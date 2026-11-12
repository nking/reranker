import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pyarrow as pa
from accelerate import PartialState
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from jinja2 import Template

from movie_lens_reranker.Movies import Movies
from movie_lens_reranker.UserHistory import UserHistory
from movie_lens_reranker.reranker_metrics import lookup_movie_tiers


class RerankerPromptFormatter:
    """
    Applies the Jinja2 YAML prompt templates identically to:
      1. Parquet-backed training/evaluation datasets (with DDP cache synchronization)
      2. Inline in-memory inference batches
    """

    def __init__(
        self,
        user_history: UserHistory,
        movies: Movies,
        movie_tiers: np.ndarray,
        query_yaml: Dict[str, Any],
        passages_yaml: Dict[str, Any],
        emb_placeholder_token: str,
        max_history: int = 10,
        top_k: int = 5,
        movie_offset: int = 0,
    ):
        self.user_history = user_history
        self.movies = movies
        self.query_yaml = query_yaml
        self.passages_yaml = passages_yaml
        self.top_k = top_k
        self.max_history = max_history
        self.emb_placeholder_token = emb_placeholder_token
        self.movie_tiers = np.asarray(movie_tiers, dtype=np.int32)
        
        self.movie_offset = int(movie_offset)

        self.sys_tmpl = Template(self.query_yaml["system_template"].strip())
        self.hist_tmpl = Template(self.passages_yaml["history_template"].strip())
        self.cand_tmpl = Template(self.passages_yaml["candidate_template"].strip())
        self.query_tmpl = Template(self.query_yaml["template"].lstrip())
        self.comp_tmpl = Template(self.query_yaml["completion_template"].strip())

        self.SEASONS = {
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
            12: "Winter",
        }

        self.pad_value = user_history.pad_value
        self.ts_pad_value = user_history.ts_pad_value

        self._TITLE_PATTERN = re.compile(r"^(.*),\s*(The|A|An)\s*(\(\d{4}\))$")

    def _format_query_time(self, timestamps: Union[np.ndarray, List[int]]) -> List[str]:
        """
        Converts a 1D NumPy array of Unix UTC timestamps into a list of strings
        e.g. ['Fall, Sat Night (Oct)', ...] in a vectorized manner.
        """
        ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
        if ts.size == 0:
            return []

        days_since_epoch = ts // 86400
        # Unix epoch (1970-01-01) was a Thursday (index 3)
        dow = (days_since_epoch + 3) % 7
        dow_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_names = [dow_map[d] for d in dow]

        hours = (ts % 86400) // 3600

        tods = np.empty(ts.shape, dtype=object)
        tods[(hours >= 5) & (hours < 12)] = "Morning"
        tods[(hours >= 12) & (hours < 17)] = "Afternoon"
        tods[(hours >= 17) & (hours < 22)] = "Evening"
        tods[(hours < 5) | (hours >= 22)] = "Night"

        out: List[str] = []
        seasons_map = self.SEASONS
        for i in range(len(ts)):
            dt = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc)
            season = seasons_map[dt.month]
            out.append(f"{season}, {day_names[i]} {tods[i]} ({dt.strftime('%b')})")

        return out

    def _format_relative_delta(
        self,
        query_ts: int,
        hist_ts: Union[np.ndarray, List[int]],
    ) -> np.ndarray:
        """Formats elapsed times `(query_ts - hist_ts)` into compact units ('20m ago', '10d ago')."""
        if len(hist_ts) == 0:
            return np.array([], dtype=str)

        hist_arr = np.asarray(hist_ts, dtype=np.int64)
        if hist_arr.size == 0:
            return np.empty(hist_arr.shape, dtype=str)

        diff_sec = np.maximum(0, int(query_ts) - hist_arr)

        condlist = [
            diff_sec < 3600,
            diff_sec < 86400,
            diff_sec < 86400 * 30,
        ]

        choicelist = [
            np.char.add(np.maximum(1, diff_sec // 60).astype(str), "m ago"),
            np.char.add((diff_sec // 3600).astype(str), "h ago"),
            np.char.add((diff_sec // 86400).astype(str), "d ago"),
        ]

        default = np.char.add((diff_sec // (86400 * 30)).astype(str), "mo ago")

        return np.select(condlist, choicelist, default=default)

    def _candidate_label(self, idx: int) -> str:
        """Returns 'A'..'Z' for idx < 26, and 'C26', 'C27', ... for larger pools before top_k subsampling."""
        return chr(ord("A") + idx) if idx < 26 else f"C{idx}"

    def _format_movielens_titles(
        self, raw_titles: Union[np.ndarray, List[str]]
    ) -> np.ndarray:
        """Converts movie titles such as 'Matrix, The (1999)' to 'The Matrix (1999)'."""
        if len(raw_titles) == 0:
            return np.array([], dtype=str)

        arr = np.asarray(raw_titles)
        pattern = self._TITLE_PATTERN

        @np.vectorize(otypes=[str])
        def _clean_title(raw_title: str) -> str:
            if not isinstance(raw_title, str):
                raw_title = str(raw_title)
            return pattern.sub(r"\2 \1 \3", raw_title.strip())

        return cast(np.ndarray, _clean_title(arr))

    def _format_movielens_genres(
        self, raw_genres: Union[np.ndarray, List[str]]
    ) -> np.ndarray:
        """Converts 'Action|Sci-Fi|Thriller' -> 'Action, Sci-Fi, Thriller'."""
        if len(raw_genres) == 0:
            return np.array([], dtype=str)

        arr = np.asarray(raw_genres)

        @np.vectorize(otypes=[str])
        def _clean_genre(raw_genre: str) -> str:
            if not isinstance(raw_genre, str):
                raw_genre = str(raw_genre)
            return raw_genre.strip().replace("|", ", ")

        return cast(np.ndarray, _clean_genre(arr))

    def _format_history_items(
        self,
        user_ids: np.ndarray,
        timestamps: np.ndarray,
        hist_movie_ids: np.ndarray,
        hist_ratings: np.ndarray,
        hist_timestamps: np.ndarray,
        hist_titles: np.ndarray,
        hist_genres: np.ndarray,
    ) -> List[Dict[str, Any]]:
        batch_history_items: List[Dict[str, Any]] = []

        for b_idx in range(len(user_ids)):
            q_ts = int(timestamps[b_idx])
            row_hist_ids = hist_movie_ids[b_idx]

            pad_mask = row_hist_ids == self.pad_value
            real_hist_idx = (
                int(np.argmax(pad_mask)) if np.any(pad_mask) else len(row_hist_ids)
            )

            user_hist_dict = {
                "movie_id": row_hist_ids[0:real_hist_idx],
                "title": self._format_movielens_titles(
                    hist_titles[b_idx][0:real_hist_idx]
                ).tolist(),
                "genres": self._format_movielens_genres(
                    hist_genres[b_idx][0:real_hist_idx]
                ).tolist(),
                "rating": (hist_ratings[b_idx][0:real_hist_idx]).tolist(),
                "recency": self._format_relative_delta(
                    q_ts, hist_timestamps[b_idx][0:real_hist_idx]
                ).tolist(),
            }
            batch_history_items.append(user_hist_dict)
        return batch_history_items

    def load_and_format_parquet_dataset(
        self,
        parquet_files: str | List[str],
        seed: int = 42,
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]], int]:
        """
        Loads preprocessed training/eval Parquet files, subsamples `top_k` candidates if
        `num_candidates > top_k`, renders `"prompt"` and `"completion"`, and builds `qrels_dict`.
        Uses `PartialState().main_process_first()` so DDP ranks never race on the Arrow cache.
        """
        state = PartialState()

        with state.main_process_first():
            ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = (
                load_dataset("parquet", data_files=parquet_files, split=None)
            )
            if isinstance(ds, DatasetDict):
                ds = ds["train"]

            arrow_schema = ds.data.schema
            num_candidates: int = arrow_schema.field("candidate_ids").type.list_size

            cand_emb_type = arrow_schema.field("candidate_embeddings").type
            if isinstance(cand_emb_type.value_type, pa.FixedSizeListType):
                ranker_out_features: int = cand_emb_type.value_type.list_size
            else:
                ranker_out_features = cand_emb_type.list_size // num_candidates

            return self.load_and_format_dataset(
                dataset=ds,
                ranker_out_features=ranker_out_features,
                num_candidates=num_candidates,
                seed=seed,
            )

    def load_and_format_dataset(
        self,
        dataset: Dataset,
        ranker_out_features: int,
        num_candidates: int,
        seed: int = 42,
        training: bool = True,
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]], int]:
        expected_columns = [
            "user_id",
            "movie_id",
            "rating",
            "timestamp",
            "candidate_ids",
            "candidate_scores",
            "user_embedding",
            "candidate_embeddings",
        ]
        ds_col_names = set(dataset.column_names)
        for col in expected_columns:
            if col not in ds_col_names:
                raise ValueError(f"dataset must have column '{col}'")

        batch_size = min(256, max(1, len(dataset)))
        cand_labels = [self._candidate_label(i) for i in range(self.top_k)]

        def _format_batch(
            batch: Dict[str, List[Any]], indices: List[int]
        ) -> Dict[str, List[Any]]:
            user_ids = batch["user_id"]
            movie_ids = batch["movie_id"]
            ratings = batch["rating"]
            timestamps = batch["timestamp"]
            candidate_ids_batch = batch["candidate_ids"]
            candidate_scores_batch = batch["candidate_scores"]
            user_embeddings_batch = batch["user_embedding"]
            candidate_embeddings_batch = batch["candidate_embeddings"]

            _user_ids = np.asarray(user_ids, dtype=np.int32)
            _movie_ids = np.asarray(movie_ids, dtype=np.int32)
            _timestamps = np.asarray(timestamps, dtype=np.int64)

            if self.movie_tiers is not None:
                batch_pos_tiers = lookup_movie_tiers(
                    _movie_ids, self.movie_tiers, self.movie_offset
                )
            else:
                # Deterministic fallback tier assignment (0, 1, 2) if movie_tiers is not provided
                batch_pos_tiers = np.mod(_movie_ids, 3).astype(np.int64)

            history_items: List[Dict[str, Any]] = self._get_and_format_user_history(
                _user_ids, _timestamps
            )
            candidate_items: List[Dict[str, Any]] = self._get_candidate_details(
                candidate_ids_batch, ranker_out_features=ranker_out_features
            )
            query_times = self._format_query_time(_timestamps)

            cur_batch_size = len(user_ids)

            out_prompts: List[str] = []
            out_completions: List[Optional[str]] = []
            out_labels_list: List[List[str]] = []
            out_label_rels: List[List[int]] = []
            out_ranker_embs: List[List[List[float]]] = []
            out_cand_scores: List[List[float]] = []
            out_movie_ids: List[int] = []
            out_pos_tiers: List[int] = []

            for i in range(cur_batch_size):
                h_i = history_items[i]

                # Skip if user has no prior watch history before query timestamp
                if len(h_i["title"]) == 0:
                    continue

                c_i = candidate_items[i]
                h_i.pop("movie_id", None)
                c_i.pop("movie_id", None)

                raw_user_emb = np.asarray(
                    user_embeddings_batch[i], dtype=np.float32
                ).reshape(-1)
                user_emb = raw_user_emb[:ranker_out_features]

                cand_ids = np.asarray(candidate_ids_batch[i], dtype=np.int32)
                cand_scores = np.asarray(candidate_scores_batch[i], dtype=np.float32)
                cand_embs = np.asarray(
                    candidate_embeddings_batch[i], dtype=np.float32
                ).reshape(num_candidates, ranker_out_features)
                cand_titles = np.asarray(c_i["title"], dtype=object)
                cand_genres = np.asarray(c_i["genres"], dtype=object)

                # Subsample to self.top_k if Parquet stores > top_k candidates
                if len(cand_ids) > self.top_k:
                    pos_matches = np.where(cand_ids == int(movie_ids[i]))[0]
                    if len(pos_matches) > 0 and pos_matches[0] >= self.top_k:
                        pos_idx = int(pos_matches[0])
                        for arr in (
                            cand_ids,
                            cand_scores,
                            cand_embs,
                            cand_titles,
                            cand_genres,
                        ):
                            arr[[0, pos_idx]] = arr[[pos_idx, 0]]
                    cand_ids = cand_ids[: self.top_k]
                    cand_scores = cand_scores[: self.top_k]
                    cand_embs = cand_embs[: self.top_k]
                    cand_titles = cand_titles[: self.top_k]
                    cand_genres = cand_genres[: self.top_k]

                row: Dict[str, Any] = {
                    "user_id": user_ids[i],
                    "movie_id": movie_ids[i],
                    "rating": ratings[i],
                    "timestamp": timestamps[i],
                    "query_time": query_times[i],
                    "user_embedding": user_emb,
                    "history_items": h_i,
                    "candidate_ids": cand_ids,
                    "candidate_scores": cand_scores,
                    "candidate_embeddings": cand_embs,
                    "candidate_items": {
                        "title": cand_titles,
                        "genres": cand_genres,
                    },
                    "labels": np.asarray(cand_labels, dtype=str),
                }

                if training:
                    label_rels = np.zeros(self.top_k, dtype=int)
                    pos_matches = np.where(row["candidate_ids"] == int(movie_ids[i]))[0]
                    pos_slot = int(pos_matches[0]) if len(pos_matches) > 0 else 0
                    label_rels[pos_slot] = 1
                    row["label_rels"] = label_rels

                    row_rng = np.random.default_rng(seed + indices[i])
                    perm = row_rng.permutation(self.top_k)
                    for key in (
                        "candidate_ids",
                        "candidate_scores",
                        "candidate_embeddings",
                        "label_rels",
                    ):
                        row[key] = row[key][perm]
                    for key in ("title", "genres"):
                        row["candidate_items"][key] = row["candidate_items"][key][perm]

                ranker_embs = np.vstack(
                    [
                        row["user_embedding"].reshape(1, ranker_out_features),
                        row["candidate_embeddings"].reshape(
                            self.top_k, ranker_out_features
                        ),
                    ]
                ).astype(np.float32)

                prompt, completion = self.render_prompt_and_completion(
                    row,
                    first_label=cand_labels[0],
                    last_label=cand_labels[-1],
                )

                out_prompts.append(prompt)
                out_completions.append(completion)
                out_labels_list.append(row["labels"].tolist())
                out_ranker_embs.append(ranker_embs.tolist())
                out_cand_scores.append(
                    np.asarray(row["candidate_scores"], dtype=np.float32).tolist()
                )
                out_movie_ids.append(int(movie_ids[i]))
                out_pos_tiers.append(int(batch_pos_tiers[i]))
                if training:
                    out_label_rels.append(row["label_rels"].tolist())

            result: Dict[str, List[Any]] = {
                "prompt": out_prompts,
                "completion": out_completions,
                "labels_list": out_labels_list,
                "ranker_embs": out_ranker_embs,
                "candidate_scores": out_cand_scores,
                "movie_id": out_movie_ids,
                "pos_tier": out_pos_tiers,
            }
            if training:
                result["label_rels_list"] = out_label_rels

            return result

        # Deterministic fingerprint so DDP worker ranks always hit rank 0's Arrow cache
        ds_fp = getattr(dataset, "_fingerprint", "in_mem")
        tiers_tag = "with_tiers" if self.movie_tiers is not None else "no_tiers"
        fp_raw = (
            f"{ds_fp}_s{seed}_k{self.top_k}_h{self.max_history}_"
            f"f{ranker_out_features}_t{training}_{tiers_tag}"
        )
        map_fingerprint = hashlib.md5(fp_raw.encode("utf-8")).hexdigest()

        state = PartialState()
        with state.main_process_first():
            formatted_ds = dataset.map(
                function=_format_batch,
                with_indices=True,
                batched=True,
                batch_size=batch_size,
                remove_columns=dataset.column_names,
                new_fingerprint=map_fingerprint,
                desc="Formatting Parquet rows into Reranker Prompts",
            )

        qrels_dict: Dict[str, Dict[str, int]] = {}

        if training and "label_rels_list" in formatted_ds.column_names:
            all_labels = formatted_ds["labels_list"]
            all_rels = formatted_ds["label_rels_list"]
            for idx, row_labels in enumerate(all_labels):
                row_rels = all_rels[idx]
                qrels_dict[f"q_{idx}"] = {
                    row_labels[c_idx]: int(row_rels[c_idx])
                    for c_idx in range(self.top_k)
                }

        formatted_ds = formatted_ds.remove_columns(["labels_list"])
        return formatted_ds, qrels_dict, ranker_out_features

    def render_prompt_and_completion(
        self,
        row: Dict[str, Any],
        first_label: str,
        last_label: str,
    ) -> Tuple[str, Optional[str]]:
        """Shared Jinja2 rendering used for both Parquet training rows and inline inference."""
        system_prompt = self.sys_tmpl.render(
            first_label=first_label,
            last_label=last_label,
        ).strip()

        history_len = len(row["history_items"]["title"])
        history_lines = [
            self.hist_tmpl.render(
                title=row["history_items"]["title"][i],
                genres=row["history_items"]["genres"][i],
                rating=row["history_items"]["rating"][i],
                recency=row["history_items"]["recency"][i],
            ).strip()
            for i in range(history_len)
        ]
        history_block = "\n".join(history_lines)

        candidate_len = len(row["candidate_items"]["title"])
        candidate_lines = [
            self.cand_tmpl.render(
                cand_emb=self.emb_placeholder_token,
                label=row["labels"][i],
                title=row["candidate_items"]["title"][i],
                genres=row["candidate_items"]["genres"][i],
            ).strip()
            for i in range(candidate_len)
        ]
        candidates_block = "\n".join(candidate_lines)

        prompt = self.query_tmpl.render(
            system_prompt=system_prompt,
            query_time=row["query_time"],
            user_emb=self.emb_placeholder_token,
            history_block=history_block,
            candidates_block=candidates_block,
        )
        if not prompt.endswith("\n"):
            prompt += "\n"

        completion = None
        if "label_rels" in row:
            scores = np.asarray(row["candidate_scores"], dtype=np.float32)
            rels = np.asarray(row["label_rels"], dtype=np.int32)
            order_idx = np.lexsort((-scores, -rels))
            ranked_labels_list = [str(row["labels"][idx]) for idx in order_idx]

            if "join" in self.query_yaml.get("completion_template", ""):
                completion = self.comp_tmpl.render(ranked_labels=ranked_labels_list)
            else:
                completion = self.comp_tmpl.render(
                    ranked_labels=",".join(ranked_labels_list)
                )

        return prompt, completion

    def _get_and_format_user_history(
        self, user_ids: np.ndarray, timestamps: np.ndarray
    ) -> List[Dict[str, Any]]:
        hist_movie_ids, hist_ratings, hist_timestamps = (
            self.user_history.get_history_with_timestamps_before_timestamp(
                user_id=user_ids,
                timestamp=timestamps,
                max_hist=self.max_history,
                most_recent_first=True,
            )
        )
        hist_titles, hist_genres = self.movies.get_titles_and_genres(hist_movie_ids)

        return self._format_history_items(
            user_ids,
            timestamps,
            hist_movie_ids,
            hist_ratings,
            hist_timestamps,
            hist_titles,
            hist_genres,
        )

    def _get_candidate_details(
        self,
        candidate_movie_ids: Union[np.ndarray, List[List[int]]],
        ranker_out_features: int,
    ) -> List[Dict[str, Any]]:
        batch_size = len(candidate_movie_ids)

        items: List[Dict[str, Any]] = []
        for b_idx in range(batch_size):
            candidate_batch = candidate_movie_ids[b_idx]
            cand_titles, cand_genres = self.movies.get_titles_and_genres(
                candidate_batch
            )
            cand_dict = {
                "movie_id": candidate_batch,
                "title": self._format_movielens_titles(cand_titles).tolist(),
                "genres": self._format_movielens_genres(cand_genres).tolist(),
            }
            items.append(cand_dict)
        return items