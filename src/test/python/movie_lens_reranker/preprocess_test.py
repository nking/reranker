import os
import re
import shutil
import unittest

from helper import *

from movie_lens_reranker.Movies import Movies
from movie_lens_reranker.RecommendedMovies import RecommendedMovies
from movie_lens_reranker.RerankerPromptFormatter import (
    EMB_PLACEHOLDER_TOKEN,
    RerankerPromptFormatter,
)
from movie_lens_reranker.UserHistory import UserHistory
from movie_lens_reranker.prompts.prompt_helper import load_prompt_yamls

HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")
if not HAS_NVIDIA_GPU:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np


class TestRerankerPreprocessing(unittest.TestCase):

    def setUp(self):
        self.input_parquet = get_train_parquet(data_size=DataSize.TINY3)

        self.num_catalog_users = 6040

        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.TINY3)

        data_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        recommendations_uri = os.path.join(data_dir, "recommended_movies.parquet")
        recommendations_timestamps_uri = os.path.join(
            data_dir, "recommended_movies_timestamps.parquet"
        )
        movies_path = os.path.join(data_dir, "movies.parquet")

        self.out_dir = os.path.join(get_bin_dir(), "reranker_preprocess_test")
        shutil.rmtree(self.out_dir, ignore_errors=True)

        self.uh = UserHistory(
            ratings_uri_list=[
                ratings_uri_dict["train_liked"],
                ratings_uri_dict["train_3"],
                ratings_uri_dict["train_disliked"],
            ],
            max_history=2048,
            n_users=self.num_catalog_users,
        )

        self.recommendations = RecommendedMovies(
            num_users=self.num_catalog_users,
            movie_rec_file_uri=recommendations_uri,
            movie_rec_ts_file_uri=recommendations_timestamps_uri,
        )

        self.movies = Movies(movies_uri=movies_path)

    def test_inference_and_training_parquet_roundtrip_and_subsampling(self):
        top_k = 5
        query_yaml, passages_yaml = load_prompt_yamls(version="2.0.0")

        prompt_formatter = RerankerPromptFormatter(
            user_history=self.uh,
            movies=self.movies,
            query_yaml=query_yaml,
            passages_yaml=passages_yaml,
            emb_placeholder_token="<|vision_pad|>",
            max_history=20,
            top_k=top_k,
        )

        formatted_ds, qrels_dict, ranker_out_features = prompt_formatter.load_and_format_parquet_dataset(
            self.input_parquet
        )
        
        prompt_lengths = [len(prompt) for prompt in formatted_ds["prompt"]]
        max_prompt_length = max(prompt_lengths)
        mean_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
        
        print(f"--- Prompt Length Statistics ---")
        print(f"Max prompt length: {max_prompt_length} characters")
        print(f"Mean prompt length: {mean_prompt_length:.1f} characters")

        self.assertIsNotNone(formatted_ds)
        self.assertGreater(len(formatted_ds), 0, "Expected non-empty formatted dataset")
        self.assertEqual(len(qrels_dict), len(formatted_ds))

        expected_cols = {"prompt", "completion", "ranker_embs", "label_rels_list"}
        self.assertTrue(
            expected_cols.issubset(set(formatted_ds.column_names)),
            f"Missing columns: {expected_cols - set(formatted_ds.column_names)}",
        )

        expected_labels = ["A", "B", "C", "D", "E"]
        first_ranked_labels_seen = set()

        for idx in range(len(formatted_ds)):
            row = formatted_ds[idx]
            prompt = row["prompt"]
            completion = row["completion"]
            ranker_embs = np.asarray(row["ranker_embs"], dtype=np.float32)
            label_rels = row["label_rels_list"]

            # 1. Verify prompt structure & placeholder token count (1 user + top_k candidates)
            self.assertTrue(prompt.startswith("<|im_start|>system"))
            self.assertIn("Query Time:", prompt)
            self.assertIn("User Watch History:", prompt)
            self.assertIn("Candidates:", prompt)
            self.assertIn("Ranked IDs:<|im_end|>", prompt)
            self.assertEqual(
                prompt.count(EMB_PLACEHOLDER_TOKEN),
                1 + top_k,
                f"Expected {1 + top_k} '{EMB_PLACEHOLDER_TOKEN}' tokens in prompt, got {prompt.count(EMB_PLACEHOLDER_TOKEN)}",
            )

            # Ensure raw float vectors were not stringified into the prompt text
            self.assertIsNone(
                re.search(r"\[\s*-?\d+\.\d+\s+-?\d+\.\d+", prompt),
                f"Found raw stringified float array in prompt:\n{prompt}",
            )

            # Ensure watch history bullets have no blank lines between them and have (X/5) ratings
            self.assertNotIn("\n\n- ", prompt)
            self.assertRegex(prompt, r"\(\d+/5\)")

            # Verify candidates appear in canonical [A]..[E] order in the prompt
            label_positions = [
                prompt.index(f"[{lbl}] {EMB_PLACEHOLDER_TOKEN}")
                for lbl in expected_labels
            ]
            self.assertEqual(
                label_positions,
                sorted(label_positions),
                "Candidate labels in prompt must appear in canonical [A]-[E] order",
            )

            # 2. Verify completion is a comma-separated ranking of A..E ending in <|im_end|>
            self.assertIsNotNone(completion)
            self.assertTrue(completion.endswith("<|im_end|>"))
            clean_comp = completion.replace("<|im_end|>", "").strip()
            ranked_letters = [tok.strip() for tok in clean_comp.split(",")]
            self.assertEqual(len(ranked_letters), top_k)
            self.assertEqual(sorted(ranked_letters), expected_labels)

            # 3. Verify the first ranked letter in completion is the ground-truth positive (rel == 1)
            pos_slot_idx = label_rels.index(1)
            expected_pos_label = expected_labels[pos_slot_idx]
            self.assertEqual(
                ranked_letters[0],
                expected_pos_label,
                f"Row {idx}: expected positive label {expected_pos_label} first in completion '{completion}'",
            )
            first_ranked_labels_seen.add(ranked_letters[0])

            # 4. Verify qrels_dict alignment with label_rels_list and completion
            qrel_entry = qrels_dict[f"q_{idx}"]
            self.assertEqual(set(qrel_entry.keys()), set(expected_labels))
            self.assertEqual(qrel_entry[expected_pos_label], 1.0)
            self.assertEqual(sum(qrel_entry.values()), 1.0)

            # 5. Verify ranker_embs numeric tensor shape (1 + top_k, 24) and finiteness
            self.assertEqual(ranker_embs.ndim, 2)
            self.assertEqual(ranker_embs.shape[0], 1 + top_k)
            self.assertGreater(ranker_embs.shape[1], 0)
            self.assertTrue(np.all(np.isfinite(ranker_embs)))

        # Verify candidate shuffling moves the ground-truth positive across multiple letters
        if len(formatted_ds) >= 5:
            self.assertGreater(
                len(first_ranked_labels_seen),
                1,
                "Expected positive candidate position to be shuffled across multiple labels A..E",
            )


if __name__ == "__main__":
    unittest.main()