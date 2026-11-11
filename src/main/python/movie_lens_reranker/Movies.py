import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq


def format_movielens_title(raw_title: str) -> str:
    """Converts 'Matrix, The (1999)' -> 'The Matrix (1999)'."""
    return re.sub(
        r"^(.*),\s*(The|A|An)\s*(\(\d{4}\))$",
        r"\2 \1 \3",
        raw_title.strip(),
    )


def format_movielens_genres(raw_genres: str) -> str:
    """Converts 'Action|Sci-Fi|Thriller' -> 'Action, Sci-Fi, Thriller'."""
    return raw_genres.replace("|", ", ")


class Movies(object):
    """
    O(1) vectorized lookup table mapping `movie_id` -> `(formatted_title, formatted_genres)`.
    Index 0 is reserved as an OOV / padding sentinel.
    """

    def __init__(
        self,
        movies_uri: Optional[str] = None,
        movies_dict: Optional[Dict[int, Tuple[str, str]]] = None,
    ):
        self.pad_value = -1
        if movies_dict is not None:
            lookup = movies_dict
            if movies_uri is not None:
                raise ValueError("expecting only movies_uri or movies_dict but not both");
        elif movies_uri is not None:
            lookup = self._read_movies_file(movies_uri)
        else:
            raise ValueError("Either movies_uri or movies_dict must be provided")

        self.max_movie_id = int(max(lookup.keys(), default=0))
        self.all_movie_ids = np.array(sorted(lookup.keys()), dtype=np.int32)

        self.titles = np.full(self.max_movie_id + 1, "", dtype=object)
        self.genres = np.full(self.max_movie_id + 1, "", dtype=object)

        for mid, (raw_title, raw_genres) in lookup.items():
            m_idx = int(mid)
            if m_idx > 0:
                self.titles[m_idx] = format_movielens_title(str(raw_title))
                self.genres[m_idx] = format_movielens_genres(str(raw_genres))

    def _read_movies_file(self, movies_uri: str) -> Dict[int, Tuple[str, str]]:
        lookup: Dict[int, Tuple[str, str]] = {}
        if movies_uri.endswith(".parquet") or movies_uri.endswith(".pq"):
            table = pq.read_table(movies_uri)
            mids = table["movie_id"].to_pylist()
            titles = table["title"].to_pylist()
            genres = table["genres"].to_pylist()
            for mid, t, g in zip(mids, titles, genres):
                if isinstance(g, (list, tuple)):
                    g = ", ".join(str(x) for x in g)
                lookup[int(mid)] = (str(t), str(g))
        else:
            # Fallback for MovieLens .dat (:: delimited)
            with open(movies_uri, "r", encoding="latin-1") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("::")
                    if len(parts) >= 3:
                        lookup[int(parts[0])] = (parts[1], parts[2])
        return lookup

    def _resolve_movie_indices(
        self, movie_ids: Union[int, List[int], np.ndarray]
    ) -> np.ndarray:
        arr = np.asarray(movie_ids, dtype=np.int64)
        return np.where((arr >= 1) & (arr <= self.max_movie_id), arr, 0)

    def get_titles_and_genres(
        self, movie_ids: Union[List[int], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a 1D or 2D array of `movie_ids`, returns `(titles, genres)` of the exact same shape.
        Padded (`-1`) or out-of-range IDs return `("", "")`.
        """
        idx = self._resolve_movie_indices(movie_ids)
        return self.titles[idx], self.genres[idx]