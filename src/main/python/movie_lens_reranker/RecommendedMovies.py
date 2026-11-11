from typing import Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


class RecommendedMovies(object):
    """
    Stores per-user recommended movie lists and their watch timestamps (or 2524608000 if unseen)
    loaded from Parquet.
    """

    def __init__(
        self,
        num_users: int,
        movie_rec_file_uri: str,
        movie_rec_ts_file_uri: Optional[str] = None,
    ):
        """
        :param num_users: number of catalog users
        :param movie_rec_file_uri: URI of movie parquet file containomh user_id and movie_ids
        :param movie_rec_ts_file_uri: URI of movie parquet file that contains user_id and timestamps
        """
        self.pad_value = -1
        self.ts_pad_value = 2524608000 #year 2050
        self.num_users = num_users
        
        self.user_ids, self.movies, self.row_length = self._read(num_users,
            file_uri=movie_rec_file_uri, pad_value=self.pad_value)
        _, self.timestamps, _ = self._read(num_users, movie_rec_ts_file_uri,
            pad_value=self.pad_value, dtype2=np.int64)
        self.num_users = num_users
        

    def _read(
        self,
        num_catalog_users: int,
        file_uri: str,
        pad_value: int,
        dtype2: type = np.int32,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        
        table = pq.read_table(file_uri)
        col_names = table.column_names
        uid_col = "user_id" if "user_id" in col_names else col_names[0]
        val_col = [c for c in col_names if c != uid_col][0]

        uids = table[uid_col].to_pylist()
        lists = table[val_col].to_pylist()
        len_arr = len(lists[0]) if len(lists) > 0 else 0

        #add empty row so that the user_ids can act as indices
        user_ids = np.arange(0, num_catalog_users + 1, dtype=np.int32)
        items = np.full((num_catalog_users + 1, len_arr), fill_value=pad_value, dtype=dtype2)

        for uid, row_vals in zip(uids, lists):
            u_idx = int(uid)
            if 0 <= u_idx <= num_catalog_users:
                items[u_idx, :] = np.asarray(row_vals, dtype=dtype2)

        return user_ids, items, len_arr

    def get_all_recommended_movies(self, user_id: np.ndarray) -> np.ndarray:
        """Returns all recommended movies (both seen and unseen) ordered by decreasing score."""
        user_idx = self.user_ids[user_id]
        return self.movies[user_idx]

    def get_unseen_movies(
        self,
        user_id: np.ndarray,
        timestamp: np.ndarray,
        latest_unseen_k: int = 200,
    ) -> np.ndarray:
        """
        Returns top `latest_unseen_k` unseen recommended movies ordered by decreasing score
        (highest-scoring unseen movies first).
        """
        if latest_unseen_k > self.row_length:
            raise ValueError(
                f"latest_unseen_k ({latest_unseen_k}) must be <= recommendations per user ({self.row_length})"
            )

        user_idx = self.user_ids[user_id]
        
        mask = self.timestamps[user_idx] > timestamp[:, np.newaxis]
        masked_movies = np.where(mask, self.movies[user_idx], self.pad_value)
        sort_indices = np.argsort(~mask, axis=1, kind="stable")
        sel_movies = np.take_along_axis(masked_movies, sort_indices, axis=1)
        return sel_movies[:, :latest_unseen_k]

    def get_lowest_scoring_unseen_movies(
        self,
        user_id: np.ndarray,
        timestamp: np.ndarray,
        worst_unseen_k: int = 200,
    ) -> np.ndarray:
        """
        Samples from the tail of the recommendation list: returns `worst_unseen_k` unseen
        recommended movies ordered by INCREASING score (worst-scoring unseen movies first).
        """
        if worst_unseen_k > self.row_length:
            raise ValueError(
                f"worst_unseen_k ({worst_unseen_k}) must be <= recommendations per user ({self.row_length})"
            )
        
        user_idx = self.user_ids[user_id]
        mask = self.timestamps[user_idx] > timestamp[:, np.newaxis]
        masked_movies = np.where(mask, self.movies[user_idx], self.pad_value)
        mask_rev = mask[:, ::-1]
        movies_rev = masked_movies[:, ::-1]
        sort_indices = np.argsort(~mask_rev, axis=1, kind="stable")
        sel_movies = np.take_along_axis(movies_rev, sort_indices, axis=1)
        return sel_movies[:, :worst_unseen_k]
