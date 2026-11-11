from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

class UserHistory(object):
    """
    Stores per-user interaction histories (movie_ids, ratings, timestamps) in dense 2D arrays
    of shape `(n_users + 1, max_history)` where outer row index `i` corresponds directly
    to `user_id = i`.
    """

    def __init__(
        self,
        ratings_uri_list: Union[str, List[str]],
        max_history: int,
        n_users: Optional[int] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
    ):
        self.pad_value = -1
        self.ts_pad_value = 2524608000  # year 2050
        self.max_history = max_history
        self.n_users = n_users
        self.min_rating = min_rating
        self.max_rating = max_rating
        # implicit indices are the user_ids
        # parallel arrays are sorted by timestamps and the empty slots are filled by the ts_pad_valu or pad_value and are at the ends of the arrays
        self.movie_ids, self.ratings, self.timestamps = self._load_history(
            ratings_uri_list, n_users=n_users
        )
    
    def _load_history(
        self,
        ratings_uri_list: Union[str, List[str]],
        n_users: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        lookup, max_observed_history = self.build_history_lookup(ratings_uri_list)
        logging.info(
            f"Max of users' histories = {max_observed_history}. self.max_history={self.max_history}"
        )
        self.max_observed_history = max(int(self.max_history),
            int(max_observed_history))
        
        max_uid = int(max(lookup.keys(), default=0))
        total_users = max(max_uid, int(n_users) if n_users is not None else 0)
        self.n_users = total_users
        
        n_rows = total_users + 1
        
        movie_ids = np.full((n_rows, self.max_observed_history),
            self.pad_value, dtype=np.int32)
        ratings = np.full((n_rows, self.max_observed_history), self.pad_value,
            dtype=np.int32)
        timestamps = np.full((n_rows, self.max_observed_history),
            self.ts_pad_value, dtype=np.int64)
        
        for user_id, (user_ts, user_movies, user_ratings) in lookup.items():
            uid = int(user_id)
            length = len(user_ts)
            if length > 0:
                timestamps[uid, :length] = user_ts
                movie_ids[uid, :length] = user_movies
                ratings[uid, :length] = user_ratings
        
        return movie_ids, ratings, timestamps
    
    def build_history_lookup(
        self,
        ratings_uri_list: Union[str, List[str]],
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
        """
        Reads Parquet file(s) containing columns `user_id`, `movie_id`, `rating`, `timestamp`
        and returns `{user_id: (sorted_ts, sorted_movies, sorted_ratings)}` and `max_history`.
        """
        if isinstance(ratings_uri_list, str):
            ratings_uri_list = [ratings_uri_list]

        lookup = defaultdict(lambda: {"ts": [], "movie_id": [], "rating": []})

        for ratings_uri in ratings_uri_list:
            table = pq.read_table(
                ratings_uri,
                columns=["user_id", "movie_id", "rating", "timestamp"],
            )
            u_col = table["user_id"].to_numpy()
            m_col = table["movie_id"].to_numpy()
            r_col = table["rating"].to_numpy()
            ts_col = table["timestamp"].to_numpy()

            # Optional rating filter (e.g., max_rating=2 or 3 for disliked history lookup)
            mask = np.ones(len(u_col), dtype=np.bool_)
            if self.min_rating is not None:
                mask &= r_col >= self.min_rating
            if self.max_rating is not None:
                mask &= r_col <= self.max_rating

            for u, m, r, ts in zip(
                u_col[mask], m_col[mask], r_col[mask], ts_col[mask]
            ):
                uid = int(u)
                lookup[uid]["movie_id"].append(int(m))
                lookup[uid]["rating"].append(int(r))
                lookup[uid]["ts"].append(int(ts))

        max_history = 0
        sorted_lookup = {}
        for u, data in lookup.items():
            ts_arr = np.asarray(data["ts"], dtype=np.int64)
            m_arr = np.asarray(data["movie_id"], dtype=np.int32)
            r_arr = np.asarray(data["rating"], dtype=np.int32)

            order = np.argsort(ts_arr, kind="stable")
            max_history = max(max_history, len(m_arr))
            sorted_lookup[u] = (ts_arr[order], m_arr[order], r_arr[order])

        return sorted_lookup, max_history

    def _resolve_user_indices(
        self, user_id: Union[int, List[int], np.ndarray]
    ) -> np.ndarray:
        user_id_arr = np.asarray(user_id, dtype=np.int64).reshape(-1)
        return np.where(
            (user_id_arr >= 1) & (user_id_arr <= self.n_users), user_id_arr, 0
        )

    def get_history_with_timestamps_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
        most_recent_first: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns `(ret_movie_ids, ret_ratings, ret_timestamps)` for the `max_hist` MOST RECENT
        interactions strictly `< timestamp`, padded at the right with `pad_value` (-1) and `ts_pad_value`.
        """
        user_idx = self._resolve_user_indices(user_id)
        n_user_selected = len(user_idx)
        
        ret_movie_ids = np.full(
            (n_user_selected, max_hist), self.pad_value,
            dtype=self.movie_ids.dtype
        )
        ret_ratings = np.full(
            (n_user_selected, max_hist), self.pad_value,
            dtype=self.ratings.dtype
        )
        ret_timestamps = np.full(
            (n_user_selected, max_hist),
            self.ts_pad_value,
            dtype=self.timestamps.dtype,
        )
        if n_user_selected == 0 or max_hist <= 0:
            return ret_movie_ids, ret_ratings, ret_timestamps
        
        sub_timestamps = self.timestamps[user_idx]
        sub_movie_ids = self.movie_ids[user_idx]
        sub_ratings = self.ratings[user_idx]
        
        # TODO: consider != self.ts_pad_value instead of < ts.pad_value
        if np.ndim(timestamp) == 0:
            mask = (sub_timestamps < timestamp) & (
                        sub_timestamps < self.ts_pad_value)
        else:
            ts_col = np.asarray(timestamp, dtype=np.int64).reshape(-1, 1)
            mask = (sub_timestamps < ts_col) & (
                        sub_timestamps < self.ts_pad_value)
            
            # Select the MOST RECENT `max_hist` valid interactions per user
        total_valid = np.sum(mask, axis=1, keepdims=True)
        occ = np.cumsum(mask, axis=1)
        recent_mask = mask & (occ > (total_valid - max_hist))
        
        recent_occ = np.cumsum(recent_mask, axis=1)
        recent_count = np.minimum(total_valid.squeeze(1), max_hist)
        
        row_coords, _ = np.where(recent_mask)
        if most_recent_first:
            new_col_coords = recent_count[row_coords] - recent_occ[recent_mask]
        else:
            new_col_coords = recent_occ[recent_mask] - 1
        
        ret_movie_ids[row_coords, new_col_coords] = sub_movie_ids[recent_mask]
        ret_ratings[row_coords, new_col_coords] = sub_ratings[recent_mask]
        ret_timestamps[row_coords, new_col_coords] = sub_timestamps[
            recent_mask]
        
        return ret_movie_ids, ret_ratings, ret_timestamps
    
    def get_history_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
        most_recent_first: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ret_movie_ids, ret_ratings, _ = (
            self.get_history_with_timestamps_before_timestamp(
                user_id=user_id,
                timestamp=timestamp,
                max_hist=max_hist,
                most_recent_first=most_recent_first,
            )
        )
        return ret_movie_ids, ret_ratings

    def get_movieids_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
        most_recent_first: bool = True,
    ) -> np.ndarray:
        ret_movie_ids, _, _ = self.get_history_with_timestamps_before_timestamp(
            user_id=user_id,
            timestamp=timestamp,
            max_hist=max_hist,
            most_recent_first=most_recent_first,
        )
        return ret_movie_ids