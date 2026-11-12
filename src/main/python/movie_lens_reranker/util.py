import json
from itertools import islice
from typing import List, Tuple
import numpy as np

def read_movie_tiers(
    movie_tiers_uri: str, batch_size: int = 1024
) -> Tuple[np.ndarray, int, int]:
    """
    Reads a JSON Lines file of `{"movie_id": int, "tier": int}` records in batches,
    sorts by `movie_id`, prepends row 0 (`movie_id=0`, `tier=2` for OOV), and returns
    `(movie_tiers, offset, n_records)`.
    """
    sentinel = 1 << 31
    min_id = sentinel - 1
    max_id = -sentinel
    n_records = 0

    id_batches: List[np.ndarray] = []
    tier_batches: List[np.ndarray] = []

    with open(movie_tiers_uri, "r", encoding="utf-8") as f:
        while True:
            lines = [line for line in islice(f, batch_size) if line.strip()]
            if not lines:
                break

            records = [json.loads(line) for line in lines]
            batch_len = len(records)

            b_ids = np.fromiter(
                (r["movie_id"] for r in records), dtype=np.int32, count=batch_len
            )
            b_tiers = np.fromiter(
                (r["tier"] for r in records), dtype=np.int32, count=batch_len
            )

            min_id = min(min_id, int(b_ids.min()))
            max_id = max(max_id, int(b_ids.max()))
            n_records += batch_len

            id_batches.append(b_ids)
            tier_batches.append(b_tiers)

    assert n_records > 0, f"No records found in {movie_tiers_uri}"
    assert min_id > 0, f"Expected min_id > 0, got {min_id}"
    n_expected = max_id - min_id + 1
    assert n_expected == n_records, (
        f"Expected contiguous movie_ids (max_id - min_id + 1 == {n_expected}), "
        f"got n_records == {n_records}"
    )

    offset = min_id - 1

    all_ids = np.concatenate(id_batches)
    all_tiers = np.concatenate(tier_batches)

    # Sort by movie_id and prepend row 0 (movie_id=0 -> tier=2)
    sort_idx = np.argsort(all_ids)
    movie_tiers = np.empty(n_records + 1, dtype=np.int32)
    movie_tiers[0] = 2
    movie_tiers[1:] = all_tiers[sort_idx]

    return movie_tiers, offset, n_records