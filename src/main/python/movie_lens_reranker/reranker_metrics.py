from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from ranx import Qrels, Run, evaluate

TIER_NAMES: Dict[int, str] = {
    0: "head",
    1: "torso",
    2: "tail",
}


def lookup_movie_tiers(
    movie_ids: Union[np.ndarray, Sequence[int]],
    movie_tiers: np.ndarray,
    movie_offset: int = 0,
) -> np.ndarray:
    """
    Maps movie_ids to popularity frequency tiers {0: head, 1: torso, 2: tail}.
    Indices in `movie_tiers` are `movie_id - movie_offset`, with row 0 reserved
    for OOV `movie_id = 0`.
    """
    ids = np.asarray(movie_ids, dtype=np.int64)
    tiers_arr = np.asarray(movie_tiers, dtype=np.int64).reshape(-1)
    if tiers_arr.size == 0:
        return np.zeros(ids.shape, dtype=np.int64)

    indices = ids - int(movie_offset)
    # Row 0 is reserved for OOV movie_id <= 0 or out-of-range IDs
    invalid_mask = (ids <= 0) | (indices < 0) | (indices >= tiers_arr.size)
    safe_indices = np.where(invalid_mask, 0, indices)
    return tiers_arr[safe_indices]


def build_movie_tiers_from_counts(
    movie_counts: np.ndarray,
    head_quantile: float = 0.80,
    torso_quantile: float = 0.50,
) -> np.ndarray:
    """
    Fallback utility to construct a `movie_tiers` array (0=head, 1=torso, 2=tail,
    row 0 = OOV) from item interaction counts when not loaded from disk.
    """
    counts = np.asarray(movie_counts, dtype=np.float64).reshape(-1)
    tiers = np.full(counts.shape, 2, dtype=np.int64)
    if counts.size <= 1:
        return tiers

    pos_counts = counts[1:][counts[1:] > 0]
    if pos_counts.size == 0:
        tiers[0] = 2
        return tiers

    head_thresh = float(np.quantile(pos_counts, head_quantile))
    torso_thresh = float(np.quantile(pos_counts, torso_quantile))

    tiers[counts >= torso_thresh] = 1
    tiers[counts >= head_thresh] = 0
    tiers[0] = 2  # OOV movie_id=0 defaults to tail tier
    return tiers


def compute_composite_reranker_metrics(
    candidate_scores: np.ndarray,
    qrels_dict: Dict[str, Dict[str, int]],
    query_tiers: Union[Sequence[int], np.ndarray, Dict[str, int]],
    top_k: int = 5,
    candidate_labels: Optional[List[str]] = None,
    tier_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Computes overall NDCG@K, MRR@K, per-tier NDCG@K (head, torso, tail),
    and `composite_ndcg@K` (sum of tier NDCGs weighted by `tier_weights`).
    """
    if candidate_labels is None:
        candidate_labels = [chr(ord("A") + i) for i in range(top_k)]

    scores_arr = np.asarray(candidate_scores, dtype=np.float64)
    n_avail_qrels = len(qrels_dict)
    n_preds = min(len(scores_arr), n_avail_qrels)

    if n_preds == 0:
        return {
            f"composite_ndcg@{top_k}": 0.0,
            f"ndcg@{top_k}": 0.0,
            f"mrr@{top_k}": 0.0,
            f"ndcg_head@{top_k}": 0.0,
            f"ndcg_torso@{top_k}": 0.0,
            f"ndcg_tail@{top_k}": 0.0,
        }

    scores_arr = scores_arr[:n_preds]
    q_keys = [f"q_{idx}" for idx in range(n_preds)]

    run_dict: Dict[str, Dict[str, float]] = {
        q_keys[idx]: {
            candidate_labels[c_idx]: float(scores_arr[idx, c_idx])
            for c_idx in range(len(candidate_labels))
        }
        for idx in range(n_preds)
    }
    active_qrels_dict: Dict[str, Dict[str, int]] = {
        k: qrels_dict[k] for k in q_keys
    }

    ndcg_key = f"ndcg@{top_k}"
    mrr_key = f"mrr@{top_k}"

    overall_res = evaluate(
        Qrels(active_qrels_dict),
        Run(run_dict),
        [ndcg_key, mrr_key],
    )

    if isinstance(query_tiers, dict):
        tiers_list = [int(query_tiers.get(k, 0)) for k in q_keys]
    else:
        tiers_arr = np.asarray(query_tiers, dtype=np.int64).reshape(-1)
        tiers_list = [
            int(tiers_arr[idx]) if idx < tiers_arr.size else 0
            for idx in range(n_preds)
        ]

    tier_ndcgs: Dict[int, float] = {}
    active_weight_sum = 0.0
    weighted_ndcg_sum = 0.0

    for tier_id, tier_name in TIER_NAMES.items():
        tier_q_keys = [
            q_keys[idx] for idx, t in enumerate(tiers_list) if t == tier_id
        ]
        w = float(tier_weights[tier_id])
        if len(tier_q_keys) > 0:
            tier_qrels = Qrels({k: active_qrels_dict[k] for k in tier_q_keys})
            tier_run = Run({k: run_dict[k] for k in tier_q_keys})
            tier_eval = evaluate(tier_qrels, tier_run, ndcg_key)
            tier_val = float(
                tier_eval[ndcg_key] if isinstance(tier_eval, dict) else tier_eval
            )
            tier_ndcgs[tier_id] = tier_val
            weighted_ndcg_sum += w * tier_val
            active_weight_sum += w
        else:
            tier_ndcgs[tier_id] = 0.0

    total_target_weight = float(sum(tier_weights))
    if active_weight_sum > 0.0:
        composite_ndcg = weighted_ndcg_sum * (total_target_weight / active_weight_sum)
    else:
        composite_ndcg = float(overall_res[ndcg_key]) * total_target_weight

    return {
        f"composite_ndcg@{top_k}": float(composite_ndcg),
        ndcg_key: float(overall_res[ndcg_key]),
        mrr_key: float(overall_res[mrr_key]),
        f"ndcg_head@{top_k}": tier_ndcgs[0],
        f"ndcg_torso@{top_k}": tier_ndcgs[1],
        f"ndcg_tail@{top_k}": tier_ndcgs[2],
    }


def make_reranker_compute_metrics(
    qrels_dict: Dict[str, Dict[str, int]],
    query_tiers: Union[Sequence[int], np.ndarray, Dict[str, int]],
    top_k: int = 5,
    candidate_labels: Optional[List[str]] = None,
    tier_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Callable:
    """
    Factory returning a Hugging Face `Trainer` `compute_metrics` callback
    that evaluates `composite_ndcg@K`, per-tier `ndcg_{head,torso,tail}@K`,
    `ndcg@K`, and `mrr@K`.
    """

    def compute_metrics(eval_pred) -> Dict[str, float]:
        candidate_scores, _ = eval_pred
        return compute_composite_reranker_metrics(
            candidate_scores=candidate_scores,
            qrels_dict=qrels_dict,
            query_tiers=query_tiers,
            top_k=top_k,
            candidate_labels=candidate_labels,
            tier_weights=tier_weights,
        )

    return compute_metrics