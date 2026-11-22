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
    Computes overall NDCG@K, MRR@K, Recall@K, Precision@{K,1,5},
    per-tier NDCG@K and Recall@K (head, torso, tail),
    `composite_ndcg@K` (sum of tier NDCGs weighted by `tier_weights`),
    and candidate score distribution diagnostics (`cand_logit_entropy`,
    `cand_logit_std`, `cand_top1_prob`).
    """
    scores_arr = np.asarray(candidate_scores, dtype=np.float64)
    if candidate_labels is None:
        n_cands = scores_arr.shape[1] if scores_arr.ndim == 2 else top_k
        candidate_labels = [chr(ord("A") + i) for i in range(n_cands)]

    n_avail_qrels = len(qrels_dict)
    n_preds = min(len(scores_arr), n_avail_qrels)

    ndcg_key = f"ndcg@{top_k}"
    mrr_key = f"mrr@{top_k}"
    recall_key = f"recall@{top_k}"
    prec_k_key = f"precision@{top_k}"
    prec_1_key = "precision@1"
    prec_5_key = "precision@5"

    if n_preds == 0:
        return {
            f"composite_ndcg@{top_k}": 0.0,
            ndcg_key: 0.0,
            mrr_key: 0.0,
            recall_key: 0.0,
            prec_k_key: 0.0,
            prec_1_key: 0.0,
            prec_5_key: 0.0,
            f"ndcg_head@{top_k}": 0.0,
            f"ndcg_torso@{top_k}": 0.0,
            f"ndcg_tail@{top_k}": 0.0,
            f"recall_head@{top_k}": 0.0,
            f"recall_torso@{top_k}": 0.0,
            f"recall_tail@{top_k}": 0.0,
            "cand_logit_entropy": 0.0,
            "cand_logit_std": 0.0,
            "cand_top1_prob": 0.0,
        }

    scores_arr = scores_arr[:n_preds]
    q_keys = [f"q_{idx}" for idx in range(n_preds)]

    # Candidate probability distribution diagnostics across queries
    shifted_scores = scores_arr - np.max(scores_arr, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.clip(
        np.sum(exp_scores, axis=-1, keepdims=True), 1e-12, None
    )
    cand_logit_entropy = float(
        -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1).mean()
    )
    cand_logit_std = float(np.std(scores_arr, axis=-1).mean())
    cand_top1_prob = float(np.max(probs, axis=-1).mean())

    n_cols = (
        min(len(candidate_labels), scores_arr.shape[1])
        if scores_arr.ndim == 2
        else len(candidate_labels)
    )
    run_dict: Dict[str, Dict[str, float]] = {
        q_keys[idx]: {
            candidate_labels[c_idx]: float(scores_arr[idx, c_idx])
            for c_idx in range(n_cols)
        }
        for idx in range(n_preds)
    }
    active_qrels_dict: Dict[str, Dict[str, int]] = {
        k: qrels_dict[k] for k in q_keys
    }

    overall_metric_keys = list(
        dict.fromkeys(
            [ndcg_key, mrr_key, recall_key, prec_k_key, prec_1_key, prec_5_key]
        )
    )
    overall_res = evaluate(
        Qrels(active_qrels_dict),
        Run(run_dict),
        overall_metric_keys,
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
    tier_recalls: Dict[int, float] = {}
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
            tier_eval = evaluate(
                tier_qrels, tier_run, [ndcg_key, recall_key]
            )
            tier_ndcg_val = float(tier_eval[ndcg_key])
            tier_recall_val = float(tier_eval[recall_key])
            tier_ndcgs[tier_id] = tier_ndcg_val
            tier_recalls[tier_id] = tier_recall_val
            weighted_ndcg_sum += w * tier_ndcg_val
            active_weight_sum += w
        else:
            tier_ndcgs[tier_id] = 0.0
            tier_recalls[tier_id] = 0.0

    total_target_weight = float(sum(tier_weights))
    if active_weight_sum > 0.0:
        composite_ndcg = weighted_ndcg_sum * (total_target_weight / active_weight_sum)
    else:
        composite_ndcg = float(overall_res[ndcg_key]) * total_target_weight

    return {
        f"composite_ndcg@{top_k}": float(composite_ndcg),
        ndcg_key: float(overall_res[ndcg_key]),
        mrr_key: float(overall_res[mrr_key]),
        recall_key: float(overall_res[recall_key]),
        prec_k_key: float(overall_res[prec_k_key]),
        prec_1_key: float(overall_res[prec_1_key]),
        prec_5_key: float(overall_res[prec_5_key]),
        f"ndcg_head@{top_k}": tier_ndcgs[0],
        f"ndcg_torso@{top_k}": tier_ndcgs[1],
        f"ndcg_tail@{top_k}": tier_ndcgs[2],
        f"recall_head@{top_k}": tier_recalls[0],
        f"recall_torso@{top_k}": tier_recalls[1],
        f"recall_tail@{top_k}": tier_recalls[2],
        "cand_logit_entropy": round(cand_logit_entropy, 5),
        "cand_logit_std": round(cand_logit_std, 5),
        "cand_top1_prob": round(cand_top1_prob, 5),
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
    that evaluates `composite_ndcg@K`, per-tier `ndcg_{head,torso,tail}@K`
    and `recall_{head,torso,tail}@K`, `ndcg@K`, `mrr@K`, `recall@K`, and
    `precision@{K,1,5}`.
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

def compute_interaction_and_calibration_diagnostics(
    candidate_scores: np.ndarray,
    qrels_dict: Dict[str, Dict[str, int]],
    query_tiers: Union[Sequence[int], np.ndarray, Dict[str, int]],
    top_k: int = 5,
    candidate_labels: Optional[List[str]] = None,
    tier_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    stage2_candidate_scores: Optional[np.ndarray] = None,
    n_ece_bins: int = 10,
) -> Dict[str, object]:
    """
    Computes 3B quality & interaction diagnostics on predicted candidate scores/log-probs:
      1. Calibration: 10-bin ECE, multi-class Brier score, NLL, and post-hoc temperature T*.
      2. Stage-2 Reranker Lift: delta vs. upstream `stage2_candidate_scores` on the exact same split.
      3. Causal-Mask Slot Bias: top-1 prediction distribution, entropy ratio, and first-half vs.
         second-half prediction rate and accuracy across candidate slots 'A'..'T'.
    """
    scores = np.asarray(candidate_scores, dtype=np.float64)
    n_preds = min(len(scores), len(qrels_dict))
    if n_preds == 0 or scores.ndim != 2:
        return {}

    scores = scores[:n_preds]
    n_cands = int(scores.shape[1])
    if candidate_labels is None:
        candidate_labels = [chr(ord("A") + i) for i in range(n_cands)]
    label_to_idx = {lbl: idx for idx, lbl in enumerate(candidate_labels[:n_cands])}

    # Extract ground-truth positive slot index per query
    pos_indices = np.full(n_preds, -1, dtype=np.int64)
    for row_idx in range(n_preds):
        rel_map = qrels_dict.get(f"q_{row_idx}", {})
        for lbl, rel in rel_map.items():
            if int(rel) > 0 and lbl in label_to_idx:
                pos_indices[row_idx] = label_to_idx[lbl]
                break

    valid_mask = pos_indices >= 0
    valid_scores = scores[valid_mask]
    valid_pos = pos_indices[valid_mask]
    n_valid = int(valid_pos.size)

    def _softmax_and_log_softmax(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        sum_exp = np.clip(np.sum(exp_z, axis=-1, keepdims=True), 1e-12, None)
        probs = exp_z / sum_exp
        log_probs = shifted - np.log(sum_exp)
        return probs, log_probs

    def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_val = 0.0
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            in_bin = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
            cnt = int(np.sum(in_bin))
            if cnt > 0:
                ece_val += (cnt / float(len(conf))) * abs(
                    float(np.mean(correct[in_bin])) - float(np.mean(conf[in_bin]))
                )
        return float(ece_val)

    # 1. Calibration + post-hoc temperature scaling T*
    calibration: Dict[str, float] = {}
    if n_valid > 0:
        probs, log_probs = _softmax_and_log_softmax(valid_scores)
        pred_top1 = np.argmax(valid_scores, axis=-1)
        top1_conf = np.max(probs, axis=-1)
        top1_correct = (pred_top1 == valid_pos).astype(np.float64)

        row_ar = np.arange(n_valid)
        raw_nll = float(-np.mean(log_probs[row_ar, valid_pos]))
        one_hot = np.zeros_like(probs)
        one_hot[row_ar, valid_pos] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
        raw_ece = _ece(top1_conf, top1_correct, n_ece_bins)

        # Coarse-to-fine 1D search for optimal temperature T* minimizing NLL
        def _nll_at_temp(t: float) -> float:
            _, lp_t = _softmax_and_log_softmax(valid_scores / t)
            return float(-np.mean(lp_t[row_ar, valid_pos]))

        coarse_ts = np.geomspace(0.1, 5.0, 50)
        coarse_nlls = [_nll_at_temp(float(t)) for t in coarse_ts]
        best_idx = int(np.argmin(coarse_nlls))
        lo_t = float(coarse_ts[max(0, best_idx - 1)])
        hi_t = float(coarse_ts[min(len(coarse_ts) - 1, best_idx + 1)])
        fine_ts = np.linspace(lo_t, hi_t, 40)
        fine_nlls = [_nll_at_temp(float(t)) for t in fine_ts]
        best_t = float(fine_ts[int(np.argmin(fine_nlls))])

        probs_t, log_probs_t = _softmax_and_log_softmax(valid_scores / best_t)
        post_temp_nll = float(-np.mean(log_probs_t[row_ar, valid_pos]))
        post_temp_ece = _ece(np.max(probs_t, axis=-1), top1_correct, n_ece_bins)

        calibration = {
            "num_valid_queries": n_valid,
            "top1_accuracy": round(float(np.mean(top1_correct)), 5),
            "mean_top1_confidence": round(float(np.mean(top1_conf)), 5),
            "confidence_minus_accuracy": round(
                float(np.mean(top1_conf) - np.mean(top1_correct)), 5
            ),
            "ece_10bin": round(raw_ece, 5),
            "brier_score": round(brier, 5),
            "nll": round(raw_nll, 5),
            "optimal_temperature": round(best_t, 4),
            "post_temp_nll": round(post_temp_nll, 5),
            "post_temp_ece_10bin": round(post_temp_ece, 5),
        }

    # 2. Causal-Mask Slot Bias across candidate positions 'A'..'T'
    all_pred_top1 = np.argmax(scores, axis=-1)
    pred_counts = np.bincount(all_pred_top1, minlength=n_cands).astype(np.float64)
    pred_rates = pred_counts / max(1.0, float(np.sum(pred_counts)))
    nonzero_rates = pred_rates[pred_rates > 0]
    slot_entropy = float(-np.sum(nonzero_rates * np.log(nonzero_rates)))
    max_entropy = float(np.log(max(2, n_cands)))

    half = n_cands // 2
    first_half_pred_rate = float(np.sum(pred_rates[:half]))
    second_half_pred_rate = float(np.sum(pred_rates[half:]))

    first_half_acc = 0.0
    second_half_acc = 0.0
    if n_valid > 0:
        valid_pred = np.argmax(valid_scores, axis=-1)
        valid_correct = (valid_pred == valid_pos).astype(np.float64)
        fh_mask = valid_pos < half
        sh_mask = valid_pos >= half
        if np.any(fh_mask):
            first_half_acc = float(np.mean(valid_correct[fh_mask]))
        if np.any(sh_mask):
            second_half_acc = float(np.mean(valid_correct[sh_mask]))

    expected_count = n_preds / float(n_cands)
    chi2_uniform = float(
        np.sum(((pred_counts - expected_count) ** 2) / max(1e-12, expected_count))
    )

    causal_slot_bias: Dict[str, object] = {
        "slot_pred_entropy": round(slot_entropy, 5),
        "slot_pred_entropy_ratio": round(slot_entropy / max(1e-12, max_entropy), 5),
        "slot_max_minus_min_rate": round(
            float(np.max(pred_rates) - np.min(pred_rates)), 5
        ),
        "first_half_pred_rate": round(first_half_pred_rate, 5),
        "second_half_pred_rate": round(second_half_pred_rate, 5),
        "second_minus_first_half_pred_rate": round(
            second_half_pred_rate - first_half_pred_rate, 5
        ),
        "first_half_top1_acc": round(first_half_acc, 5),
        "second_half_top1_acc": round(second_half_acc, 5),
        "second_minus_first_half_top1_acc": round(
            second_half_acc - first_half_acc, 5
        ),
        "slot_chi2_uniform": round(chi2_uniform, 4),
        "slot_pred_rates": {
            candidate_labels[i]: round(float(pred_rates[i]), 4)
            for i in range(n_cands)
        },
    }

    result: Dict[str, object] = {
        "calibration": calibration,
        "causal_slot_bias": causal_slot_bias,
    }

    # 3. Stage-2 Reranker Lift (reranker metrics minus upstream Stage-2 candidate_scores)
    if stage2_candidate_scores is not None:
        s2_arr = np.asarray(stage2_candidate_scores, dtype=np.float64)[:n_preds]
        if s2_arr.shape == scores.shape:
            reranker_m = compute_composite_reranker_metrics(
                candidate_scores=scores,
                qrels_dict=qrels_dict,
                query_tiers=query_tiers,
                top_k=top_k,
                candidate_labels=candidate_labels,
                tier_weights=tier_weights,
            )
            stage2_m = compute_composite_reranker_metrics(
                candidate_scores=s2_arr,
                qrels_dict=qrels_dict,
                query_tiers=query_tiers,
                top_k=top_k,
                candidate_labels=candidate_labels,
                tier_weights=tier_weights,
            )
            lift_keys = [
                f"composite_ndcg@{top_k}",
                f"ndcg@{top_k}",
                f"mrr@{top_k}",
                f"recall@{top_k}",
                "precision@1",
                f"ndcg_head@{top_k}",
                f"ndcg_torso@{top_k}",
                f"ndcg_tail@{top_k}",
            ]
            result["stage2_baseline"] = {
                k: round(float(stage2_m[k]), 5) for k in lift_keys if k in stage2_m
            }
            result["lift_vs_stage2"] = {
                k: round(float(reranker_m[k] - stage2_m[k]), 5)
                for k in lift_keys
                if k in reranker_m and k in stage2_m
            }

    return result