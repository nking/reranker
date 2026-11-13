from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer

from movie_lens_reranker.Callbacks import LayerDiagnosticsCallback

class RankerEmbeddingProjector(nn.Linear):
    """Linear projection layer mapping stage-2 ranker embeddings into the LLM hidden dimension."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)


class ListwiseSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that:
      1. Injects projected Stage-2 ranker embeddings into `<|vision_pad|>` token positions[cite: 5]
      2. Aligns training with `composite_ndcg@K` via a 5-way Listwise Candidate Focal +
         Tier-Weighted (IPW) softmax loss at `pred_pos`[cite: 5]
      3. Supports Option A Cross-Stage Score Distillation (`distill_alpha > 0`) using
         z-score normalized Stage-2 `candidate_scores` via temperature-scaled KL divergence.
    """

    def __init__(
        self,
        *args,
        include_ranker_embeddings: bool = True,
        ranker_emb_dim: int = 24,
        emb_token_id: Optional[int] = None,
        candidate_token_ids: Optional[List[int]] = None,
        tier_loss_weights: Tuple[float, float, float] = (0.333, 0.333, 0.333),
        focal_gamma: float = 2.0,
        listwise_loss_alpha: float = 0.5,
        distill_alpha: float = 0.15,
        distill_temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_ranker_embeddings = include_ranker_embeddings
        self.ranker_emb_dim = ranker_emb_dim
        self.emb_token_id = emb_token_id
        self.candidate_token_ids = candidate_token_ids
        self.tier_loss_weights = tier_loss_weights
        self.focal_gamma = float(focal_gamma)
        self.listwise_loss_alpha = float(listwise_loss_alpha)
        self.distill_alpha = float(distill_alpha)
        self.distill_temperature = float(distill_temperature)

        self._current_ranker_embs: Optional[torch.Tensor] = None
        self._current_input_ids: Optional[torch.Tensor] = None
        self._embed_hook_handle = None
        self._current_ranker_embs: Optional[torch.Tensor] = None
        self._current_input_ids: Optional[torch.Tensor] = None
        self._embed_hook_handle = None
        self._latest_step_diagnostics: Dict[str, float] = {}
        
        # Wrap the data collator so `ranker_embs`, `pos_tier`, and `candidate_scores` survive collation[cite: 5]
        orig_collator = self.data_collator
        
        
        def _collator_with_extras(features: List[Dict[str, Any]]) -> Dict[str, Any]:
            ranker_embs_list = [
                f.get("ranker_embs") for f in features if "ranker_embs" in f
            ]
            pos_tier_list = [f.get("pos_tier") for f in features if "pos_tier" in f]
            cand_scores_list = [
                f.get("candidate_scores") for f in features if "candidate_scores" in f
            ]

            stripped_features = [
                {
                    k: v
                    for k, v in f.items()
                    if k
                    not in (
                        "ranker_embs",
                        "pos_tier",
                        "candidate_scores",
                        "movie_id",
                        "label_rels_list",
                    )
                }
                for f in features
            ]
            batch = orig_collator(stripped_features)

            if self.include_ranker_embeddings and len(ranker_embs_list) == len(
                features
            ):
                batch["ranker_embs"] = torch.tensor(
                    ranker_embs_list, dtype=torch.float32
                )
            if len(pos_tier_list) == len(features):
                batch["pos_tier"] = torch.tensor(pos_tier_list, dtype=torch.long)
            if len(cand_scores_list) == len(features):
                batch["candidate_scores"] = torch.tensor(
                    cand_scores_list, dtype=torch.float32
                )

            return batch

        #the collator strips out the non-text columns and puts them in a batch dictionary
        self.data_collator = _collator_with_extras

        if self.include_ranker_embeddings:
            self._init_projector_and_hook()
        
        diag_cb = LayerDiagnosticsCallback(self)
        self.callback_handler.callbacks.insert(0, diag_cb)

    def _unwrap_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        """Unwraps DDP / Accelerate model wrappers (called by both Trainer and export.py)[cite: 5]."""
        target = model if model is not None else self.model
        if hasattr(self, "accelerator") and self.accelerator is not None:
            return self.accelerator.unwrap_model(target)
        while hasattr(target, "module"):
            target = target.module
        return target

    _unwrap_base_model = _unwrap_model

    def _init_projector_and_hook(self):
        unwrapped = self._unwrap_model(self.model)
        hidden_size = unwrapped.config.hidden_size
        device = next(unwrapped.parameters()).device

        projector = RankerEmbeddingProjector(self.ranker_emb_dim, hidden_size).to(
            device=device, dtype=torch.float32
        )

        # Register on the model BEFORE DDP wrapping so DDP synchronizes projector gradients[cite: 5]
        unwrapped.ranker_emb_projector = projector
        self.ranker_emb_projector = projector

        embed_tokens = unwrapped.get_input_embeddings()

        def _inject_ranker_embs_hook(module, hook_inputs, output):
            if (
                not self.include_ranker_embeddings
                or self._current_ranker_embs is None
                or self._current_input_ids is None
                or self.emb_token_id is None
            ):
                return output

            input_ids = self._current_input_ids
            ranker_embs = self._current_ranker_embs.to(
                device=output.device, dtype=torch.float32
            )
            proj_module = getattr(
                self._unwrap_model(self.model),
                "ranker_emb_projector",
                self.ranker_emb_projector,
            )
            proj_embs = proj_module(ranker_embs).to(dtype=output.dtype)

            mask = input_ids == self.emb_token_id
            batch_size, n_slots, _ = proj_embs.shape
            out_cloned = output.clone()

            for b in range(min(batch_size, out_cloned.size(0))):
                positions = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
                n_match = min(positions.numel(), n_slots)
                if n_match > 0:
                    out_cloned[b, positions[:n_match]] = proj_embs[b, :n_match]

            return out_cloned

        self._embed_hook_handle = embed_tokens.register_forward_hook(
            _inject_ranker_embs_hook
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        ranker_embs = inputs.pop("ranker_embs", None)
        pos_tier = inputs.pop("pos_tier", None)
        candidate_scores = inputs.pop("candidate_scores", None)

        if self.include_ranker_embeddings and ranker_embs is not None:
            self._current_ranker_embs = ranker_embs
            self._current_input_ids = inputs.get("input_ids")
        else:
            self._current_ranker_embs = None
            self._current_input_ids = None

        try:
            outputs = model(**inputs)
        finally:
            self._current_ranker_embs = None
            self._current_input_ids = None

        base_sft_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        labels = inputs.get("labels")
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

        if (
            labels is not None
            and logits is not None
            and self.candidate_token_ids is not None
            and len(self.candidate_token_ids) > 0
            and (self.listwise_loss_alpha > 0.0 or self.distill_alpha > 0.0)
        ):
            valid_mask = labels != -100
            has_target = valid_mask.any(dim=-1)
            if has_target.any():
                first_target_idx = valid_mask.int().argmax(dim=-1)
                pred_pos = (first_target_idx - 1).clamp(min=0)
                batch_indices = torch.arange(logits.size(0), device=logits.device)

                cand_token_tensor = torch.tensor(
                    self.candidate_token_ids,
                    device=logits.device,
                    dtype=torch.long,
                )
                first_step_cand_logits = logits[batch_indices, pred_pos][
                    :, cand_token_tensor
                ].float()

                target_token_ids = labels[batch_indices, first_target_idx]
                match_matrix = (
                    target_token_ids.unsqueeze(-1) == cand_token_tensor.unsqueeze(0)
                )
                valid_cand_row = has_target & match_matrix.any(dim=-1)

                if valid_cand_row.any():
                    target_cand_idx = match_matrix.int().argmax(dim=-1)
                    log_probs = F.log_softmax(first_step_cand_logits, dim=-1)
                    pos_log_probs = log_probs[batch_indices, target_cand_idx]
                    pos_probs = pos_log_probs.exp().clamp(min=1e-6, max=1.0 - 1e-6)

                    # Focal modulation: (1 - p_pos)^gamma[cite: 5]
                    focal_factor = (1.0 - pos_probs) ** self.focal_gamma
                    per_sample_ce = -pos_log_probs * focal_factor

                    # Popularity tier IPW weighting (0=head, 1=torso, 2=tail)[cite: 5]
                    if pos_tier is not None:
                        tier_w_tensor = torch.tensor(
                            self.tier_loss_weights,
                            device=logits.device,
                            dtype=torch.float32,
                        )
                        clamped_tiers = pos_tier.to(device=logits.device).clamp(
                            min=0, max=len(self.tier_loss_weights) - 1
                        )
                        sample_tier_weights = tier_w_tensor[clamped_tiers]
                        per_sample_ce = per_sample_ce * sample_tier_weights

                    listwise_focal_loss = per_sample_ce[valid_cand_row].mean()

                    # Option A: Cross-Stage Score Distillation from Stage-2 candidate_scores
                    kd_loss = torch.tensor(
                        0.0, device=logits.device, dtype=torch.float32
                    )
                    effective_distill_alpha = 0.0
                    if candidate_scores is not None and self.distill_alpha > 0.0:
                        scores_f = candidate_scores.to(
                            device=logits.device, dtype=torch.float32
                        )[valid_cand_row]
                        # Row-wise z-score normalization ensures invariant scale across queries
                        mean_s = scores_f.mean(dim=-1, keepdim=True)
                        std_s = scores_f.std(dim=-1, keepdim=True).clamp(min=1e-6)
                        norm_scores = (scores_f - mean_s) / std_s

                        tau = max(1e-3, self.distill_temperature)
                        teacher_probs = F.softmax(norm_scores / tau, dim=-1)
                        student_log_probs = F.log_softmax(
                            first_step_cand_logits[valid_cand_row] / tau, dim=-1
                        )
                        kd_loss = F.kl_div(
                            student_log_probs, teacher_probs, reduction="batchmean"
                        ) * (tau ** 2)
                        effective_distill_alpha = self.distill_alpha

                    supervised_weight = max(
                        0.0, 1.0 - self.listwise_loss_alpha - effective_distill_alpha
                    )
                    total_loss = (
                        supervised_weight * base_sft_loss
                        + self.listwise_loss_alpha * listwise_focal_loss.to(dtype=base_sft_loss.dtype)
                        + effective_distill_alpha * kd_loss.to(dtype=base_sft_loss.dtype)
                    )
                    
                    # Stash unweighted sub-losses and candidate logit distribution diagnostics
                    if model.training:
                        with torch.no_grad():
                            valid_log_probs = log_probs[valid_cand_row]
                            probs = valid_log_probs.exp()
                            entropy = -(probs * valid_log_probs).sum(
                                dim=-1).mean()
                            cand_std = (
                                first_step_cand_logits[valid_cand_row]
                                .std(dim=-1)
                                .mean()
                            )
                            top1_prob = probs.max(dim=-1).values.mean()
                            self._latest_step_diagnostics = {
                                "loss_sft_raw": base_sft_loss.detach().float().item(),
                                "loss_listwise_raw": listwise_focal_loss.detach().float().item(),
                                "loss_distill_raw": kd_loss.detach().float().item(),
                                "cand_logit_entropy": entropy.item(),
                                "cand_logit_std": cand_std.item(),
                                "cand_top1_prob": top1_prob.item(),
                            }
                    
                    if isinstance(outputs, dict):
                        outputs["loss"] = total_loss
                    
                    return (total_loss, outputs) if return_outputs else total_loss
                    
        return (base_sft_loss, outputs) if return_outputs else base_sft_loss