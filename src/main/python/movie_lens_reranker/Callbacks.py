import math
import re
import sys
from typing import Dict, List
import torch
from transformers import TrainerCallback

class LayerDiagnosticsCallback(TrainerCallback):
    """
    Logs lightweight LoRA update norms by depth (early/mid/late layers),
    LoRA effective rank saturation ratio, and RankerEmbeddingProjector norms
    directly into `trainer.state.log_history`.
    """

    def __init__(self, trainer_ref):
        self.trainer_ref = trainer_ref

    @torch.no_grad()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        model = self.trainer_ref._unwrap_model()
        n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 24)
        early_cut = max(1, n_layers // 3)
        mid_cut = max(2, (2 * n_layers) // 3)

        bucket_norms: Dict[str, List[float]] = {"early": [], "mid": [], "late": []}
        rank_ratios: List[float] = []

        # Collect LoRA A/B pairs across PEFT modules
        lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        lora_scalings: Dict[str, float] = {}

        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                active = getattr(mod, "active_adapter", "default")
                if isinstance(active, (list, tuple)):
                    active = active[0]
                if active in mod.lora_A and active in mod.lora_B:
                    lora_pairs[name] = {
                        "A": mod.lora_A[active].weight.detach().float(),
                        "B": mod.lora_B[active].weight.detach().float(),
                    }
                    scaling_dict = getattr(mod, "scaling", {})
                    lora_scalings[name] = float(
                        scaling_dict.get(active, 1.0)
                        if isinstance(scaling_dict, dict)
                        else 1.0
                    )

        for name, pair in lora_pairs.items():
            A, B = pair["A"], pair["B"]  # A: [r, d_in], B: [d_out, r]
            r = A.size(0)
            scale = lora_scalings.get(name, 1.0)

            # Fast O(r^3) Gram matrix M = (B^T B) @ (A A^T) of shape [r, r]
            gram = (B.T @ B) @ (A @ A.T)
            frob_sq = torch.trace(gram).clamp(min=0.0).item()
            eff_frob_norm = scale * math.sqrt(frob_sq)

            m_layer = re.search(r"\.layers\.(\d+)\.", name)
            if m_layer:
                layer_idx = int(m_layer.group(1))
                if layer_idx < early_cut:
                    bucket_norms["early"].append(eff_frob_norm)
                elif layer_idx < mid_cut:
                    bucket_norms["mid"].append(eff_frob_norm)
                else:
                    bucket_norms["late"].append(eff_frob_norm)

            if r > 1 and frob_sq > 1e-12:
                # Spectral norm squared = largest singular value squared of (B @ A)
                sym_gram = (B.T @ B) * (A @ A.T)
                eigvals = torch.linalg.eigvalsh(sym_gram)
                max_sv_sq = eigvals[-1].clamp(min=1e-12).item()
                stable_rank = frob_sq / max_sv_sq
                rank_ratios.append(min(1.0, stable_rank / float(r)))

        for bucket, vals in bucket_norms.items():
            if vals:
                logs[f"lora_update_norm_{bucket}"] = round(sum(vals) / len(vals), 6)

        if rank_ratios:
            logs["lora_effective_rank_ratio"] = round(
                sum(rank_ratios) / len(rank_ratios), 4
            )

        # Projector weight & gradient norm diagnostics
        proj = getattr(model, "ranker_emb_projector", None)
        if proj is not None and hasattr(proj, "weight"):
            logs["projector_weight_norm"] = round(
                proj.weight.detach().float().norm().item(), 5
            )
            if proj.weight.grad is not None:
                logs["projector_grad_norm"] = round(
                    proj.weight.grad.detach().float().norm().item(), 5
                )

        # Stashed per-step loss components and candidate logit entropy from compute_loss
        for k, v in getattr(self.trainer_ref, "_latest_step_diagnostics", {}).items():
            logs[k] = round(float(v), 5)
            

class StepHeartbeatCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % max(1, args.logging_steps) == 0:
            print(
                f"\n[Heartbeat] Starting optimizer step {state.global_step + 1}/{state.max_steps}...",
                file=sys.stderr,
                flush=True,
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % max(1, args.logging_steps) == 0:
            print(
                f"[Heartbeat] Completed optimizer step {state.global_step}/{state.max_steps}!",
                file=sys.stderr,
                flush=True,
            )
