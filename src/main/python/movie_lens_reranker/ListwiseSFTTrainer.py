import os
from typing import Optional

import torch
from accelerate import PartialState
from trl import SFTConfig, SFTTrainer
from safetensors.torch import load_file, save_file

"""
For multi-GPU:
   torchrun --nproc_per_node=2
   accelerate wraps the model in torch.nn.parallel.DistributedDataParallel
   
"""

class ListwiseSFTTrainer(SFTTrainer):
    """
    DDP-ready SFTTrainer subclass that:
    1. Synchronizes dataset tokenization across ranks via `main_process_first()`.
    2. Injects 24-dim GATv2 embeddings via `ranker_emb_projector` safely under DDP.
    3. Runs a single forward pass during evaluation without TRL's deprecated `return_outputs=True`.
    4. Saves and restores `ranker_projector.safetensors` on the main process.
    """

    def __init__(
        self,
        *args,
        include_ranker_embeddings: bool = False,
        ranker_emb_dim: int = None,
        emb_token_id: int = None,
        **kwargs,
    ):
        if ranker_emb_dim is None:
            raise ValueError("ranker_emb_dim is required")

        # Ensure Rank 0 runs SFTTrainer's internal dataset tokenization (.map) first
        with PartialState().main_process_first():
            super().__init__(*args, **kwargs)

        self.include_ranker_embeddings = include_ranker_embeddings
        self.ranker_emb_dim = ranker_emb_dim
        self.emb_token_id = emb_token_id
        self._current_input_ids = None
        self._current_ranker_embs = None

        if self.include_ranker_embeddings:
            if self.emb_token_id is None:
                raise ValueError(
                    "emb_token_id must be provided when include_ranker_embeddings=True"
                )

            unwrapped = self._unwrap_model()
            hidden_size = unwrapped.config.hidden_size
            device = next(unwrapped.parameters()).device
            projector = torch.nn.Linear(
                ranker_emb_dim,
                hidden_size,
                device=device,
                dtype=torch.float32,
            )
            projector.requires_grad_(True)
            unwrapped.ranker_emb_projector = projector

            orig_collator = self.data_collator

            def collate_with_ranker_embs(features):
                copied_features = [dict(f) for f in features]
                embs = [f.pop("ranker_embs", None) for f in copied_features]
                batch = orig_collator(copied_features)
                if embs[0] is not None:
                    batch["ranker_embs"] = torch.tensor(embs, dtype=torch.float32)
                return batch

            self.data_collator = collate_with_ranker_embs

            embed_layer = unwrapped.get_input_embeddings()
            embed_layer.register_forward_hook(self._inject_ranker_embeddings_hook)

    def _unwrap_model(self):
        if hasattr(self, "accelerator") and self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model.module if hasattr(self.model, "module") else self.model

    def _inject_ranker_embeddings_hook(self, module, module_inputs, output):
        if not self.include_ranker_embeddings or self._current_ranker_embs is None:
            return output

        unwrapped = self._unwrap_model()
        projector = unwrapped.ranker_emb_projector
        input_ids = module_inputs[0]

        ranker_embs = self._current_ranker_embs.to(
            device=output.device,
            dtype=projector.weight.dtype,
        )
        projected = projector(ranker_embs).to(dtype=output.dtype)

        # Loop-free vectorized replacement at `<|vision_pad|>` token positions
        mask = input_ids == self.emb_token_id
        num_slots = projected.size(1)
        slot_idx = (mask.long().cumsum(dim=1) - 1).clamp(min=0, max=num_slots - 1)
        gather_idx = slot_idx.unsqueeze(-1).expand(-1, -1, projected.size(-1))
        gathered_embs = torch.gather(projected, dim=1, index=gather_idx)
        return torch.where(mask.unsqueeze(-1), gathered_embs, output)

    def _prepare_inputs(self, inputs):
        ranker_embs = inputs.pop("ranker_embs", None)
        inputs = super()._prepare_inputs(inputs)
        self._current_input_ids = inputs.get("input_ids")
        if ranker_embs is not None:
            self._current_ranker_embs = ranker_embs
        return inputs

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if hasattr(self, "accelerator") and not self.accelerator.is_main_process:
            return

        save_dir = output_dir if output_dir is not None else self.args.output_dir
        unwrapped = self._unwrap_model()
        if self.include_ranker_embeddings and hasattr(
            unwrapped, "ranker_emb_projector"
        ):
            os.makedirs(save_dir, exist_ok=True)
            proj_state = {
                k: v.detach().cpu().to(dtype=torch.float32).contiguous()
                for k, v in unwrapped.ranker_emb_projector.state_dict().items()
            }
            save_file(
                proj_state, os.path.join(save_dir, "ranker_projector.safetensors")
            )

    def _load_best_model(self):
        super()._load_best_model()
        unwrapped = self._unwrap_model()
        if (
            self.include_ranker_embeddings
            and self.state.best_model_checkpoint is not None
            and hasattr(unwrapped, "ranker_emb_projector")
        ):
            proj_st_path = os.path.join(
                self.state.best_model_checkpoint, "ranker_projector.safetensors"
            )
            if os.path.exists(proj_st_path):
                device = next(unwrapped.ranker_emb_projector.parameters()).device
                proj_state = load_file(proj_st_path, device=str(device))
                proj_state = {
                    k: v.to(
                        device=device,
                        dtype=unwrapped.ranker_emb_projector.weight.dtype,
                    )
                    for k, v in proj_state.items()
                }
                unwrapped.ranker_emb_projector.load_state_dict(proj_state)

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
                loss = outputs.loss.mean().detach()
                logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits.detach(), labels.detach())