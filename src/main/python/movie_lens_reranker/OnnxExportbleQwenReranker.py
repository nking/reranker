from typing import List, Optional
import torch
import torch.nn as nn

class OnnxExportableQwenReranker(nn.Module):
    """
    Wraps the merged Qwen2 transformer backbone (`qwen_model.model`), the trained
    24->hidden_size GATv2 projector, and the 5-candidate output projection head
    into a single loop-free, ONNX-traceable module.
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        projector: Optional[nn.Linear],
        emb_token_id: int,
        candidate_token_ids: List[int],
        include_ranker_embeddings: bool = True,
    ):
        super().__init__()
        # Use the transformer backbone directly so we only project the final token over the 5 candidates
        self.backbone = qwen_model.model
        self.projector = projector
        self.emb_token_id = int(emb_token_id)
        self.include_ranker_embeddings = include_ranker_embeddings
        self.num_emb_slots = 1 + len(candidate_token_ids)

        cand_ids_tensor = torch.tensor(candidate_token_ids, dtype=torch.long)
        self.register_buffer("candidate_token_ids", cand_ids_tensor, persistent=True)

        # Extract only the [top_k, hidden_size] rows of lm_head corresponding to [A]..[E]
        lm_head = qwen_model.get_output_embeddings()
        hidden_size = lm_head.weight.size(1)
        top_k = len(candidate_token_ids)
        self.candidate_head = nn.Linear(
            hidden_size,
            top_k,
            bias=lm_head.bias is not None,
            dtype=lm_head.weight.dtype,
            device=lm_head.weight.device,
        )
        with torch.no_grad():
            self.candidate_head.weight.copy_(
                torch.index_select(lm_head.weight, dim=0, index=cand_ids_tensor)
            )
            if lm_head.bias is not None:
                self.candidate_head.bias.copy_(
                    torch.index_select(lm_head.bias, dim=0, index=cand_ids_tensor)
                )

    def forward(
        self,
        input_ids: torch.Tensor,       # int64 [batch_size, seq_len] (left-padded)
        attention_mask: torch.Tensor,  # int64 [batch_size, seq_len]
        ranker_embs: torch.Tensor,     # float32 [batch_size, 1 + top_k, 24]
    ) -> torch.Tensor:                 # float32 [batch_size, top_k] log-probabilities
        inputs_embeds = self.backbone.embed_tokens(input_ids)

        if self.include_ranker_embeddings and self.projector is not None:
            projected = self.projector(
                ranker_embs.to(dtype=self.projector.weight.dtype)
            ).to(dtype=inputs_embeds.dtype)

            # Loop-free ONNX-traceable replacement of `<|vision_pad|>` token positions
            mask = input_ids == self.emb_token_id
            slot_idx = (mask.long().cumsum(dim=1) - 1).clamp(
                min=0, max=self.num_emb_slots - 1
            )
            gather_idx = slot_idx.unsqueeze(-1).expand(-1, -1, projected.size(-1))
            gathered_embs = torch.gather(projected, dim=1, index=gather_idx)
            inputs_embeds = torch.where(
                mask.unsqueeze(-1), gathered_embs, inputs_embeds
            )

        # Explicit position_ids so left-padded batches have identical RoPE positions to unpadded inputs
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)

        backbone_outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = backbone_outputs[0]

        # Use non-negative index `seq_len - 1` to avoid ONNX aten::index negative-index warnings
        last_token_idx = inputs_embeds.size(1) - 1
        last_hidden = hidden_states[:, last_token_idx, :]

        cand_logits = self.candidate_head(last_hidden)
        return torch.log_softmax(cand_logits.float(), dim=-1)