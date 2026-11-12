import json
import os
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from movie_lens_reranker.ListwiseSFTTrainer import ListwiseSFTTrainer
from movie_lens_reranker.OnnxExportbleQwenReranker import OnnxExportableQwenReranker

@torch.no_grad()
def apply_smoothquant_smoothing(
    qwen_model: nn.Module,
    act_max: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> nn.Module:
    """
    Song Han MIT HAN Lab SmoothQuant (Xiao et al., ICML 2023):
    Migrates activation quantization difficulty from activations into weights
    via `s_j = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha)` (normalized by geometric mean),
    folding `s^{-1}` into `Qwen2RMSNorm` and `up_proj` rows, and `s` into downstream
    Linear columns (`[q,k,v]_proj`, `[gate,up]_proj`, and `down_proj`).
    """
    layers = qwen_model.model.layers
    for idx, layer in enumerate(layers):
        prefix = f"model.layers.{idx}"

        # 1. Smooth input_layernorm -> [q_proj, k_proj, v_proj]
        q_key = f"{prefix}.self_attn.q_proj"
        if q_key in act_max:
            s_x = act_max[q_key].to(
                device=layer.input_layernorm.weight.device, dtype=torch.float32
            ).clamp(min=1e-5)
            w_cat = torch.cat(
                [
                    layer.self_attn.q_proj.weight.detach().float(),
                    layer.self_attn.k_proj.weight.detach().float(),
                    layer.self_attn.v_proj.weight.detach().float(),
                ],
                dim=0,
            )
            s_w = w_cat.abs().amax(dim=0).clamp(min=1e-5)
            scales = s_x.pow(alpha) / s_w.pow(1.0 - alpha)
            scales = (scales / scales.log().mean().exp()).clamp(min=1e-3, max=1e3)

            layer.input_layernorm.weight.copy_(
                (layer.input_layernorm.weight.float() / scales).to(
                    layer.input_layernorm.weight.dtype
                )
            )
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.weight.copy_(
                    (proj.weight.float() * scales.unsqueeze(0)).to(proj.weight.dtype)
                )

        # 2. Smooth post_attention_layernorm -> [gate_proj, up_proj]
        gate_key = f"{prefix}.mlp.gate_proj"
        if gate_key in act_max:
            s_x = act_max[gate_key].to(
                device=layer.post_attention_layernorm.weight.device, dtype=torch.float32
            ).clamp(min=1e-5)
            w_cat = torch.cat(
                [
                    layer.mlp.gate_proj.weight.detach().float(),
                    layer.mlp.up_proj.weight.detach().float(),
                ],
                dim=0,
            )
            s_w = w_cat.abs().amax(dim=0).clamp(min=1e-5)
            scales = s_x.pow(alpha) / s_w.pow(1.0 - alpha)
            scales = (scales / scales.log().mean().exp()).clamp(min=1e-3, max=1e3)

            layer.post_attention_layernorm.weight.copy_(
                (layer.post_attention_layernorm.weight.float() / scales).to(
                    layer.post_attention_layernorm.weight.dtype
                )
            )
            for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
                proj.weight.copy_(
                    (proj.weight.float() * scales.unsqueeze(0)).to(proj.weight.dtype)
                )

        # 3. Smooth post-SwiGLU activations: up_proj -> down_proj
        down_key = f"{prefix}.mlp.down_proj"
        if down_key in act_max and layer.mlp.up_proj.bias is None:
            s_x = act_max[down_key].to(
                device=layer.mlp.down_proj.weight.device, dtype=torch.float32
            ).clamp(min=1e-5)
            s_w = (
                layer.mlp.down_proj.weight.detach()
                .float()
                .abs()
                .amax(dim=0)
                .clamp(min=1e-5)
            )
            scales = s_x.pow(alpha) / s_w.pow(1.0 - alpha)
            scales = (scales / scales.log().mean().exp()).clamp(min=1e-3, max=1e3)

            layer.mlp.up_proj.weight.copy_(
                (layer.mlp.up_proj.weight.float() / scales.unsqueeze(1)).to(
                    layer.mlp.up_proj.weight.dtype
                )
            )
            layer.mlp.down_proj.weight.copy_(
                (layer.mlp.down_proj.weight.float() * scales.unsqueeze(0)).to(
                    layer.mlp.down_proj.weight.dtype
                )
            )

    return qwen_model


def _quantize_onnx_model(
    fp32_onnx_path: str,
    final_onnx_path: str,
    quantize_mode: str,
) -> str:
    """Applies W4A16 (`int4` / `awq_w4a16`) or W8A8 (`int8` / `smoothquant_w8a8`) quantization."""
    mode = quantize_mode.lower()
    if mode in (
        "int4",
        "awq",
        "awq_w4a16",
        "int8",
        "smoothquant",
        "smoothquant_w8a8",
    ):
        # Keep the 24->896 projector and 896->5 candidate_head in full FP32
        model_proto = onnx.load(fp32_onnx_path)
        excluded_nodes = [
            node.name
            for node in model_proto.graph.node
            if "projector" in node.name or "candidate_head" in node.name
        ]

        if mode in ("int4", "awq", "awq_w4a16"):
            quantizer = MatMulNBitsQuantizer(
                model=model_proto,
                bits=4,
                block_size=128,
                is_symmetric=True,
                nodes_to_exclude=excluded_nodes,
            )
            quantizer.process()
            quantizer.model.save_model_to_file(
                final_onnx_path, use_external_data_format=False
            )
        else:
            quantize_dynamic(
                model_input=fp32_onnx_path,
                model_output=final_onnx_path,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["MatMul", "Gemm"],
                nodes_to_exclude=excluded_nodes,
            )

        if fp32_onnx_path != final_onnx_path and os.path.exists(fp32_onnx_path):
            os.remove(fp32_onnx_path)
        return final_onnx_path

    if fp32_onnx_path != final_onnx_path:
        shutil.move(fp32_onnx_path, final_onnx_path)
    return final_onnx_path

def _run_torch_onnx_export(
    wrapper: torch.nn.Module,
    dummy_args: tuple,
    output_onnx_path: str,
) -> None:
    """Exports `wrapper` to ONNX opset 17 while silencing TorchScript deprecation/tracer noise[cite: 6]."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore", message=".*Exporting aten::index operator.*"
        )
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_args,
                output_onnx_path,
                input_names=["input_ids", "attention_mask", "ranker_embs"],
                output_names=["candidate_log_probs"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"},
                    "ranker_embs": {0: "batch_size"},
                    "candidate_log_probs": {0: "batch_size"},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )


def _collect_channel_activation_stats(
    qwen_model: nn.Module,
    tokenizer: Any,
    calibration_prompts: Optional[List[str]],
    emb_token_id: int,
    top_k: int,
    ranker_emb_dim: int,
    projector: Optional[nn.Module] = None,
    max_calib_samples: int = 64,
    max_seq_len: int = 768,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Runs a lightweight calibration forward pass to collect per-channel:
      - `act_max`: max(|X|) for SmoothQuant (W8A8)
      - `act_mean`: mean(|X|) for AWQ (W4A16)
    """
    device = next(qwen_model.parameters()).device
    dtype = next(qwen_model.parameters()).dtype
    qwen_model.eval()

    act_max: Dict[str, torch.Tensor] = {}
    act_sum: Dict[str, torch.Tensor] = {}
    act_count: Dict[str, int] = {}
    handles = []

    target_suffixes = ("q_proj", "gate_proj", "down_proj")

    for name, mod in qwen_model.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith(target_suffixes):

            def _make_hook(layer_name: str):
                def _hook(m, inp, out):
                    x = inp[0].detach().float()
                    flat = x.reshape(-1, x.size(-1))
                    cur_max = flat.abs().amax(dim=0).cpu()
                    cur_sum = flat.abs().sum(dim=0).cpu()
                    if layer_name not in act_max:
                        act_max[layer_name] = cur_max
                        act_sum[layer_name] = cur_sum
                        act_count[layer_name] = flat.size(0)
                    else:
                        act_max[layer_name] = torch.maximum(
                            act_max[layer_name], cur_max
                        )
                        act_sum[layer_name] += cur_sum
                        act_count[layer_name] += flat.size(0)

                return _hook

            handles.append(mod.register_forward_hook(_make_hook(name)))

    prompts = (
        calibration_prompts[:max_calib_samples]
        if calibration_prompts and len(calibration_prompts) > 0
        else None
    )

    try:
        with torch.no_grad():
            if prompts is not None:
                orig_pad = tokenizer.padding_side
                tokenizer.padding_side = "left"
                try:
                    batch_size = 8
                    for start in range(0, len(prompts), batch_size):
                        batch_prompts = prompts[start : start + batch_size]
                        enc = tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=max_seq_len,
                        )
                        input_ids = enc["input_ids"].to(device)
                        attention_mask = enc["attention_mask"].to(device)
                        inputs_embeds = qwen_model.model.embed_tokens(input_ids)

                        if projector is not None:
                            dummy_embs = torch.zeros(
                                (input_ids.size(0), 1 + top_k, ranker_emb_dim),
                                device=device,
                                dtype=torch.float32,
                            )
                            proj = projector.to(device)(dummy_embs).to(dtype=dtype)
                            mask = input_ids == emb_token_id
                            slot_idx = (mask.long().cumsum(dim=1) - 1).clamp(
                                min=0, max=top_k
                            )
                            gather_idx = slot_idx.unsqueeze(-1).expand(
                                -1, -1, proj.size(-1)
                            )
                            inputs_embeds = torch.where(
                                mask.unsqueeze(-1),
                                torch.gather(proj, dim=1, index=gather_idx),
                                inputs_embeds,
                            )

                        qwen_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                finally:
                    tokenizer.padding_side = orig_pad
            else:
                # Fallback synthetic calibration batch if no prompts were passed
                dummy_ids = torch.randint(10, 1000, (4, 64), device=device)
                dummy_mask = torch.ones_like(dummy_ids)
                qwen_model(
                    input_ids=dummy_ids,
                    attention_mask=dummy_mask,
                    use_cache=False,
                )
    finally:
        for h in handles:
            h.remove()

    act_mean = {
        k: (act_sum[k] / max(1, act_count[k])).clamp(min=1e-5)
        for k in act_sum
    }
    return act_max, act_mean



@torch.no_grad()
def apply_awq_salient_scaling(
    qwen_model: nn.Module,
    act_mean: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> nn.Module:
    """
    Song Han MIT HAN Lab AWQ (Lin et al., MLSys 2024):
    Protects salient weight channels by scaling them up by `s_j = mean(|X_j|)^alpha`
    (normalized by geometric mean) before 4-bit block quantization, folding `s_j^{-1}`
    into `Qwen2RMSNorm` and `up_proj` output rows.
    """
    layers = qwen_model.model.layers
    for idx, layer in enumerate(layers):
        prefix = f"model.layers.{idx}"
        q_key = f"{prefix}.self_attn.q_proj"
        if q_key in act_mean:
            s_x = act_mean[q_key].to(
                device=layer.input_layernorm.weight.device, dtype=torch.float32
            ).clamp(min=1e-5)
            scales = s_x.pow(alpha)
            scales = (scales / scales.log().mean().exp()).clamp(min=1e-3, max=1e3)

            layer.input_layernorm.weight.copy_(
                (layer.input_layernorm.weight.float() / scales).to(
                    layer.input_layernorm.weight.dtype
                )
            )
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.weight.copy_(
                    (proj.weight.float() * scales.unsqueeze(0)).to(proj.weight.dtype)
                )

        gate_key = f"{prefix}.mlp.gate_proj"
        if gate_key in act_mean:
            s_x = act_mean[gate_key].to(
                device=layer.post_attention_layernorm.weight.device, dtype=torch.float32
            ).clamp(min=1e-5)
            scales = s_x.pow(alpha)
            scales = (scales / scales.log().mean().exp()).clamp(min=1e-3, max=1e3)

            layer.post_attention_layernorm.weight.copy_(
                (layer.post_attention_layernorm.weight.float() / scales).to(
                    layer.post_attention_layernorm.weight.dtype
                )
            )
            for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
                proj.weight.copy_(
                    (proj.weight.float() * scales.unsqueeze(0)).to(proj.weight.dtype)
                )

        down_key = f"{prefix}.mlp.down_proj"
        if down_key in act_mean and layer.mlp.up_proj.bias is None:
            s_x = act_mean[down_key].to(
                device=layer.mlp.down_proj.weight.device, dtype=torch.float32
            ).clamp(min=1e-5)
            scales = s_x.pow(alpha)
            scales = (scales / scales.log().mean().exp()).clamp(min=1e-3, max=1e3)

            layer.mlp.up_proj.weight.copy_(
                (layer.mlp.up_proj.weight.float() / scales.unsqueeze(1)).to(
                    layer.mlp.up_proj.weight.dtype
                )
            )
            layer.mlp.down_proj.weight.copy_(
                (layer.mlp.down_proj.weight.float() * scales.unsqueeze(0)).to(
                    layer.mlp.down_proj.weight.dtype
                )
            )

    return qwen_model

def save_reusable_checkpoint(
    trained_trainer: ListwiseSFTTrainer,
    tokenizer: Any,
    checkpoint_dir: str,
    base_model_id: str,
    emb_token_id: int,
    candidate_token_ids: List[int],
    top_k: int,
    ranker_emb_dim: int,
    candidate_labels: List[str],
    include_ranker_embeddings: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves a lightweight (~11.5 MB) portable checkpoint (equivalent to Orbax best checkpoint)
    containing only the trained LoRA adapters, `ranker_projector.safetensors`, tokenizer,
    and `reranker_config.json` so alternative ONNX exports (FP32, SmoothQuant W8A8, AWQ W4A16)
    can be generated anytime later without re-training.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)
    trained_trainer.save_model(checkpoint_dir)

    unwrapped = trained_trainer._unwrap_model()
    if include_ranker_embeddings and hasattr(unwrapped, "ranker_emb_projector"):
        proj_state = {
            k: v.detach().cpu().float().contiguous()
            for k, v in unwrapped.ranker_emb_projector.state_dict().items()
        }
        save_file(
            proj_state, os.path.join(checkpoint_dir, "ranker_projector.safetensors")
        )

    config_payload: Dict[str, Any] = {
        "base_model_id": base_model_id,
        "top_k": int(top_k),
        "candidate_labels": list(candidate_labels),
        "candidate_token_ids": [int(x) for x in candidate_token_ids],
        "emb_token_id": int(emb_token_id),
        "ranker_emb_dim": int(ranker_emb_dim),
        "include_ranker_embeddings": bool(include_ranker_embeddings),
    }
    if extra_metadata:
        config_payload.update(extra_metadata)

    with open(
        os.path.join(checkpoint_dir, "reranker_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config_payload, f, indent=2)

    return checkpoint_dir


def _build_triton_artifacts(
    merged_qwen: nn.Module,
    projector_cpu: Optional[nn.Module],
    tokenizer: Any,
    emb_token_id: int,
    candidate_token_ids: List[int],
    top_k: int,
    ranker_emb_dim: int,
    candidate_labels: List[str],
    triton_model_dir: str,
    include_ranker_embeddings: bool = True,
    quantize_mode: str = "none",
    calibration_prompts: Optional[List[str]] = None,
) -> str:
    version_dir = os.path.join(triton_model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    tokenizer.save_pretrained(triton_model_dir)

    mode = quantize_mode.lower()
    if mode in ("smoothquant", "smoothquant_w8a8", "awq", "awq_w4a16"):
        act_max, act_mean = _collect_channel_activation_stats(
            qwen_model=merged_qwen,
            tokenizer=tokenizer,
            calibration_prompts=calibration_prompts,
            emb_token_id=emb_token_id,
            top_k=top_k,
            ranker_emb_dim=ranker_emb_dim,
            projector=projector_cpu if include_ranker_embeddings else None,
        )
        if mode in ("smoothquant", "smoothquant_w8a8"):
            apply_smoothquant_smoothing(merged_qwen, act_max=act_max, alpha=0.5)
        else:
            apply_awq_salient_scaling(merged_qwen, act_mean=act_mean, alpha=0.5)

    onnx_wrapper = OnnxExportableQwenReranker(
        qwen_model=merged_qwen,
        projector=projector_cpu,
        emb_token_id=emb_token_id,
        candidate_token_ids=candidate_token_ids,
        include_ranker_embeddings=include_ranker_embeddings,
    ).eval()

    dummy_batch, dummy_seq = 2, 32
    dummy_input_ids = torch.ones((dummy_batch, dummy_seq), dtype=torch.long)
    dummy_input_ids[:, 2 : 2 + 1 + top_k] = emb_token_id
    dummy_attention_mask = torch.ones((dummy_batch, dummy_seq), dtype=torch.long)
    dummy_ranker_embs = torch.randn(
        (dummy_batch, 1 + top_k, ranker_emb_dim), dtype=torch.float32
    )

    tmp_fp32_onnx = os.path.join(version_dir, "_tmp_model_fp32.onnx")
    final_onnx_path = os.path.join(version_dir, "model.onnx")

    _run_torch_onnx_export(
        onnx_wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_ranker_embs),
        tmp_fp32_onnx,
    )
    _quantize_onnx_model(
        fp32_onnx_path=tmp_fp32_onnx,
        final_onnx_path=final_onnx_path,
        quantize_mode=mode,
    )

    config_pbtxt = (
        'name: "qwen-reranker"\n'
        'platform: "onnxruntime_onnx"\n'
        "max_batch_size: 32\n"
        "input [\n"
        '  { name: "input_ids", data_type: TYPE_INT64, dims: [ -1 ] },\n'
        '  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1 ] },\n'
        f'  {{ name: "ranker_embs", data_type: TYPE_FP32, dims: [ {1 + top_k}, {ranker_emb_dim} ] }}\n'
        "]\n"
        "output [\n"
        f'  {{ name: "candidate_log_probs", data_type: TYPE_FP32, dims: [ {top_k} ] }}\n'
        "]\n"
        "instance_group [{ count: 1, kind: KIND_CPU }]\n"
    )
    with open(os.path.join(triton_model_dir, "config.pbtxt"), "w") as f:
        f.write(config_pbtxt)

    with open(os.path.join(triton_model_dir, "reranker_config.json"), "w") as f:
        json.dump(
            {
                "top_k": top_k,
                "candidate_labels": candidate_labels,
                "candidate_token_ids": candidate_token_ids,
                "emb_token_id": int(emb_token_id),
                "ranker_emb_dim": ranker_emb_dim,
                "quantize_mode": mode,
            },
            f,
            indent=2,
        )

    return final_onnx_path


def export_from_reusable_checkpoint(
    checkpoint_dir: str,
    triton_model_dir: str,
    base_model_path: Optional[str] = None,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    quantize_mode: str = "awq_w4a16",  # "none", "smoothquant_w8a8", or "awq_w4a16"
    calibration_prompts: Optional[List[str]] = None,
) -> str:
    """
    Reloads a saved ~11.5 MB Portable Checkpoint (`save_reusable_checkpoint`), merges
    LoRA into the base model on CPU, optionally runs SmoothQuant W8A8 or AWQ W4A16
    calibration, and exports a Triton ONNX repository.
    """
    with open(
        os.path.join(checkpoint_dir, "reranker_config.json"), "r", encoding="utf-8"
    ) as f:
        cfg = json.load(f)

    resolved_base = base_model_path or cfg["base_model_id"]
    top_k = int(cfg["top_k"])
    emb_dim = int(cfg["ranker_emb_dim"])
    emb_token_id = int(cfg["emb_token_id"])
    candidate_token_ids = [int(x) for x in cfg["candidate_token_ids"]]
    candidate_labels = list(cfg["candidate_labels"])
    include_ranker_embeddings = bool(cfg.get("include_ranker_embeddings", True))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)

    cpu_init_kwargs = dict(model_init_kwargs or {})
    cpu_init_kwargs["dtype"] = torch.float32
    cpu_init_kwargs["device_map"] = None
    cpu_init_kwargs["attn_implementation"] = "eager"

    base_fp32 = AutoModelForCausalLM.from_pretrained(
        resolved_base, **cpu_init_kwargs
    ).cpu()
    merged_qwen = PeftModel.from_pretrained(
        base_fp32, checkpoint_dir
    ).merge_and_unload()
    merged_qwen.eval()

    projector_cpu = None
    proj_file = os.path.join(checkpoint_dir, "ranker_projector.safetensors")
    if include_ranker_embeddings and os.path.exists(proj_file):
        hidden_size = int(merged_qwen.config.hidden_size)
        projector_cpu = torch.nn.Linear(emb_dim, hidden_size, dtype=torch.float32)
        proj_state = load_file(proj_file)
        projector_cpu.load_state_dict({k: v.float() for k, v in proj_state.items()})
        projector_cpu.eval()

    return _build_triton_artifacts(
        merged_qwen=merged_qwen,
        projector_cpu=projector_cpu,
        tokenizer=tokenizer,
        emb_token_id=emb_token_id,
        candidate_token_ids=candidate_token_ids,
        top_k=top_k,
        ranker_emb_dim=emb_dim,
        candidate_labels=candidate_labels,
        triton_model_dir=triton_model_dir,
        include_ranker_embeddings=include_ranker_embeddings,
        quantize_mode=quantize_mode,
        calibration_prompts=calibration_prompts,
    )


def export_and_quantize_onnx(
    safetensors_dir: str,
    onnx_output_dir: str,
    quantize_mode: str = "int4",  # "int4", "awq_w4a16", "int8", "smoothquant_w8a8", or "none"
    calibration_prompts: Optional[List[str]] = None,
) -> str:
    os.makedirs(onnx_output_dir, exist_ok=True)

    with open(os.path.join(safetensors_dir, "reranker_config.json"), "r") as f:
        cfg = json.load(f)

    top_k = int(cfg["top_k"])
    emb_dim = int(cfg["ranker_emb_dim"])
    emb_token_id = int(cfg["emb_token_id"])
    candidate_token_ids = list(cfg["candidate_token_ids"])

    qwen = AutoModelForCausalLM.from_pretrained(
        safetensors_dir,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).cpu()
    qwen.eval()

    hidden_size = int(qwen.config.hidden_size)
    projector = torch.nn.Linear(emb_dim, hidden_size, dtype=torch.float32)
    proj_state = load_file(
        os.path.join(safetensors_dir, "ranker_projector.safetensors")
    )
    projector.load_state_dict({k: v.float() for k, v in proj_state.items()})
    projector.eval()

    mode = quantize_mode.lower()
    if mode in ("smoothquant", "smoothquant_w8a8", "awq", "awq_w4a16"):
        tokenizer = AutoTokenizer.from_pretrained(
            safetensors_dir, local_files_only=True
        )
        act_max, act_mean = _collect_channel_activation_stats(
            qwen_model=qwen,
            tokenizer=tokenizer,
            calibration_prompts=calibration_prompts,
            emb_token_id=emb_token_id,
            top_k=top_k,
            ranker_emb_dim=emb_dim,
            projector=projector,
        )
        if mode in ("smoothquant", "smoothquant_w8a8"):
            apply_smoothquant_smoothing(qwen, act_max=act_max, alpha=0.5)
        else:
            apply_awq_salient_scaling(qwen, act_mean=act_mean, alpha=0.5)

    wrapper = OnnxExportableQwenReranker(
        qwen_model=qwen,
        projector=projector,
        emb_token_id=emb_token_id,
        candidate_token_ids=candidate_token_ids,
    ).eval()

    batch_size, seq_len = 2, 32
    dummy_input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_input_ids[:, 2 : 2 + 1 + top_k] = emb_token_id
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_ranker_embs = torch.randn(
        (batch_size, 1 + top_k, emb_dim), dtype=torch.float32
    )

    fp32_onnx_path = os.path.join(onnx_output_dir, "model_fp32.onnx")
    _run_torch_onnx_export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_ranker_embs),
        fp32_onnx_path,
    )

    final_onnx_path = (
        fp32_onnx_path
        if mode == "none"
        else os.path.join(onnx_output_dir, "model.onnx")
    )
    return _quantize_onnx_model(
        fp32_onnx_path=fp32_onnx_path,
        final_onnx_path=final_onnx_path,
        quantize_mode=mode,
    )


def export_to_triton_onnx(
    trained_trainer: ListwiseSFTTrainer,
    tokenizer: Any,
    model_init_kwargs: dict,
    emb_token_id: int,
    candidate_token_ids: list[int],
    top_k: int,
    ranker_emb_dim: int,
    candidate_labels: list[str],
    base_model_path: str,
    triton_model_dir: str,
    include_ranker_embeddings: bool = True,
    quantize_mode: str = "none",  # "none", "smoothquant_w8a8", or "awq_w4a16"
    calibration_prompts: Optional[List[str]] = None,
) -> str:
    """
    Exports the trained ListwiseSFTTrainer model directly to a Triton ONNX repository[cite: 6]:
      triton_model_dir/
        ├── config.pbtxt
        ├── tokenizer.json
        ├── reranker_config.json
        └── 1/
            └── model.onnx
    """
    unwrapped = trained_trainer._unwrap_model()
    projector_cpu = None
    if include_ranker_embeddings and hasattr(unwrapped, "ranker_emb_projector"):
        orig_proj = unwrapped.ranker_emb_projector
        projector_cpu = torch.nn.Linear(
            orig_proj.in_features,
            orig_proj.out_features,
            dtype=torch.float32,
        )
        projector_cpu.load_state_dict(
            {k: v.detach().cpu().float() for k, v in orig_proj.state_dict().items()}
        )
        projector_cpu.eval()

    tmp_adapter_dir = os.path.join(triton_model_dir, "_tmp_lora_adapter")
    trained_trainer.save_model(tmp_adapter_dir)

    cpu_init_kwargs = dict(model_init_kwargs)
    cpu_init_kwargs["dtype"] = torch.float32
    cpu_init_kwargs["device_map"] = None
    cpu_init_kwargs["attn_implementation"] = "eager"

    base_fp32 = AutoModelForCausalLM.from_pretrained(
        base_model_path, **cpu_init_kwargs
    ).cpu()
    merged_qwen = PeftModel.from_pretrained(
        base_fp32, tmp_adapter_dir
    ).merge_and_unload()
    merged_qwen.eval()
    shutil.rmtree(tmp_adapter_dir, ignore_errors=True)

    return _build_triton_artifacts(
        merged_qwen=merged_qwen,
        projector_cpu=projector_cpu,
        tokenizer=tokenizer,
        emb_token_id=emb_token_id,
        candidate_token_ids=candidate_token_ids,
        top_k=top_k,
        ranker_emb_dim=ranker_emb_dim,
        candidate_labels=candidate_labels,
        triton_model_dir=triton_model_dir,
        include_ranker_embeddings=include_ranker_embeddings,
        quantize_mode=quantize_mode,
        calibration_prompts=calibration_prompts,
    )

def load_reusable_checkpoint_in_memory(
    checkpoint_dir: str,
    base_model_path: Optional[str] = None,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[OnnxExportableQwenReranker, Any, Dict[str, Any]]:
    """
    Loads a saved Portable Checkpoint (`save_reusable_checkpoint`) directly into memory
    as a merged `OnnxExportableQwenReranker` PyTorch module with the serving signature:
        `candidate_log_probs = model(input_ids, attention_mask, ranker_embs)`
    Returns `(in_memory_model, tokenizer, reranker_config_dict)`.
    """
    with open(
        os.path.join(checkpoint_dir, "reranker_config.json"), "r", encoding="utf-8"
    ) as f:
        cfg = json.load(f)

    resolved_base = base_model_path or cfg["base_model_id"]
    emb_dim = int(cfg["ranker_emb_dim"])
    emb_token_id = int(cfg["emb_token_id"])
    candidate_token_ids = [int(x) for x in cfg["candidate_token_ids"]]
    include_ranker_embeddings = bool(cfg.get("include_ranker_embeddings", True))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)

    init_kwargs = dict(model_init_kwargs or {})
    init_kwargs["dtype"] = torch.float32
    init_kwargs["device_map"] = None
    init_kwargs["attn_implementation"] = "eager"

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base, **init_kwargs
    ).to(device)
    merged_qwen = PeftModel.from_pretrained(
        base_model, checkpoint_dir
    ).merge_and_unload()
    merged_qwen.eval()

    projector = None
    proj_file = os.path.join(checkpoint_dir, "ranker_projector.safetensors")
    if include_ranker_embeddings and os.path.exists(proj_file):
        hidden_size = int(merged_qwen.config.hidden_size)
        projector = torch.nn.Linear(
            emb_dim, hidden_size, dtype=torch.float32
        ).to(device)
        proj_state = load_file(proj_file)
        projector.load_state_dict({k: v.float() for k, v in proj_state.items()})
        projector.eval()

    in_memory_model = (
        OnnxExportableQwenReranker(
            qwen_model=merged_qwen,
            projector=projector,
            emb_token_id=emb_token_id,
            candidate_token_ids=candidate_token_ids,
            include_ranker_embeddings=include_ranker_embeddings,
        )
        .to(device)
        .eval()
    )

    return in_memory_model, tokenizer, cfg