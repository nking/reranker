import gc
import json
import os
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from movie_lens_reranker.ListwiseSFTTrainer import ListwiseSFTTrainer
from movie_lens_reranker.OnnxExportbleQwenReranker import OnnxExportableQwenReranker


def _log_export(msg: str, t0: Optional[float] = None) -> None:
    if t0 is not None:
        print(f"[Export] {msg} ({time.monotonic() - t0:.2f}s)", flush=True)
    else:
        print(f"[Export] {msg}", flush=True)


def _ensure_peft_torchao_compat() -> None:
    """Prevents PEFT >=0.21 from crashing on Kaggle's preinstalled torchao 0.10.0 during CPU LoRA merge."""
    try:
        import peft.import_utils as peft_import_utils
        peft_import_utils.is_torchao_available()
    except ImportError:
        try:
            import peft.import_utils as peft_import_utils
            peft_import_utils.is_torchao_available = lambda: False
        except Exception:
            pass
        try:
            import peft.tuners.lora.torchao as peft_lora_torchao
            peft_lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass


def _compute_safe_smoothquant_scales(
    s_x: torch.Tensor,
    s_w: torch.Tensor,
    alpha: float,
    min_scale: float = 0.35,
    max_scale: float = 2.5,
) -> torch.Tensor:
    """
    Computes bounded per-channel SmoothQuant migration scales.
    Floors sparse/inactive calibration channels at 10% of median activation magnitude
    and clamps scales to `[min_scale, max_scale]` so RMSNorm and Linear weights are
    not over-amplified across 24 layers.
    """
    med_x = float(s_x.median().item())
    x_floor = max(1e-2, 0.1 * med_x)
    s_x_safe = s_x.clamp(min=x_floor)
    s_w_safe = s_w.clamp(min=1e-4)

    scales = s_x_safe.pow(alpha) / s_w_safe.pow(1.0 - alpha)
    geo_mean = scales.log().mean().exp()
    return (scales / geo_mean).clamp(min=min_scale, max=max_scale)


@torch.no_grad()
def apply_smoothquant_smoothing(
    qwen_model: nn.Module,
    act_max: Dict[str, torch.Tensor],
    alpha: float = 0.35,
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
            )
            w_cat = torch.cat(
                [
                    layer.self_attn.q_proj.weight.detach().float(),
                    layer.self_attn.k_proj.weight.detach().float(),
                    layer.self_attn.v_proj.weight.detach().float(),
                ],
                dim=0,
            )
            s_w = w_cat.abs().amax(dim=0)
            scales = _compute_safe_smoothquant_scales(s_x, s_w, alpha=alpha)

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
            )
            w_cat = torch.cat(
                [
                    layer.mlp.gate_proj.weight.detach().float(),
                    layer.mlp.up_proj.weight.detach().float(),
                ],
                dim=0,
            )
            s_w = w_cat.abs().amax(dim=0)
            scales = _compute_safe_smoothquant_scales(s_x, s_w, alpha=alpha)

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
            )
            s_w = layer.mlp.down_proj.weight.detach().float().abs().amax(dim=0)
            scales = _compute_safe_smoothquant_scales(s_x, s_w, alpha=alpha)

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


@torch.no_grad()
def _collect_channel_activation_stats(
    qwen_model: nn.Module,
    tokenizer: Any,
    calibration_prompts: Optional[List[str]],
    emb_token_id: int,
    top_k: int,
    ranker_emb_dim: int,
    projector: Optional[nn.Module] = None,
    calibration_embs: Optional[List[Any]] = None,
    max_calib_samples: int = 64,
    max_seq_len: int = 768,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Runs a lightweight calibration forward pass to collect per-channel max(|X|) for SmoothQuant.
    """
    t0 = time.monotonic()
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
    has_real_embs = (
        prompts is not None
        and calibration_embs is not None
        and len(calibration_embs) >= len(prompts)
    )
    n_calib = len(prompts) if prompts is not None else 4
    _log_export(
        f"  Collecting activation stats on {device} ({n_calib} samples, real_embs={has_real_embs})..."
    )

    try:
        with torch.no_grad():
            if prompts is not None:
                orig_pad = tokenizer.padding_side
                tokenizer.padding_side = "left"
                try:
                    batch_size = 8
                    for start in range(0, len(prompts), batch_size):
                        t_batch = time.monotonic()
                        end = min(start + batch_size, len(prompts))
                        batch_prompts = prompts[start:end]
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
                            if has_real_embs and calibration_embs is not None:
                                raw_embs = np.asarray(
                                    calibration_embs[start:end], dtype=np.float32
                                )
                                batch_embs = torch.from_numpy(raw_embs).to(device)
                            else:
                                batch_embs = torch.zeros(
                                    (input_ids.size(0), 1 + top_k, ranker_emb_dim),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            proj = projector.to(device)(batch_embs).to(dtype=dtype)
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
                        _log_export(
                            f"    Calib batch {end}/{len(prompts)} "
                            f"(seq_len={input_ids.size(1)})",
                            t0=t_batch,
                        )
                finally:
                    tokenizer.padding_side = orig_pad
            else:
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
    _log_export("  Completed activation stats collection", t0=t0)
    return act_max, act_mean


class RerankerCalibrationDataReader(CalibrationDataReader):
    """Feeds calibration prompts and ranker embeddings into onnxruntime.quantization.quantize_static."""

    def __init__(
        self,
        tokenizer: Any,
        prompts: Optional[List[str]],
        embs: Optional[List[Any]],
        emb_token_id: int,
        top_k: int,
        ranker_emb_dim: int,
        max_seq_len: int = 768,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts if prompts and len(prompts) > 0 else ["Default calibration prompt for reranker."]
        self.embs = embs
        self.emb_token_id = int(emb_token_id)
        self.top_k = int(top_k)
        self.ranker_emb_dim = int(ranker_emb_dim)
        self.max_seq_len = int(max_seq_len)
        self.enum_data = iter(range(len(self.prompts)))

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            idx = next(self.enum_data)
        except StopIteration:
            return None

        prompt = self.prompts[idx]
        emb = (
            self.embs[idx]
            if self.embs is not None and idx < len(self.embs)
            else np.zeros((1 + self.top_k, self.ranker_emb_dim), dtype=np.float32)
        )

        orig_pad = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                [prompt],
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            input_ids = enc["input_ids"].astype(np.int64)
            attention_mask = enc["attention_mask"].astype(np.int64)
            ranker_embs = np.expand_dims(np.asarray(emb, dtype=np.float32), axis=0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "ranker_embs": ranker_embs,
            }
        finally:
            self.tokenizer.padding_side = orig_pad


def _quantize_onnx_model(
    fp32_onnx_path: str,
    final_onnx_path: str,
    quantize_mode: str,
    tokenizer: Optional[Any] = None,
    calibration_prompts: Optional[List[str]] = None,
    calibration_embs: Optional[List[Any]] = None,
    emb_token_id: int = 0,
    top_k: int = 20,
    ranker_emb_dim: int = 24,
) -> str:
    """Applies W4A16 (`int4` / `awq_w4a16`) or W8A8 (`int8` / `smoothquant_w8a8`) quantization."""
    t0 = time.monotonic()
    mode = quantize_mode.lower()
    if mode in (
        "int4",
        "awq",
        "awq_w4a16",
        "int8",
        "smoothquant",
        "smoothquant_w8a8",
    ):
        _log_export(f"  Quantizing ONNX graph ({mode}) -> {final_onnx_path}...")
        model_proto = onnx.load(fp32_onnx_path)
        num_layers = max(
            (
                int(part.split(".")[1]) + 1
                for node in model_proto.graph.node
                for part in node.name.split("/")
                if part.startswith("layers.") and part.split(".")[1].isdigit()
            ),
            default=0,
        )
        last_layer_tag = f"layers.{num_layers - 1}/" if num_layers > 2 else None

        # Keep projector, candidate_head, and first/last boundary layers in FP32
        excluded_nodes = []
        for node in model_proto.graph.node:
            if "projector" in node.name or "candidate_head" in node.name:
                excluded_nodes.append(node.name)
            elif mode in ("int8", "smoothquant", "smoothquant_w8a8") and (
                "layers.0/" in node.name
                or (last_layer_tag is not None and last_layer_tag in node.name)
            ):
                excluded_nodes.append(node.name)
        del model_proto

        if mode in ("int4", "awq", "awq_w4a16"):
            model_proto_awq = onnx.load(fp32_onnx_path)
            quantizer = MatMulNBitsQuantizer(
                model=model_proto_awq,
                bits=4,
                block_size=128,
                is_symmetric=True,
                nodes_to_exclude=excluded_nodes,
            )
            quantizer.process()
            quantizer.model.save_model_to_file(
                final_onnx_path, use_external_data_format=False
            )
            del quantizer, model_proto_awq
        else:
            preprocessed_onnx_path = fp32_onnx_path.replace(
                ".onnx", "_preprocessed.onnx"
            )
            quant_input_path = fp32_onnx_path
            try:
                quant_pre_process(
                    input_model_path=fp32_onnx_path,
                    output_model_path=preprocessed_onnx_path,
                    skip_optimization=False,
                    skip_onnx_shape=False,
                    skip_symbolic_shape=True,
                )
                if os.path.exists(preprocessed_onnx_path):
                    quant_input_path = preprocessed_onnx_path
            except Exception as prep_exc:
                _log_export(
                    f"  [WARNING] quant_pre_process skipped due to: {prep_exc}"
                )

            try:
                reader = RerankerCalibrationDataReader(
                    tokenizer=tokenizer,
                    prompts=calibration_prompts,
                    embs=calibration_embs,
                    emb_token_id=emb_token_id,
                    top_k=top_k,
                    ranker_emb_dim=ranker_emb_dim,
                )
                quantize_static(
                    model_input=quant_input_path,
                    model_output=final_onnx_path,
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8,
                    per_channel=True,
                    reduce_range=True,
                    op_types_to_quantize=["MatMul", "Gemm"],
                    nodes_to_exclude=excluded_nodes,
                    calibrate_method=CalibrationMethod.Percentile,
                )
            finally:
                if os.path.exists(preprocessed_onnx_path):
                    os.remove(preprocessed_onnx_path)

        if fp32_onnx_path != final_onnx_path and os.path.exists(fp32_onnx_path):
            os.remove(fp32_onnx_path)

        gc.collect()
        size_mb = os.path.getsize(final_onnx_path) / (1024 * 1024)
        _log_export(
            f"  Finished quantization ({mode}, {size_mb:.1f} MB)", t0=t0
        )
        return final_onnx_path

    if fp32_onnx_path != final_onnx_path:
        shutil.move(fp32_onnx_path, final_onnx_path)
    size_mb = os.path.getsize(final_onnx_path) / (1024 * 1024)
    _log_export(f"  Saved unquantized FP32 ONNX ({size_mb:.1f} MB)", t0=t0)
    return final_onnx_path


def _run_torch_onnx_export(
    wrapper: torch.nn.Module,
    dummy_args: tuple,
    output_onnx_path: str,
) -> None:
    """Exports `wrapper` to ONNX opset 17 while silencing TorchScript deprecation/tracer noise."""
    t0 = time.monotonic()
    _log_export(
        f"  Running torch.onnx.export (opset=17, dynamo=False) -> {output_onnx_path}..."
    )
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
    size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    _log_export(f"  Finished torch.onnx.export ({size_mb:.1f} MB)", t0=t0)


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
    Saves a lightweight (~11.5 MB) portable checkpoint containing trained LoRA adapters,
    `ranker_projector.safetensors`, tokenizer, and `reranker_config.json`.
    """
    t0 = time.monotonic()
    _log_export(f"Saving portable checkpoint -> {checkpoint_dir}...")
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

    _log_export(f"Saved portable checkpoint to {checkpoint_dir}", t0=t0)
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
    calibration_embs: Optional[List[Any]] = None,
    smoothquant_alpha: float = 0.35,
) -> str:
    t0 = time.monotonic()
    version_dir = os.path.join(triton_model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    tokenizer.save_pretrained(triton_model_dir)

    mode = quantize_mode.lower()
    if mode in ("smoothquant", "smoothquant_w8a8"):
        act_max, act_mean = _collect_channel_activation_stats(
            qwen_model=merged_qwen,
            tokenizer=tokenizer,
            calibration_prompts=calibration_prompts,
            emb_token_id=emb_token_id,
            top_k=top_k,
            ranker_emb_dim=ranker_emb_dim,
            projector=projector_cpu if include_ranker_embeddings else None,
            calibration_embs=calibration_embs,
        )
        t_scale = time.monotonic()
        apply_smoothquant_smoothing(merged_qwen, act_max=act_max, alpha=smoothquant_alpha)
        _log_export(f"  Applied {mode} channel smoothing", t0=t_scale)

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
    del onnx_wrapper
    gc.collect()

    _quantize_onnx_model(
        fp32_onnx_path=tmp_fp32_onnx,
        final_onnx_path=final_onnx_path,
        quantize_mode=mode,
        tokenizer=tokenizer,
        calibration_prompts=calibration_prompts,
        calibration_embs=calibration_embs,
        emb_token_id=emb_token_id,
        top_k=top_k,
        ranker_emb_dim=ranker_emb_dim,
    )

    triton_kind = (
        "KIND_CPU"
        if mode in ("int8", "smoothquant", "smoothquant_w8a8")
        else "KIND_GPU"
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
        f"instance_group [{{ count: 1, kind: {triton_kind} }}]\n"
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
                "include_ranker_embeddings": bool(include_ranker_embeddings),
                "quantize_mode": mode,
            },
            f,
            indent=2,
        )

    _log_export(
        f"Completed Triton bundle ({mode}) -> {triton_model_dir}", t0=t0
    )
    return final_onnx_path


def export_from_reusable_checkpoint(
    checkpoint_dir: str,
    triton_model_dir: str,
    base_model_path: Optional[str] = None,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    quantize_mode: str = "awq_w4a16",
    calibration_prompts: Optional[List[str]] = None,
    calibration_embs: Optional[List[Any]] = None,
    smoothquant_alpha: float = 0.35,
) -> str:
    """
    Reloads a saved portable checkpoint, merges LoRA into base model on CPU,
    and exports a Triton ONNX repository.
    """
    t_total = time.monotonic()
    _log_export(
        f"[{quantize_mode}] Starting export_from_reusable_checkpoint -> {triton_model_dir}"
    )

    t_step = time.monotonic()
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
    _log_export(f"[{quantize_mode}] 1/4 Loaded config & tokenizer", t0=t_step)

    t_step = time.monotonic()
    cpu_init_kwargs = dict(model_init_kwargs or {})
    cpu_init_kwargs["dtype"] = torch.float32
    cpu_init_kwargs["device_map"] = None
    cpu_init_kwargs["attn_implementation"] = "eager"
    if os.path.exists(str(resolved_base)):
        cpu_init_kwargs["local_files_only"] = True

    base_fp32 = AutoModelForCausalLM.from_pretrained(
        resolved_base, **cpu_init_kwargs
    ).cpu()
    _log_export(f"[{quantize_mode}] 2/4 Loaded FP32 base model on CPU", t0=t_step)

    t_step = time.monotonic()
    _ensure_peft_torchao_compat()
    merged_qwen = PeftModel.from_pretrained(
        base_fp32, checkpoint_dir
    ).merge_and_unload()
    del base_fp32
    merged_qwen.eval()
    _log_export(f"[{quantize_mode}] 3/4 Merged LoRA adapters", t0=t_step)

    projector_cpu = None
    proj_file = os.path.join(checkpoint_dir, "ranker_projector.safetensors")
    if include_ranker_embeddings and os.path.exists(proj_file):
        hidden_size = int(merged_qwen.config.hidden_size)
        projector_cpu = torch.nn.Linear(emb_dim, hidden_size, dtype=torch.float32)
        proj_state = load_file(proj_file)
        projector_cpu.load_state_dict({k: v.float() for k, v in proj_state.items()})
        projector_cpu.eval()

    _log_export(f"[{quantize_mode}] 4/4 Building Triton ONNX artifacts...")
    try:
        out_path = _build_triton_artifacts(
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
            calibration_embs=calibration_embs,
            smoothquant_alpha=smoothquant_alpha,
        )
    finally:
        del merged_qwen, projector_cpu
        gc.collect()

    _log_export(
        f"[{quantize_mode}] Finished full export_from_reusable_checkpoint",
        t0=t_total,
    )
    return out_path


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
    quantize_mode: str = "none",
    calibration_prompts: Optional[List[str]] = None,
    calibration_embs: Optional[List[Any]] = None,
    smoothquant_alpha: float = 0.35,
) -> str:
    """Exports trained model directly to a Triton ONNX repository."""
    t0 = time.monotonic()
    _log_export(f"[{quantize_mode}] Starting export_to_triton_onnx -> {triton_model_dir}")
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
    if os.path.exists(str(base_model_path)):
        cpu_init_kwargs["local_files_only"] = True

    base_fp32 = AutoModelForCausalLM.from_pretrained(
        base_model_path, **cpu_init_kwargs
    ).cpu()
    _ensure_peft_torchao_compat()
    merged_qwen = PeftModel.from_pretrained(
        base_fp32, tmp_adapter_dir
    ).merge_and_unload()
    del base_fp32
    merged_qwen.eval()
    shutil.rmtree(tmp_adapter_dir, ignore_errors=True)

    try:
        out_path = _build_triton_artifacts(
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
            calibration_embs=calibration_embs,
            smoothquant_alpha=smoothquant_alpha,
        )
    finally:
        del merged_qwen, projector_cpu
        gc.collect()

    _log_export(f"[{quantize_mode}] Finished export_to_triton_onnx", t0=t0)
    return out_path


def load_reusable_checkpoint_in_memory(
    checkpoint_dir: str,
    base_model_path: Optional[str] = None,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[OnnxExportableQwenReranker, Any, Dict[str, Any]]:
    """Loads a saved portable checkpoint directly into memory as a PyTorch module."""
    t0 = time.monotonic()
    _log_export(
        f"Loading portable checkpoint into memory on {device} from {checkpoint_dir}..."
    )
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
    if os.path.exists(str(resolved_base)):
        init_kwargs["local_files_only"] = True

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base, **init_kwargs
    ).to(device)
    _ensure_peft_torchao_compat()
    merged_qwen = PeftModel.from_pretrained(
        base_model, checkpoint_dir
    ).merge_and_unload()
    del base_model
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

    _log_export("Loaded portable checkpoint into memory", t0=t0)
    return in_memory_model, tokenizer, cfg