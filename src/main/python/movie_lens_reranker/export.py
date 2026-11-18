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
from onnxruntime.quantization import QuantType, quantize_dynamic
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


def _quantize_onnx_model(
    fp32_onnx_path: str,
    final_onnx_path: str,
    quantize_mode: str,
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

        # Keep projector, candidate_head, and first/last boundary layers in FP32 for maximum precision
        excluded_nodes = []
        for node in model_proto.graph.node:
            if "projector" in node.name or "candidate_head" in node.name:
                excluded_nodes.append(node.name)
            elif mode in ("int8", "smoothquant", "smoothquant_w8a8") and (
                "layers.0/" in node.name
                or (last_layer_tag is not None and last_layer_tag in node.name)
            ):
                excluded_nodes.append(node.name)

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
            del quantizer, model_proto
        else:
            del model_proto
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
) -> str:
    t0 = time.monotonic()
    version_dir = os.path.join(triton_model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    tokenizer.save_pretrained(triton_model_dir)

    mode = quantize_mode.lower()
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