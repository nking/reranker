import json
import os
import shutil
import warnings
from typing import Any

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from movie_lens_reranker.ListwiseSFTTrainer import ListwiseSFTTrainer
from movie_lens_reranker.OnnxExportbleQwenReranker import OnnxExportableQwenReranker


def _run_torch_onnx_export(
    wrapper: torch.nn.Module,
    dummy_args: tuple,
    output_onnx_path: str,
) -> None:
    """Exports `wrapper` to ONNX opset 17 while silencing TorchScript deprecation/tracer noise."""
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


def export_and_quantize_onnx(
    safetensors_dir: str,
    onnx_output_dir: str,
    quantize_mode: str = "int4",  # "int4", "int8", or "none"
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

    if quantize_mode == "int4":
        int4_onnx_path = os.path.join(onnx_output_dir, "model.onnx")
        model_proto = onnx.load(fp32_onnx_path)
        quantizer = MatMulNBitsQuantizer(
            model=model_proto,
            bits=4,
            block_size=128,
            is_symmetric=True,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(
            int4_onnx_path, use_external_data_format=False
        )
        return int4_onnx_path

    if quantize_mode == "int8":
        int8_onnx_path = os.path.join(onnx_output_dir, "model.onnx")
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8,
        )
        return int8_onnx_path

    return fp32_onnx_path


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
) -> str:
    """
    Exports the trained ListwiseSFTTrainer model directly to a Triton ONNX repository:
      triton_model_dir/
        ├── config.pbtxt
        ├── tokenizer.json
        ├── reranker_config.json
        └── 1/
            └── model.onnx
    """
    version_dir = os.path.join(triton_model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    # 1. Save tokenizer for Rust client
    tokenizer.save_pretrained(triton_model_dir)

    # 2. Extract trained projector in FP32 on CPU
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

    # 3. Save adapter weights to temporary dir and reload into clean FP32 base model on CPU
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

    # 4. Wrap in OnnxExportableQwenReranker and export to 1/model.onnx
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

    onnx_file_path = os.path.join(version_dir, "model.onnx")
    _run_torch_onnx_export(
        onnx_wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_ranker_embs),
        onnx_file_path,
    )

    # 5. Write Triton config.pbtxt and reranker_config.json
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
            },
            f,
            indent=2,
        )

    return onnx_file_path