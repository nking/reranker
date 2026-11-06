import copy
import gc
import math
import os
import re
import shutil
import sys
import unittest
from typing import Any, Dict, List

HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not HAS_NVIDIA_GPU:
    # Local CPU: Force single-threaded OpenMP to prevent C++ autograd barrier hangs
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
else:
    # Kaggle T4x2 (4 vCPUs / 2 GPU processes = 2 threads per process)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import optuna
import torch
import yaml
from accelerate import PartialState
from datasets import Dataset
from huggingface_hub import snapshot_download
from jinja2 import Template
from peft import LoraConfig, PeftModel
from ranx import Qrels, Run, evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

if not torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

from helper import *

try:
    from prompt_helper import Prompt_Type, get_yaml_prompt_path
except ImportError:
    Prompt_Type = None
    get_yaml_prompt_path = None

TOP_K = 5
MAX_HISTORY_ITEMS = 2
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
USE_CUDA = torch.cuda.is_available()
CANDIDATE_LABELS = [chr(ord("A") + i) for i in range(TOP_K)]

# --- GATv2 Ranker Embedding Injection Settings ---
INCLUDE_RANKER_EMBEDDINGS = True
RANKER_EMB_DIM = 24
EMB_PLACEHOLDER_TOKEN = "<|vision_pad|>"

# --- Default YAML Configurations (Fallback if prompts/query-v2.0.0.yaml is not on disk) ---
DEFAULT_QUERY_YAML = """
name: "qwen listwise reranker query prompt"
version: "2.0.0"
author: "Nichole"
created_at: "2026-11-05"

model_config:
  model: "Qwen/Qwen2.5-0.5B-Instruct"
  top_k: 5
  max_history_items: 10
  safety_margin_tokens: 16
  worst_case_title: "Everything You Always Wanted to Know About Sex * But Were Afraid to Ask (1972)"
  worst_case_genres: "Action, Adventure, Animation, Children's, Comedy, Sci-Fi"
  worst_case_recency: "12mo ago"

system_template: |
  Rank candidate movies [{{ first_label }}]-[{{ last_label }}] by relevance.

template: |
  <|im_start|>system
  {{ system_prompt }}<|im_end|>
  <|im_start|>user
  Query Time: {{ query_time }}
  {% if user_emb_tag %}{{ user_emb_tag }}
  {% endif %}User Watch History:
  {{ history_block }}
  Candidates:
  {{ candidates_block }}
  Ranked IDs:<|im_end|>
  <|im_start|>assistant

completion_template: "{{ ranked_labels }}<|im_end|>"
eval_dataset_id: ""
"""

DEFAULT_PASSAGES_YAML = """
name: "qwen listwise reranker passages prompt"
version: "2.0.0"
author: "Nichole"
created_at: "2026-11-05"

model_config:
  model: "Qwen/Qwen2.5-0.5B-Instruct"

history_template: "- {{ title }} | {{ genres }} | {{ rating }}/5 | {{ recency }}"
candidate_template: "[{{ label }}] {% if cand_emb_tag %}{{ cand_emb_tag }} {% endif %}{{ title }} | {{ genres }}"

template: |
  [{{ label }}] {% if cand_emb_tag %}{{ cand_emb_tag }} {% endif %}{{ title }} | {{ genres }}

eval_dataset_id: ""
"""


def format_movielens_title(raw_title: str) -> str:
    """Converts 'Matrix, The (1999)' -> 'The Matrix (1999)'."""
    return re.sub(
        r"^(.*),\s*(The|A|An)\s*(\(\d{4}\))$",
        r"\2 \1 \3",
        raw_title.strip(),
    )


def format_movielens_genres(raw_genres: str) -> str:
    """Converts 'Action|Sci-Fi|Thriller' -> 'Action, Sci-Fi, Thriller'."""
    return raw_genres.replace("|", ", ")


class StepHeartbeatCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        print(
            f"\n[Heartbeat] Starting optimizer step {state.global_step + 1}/{state.max_steps}...",
            file=sys.stderr,
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        print(
            f"[Heartbeat] Completed optimizer step {state.global_step}/{state.max_steps}!",
            file=sys.stderr,
            flush=True,
        )


class ListwiseSFTTrainer(SFTTrainer):
    """
    Subclasses SFTTrainer to:
    1. Inject 24-dim GATv2 user & candidate embeddings at `<|vision_pad|>` token positions
       via a trainable linear projection (`nn.Linear(24, hidden_size)`).
    2. Run a single forward pass during evaluation without TRL's deprecated `return_outputs=True`.
    """

    def __init__(
        self,
        *args,
        include_ranker_embeddings: bool = False,
        ranker_emb_dim: int = RANKER_EMB_DIM,
        emb_token_id: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_ranker_embeddings = include_ranker_embeddings
        self.emb_token_id = emb_token_id
        self._current_input_ids = None
        self._current_ranker_embs = None

        if self.include_ranker_embeddings:
            if self.emb_token_id is None:
                raise ValueError(
                    "emb_token_id must be provided when include_ranker_embeddings=True"
                )

            # 1. Attach trainable 24 -> hidden_size (896) projector onto the model
            hidden_size = self.model.config.hidden_size
            device = next(self.model.parameters()).device
            projector = torch.nn.Linear(
                ranker_emb_dim,
                hidden_size,
                device=device,
                dtype=torch.float32,
            )
            projector.requires_grad_(True)
            self.model.ranker_emb_projector = projector

            # 2. Wrap data_collator so "ranker_embs" is collated into batches without mutating dataset rows
            orig_collator = self.data_collator

            def collate_with_ranker_embs(features):
                copied_features = [dict(f) for f in features]
                embs = [f.pop("ranker_embs", None) for f in copied_features]
                batch = orig_collator(copied_features)
                if embs[0] is not None:
                    batch["ranker_embs"] = torch.tensor(embs, dtype=torch.float32)
                return batch

            self.data_collator = collate_with_ranker_embs

            # 3. Register forward hook on token embeddings to swap in projected GATv2 vectors
            embed_layer = self.model.get_input_embeddings()
            embed_layer.register_forward_hook(self._inject_ranker_embeddings_hook)

    def _inject_ranker_embeddings_hook(self, module, module_inputs, output):
        if not self.include_ranker_embeddings or self._current_ranker_embs is None:
            return output

        input_ids = module_inputs[0]
        ranker_embs = self._current_ranker_embs.to(
            device=output.device,
            dtype=self.model.ranker_emb_projector.weight.dtype,
        )
        projected = self.model.ranker_emb_projector(ranker_embs).to(dtype=output.dtype)

        inputs_embeds = output.clone()
        for b_idx in range(input_ids.size(0)):
            pos = (input_ids[b_idx] == self.emb_token_id).nonzero(as_tuple=True)[0]
            n_slots = min(len(pos), projected.size(1))
            if n_slots > 0:
                inputs_embeds[b_idx, pos[:n_slots]] = projected[b_idx, :n_slots]

        return inputs_embeds

    def _prepare_inputs(self, inputs):
        ranker_embs = inputs.pop("ranker_embs", None)
        inputs = super()._prepare_inputs(inputs)
        self._current_input_ids = inputs.get("input_ids")
        if ranker_embs is not None:
            self._current_ranker_embs = ranker_embs
        return inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
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


class TestSupervisedFineTuning(unittest.TestCase):

    def setUp(self):
        self.token = get_hf_token()

        local_cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs"
        )
        local_cache_exists = (
            os.path.exists(local_cache_path) and len(os.listdir(local_cache_path)) > 0
        )

        if local_cache_exists:
            os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_path = (
            snapshot_download(repo_id=MODEL_ID, local_files_only=True)
            if local_cache_exists
            else MODEL_ID
        )

        self.output_dir = os.path.join(get_bin_dir(), "prototype_qwen_reranker")
        self.use_qlora_4bit = USE_CUDA

        # 1. Load YAML Prompt Templates
        self.query_yaml, self.passages_yaml = self.load_prompt_yamls(version="2.0.0")

        # 2. Tokenizer & Candidate Token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            token=self.token,
            local_files_only=local_cache_exists,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.candidate_token_ids = [
            self.tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in CANDIDATE_LABELS
        ]

        # Resolve single token ID for `<|vision_pad|>` placeholder
        emb_token_ids = self.tokenizer.encode(
            EMB_PLACEHOLDER_TOKEN, add_special_tokens=False
        )
        self.assertEqual(
            len(emb_token_ids),
            1,
            f"Expected {EMB_PLACEHOLDER_TOKEN} to map to a single token ID, got {emb_token_ids}",
        )
        self.emb_token_id = emb_token_ids[0]

        # 3. Quantization & Base LoRA Target Modules
        self.bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            if self.use_qlora_4bit
            else None
        )

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        # 4. Model Init Kwargs
        self.model_init_kwargs = self.build_model_init_kwargs(
            token=self.token,
            local_cache_exists=local_cache_exists,
            cpu_smoke_test=True,
        )

    def tearDown(self):
        if PartialState().is_main_process and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def load_prompt_yamls(self, version: str = "2.0.0"):
        query_cfg = None
        passages_cfg = None

        if get_yaml_prompt_path is not None and Prompt_Type is not None:
            try:
                q_path = get_yaml_prompt_path(Prompt_Type.QUERY, version=version)
                with open(q_path, "r", encoding="utf-8") as f:
                    query_cfg = yaml.safe_load(f)
            except Exception:
                query_cfg = None

            try:
                p_path = get_yaml_prompt_path(Prompt_Type.PASSAGES, version=version)
                with open(p_path, "r", encoding="utf-8") as f:
                    passages_cfg = yaml.safe_load(f)
            except Exception:
                passages_cfg = None

        if not query_cfg or "system_template" not in query_cfg:
            query_cfg = yaml.safe_load(DEFAULT_QUERY_YAML)
        if not passages_cfg or "candidate_template" not in passages_cfg:
            passages_cfg = yaml.safe_load(DEFAULT_PASSAGES_YAML)

        return query_cfg, passages_cfg

    def render_prompt_and_completion(
        self,
        query_time: str,
        history_items: List[Dict[str, Any]],
        candidate_items: List[Dict[str, Any]],
        ranked_labels_List: List[str],
        include_ranker_embeddings: bool = INCLUDE_RANKER_EMBEDDINGS,
    ):
        candidate_labels = [c["label"] for c in candidate_items]

        sys_tmpl = Template(self.query_yaml["system_template"].strip())
        system_prompt = sys_tmpl.render(
            first_label=candidate_labels[0],
            last_label=candidate_labels[-1],
        )

        hist_tmpl = Template(self.passages_yaml["history_template"].strip())
        history_lines = [hist_tmpl.render(**item) for item in history_items]
        history_block = "\n".join(history_lines)

        cand_tmpl = Template(self.passages_yaml["candidate_template"].strip())
        cand_emb_tag = EMB_PLACEHOLDER_TOKEN if include_ranker_embeddings else ""
        candidate_lines = [
            cand_tmpl.render(cand_emb_tag=cand_emb_tag, **item)
            for item in candidate_items
        ]
        candidates_block = "\n".join(candidate_lines)

        user_emb_tag = (
            f"User Graph Embedding: {EMB_PLACEHOLDER_TOKEN}"
            if include_ranker_embeddings
            else ""
        )

        query_tmpl = Template(self.query_yaml["template"].lstrip())
        prompt = query_tmpl.render(
            system_prompt=system_prompt,
            query_time=query_time,
            user_emb_tag=user_emb_tag,
            history_block=history_block,
            candidates_block=candidates_block,
        )
        if not prompt.endswith("\n"):
            prompt += "\n"

        comp_tmpl = Template(self.query_yaml["completion_template"].strip())
        completion = comp_tmpl.render(ranked_labels=" ".join(ranked_labels_List))

        return prompt, completion

    def calculate_max_length(
        self,
        top_k: int = TOP_K,
        max_history_items: int = MAX_HISTORY_ITEMS,
        include_ranker_embeddings: bool = INCLUDE_RANKER_EMBEDDINGS,
    ) -> int:
        model_cfg = self.query_yaml.get("model_config", {})
        worst_title = model_cfg.get(
            "worst_case_title",
            "Everything You Always Wanted to Know About Sex * But Were Afraid to Ask (1972)",
        )
        worst_genres = model_cfg.get(
            "worst_case_genres",
            "Action, Adventure, Animation, Children's, Comedy, Sci-Fi",
        )
        worst_recency = model_cfg.get("worst_case_recency", "12mo ago")
        safety_margin = int(model_cfg.get("safety_margin_tokens", 16))

        labels = [chr(ord("A") + i) for i in range(top_k)]

        worst_history = [
            {
                "title": worst_title,
                "genres": worst_genres,
                "rating": 5,
                "recency": worst_recency,
            }
            for _ in range(max_history_items)
        ]
        worst_candidates = [
            {
                "label": lbl,
                "title": worst_title,
                "genres": worst_genres,
            }
            for lbl in labels
        ]

        worst_prompt, worst_completion = self.render_prompt_and_completion(
            query_time="Winter, Wed Afternoon (Dec)",
            history_items=worst_history,
            candidate_items=worst_candidates,
            ranked_labels_List=labels,
            include_ranker_embeddings=include_ranker_embeddings,
        )

        full_text = worst_prompt + worst_completion
        token_count = len(self.tokenizer.encode(full_text, add_special_tokens=True))
        total_with_margin = token_count + safety_margin
        aligned_max_length = int(math.ceil(total_with_margin / 32.0) * 32)
        return aligned_max_length

    def build_model_init_kwargs(
        self,
        token: str,
        local_cache_exists: bool,
        cpu_smoke_test: bool = False,
    ) -> dict:
        use_cuda = torch.cuda.is_available()
        device_index = PartialState().process_index

        kwargs = {
            "dtype": torch.float16 if use_cuda else torch.float32,
            "device_map": {"": device_index} if use_cuda else None,
            "attn_implementation": "sdpa" if use_cuda else "eager",
            "token": token,
            "local_files_only": local_cache_exists,
        }
        if not use_cuda and cpu_smoke_test:
            kwargs["num_hidden_layers"] = 2

        return kwargs

    def build_sft_config(
        self,
        output_dir: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        step_freq: int = 5 if USE_CUDA else 2,
        include_ranker_embeddings: bool = INCLUDE_RANKER_EMBEDDINGS,
    ) -> SFTConfig:
        train_batch_size = 4 if USE_CUDA else 1
        eval_batch_size = 4 if USE_CUDA else 2
        grad_accum = 2 if USE_CUDA else 1
        epochs = 2 if USE_CUDA else 1

        computed_max_length = self.calculate_max_length(
            top_k=TOP_K,
            max_history_items=MAX_HISTORY_ITEMS,
            include_ranker_embeddings=include_ranker_embeddings,
        )

        return SFTConfig(
            output_dir=output_dir,
            model_init_kwargs=self.model_init_kwargs,
            gradient_checkpointing=False,
            max_length=computed_max_length,
            packing=False,
            completion_only_loss=True,
            eos_token="<|im_end|>",
            remove_unused_columns=not include_ranker_embeddings,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            dataloader_pin_memory=USE_CUDA,
            dataloader_num_workers=0,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            fp16=USE_CUDA,
            use_cpu=not USE_CUDA,
            logging_first_step=True,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=step_freq,
            save_strategy="steps",
            save_steps=step_freq,
            save_total_limit=2,
            load_best_model_at_end=True,  # Automatically loads best checkpoint based on metric
            metric_for_best_model=f"eval_ndcg@{TOP_K}",
            greater_is_better=True,
            ddp_find_unused_parameters=False if USE_CUDA else None,
            report_to="none",
        )

    def print_model_characteristics(self, model, label: str = "PEFT/QLoRA Model"):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        param_size_bytes = 0
        for param in model.parameters():
            num_el = param.numel()
            if (
                param.__class__.__name__ == "Params4bit"
                or getattr(param, "quant_state", None) is not None
            ):
                param_size_bytes += num_el * 0.5
            elif param.dtype in [torch.float16, torch.bfloat16]:
                param_size_bytes += num_el * 2
            elif param.dtype == torch.float32:
                param_size_bytes += num_el * 4
            else:
                try:
                    param_size_bytes += param.nelement() * param.element_size()
                except Exception:
                    param_size_bytes += num_el * 2

        memory_mb = param_size_bytes / (1024 * 1024)

        print("\n" + "=" * 50, file=sys.stderr)
        print(f"MODEL CHARACTERISTICS [{label}]:", file=sys.stderr)
        print(f"  - Total Parameters:     {total_params:,}", file=sys.stderr)
        print(f"  - Trainable Parameters: {trainable_params:,}", file=sys.stderr)
        print(
            f"  - Trainable Ratio:      {(trainable_params / total_params) * 100:.2f}%",
            file=sys.stderr,
        )
        print(f"  - Est. Memory Footprint: {memory_mb:.2f} MB", file=sys.stderr)
        print("=" * 50 + "\n", file=sys.stderr)
    
    def assertLoadedModelIsEqual(
        self,
        trained_trainer,
        save_path: str,
        test_dataset: Dataset,
        eval_qrels_dict: Dict[str, Dict[str, float]],
        include_ranker_embeddings: bool,
    ):
        """Asserts that reloading the saved model produces identical evaluation metrics."""
        os.makedirs(save_path, exist_ok=True)

        # 1. Save model, tokenizer, and custom ranker projector weights
        trained_trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        trained_model = trained_trainer.model
        if hasattr(trained_model, "ranker_emb_projector"):
            torch.save(
                trained_model.ranker_emb_projector.state_dict(),
                os.path.join(save_path, "ranker_projector.pt"),
            )

        # 2. Instantiate an evaluation-only trainer using base model path
        sft_config_reload = self.build_sft_config(
            output_dir=os.path.join(save_path, "eval_temp"),
            include_ranker_embeddings=include_ranker_embeddings,
        )
        sft_config_reload.do_train = False
        
        reloaded_trainer = ListwiseSFTTrainer(
            model=self.model_path,
            args=sft_config_reload,
            train_dataset=test_dataset,
            eval_dataset=test_dataset,
            processing_class=self.tokenizer,
            quantization_config=self.bnb_config,
            peft_config=None,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.make_compute_metrics(eval_qrels_dict),
            include_ranker_embeddings=include_ranker_embeddings,
            ranker_emb_dim=RANKER_EMB_DIM,
            emb_token_id=self.emb_token_id,
        )
        
        # Load the saved PEFT adapter weights directly onto the trainer's model
        device = next(reloaded_trainer.model.parameters()).device
        reloaded_trainer.model = PeftModel.from_pretrained(
            reloaded_trainer.model,
            save_path,
        )
        
        # Load projector weights into reloaded trainer model if present
        proj_weights_path = os.path.join(save_path, "ranker_projector.pt")
        if os.path.exists(proj_weights_path):
            hidden_size = reloaded_trainer.model.config.hidden_size
            reloaded_trainer.model.ranker_emb_projector = torch.nn.Linear(
                RANKER_EMB_DIM, hidden_size, device=device,
                dtype=torch.float32
            )
            reloaded_trainer.model.ranker_emb_projector.load_state_dict(
                torch.load(proj_weights_path, map_location=device)
            )

        # 3. Evaluate both trainers and assert metrics match
        trained_metrics = trained_trainer.evaluate()
        reloaded_metrics = reloaded_trainer.evaluate()

        print(f"\n[Verification] Trained NDCG@5: {trained_metrics.get(f'eval_ndcg@{TOP_K}')}", file=sys.stderr)
        print(f"[Verification] Reloaded NDCG@5: {reloaded_metrics.get(f'eval_ndcg@{TOP_K}')}", file=sys.stderr)

        self.assertAlmostEqual(
            trained_metrics[f"eval_ndcg@{TOP_K}"],
            reloaded_metrics[f"eval_ndcg@{TOP_K}"],
            places=4,
            msg="Saved and reloaded model evaluation metrics do not match!",
        )
        print("\n[Verification] Saved model reloaded and evaluated successfully with identical metrics!", file=sys.stderr)
        
    def testTorchRun(self):
        n_train = 40 if USE_CUDA else 4
        n_eval = 16 if USE_CUDA else 2

        train_dataset, _ = self.build_fake_dataset(
            num_samples=n_train,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )
        eval_dataset, eval_qrels_dict = self.build_fake_dataset(
            num_samples=n_eval,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )

        sft_config = self.build_sft_config(
            output_dir=self.output_dir,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )

        print(
            "\n[Info] Loading temporary base model to inspect original footprint...",
            file=sys.stderr,
        )
        base_model_temp = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **self.model_init_kwargs,
        )
        self.print_model_characteristics(
            base_model_temp,
            label="Base Model (FP32/FP16 Unmodified)",
        )

        del base_model_temp
        gc.collect()
        if USE_CUDA:
            torch.cuda.empty_cache()

        trainer = ListwiseSFTTrainer(
            model=self.model_path,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            quantization_config=self.bnb_config,
            peft_config=self.lora_config,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.make_compute_metrics(eval_qrels_dict),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),
                StepHeartbeatCallback(),
            ],
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            ranker_emb_dim=RANKER_EMB_DIM,
            emb_token_id=self.emb_token_id,
        )

        self.print_model_characteristics(
            trainer.model,
            label=f"Qwen2.5-0.5B + QLoRA (include_ranker_embeddings={INCLUDE_RANKER_EMBEDDINGS})",
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Final Evaluation Metrics:", metrics)
        self.assertIn(f"eval_ndcg@{TOP_K}", metrics)

        # Assert model saving, loading, and equivalence
        save_test_path = os.path.join(self.output_dir, "saved_checkpoint")
        self.assertLoadedModelIsEqual(
            trained_trainer=trainer,
            save_path=save_test_path,
            test_dataset=eval_dataset,
            eval_qrels_dict=eval_qrels_dict,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )

    def testOptunaHPO(self):
        n_train = 40 if USE_CUDA else 2
        n_eval = 16 if USE_CUDA else 2
        n_trials = 4 if USE_CUDA else 2
        step_freq = 5 if USE_CUDA else 1

        train_dataset, _ = self.build_fake_dataset(
            num_samples=n_train,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )
        eval_dataset, eval_qrels_dict = self.build_fake_dataset(
            num_samples=n_eval,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )
        compute_metrics_fn = self.make_compute_metrics(eval_qrels_dict)

        best_trained_trainer = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_trained_trainer
            r_val = trial.suggest_categorical("lora_r", [8, 16, 32])
            lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1, step=0.05)
            lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

            trial_lora = LoraConfig(
                r=r_val,
                lora_alpha=r_val * 2,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.lora_config.target_modules,
            )

            trial_output_dir = os.path.join(self.output_dir, f"trial_{trial.number}")
            trial_sft_config = self.build_sft_config(
                output_dir=trial_output_dir,
                learning_rate=lr,
                weight_decay=weight_decay,
                step_freq=step_freq,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )

            trainer = ListwiseSFTTrainer(
                model=self.model_path,
                args=trial_sft_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,
                quantization_config=self.bnb_config,
                peft_config=trial_lora,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
                compute_metrics=compute_metrics_fn,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=2),
                    StepHeartbeatCallback(),
                ],
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
                ranker_emb_dim=RANKER_EMB_DIM,
                emb_token_id=self.emb_token_id,
            )

            trainer.train()
            if trainer.state.best_metric is not None:
                target_score = float(trainer.state.best_metric)
            else:
                metrics = trainer.evaluate()
                target_score = float(metrics[f"eval_ndcg@{TOP_K}"])

            # Keep reference to the best trainer for final saving and equivalence assertion
            if trial.number == 0 or target_score >= trial.study.best_value:
                best_trained_trainer = trainer

            return target_score

        study = optuna.create_study(
            study_name=f"qwen_reranker_hpo_emb_{INCLUDE_RANKER_EMBEDDINGS}",
            direction="maximize",
        )
        study.optimize(objective, n_trials=n_trials)

        print("Best Trial Number:", study.best_trial.number)
        print(
            f"Best eval_ndcg@{TOP_K} (include_ranker_embeddings={INCLUDE_RANKER_EMBEDDINGS}):",
            study.best_value,
        )
        print("Best Hyperparameters:", study.best_params)

        self.assertIsNotNone(study.best_value)
        self.assertGreaterEqual(study.best_value, 0.0)

        # Assert model saving, loading, and equivalence on the best HPO trial
        if best_trained_trainer is not None:
            save_test_path = os.path.join(self.output_dir, "hpo_best_checkpoint")
            self.assertLoadedModelIsEqual(
                trained_trainer=best_trained_trainer,
                save_path=save_test_path,
                test_dataset=eval_dataset,
                eval_qrels_dict=eval_qrels_dict,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )

    def build_fake_dataset(
        self,
        num_samples: int = 40,
        include_ranker_embeddings: bool = INCLUDE_RANKER_EMBEDDINGS,
    ):
        records = []
        qrels_dict = {}

        gen = torch.Generator().manual_seed(42)

        sample_history = [
            {
                "title": format_movielens_title("Matrix, The (1999)"),
                "genres": format_movielens_genres("Action|Sci-Fi|Thriller"),
                "rating": 5,
                "recency": "10d ago",
            },
            {
                "title": format_movielens_title(
                    "Open Your Eyes (Abre los ojos) (1997)"
                ),
                "genres": format_movielens_genres("Drama|Romance|Sci-Fi"),
                "rating": 4,
                "recency": "1d ago",
            },
        ]

        raw_candidates = [
            ("A", "Toy Story 2 (1999)", "Animation|Children's|Comedy"),
            (
                "B",
                "Star Wars: Episode IV - A New Hope (1977)",
                "Action|Adventure|Fantasy|Sci-Fi",
            ),
            ("C", "Across the Sea of Time (1995)", "Documentary"),
            ("D", "Big Green, The (1995)", "Children's|Comedy"),
            ("E", "Sense and Sensibility (1995)", "Drama|Romance"),
        ]
        sample_candidates = [
            {
                "label": lbl,
                "title": format_movielens_title(title),
                "genres": format_movielens_genres(genres),
            }
            for lbl, title, genres in raw_candidates[:TOP_K]
        ]
        ranked_order = ["B", "E", "A", "C", "D"][:TOP_K]

        for i in range(num_samples):
            prompt, completion = self.render_prompt_and_completion(
                query_time="Fall, Sat Night (Oct)",
                history_items=sample_history,
                candidate_items=sample_candidates,
                ranked_labels_List=ranked_order,
                include_ranker_embeddings=include_ranker_embeddings,
            )
            record = {"prompt": prompt, "completion": completion}

            if include_ranker_embeddings:
                user_vec = torch.randn(1, RANKER_EMB_DIM, generator=gen)
                cand_vecs = torch.randn(TOP_K, RANKER_EMB_DIM, generator=gen)
                cand_vecs[1] = user_vec[0] + 0.1 * torch.randn(
                    RANKER_EMB_DIM, generator=gen
                )
                all_embs = torch.cat([user_vec, cand_vecs], dim=0).tolist()
                record["ranker_embs"] = all_embs

            records.append(record)
            qrels_dict[f"q_{i}"] = {
                "B": 1.0,
                "E": 0.0,
                "A": 0.0,
                "C": 0.0,
                "D": 0.0,
            }

        return Dataset.from_list(records), qrels_dict

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.ndim == 2:
            return logits

        first_target_idx = (labels != -100).int().argmax(dim=-1)
        pred_pos = (first_target_idx - 1).clamp(min=0)

        batch_indices = torch.arange(logits.size(0), device=logits.device)
        first_step_logits = logits[batch_indices, pred_pos]

        cand_token_tensor = torch.tensor(
            self.candidate_token_ids, device=logits.device
        )
        return first_step_logits[:, cand_token_tensor]

    def make_compute_metrics(self, qrels_dict: Dict[str, Dict[str, float]]):
        qrels = Qrels(qrels_dict)
        ndcg_key = f"ndcg@{TOP_K}"
        mrr_key = f"mrr@{TOP_K}"

        def compute_metrics(eval_pred):
            candidate_scores, _ = eval_pred
            run_dict = {
                f"q_{idx}": {
                    CANDIDATE_LABELS[c_idx]: float(scores[c_idx])
                    for c_idx in range(len(CANDIDATE_LABELS))
                }
                for idx, scores in enumerate(candidate_scores)
            }
            results = evaluate(qrels, Run(run_dict), [ndcg_key, mrr_key])
            return {ndcg_key: results[ndcg_key], mrr_key: results[mrr_key]}

        return compute_metrics


if __name__ == "__main__":
    unittest.main()