import gc
import math
import os
import re
import shutil
import sys
import unittest
from typing import Any, Dict, List, Tuple

HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_LOCAL_CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs"
)
LOCAL_CACHE_EXISTS = (
    os.path.exists(_LOCAL_CACHE_PATH) and len(os.listdir(_LOCAL_CACHE_PATH)) > 0
)
if LOCAL_CACHE_EXISTS:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

if not HAS_NVIDIA_GPU:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
else:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np
import onnxruntime as ort
import optuna
import torch
import yaml
from accelerate import PartialState
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from ranx import Qrels, Run, evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig

from movie_lens_reranker.export import export_to_triton_onnx
from movie_lens_reranker.ListwiseSFTTrainer import ListwiseSFTTrainer
from movie_lens_reranker.RerankerPromptFormatter import RerankerPromptFormatter

if not torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

from helper import *

from movie_lens_reranker.prompts.prompt_helper import Prompt_Type, \
    get_yaml_prompt_path, load_prompt_yamls

TOP_K = 5
MAX_HISTORY_ITEMS = 2
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
USE_CUDA = torch.cuda.is_available()
CANDIDATE_LABELS = [chr(ord("A") + i) for i in range(TOP_K)]

INCLUDE_RANKER_EMBEDDINGS = True
RANKER_EMB_DIM = 24
EMB_PLACEHOLDER_TOKEN = "<|vision_pad|>"
DTYPE = torch.float16 if USE_CUDA else torch.float32


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


class TestSupervisedFineTuning(unittest.TestCase):

    def setUp(self):
        self.token = get_hf_token()
        local_cache_exists = LOCAL_CACHE_EXISTS

        self.base_model_path = (
            snapshot_download(repo_id=MODEL_ID, local_files_only=True)
            if local_cache_exists
            else MODEL_ID
        )

        self.output_dir = os.path.join(get_bin_dir(), "prototype_qwen_reranker")
        os.makedirs(self.output_dir, exist_ok=True)
        self.use_qlora_4bit = USE_CUDA

        # 1. Load YAML Prompt Templates & Instantiate RerankerPromptFormatter
        self.query_yaml, self.passages_yaml = load_prompt_yamls(version="2.0.0")
        self.prompt_formatter = RerankerPromptFormatter(
            query_yaml=self.query_yaml,
            passages_yaml=self.passages_yaml,
            emb_placeholder_token=EMB_PLACEHOLDER_TOKEN,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            top_k=TOP_K,
        )

        # 2. Tokenizer & Candidate Token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            token=self.token,
            local_files_only=local_cache_exists,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.candidate_token_ids: list[int] = [
            self.tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in CANDIDATE_LABELS
        ]

        emb_token_ids: list[int] = self.tokenizer.encode(
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
                bnb_4bit_compute_dtype=DTYPE,
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

    def testTorchRun(self):
        n_train = 40 if USE_CUDA else 4
        n_eval = 16 if USE_CUDA else 2

        train_dataset, _ = self.build_fake_dataset(
            num_samples=n_train,
            parquet_filename="train_diagnostic.parquet",
        )
        eval_dataset, eval_qrels_dict = self.build_fake_dataset(
            num_samples=n_eval,
            parquet_filename="eval_diagnostic.parquet",
        )

        sft_config = self.build_sft_config(
            output_dir=self.output_dir,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )

        base_model_temp = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
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
            model=self.base_model_path,
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

        if PartialState().is_main_process:
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
            parquet_filename="hpo_train_diagnostic.parquet",
        )
        eval_dataset, eval_qrels_dict = self.build_fake_dataset(
            num_samples=n_eval,
            parquet_filename="hpo_eval_diagnostic.parquet",
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
                model=self.base_model_path,
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

            if trial.number == 0 or target_score >= trial.study.best_value:
                best_trained_trainer = trainer

            return target_score

        study = optuna.create_study(
            study_name=f"qwen_reranker_hpo_emb_{INCLUDE_RANKER_EMBEDDINGS}",
            direction="maximize",
        )
        study.optimize(objective, n_trials=n_trials)

        self.assertIsNotNone(study.best_value)
        self.assertGreaterEqual(study.best_value, 0.0)

        if best_trained_trainer is not None and PartialState().is_main_process:
            save_test_path = os.path.join(self.output_dir, "hpo_best_checkpoint")
            self.assertLoadedModelIsEqual(
                trained_trainer=best_trained_trainer,
                save_path=save_test_path,
                test_dataset=eval_dataset,
                eval_qrels_dict=eval_qrels_dict,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )

    
    def calculate_max_length(
        self,
        top_k: int = TOP_K,
        max_history_items: int = MAX_HISTORY_ITEMS,
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

        worst_prompt, worst_completion = (
            self.prompt_formatter.render_prompt_and_completion(
                query_time="Winter, Wed Afternoon (Dec)",
                history_items=worst_history,
                candidate_items=worst_candidates,
                ranked_labels=labels,
            )
        )

        full_text = worst_prompt + worst_completion
        token_count = len(self.tokenizer.encode(full_text, add_special_tokens=True))
        total_with_margin = token_count + safety_margin
        return int(math.ceil(total_with_margin / 32.0) * 32)

    def build_model_init_kwargs(
        self,
        token: str,
        local_cache_exists: bool,
        cpu_smoke_test: bool = False,
    ) -> dict:
        use_cuda = torch.cuda.is_available()
        device_index = PartialState().process_index

        kwargs = {
            "dtype": DTYPE,
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
            fp16=(DTYPE == torch.float16),
            bf16=(DTYPE == torch.bfloat16),
            use_cpu=not USE_CUDA,
            logging_first_step=True,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=step_freq,
            save_strategy="steps",
            save_steps=step_freq,
            save_total_limit=2,
            load_best_model_at_end=True,
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

    def build_fake_dataset(
        self,
        num_samples: int = 40,
        parquet_filename: str = "fake_stage2_diagnostic.parquet",
    ) -> Tuple[Dataset, Dict[str, Dict[str, float]]]:
        gen = torch.Generator().manual_seed(42)

        sample_history = [
            {
                "movie_id": 6041,
                "title": format_movielens_title("Matrix, The (1999)"),
                "genres": format_movielens_genres("Action|Sci-Fi|Thriller"),
                "rating": 5,
                "recency": "10d ago",
            },
            {
                "movie_id": 6042,
                "title": format_movielens_title(
                    "Open Your Eyes (Abre los ojos) (1997)"
                ),
                "genres": format_movielens_genres("Drama|Romance|Sci-Fi"),
                "rating": 4,
                "recency": "1d ago",
            },
        ]

        raw_candidates = [
            (6050, "A", "Toy Story 2 (1999)", "Animation|Children's|Comedy", 0, 0.40),
            (
                6051,
                "B",
                "Star Wars: Episode IV - A New Hope (1977)",
                "Action|Adventure|Fantasy|Sci-Fi",
                1,
                0.95,
            ),
            (6052, "C", "Across the Sea of Time (1995)", "Documentary", 0, 0.20),
            (6053, "D", "Big Green, The (1995)", "Children's|Comedy", 0, 0.10),
            (6054, "E", "Sense and Sensibility (1995)", "Drama|Romance", 0, 0.80),
        ]

        cand_items = [
            {
                "label": lbl,
                "movie_id": mid,
                "title": format_movielens_title(title),
                "genres": format_movielens_genres(genres),
                "score": score,
            }
            for mid, lbl, title, genres, _, score in raw_candidates[:TOP_K]
        ]
        labels_list = [lbl_val for _, _, _, _, lbl_val, _ in raw_candidates[:TOP_K]]
        scores_list = [score for _, _, _, _, _, score in raw_candidates[:TOP_K]]
        ranked_order = ["B", "E", "A", "C", "D"][:TOP_K]

        parquet_records = []
        for i in range(num_samples):
            user_vec = torch.randn(1, RANKER_EMB_DIM, generator=gen)
            cand_vecs = torch.randn(TOP_K, RANKER_EMB_DIM, generator=gen)
            cand_vecs[1] = user_vec[0] + 0.1 * torch.randn(
                RANKER_EMB_DIM, generator=gen
            )
            combined_embs = torch.cat([user_vec, cand_vecs], dim=0)

            parquet_records.append(
                {
                    "user_id": i + 1,
                    "movie_id": 6051,
                    "rating": 5,
                    "timestamp": 972768000 + i,
                    "query_time": "Fall, Sat Night (Oct)",
                    "history_items": sample_history,
                    "candidate_items": cand_items,
                    "candidate_scores": scores_list,
                    "labels": labels_list,
                    "ranked_labels": ranked_order,
                    "user_embedding": user_vec[0].tolist(),
                    "candidate_embeddings": cand_vecs.tolist(),
                    "ranker_embs": combined_embs.tolist(),
                }
            )

        parquet_path = os.path.join(self.output_dir, parquet_filename)
        if PartialState().is_main_process:
            RerankerPreprocessor.write_training_parquet(
                parquet_records, parquet_path, append=False
            )

        return self.prompt_formatter.load_and_format_parquet_dataset(
            parquet_files=parquet_path,
            seed=42,
        )

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

    def make_compute_metrics(self, qrels_dict: dict[str, dict[str, float]]):
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

    def assertLoadedModelIsEqual(
        self,
        trained_trainer: ListwiseSFTTrainer,
        save_path: str,
        test_dataset: Dataset,
        eval_qrels_dict: Dict[str, Dict[str, float]],
        include_ranker_embeddings: bool,
    ):
        trained_metrics = trained_trainer.evaluate()

        onnx_path = export_to_triton_onnx(
            trained_trainer=trained_trainer,
            tokenizer=self.tokenizer,
            model_init_kwargs=self.model_init_kwargs,
            emb_token_id=self.emb_token_id,
            candidate_token_ids=self.candidate_token_ids,
            top_k=TOP_K,
            ranker_emb_dim=RANKER_EMB_DIM,
            candidate_labels=CANDIDATE_LABELS,
            base_model_path=self.base_model_path,
            triton_model_dir=save_path,
            include_ranker_embeddings=include_ranker_embeddings,
        )
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.pbtxt")))

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        self.tokenizer.padding_side = "left"
        prompts = [row["prompt"] for row in test_dataset]
        enc = self.tokenizer(
            prompts, return_tensors="np", padding=True, truncation=True
        )

        n_samples = len(prompts)
        if include_ranker_embeddings:
            ranker_embs_np = np.asarray(
                [row["ranker_embs"] for row in test_dataset], dtype=np.float32
            )
        else:
            ranker_embs_np = np.zeros(
                (n_samples, 1 + TOP_K, RANKER_EMB_DIM), dtype=np.float32
            )

        onnx_outputs = sess.run(
            ["candidate_log_probs"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "ranker_embs": ranker_embs_np,
            },
        )
        onnx_cand_log_probs = onnx_outputs[0]
        self.assertEqual(onnx_cand_log_probs.shape, (n_samples, TOP_K))

        trained_model = trained_trainer.model
        trained_model.eval()
        device = next(trained_model.parameters()).device
        enc0 = self.tokenizer(prompts[0], return_tensors="pt", padding=False)
        eval_inputs0 = {k: v.to(device) for k, v in enc0.items()}

        with torch.no_grad():
            if include_ranker_embeddings:
                trained_trainer._current_ranker_embs = torch.tensor(
                    ranker_embs_np[0:1], dtype=torch.float32, device=device
                )
                trained_trainer._current_input_ids = eval_inputs0["input_ids"]
            pt_out0 = trained_model(**eval_inputs0)
            cand_token_tensor = torch.tensor(self.candidate_token_ids, device=device)
            pt_log_probs0 = (
                torch.log_softmax(
                    pt_out0.logits[0, -1, cand_token_tensor].float(), dim=-1
                )
                .cpu()
                .numpy()
            )

        if not self.use_qlora_4bit:
            np.testing.assert_allclose(
                onnx_cand_log_probs[0],
                pt_log_probs0,
                rtol=1e-3,
                atol=1e-3,
                err_msg="ONNX Runtime candidate log-probs do not match PyTorch model!",
            )

        compute_metrics_fn = self.make_compute_metrics(eval_qrels_dict)
        onnx_metrics = compute_metrics_fn((onnx_cand_log_probs, None))

        self.assertAlmostEqual(
            float(trained_metrics[f"eval_ndcg@{TOP_K}"]),
            float(onnx_metrics[f"ndcg@{TOP_K}"]),
            places=4,
        )
        self.assertAlmostEqual(
            float(trained_metrics[f"eval_mrr@{TOP_K}"]),
            float(onnx_metrics[f"mrr@{TOP_K}"]),
            places=4,
        )
        print(
            f"\n[Verification] Exported Triton ONNX model verified in-memory! "
            f"NDCG@{TOP_K}={onnx_metrics[f'ndcg@{TOP_K}']:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    unittest.main()