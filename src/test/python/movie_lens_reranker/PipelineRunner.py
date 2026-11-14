import gc
import json
import math
import os
import shutil
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ON_KAGGLE = os.path.exists("/kaggle") or ("kaggle" in os.getcwd())
if ON_KAGGLE:
    HF_HOME_DIR = "/tmp/.cache/huggingface"
else:
    HF_HOME_DIR = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME_DIR, "datasets")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_MODEL_REPO_DIR = f"models--{MODEL_ID.replace('/', '--')}"
_LOCAL_BLOBS_PATH = os.path.join(
    os.environ["HF_HUB_CACHE"], _MODEL_REPO_DIR, "blobs"
)
LOCAL_CACHE_EXISTS = (
    os.path.exists(_LOCAL_BLOBS_PATH) and len(os.listdir(_LOCAL_BLOBS_PATH)) > 0
)

if LOCAL_CACHE_EXISTS:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

CPU_THREADS = str(min(4, max(1, os.cpu_count() or 1)))
if not HAS_NVIDIA_GPU:
    os.environ.setdefault("OMP_NUM_THREADS", CPU_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", CPU_THREADS)
    os.environ.setdefault("NUMBA_NUM_THREADS", CPU_THREADS)
else:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import torch

if not torch.cuda.is_available():
    torch.set_num_threads(int(CPU_THREADS))
    torch.set_num_interop_threads(1)

import numpy as np
import onnxruntime as ort
import optuna
from accelerate import PartialState
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig

sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker"))
sys.path.append(os.path.join(os.getcwd(), "src/main/python/movie_lens_reranker"))
from helper import *
from movie_lens_reranker.Callbacks import StepHeartbeatCallback
from movie_lens_reranker.Movies import Movies
from movie_lens_reranker.RecommendedMovies import RecommendedMovies
from movie_lens_reranker.UserHistory import UserHistory
from movie_lens_reranker.util import read_movie_tiers
from movie_lens_reranker.export import (
    export_from_reusable_checkpoint,
    export_to_triton_onnx,
    load_reusable_checkpoint_in_memory,
    save_reusable_checkpoint,
)
from movie_lens_reranker.ListwiseSFTTrainer import ListwiseSFTTrainer
from movie_lens_reranker.RerankerPromptFormatter import RerankerPromptFormatter
from movie_lens_reranker.reranker_metrics import (
    make_reranker_compute_metrics,
)
from movie_lens_reranker.prompts.prompt_helper import load_prompt_yamls

NUM_CANDIDATES = 20  # Number of candidates in prompt & model output head ('A'..'T')
EVAL_TOP_K = 5       # Ranking evaluation cutoff for NDCG@K and MRR@K ('@5')
TOP_K = NUM_CANDIDATES
MAX_SEQ_TOKEN_LENGTH = 1024
MAX_PROMPT_HISTORY = 10
USE_CUDA = torch.cuda.is_available()
CANDIDATE_LABELS = RerankerPromptFormatter.candidate_labels(NUM_CANDIDATES)

INCLUDE_RANKER_EMBEDDINGS = True
EMB_PLACEHOLDER_TOKEN = "<|vision_pad|>"
DTYPE = torch.float16 if USE_CUDA else torch.float32

# Metric used for EarlyStoppingCallback, checkpoint selection, and Optuna HPO
PRIMARY_METRIC_NAME = f"composite_ndcg@{EVAL_TOP_K}"
EVAL_PRIMARY_METRIC_NAME = f"eval_{PRIMARY_METRIC_NAME}"

with open(
    os.path.join(get_data_dir(), "training_hyperparameters.json"), encoding="utf-8"
) as f:
    ranker_metadata_dict = json.load(f)
TIER_WEIGHTS: Tuple[float, float, float] = tuple(ranker_metadata_dict["tier_weights"])
IPW_TIER_LOSS_WEIGHTS: Tuple[float, float, float] = (0.333, 0.333, 0.333)
NUM_CATALOG_USERS = ranker_metadata_dict["num_users"]  # 6040
RANKER_OUT_FEATURES = ranker_metadata_dict["out_dim"]

# Bounds for periodic in-training eval, CPU is_unit_test tests, and HPO
MAX_CPU_TRAIN_SAMPLES = 16
MAX_ACTIVE_EVAL_SAMPLES = 2048 if USE_CUDA else 16
MAX_HPO_TRAIN_SAMPLES = 20_000 if USE_CUDA else 16
MAX_ONNX_VERIFY_SAMPLES = 256 if USE_CUDA else 8


class TrainPhase(Enum):
    BEST = "BEST"
    FIXED = "FIXED"


class PipelineRunner:
    _dataset_cache: Dict[
        Tuple[Any, str], Tuple[Dataset, Dict[str, Dict[str, int]], int]
    ] = {}

    def __init__(self, is_unit_test: bool = False):
        self.is_unit_test = bool(is_unit_test)
        self.token = get_hf_token()

        state = PartialState()

        # On Kaggle T4x2 DDP, both processes share /tmp; let Rank 0 download first if not cached
        if not LOCAL_CACHE_EXISTS and state.is_main_process:
            snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=os.environ["HF_HUB_CACHE"],
                token=self.token,
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Resolve the exact local snapshot folder inside /tmp/.cache/huggingface/hub
        self.base_model_path = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=os.environ["HF_HUB_CACHE"],
            local_files_only=True,
        )

        self.output_dir = os.path.join(get_output_dir(), "prototype_qwen_reranker")
        if PartialState().is_main_process:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir, ignore_errors=True)
            os.makedirs(self.output_dir, exist_ok=True)
        self.use_qlora_4bit = USE_CUDA

        self.output_working_dir = get_output_working_dir()
        if PartialState().is_main_process:
            os.makedirs(self.output_working_dir, exist_ok=True)

        # Tokenizer & Candidate Token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            token=self.token,
            local_files_only=LOCAL_CACHE_EXISTS,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization & Base LoRA Target Modules
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

        self.lora_config_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        # Model Init Kwargs
        self.model_init_kwargs = self._build_model_init_kwargs(
            token=self.token,
            is_unit_test=self.is_unit_test,
        )

        # DATA
        data_dir = get_data_dir()
        recommendations_uri = os.path.join(data_dir, "recommended_movies.parquet")
        recommendations_timestamps_uri = os.path.join(
            data_dir, "recommended_movies_timestamps.parquet"
        )
        self.recommendations = RecommendedMovies(
            num_users=NUM_CATALOG_USERS,
            movie_rec_file_uri=recommendations_uri,
            movie_rec_ts_file_uri=recommendations_timestamps_uri,
        )
        movies_path = os.path.join(data_dir, "movies.parquet")
        self.movies = Movies(movies_uri=movies_path)

        # Movie popularity tiers: indices are (movie_id - movie_offset), values in {0, 1, 2}
        # for head, torso, tail, with row 0 reserved for OOV movie_id=0.
        self.movie_tiers, self.movie_offset, self.num_catalog_movies = read_movie_tiers(
            os.path.join(data_dir, "movie_tiers.json")
        )

        self.DataSize = DataSize.TINY3 if self.is_unit_test else DataSize.FULL

        self.candidate_token_ids: List[int] = [
            self.tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in CANDIDATE_LABELS
        ]

        emb_token_ids: List[int] = self.tokenizer.encode(
            EMB_PLACEHOLDER_TOKEN, add_special_tokens=False
        )
        assert len(emb_token_ids) == 1, (
            f"Expected {EMB_PLACEHOLDER_TOKEN} to map to a single token ID, got {emb_token_ids}"
        )
        self.emb_token_id = emb_token_ids[0]

        full_train_dataset, full_train_qrels, ranker_out_features = self.get_dataset(
            is_train=True,
        )
        assert RANKER_OUT_FEATURES == ranker_out_features, (
            f"Expected RANKER_OUT_FEATURES={RANKER_OUT_FEATURES}, got {ranker_out_features}"
        )
        full_eval_dataset, full_eval_qrels, _ = self.get_dataset(
            is_train=False,
        )

        train_cap = (
            MAX_CPU_TRAIN_SAMPLES
            if (self.is_unit_test and not USE_CUDA)
            else MAX_HPO_TRAIN_SAMPLES
        )
        eval_cap = (
            MAX_CPU_TRAIN_SAMPLES
            if (self.is_unit_test and not USE_CUDA)
            else MAX_ACTIVE_EVAL_SAMPLES
        )

        self.train_dataset, self.train_qrels_dict = self._subsample_dataset_and_qrels(
            full_train_dataset, full_train_qrels, train_cap
        )
        self.eval_dataset, self.eval_qrels_dict = self._subsample_dataset_and_qrels(
            full_eval_dataset, full_eval_qrels, eval_cap
        )

        # Output paths
        self.best_params_path = os.path.join(
            self.output_working_dir, "best_hyperparameters.json"
        )
        self.portable_ckpt_dir = os.path.join(
            self.output_working_dir, "portable_checkpoint"
        )

    def _compute_training_schedule(
        self,
        n_train: int,
        n_eval: int,
        is_hpo: bool = False,
    ) -> Dict[str, Any]:
        """
        Computes batch sizes, gradient accumulation, epochs, max_steps, eval/save step
        frequency, early stopping patience, and HPO trial counts dynamically based on
        dataset size and hardware (Local/Kaggle CPU vs. Kaggle T4x2 GPUs).
        """
        num_procs = max(1, PartialState().num_processes)

        if USE_CUDA and not self.is_unit_test:
            # Kaggle T4 (16GB VRAM per GPU) with Qwen2.5-0.5B 4-bit NF4 and <=1024 token sequences
            max_steps = -1
            if n_train <= 1_000:
                train_batch_size = 4
                eval_batch_size = 8
                grad_accum = 2
                epochs = 1 if is_hpo else 2
                n_trials = 4
                early_stopping_patience = 2
            elif n_train <= 50_000:
                train_batch_size = 16
                eval_batch_size = 32
                grad_accum = 2
                epochs = 4 if is_hpo else 4
                n_trials = 6
                early_stopping_patience = 3
            else:
                # Full scale positives (~460,000 train prompts, ~52,000 eval prompts)
                train_batch_size = 16
                eval_batch_size = 32
                grad_accum = 4
                epochs = 1
                n_trials = 6
                early_stopping_patience = 3
        else:
            # Local / Kaggle CPU fast prototype unit test
            train_batch_size = 4
            eval_batch_size = 8
            grad_accum = 1
            epochs = 1
            max_steps = 2
            n_trials = 2
            early_stopping_patience = 2

        eff_batch = max(1, train_batch_size * num_procs * grad_accum)
        steps_per_epoch = max(1, math.ceil(n_train / eff_batch))
        total_steps = (
            max_steps if max_steps > 0 else max(1, steps_per_epoch * epochs)
        )

        if total_steps <= 20:
            step_freq = max(1, total_steps)
            logging_steps = 1
        else:
            step_freq = max(10, steps_per_epoch)
            logging_steps = max(1, steps_per_epoch // 5)

        return {
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "grad_accum": grad_accum,
            "epochs": epochs,
            "max_steps": max_steps,
            "step_freq": step_freq,
            "logging_steps": logging_steps,
            "n_trials": n_trials,
            "early_stopping_patience": early_stopping_patience,
            "use_grad_checkpointing": USE_CUDA and train_batch_size >= 16,
        }

    @staticmethod
    def _subsample_dataset_and_qrels(
        dataset: Dataset,
        qrels_dict: Dict[str, Dict[str, int]],
        max_samples: int,
        seed: int = 42,
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]]]:
        """Deterministically subsamples a dataset stratified by `pos_tier` and re-indexes qrels."""
        if len(dataset) <= max_samples:
            return dataset, qrels_dict

        rng = np.random.default_rng(seed)
        total = len(dataset)

        if "pos_tier" not in dataset.column_names:
            chosen_indices = np.sort(
                rng.choice(total, size=max_samples, replace=False)
            ).tolist()
        else:
            tiers = np.asarray(dataset["pos_tier"], dtype=np.int64)
            chosen_parts = []
            for tier_id in (0, 1, 2):
                tier_indices = np.where(tiers == tier_id)[0]
                if len(tier_indices) == 0:
                    continue
                n_take = max(
                    1, int(round(max_samples * (len(tier_indices) / total)))
                )
                n_take = min(n_take, len(tier_indices))
                chosen_parts.append(
                    rng.choice(tier_indices, size=n_take, replace=False)
                )

            chosen_indices = np.sort(np.concatenate(chosen_parts))
            if len(chosen_indices) > max_samples:
                chosen_indices = np.sort(
                    rng.choice(chosen_indices, size=max_samples, replace=False)
                )
            chosen_indices = chosen_indices.tolist()

        sub_ds = dataset.select(chosen_indices)
        sub_qrels = {
            f"q_{new_idx}": qrels_dict[f"q_{orig_idx}"]
            for new_idx, orig_idx in enumerate(chosen_indices)
        }
        return sub_ds, sub_qrels

    def run_pipeline(self):
        self.run_hpo_phase()
        self.run_train_phase()
        return self.run_test_phase()

    def run_train_phase(
        self,
        train_phase: TrainPhase = TrainPhase.BEST,
        hyperparams: Optional[Dict[str, Union[float, int]]] = None,
    ):
        if train_phase == TrainPhase.BEST:
            if os.path.exists(self.best_params_path):
                with open(self.best_params_path, "r", encoding="utf-8") as f:
                    loaded_hpo = json.load(f)
                    hyperparams = loaded_hpo["best_params"]
            else:
                raise ValueError(
                    f"Cannot find best params file: {self.best_params_path}"
                )
        elif hyperparams is None:
            hyperparams = {
                "lora_r": 16 if USE_CUDA else 8,
                "lora_dropout": 0.05,
                "learning_rate": 2e-4,
                "weight_decay": 0.0,
            }

        for key in ("lora_r", "lora_dropout", "learning_rate", "weight_decay"):
            if key not in hyperparams:
                raise ValueError(f"Cannot find expected {key} in hyperparams")

        r_val = int(hyperparams["lora_r"])
        lora_config = LoraConfig(
            r=r_val,
            lora_alpha=r_val * 2,
            lora_dropout=float(hyperparams["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_config_target_modules,
        )

        sched = self._compute_training_schedule(
            n_train=len(self.train_dataset),
            n_eval=len(self.eval_dataset),
            is_hpo=False,
        )

        sft_config = self._build_sft_config(
            output_dir=self.output_dir,
            learning_rate=float(hyperparams["learning_rate"]),
            weight_decay=float(hyperparams["weight_decay"]),
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            train_batch_size=sched["train_batch_size"],
            eval_batch_size=sched["eval_batch_size"],
            grad_accum=sched["grad_accum"],
            epochs=sched["epochs"],
            max_steps=sched["max_steps"],
            step_freq=sched["step_freq"],
            logging_steps=sched["logging_steps"],
            gradient_checkpointing=sched["use_grad_checkpointing"],
        )

        if USE_CUDA:
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
            torch.cuda.empty_cache()

        trainer = ListwiseSFTTrainer(
            model=self.base_model_path,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            quantization_config=self.bnb_config,
            peft_config=lora_config,
            preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
            compute_metrics=self._make_compute_metrics(
                self.eval_qrels_dict, self.eval_dataset
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=sched["early_stopping_patience"],
                    early_stopping_threshold=1e-4,
                ),
                StepHeartbeatCallback(),
            ],
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            ranker_emb_dim=RANKER_OUT_FEATURES,
            emb_token_id=self.emb_token_id,
            candidate_token_ids=self.candidate_token_ids,
            tier_loss_weights=IPW_TIER_LOSS_WEIGHTS,
            eval_top_k=EVAL_TOP_K,
            eval_tier_weights=TIER_WEIGHTS,
            distill_alpha=0.15,
            distill_temperature=2.0,
        )

        self.print_model_characteristics(
            trainer.model,
            label=f"Qwen2.5-0.5B + QLoRA (include_ranker_embeddings={INCLUDE_RANKER_EMBEDDINGS})",
        )

        trainer.train()

        if PartialState().is_main_process:
            history_path = os.path.join(
                self.output_working_dir, "training_metrics_history.json"
            )
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(
                f"[MetricsTracker] Saved training and evaluation history to {history_path}"
            )

        metrics = trainer.evaluate()
        print("Final Evaluation Metrics:", metrics)
        assert EVAL_PRIMARY_METRIC_NAME in metrics, (
            f"Missing {EVAL_PRIMARY_METRIC_NAME} in {metrics.keys()}"
        )
        assert f"eval_ndcg@{EVAL_TOP_K}" in metrics, (
            f"Missing eval_ndcg@{EVAL_TOP_K} in {metrics.keys()}"
        )

        if PartialState().is_main_process:
            # Save ~11.5 MB portable checkpoint (Orbax equivalent)
            save_reusable_checkpoint(
                trained_trainer=trainer,
                tokenizer=self.tokenizer,
                checkpoint_dir=self.portable_ckpt_dir,
                base_model_id=MODEL_ID,
                emb_token_id=self.emb_token_id,
                candidate_token_ids=self.candidate_token_ids,
                top_k=NUM_CANDIDATES,
                ranker_emb_dim=RANKER_OUT_FEATURES,
                candidate_labels=CANDIDATE_LABELS,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )

        return metrics

    def run_test_phase(self, quantize_mode: str = "smoothquant_w8a8") -> Dict[str, Any]:
        """
        Exports the Portable Checkpoint to FP32, SmoothQuant W8A8, and AWQ W4A16 Triton ONNX,
        verifies in-memory signature and parity, and evaluates on the test split
        (subsampled to 2,048 if `is_unit_test=True`, otherwise the full ~52,000 test split).
        """
        if not os.path.exists(self.portable_ckpt_dir):
            raise ValueError(
                f"Portable ckpt dir not found at {self.portable_ckpt_dir}"
            )

        full_test_ds, full_test_qrels, _ = self.get_dataset(
            is_train=False, is_test=True
        )
        if self.is_unit_test:
            test_cap = 2048 if USE_CUDA else 16
            test_dataset, test_qrels_dict = self._subsample_dataset_and_qrels(
                full_test_ds, full_test_qrels, test_cap
            )
        else:
            test_dataset, test_qrels_dict = full_test_ds, full_test_qrels

        calib_prompts = [row["prompt"] for row in self.eval_dataset][:32]

        fp32_onnx = export_from_reusable_checkpoint(
            checkpoint_dir=self.portable_ckpt_dir,
            triton_model_dir=os.path.join(self.output_dir, "triton_fp32"),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="fp32",
        )
        sq_onnx = export_from_reusable_checkpoint(
            checkpoint_dir=self.portable_ckpt_dir,
            triton_model_dir=os.path.join(
                self.output_working_dir, "triton_smoothquant_w8a8"
            ),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="smoothquant_w8a8",
            calibration_prompts=calib_prompts,
        )
        awq_onnx = export_from_reusable_checkpoint(
            checkpoint_dir=self.portable_ckpt_dir,
            triton_model_dir=os.path.join(
                self.output_working_dir, "triton_awq_w4a16"
            ),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="awq_w4a16",
            calibration_prompts=calib_prompts,
        )

        # Verify in-memory signature & parity across PyTorch, FP32, SmoothQuant W8A8, and AWQ W4A16
        self._assert_exported_models_work_in_memory(
            portable_ckpt_dir=self.portable_ckpt_dir,
            fp32_onnx_path=fp32_onnx,
            sq_onnx_path=sq_onnx,
            awq_onnx_path=awq_onnx,
            eval_dataset=test_dataset,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            num_examples=min(2, len(test_dataset)),
        )

        # Evaluate selected quantized ONNX model on the test dataset
        active_onnx = (
            awq_onnx
            if quantize_mode.lower() in ("int4", "awq", "awq_w4a16")
            else sq_onnx
        )
        test_log_probs = self._run_onnx_batched_inference(
            onnx_path=active_onnx,
            dataset=test_dataset,
            batch_size=32 if USE_CUDA else 8,
            include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
        )
        compute_metrics_fn = self._make_compute_metrics(
            test_qrels_dict, test_dataset
        )
        test_metrics = compute_metrics_fn((test_log_probs, None))

        if PartialState().is_main_process:
            test_metrics_path = os.path.join(
                self.output_working_dir, "test_metrics.json"
            )
            with open(test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "quantize_mode": quantize_mode,
                        "num_test_examples": len(test_dataset),
                        "metrics": test_metrics,
                    },
                    f,
                    indent=2,
                )
            print(
                f"\n[Test Phase ({quantize_mode})] Evaluated {len(test_dataset)} examples -> "
                f"Composite_NDCG@{EVAL_TOP_K}={test_metrics[PRIMARY_METRIC_NAME]:.4f}, "
                f"NDCG@{EVAL_TOP_K}={test_metrics[f'ndcg@{EVAL_TOP_K}']:.4f}, "
                f"MRR@{EVAL_TOP_K}={test_metrics[f'mrr@{EVAL_TOP_K}']:.4f}",
                file=sys.stderr,
            )

        return test_metrics

    def run_hpo_phase(self):
        sched = self._compute_training_schedule(
            n_train=len(self.train_dataset),
            n_eval=len(self.eval_dataset),
            is_hpo=True,
        )
        train_batch_size = sched["train_batch_size"]
        eval_batch_size = sched["eval_batch_size"]
        grad_accum = sched["grad_accum"]
        epochs = sched["epochs"]
        max_steps = sched["max_steps"]
        n_trials = sched["n_trials"]
        step_freq = sched["step_freq"]

        compute_metrics_fn = self._make_compute_metrics(
            self.eval_qrels_dict, self.eval_dataset
        )

        best_trained_trainer: Optional[ListwiseSFTTrainer] = None
        best_score_so_far = -float("inf")
        state = PartialState()

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_trained_trainer, best_score_so_far
            if state.is_main_process:
                sampled_params = {
                    "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
                    "lora_dropout": trial.suggest_float(
                        "lora_dropout", 0.0, 0.1, step=0.05
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 5e-5, 5e-4, log=True
                    ),
                    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                }
            else:
                sampled_params = {}

            # Synchronize sampled HPO parameters across DDP ranks
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                obj_list = [sampled_params]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                sampled_params = obj_list[0]

            r_val = int(sampled_params["lora_r"])
            lora_dropout = float(sampled_params["lora_dropout"])
            lr = float(sampled_params["learning_rate"])
            weight_decay = float(sampled_params["weight_decay"])

            trial_lora = LoraConfig(
                r=r_val,
                lora_alpha=r_val * 2,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.lora_config_target_modules,
            )

            trial_output_dir = os.path.join(self.output_dir, f"trial_{trial.number}")
            trial_sft_config = self._build_sft_config(
                output_dir=trial_output_dir,
                learning_rate=lr,
                weight_decay=weight_decay,
                step_freq=step_freq,
                logging_steps=sched["logging_steps"],
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                grad_accum=grad_accum,
                epochs=epochs,
                max_steps=max_steps,
                gradient_checkpointing=sched["use_grad_checkpointing"],
            )

            trainer = ListwiseSFTTrainer(
                model=self.base_model_path,
                args=trial_sft_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                quantization_config=self.bnb_config,
                peft_config=trial_lora,
                preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
                compute_metrics=compute_metrics_fn,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=sched["early_stopping_patience"],
                        early_stopping_threshold=1e-4,
                    ),
                    StepHeartbeatCallback(),
                ],
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
                ranker_emb_dim=RANKER_OUT_FEATURES,
                emb_token_id=self.emb_token_id,
                candidate_token_ids=self.candidate_token_ids,
                tier_loss_weights=IPW_TIER_LOSS_WEIGHTS,
                eval_top_k=EVAL_TOP_K,
                eval_tier_weights=TIER_WEIGHTS,
                distill_alpha=0.15,
                distill_temperature=2.0,
            )

            trainer.train()
            if trainer.state.best_metric is not None:
                target_score = float(trainer.state.best_metric)
            else:
                metrics = trainer.evaluate()
                target_score = float(metrics[EVAL_PRIMARY_METRIC_NAME])

            # Remove heavy HF trial checkpoints from /tmp once best weights are loaded in memory
            if state.is_main_process and os.path.exists(trial_output_dir):
                shutil.rmtree(trial_output_dir, ignore_errors=True)

            trainer.optimizer = None
            trainer.lr_scheduler = None

            if target_score > best_score_so_far:
                best_score_so_far = target_score
                if best_trained_trainer is not None:
                    del best_trained_trainer
                best_trained_trainer = trainer
            else:
                del trainer

            gc.collect()
            if USE_CUDA:
                torch.cuda.empty_cache()

            return target_score

        study = optuna.create_study(
            study_name=f"qwen_reranker_hpo_emb_{INCLUDE_RANKER_EMBEDDINGS}",
            direction="maximize",
        )
        study.optimize(objective, n_trials=n_trials)

        assert study.best_value is not None, "Optuna study.best_value should not be None"
        assert study.best_value >= 0.0, f"Expected best_value >= 0.0, got {study.best_value}"

        if best_trained_trainer is not None and PartialState().is_main_process:
            save_test_path = os.path.join(self.output_dir, "hpo_best_checkpoint")
            self._assert_loaded_model_is_equal(
                trained_trainer=best_trained_trainer,
                save_path=save_test_path,
                test_dataset=self.eval_dataset,
                eval_qrels_dict=self.eval_qrels_dict,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )

            with open(self.best_params_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_value": float(study.best_value),
                        "best_params": study.best_params,
                    },
                    f,
                    indent=2,
                )
            print(
                f"\n[Optuna HPO] Best trial value ({study.best_value:.4f}) and parameters saved to {self.best_params_path}",
                file=sys.stderr,
            )

    def _run_onnx_batched_inference(
        self,
        onnx_path: str,
        dataset: Dataset,
        batch_size: int = 32,
        include_ranker_embeddings: bool = True,
    ) -> np.ndarray:
        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if USE_CUDA and "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(onnx_path, providers=providers)

        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            prompts = [row["prompt"] for row in dataset]
            n_samples = len(prompts)
            if include_ranker_embeddings:
                ranker_embs_np = np.asarray(
                    [row["ranker_embs"] for row in dataset], dtype=np.float32
                )
            else:
                ranker_embs_np = np.zeros(
                    (n_samples, 1 + NUM_CANDIDATES, RANKER_OUT_FEATURES),
                    dtype=np.float32,
                )

            onnx_batches = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                enc_batch = self.tokenizer(
                    prompts[start_idx:end_idx],
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_TOKEN_LENGTH,
                )
                batch_out = sess.run(
                    ["candidate_log_probs"],
                    {
                        "input_ids": enc_batch["input_ids"].astype(np.int64),
                        "attention_mask": enc_batch["attention_mask"].astype(np.int64),
                        "ranker_embs": ranker_embs_np[start_idx:end_idx],
                    },
                )[0]
                onnx_batches.append(batch_out)

            onnx_cand_log_probs = np.concatenate(onnx_batches, axis=0)
            assert onnx_cand_log_probs.shape == (n_samples, NUM_CANDIDATES), (
                f"Expected shape {(n_samples, NUM_CANDIDATES)}, got {onnx_cand_log_probs.shape}"
            )
            return onnx_cand_log_probs
        finally:
            self.tokenizer.padding_side = orig_padding_side

    def _assert_loaded_model_is_equal(
        self,
        trained_trainer: ListwiseSFTTrainer,
        save_path: str,
        test_dataset: Dataset,
        eval_qrels_dict: Dict[str, Dict[str, int]],
        include_ranker_embeddings: bool,
    ):
        verify_ds, verify_qrels = self._subsample_dataset_and_qrels(
            test_dataset, eval_qrels_dict, MAX_ONNX_VERIFY_SAMPLES
        )

        orig_compute_metrics = trained_trainer.compute_metrics
        trained_trainer.compute_metrics = self._make_compute_metrics(
            verify_qrels, verify_ds
        )
        try:
            trained_metrics = trained_trainer.evaluate(eval_dataset=verify_ds)
        finally:
            trained_trainer.compute_metrics = orig_compute_metrics

        onnx_path = export_to_triton_onnx(
            trained_trainer=trained_trainer,
            tokenizer=self.tokenizer,
            model_init_kwargs=self.model_init_kwargs,
            emb_token_id=self.emb_token_id,
            candidate_token_ids=self.candidate_token_ids,
            top_k=NUM_CANDIDATES,
            ranker_emb_dim=RANKER_OUT_FEATURES,
            candidate_labels=CANDIDATE_LABELS,
            base_model_path=self.base_model_path,
            triton_model_dir=save_path,
            include_ranker_embeddings=include_ranker_embeddings,
        )
        assert os.path.exists(onnx_path), f"Missing exported ONNX file: {onnx_path}"
        assert os.path.exists(os.path.join(save_path, "config.pbtxt")), (
            "Missing Triton config.pbtxt"
        )

        batch_size = max(1, trained_trainer.args.per_device_eval_batch_size)
        onnx_cand_log_probs = self._run_onnx_batched_inference(
            onnx_path=onnx_path,
            dataset=verify_ds,
            batch_size=batch_size,
            include_ranker_embeddings=include_ranker_embeddings,
        )

        compute_metrics_fn = self._make_compute_metrics(verify_qrels, verify_ds)
        onnx_metrics = compute_metrics_fn((onnx_cand_log_probs, None))

        np.testing.assert_allclose(
            float(trained_metrics[EVAL_PRIMARY_METRIC_NAME]),
            float(onnx_metrics[PRIMARY_METRIC_NAME]),
            rtol=1e-3,
            atol=1e-3,
        )
        print(
            f"\n[Verification] Exported Triton ONNX model verified in-memory! "
            f"Composite_NDCG@{EVAL_TOP_K}={onnx_metrics[PRIMARY_METRIC_NAME]:.4f}, "
            f"NDCG@{EVAL_TOP_K}={onnx_metrics[f'ndcg@{EVAL_TOP_K}']:.4f}",
            file=sys.stderr,
        )

    def _assert_onnx_session_signature_and_inference(
        self,
        onnx_path: str,
        sample_dataset: Dataset,
        include_ranker_embeddings: bool = True,
        label: str = "ONNX Export",
    ) -> np.ndarray:
        assert os.path.exists(onnx_path), f"Missing ONNX file: {onnx_path}"

        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if USE_CUDA and "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(onnx_path, providers=providers)

        inputs_meta = {inp.name: inp for inp in sess.get_inputs()}
        outputs_meta = {out.name: out for out in sess.get_outputs()}

        assert set(inputs_meta.keys()) == {
            "input_ids",
            "attention_mask",
            "ranker_embs",
        }
        assert set(outputs_meta.keys()) == {"candidate_log_probs"}

        cand_log_probs = self._run_onnx_batched_inference(
            onnx_path=onnx_path,
            dataset=sample_dataset,
            batch_size=len(sample_dataset),
            include_ranker_embeddings=include_ranker_embeddings,
        )
        assert np.all(np.isfinite(cand_log_probs)), (
            f"[{label}] Non-finite log-probs detected"
        )
        prob_sums = np.exp(cand_log_probs).sum(axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones(len(sample_dataset), dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
        )

        top1_labels = [
            CANDIDATE_LABELS[int(np.argmax(row_scores))]
            for row_scores in cand_log_probs
        ]
        print(
            f"[In-Memory Verify: {label}] {len(sample_dataset)} eval examples -> "
            f"top-1 predictions={top1_labels}, log_probs[0]={np.round(cand_log_probs[0], 4)}",
            file=sys.stderr,
        )
        return cand_log_probs

    def _assert_exported_models_work_in_memory(
        self,
        portable_ckpt_dir: str,
        fp32_onnx_path: str,
        sq_onnx_path: str,
        awq_onnx_path: str,
        eval_dataset: Dataset,
        include_ranker_embeddings: bool = True,
        num_examples: int = 2,
    ) -> None:
        sample_ds = eval_dataset.select(
            range(min(num_examples, len(eval_dataset)))
        )
        n_samples = len(sample_ds)

        pt_model, ckpt_tokenizer, ckpt_cfg = load_reusable_checkpoint_in_memory(
            checkpoint_dir=portable_ckpt_dir,
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            device="cpu",
        )
        assert int(ckpt_cfg["top_k"]) == NUM_CANDIDATES

        ckpt_tokenizer.padding_side = "left"
        prompts = [row["prompt"] for row in sample_ds]
        enc_pt = ckpt_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_TOKEN_LENGTH,
        )
        if include_ranker_embeddings:
            embs_pt = torch.tensor(
                [row["ranker_embs"] for row in sample_ds], dtype=torch.float32
            )
        else:
            embs_pt = torch.zeros(
                (n_samples, 1 + NUM_CANDIDATES, RANKER_OUT_FEATURES),
                dtype=torch.float32,
            )

        with torch.no_grad():
            pt_log_probs = (
                pt_model(
                    input_ids=enc_pt["input_ids"],
                    attention_mask=enc_pt["attention_mask"],
                    ranker_embs=embs_pt,
                )
                .cpu()
                .numpy()
            )
        assert pt_log_probs.shape == (n_samples, NUM_CANDIDATES)

        fp32_log_probs = self._assert_onnx_session_signature_and_inference(
            onnx_path=fp32_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="FP32 ONNX",
        )
        sq_log_probs = self._assert_onnx_session_signature_and_inference(
            onnx_path=sq_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="SmoothQuant W8A8 ONNX",
        )
        awq_log_probs = self._assert_onnx_session_signature_and_inference(
            onnx_path=awq_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="AWQ W4A16 ONNX",
        )

        np.testing.assert_allclose(
            pt_log_probs,
            fp32_log_probs,
            rtol=1e-3,
            atol=1e-3,
            err_msg="In-memory Portable Checkpoint log-probs do not match FP32 ONNX!",
        )
        np.testing.assert_allclose(
            sq_log_probs,
            fp32_log_probs,
            rtol=0.25,
            atol=0.25,
            err_msg="SmoothQuant W8A8 log-probs drifted too far from FP32 ONNX!",
        )
        np.testing.assert_allclose(
            awq_log_probs,
            fp32_log_probs,
            rtol=0.35,
            atol=0.35,
            err_msg="AWQ W4A16 log-probs drifted too far from FP32 ONNX!",
        )

    def _build_model_init_kwargs(
        self,
        token: str,
        is_unit_test: bool = False,
    ) -> dict:
        use_cuda = torch.cuda.is_available()
        device_index = PartialState().process_index

        kwargs = {
            "dtype": DTYPE,
            "device_map": {"": device_index} if use_cuda else None,
            "attn_implementation": "sdpa" if use_cuda else "eager",
            "token": token,
            "local_files_only": LOCAL_CACHE_EXISTS,
        }
        if not use_cuda and is_unit_test:
            kwargs["num_hidden_layers"] = 2

        return kwargs

    build_model_init_kwargs = _build_model_init_kwargs

    def _build_sft_config(
        self,
        output_dir: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        step_freq: int = 5 if USE_CUDA else 2,
        logging_steps: int = 1,
        include_ranker_embeddings: bool = INCLUDE_RANKER_EMBEDDINGS,
        train_batch_size: int = 4,
        eval_batch_size: int = 8,
        grad_accum: int = 1,
        epochs: int = 1,
        max_steps: int = -1,
        gradient_checkpointing: bool = False,
    ) -> SFTConfig:
        # Full training (max_steps <= 0) logs & evaluates strictly at the end of every epoch;
        # 2-step CPU unit tests (max_steps > 0) log & evaluate by steps.
        strategy = "epoch" if max_steps <= 0 else "steps"
        cfg = SFTConfig(
            output_dir=output_dir,
            model_init_kwargs=self.model_init_kwargs,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if gradient_checkpointing else None
            ),
            max_length=MAX_SEQ_TOKEN_LENGTH,
            packing=False,
            completion_only_loss=True,
            eos_token="<|im_end|>",
            remove_unused_columns=False,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            dataloader_pin_memory=USE_CUDA,
            dataloader_num_workers=2 if USE_CUDA else 0,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            max_steps=max_steps,
            fp16=(DTYPE == torch.float16),
            bf16=(DTYPE == torch.bfloat16),
            use_cpu=not USE_CUDA,
            logging_first_step=(max_steps > 0),
            logging_strategy=strategy,
            logging_steps=logging_steps,
            eval_strategy=strategy,
            eval_steps=step_freq if strategy == "steps" else None,
            save_strategy=strategy,
            save_steps=step_freq if strategy == "steps" else 500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=EVAL_PRIMARY_METRIC_NAME,
            greater_is_better=True,
            ddp_find_unused_parameters=False if USE_CUDA else None,
            report_to="none",
        )
        if (
            USE_CUDA
            and self.use_qlora_4bit
            and PartialState().num_processes == 1
            and torch.cuda.device_count() > 1
        ):
            cfg._n_gpu = 1
        return cfg

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

    def _preprocess_logits_for_metrics(self, logits, labels):
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

    def _make_compute_metrics(
        self,
        qrels_dict: Dict[str, Dict[str, int]],
        eval_dataset: Optional[Dataset] = None,
    ):
        if eval_dataset is not None and "pos_tier" in eval_dataset.column_names:
            query_tiers = np.asarray(eval_dataset["pos_tier"], dtype=np.int64)
        else:
            query_tiers = np.arange(len(qrels_dict), dtype=np.int64) % 3

        return make_reranker_compute_metrics(
            qrels_dict=qrels_dict,
            query_tiers=query_tiers,
            top_k=EVAL_TOP_K,
            candidate_labels=CANDIDATE_LABELS,
            tier_weights=TIER_WEIGHTS,
        )

    def get_dataset(
        self, is_train: bool = True, is_test: bool = False
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]], int]:
        split_name = "test" if is_test else ("train" if is_train else "val")
        cache_key = (self.DataSize, split_name)
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        ratings_uri_dict = get_train_val_test_liked_uris(data_size=self.DataSize)
        ratings_uri_list = [
            ratings_uri_dict["train_liked"],
            ratings_uri_dict["train_3"],
            ratings_uri_dict["train_disliked"],
        ]
        if not is_train or is_test:
            ratings_uri_list.extend(
                [
                    ratings_uri_dict["val_liked"],
                    ratings_uri_dict["val_3"],
                    ratings_uri_dict["val_disliked"],
                ]
            )
        if is_test:
            for k in ("test_liked", "test_3", "test_disliked"):
                if k in ratings_uri_dict and os.path.exists(ratings_uri_dict[k]):
                    ratings_uri_list.append(ratings_uri_dict[k])

        query_yaml, passages_yaml = load_prompt_yamls(version="2.0.0")

        uh = UserHistory(
            ratings_uri_list=ratings_uri_list,
            max_history=2048,
            n_users=NUM_CATALOG_USERS,
        )

        prompt_formatter = RerankerPromptFormatter(
            user_history=uh,
            movies=self.movies,
            query_yaml=query_yaml,
            passages_yaml=passages_yaml,
            emb_placeholder_token=EMB_PLACEHOLDER_TOKEN,
            max_history=MAX_PROMPT_HISTORY,
            top_k=NUM_CANDIDATES,
            movie_tiers=self.movie_tiers,
            movie_offset=self.movie_offset,
        )

        if is_test:
            if "get_test_parquet" in globals():
                parquet_files = globals()["get_test_parquet"](data_size=self.DataSize)
            elif "get_testing_parquet" in globals():
                parquet_files = globals()["get_testing_parquet"](
                    data_size=self.DataSize
                )
            else:
                train_pq = get_train_parquet(data_size=self.DataSize)
                test_pq = train_pq.replace("training", "testing")
                if not os.path.exists(test_pq):
                    test_pq = train_pq.replace("training", "test")
                val_pq = train_pq.replace("training", "validation")
                parquet_files = (
                    test_pq
                    if os.path.exists(test_pq)
                    else (val_pq if os.path.exists(val_pq) else train_pq)
                )
        elif is_train:
            parquet_files = get_train_parquet(data_size=self.DataSize)
        else:
            if "get_val_parquet" in globals():
                parquet_files = globals()["get_val_parquet"](data_size=self.DataSize)
            elif "get_validation_parquet" in globals():
                parquet_files = globals()["get_validation_parquet"](
                    data_size=self.DataSize
                )
            else:
                train_pq = get_train_parquet(data_size=self.DataSize)
                val_pq = train_pq.replace("training", "validation")
                parquet_files = val_pq if os.path.exists(val_pq) else train_pq

        res = prompt_formatter.load_and_format_parquet_dataset(
            parquet_files=parquet_files
        )
        if len(res) == 3:
            formatted_ds, qrels_dict, ranker_out_features = res
        else:
            formatted_ds, qrels_dict = res
            ranker_out_features = len(formatted_ds[0]["ranker_embs"][0])
        # The prompts in formatted_ds have placeholders "<|vision_pad|>" for each embedding.

        self._dataset_cache[cache_key] = (
            formatted_ds,
            qrels_dict,
            ranker_out_features,
        )
        return formatted_ds, qrels_dict, ranker_out_features