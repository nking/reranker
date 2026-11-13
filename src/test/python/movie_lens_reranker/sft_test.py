import gc
import math
import os
import shutil
import sys
import unittest
from typing import Any, Tuple

HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ON_KAGGLE = "kaggle" in os.getcwd()
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

from movie_lens_reranker.Movies import Movies
from movie_lens_reranker.RecommendedMovies import RecommendedMovies
from movie_lens_reranker.UserHistory import UserHistory
from movie_lens_reranker.util import read_movie_tiers
from movie_lens_reranker.export import (
    export_from_reusable_checkpoint,
    save_reusable_checkpoint, load_reusable_checkpoint_in_memory,
)

CPU_THREADS = str(min(4, max(1, os.cpu_count() or 1)))
if not HAS_NVIDIA_GPU:
    os.environ.setdefault("OMP_NUM_THREADS", CPU_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", CPU_THREADS)
    os.environ.setdefault("NUMBA_NUM_THREADS", CPU_THREADS)
else:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np
import onnxruntime as ort
import optuna
import torch
from accelerate import PartialState
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig
import json
from movie_lens_reranker.export import export_to_triton_onnx
from movie_lens_reranker.ListwiseSFTTrainer import ListwiseSFTTrainer
from movie_lens_reranker.RerankerPromptFormatter import RerankerPromptFormatter
from movie_lens_reranker.reranker_metrics import (
    make_reranker_compute_metrics,
)

if not torch.cuda.is_available():
    torch.set_num_threads(int(CPU_THREADS))
    torch.set_num_interop_threads(1)

from helper import *

from movie_lens_reranker.prompts.prompt_helper import load_prompt_yamls

TOP_K = 20
#MAX_PROMPT_CHARS = 3421 for TOP_K=20 on full train positives dataset
# the Qwen tokenizer compresses on average 3.5 -4 characters per token.
# max / 3.6-ish ~ 960 tokens.
# 21 placholder tokens for the embeddings.  1 completion token
MAX_SEQ_TOKEN_LENGTH = 1024
MAX_PROMPT_HISTORY = 10
USE_CUDA = torch.cuda.is_available()
CANDIDATE_LABELS = RerankerPromptFormatter.candidate_labels(TOP_K)

INCLUDE_RANKER_EMBEDDINGS = True
EMB_PLACEHOLDER_TOKEN = "<|vision_pad|>"
DTYPE = torch.float16 if USE_CUDA else torch.float32

# Metric used for EarlyStoppingCallback, checkpoint selection, and Optuna HPO
PRIMARY_METRIC_NAME = f"composite_ndcg@{TOP_K}"
EVAL_PRIMARY_METRIC_NAME = f"eval_{PRIMARY_METRIC_NAME}"

with open(os.path.join(get_data_dir(), "training_hyperparameters.json"), encoding="utf-8") as f: ranker_metadata_dict = json.load(f)
TIER_WEIGHTS: Tuple[float, float, float] = ranker_metadata_dict["tier_weights"]
IPW_TIER_LOSS_WEIGHTS: Tuple[float, float, float] = (0.333, 0.333, 0.333) # or 0.3959521 ,  1.13960397, 34.62384615
NUM_CATALOG_USERS = ranker_metadata_dict["num_users"] #6040
RANKER_OUT_FEATURES = ranker_metadata_dict["out_dim"]

# Bounds for periodic in-training eval, CPU smoke tests, and HPO
MAX_CPU_TRAIN_SAMPLES = 16
MAX_ACTIVE_EVAL_SAMPLES = 2048 if USE_CUDA else 16
MAX_HPO_TRAIN_SAMPLES = 20_000 if USE_CUDA else 16
MAX_ONNX_VERIFY_SAMPLES = 256 if USE_CUDA else 8

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


class TestSupervisedFineTuning(unittest.TestCase):
    _dataset_cache: Dict[
        Tuple[Any, bool], Tuple[Dataset, Dict[str, Dict[str, int]], int]
    ] = {}

    def setUp(self):
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

        self.lora_config = LoraConfig(
            r=16 if USE_CUDA else 8,
            lora_alpha=32 if USE_CUDA else 16,
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

        # Model Init Kwargs
        self.model_init_kwargs = self.build_model_init_kwargs(
            token=self.token,
            cpu_smoke_test=True,
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
        self.movie_tiers, self.movie_offset, self.num_catalog_movies = read_movie_tiers(os.path.join(data_dir, "movie_tiers.json"))

        self.DataSize = DataSize.TINY3

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
        
        full_train_dataset, full_train_qrels, ranker_out_features = self.get_dataset(
            is_train=True,
        )
        self.assertEqual(RANKER_OUT_FEATURES, ranker_out_features)
        
        full_eval_dataset, full_eval_qrels, _ = self.get_dataset(
            is_train=False,
        )
        
        self.train_dataset, self.train_qrels_dict = self._subsample_dataset_and_qrels(
            full_train_dataset, full_train_qrels, MAX_HPO_TRAIN_SAMPLES
        )
        self.eval_dataset, self.eval_qrels_dict = self._subsample_dataset_and_qrels(
            full_eval_dataset, full_eval_qrels, MAX_ACTIVE_EVAL_SAMPLES
        )
    
    def tearDown(self):
        pass

    def compute_training_schedule(
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

        if USE_CUDA:
            # Kaggle T4 (16GB VRAM per GPU) with Qwen2.5-0.5B 4-bit NF4 and <=768 token sequences
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
                epochs = 4 if is_hpo else 2
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
            # Local / Kaggle CPU fast prototype smoke test
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
            step_freq = max(10, total_steps // (4 * epochs))
            logging_steps = max(1, step_freq // 5)

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
        seed: int=42
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]]]:
        """Deterministically slices a dataset and re-indexes its qrels_dict to q_0 .. q_{K-1}."""
        if len(dataset) <= max_samples:
            return dataset, qrels_dict
        rng = np.random.default_rng(seed)
        chosen_indices = np.sort(
            rng.choice(len(dataset), size=max_samples, replace=False)
        ).tolist()
        sub_ds = dataset.select(chosen_indices)
        sub_qrels = {
            f"q_{new_idx}": qrels_dict[f"q_{orig_idx}"]
            for new_idx, orig_idx in enumerate(chosen_indices)
        }
        return sub_ds, sub_qrels
        
    def testTorchRun(self):
        
        sched = self.compute_training_schedule(
            n_train=len(self.train_dataset),
            n_eval=len(self.eval_dataset),
            is_hpo=False,
        )

        sft_config = self.build_sft_config(
            output_dir=self.output_dir,
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
            peft_config=self.lora_config,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
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
            distill_alpha=0.15,
            distill_temperature=2.0,
        )

        self.print_model_characteristics(
            trainer.model,
            label=f"Qwen2.5-0.5B + QLoRA (include_ranker_embeddings={INCLUDE_RANKER_EMBEDDINGS})",
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Final Evaluation Metrics:", metrics)
        self.assertIn(EVAL_PRIMARY_METRIC_NAME, metrics)
        self.assertIn(f"eval_ndcg@{TOP_K}", metrics)

        if PartialState().is_main_process:
            history_path = os.path.join(self.output_dir, "training_metrics_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(f"[MetricsTracker] Saved training and evaluation history to {history_path}")
        
        if PartialState().is_main_process:
            # Save ~11.5 MB portable checkpoint (Orbax equivalent)
            portable_ckpt_dir = os.path.join(self.output_dir, "portable_checkpoint")
            save_reusable_checkpoint(
                trained_trainer=trainer,
                tokenizer=self.tokenizer,
                checkpoint_dir=portable_ckpt_dir,
                base_model_id=MODEL_ID,
                emb_token_id=self.emb_token_id,
                candidate_token_ids=self.candidate_token_ids,
                top_k=TOP_K,
                ranker_emb_dim=RANKER_OUT_FEATURES,
                candidate_labels=CANDIDATE_LABELS,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )
            
            # Verify FP32 ONNX parity
            save_test_path = os.path.join(self.output_dir, "saved_checkpoint")
            self.assertLoadedModelIsEqual(
                trained_trainer=trainer,
                save_path=save_test_path,
                test_dataset=self.eval_dataset,
                eval_qrels_dict=self.eval_qrels_dict,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )
            fp32_onnx = os.path.join(save_test_path, "1", "model.onnx")
            
            # Export SmoothQuant (W8A8) & AWQ (W4A16) from the portable checkpoint
            calib_prompts = [row["prompt"] for row in self.eval_dataset][:32]
            sq_onnx = export_from_reusable_checkpoint(
                checkpoint_dir=portable_ckpt_dir,
                triton_model_dir=os.path.join(self.output_dir,
                    "triton_smoothquant_w8a8"),
                base_model_path=self.base_model_path,
                model_init_kwargs=self.model_init_kwargs,
                quantize_mode="smoothquant_w8a8",
                calibration_prompts=calib_prompts,
            )
            awq_onnx = export_from_reusable_checkpoint(
                checkpoint_dir=portable_ckpt_dir,
                triton_model_dir=os.path.join(self.output_dir,
                    "triton_awq_w4a16"),
                base_model_path=self.base_model_path,
                model_init_kwargs=self.model_init_kwargs,
                quantize_mode="awq_w4a16",
                calibration_prompts=calib_prompts,
            )
            self.assertTrue(os.path.exists(sq_onnx))
            self.assertTrue(os.path.exists(awq_onnx))
            
            self.assertExportedModelsWorkInMemory(
                portable_ckpt_dir=portable_ckpt_dir,
                fp32_onnx_path=fp32_onnx,
                sq_onnx_path=sq_onnx,
                awq_onnx_path=awq_onnx,
                eval_dataset=self.eval_dataset,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
                num_examples=2,
            )

    def testOptunaHPO(self):
        
        #the datasets have prompts in them and those prompts have placeholders for the embeddings: "<|vision_pad|>"

        sched = self.compute_training_schedule(
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

        compute_metrics_fn = self._make_compute_metrics(self.eval_qrels_dict, self.eval_dataset)

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
                target_modules=self.lora_config.target_modules,
            )

            trial_output_dir = os.path.join(self.output_dir, f"trial_{trial.number}")
            trial_sft_config = self.build_sft_config(
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
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
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
                distill_alpha=0.15,
                distill_temperature=2.0,
            )

            trainer.train()
            if trainer.state.best_metric is not None:
                target_score = float(trainer.state.best_metric)
            else:
                metrics = trainer.evaluate()
                target_score = float(metrics[EVAL_PRIMARY_METRIC_NAME])

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

        self.assertIsNotNone(study.best_value)
        self.assertGreaterEqual(study.best_value, 0.0)

        if best_trained_trainer is not None and PartialState().is_main_process:
            save_test_path = os.path.join(self.output_dir, "hpo_best_checkpoint")
            self.assertLoadedModelIsEqual(
                trained_trainer=best_trained_trainer,
                save_path=save_test_path,
                test_dataset=self.eval_dataset,
                eval_qrels_dict=self.eval_qrels_dict,
                include_ranker_embeddings=INCLUDE_RANKER_EMBEDDINGS,
            )
            
            best_params_path = os.path.join(self.output_dir,
                "best_hyperparameters.json")
            with open(best_params_path, "w", encoding="utf-8") as f:
                json.dump({
                        "best_value": float(study.best_value),
                        "best_params": study.best_params,
                    }, f, indent=2,
                )
            print(
                f"\n[Optuna HPO] Best trial value ({study.best_value:.4f}) and parameters saved to {best_params_path}",
                file=sys.stderr,
            )

    def build_model_init_kwargs(
        self,
        token: str,
        cpu_smoke_test: bool = False,
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
        if not use_cuda and cpu_smoke_test:
            kwargs["num_hidden_layers"] = 2

        return kwargs

    def build_sft_config(
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
            logging_first_step=True,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=step_freq,
            save_strategy="steps",
            save_steps=step_freq,
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
            top_k=TOP_K,
            candidate_labels=CANDIDATE_LABELS,
            tier_weights=TIER_WEIGHTS,
        )

    def assertLoadedModelIsEqual(
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
            top_k=TOP_K,
            ranker_emb_dim=RANKER_OUT_FEATURES,
            candidate_labels=CANDIDATE_LABELS,
            base_model_path=self.base_model_path,
            triton_model_dir=save_path,
            include_ranker_embeddings=include_ranker_embeddings,
        )
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.pbtxt")))

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
            prompts = [row["prompt"] for row in verify_ds]
            n_samples = len(prompts)
            if include_ranker_embeddings:
                ranker_embs_np = np.asarray(
                    [row["ranker_embs"] for row in verify_ds], dtype=np.float32
                )
            else:
                ranker_embs_np = np.zeros(
                    (n_samples, 1 + TOP_K, RANKER_OUT_FEATURES), dtype=np.float32
                )

            batch_size = max(1, trained_trainer.args.per_device_eval_batch_size)
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
            self.assertEqual(onnx_cand_log_probs.shape, (n_samples, TOP_K))
        finally:
            self.tokenizer.padding_side = orig_padding_side

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

        compute_metrics_fn = self._make_compute_metrics(verify_qrels, verify_ds)
        onnx_metrics = compute_metrics_fn((onnx_cand_log_probs, None))

        self.assertAlmostEqual(
            float(trained_metrics[EVAL_PRIMARY_METRIC_NAME]),
            float(onnx_metrics[PRIMARY_METRIC_NAME]),
            places=4,
        )
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
            f"Composite_NDCG@{TOP_K}={onnx_metrics[PRIMARY_METRIC_NAME]:.4f}, "
            f"NDCG@{TOP_K}={onnx_metrics[f'ndcg@{TOP_K}']:.4f}",
            file=sys.stderr,
        )

    def get_dataset(
        self, is_train: bool
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]], int]:
        cache_key = (self.DataSize, bool(is_train))
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        ratings_uri_dict = get_train_val_test_liked_uris(data_size=self.DataSize)
        ratings_uri_list = [
            ratings_uri_dict["train_liked"],
            ratings_uri_dict["train_3"],
            ratings_uri_dict["train_disliked"],
        ]
        if not is_train:
            ratings_uri_list.extend(
                [
                    ratings_uri_dict["val_liked"],
                    ratings_uri_dict["val_3"],
                    ratings_uri_dict["val_disliked"],
                ]
            )

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
            top_k=TOP_K,
            movie_tiers=self.movie_tiers,
            movie_offset=self.movie_offset,
        )

        if is_train:
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
        #the prompts in the formatted_ds have placeholders "<|vision_pad|>" for each embedding.

        self._dataset_cache[cache_key] = (
            formatted_ds,
            qrels_dict,
            ranker_out_features,
        )
        return formatted_ds, qrels_dict, ranker_out_features
    
    def assertOnnxSessionSignatureAndInference(
            self,
            onnx_path: str,
            sample_dataset: Dataset,
            include_ranker_embeddings: bool = True,
            label: str = "ONNX Export",
    ) -> np.ndarray:
        """
        Loads an exported ONNX model into memory, asserts its input/output serving
        signature (equivalent to TF SavedModel signature inspection), runs inference
        on a few eval examples, and verifies valid log-probability outputs.
        """
        self.assertTrue(os.path.exists(onnx_path),
            f"Missing ONNX file: {onnx_path}")
        
        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if USE_CUDA and "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(onnx_path, providers=providers)
        
        # 1. Verify serving signature inputs & outputs
        inputs_meta = {inp.name: inp for inp in sess.get_inputs()}
        outputs_meta = {out.name: out for out in sess.get_outputs()}
        
        self.assertEqual(
            set(inputs_meta.keys()),
            {"input_ids", "attention_mask", "ranker_embs"},
        )
        self.assertEqual(set(outputs_meta.keys()), {"candidate_log_probs"})
        self.assertEqual(inputs_meta["input_ids"].type, "tensor(int64)")
        self.assertEqual(inputs_meta["attention_mask"].type, "tensor(int64)")
        self.assertEqual(inputs_meta["ranker_embs"].type, "tensor(float)")
        self.assertEqual(outputs_meta["candidate_log_probs"].type,
            "tensor(float)")
        
        # 2. Run in-memory inference on the sample eval examples
        n_samples = len(sample_dataset)
        prompts = [row["prompt"] for row in sample_dataset]
        
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                prompts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_TOKEN_LENGTH,
            )
        finally:
            self.tokenizer.padding_side = orig_padding_side
        
        if include_ranker_embeddings:
            ranker_embs_np = np.asarray(
                [row["ranker_embs"] for row in sample_dataset],
                dtype=np.float32
            )
        else:
            ranker_embs_np = np.zeros(
                (n_samples, 1 + TOP_K, RANKER_OUT_FEATURES), dtype=np.float32
            )
        
        cand_log_probs = sess.run(
            ["candidate_log_probs"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "ranker_embs": ranker_embs_np,
            },
        )[0]
        
        # 3. Assert output shape, finiteness, and valid probability distribution
        self.assertEqual(cand_log_probs.shape, (n_samples, TOP_K))
        self.assertTrue(
            np.all(np.isfinite(cand_log_probs)),
            f"[{label}] Non-finite log-probs detected: {cand_log_probs}",
        )
        prob_sums = np.exp(cand_log_probs).sum(axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones(n_samples, dtype=np.float32),
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"[{label}] Candidate probabilities do not sum to 1.0!",
        )
        
        top1_labels = [
            CANDIDATE_LABELS[int(np.argmax(row_scores))]
            for row_scores in cand_log_probs
        ]
        print(
            f"[In-Memory Verify: {label}] {n_samples} eval examples -> "
            f"top-1 predictions={top1_labels}, log_probs[0]={np.round(cand_log_probs[0], 4)}",
            file=sys.stderr,
        )
        return cand_log_probs
    
    def assertExportedModelsWorkInMemory(
            self,
            portable_ckpt_dir: str,
            fp32_onnx_path: str,
            sq_onnx_path: str,
            awq_onnx_path: str,
            eval_dataset: Dataset,
            include_ranker_embeddings: bool = True,
            num_examples: int = 2,
    ) -> None:
        """
        Loads the Portable Checkpoint (PyTorch) and all 3 ONNX exports (FP32, SmoothQuant W8A8,
        AWQ W4A16) into memory and verifies their signatures and inference on `num_examples`
        from `eval_dataset`.
        """
        sample_ds = eval_dataset.select(
            range(min(num_examples, len(eval_dataset))))
        n_samples = len(sample_ds)
        
        # 1. Load Portable Checkpoint in-memory as PyTorch OnnxExportableQwenReranker
        pt_model, ckpt_tokenizer, ckpt_cfg = load_reusable_checkpoint_in_memory(
            checkpoint_dir=portable_ckpt_dir,
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            device="cpu",
        )
        self.assertEqual(int(ckpt_cfg["top_k"]), TOP_K)
        
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
                (n_samples, 1 + TOP_K, RANKER_OUT_FEATURES),
                dtype=torch.float32
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
        self.assertEqual(pt_log_probs.shape, (n_samples, TOP_K))
        
        # 2. Load & verify each ONNX model in memory
        fp32_log_probs = self.assertOnnxSessionSignatureAndInference(
            onnx_path=fp32_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="FP32 ONNX",
        )
        sq_log_probs = self.assertOnnxSessionSignatureAndInference(
            onnx_path=sq_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="SmoothQuant W8A8 ONNX",
        )
        awq_log_probs = self.assertOnnxSessionSignatureAndInference(
            onnx_path=awq_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="AWQ W4A16 ONNX",
        )
        
        # 3. Assert exact parity between in-memory PyTorch checkpoint and FP32 ONNX
        np.testing.assert_allclose(
            pt_log_probs,
            fp32_log_probs,
            rtol=1e-3,
            atol=1e-3,
            err_msg="In-memory Portable Checkpoint log-probs do not match FP32 ONNX!",
        )
        
        # 4. Assert quantized exports stay within bounded tolerance of FP32 log-probs
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

if __name__ == "__main__":
    unittest.main()