import gc
import json
import math
import os
import shutil
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import faulthandler
import io
import threading
import time
import traceback
from contextlib import contextmanager


def _get_faulthandler_stream():
    """Returns a stream with a valid OS file descriptor for `faulthandler`, or None."""
    for stream in (sys.stderr, sys.__stderr__):
        if stream is None:
            continue
        try:
            fd = stream.fileno()
            if isinstance(fd, int) and fd >= 0:
                return stream
        except (io.UnsupportedOperation, AttributeError, ValueError, OSError):
            continue
    return None


@contextmanager
def stage_watchdog(
        stage_name: str,
        timeout_seconds: int,
        heartbeat_interval: int = 60,
):
    """
    Prints a live heartbeat to stdout every `heartbeat_interval` seconds during long
    C++/CPU stages, and dumps all thread tracebacks and exits if `timeout_seconds` is exceeded.
    """
    fh_stream = _get_faulthandler_stream()
    if fh_stream is not None:
        try:
            faulthandler.enable(file=fh_stream, all_threads=True)
        except Exception:
            fh_stream = None
    
    stop_event = threading.Event()
    t0 = time.monotonic()
    
    def _monitor():
        while not stop_event.wait(heartbeat_interval):
            elapsed = time.monotonic() - t0
            if elapsed >= timeout_seconds:
                print(
                    f"\n[WATCHDOG TIMEOUT] Stage '{stage_name}' exceeded {timeout_seconds}s! "
                    "Dumping all thread tracebacks and exiting...",
                    file=sys.stderr,
                    flush=True,
                )
                if fh_stream is not None:
                    try:
                        faulthandler.dump_traceback(file=fh_stream,
                            all_threads=True)
                    except Exception:
                        pass
                else:
                    for tid, frame in sys._current_frames().items():
                        print(f"\n--- Thread {tid} ---", file=sys.stderr)
                        traceback.print_stack(frame, file=sys.stderr)
                    sys.stderr.flush()
                os._exit(1)
            print(
                f"[Watchdog][{stage_name}] Still running ({elapsed:.0f}s / {timeout_seconds}s)...",
                flush=True,
            )
    
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()


HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists(
    "/proc/driver/nvidia")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
os.environ.setdefault("NCCL_TIMEOUT", "3600")

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
        os.path.exists(_LOCAL_BLOBS_PATH) and len(
    os.listdir(_LOCAL_BLOBS_PATH)) > 0
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

sys.path.append(
    os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker"))
sys.path.append(
    os.path.join(os.getcwd(), "src/main/python/movie_lens_reranker"))
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


class TrainPhase(Enum):
    BEST = "BEST"
    FIXED = "FIXED"


class TestCheckPointType(Enum):
    TRAIN = "FromTrainPhase"
    HPO = "FromHPOPhase"


class SplitName(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class PipelineRunner:
    _dataset_cache: Dict[
        Tuple[Any, str, Optional[int], bool],
        Tuple[Dataset, Dict[str, Dict[str, int]], int],
    ] = {}
    
    def __init__(self, is_unit_test: bool = False):
        self.is_unit_test = bool(is_unit_test)
        self.token = get_hf_token()
        
        self.DataSize = DataSize.TINY3 if self.is_unit_test else DataSize.FULL
        
        self.NUM_CANDIDATES = 20  # Number of candidates in prompt & model output head ('A'..'T')
        self.EVAL_TOP_K = 5  # Ranking evaluation cutoff for NDCG@K and MRR@K ('@5')
        self.TOP_K = self.NUM_CANDIDATES
        self.MAX_SEQ_TOKEN_LENGTH = 1024
        self.MAX_PROMPT_HISTORY = 10
        self.USE_CUDA = torch.cuda.is_available()
        self.CANDIDATE_LABELS = RerankerPromptFormatter.candidate_labels(
            self.NUM_CANDIDATES
        )
        
        self.INCLUDE_RANKER_EMBEDDINGS = True
        self.EMB_PLACEHOLDER_TOKEN = "<|vision_pad|>"
        if self.USE_CUDA:
            cc_major, _ = torch.cuda.get_device_capability(0)
            self.DTYPE = torch.bfloat16 if cc_major >= 8 else torch.float16
        else:
            self.DTYPE = torch.float32
        
        # Metric used for EarlyStoppingCallback, checkpoint selection, and Optuna HPO
        self.PRIMARY_METRIC_NAME = f"composite_ndcg@{self.EVAL_TOP_K}"
        self.EVAL_PRIMARY_METRIC_NAME = f"eval_{self.PRIMARY_METRIC_NAME}"
        
        meta_file = os.path.join(
            get_data_dir(), "ranker_training_hyperparameters.json"
        )
        if not os.path.exists(meta_file):
            meta_file = os.path.join(
                get_data_dir(), "training_hyperparameters.json"
            )
        with open(meta_file, encoding="utf-8") as f:
            ranker_metadata_dict = json.load(f)
        self.TIER_WEIGHTS: Tuple[float, float, float] = tuple(
            ranker_metadata_dict["tier_weights"]
        )
        self.IPW_TIER_LOSS_WEIGHTS: Tuple[float, float, float] = (
            0.333,
            0.333,
            0.333,
        )
        self.NUM_CATALOG_USERS = ranker_metadata_dict["num_users"]
        self.RANKER_OUT_FEATURES = ranker_metadata_dict["out_dim"]
        
        # Bounds for periodic in-training eval, CPU is_unit_test tests, and HPO
        self.MAX_CPU_TRAIN_SAMPLES = 16
        self.MAX_ACTIVE_EVAL_SAMPLES = 2048 if self.USE_CUDA else 16
        self.MAX_HPO_TRAIN_SAMPLES = 20_000 if self.USE_CUDA else 16
        self.MAX_ONNX_VERIFY_SAMPLES = 256 if self.USE_CUDA else 8
        self.MAX_CPU_W8A8_TEST_SAMPLES = 500
        
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
        self.use_qlora_4bit = self.USE_CUDA
        
        base_output_dir = os.path.join(get_output_dir(),
            f"{self.DataSize.value}")
        self.output_dir = os.path.join(base_output_dir,
            "prototype_qwen_reranker")
        if PartialState().is_main_process:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir, ignore_errors=True)
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_working_dir = os.path.join(get_output_working_dir(),
            f"{self.DataSize.value}")
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
                bnb_4bit_compute_dtype=self.DTYPE,
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
        recommendations_uri = os.path.join(data_dir,
            "recommended_movies.parquet")
        recommendations_timestamps_uri = os.path.join(
            data_dir, "recommended_movies_timestamps.parquet"
        )
        self.recommendations = RecommendedMovies(
            num_users=self.NUM_CATALOG_USERS,
            movie_rec_file_uri=recommendations_uri,
            movie_rec_ts_file_uri=recommendations_timestamps_uri,
        )
        movies_path = os.path.join(data_dir, "movies.parquet")
        self.movies = Movies(movies_uri=movies_path)
        
        self.movie_tiers, self.movie_offset, self.num_catalog_movies = read_movie_tiers(
            os.path.join(data_dir, "movie_tiers.json")
        )
        
        self.candidate_token_ids: List[int] = [
            self.tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in self.CANDIDATE_LABELS
        ]
        
        emb_token_ids: List[int] = self.tokenizer.encode(
            self.EMB_PLACEHOLDER_TOKEN, add_special_tokens=False
        )
        assert len(emb_token_ids) == 1, (
            f"Expected {self.EMB_PLACEHOLDER_TOKEN} to map to a single token ID, got {emb_token_ids}"
        )
        self.emb_token_id = emb_token_ids[0]
        
        # Output paths (Portable checkpoints & metadata go to working_dir for easy download)
        self.best_params_path = os.path.join(
            self.output_working_dir, "best_hyperparameters.json"
        )
        self.train_params_path = os.path.join(
            self.output_working_dir, "training_hyperparameters.json"
        )
        self.portable_ckpt_dir = os.path.join(
            self.output_working_dir, "portable_train_checkpoint"
        )
        self.portable_best_hpo_ckpt_dir = os.path.join(
            self.output_working_dir, "portable_best_hpo_checkpoint"
        )
        
        self.train_history_path = os.path.join(
            self.output_working_dir, "train_metrics_history.json"
        )
        self.best_hpo_history_path = os.path.join(
            self.output_working_dir, "best_hpo_metrics_history.json"
        )
        self.test_history_path = os.path.join(
            self.output_working_dir, "test_metrics_history.json"
        )
    
    def _build_ort_session_options(self,
            use_cpu_provider: bool) -> ort.SessionOptions:
        sess_opts = ort.SessionOptions()
        if use_cpu_provider:
            num_threads = max(1, os.cpu_count() or 4)
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
        return sess_opts
    
    def _compute_training_schedule(
            self,
            n_train: int,
            n_eval: int,
            is_hpo: bool = False,
    ) -> Dict[str, Any]:
        num_procs = max(1, PartialState().num_processes)
        
        if self.USE_CUDA and not self.is_unit_test:
            max_steps = -1
            if n_train <= 1_000:
                train_batch_size = 4
                eval_batch_size = 8
                grad_accum = 2
                epochs = 2
                evals_per_epoch = 2
                n_trials = 4
                early_stopping_patience = 3
            elif n_train <= 50_000:
                train_batch_size = 4
                eval_batch_size = 8
                grad_accum = 8
                epochs = 3
                evals_per_epoch = 4
                n_trials = 4
                early_stopping_patience = 4
            else:
                train_batch_size = 4
                eval_batch_size = 8
                grad_accum = 8
                epochs = 1
                evals_per_epoch = 8
                n_trials = 4
                early_stopping_patience = 4
        else:
            train_batch_size = 4
            eval_batch_size = 8
            grad_accum = 1
            epochs = 1
            max_steps = 2
            evals_per_epoch = 2
            n_trials = 2
            early_stopping_patience = 2
        
        eff_batch = max(1, train_batch_size * num_procs * grad_accum)
        steps_per_epoch = max(1, math.ceil(n_train / eff_batch))
        total_steps = (
            max_steps if max_steps > 0 else max(1, steps_per_epoch * epochs)
        )
        
        if max_steps > 0 or total_steps <= 20:
            step_freq = 1
            logging_steps = 1
        else:
            step_freq = max(1, steps_per_epoch // evals_per_epoch)
            logging_steps = step_freq
        
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
            "use_grad_checkpointing": self.USE_CUDA and train_batch_size >= 16,
        }
    
    @staticmethod
    def _subsample_dataset_and_qrels(
            dataset: Dataset,
            qrels_dict: Dict[str, Dict[str, int]],
            max_samples: int,
            seed: int = 42,
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]]]:
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
    
    def run_pipeline(self) -> Dict[str, Any]:
        if get_kaggle():
            self._verify_kaggle_t4x2_readiness(
                require_ddp_two_processes=True
            )
        
        state = PartialState()
        is_main = state.is_main_process
        rank = state.process_index
        
        try:
            self.run_hpo_phase()
            
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as exc:
            print(
                f"[PipelineRunner][Rank {rank}] HPO phase failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            for env_key in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "GROUP_RANK",
                    "LOCAL_WORLD_SIZE",
                    "ROLE_RANK",
                    "ROLE_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID",
            ):
                os.environ.pop(env_key, None)
            PartialState._reset_state()
            torch.set_num_threads(os.cpu_count() or 4)
        
        if is_main:
            test_timeout = 2400 if self.is_unit_test else 10800
            try:
                with stage_watchdog(
                        "run_test_phase",
                        timeout_seconds=test_timeout,
                        heartbeat_interval=60,
                ):
                    return self.run_test_phase()
            except Exception as exc:
                print(
                    f"[PipelineRunner][Rank {rank}] Test phase failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
        
        return {}
    
    def run_train_phase(
            self,
            train_phase: TrainPhase = TrainPhase.BEST,
            hyperparams: Optional[Dict[str, Union[float, int]]] = None,
            assert_saved_checkpoint: bool = False,
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
                "lora_r": 16 if self.USE_CUDA else 8,
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
        
        self.load_train_and_val_parquets()
        
        sched = self._compute_training_schedule(
            n_train=len(self.train_dataset),
            n_eval=len(self.eval_dataset),
            is_hpo=False,
        )
        
        sft_config = self._build_sft_config(
            output_dir=self.output_dir,
            learning_rate=float(hyperparams["learning_rate"]),
            weight_decay=float(hyperparams["weight_decay"]),
            include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            train_batch_size=sched["train_batch_size"],
            eval_batch_size=sched["eval_batch_size"],
            grad_accum=sched["grad_accum"],
            epochs=sched["epochs"],
            max_steps=sched["max_steps"],
            step_freq=sched["step_freq"],
            logging_steps=sched["logging_steps"],
            gradient_checkpointing=sched["use_grad_checkpointing"],
        )
        
        if self.USE_CUDA:
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
            include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            ranker_emb_dim=self.RANKER_OUT_FEATURES,
            emb_token_id=self.emb_token_id,
            candidate_token_ids=self.candidate_token_ids,
            tier_loss_weights=self.IPW_TIER_LOSS_WEIGHTS,
            eval_top_k=self.EVAL_TOP_K,
            eval_tier_weights=self.TIER_WEIGHTS,
            distill_alpha=0.15,
            distill_temperature=2.0,
        )
        
        self.print_model_characteristics(
            trainer.model,
            label=f"Qwen2.5-0.5B + QLoRA (include_ranker_embeddings={self.INCLUDE_RANKER_EMBEDDINGS})",
        )
        
        trainer.train()
        
        if PartialState().is_main_process:
            with open(self.train_history_path, "w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(
                f"[MetricsTracker] Saved training and evaluation history to {self.train_history_path}"
            )
            
            with open(self.train_params_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._build_run_metadata_dict(hyperparams, sched),
                    f,
                    indent=2,
                )
        
        metrics = trainer.evaluate()
        print("Final Evaluation Metrics:", metrics)
        assert self.EVAL_PRIMARY_METRIC_NAME in metrics, (
            f"Missing {self.EVAL_PRIMARY_METRIC_NAME} in {metrics.keys()}"
        )
        assert f"eval_ndcg@{self.EVAL_TOP_K}" in metrics, (
            f"Missing eval_ndcg@{self.EVAL_TOP_K} in {metrics.keys()}"
        )
        
        if PartialState().is_main_process:
            save_reusable_checkpoint(
                trained_trainer=trainer,
                tokenizer=self.tokenizer,
                checkpoint_dir=self.portable_ckpt_dir,
                base_model_id=MODEL_ID,
                emb_token_id=self.emb_token_id,
                candidate_token_ids=self.candidate_token_ids,
                top_k=self.NUM_CANDIDATES,
                ranker_emb_dim=self.RANKER_OUT_FEATURES,
                candidate_labels=self.CANDIDATE_LABELS,
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            )
        
        return metrics
    
    def run_test_phase(
            self,
            quantize_mode: str = "smoothquant_w8a8",
            chkpt: TestCheckPointType = TestCheckPointType.HPO,
    ) -> Dict[str, Any]:
        """
        Exports the Portable Checkpoint to FP32, SmoothQuant W8A8, and/or AWQ W4A16 Triton ONNX
        (written to self.output_dir in /tmp so zip downloads remain lightweight),
        verifies in-memory signature and parity, and evaluates on the test split.
        """
        t_phase = time.monotonic()
        ckpt_dir = (
            self.portable_best_hpo_ckpt_dir
            if chkpt == TestCheckPointType.HPO
            else self.portable_ckpt_dir
        )
        if not os.path.exists(ckpt_dir):
            raise ValueError(
                f"Portable ckpt dir {chkpt} not found at {ckpt_dir}"
            )
        
        self.load_train_and_val_parquets()
        
        t_step = time.monotonic()
        print("[Test Phase] Loading test split dataset...", flush=True)
        test_cap = (
            2048 if self.USE_CUDA else 16) if self.is_unit_test else None
        test_dataset, test_qrels_dict, _ = self.get_dataset(
            SplitName.TEST,
            subsample=test_cap is not None,
            max_samples=test_cap,
            training=True,
        )
        print(
            f"[Test Phase] Prepared {len(test_dataset)} test examples ({time.monotonic() - t_step:.2f}s)",
            flush=True,
        )
        
        n_calib = 10 if self.is_unit_test else 24
        calib_rows = [
            self.eval_dataset[i]
            for i in range(min(n_calib, len(self.eval_dataset)))
        ]
        calib_prompts = [row["prompt"] for row in calib_rows]
        calib_embs = (
            [row["ranker_embs"] for row in calib_rows]
            if self.INCLUDE_RANKER_EMBEDDINGS
            else None
        )
        
        mode_lower = quantize_mode.lower()
        want_fp32_only = mode_lower in ("fp32", "none")
        want_awq = mode_lower in ("int4", "awq", "awq_w4a16")
        
        # All exported ONNX bundles are written to self.output_dir (/tmp) to keep working_dir downloads clean
        print(f"[Test Phase] Exporting FP32 ONNX from {ckpt_dir}...",
            flush=True)
        fp32_onnx = export_from_reusable_checkpoint(
            checkpoint_dir=ckpt_dir,
            triton_model_dir=os.path.join(self.output_dir, "triton_fp32"),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="fp32",
            calibration_prompts=calib_prompts,
            calibration_embs=calib_embs
        )
        
        sq_onnx = None
        if not want_fp32_only and (not self.is_unit_test or not want_awq):
            print(
                f"[Test Phase] Exporting SmoothQuant W8A8 ONNX ({n_calib} calib prompts)...",
                flush=True,
            )
            sq_onnx = export_from_reusable_checkpoint(
                checkpoint_dir=ckpt_dir,
                triton_model_dir=os.path.join(
                    self.output_dir, "triton_smoothquant_w8a8"
                ),
                base_model_path=self.base_model_path,
                model_init_kwargs=self.model_init_kwargs,
                quantize_mode="smoothquant_w8a8",
                calibration_prompts=calib_prompts,
                calibration_embs=calib_embs
            )
        
        awq_onnx = None
        if not want_fp32_only and (not self.is_unit_test or want_awq):
            print(
                f"[Test Phase] Exporting AWQ W4A16 ONNX ({n_calib} calib prompts)...",
                flush=True,
            )
            awq_onnx = export_from_reusable_checkpoint(
                checkpoint_dir=ckpt_dir,
                triton_model_dir=os.path.join(
                    self.output_dir, "triton_awq_w4a16"
                ),
                base_model_path=self.base_model_path,
                model_init_kwargs=self.model_init_kwargs,
                quantize_mode="awq_w4a16",
                calibration_prompts=calib_prompts,
                calibration_embs=calib_embs
            )
        
        t_step = time.monotonic()
        print("[Test Phase] Verifying exported ONNX parity in-memory...",
            flush=True)
        self._assert_exported_models_work_in_memory(
            portable_ckpt_dir=ckpt_dir,
            fp32_onnx_path=fp32_onnx,
            sq_onnx_path=sq_onnx,
            awq_onnx_path=awq_onnx,
            eval_dataset=test_dataset,
            include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            num_examples=min(2, len(test_dataset)),
        )
        print(
            f"[Test Phase] Parity check passed ({time.monotonic() - t_step:.2f}s)",
            flush=True,
        )
        
        if want_fp32_only:
            primary_key = "fp32"
        elif want_awq:
            primary_key = "awq_w4a16"
        else:
            primary_key = "smoothquant_w8a8"
        
        all_variants: List[Tuple[str, Optional[str]]] = [
            ("fp32", fp32_onnx),
            ("smoothquant_w8a8", sq_onnx),
            ("awq_w4a16", awq_onnx),
        ]
        if self.is_unit_test:
            eval_variants = [
                (name, path)
                for name, path in all_variants
                if name == primary_key and path is not None
            ]
        else:
            eval_variants = [
                (name, path) for name, path in all_variants if path is not None
            ]
        
        assert any(name == primary_key for name, _ in eval_variants), (
            f"Primary ONNX export missing for quantize_mode={quantize_mode}"
        )
        
        eval_batch_size = 32 if self.USE_CUDA else 8
        comparison_results: Dict[str, Dict[str, Any]] = {}
        
        for variant_name, onnx_path in eval_variants:
            t_var = time.monotonic()
            if variant_name == "smoothquant_w8a8" and not self.is_unit_test:
                var_ds, var_qrels = self._subsample_dataset_and_qrels(
                    test_dataset, test_qrels_dict,
                    self.MAX_CPU_W8A8_TEST_SAMPLES
                )
            else:
                var_ds, var_qrels = test_dataset, test_qrels_dict
            var_compute_metrics_fn = self._make_compute_metrics(var_qrels,
                var_ds)
            print(
                f"[Test Phase] Running batched ONNX inference ({variant_name}) on {len(var_ds)} examples...",
                flush=True,
            )
            variant_log_probs = self._run_onnx_batched_inference(
                onnx_path=onnx_path,
                dataset=var_ds,
                batch_size=eval_batch_size,
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            )
            variant_metrics = var_compute_metrics_fn((variant_log_probs, None))
            elapsed_var = time.monotonic() - t_var
            ms_per_example = (elapsed_var * 1000.0) / max(1, len(var_ds))
            comparison_results[variant_name] = {
                "num_examples": len(var_ds),
                "elapsed_sec": round(elapsed_var, 2),
                "ms_per_example": round(ms_per_example, 3),
                "metrics": variant_metrics,
            }
            print(
                f"[Test Phase ({variant_name})] Completed in {elapsed_var:.2f}s "
                f"({ms_per_example:.2f} ms/query) -> "
                f"Composite_NDCG@{self.EVAL_TOP_K}={variant_metrics[self.PRIMARY_METRIC_NAME]:.4f}, "
                f"NDCG@{self.EVAL_TOP_K}={variant_metrics[f'ndcg@{self.EVAL_TOP_K}']:.4f}, "
                f"MRR@{self.EVAL_TOP_K}={variant_metrics[f'mrr@{self.EVAL_TOP_K}']:.4f}",
                flush=True,
            )
        
        test_metrics = comparison_results[primary_key]["metrics"]
        
        if len(comparison_results) > 1:
            fp32_comp = (
                comparison_results["fp32"]["metrics"][self.PRIMARY_METRIC_NAME]
                if "fp32" in comparison_results
                else None
            )
            print("\n" + "=" * 78, flush=True)
            print(
                f"TEST SPLIT QUANTIZATION COMPARISON (base n={len(test_dataset)} examples):",
                flush=True,
            )
            for variant_name, entry in comparison_results.items():
                m = entry["metrics"]
                comp_val = m[self.PRIMARY_METRIC_NAME]
                delta_str = (
                    f" (Δ vs FP32: {comp_val - fp32_comp:+.4f})"
                    if fp32_comp is not None and variant_name != "fp32"
                    else ""
                )
                print(
                    f"  - {variant_name:<18} | n={entry['num_examples']:<6} | "
                    f"Composite_NDCG@{self.EVAL_TOP_K}={comp_val:.4f}{delta_str} | "
                    f"NDCG@{self.EVAL_TOP_K}={m[f'ndcg@{self.EVAL_TOP_K}']:.4f} | "
                    f"MRR@{self.EVAL_TOP_K}={m[f'mrr@{self.EVAL_TOP_K}']:.4f} | "
                    f"{entry['ms_per_example']:.2f} ms/query",
                    flush=True,
                )
            print("=" * 78 + "\n", flush=True)
        
        if PartialState().is_main_process:
            with open(self.test_history_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "quantize_mode": quantize_mode,
                        "primary_model": primary_key,
                        "num_test_examples": len(test_dataset),
                        "total_test_phase_sec": round(
                            time.monotonic() - t_phase, 2),
                        "metrics": test_metrics,
                        "comparison": comparison_results,
                    },
                    f,
                    indent=2,
                )
        
        return test_metrics
    
    def run_hpo_phase(self, assert_saved_checkpoint: bool = False):
        self.load_train_and_val_parquets()
        
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
                    "weight_decay": trial.suggest_float("weight_decay", 0.0,
                        0.1),
                }
            else:
                sampled_params = {}
            
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
            
            trial_output_dir = os.path.join(self.output_dir,
                f"trial_{trial.number}")
            trial_sft_config = self._build_sft_config(
                output_dir=trial_output_dir,
                learning_rate=lr,
                weight_decay=weight_decay,
                step_freq=step_freq,
                logging_steps=sched["logging_steps"],
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
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
                        early_stopping_patience=sched[
                            "early_stopping_patience"],
                        early_stopping_threshold=1e-4,
                    ),
                    StepHeartbeatCallback(),
                ],
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
                ranker_emb_dim=self.RANKER_OUT_FEATURES,
                emb_token_id=self.emb_token_id,
                candidate_token_ids=self.candidate_token_ids,
                tier_loss_weights=self.IPW_TIER_LOSS_WEIGHTS,
                eval_top_k=self.EVAL_TOP_K,
                eval_tier_weights=self.TIER_WEIGHTS,
                distill_alpha=0.15,
                distill_temperature=2.0,
            )
            
            trainer.train()
            if trainer.state.best_metric is not None:
                target_score = float(trainer.state.best_metric)
            else:
                metrics = trainer.evaluate()
                target_score = float(metrics[self.EVAL_PRIMARY_METRIC_NAME])
            
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
            if self.USE_CUDA:
                torch.cuda.empty_cache()
            
            return target_score
        
        study = optuna.create_study(
            study_name=f"qwen_reranker_hpo_emb_{self.INCLUDE_RANKER_EMBEDDINGS}",
            direction="maximize",
        )
        study.optimize(objective, n_trials=n_trials)
        
        assert study.best_value is not None, "Optuna study.best_value should not be None"
        assert study.best_value >= 0.0, (
            f"Expected best_value >= 0.0, got {study.best_value}"
        )
        
        if best_trained_trainer is not None and PartialState().is_main_process:
            with open(self.best_hpo_history_path, "w", encoding="utf-8") as f:
                json.dump(best_trained_trainer.state.log_history, f, indent=2)
            print(
                f"[MetricsTracker] Saved best HPO training and evaluation history to {self.best_hpo_history_path}"
            )
            
            with open(self.best_params_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_value": float(study.best_value),
                        "best_params": study.best_params,
                        **self._build_run_metadata_dict(study.best_params,
                            sched),
                    },
                    f,
                    indent=2,
                )
            print(
                f"\n[Optuna HPO] Best trial value ({study.best_value:.4f}) and parameters saved to {self.best_params_path}",
                file=sys.stderr, flush=True
            )
            
            save_reusable_checkpoint(
                trained_trainer=best_trained_trainer,
                tokenizer=self.tokenizer,
                checkpoint_dir=self.portable_best_hpo_ckpt_dir,
                base_model_id=MODEL_ID,
                emb_token_id=self.emb_token_id,
                candidate_token_ids=self.candidate_token_ids,
                top_k=self.NUM_CANDIDATES,
                ranker_emb_dim=self.RANKER_OUT_FEATURES,
                candidate_labels=self.CANDIDATE_LABELS,
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            )
        
        if best_trained_trainer is not None:
            del best_trained_trainer
            gc.collect()
            if self.USE_CUDA:
                torch.cuda.empty_cache()
    
    def _run_onnx_batched_inference(
            self,
            onnx_path: str,
            dataset: Dataset,
            batch_size: int = 32,
            include_ranker_embeddings: bool = True,
    ) -> np.ndarray:
        is_w8a8 = "smoothquant" in onnx_path.lower() or "w8a8" in onnx_path.lower()
        available_providers = ort.get_available_providers()
        use_cpu_provider = not (
                self.USE_CUDA
                and not is_w8a8
                and "CUDAExecutionProvider" in available_providers
        )
        providers = (
            ["CPUExecutionProvider"]
            if use_cpu_provider
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        sess_opts = self._build_ort_session_options(
            use_cpu_provider=use_cpu_provider
        )
        
        sess = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers
        )
        
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
                    (n_samples, 1 + self.NUM_CANDIDATES,
                        self.RANKER_OUT_FEATURES),
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
                    max_length=self.MAX_SEQ_TOKEN_LENGTH,
                )
                batch_out = sess.run(
                    ["candidate_log_probs"],
                    {
                        "input_ids": enc_batch["input_ids"].astype(np.int64),
                        "attention_mask": enc_batch["attention_mask"].astype(
                            np.int64),
                        "ranker_embs": ranker_embs_np[start_idx:end_idx],
                    },
                )[0]
                onnx_batches.append(batch_out)
            
            onnx_cand_log_probs = np.concatenate(onnx_batches, axis=0)
            assert onnx_cand_log_probs.shape == (n_samples,
                self.NUM_CANDIDATES), (
                f"Expected shape {(n_samples, self.NUM_CANDIDATES)}, got {onnx_cand_log_probs.shape}"
            )
            return onnx_cand_log_probs
        finally:
            self.tokenizer.padding_side = orig_padding_side
    
    def _assert_onnx_session_signature_and_inference(
            self,
            onnx_path: str,
            sample_dataset: Dataset,
            include_ranker_embeddings: bool = True,
            label: str = "ONNX Export",
    ) -> np.ndarray:
        assert os.path.exists(onnx_path), f"Missing ONNX file: {onnx_path}"
        
        is_w8a8 = "smoothquant" in onnx_path.lower() or "w8a8" in onnx_path.lower()
        available_providers = ort.get_available_providers()
        use_cpu_provider = not (
                self.USE_CUDA
                and not is_w8a8
                and "CUDAExecutionProvider" in available_providers
        )
        providers = (
            ["CPUExecutionProvider"]
            if use_cpu_provider
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        sess_opts = self._build_ort_session_options(
            use_cpu_provider=use_cpu_provider
        )
        
        sess = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers
        )
        
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
            self.CANDIDATE_LABELS[int(np.argmax(row_scores))]
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
            sq_onnx_path: Optional[str] = None,
            awq_onnx_path: Optional[str] = None,
            eval_dataset: Optional[Dataset] = None,
            include_ranker_embeddings: bool = True,
            num_examples: int = 2,
    ) -> None:
        if eval_dataset is None:
            raise ValueError("eval_dataset must not be None")
        
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
        assert int(ckpt_cfg["top_k"]) == self.NUM_CANDIDATES
        
        ckpt_tokenizer.padding_side = "left"
        prompts = [row["prompt"] for row in sample_ds]
        enc_pt = ckpt_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_SEQ_TOKEN_LENGTH,
        )
        if include_ranker_embeddings:
            embs_pt = torch.tensor(
                [row["ranker_embs"] for row in sample_ds], dtype=torch.float32
            )
        else:
            embs_pt = torch.zeros(
                (n_samples, 1 + self.NUM_CANDIDATES, self.RANKER_OUT_FEATURES),
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
        assert pt_log_probs.shape == (n_samples, self.NUM_CANDIDATES)
        
        fp32_log_probs = self._assert_onnx_session_signature_and_inference(
            onnx_path=fp32_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="FP32 ONNX",
        )
        
        # 1. Exact parity between in-memory PyTorch checkpoint and FP32 ONNX
        np.testing.assert_allclose(
            pt_log_probs,
            fp32_log_probs,
            rtol=1e-3,
            atol=1e-3,
            err_msg="In-memory Portable Checkpoint log-probs do not match FP32 ONNX!",
        )
        
        cand_scale = math.log(max(2, self.NUM_CANDIDATES)) / math.log(5.0)
        
        if sq_onnx_path is not None:
            sq_log_probs = self._assert_onnx_session_signature_and_inference(
                onnx_path=sq_onnx_path,
                sample_dataset=sample_ds,
                include_ranker_embeddings=include_ranker_embeddings,
                label="SmoothQuant W8A8 ONNX",
            )
            sq_log_rtol = 0.75 if self.is_unit_test else 0.40
            sq_log_atol = round(
                (1.50 if self.is_unit_test else 0.40) * cand_scale, 4
            )
            np.testing.assert_allclose(
                np.exp(sq_log_probs),
                np.exp(fp32_log_probs),
                rtol=0.35 if self.is_unit_test else 0.25,
                atol=0.55,
                err_msg="SmoothQuant W8A8 probabilities drifted too far from FP32 ONNX!",
            )
            np.testing.assert_allclose(
                sq_log_probs,
                fp32_log_probs,
                rtol=sq_log_rtol,
                atol=sq_log_atol,
                err_msg="SmoothQuant W8A8 log-probs drifted too far from FP32 ONNX!",
            )
        
        if awq_onnx_path is not None:
            awq_log_probs = self._assert_onnx_session_signature_and_inference(
                onnx_path=awq_onnx_path,
                sample_dataset=sample_ds,
                include_ranker_embeddings=include_ranker_embeddings,
                label="AWQ W4A16 ONNX",
            )
            awq_log_atol = round(0.45 * cand_scale, 4)
            np.testing.assert_allclose(
                np.exp(awq_log_probs),
                np.exp(fp32_log_probs),
                rtol=0.35,
                atol=0.55,
                err_msg="AWQ W4A16 probabilities drifted too far from FP32 ONNX!",
            )
            np.testing.assert_allclose(
                awq_log_probs,
                fp32_log_probs,
                rtol=0.45,
                atol=awq_log_atol,
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
            "dtype": self.DTYPE,
            "device_map": {"": device_index} if use_cuda else None,
            "attn_implementation": "sdpa" if use_cuda else "eager",
            "token": token,
            "local_files_only": LOCAL_CACHE_EXISTS,
        }
        if not use_cuda and is_unit_test:
            kwargs["num_hidden_layers"] = 2
        
        return kwargs
    
    def _build_sft_config(
            self,
            output_dir: str,
            learning_rate: float = 2e-4,
            weight_decay: float = 0.0,
            step_freq: int = 5,
            logging_steps: int = 1,
            include_ranker_embeddings: bool = True,
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
            max_length=self.MAX_SEQ_TOKEN_LENGTH,
            packing=False,
            completion_only_loss=True,
            eos_token="<|im_end|>",
            remove_unused_columns=False,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            dataloader_pin_memory=self.USE_CUDA,
            dataloader_num_workers=2 if self.USE_CUDA else 0,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            max_steps=max_steps,
            fp16=(self.DTYPE == torch.float16),
            bf16=(self.DTYPE == torch.bfloat16),
            use_cpu=not self.USE_CUDA,
            logging_first_step=False,
            logging_strategy="steps",
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=step_freq,
            save_strategy="steps",
            save_steps=step_freq,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=self.EVAL_PRIMARY_METRIC_NAME,
            greater_is_better=True,
            ddp_find_unused_parameters=False if self.USE_CUDA else None,
            report_to="none",
        )
        if (
                self.USE_CUDA
                and self.use_qlora_4bit
                and PartialState().num_processes == 1
                and torch.cuda.device_count() > 1
        ):
            cfg._n_gpu = 1
        return cfg
    
    def print_model_characteristics(self, model,
            label: str = "PEFT/QLoRA Model"):
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
        print(f"  - Trainable Parameters: {trainable_params:,}",
            file=sys.stderr)
        print(
            f"  - Trainable Ratio:      {(trainable_params / total_params) * 100:.2f}%",
            file=sys.stderr,
        )
        print(f"  - Est. Memory Footprint: {memory_mb:.2f} MB",
            file=sys.stderr)
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
            top_k=self.EVAL_TOP_K,
            candidate_labels=self.CANDIDATE_LABELS,
            tier_weights=self.TIER_WEIGHTS,
        )
    
    def get_dataset(
            self,
            split: SplitName,
            subsample: bool = False,
            max_samples: Optional[int] = None,
            training: bool = True,
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]], int]:
        effective_max = max_samples if subsample else None
        cache_key = (self.DataSize, split.value, effective_max, training)
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        ratings_uri_dict = get_train_val_test_liked_uris(
            data_size=self.DataSize)
        ratings_uri_list = [
            ratings_uri_dict["train_liked"],
            ratings_uri_dict["train_3"],
            ratings_uri_dict["train_disliked"],
        ]
        if split != SplitName.TRAIN:
            ratings_uri_list.extend(
                [
                    ratings_uri_dict["val_liked"],
                    ratings_uri_dict["val_3"],
                    ratings_uri_dict["val_disliked"],
                ]
            )
        if split == SplitName.TEST:
            for k in ("test_liked", "test_3", "test_disliked"):
                if k in ratings_uri_dict and os.path.exists(
                        ratings_uri_dict[k]):
                    ratings_uri_list.append(ratings_uri_dict[k])
        
        query_yaml, passages_yaml = load_prompt_yamls(version="2.0.0")
        
        uh = UserHistory(
            ratings_uri_list=ratings_uri_list,
            max_history=2048,
            n_users=self.NUM_CATALOG_USERS,
        )
        
        prompt_formatter = RerankerPromptFormatter(
            user_history=uh,
            movies=self.movies,
            query_yaml=query_yaml,
            passages_yaml=passages_yaml,
            emb_placeholder_token=self.EMB_PLACEHOLDER_TOKEN,
            max_history=self.MAX_PROMPT_HISTORY,
            top_k=self.NUM_CANDIDATES,
            movie_tiers=self.movie_tiers,
            movie_offset=self.movie_offset,
        )
        
        if split == SplitName.TEST:
            parquet_files = get_parquet(data_size=self.DataSize,
                split=Split.TEST)
        elif split == SplitName.TRAIN:
            parquet_files = get_parquet(data_size=self.DataSize,
                split=Split.TRAIN)
        else:
            parquet_files = get_parquet(data_size=self.DataSize,
                split=Split.VAL)
        
        formatted_ds, qrels_dict, ranker_out_features = (
            prompt_formatter.load_and_format_parquet_dataset_subset(
                parquet_files=parquet_files,
                max_samples=effective_max,
                training=training,
            )
        )
        
        self._dataset_cache[cache_key] = (
            formatted_ds,
            qrels_dict,
            ranker_out_features,
        )
        return formatted_ds, qrels_dict, ranker_out_features
    
    def _assert_loaded_model_is_equal(
            self,
            trained_trainer: ListwiseSFTTrainer,
            save_to_triton_onnx_path: str,
            test_dataset: Dataset,
            eval_qrels_dict: Dict[str, Dict[str, int]],
            include_ranker_embeddings: bool,
    ):
        verify_ds, verify_qrels = self._subsample_dataset_and_qrels(
            test_dataset, eval_qrels_dict, self.MAX_ONNX_VERIFY_SAMPLES
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
            top_k=self.NUM_CANDIDATES,
            ranker_emb_dim=self.RANKER_OUT_FEATURES,
            candidate_labels=self.CANDIDATE_LABELS,
            base_model_path=self.base_model_path,
            triton_model_dir=save_to_triton_onnx_path,
            include_ranker_embeddings=include_ranker_embeddings,
        )
        assert os.path.exists(onnx_path)
        assert os.path.exists(
            os.path.join(save_to_triton_onnx_path, "config.pbtxt"))
        
        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.USE_CUDA and "CUDAExecutionProvider" in available_providers
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
                    (n_samples, 1 + self.NUM_CANDIDATES,
                        self.RANKER_OUT_FEATURES),
                    dtype=np.float32,
                )
            
            batch_size = max(1,
                trained_trainer.args.per_device_eval_batch_size)
            onnx_batches = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                enc_batch = self.tokenizer(
                    prompts[start_idx:end_idx],
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_TOKEN_LENGTH,
                )
                batch_out = sess.run(
                    ["candidate_log_probs"],
                    {
                        "input_ids": enc_batch["input_ids"].astype(np.int64),
                        "attention_mask": enc_batch["attention_mask"].astype(
                            np.int64),
                        "ranker_embs": ranker_embs_np[start_idx:end_idx],
                    },
                )[0]
                onnx_batches.append(batch_out)
            
            onnx_cand_log_probs = np.concatenate(onnx_batches, axis=0)
            assert onnx_cand_log_probs.shape == (n_samples,
                self.NUM_CANDIDATES)
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
            cand_token_tensor = torch.tensor(self.candidate_token_ids,
                device=device)
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
        
        compute_metrics_fn = self._make_compute_metrics(verify_qrels,
            verify_ds)
        onnx_metrics = compute_metrics_fn((onnx_cand_log_probs, None))
        
        np.testing.assert_almost_equal(
            float(trained_metrics[self.EVAL_PRIMARY_METRIC_NAME]),
            float(onnx_metrics[self.PRIMARY_METRIC_NAME]),
            decimal=4,
        )
        np.testing.assert_almost_equal(
            float(trained_metrics[f"eval_ndcg@{self.EVAL_TOP_K}"]),
            float(onnx_metrics[f"ndcg@{self.EVAL_TOP_K}"]),
            decimal=4,
        )
        np.testing.assert_almost_equal(
            float(trained_metrics[f"eval_mrr@{self.EVAL_TOP_K}"]),
            float(onnx_metrics[f"mrr@{self.EVAL_TOP_K}"]),
            decimal=4,
        )
        print(
            f"\n[Verification] Exported Triton ONNX model verified in-memory! "
            f"Composite_NDCG@{self.EVAL_TOP_K}={onnx_metrics[self.PRIMARY_METRIC_NAME]:.4f}, "
            f"NDCG@{self.EVAL_TOP_K}={onnx_metrics[f'ndcg@{self.EVAL_TOP_K}']:.4f}",
            file=sys.stderr,
        )
    
    def load_train_and_val_parquets(self):
        if hasattr(self, "eval_dataset") and hasattr(self, "train_dataset"):
            return
        
        train_cap = (
            self.MAX_CPU_TRAIN_SAMPLES
            if (self.is_unit_test and not self.USE_CUDA)
            else self.MAX_HPO_TRAIN_SAMPLES
        )
        eval_cap = (
            self.MAX_CPU_TRAIN_SAMPLES
            if (self.is_unit_test and not self.USE_CUDA)
            else self.MAX_ACTIVE_EVAL_SAMPLES
        )
        
        self.train_dataset, self.train_qrels_dict, ranker_out_features = self.get_dataset(
            SplitName.TRAIN,
            subsample=True,
            max_samples=train_cap,
            training=True,
        )
        assert self.RANKER_OUT_FEATURES == ranker_out_features, (
            f"Expected self.RANKER_OUT_FEATURES={self.RANKER_OUT_FEATURES}, got {ranker_out_features}"
        )
        self.eval_dataset, self.eval_qrels_dict, _ = self.get_dataset(
            SplitName.VAL,
            subsample=True,
            max_samples=eval_cap,
            training=True,
        )
    
    def _verify_kaggle_t4x2_readiness(
            self,
            require_ddp_two_processes: bool = False,
            min_free_vram_gb: float = 13.5,
    ) -> None:
        if not self.USE_CUDA:
            raise RuntimeError(
                "CUDA is not available. In Kaggle, select Settings -> Accelerator -> GPU T4 x2."
            )
        
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            gpu_name = torch.cuda.get_device_name(0) if n_gpus == 1 else "None"
            raise RuntimeError(
                f"Expected 2 GPUs (GPU T4 x2), but found {n_gpus} ({gpu_name}). "
                "Please switch Kaggle Accelerator from GPU P100 to GPU T4 x2."
            )
        
        for idx in range(n_gpus):
            dev_name = torch.cuda.get_device_name(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            
            if "T4" not in dev_name.upper() or (major, minor) < (7, 5):
                raise RuntimeError(
                    f"GPU {idx} is '{dev_name}' (compute capability {major}.{minor}); "
                    "expected NVIDIA Tesla T4 (compute capability >= 7.5)."
                )
            if free_gb < min_free_vram_gb:
                raise RuntimeError(
                    f"GPU {idx} ({dev_name}) has only {free_gb:.2f}/{total_gb:.2f} GB free VRAM "
                    f"(minimum required: {min_free_vram_gb:.1f} GB)."
                )
            
            if PartialState().is_main_process:
                print(
                    f"[GPU Check] cuda:{idx} = {dev_name} (CC {major}.{minor}) | "
                    f"Free VRAM: {free_gb:.2f}/{total_gb:.2f} GB",
                    file=sys.stderr,
                )
        
        ort_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in ort_providers:
            raise RuntimeError(
                f"ONNX Runtime is missing 'CUDAExecutionProvider' (found {ort_providers}). "
                "Ensure 'onnxruntime-gpu' is installed so run_test_phase() runs on GPU."
            )
        
        num_procs = PartialState().num_processes
        if num_procs < 2:
            msg = (
                f"2 T4 GPUs detected, but PartialState().num_processes == {num_procs}. "
                "Launch via `accelerate launch --num_processes=2` to use both GPUs in DDP."
            )
            if require_ddp_two_processes:
                raise RuntimeError(msg)
            print(f"[GPU Check WARNING] {msg}", file=sys.stderr)
    
    def _build_run_metadata_dict(
            self,
            hyperparams: Dict[str, Any],
            sched: Dict[str, Any],
    ) -> Dict[str, Any]:
        num_procs = max(1, PartialState().num_processes)
        eff_batch = (
                sched["train_batch_size"] * num_procs * sched["grad_accum"]
        )
        return {
            **hyperparams,
            "num_candidates": self.NUM_CANDIDATES,
            "eval_top_k": self.EVAL_TOP_K,
            "num_catalog_movies": self.num_catalog_movies,
            "num_catalog_users": self.NUM_CATALOG_USERS,
            "focal_gamma": 2.0,
            "listwise_loss_alpha": 0.50,
            "distill_alpha": 0.15,
            "distill_temperature": 2.0,
            "tier_weights": list(self.TIER_WEIGHTS),
            "ipw_tier_loss_weights": list(self.IPW_TIER_LOSS_WEIGHTS),
            "per_device_train_batch_size": sched["train_batch_size"],
            "per_device_eval_batch_size": sched["eval_batch_size"],
            "gradient_accumulation_steps": sched["grad_accum"],
            "num_processes": num_procs,
            "effective_batch_size": eff_batch,
            "num_train_samples": len(self.train_dataset),
            "num_eval_samples": len(self.eval_dataset),
            "epochs": sched["epochs"],
            "max_steps": sched["max_steps"],
            "step_freq": sched["step_freq"],
            "logging_steps": sched["logging_steps"],
            "base_model_id": MODEL_ID,
            "include_ranker_embeddings": self.INCLUDE_RANKER_EMBEDDINGS,
            "ranker_emb_dim": self.RANKER_OUT_FEATURES,
            "max_seq_token_length": self.MAX_SEQ_TOKEN_LENGTH,
            "max_prompt_history": self.MAX_PROMPT_HISTORY,
            "data_size": self.DataSize.name,
            "git_commit_hash": get_git_commit_hash(),
            "package_version": get_package_version()
        }