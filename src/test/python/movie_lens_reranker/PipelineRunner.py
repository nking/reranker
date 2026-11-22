import gc
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faulthandler


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
                        faulthandler.dump_traceback(
                            file=fh_stream, all_threads=True
                        )
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
    "/proc/driver/nvidia"
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
os.environ.setdefault("NCCL_TIMEOUT", "3600")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    os.path.exists(_LOCAL_BLOBS_PATH)
    and len(os.listdir(_LOCAL_BLOBS_PATH)) > 0
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
    os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker")
)
sys.path.append(
    os.path.join(os.getcwd(), "src/main/python/movie_lens_reranker")
)
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
    compute_interaction_and_calibration_diagnostics,
    make_reranker_compute_metrics,
)
from movie_lens_reranker.prompts.prompt_helper import (
    Prompt_Type,
    get_yaml_prompt_path,
    load_prompt_yamls,
)


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

    def __init__(self, is_unit_test: bool = False, seed: int = 42):
        self.is_unit_test = bool(is_unit_test)
        self.token = get_hf_token()
        self.seed = seed

        self.yaml_version = "2.0.0"

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

        # kaggle: /tmp/output, local: /bin
        base_output_dir = os.path.join(
            get_output_dir(), f"{self.DataSize.value}"
        )
        self.output_dir = os.path.join(
            base_output_dir, "prototype_qwen_reranker"
        )
        if PartialState().is_main_process:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir, ignore_errors=True)
            os.makedirs(self.output_dir, exist_ok=True)

        self.output_working_dir = os.path.join(
            get_output_working_dir(), f"{self.DataSize.value}"
        )
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
        recommendations_uri = os.path.join(
            data_dir, "recommended_movies.parquet"
        )
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

        self.movie_tiers, self.movie_offset, self.num_catalog_movies = (
            read_movie_tiers(os.path.join(data_dir, "movie_tiers.json"))
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
        self.test_parity_path = os.path.join(
            self.output_working_dir, "test_parity_comparison.json"
        )
        self.test_logits_dir = os.path.join(self.output_dir, "test_logits")
        self.test_diagnostics_path = os.path.join(
            self.output_working_dir, "test_diagnostics.json"
        )

        self.triton_onnx_path = os.path.join(
            self.output_dir, "triton_onnx_export"
        )

    def _release_cached_datasets(self, keep_eval: bool = False) -> None:
        """Releases cached train/val datasets from host RAM before ONNX export/inference."""
        self._dataset_cache.clear()
        attrs = ["train_dataset", "train_qrels_dict"]
        if not keep_eval:
            attrs.extend(["eval_dataset", "eval_qrels_dict"])
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if self.USE_CUDA:
            torch.cuda.empty_cache()

    def _run_test_phase_in_fresh_subprocess(
            self,
            quantize_mode: str = "smoothquant_w8a8",
            chkpt: TestCheckPointType = TestCheckPointType.HPO,
            fail_on_quant_parity: bool = False,
            checkpoint_dir: Optional[str] = None,
            max_test_samples: Optional[int] = None,
            timeout_seconds: int = 28800,
    ) -> Dict[str, Any]:
        """
        Spawns a fresh Python subprocess to execute `run_test_phase()` with clean host RAM
        and a fresh ONNX Runtime CUDA environment (free of training-phase Python/PyTorch
        heap state). Streams stdout/stderr live and returns the saved primary test metrics.
        """
        payload = {
            "runner_dir": os.path.dirname(os.path.abspath(__file__)),
            "module_name": Path(__file__).stem,
            "is_unit_test": self.is_unit_test,
            "seed": self.seed,
            "yaml_version": self.yaml_version,
            "include_ranker_embeddings": self.INCLUDE_RANKER_EMBEDDINGS,
            "quantize_mode": quantize_mode,
            "chkpt_name": chkpt.name,
            "fail_on_quant_parity": bool(fail_on_quant_parity),
            "checkpoint_dir": checkpoint_dir,
            "max_test_samples": max_test_samples,
        }
        child_code = (
            "import importlib, json, sys\n"
            "cfg = json.loads(sys.argv[1])\n"
            "if cfg['runner_dir'] not in sys.path:\n"
            "    sys.path.insert(0, cfg['runner_dir'])\n"
            "mod = importlib.import_module(cfg['module_name'])\n"
            "runner = mod.PipelineRunner(\n"
            "    is_unit_test=cfg['is_unit_test'],\n"
            "    seed=cfg['seed'],\n"
            ")\n"
            "runner.yaml_version = cfg['yaml_version']\n"
            "runner.INCLUDE_RANKER_EMBEDDINGS = cfg['include_ranker_embeddings']\n"
            "runner.run_test_phase(\n"
            "    quantize_mode=cfg['quantize_mode'],\n"
            "    chkpt=mod.TestCheckPointType[cfg['chkpt_name']],\n"
            "    fail_on_quant_parity=cfg['fail_on_quant_parity'],\n"
            "    checkpoint_dir=cfg['checkpoint_dir'],\n"
            "    max_test_samples=cfg['max_test_samples'],\n"
            ")\n"
        )
        env = os.environ.copy()
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
            env.pop(env_key, None)

        print(
            f"[PipelineRunner] Launching fresh subprocess for run_test_phase "
            f"(chkpt={chkpt.name}, seed={self.seed}, timeout={timeout_seconds}s)...",
            flush=True,
        )
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", child_code, json.dumps(payload)],
            env=env,
        )
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(
                f"Subprocess run_test_phase exceeded {timeout_seconds}s"
            )
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise

        if rc != 0:
            raise RuntimeError(
                f"Subprocess run_test_phase failed with exit code {rc}"
            )

        if os.path.exists(self.test_history_path):
            with open(self.test_history_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return saved.get("metrics", {})
        return {}

    def _build_ort_session_options(
            self, use_cpu_provider: bool
    ) -> ort.SessionOptions:
        sess_opts = ort.SessionOptions()
        # Prevent BFCArena bloat across varying dynamic sequence lengths
        sess_opts.enable_mem_pattern = False
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

        # Qwen's vocab is 152_000
        # so VRAM needed is batch_size * 1024 max seq token length * 152_000 * 4 Bytes
        # and the T4 GPU VRAM is ~ 15GB

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
                eval_batch_size = 4
                grad_accum = 8
                epochs = 3
                evals_per_epoch = 4
                n_trials = 3
                early_stopping_patience = 3
            else:
                train_batch_size = 4
                eval_batch_size = 8
                grad_accum = 8
                epochs = 1
                evals_per_epoch = 8
                n_trials = 3
                early_stopping_patience = 3
        else:
            train_batch_size = 4
            eval_batch_size = 4
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
            "use_grad_checkpointing": self.USE_CUDA,
        }

    @staticmethod
    def _select_stratified_subsample_indices(
            dataset: Dataset,
            max_samples: int,
            seed: int = 42,
    ) -> List[int]:
        total = len(dataset)
        if total <= max_samples:
            return list(range(total))

        rng = np.random.default_rng(seed)
        if "pos_tier" not in dataset.column_names:
            return np.sort(
                rng.choice(total, size=max_samples, replace=False)
            ).tolist()

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
        return chosen_indices.tolist()

    @classmethod
    def _subsample_dataset_and_qrels(
            cls,
            dataset: Dataset,
            qrels_dict: Dict[str, Dict[str, int]],
            max_samples: int,
            seed: int = 42,
    ) -> Tuple[Dataset, Dict[str, Dict[str, int]]]:
        if len(dataset) <= max_samples:
            return dataset, qrels_dict

        chosen_indices = cls._select_stratified_subsample_indices(
            dataset=dataset, max_samples=max_samples, seed=seed
        )
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
            self._release_cached_datasets(keep_eval=False)

        if is_main:
            test_timeout = 2400 if self.is_unit_test else 28800
            try:
                with stage_watchdog(
                        "run_test_phase",
                        timeout_seconds=test_timeout + 60,
                        heartbeat_interval=60,
                ):
                    if self.USE_CUDA:
                        return self._run_test_phase_in_fresh_subprocess(
                            chkpt=TestCheckPointType.HPO,
                            timeout_seconds=test_timeout,
                        )
                    return self.run_test_phase(chkpt=TestCheckPointType.HPO)
            except Exception as exc:
                print(
                    f"[PipelineRunner][Rank {rank}] Test phase failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        return {}

    def run_train_and_test(
            self,
            train_phase: TrainPhase = TrainPhase.BEST,
            hyperparams: Optional[Dict[str, Union[float, int]]] = None,
            assert_saved_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        if get_kaggle():
            self._verify_kaggle_t4x2_readiness(
                require_ddp_two_processes=True
            )

        state = PartialState()
        is_main = state.is_main_process
        rank = state.process_index

        try:
            self.run_train_phase(
                train_phase, hyperparams, assert_saved_checkpoint
            )

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as exc:
            print(
                f"[PipelineRunner][Rank {rank}] Train phase failed: {exc}",
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
            self._release_cached_datasets(keep_eval=False)

        if is_main:
            test_timeout = 2400 if self.is_unit_test else 28800
            try:
                with stage_watchdog(
                        "run_test_phase",
                        timeout_seconds=test_timeout + 60,
                        heartbeat_interval=60,
                ):
                    if self.USE_CUDA:
                        return self._run_test_phase_in_fresh_subprocess(
                            chkpt=TestCheckPointType.TRAIN,
                            timeout_seconds=test_timeout,
                        )
                    return self.run_test_phase(chkpt=TestCheckPointType.TRAIN)
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
                    hyperparams = json.load(f)
                    print("loaded from best_params_path")
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
            print("creating hyperparams dictionary")

        print(f"run_train_phase hyperparams: {hyperparams}")

        for key in ("lora_r", "lora_dropout", "learning_rate", "weight_decay"):
            if key not in hyperparams:
                raise ValueError(f"Cannot find expected {key} in hyperparams")

        # this must occur before any data loading
        if "seed" in hyperparams:
            self.seed = hyperparams["seed"]

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

        callbacks = [StepHeartbeatCallback()]
        early_stopping_callback = None
        if False:  # Keep disabled for fixed-HP runs so Epoch-1 plateau is not aborted early
            if "early_stopping_patience" in hyperparams:
                p = hyperparams["early_stopping_patience"]
                t = hyperparams["early_stopping_threshold"]
            else:
                p = sched["early_stopping_patience"]
                t = 3e-4
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=p,
                early_stopping_threshold=t,
            )
            callbacks.append(early_stopping_callback)

        sft_config = self._build_sft_config(
            output_dir=self.output_dir,
            learning_rate=float(hyperparams["learning_rate"]),
            weight_decay=float(hyperparams["weight_decay"]),
            step_freq=sched["step_freq"],
            logging_steps=sched["logging_steps"],
            include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            train_batch_size=sched["train_batch_size"],
            eval_batch_size=sched["eval_batch_size"],
            grad_accum=sched["grad_accum"],
            epochs=sched["epochs"],
            max_steps=sched["max_steps"],
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
            callbacks=callbacks,
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

        try:
            trainer.train()

            if PartialState().is_main_process:
                with open(self.train_history_path, "w", encoding="utf-8") as f:
                    json.dump(trainer.state.log_history, f, indent=2)
                print(
                    f"[MetricsTracker] Saved training and evaluation history to {self.train_history_path}"
                )

                with open(self.train_params_path, "w", encoding="utf-8") as f:
                    json.dump(
                        self._build_run_metadata_dict(
                            hyperparams, sched, early_stopping_callback
                        ),
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
                    training_hyperparams_path=self.train_params_path,
                    query_yaml_path=get_yaml_prompt_path(
                        prompt_type=Prompt_Type.QUERY,
                        version=self.yaml_version,
                    ),
                    passages_yaml_path=get_yaml_prompt_path(
                        prompt_type=Prompt_Type.PASSAGES,
                        version=self.yaml_version,
                    ),
                )

                if assert_saved_checkpoint:
                    self._assert_loaded_model_is_equal(
                        trained_trainer=trainer,
                        save_to_triton_onnx_path=self.triton_onnx_path,
                        test_dataset=self.eval_dataset,
                        eval_qrels_dict=self.eval_qrels_dict,
                        include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
                        training_hyperparams_path=self.train_params_path,
                    )

            return metrics
        finally:
            trainer.optimizer = None
            trainer.lr_scheduler = None
            del trainer
            gc.collect()
            if self.USE_CUDA:
                torch.cuda.empty_cache()

    def _write_test_history(
            self,
            quantize_mode: str,
            primary_key: str,
            num_test_examples: int,
            t_phase: float,
            comparison_results: Dict[str, Dict[str, Any]],
            parity_comparison: Optional[Dict[str, Any]] = None,
            diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not PartialState().is_main_process:
            return
        primary_metrics = (
            comparison_results[primary_key]["metrics"]
            if primary_key in comparison_results
            else {}
        )
        payload: Dict[str, Any] = {
            "quantize_mode": quantize_mode,
            "primary_model": primary_key,
            "num_test_examples": num_test_examples,
            "total_test_phase_sec": round(time.monotonic() - t_phase, 2),
            "metrics": primary_metrics,
            "comparison": comparison_results,
        }
        if parity_comparison is not None:
            payload["parity_comparison"] = parity_comparison
        if diagnostics is not None:
            payload["diagnostics"] = diagnostics
        with open(self.test_history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _assert_onnx_session_signature(self, onnx_path: str) -> None:
        assert os.path.exists(onnx_path), f"Missing ONNX file: {onnx_path}"
        # Inspect graph I/O metadata on CPU so we never reserve a CUDA arena prematurely
        sess_opts = self._build_ort_session_options(use_cpu_provider=True)
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        try:
            inputs_meta = {inp.name: inp for inp in sess.get_inputs()}
            outputs_meta = {out.name: out for out in sess.get_outputs()}
            assert set(inputs_meta.keys()) == {
                "input_ids",
                "attention_mask",
                "ranker_embs",
            }
            assert set(outputs_meta.keys()) == {"candidate_log_probs"}
        finally:
            del sess
            gc.collect()

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
        use_hpo_early_stopping = True
        hpo_es_patience = sched["early_stopping_patience"]
        hpo_es_threshold = 3e-4

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_trained_trainer, best_score_so_far
            if state.is_main_process:
                # lora_r [8, 16]
                sampled_params = {
                    "lora_r": trial.suggest_categorical("lora_r", [16]),
                    "lora_dropout": trial.suggest_float(
                        "lora_dropout", 0.0, 0.1, step=0.05
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1.5e-4, 5e-4, log=True
                    ),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 0.0, 0.1
                    ),
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

            trial_output_dir = os.path.join(
                self.output_dir, f"trial_{trial.number}"
            )
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

            # Instantiate fresh callbacks per trial so EarlyStoppingCallback state never leaks across trials
            trial_callbacks = [StepHeartbeatCallback()]
            if use_hpo_early_stopping:
                trial_callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=hpo_es_patience,
                        early_stopping_threshold=hpo_es_threshold,
                    )
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
                callbacks=trial_callbacks,
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
        study.optimize(objective, n_trials=n_trials, catch=())

        assert study.best_value is not None, (
            "Optuna study.best_value should not be None"
        )
        assert study.best_value >= 0.0, (
            f"Expected best_value >= 0.0, got {study.best_value}"
        )

        meta_es_cb = (
            EarlyStoppingCallback(
                early_stopping_patience=hpo_es_patience,
                early_stopping_threshold=hpo_es_threshold,
            )
            if use_hpo_early_stopping
            else None
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
                        **self._build_run_metadata_dict(
                            study.best_params, sched, meta_es_cb
                        ),
                    },
                    f,
                    indent=2,
                )
            print(
                f"\n[Optuna HPO] Best trial value ({study.best_value:.4f}) and parameters saved to {self.best_params_path}",
                file=sys.stderr,
                flush=True,
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
                training_hyperparams_path=self.best_params_path,
                query_yaml_path=get_yaml_prompt_path(
                    prompt_type=Prompt_Type.QUERY,
                    version=self.yaml_version,
                ),
                passages_yaml_path=get_yaml_prompt_path(
                    prompt_type=Prompt_Type.PASSAGES,
                    version=self.yaml_version,
                ),
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
            batch_size: int = 4,
            include_ranker_embeddings: bool = True,
            override_ranker_embs: Optional[np.ndarray] = None,
            parallel_gpus: bool = True,
    ) -> np.ndarray:
        """
        Runs batched ONNX inference on `dataset` in its original order (no prompt-length
        reordering):
          - When running on CUDA with >=2 GPUs (e.g. Kaggle T4x2) and `parallel_gpus=True`,
            uses striped sharding across all GPUs and aborts all workers immediately if any
            worker raises an error.
          - Automatically splits a batch in half on `BFCArena` / OOM errors and periodically
            shrinks the CUDA memory arena to prevent fragmentation.
        """
        n_samples = len(dataset)
        if n_samples == 0:
            return np.empty((0, self.NUM_CANDIDATES), dtype=np.float32)

        is_w8a8 = "smoothquant" in onnx_path.lower() or "w8a8" in onnx_path.lower()
        available_providers = ort.get_available_providers()
        use_cpu_provider = not (
            self.USE_CUDA
            and not is_w8a8
            and "CUDAExecutionProvider" in available_providers
        )

        prompts = [row["prompt"] for row in dataset]
        if override_ranker_embs is not None:
            ranker_embs_np = np.asarray(override_ranker_embs, dtype=np.float32)
            assert ranker_embs_np.shape[0] == n_samples, (
                f"override_ranker_embs rows ({ranker_embs_np.shape[0]}) != n_samples ({n_samples})"
            )
        elif include_ranker_embeddings:
            ranker_embs_np = np.asarray(
                [row["ranker_embs"] for row in dataset], dtype=np.float32
            )
        else:
            ranker_embs_np = np.zeros(
                (n_samples, 1 + self.NUM_CANDIDATES, self.RANKER_OUT_FEATURES),
                dtype=np.float32,
            )

        all_indices = np.arange(n_samples, dtype=np.int64)

        n_gpus = (
            torch.cuda.device_count()
            if (not use_cpu_provider and parallel_gpus)
            else 0
        )
        use_multi_gpu = n_gpus >= 2 and n_samples >= max(2, batch_size * 2)

        tok_lock = threading.Lock()
        prog_lock = threading.Lock()
        stop_event = threading.Event()
        done_counter = [0]
        t0 = time.monotonic()
        log_every = max(batch_size * 50, 500)
        next_log = [log_every]

        def _run_shard(
                shard_orig_indices: np.ndarray,
                device_id: Optional[int],
        ) -> Tuple[np.ndarray, np.ndarray]:
            if use_cpu_provider:
                providers: List[Any] = ["CPUExecutionProvider"]
            else:
                cuda_opts: Dict[str, Any] = {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                }
                if device_id is not None:
                    cuda_opts["device_id"] = int(device_id)
                providers = [
                    ("CUDAExecutionProvider", cuda_opts),
                    "CPUExecutionProvider",
                ]

            shrink_opts: Optional[ort.RunOptions] = None
            if not use_cpu_provider and device_id is not None:
                try:
                    ro = ort.RunOptions()
                    ro.add_run_config_entry(
                        "memory.enable_memory_arena_shrinkage",
                        f"gpu:{int(device_id)}",
                    )
                    shrink_opts = ro
                except Exception:
                    shrink_opts = None

            sess_opts = self._build_ort_session_options(
                use_cpu_provider=use_cpu_provider
            )
            sess = ort.InferenceSession(
                onnx_path, sess_options=sess_opts, providers=providers
            )

            def _run_prompts_with_oom_fallback(
                    sub_prompts: List[str],
                    sub_embs: np.ndarray,
                    run_opts: Optional[ort.RunOptions] = None,
            ) -> np.ndarray:
                if stop_event.is_set():
                    return np.empty((0, self.NUM_CANDIDATES), dtype=np.float32)
                with tok_lock:
                    enc_batch = self.tokenizer(
                        sub_prompts,
                        return_tensors="np",
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_SEQ_TOKEN_LENGTH,
                    )
                input_ids = enc_batch["input_ids"].astype(np.int64)
                attention_mask = enc_batch["attention_mask"].astype(np.int64)
                try:
                    return sess.run(
                        ["candidate_log_probs"],
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "ranker_embs": sub_embs,
                        },
                        run_options=run_opts,
                    )[0]
                except Exception as exc:
                    err_str = str(exc)
                    is_oom = (
                        "BFCArena" in err_str
                        or "Failed to allocate memory" in err_str
                        or "out of memory" in err_str.lower()
                    )
                    if len(sub_prompts) <= 1 or not is_oom:
                        raise
                    mid = len(sub_prompts) // 2
                    print(
                        f"  [ONNX OOM Fallback] Retrying batch size {len(sub_prompts)} "
                        f"as {mid} + {len(sub_prompts) - mid} "
                        f"(seq_len={input_ids.shape[1]}, device={device_id})...",
                        file=sys.stderr,
                        flush=True,
                    )
                    retry_opts = (
                        shrink_opts if shrink_opts is not None else run_opts
                    )
                    left = _run_prompts_with_oom_fallback(
                        sub_prompts[:mid],
                        sub_embs[:mid],
                        run_opts=retry_opts,
                    )
                    right = _run_prompts_with_oom_fallback(
                        sub_prompts[mid:],
                        sub_embs[mid:],
                        run_opts=retry_opts,
                    )
                    return np.concatenate([left, right], axis=0)

            try:
                shard_prompts = [prompts[i] for i in shard_orig_indices]
                shard_embs = ranker_embs_np[shard_orig_indices]
                shard_n = len(shard_prompts)
                shard_batches = []

                for batch_idx, start_idx in enumerate(
                    range(0, shard_n, batch_size)
                ):
                    if stop_event.is_set():
                        return (
                            np.empty(0, dtype=np.int64),
                            np.empty(
                                (0, self.NUM_CANDIDATES), dtype=np.float32
                            ),
                        )
                    end_idx = min(start_idx + batch_size, shard_n)
                    run_opts = (
                        shrink_opts
                        if (
                            shrink_opts is not None
                            and batch_idx > 0
                            and batch_idx % 50 == 0
                        )
                        else None
                    )
                    batch_out = _run_prompts_with_oom_fallback(
                        shard_prompts[start_idx:end_idx],
                        shard_embs[start_idx:end_idx],
                        run_opts=run_opts,
                    )
                    if stop_event.is_set():
                        return (
                            np.empty(0, dtype=np.int64),
                            np.empty(
                                (0, self.NUM_CANDIDATES), dtype=np.float32
                            ),
                        )
                    shard_batches.append(batch_out)

                    with prog_lock:
                        done_counter[0] += end_idx - start_idx
                        cur = done_counter[0]
                        if cur >= next_log[0] or cur == n_samples:
                            elapsed = max(1e-6, time.monotonic() - t0)
                            rate = cur / elapsed
                            eta = (n_samples - cur) / max(rate, 1e-6)
                            dev_tag = (
                                f"{n_gpus}xGPU"
                                if use_multi_gpu
                                else ("CPU" if use_cpu_provider else "1xGPU")
                            )
                            print(
                                f"  [ONNX Inference ({dev_tag})] {cur}/{n_samples} "
                                f"({100.0 * cur / n_samples:.1f}%) | "
                                f"{rate:.2f} samples/s | ETA {eta:.0f}s",
                                flush=True,
                            )
                            next_log[0] += log_every

                return shard_orig_indices, np.concatenate(
                    shard_batches, axis=0
                )
            except Exception:
                stop_event.set()
                raise
            finally:
                del sess
                gc.collect()

        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            onnx_cand_log_probs = np.empty(
                (n_samples, self.NUM_CANDIDATES), dtype=np.float32
            )
            if use_multi_gpu:
                shards = [all_indices[dev::n_gpus] for dev in range(n_gpus)]
                print(
                    f"  [ONNX Parallel] Sharding {n_samples} examples across {n_gpus} GPUs "
                    f"(batch_size={batch_size}, shard_sizes={[len(s) for s in shards]})...",
                    flush=True,
                )
                with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                    futures = [
                        pool.submit(_run_shard, shards[dev], dev)
                        for dev in range(n_gpus)
                    ]
                    for fut in futures:
                        idx_arr, out_arr = fut.result()
                        onnx_cand_log_probs[idx_arr] = out_arr
            else:
                idx_arr, out_arr = _run_shard(
                    all_indices, 0 if not use_cpu_provider else None
                )
                onnx_cand_log_probs[idx_arr] = out_arr

            assert onnx_cand_log_probs.shape == (
                n_samples,
                self.NUM_CANDIDATES,
            ), (
                f"Expected shape {(n_samples, self.NUM_CANDIDATES)}, "
                f"got {onnx_cand_log_probs.shape}"
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
        self._assert_onnx_session_signature(onnx_path)

        cand_log_probs = self._run_onnx_batched_inference(
            onnx_path=onnx_path,
            dataset=sample_dataset,
            batch_size=len(sample_dataset),
            include_ranker_embeddings=include_ranker_embeddings,
            parallel_gpus=False,
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
            fail_on_quant_parity: bool = False,
    ) -> Dict[str, Any]:
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
        try:
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
                    [row["ranker_embs"] for row in sample_ds],
                    dtype=torch.float32,
                )
            else:
                embs_pt = torch.zeros(
                    (
                        n_samples,
                        1 + self.NUM_CANDIDATES,
                        self.RANKER_OUT_FEATURES,
                    ),
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
        finally:
            del pt_model
            gc.collect()

        assert pt_log_probs.shape == (n_samples, self.NUM_CANDIDATES)

        fp32_log_probs = self._assert_onnx_session_signature_and_inference(
            onnx_path=fp32_onnx_path,
            sample_dataset=sample_ds,
            include_ranker_embeddings=include_ranker_embeddings,
            label="FP32 ONNX",
        )

        report: Dict[str, Any] = {
            "comparisons": {},
            "parity_errors": [],
            "all_passed": True,
        }

        pt_entry = self._evaluate_parity_arrays(
            comparison_name="pytorch_vs_fp32",
            actual_log_probs=pt_log_probs,
            desired_log_probs=fp32_log_probs,
            prob_rtol=1e-3,
            prob_atol=1e-3,
            log_rtol=1e-3,
            log_atol=1e-3,
            check_first_n=n_samples,
        )
        report["comparisons"]["pytorch_vs_fp32"] = pt_entry
        report["parity_errors"].extend(pt_entry["errors"])

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
            sq_entry = self._evaluate_parity_arrays(
                comparison_name="smoothquant_w8a8_vs_fp32",
                actual_log_probs=sq_log_probs,
                desired_log_probs=fp32_log_probs,
                prob_rtol=0.35 if self.is_unit_test else 0.25,
                prob_atol=0.55,
                log_rtol=sq_log_rtol,
                log_atol=sq_log_atol,
                check_first_n=n_samples,
            )
            report["comparisons"]["smoothquant_w8a8_vs_fp32"] = sq_entry
            report["parity_errors"].extend(sq_entry["errors"])

        if awq_onnx_path is not None:
            awq_log_probs = self._assert_onnx_session_signature_and_inference(
                onnx_path=awq_onnx_path,
                sample_dataset=sample_ds,
                include_ranker_embeddings=include_ranker_embeddings,
                label="AWQ W4A16 ONNX",
            )
            awq_log_atol = round(0.45 * cand_scale, 4)
            awq_entry = self._evaluate_parity_arrays(
                comparison_name="awq_w4a16_vs_fp32",
                actual_log_probs=awq_log_probs,
                desired_log_probs=fp32_log_probs,
                prob_rtol=0.35,
                prob_atol=0.55,
                log_rtol=0.45,
                log_atol=awq_log_atol,
                check_first_n=n_samples,
            )
            report["comparisons"]["awq_w4a16_vs_fp32"] = awq_entry
            report["parity_errors"].extend(awq_entry["errors"])

        report["all_passed"] = len(report["parity_errors"]) == 0
        if fail_on_quant_parity and report["parity_errors"]:
            raise AssertionError(
                "In-memory parity tolerance violations: "
                + json.dumps(report["parity_errors"])
            )
        return report

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

    def print_model_characteristics(
            self, model, label: str = "PEFT/QLoRA Model"
    ):
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
        print(
            f"  - Trainable Parameters: {trainable_params:,}",
            file=sys.stderr,
        )
        print(
            f"  - Trainable Ratio:      {(trainable_params / total_params) * 100:.2f}%",
            file=sys.stderr,
        )
        print(
            f"  - Est. Memory Footprint: {memory_mb:.2f} MB",
            file=sys.stderr,
        )
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
            data_size=self.DataSize
        )
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
                    ratings_uri_dict[k]
                ):
                    ratings_uri_list.append(ratings_uri_dict[k])

        query_yaml, passages_yaml = load_prompt_yamls(self.yaml_version)

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
            parquet_files = get_parquet(
                data_size=self.DataSize, split=Split.TEST
            )
        elif split == SplitName.TRAIN:
            parquet_files = get_parquet(
                data_size=self.DataSize, split=Split.TRAIN
            )
        else:
            parquet_files = get_parquet(
                data_size=self.DataSize, split=Split.VAL
            )

        formatted_ds, qrels_dict, ranker_out_features = (
            prompt_formatter.load_and_format_parquet_dataset_subset(
                parquet_files=parquet_files,
                max_samples=effective_max,
                seed=self.seed,
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
            training_hyperparams_path: Optional[str] = None,
    ):
        verify_ds, verify_qrels = self._subsample_dataset_and_qrels(
            test_dataset,
            eval_qrels_dict,
            self.MAX_ONNX_VERIFY_SAMPLES,
            seed=self.seed,
        )

        orig_compute_metrics = trained_trainer.compute_metrics
        trained_trainer.compute_metrics = self._make_compute_metrics(
            verify_qrels, verify_ds
        )
        try:
            trained_metrics = trained_trainer.evaluate(eval_dataset=verify_ds)
        finally:
            trained_trainer.compute_metrics = orig_compute_metrics

        q_path = Path(
            get_yaml_prompt_path(
                prompt_type=Prompt_Type.QUERY, version=self.yaml_version
            )
        )
        y_path = Path(
            get_yaml_prompt_path(
                prompt_type=Prompt_Type.PASSAGES, version=self.yaml_version
            )
        )
        yaml_paths = [q_path, y_path]

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
            training_hyperparams_path=training_hyperparams_path,
            yaml_paths=yaml_paths,
        )
        assert os.path.exists(onnx_path)
        assert os.path.exists(
            os.path.join(save_to_triton_onnx_path, "config.pbtxt")
        )

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
                    (
                        n_samples,
                        1 + self.NUM_CANDIDATES,
                        self.RANKER_OUT_FEATURES,
                    ),
                    dtype=np.float32,
                )

            batch_size = max(
                1, trained_trainer.args.per_device_eval_batch_size
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
                            np.int64
                        ),
                        "ranker_embs": ranker_embs_np[start_idx:end_idx],
                    },
                )[0]
                onnx_batches.append(batch_out)

            onnx_cand_log_probs = np.concatenate(onnx_batches, axis=0)
            assert onnx_cand_log_probs.shape == (
                n_samples,
                self.NUM_CANDIDATES,
            )
        finally:
            self.tokenizer.padding_side = orig_padding_side
            del sess
            gc.collect()

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
            cand_token_tensor = torch.tensor(
                self.candidate_token_ids, device=device
            )
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

        compute_metrics_fn = self._make_compute_metrics(
            verify_qrels, verify_ds
        )
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

        self.train_dataset, self.train_qrels_dict, ranker_out_features = (
            self.get_dataset(
                SplitName.TRAIN,
                subsample=True,
                max_samples=train_cap,
                training=True,
            )
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
            early_stopping_callback: Optional[EarlyStoppingCallback] = None,
    ) -> Dict[str, Any]:
        num_procs = max(1, PartialState().num_processes)
        eff_batch = (
            sched["train_batch_size"] * num_procs * sched["grad_accum"]
        )
        d = {
            "seed": self.seed,
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
            "package_version": get_package_version(),
        }
        # Merge first, then explicitly set or clear early-stopping keys
        merged = hyperparams | d
        if early_stopping_callback is not None:
            merged["early_stopping_threshold"] = (
                early_stopping_callback.early_stopping_threshold
            )
            merged["early_stopping_patience"] = (
                early_stopping_callback.early_stopping_patience
            )
        else:
            merged.pop("early_stopping_threshold", None)
            merged.pop("early_stopping_patience", None)
        return merged

    def _evaluate_parity_arrays(
            self,
            comparison_name: str,
            actual_log_probs: np.ndarray,
            desired_log_probs: np.ndarray,
            prob_rtol: float,
            prob_atol: float,
            log_rtol: float,
            log_atol: float,
            check_first_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Computes structured quantization-drift metrics across all aligned rows and
        evaluates probability-space and log-prob-space tolerances without raising.
        Emits single-line `[PARITY_SUMMARY]` and `[PARITY_JSON]` log lines for easy parsing.
        """
        n_total = int(actual_log_probs.shape[0])
        n_check = (
            min(int(check_first_n), n_total)
            if check_first_n is not None
            else n_total
        )

        p_actual = np.exp(actual_log_probs)
        p_desired = np.exp(desired_log_probs)
        prob_abs_all = np.abs(p_actual - p_desired)
        log_abs_all = np.abs(actual_log_probs - desired_log_probs)
        kl_per_example = np.sum(
            p_desired * (desired_log_probs - actual_log_probs), axis=-1
        )
        top1_agree = float(
            np.mean(
                np.argmax(actual_log_probs, axis=-1)
                == np.argmax(desired_log_probs, axis=-1)
            )
        )

        k_top = min(self.EVAL_TOP_K, int(actual_log_probs.shape[-1]))
        actual_topk = np.argsort(-actual_log_probs, axis=-1)[:, :k_top]
        desired_topk = np.argsort(-desired_log_probs, axis=-1)[:, :k_top]
        topk_overlap = float(
            np.mean(
                [
                    len(set(a_row.tolist()) & set(d_row.tolist()))
                    / float(k_top)
                    for a_row, d_row in zip(actual_topk, desired_topk)
                ]
            )
        )

        def _check_space(
                actual_slice: np.ndarray,
                desired_slice: np.ndarray,
                rtol: float,
                atol: float,
        ) -> Dict[str, Any]:
            abs_diff = np.abs(actual_slice - desired_slice)
            tol = atol + rtol * np.abs(desired_slice)
            mismatch_mask = abs_diff > tol
            denom = np.where(
                np.abs(desired_slice) > 1e-12, np.abs(desired_slice), 1e-12
            )
            rel_diff = abs_diff / denom
            mismatched = int(np.sum(mismatch_mask))
            total_el = int(actual_slice.size)
            max_abs = float(
                np.max(abs_diff[mismatch_mask])
                if mismatched > 0
                else np.max(abs_diff)
            )
            max_rel = float(
                np.max(rel_diff[mismatch_mask])
                if mismatched > 0
                else np.max(rel_diff)
            )
            return {
                "passed": bool(mismatched == 0),
                "rtol": float(rtol),
                "atol": float(atol),
                "num_examples_checked": int(actual_slice.shape[0]),
                "mismatched_elements": mismatched,
                "total_elements": total_el,
                "mismatch_pct": round(
                    100.0 * mismatched / max(1, total_el), 2
                ),
                "max_abs_diff": round(max_abs, 6),
                "max_rel_diff": round(max_rel, 6),
            }

        prob_check = _check_space(
            p_actual[:n_check], p_desired[:n_check], prob_rtol, prob_atol
        )
        log_check = _check_space(
            actual_log_probs[:n_check],
            desired_log_probs[:n_check],
            log_rtol,
            log_atol,
        )

        errors: List[Dict[str, Any]] = []
        if not prob_check["passed"]:
            errors.append(
                {
                    "comparison": comparison_name,
                    "metric_space": "prob",
                    **prob_check,
                }
            )
        if not log_check["passed"]:
            errors.append(
                {
                    "comparison": comparison_name,
                    "metric_space": "log_prob",
                    **log_check,
                }
            )

        passed = bool(prob_check["passed"] and log_check["passed"])
        entry: Dict[str, Any] = {
            "comparison": comparison_name,
            "status": "PASS" if passed else "WARN",
            "passed": passed,
            "num_shared_examples": n_total,
            "top1_agreement": round(top1_agree, 4),
            f"top{k_top}_overlap": round(topk_overlap, 4),
            "mean_prob_abs_diff": round(float(np.mean(prob_abs_all)), 6),
            "max_prob_abs_diff": round(float(np.max(prob_abs_all)), 6),
            "p95_prob_abs_diff": round(
                float(np.percentile(prob_abs_all, 95)), 6
            ),
            "mean_log_prob_abs_diff": round(float(np.mean(log_abs_all)), 6),
            "max_log_prob_abs_diff": round(float(np.max(log_abs_all)), 6),
            "p95_log_prob_abs_diff": round(
                float(np.percentile(log_abs_all, 95)), 6
            ),
            "mean_kl_divergence": round(float(np.mean(kl_per_example)), 6),
            "prob_check": prob_check,
            "log_prob_check": log_check,
            "errors": errors,
        }

        print(
            f"[PARITY_SUMMARY] comparison={comparison_name} status={entry['status']} "
            f"n={n_total} top1_agree={entry['top1_agreement']:.4f} "
            f"top{k_top}_overlap={entry[f'top{k_top}_overlap']:.4f} "
            f"mean_kl={entry['mean_kl_divergence']:.4f} "
            f"prob_ok={prob_check['passed']} prob_mismatch={prob_check['mismatched_elements']}/{prob_check['total_elements']}({prob_check['mismatch_pct']:.2f}%) "
            f"log_ok={log_check['passed']} log_mismatch={log_check['mismatched_elements']}/{log_check['total_elements']}({log_check['mismatch_pct']:.2f}%) "
            f"log_max_abs={log_check['max_abs_diff']:.4f}",
            flush=True,
        )
        print(
            f"[PARITY_JSON] {json.dumps(entry, sort_keys=True)}",
            flush=True,
        )
        return entry

    def run_test_phase(
            self,
            quantize_mode: str = "smoothquant_w8a8",
            chkpt: TestCheckPointType = TestCheckPointType.HPO,
            fail_on_quant_parity: bool = False,
            checkpoint_dir: Optional[str] = None,
            max_test_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Exports the Portable Checkpoint to FP32, SmoothQuant W8A8, and/or AWQ W4A16 Triton ONNX
        (written to self.output_dir in /tmp so zip downloads remain lightweight),
        evaluates on the test split, computes 3A/3B diagnostics, and runs structured parity
        comparisons.

        By default (`fail_on_quant_parity=False`), parity tolerance violations are logged in
        structured `[PARITY_SUMMARY]` / `[PARITY_JSON]` lines and saved to
        `test_parity_comparison.json` without aborting the run.
        """
        t_phase = time.monotonic()

        # Free any cached training split before loading test data / running CPU ONNX exports
        self._release_cached_datasets(keep_eval=True)

        ckpt_dir = checkpoint_dir or (
            self.portable_best_hpo_ckpt_dir
            if chkpt == TestCheckPointType.HPO
            else self.portable_ckpt_dir
        )
        if not os.path.exists(ckpt_dir):
            raise ValueError(
                f"Portable ckpt dir {chkpt} not found at {ckpt_dir}"
            )

        t_step = time.monotonic()
        print("[Test Phase] Loading test split dataset...", flush=True)
        test_cap = (
            max_test_samples
            if max_test_samples is not None
            else (
                (2048 if self.USE_CUDA else 16)
                if self.is_unit_test
                else None
            )
        )

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
        mode_lower = quantize_mode.lower()
        want_fp32_only = mode_lower in ("fp32", "none")
        want_awq = mode_lower in ("int4", "awq", "awq_w4a16")
        need_sq_export = not want_fp32_only and (
            not self.is_unit_test or not want_awq
        )
        need_awq_export = not want_fp32_only and (
            not self.is_unit_test or want_awq
        )

        # Only fetch lightweight calibration prompts when SmoothQuant W8A8 is exported
        if need_sq_export:
            self.create_calib_rows_embs(n_calib)

        # Drop cached val/test entries from `_dataset_cache` so host RAM is free for CPU ONNX export
        self._release_cached_datasets(keep_eval=False)

        # All exported ONNX bundles are written to self.output_dir (/tmp) to keep working_dir downloads clean
        print(
            f"[Test Phase] Exporting FP32 ONNX from {ckpt_dir}...", flush=True
        )
        fp32_onnx = self.make_fp32_export(ckpt_dir, self.output_dir)

        sq_onnx = None
        if need_sq_export:
            print(
                f"[Test Phase] Exporting SmoothQuant W8A8 ONNX ({n_calib} calib prompts)...",
                flush=True,
            )
            sq_onnx = self.make_smoothquant_w8a8_export(
                ckpt_dir, self.output_dir
            )

        awq_onnx = None
        if need_awq_export:
            print(
                "[Test Phase] Exporting AWQ W4A16 ONNX...",
                flush=True,
            )
            awq_onnx = self.make_awq_w4a16_export(ckpt_dir, self.output_dir)

        parity_report: Optional[Dict[str, Any]] = None
        if self.is_unit_test:
            t_step = time.monotonic()
            print(
                "[Test Phase] Verifying exported ONNX parity in-memory...",
                flush=True,
            )
            parity_report = self._assert_exported_models_work_in_memory(
                portable_ckpt_dir=ckpt_dir,
                fp32_onnx_path=fp32_onnx,
                sq_onnx_path=sq_onnx,
                awq_onnx_path=awq_onnx,
                eval_dataset=test_dataset,
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
                num_examples=min(2, len(test_dataset)),
                fail_on_quant_parity=bool(fail_on_quant_parity),
            )
            print(
                f"[Test Phase] In-memory parity comparison finished ({time.monotonic() - t_step:.2f}s) "
                f"(all_passed={parity_report.get('all_passed', True)})",
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

        os.makedirs(self.test_logits_dir, exist_ok=True)
        # Keep GPU ONNX batch_size=4 so eager (B, 14, S, S) attention stays safely within T4 14.56 GB VRAM
        gpu_eval_batch_size = 4 if self.USE_CUDA else 8
        comparison_results: Dict[str, Dict[str, Any]] = {}
        diagnostics_by_variant: Dict[str, Any] = {}
        saved_logits_meta: Dict[str, Dict[str, str]] = {}
        ablation_target_variant = (
            "fp32"
            if any(name == "fp32" for name, _ in eval_variants)
            else primary_key
        )

        for variant_name, onnx_path in eval_variants:
            t_var = time.monotonic()
            self._assert_onnx_session_signature(onnx_path)

            if variant_name == "smoothquant_w8a8" and not self.is_unit_test:
                var_indices = self._select_stratified_subsample_indices(
                    test_dataset,
                    self.MAX_CPU_W8A8_TEST_SAMPLES,
                    seed=self.seed,
                )
                var_ds = test_dataset.select(var_indices)
                var_qrels = {
                    f"q_{new_idx}": test_qrels_dict[f"q_{orig_idx}"]
                    for new_idx, orig_idx in enumerate(var_indices)
                }
            else:
                var_indices = list(range(len(test_dataset)))
                var_ds, var_qrels = test_dataset, test_qrels_dict

            var_batch_size = (
                8
                if (variant_name == "smoothquant_w8a8" or not self.USE_CUDA)
                else gpu_eval_batch_size
            )
            var_compute_metrics_fn = self._make_compute_metrics(
                var_qrels, var_ds
            )
            print(
                f"[Test Phase] Running batched ONNX inference ({variant_name}) on {len(var_ds)} examples...",
                flush=True,
            )
            variant_log_probs = self._run_onnx_batched_inference(
                onnx_path=onnx_path,
                dataset=var_ds,
                batch_size=var_batch_size,
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
            )

            assert np.all(np.isfinite(variant_log_probs)), (
                f"[{variant_name}] Non-finite log-probs detected"
            )
            prob_sums = np.exp(variant_log_probs).sum(axis=-1)
            np.testing.assert_allclose(
                prob_sums,
                np.ones(len(var_ds), dtype=np.float32),
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"[{variant_name}] Probabilities do not sum to 1.0",
            )

            logits_path = os.path.join(
                self.test_logits_dir, f"{variant_name}_log_probs.npy"
            )
            indices_path = os.path.join(
                self.test_logits_dir, f"{variant_name}_indices.npy"
            )
            np.save(logits_path, variant_log_probs.astype(np.float32))
            np.save(indices_path, np.asarray(var_indices, dtype=np.int64))
            saved_logits_meta[variant_name] = {
                "logits_path": logits_path,
                "indices_path": indices_path,
            }

            variant_metrics = var_compute_metrics_fn((variant_log_probs, None))
            elapsed_var = time.monotonic() - t_var
            ms_per_example = (elapsed_var * 1000.0) / max(1, len(var_ds))
            variant_entry = {
                "variant": variant_name,
                "num_examples": len(var_ds),
                "elapsed_sec": round(elapsed_var, 2),
                "ms_per_example": round(ms_per_example, 3),
                "logits_path": logits_path,
                "metrics": variant_metrics,
            }
            comparison_results[variant_name] = variant_entry

            print(
                f"[Test Phase ({variant_name})] Completed in {elapsed_var:.2f}s "
                f"({ms_per_example:.2f} ms/query) -> "
                f"Composite_NDCG@{self.EVAL_TOP_K}={variant_metrics[self.PRIMARY_METRIC_NAME]:.4f}, "
                f"NDCG@{self.EVAL_TOP_K}={variant_metrics[f'ndcg@{self.EVAL_TOP_K}']:.4f}, "
                f"MRR@{self.EVAL_TOP_K}={variant_metrics[f'mrr@{self.EVAL_TOP_K}']:.4f}",
                flush=True,
            )

            # Persist per-variant metrics immediately BEFORE running diagnostics
            if PartialState().is_main_process:
                per_variant_path = os.path.join(
                    self.output_working_dir,
                    f"test_metrics_{variant_name}.json",
                )
                with open(per_variant_path, "w", encoding="utf-8") as f:
                    json.dump(variant_entry, f, indent=2)
                print(
                    f"[Test Phase] Saved individual metrics -> {per_variant_path}",
                    flush=True,
                )
                self._write_test_history(
                    quantize_mode=quantize_mode,
                    primary_key=primary_key,
                    num_test_examples=len(test_dataset),
                    t_phase=t_phase,
                    comparison_results=comparison_results,
                    parity_comparison=parity_report,
                    diagnostics=diagnostics_by_variant,
                )

            # 3B: Calibration, Stage-2 Lift, and Causal-Mask Slot Bias diagnostics
            stage2_scores = (
                np.asarray(var_ds["candidate_scores"], dtype=np.float64)
                if "candidate_scores" in var_ds.column_names
                else None
            )
            var_tiers = (
                np.asarray(var_ds["pos_tier"], dtype=np.int64)
                if "pos_tier" in var_ds.column_names
                else np.arange(len(var_ds), dtype=np.int64) % 3
            )
            var_diag = compute_interaction_and_calibration_diagnostics(
                candidate_scores=variant_log_probs,
                qrels_dict=var_qrels,
                query_tiers=var_tiers,
                top_k=self.EVAL_TOP_K,
                candidate_labels=self.CANDIDATE_LABELS,
                tier_weights=self.TIER_WEIGHTS,
                stage2_candidate_scores=stage2_scores,
            )

            # 3A: Counterfactual Embedding Ablation (run once on fp32, or primary_key in unit tests)
            if variant_name == ablation_target_variant:
                ablation_cap = 4 if self.is_unit_test else 256
                var_diag["embedding_ablation"] = (
                    self.run_embedding_ablation_diagnostic(
                        onnx_path=onnx_path,
                        dataset=var_ds,
                        qrels_dict=var_qrels,
                        base_log_probs=variant_log_probs,
                        max_samples=min(ablation_cap, len(var_ds)),
                        batch_size=var_batch_size,
                    )
                )

            del variant_log_probs
            gc.collect()

            diagnostics_by_variant[variant_name] = var_diag
            cal = var_diag.get("calibration", {})
            slot = var_diag.get("causal_slot_bias", {})
            lift = var_diag.get("lift_vs_stage2", {})
            print(
                f"[DIAG_SUMMARY] variant={variant_name} "
                f"ece={cal.get('ece_10bin')} T*={cal.get('optimal_temperature')} "
                f"post_T_ece={cal.get('post_temp_ece_10bin')} "
                f"slot_ent_ratio={slot.get('slot_pred_entropy_ratio')} "
                f"half_acc_delta={slot.get('second_minus_first_half_top1_acc')} "
                f"lift_comp_ndcg={lift.get(self.PRIMARY_METRIC_NAME)}",
                flush=True,
            )

            if PartialState().is_main_process:
                with open(
                    self.test_diagnostics_path, "w", encoding="utf-8"
                ) as f:
                    json.dump(diagnostics_by_variant, f, indent=2)
                self._write_test_history(
                    quantize_mode=quantize_mode,
                    primary_key=primary_key,
                    num_test_examples=len(test_dataset),
                    t_phase=t_phase,
                    comparison_results=comparison_results,
                    parity_comparison=parity_report,
                    diagnostics=diagnostics_by_variant,
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

        if not self.is_unit_test:
            t_step = time.monotonic()
            print(
                f"[Test Phase] Running postponed quantization parity comparison from {self.test_logits_dir}...",
                flush=True,
            )
            parity_report = self._compare_saved_logits_parity(
                portable_ckpt_dir=ckpt_dir,
                saved_logits_meta=saved_logits_meta,
                eval_dataset=test_dataset,
                include_ranker_embeddings=self.INCLUDE_RANKER_EMBEDDINGS,
                num_pt_examples=min(2, len(test_dataset)),
            )

        if parity_report is not None and PartialState().is_main_process:
            with open(self.test_parity_path, "w", encoding="utf-8") as f:
                json.dump(parity_report, f, indent=2)
            self._write_test_history(
                quantize_mode=quantize_mode,
                primary_key=primary_key,
                num_test_examples=len(test_dataset),
                t_phase=t_phase,
                comparison_results=comparison_results,
                parity_comparison=parity_report,
                diagnostics=diagnostics_by_variant,
            )
            print(
                f"[Test Phase] Saved structured parity report -> {self.test_parity_path} "
                f"(all_passed={parity_report.get('all_passed', True)}, "
                f"num_errors={len(parity_report.get('parity_errors', []))})",
                flush=True,
            )

        if (
            fail_on_quant_parity
            and parity_report is not None
            and parity_report.get("parity_errors")
        ):
            raise AssertionError(
                "Parity tolerance violations recorded: "
                + json.dumps(parity_report["parity_errors"])
            )

        return test_metrics

    def create_calib_rows_embs(self, n_calib: int):
        has_prompts = (
            getattr(self, "calib_prompts", None) is not None
            and len(self.calib_prompts) >= n_calib
        )
        has_embs = not self.INCLUDE_RANKER_EMBEDDINGS or (
            getattr(self, "calib_embs", None) is not None
            and len(self.calib_embs) >= n_calib
        )
        if has_prompts and has_embs:
            return

        # Reuse eval_dataset if already loaded in memory (e.g. right after HPO);
        # otherwise fetch only a tiny validation slice (~2x n_calib) instead of 20k train + 2k val.
        existing_eval = getattr(self, "eval_dataset", None)
        if existing_eval is not None and len(existing_eval) >= n_calib:
            calib_ds = existing_eval
        else:
            calib_fetch = max(n_calib * 2, 32)
            calib_ds, _, _ = self.get_dataset(
                SplitName.VAL,
                subsample=True,
                max_samples=calib_fetch,
                training=True,
            )

        calib_rows = [
            calib_ds[i] for i in range(min(n_calib, len(calib_ds)))
        ]
        self.calib_prompts = [row["prompt"] for row in calib_rows]
        self.calib_embs = (
            [row["ranker_embs"] for row in calib_rows]
            if self.INCLUDE_RANKER_EMBEDDINGS
            else None
        )

    def run_embedding_ablation_diagnostic(
            self,
            onnx_path: str,
            dataset: Dataset,
            qrels_dict: Dict[str, Dict[str, int]],
            base_log_probs: Optional[np.ndarray] = None,
            max_samples: int = 256,
            batch_size: int = 4,
    ) -> Dict[str, Any]:
        """
        3A Counterfactual Embedding Ablation:
        Re-evaluates `onnx_path` on a stratified slice of `dataset` with:
          1. `zeroed_embs`: all Stage-2 ranker embeddings set to 0.0
          2. `shuffled_embs`: Stage-2 ranker embeddings randomly permuted across queries
             (preserves norm distribution while destroying query-item alignment)
        Quantifies how much ranking quality comes from injected embeddings vs. text semantics.
        """
        if not self.INCLUDE_RANKER_EMBEDDINGS:
            return {
                "skipped": True,
                "reason": "INCLUDE_RANKER_EMBEDDINGS is False",
            }

        sub_indices = self._select_stratified_subsample_indices(
            dataset=dataset,
            max_samples=min(max_samples, len(dataset)),
            seed=self.seed,
        )
        sub_ds = dataset.select(sub_indices)
        sub_qrels = {
            f"q_{new_idx}": qrels_dict[f"q_{orig_idx}"]
            for new_idx, orig_idx in enumerate(sub_indices)
        }
        compute_fn = self._make_compute_metrics(sub_qrels, sub_ds)

        if base_log_probs is not None:
            base_lp = np.asarray(base_log_probs, dtype=np.float32)[sub_indices]
        else:
            base_lp = self._run_onnx_batched_inference(
                onnx_path=onnx_path,
                dataset=sub_ds,
                batch_size=batch_size,
                include_ranker_embeddings=True,
            )

        base_metrics = compute_fn((base_lp, None))
        base_probs = np.exp(base_lp)
        base_top1 = np.argmax(base_lp, axis=-1)

        real_embs = np.asarray(
            [row["ranker_embs"] for row in sub_ds], dtype=np.float32
        )
        rng = np.random.default_rng(self.seed + 999)
        if len(sub_ds) > 1:
            perm = rng.permutation(len(sub_ds))
            # Guarantee a derangement-like shift if permutation leaves items aligned
            if np.all(perm == np.arange(len(sub_ds))):
                perm = np.roll(perm, 1)
            shuffled_embs = real_embs[perm]
        else:
            shuffled_embs = np.zeros_like(real_embs)

        zero_lp = self._run_onnx_batched_inference(
            onnx_path=onnx_path,
            dataset=sub_ds,
            batch_size=batch_size,
            include_ranker_embeddings=False,
        )
        shuf_lp = self._run_onnx_batched_inference(
            onnx_path=onnx_path,
            dataset=sub_ds,
            batch_size=batch_size,
            include_ranker_embeddings=True,
            override_ranker_embs=shuffled_embs,
        )

        def _summarize_ablation(
                mode_name: str, abl_lp: np.ndarray
        ) -> Dict[str, Any]:
            abl_m = compute_fn((abl_lp, None))
            kl = float(
                np.mean(np.sum(base_probs * (base_lp - abl_lp), axis=-1))
            )
            top1_flip_rate = float(
                np.mean(np.argmax(abl_lp, axis=-1) != base_top1)
            )
            comp_key = self.PRIMARY_METRIC_NAME
            ndcg_key = f"ndcg@{self.EVAL_TOP_K}"
            entry = {
                "mode": mode_name,
                "num_examples": len(sub_ds),
                f"ablation_{comp_key}": round(float(abl_m[comp_key]), 5),
                f"delta_{comp_key}": round(
                    float(abl_m[comp_key] - base_metrics[comp_key]), 5
                ),
                f"ablation_{ndcg_key}": round(float(abl_m[ndcg_key]), 5),
                f"delta_{ndcg_key}": round(
                    float(abl_m[ndcg_key] - base_metrics[ndcg_key]), 5
                ),
                "top1_flip_rate": round(top1_flip_rate, 5),
                "mean_kl_from_baseline": round(kl, 6),
            }
            print(
                f"[DIAG_EMBEDDING_ABLATION] mode={mode_name} n={len(sub_ds)} "
                f"delta_{comp_key}={entry[f'delta_{comp_key}']:+.4f} "
                f"delta_{ndcg_key}={entry[f'delta_{ndcg_key}']:+.4f} "
                f"top1_flip_rate={top1_flip_rate:.4f} mean_kl={kl:.4f}",
                flush=True,
            )
            return entry

        return {
            "num_examples": len(sub_ds),
            "baseline_composite_ndcg": round(
                float(base_metrics[self.PRIMARY_METRIC_NAME]), 5
            ),
            "baseline_ndcg": round(
                float(base_metrics[f"ndcg@{self.EVAL_TOP_K}"]), 5
            ),
            "zeroed_embs": _summarize_ablation("zeroed_embs", zero_lp),
            "shuffled_embs": _summarize_ablation("shuffled_embs", shuf_lp),
        }

    def _compare_saved_logits_parity(
            self,
            portable_ckpt_dir: str,
            saved_logits_meta: Dict[str, Dict[str, str]],
            eval_dataset: Dataset,
            include_ranker_embeddings: bool = True,
            num_pt_examples: int = 2,
    ) -> Dict[str, Any]:
        """
        Compares all saved ONNX test logits stored in `self.test_logits_dir` without aborting:
          1. In-memory PyTorch Portable Checkpoint vs saved FP32 ONNX logits (`num_pt_examples`).
          2. Each quantized variant (`smoothquant_w8a8`, `awq_w4a16`) vs saved FP32 ONNX logits
             across all shared test indices.
        Returns a structured report containing all metrics and any tolerance violations.
        """
        report: Dict[str, Any] = {
            "comparisons": {},
            "parity_errors": [],
            "all_passed": True,
        }
        if "fp32" not in saved_logits_meta:
            return report

        fp32_log_probs = np.load(saved_logits_meta["fp32"]["logits_path"])
        fp32_indices = np.load(saved_logits_meta["fp32"]["indices_path"])
        fp32_idx_to_pos = {
            int(orig_idx): pos
            for pos, orig_idx in enumerate(fp32_indices.tolist())
        }

        # 1. PyTorch Portable Checkpoint vs saved FP32 ONNX logits
        n_pt = min(num_pt_examples, len(fp32_indices))
        pt_orig_indices = fp32_indices[:n_pt].tolist()
        sample_ds = eval_dataset.select(pt_orig_indices)

        pt_model, ckpt_tokenizer, ckpt_cfg = load_reusable_checkpoint_in_memory(
            checkpoint_dir=portable_ckpt_dir,
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            device="cpu",
        )
        try:
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
                    [row["ranker_embs"] for row in sample_ds],
                    dtype=torch.float32,
                )
            else:
                embs_pt = torch.zeros(
                    (n_pt, 1 + self.NUM_CANDIDATES, self.RANKER_OUT_FEATURES),
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
        finally:
            del pt_model
            gc.collect()

        pt_entry = self._evaluate_parity_arrays(
            comparison_name="pytorch_vs_fp32",
            actual_log_probs=pt_log_probs,
            desired_log_probs=fp32_log_probs[:n_pt],
            prob_rtol=1e-3,
            prob_atol=1e-3,
            log_rtol=1e-3,
            log_atol=1e-3,
            check_first_n=n_pt,
        )
        report["comparisons"]["pytorch_vs_fp32"] = pt_entry
        report["parity_errors"].extend(pt_entry["errors"])

        # 2. Quantized variants vs saved FP32 ONNX logits across all shared test examples
        cand_scale = math.log(max(2, self.NUM_CANDIDATES)) / math.log(5.0)
        quant_specs = [
            (
                "smoothquant_w8a8",
                0.25,
                0.55,
                0.40,
                round(0.40 * cand_scale, 4),
            ),
            ("awq_w4a16", 0.35, 0.55, 0.45, round(0.45 * cand_scale, 4)),
        ]

        for variant_name, prob_rtol, prob_atol, log_rtol, log_atol in quant_specs:
            if variant_name not in saved_logits_meta:
                continue

            var_log_probs = np.load(
                saved_logits_meta[variant_name]["logits_path"]
            )
            var_indices = np.load(
                saved_logits_meta[variant_name]["indices_path"]
            )

            shared_var_pos: List[int] = []
            shared_fp32_pos: List[int] = []
            for v_pos, orig_idx in enumerate(var_indices.tolist()):
                f_pos = fp32_idx_to_pos.get(int(orig_idx))
                if f_pos is not None:
                    shared_var_pos.append(v_pos)
                    shared_fp32_pos.append(f_pos)

            var_aligned = var_log_probs[shared_var_pos]
            fp32_aligned = fp32_log_probs[shared_fp32_pos]

            comp_key = f"{variant_name}_vs_fp32"
            var_entry = self._evaluate_parity_arrays(
                comparison_name=comp_key,
                actual_log_probs=var_aligned,
                desired_log_probs=fp32_aligned,
                prob_rtol=prob_rtol,
                prob_atol=prob_atol,
                log_rtol=log_rtol,
                log_atol=log_atol,
                check_first_n=len(shared_var_pos),
            )
            report["comparisons"][comp_key] = var_entry
            report["parity_errors"].extend(var_entry["errors"])

            del var_log_probs, var_indices, var_aligned, fp32_aligned
            gc.collect()

        del fp32_log_probs, fp32_indices
        gc.collect()

        report["all_passed"] = len(report["parity_errors"]) == 0
        return report

    def make_fp32_export(self, ckpt_dir: str, output_dir: str):
        return export_from_reusable_checkpoint(
            checkpoint_dir=ckpt_dir,
            triton_model_dir=os.path.join(output_dir, "triton_fp32"),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="fp32",
            calibration_prompts=None,
            calibration_embs=None,
        )

    def make_smoothquant_w8a8_export(self, ckpt_dir: str, output_dir: str):
        n_calib = 10 if self.is_unit_test else 24
        self.create_calib_rows_embs(n_calib)
        return export_from_reusable_checkpoint(
            checkpoint_dir=ckpt_dir,
            triton_model_dir=os.path.join(
                output_dir, "triton_smoothquant_w8a8"
            ),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="smoothquant_w8a8",
            calibration_prompts=self.calib_prompts,
            calibration_embs=self.calib_embs,
        )

    def make_awq_w4a16_export(self, ckpt_dir: str, output_dir: str):
        return export_from_reusable_checkpoint(
            checkpoint_dir=ckpt_dir,
            triton_model_dir=os.path.join(output_dir, "triton_awq_w4a16"),
            base_model_path=self.base_model_path,
            model_init_kwargs=self.model_init_kwargs,
            quantize_mode="awq_w4a16",
            calibration_prompts=None,
            calibration_embs=None,
        )