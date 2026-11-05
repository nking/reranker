import os

HAS_NVIDIA_GPU = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not HAS_NVIDIA_GPU:
    # Local CPU: Force single-threaded OpenMP to prevent C++ autograd barrier hangs
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
else:
    # Kaggle T4x2 (4 vCPUs / 2 GPU processes = 2 threads per process)
    # setdefault allows torchrun / accelerate launch to override if needed
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    
# Force single-threaded OpenMP on CPU BEFORE importing torch or ranx
#    to prevent C++ _engine_run_backward barrier deadlocks in Conda
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import shutil
import unittest
from typing import Dict

import torch
if not torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig
from ranx import Qrels, Run, evaluate
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

from helper import *

TOP_K = 5
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
USE_CUDA = torch.cuda.is_available()
CANDIDATE_LABELS = [chr(ord("A") + i) for i in range(TOP_K)]

#to see progress of first model download, can use this in terminal:
#   ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/

#hugging face automatically caches into ~/.cache/huggingface/hub/
#  and subsequently loads from there instead of from hugging face again.

class TestSupervisedFineTuning(unittest.TestCase):

    def setUp(self):
        self.token = get_hf_token()
        
        local_cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs"
        )
        local_cache_exists = (
                os.path.exists(local_cache_path) and len(
            os.listdir(local_cache_path)) > 0
        )
        
        if local_cache_exists:
            os.environ["HF_HUB_OFFLINE"] = "1"  # Must be a string
        
        self.output_dir = os.path.join(get_bin_dir(), "prototype_qwen_reranker")
        
        self.use_qlora_4bit = USE_CUDA

        # 1. Tokenizer & Candidate Token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=self.token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.candidate_token_ids = [
            self.tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in CANDIDATE_LABELS
        ]

        # 2. Quantization (BitsAndBytesConfig) & LoRA (LoraConfig)
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        
        device_index = PartialState().process_index
        self.model_init_kwargs = {
            "dtype": torch.float16 if USE_CUDA else torch.float32,
            "device_map": {"": device_index} if USE_CUDA else None,
            "attn_implementation": "sdpa" if USE_CUDA else "eager",
            "token": self.token,
            "local_files_only": local_cache_exists,
        }
        if not USE_CUDA:
            # Load only 2 of the 24 Qwen layers for fast CPU unit testing
            self.model_init_kwargs["num_hidden_layers"] = 2

    def tearDown(self):
        if PartialState().is_main_process and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def testTorchRun(self):
        n_train = 40 if USE_CUDA else 4
        n_eval = 16 if USE_CUDA else 2
        train_batch_size = 4 if USE_CUDA else 1
        eval_batch_size = 4 if USE_CUDA else 2
        grad_accum = 2 if USE_CUDA else 1
        epochs = 2 if USE_CUDA else 1
        step_freq = 5 if USE_CUDA else 2
        
        train_dataset, _ = self.build_fake_dataset(num_samples=n_train)
        eval_dataset, eval_qrels_dict = self.build_fake_dataset(
            num_samples=n_eval)
        
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            model_init_kwargs=self.model_init_kwargs,
            gradient_checkpointing=False,
            max_length=256,  # Must exceed prompt length (~175 tokens)
            packing=False,
            completion_only_loss=True,
            eos_token="<|im_end|>",
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            dataloader_pin_memory=USE_CUDA,
            dataloader_num_workers=0,
            learning_rate=2e-4,
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
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_ndcg@{TOP_K}",
            greater_is_better=True,
            ddp_find_unused_parameters=False if USE_CUDA else None,
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=MODEL_ID,
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
        )
        
        trainer.train()
        metrics = trainer.evaluate()
        print("Final Evaluation Metrics:", metrics)
        self.assertIn(f"eval_ndcg@{TOP_K}", metrics)
    
    def build_model_init_kwargs(
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
    
    def build_fake_dataset(self, num_samples: int = 40):
        """Generates fake prompt-completion pairs and ground-truth qrels for ranx."""
        records = []
        qrels_dict = {}

        for i in range(num_samples):
            prompt = (
                "<|im_start|>system\nYou are a personalized movie-ranking assistant."
                "Given the user's recent movie ratings in Watch History and"
                "a list of Candidate movies,"
                "re-order the candidates from most relevant to least relevant, but return only the candidate movie letter IDs that are in square brackets.<|im_end|>\n"
                "<|im_start|>user\nUser Watch History: Matrix, The (1999)::Action|Sci-Fi|Thriller::(rating 5/5);"
                " Open Your Eyes (Abre los ojos) (1997)::Drama|Romance|Sci-Fi::(rating 4/5);"
                "\n"
                "Candidates:\n"
                "[A] Toy Story 2 (1999)::Animation|Children's|Comedy\n"
                "[B] Star Wars: Episode IV - A New Hope (1977)::Action|Adventure|Fantasy|Sci-Fi\n"
                "[C] Across the Sea of Time (1995)::Documentary\n"
                "[D] Big Green, The (1995)::Children's|Comedy\n"
                "[E] Sense and Sensibility (1995)::Drama|Romance\n"
                "Ranked IDs:<|im_end|>\n<|im_start|>assistant\n"
            )
            completion = "B E A C D<|im_end|>"
            records.append({"prompt": prompt, "completion": completion})
            qrels_dict[f"q_{i}"] = {"B": 1.0, "E": 0.0, "A": 0.0, "C": 0.0, "D": 0.0}

        return Dataset.from_list(records), qrels_dict

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
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

    """
    import optuna
    def objective(self, trial: optuna.Trial) -> float:
        r_val = trial.suggest_categorical("lora_r", [8, 16, 32])
        lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    
        trial_lora = LoraConfig(
            r=r_val,
            lora_alpha=r_val * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_config.target_modules,
        )
        trial_sft_config = SFTConfig(
            output_dir=f"{self.output_dir}/trial_{trial.number}",
            model_init_kwargs=self.model_init_kwargs,
            learning_rate=lr,
            # ... rest of sft_config ...
        )
    
        trainer = SFTTrainer(
            model=MODEL_ID,
            args=trial_sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            quantization_config=self.bnb_config,
            peft_config=trial_lora,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.make_compute_metrics(eval_qrels_dict),
        )
        trainer.train()
        metrics = trainer.evaluate()
        return metrics[f"eval_ndcg@{TOP_K}"]
    
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=4)
    """
    

from transformers import TrainerCallback

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
        
if __name__ == "__main__":
    unittest.main()