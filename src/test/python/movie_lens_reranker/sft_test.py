import os
import sys
import unittest

sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker"))
sys.path.append(os.path.join(os.getcwd(), "src/main/python/movie_lens_reranker"))
from PipelineRunner import *

import numpy as np
import onnxruntime as ort
import torch
from accelerate import PartialState
from datasets import Dataset

class TestSupervisedFineTuning(unittest.TestCase):
    _dataset_cache: Dict[
        Tuple[Any, bool], Tuple[Dataset, Dict[str, Dict[str, int]], int]
    ] = {}
    
    @classmethod
    def setUpClass(cls):
        cls.pipeline_runner = PipelineRunner(is_unit_test=True)
    
    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def testTorchRun(self):
        
        TestSupervisedFineTuning.pipeline_runner.run_train_phase(
            train_phase=TrainPhase.FIXED,
            assert_saved_checkpoint=True
        )
        
        if PartialState().is_main_process:
            history_path = TestSupervisedFineTuning.pipeline_runner.history_path
            self.assertTrue(os.path.exists(history_path))
            
            with open(history_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
       
            # Save ~11.5 MB portable checkpoint (Orbax equivalent)
            portable_ckpt_dir = TestSupervisedFineTuning.pipeline_runner.portable_ckpt_dir
            
            if not get_kaggle():
                # Verify FP32 ONNX parity
                fp32_onnx = os.path.join(TestSupervisedFineTuning.pipeline_runner._triton_onnx_path, "1", "model.onnx")
    
                # Export SmoothQuant (W8A8) & AWQ (W4A16) from the portable checkpoint
                calib_prompts = [row["prompt"] for row in TestSupervisedFineTuning.pipeline_runner.eval_dataset][:32]
                sq_onnx = export_from_reusable_checkpoint(
                    checkpoint_dir=portable_ckpt_dir,
                    triton_model_dir=os.path.join(
                        TestSupervisedFineTuning.pipeline_runner.output_dir, "triton_smoothquant_w8a8"
                    ),
                    base_model_path=TestSupervisedFineTuning.pipeline_runner.base_model_path,
                    model_init_kwargs=TestSupervisedFineTuning.pipeline_runner.model_init_kwargs,
                    quantize_mode="smoothquant_w8a8",
                    calibration_prompts=calib_prompts,
                )
                awq_onnx = export_from_reusable_checkpoint(
                    checkpoint_dir=portable_ckpt_dir,
                    triton_model_dir=os.path.join(TestSupervisedFineTuning.pipeline_runner.output_dir, "triton_awq_w4a16"),
                    base_model_path=TestSupervisedFineTuning.pipeline_runner.base_model_path,
                    model_init_kwargs=TestSupervisedFineTuning.pipeline_runner.model_init_kwargs,
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
                    eval_dataset=TestSupervisedFineTuning.pipeline_runner.eval_dataset,
                    include_ranker_embeddings=TestSupervisedFineTuning.pipeline_runner.INCLUDE_RANKER_EMBEDDINGS,
                    num_examples=2,
                )

    def testOptunaHPO(self):
        
        TestSupervisedFineTuning.pipeline_runner.run_hpo_phase(assert_saved_checkpoint=True)

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
        self.assertTrue(
            os.path.exists(onnx_path), f"Missing ONNX file: {onnx_path}"
        )

        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if TestSupervisedFineTuning.pipeline_runner.USE_CUDA and "CUDAExecutionProvider" in available_providers
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
        self.assertEqual(outputs_meta["candidate_log_probs"].type, "tensor(float)")

        # 2. Run in-memory inference on the sample eval examples
        n_samples = len(sample_dataset)
        prompts = [row["prompt"] for row in sample_dataset]

        orig_padding_side = TestSupervisedFineTuning.pipeline_runner.tokenizer.padding_side
        TestSupervisedFineTuning.pipeline_runner.tokenizer.padding_side = "left"
        try:
            enc = TestSupervisedFineTuning.pipeline_runner.tokenizer(
                prompts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=TestSupervisedFineTuning.pipeline_runner.MAX_SEQ_TOKEN_LENGTH,
            )
        finally:
            TestSupervisedFineTuning.pipeline_runner.tokenizer.padding_side = orig_padding_side

        if include_ranker_embeddings:
            ranker_embs_np = np.asarray(
                [row["ranker_embs"] for row in sample_dataset], dtype=np.float32
            )
        else:
            ranker_embs_np = np.zeros(
                (n_samples, 1 + TestSupervisedFineTuning.pipeline_runner.NUM_CANDIDATES, TestSupervisedFineTuning.pipeline_runner.RANKER_OUT_FEATURES),
                dtype=np.float32,
            )

        self.assertEqual(
            ranker_embs_np.shape,
            (n_samples, 1 + TestSupervisedFineTuning.pipeline_runner.NUM_CANDIDATES, TestSupervisedFineTuning.pipeline_runner.RANKER_OUT_FEATURES),
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
        self.assertEqual(cand_log_probs.shape, (n_samples, TestSupervisedFineTuning.pipeline_runner.NUM_CANDIDATES))
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
            TestSupervisedFineTuning.pipeline_runner.CANDIDATE_LABELS[int(np.argmax(row_scores))]
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
            range(min(num_examples, len(eval_dataset)))
        )
        n_samples = len(sample_ds)

        # 1. Load Portable Checkpoint in-memory as PyTorch OnnxExportableQwenReranker
        pt_model, ckpt_tokenizer, ckpt_cfg = load_reusable_checkpoint_in_memory(
            checkpoint_dir=portable_ckpt_dir,
            base_model_path=TestSupervisedFineTuning.pipeline_runner.base_model_path,
            model_init_kwargs=TestSupervisedFineTuning.pipeline_runner.model_init_kwargs,
            device="cpu",
        )
        self.assertEqual(int(ckpt_cfg["top_k"]), TestSupervisedFineTuning.pipeline_runner.NUM_CANDIDATES)

        ckpt_tokenizer.padding_side = "left"
        prompts = [row["prompt"] for row in sample_ds]
        enc_pt = ckpt_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TestSupervisedFineTuning.pipeline_runner.MAX_SEQ_TOKEN_LENGTH,
        )
        if include_ranker_embeddings:
            embs_pt = torch.tensor(
                [row["ranker_embs"] for row in sample_ds], dtype=torch.float32
            )
        else:
            embs_pt = torch.zeros(
                (n_samples, 1 + TestSupervisedFineTuning.pipeline_runner.NUM_CANDIDATES, TestSupervisedFineTuning.pipeline_runner.RANKER_OUT_FEATURES),
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
        self.assertEqual(pt_log_probs.shape, (n_samples, TestSupervisedFineTuning.pipeline_runner.NUM_CANDIDATES))

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