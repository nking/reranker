# isort: skip_file
import json
import os
import sys
import unittest

from pathlib import Path

def _get_project_dir() -> str:
    cwd = os.getcwd()
    head = cwd
    proj_dir = ""
    while head and head != os.sep:
        head, tail = os.path.split(head)
        if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
            if tail == "reranker":
                proj_dir = os.path.join(head, tail)
                break
    return proj_dir
proj_dir = _get_project_dir()
for path in ("src/main/python/movie_lens_reranker", "src/test/python/movie_lens_reranker"):
    target_path = os.path.join(proj_dir, path)
    target_path = str(Path(target_path).resolve())
    if target_path not in sys.path:
        sys.path.append(target_path)

# Must be imported before torch/datasets/transformers so PipelineRunner configures
# HF_HOME (/tmp on Kaggle) and CPU thread limits first.
from PipelineRunner import *

from accelerate import PartialState


class TestSupervisedFineTuning(unittest.TestCase):
    def setUp(self):
        self.pipeline_runner = PipelineRunner(is_unit_test=True)

    def tearDown(self):
        pass

    def testTorchRun(self):
        self.pipeline_runner.run_train_phase(
            train_phase=TrainPhase.FIXED,
            assert_saved_checkpoint=True,
        )

        if PartialState().is_main_process:
            history_path = self.pipeline_runner.train_history_path
            self.assertTrue(os.path.exists(history_path))
            with open(history_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            train_points = [
                m for m in metrics if "train_composite_ndcg@5" in m
            ]
            eval_points = [
                m for m in metrics if "eval_composite_ndcg@5" in m
            ]
            self.assertGreaterEqual(
                len(train_points),
                2,
                f"Expected >= 2 train curve points, got {len(train_points)}",
            )
            self.assertGreaterEqual(
                len(eval_points),
                2,
                f"Expected >= 2 eval curve points, got {len(eval_points)}",
            )

            params_path = self.pipeline_runner.train_params_path
            self.assertTrue(os.path.exists(params_path))

            portable_ckpt_dir = self.pipeline_runner.portable_ckpt_dir
            self.assertTrue(os.path.exists(portable_ckpt_dir))

            if not get_kaggle():
                fp32_onnx = os.path.join(
                    self.pipeline_runner._triton_onnx_path, "1", "model.onnx"
                )
                calib_prompts = [
                    row["prompt"] for row in self.pipeline_runner.eval_dataset
                ][:32]

                sq_onnx = export_from_reusable_checkpoint(
                    checkpoint_dir=portable_ckpt_dir,
                    triton_model_dir=os.path.join(
                        self.pipeline_runner.output_dir,
                        "triton_smoothquant_w8a8",
                    ),
                    base_model_path=self.pipeline_runner.base_model_path,
                    model_init_kwargs=self.pipeline_runner.model_init_kwargs,
                    quantize_mode="smoothquant_w8a8",
                    calibration_prompts=calib_prompts,
                )
                awq_onnx = export_from_reusable_checkpoint(
                    checkpoint_dir=portable_ckpt_dir,
                    triton_model_dir=os.path.join(
                        self.pipeline_runner.output_dir, "triton_awq_w4a16"
                    ),
                    base_model_path=self.pipeline_runner.base_model_path,
                    model_init_kwargs=self.pipeline_runner.model_init_kwargs,
                    quantize_mode="awq_w4a16",
                    calibration_prompts=calib_prompts,
                )
                self.assertTrue(os.path.exists(sq_onnx))
                self.assertTrue(os.path.exists(awq_onnx))

                self.pipeline_runner._assert_exported_models_work_in_memory(
                    portable_ckpt_dir=portable_ckpt_dir,
                    fp32_onnx_path=fp32_onnx,
                    sq_onnx_path=sq_onnx,
                    awq_onnx_path=awq_onnx,
                    eval_dataset=self.pipeline_runner.eval_dataset,
                    include_ranker_embeddings=self.pipeline_runner.INCLUDE_RANKER_EMBEDDINGS,
                    num_examples=2,
                )

    def testOptunaHPO(self):
        self.pipeline_runner.run_hpo_phase(assert_saved_checkpoint=True)

        if PartialState().is_main_process:
            self.assert_hpo_outputs()
           

    def testRunPipeline(self):
        self.pipeline_runner.run_pipeline()
        
        if PartialState().is_main_process:
            # outputs from the hpo phase are the checkpoint, best hyperparameters and metrics
            #assert hpo files written
            self.assert_hpo_outputs()
            
            #outpus from the tes phase are metrics
            test_history_path = self.pipeline_runner.test_history_path
            with open(test_history_path, "r", encoding="utf-8") as f:
                test_metrics_history = json.load(f)
                for key in ("quantize_mode", "num_test_examples", "metrics"):
                    self.assertIn(key, test_metrics_history)
                test_metrics = test_metrics_history["metrics"]
                self.assertIn(
                    self.pipeline_runner.PRIMARY_METRIC_NAME, test_metrics
                )
            
    def assert_hpo_outputs(self):
        history_path = self.pipeline_runner.best_hpo_history_path
        self.assertTrue(os.path.exists(history_path))
        with open(history_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        train_points = [
            m for m in metrics if "train_composite_ndcg@5" in m
        ]
        eval_points = [
            m for m in metrics if "eval_composite_ndcg@5" in m
        ]
        self.assertGreaterEqual(
            len(train_points),
            2,
            f"Expected >= 2 HPO train curve points, got {len(train_points)}",
        )
        self.assertGreaterEqual(
            len(eval_points),
            2,
            f"Expected >= 2 HPO eval curve points, got {len(eval_points)}",
        )
        
        params_path = self.pipeline_runner.best_params_path
        self.assertTrue(os.path.exists(params_path))
        
        portable_ckpt_dir = self.pipeline_runner.portable_best_hpo_ckpt_dir
        self.assertTrue(os.path.exists(portable_ckpt_dir))

if __name__ == "__main__":
    unittest.main()