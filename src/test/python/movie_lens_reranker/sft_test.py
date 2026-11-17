# isort: skip_file
import json
import os
import sys
import unittest

from pathlib import Path

from movie_lens_reranker.reranker_metrics import lookup_movie_tiers, TIER_NAMES


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
            
            self._print_dataset_tier_counts()
            
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
    
    def _count_tiers_in_dataset(self, dataset) -> dict:
        counts = {"head": 0, "torso": 0, "tail": 0}
        if dataset is None or len(dataset) == 0:
            return counts
        
        if "pos_tier" in dataset.column_names:
            tiers = np.asarray(dataset["pos_tier"], dtype=np.int64)
        else:
            movie_col = next(
                (
                    col
                    for col in ("pos_movie_id", "movie_id")
                    if col in dataset.column_names
                ),
                None,
            )
            if movie_col is None:
                return counts
            tiers = lookup_movie_tiers(
                dataset[movie_col],
                self.pipeline_runner.movie_tiers,
                self.pipeline_runner.movie_offset,
            )
        
        for tier_id, tier_name in TIER_NAMES.items():
            counts[tier_name] = int(np.sum(tiers == tier_id))
        return counts
    
    def _print_dataset_tier_counts(self):
        # 1. Catalog movie_tiers distribution (row 0 is reserved for OOV movie_id=0)
        catalog_tiers = np.asarray(self.pipeline_runner.movie_tiers,
            dtype=np.int64)
        valid_catalog = (
            catalog_tiers[1:] if catalog_tiers.size > 1 else catalog_tiers
        )
        catalog_counts = {
            tier_name: int(np.sum(valid_catalog == tier_id))
            for tier_id, tier_name in TIER_NAMES.items()
        }
        
        # 2. Full (cached) and active (subsampled) datasets for train, eval, and test
        full_train_ds, _, _ = self.pipeline_runner.get_dataset(SplitName.TRAIN)
        full_eval_ds, _, _ = self.pipeline_runner.get_dataset(SplitName.VAL)
        full_test_ds, full_test_qrels, _ = self.pipeline_runner.get_dataset(
            SplitName.TEST
        )
        
        active_train_ds = getattr(self.pipeline_runner, "train_dataset", None)
        active_eval_ds = getattr(self.pipeline_runner, "eval_dataset", None)
        active_test_ds = getattr(self.pipeline_runner, "test_dataset", None)
        if active_test_ds is None:
            if self.pipeline_runner.is_unit_test:
                test_cap = 2048 if self.pipeline_runner.USE_CUDA else 16
                active_test_ds, _ = (
                    self.pipeline_runner._subsample_dataset_and_qrels(
                        full_test_ds, full_test_qrels, test_cap
                    )
                )
            else:
                active_test_ds = full_test_ds
        
        print("\n" + "=" * 60, flush=True)
        print(
            f"[Tier Distribution ({self.pipeline_runner.DataSize.name})]",
            flush=True,
        )
        print(
            f"  Catalog movie_tiers (total={len(valid_catalog)}): {catalog_counts}",
            flush=True,
        )
        for label, ds in (
                ("train (active)", active_train_ds),
                ("eval  (active)", active_eval_ds),
                ("test  (active)", active_test_ds),
                ("train (full)  ", full_train_ds),
                ("eval  (full)  ", full_eval_ds),
                ("test  (full)  ", full_test_ds),
        ):
            tier_counts = self._count_tiers_in_dataset(ds)
            total = len(ds) if ds is not None else 0
            print(
                f"  {label} (n={total:4d}): {tier_counts}",
                flush=True,
            )
        print("=" * 60 + "\n", flush=True)
        
if __name__ == "__main__":
    unittest.main()