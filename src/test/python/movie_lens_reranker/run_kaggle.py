import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker"))
sys.path.append(os.path.join(os.getcwd(), "src/main/python/movie_lens_reranker"))

from PipelineRunner import *

if __name__ == "__main__":
    pipeline_runner = PipelineRunner(is_unit_test=False)
    if get_kaggle():
        pipeline_runner._verify_kaggle_t4x2_readiness()

    # Both GPU ranks participate in DDP training during HPO
    pipeline_runner.run_hpo_phase()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Only Rank 0 exports ONNX models and evaluates on the test split
    if PartialState().is_main_process:
        pipeline_runner.run_test_phase()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
