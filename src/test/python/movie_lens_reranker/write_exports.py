import os
import sys
import argparse
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

from PipelineRunner import *

def _str_to_bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    val_lower = val.strip().lower()
    if val_lower in ("true", "1", "yes", "y"):
        return True
    if val_lower in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{val}'")

if __name__ == "__main__":
    
    import argparse
    import textwrap
    
    epilog = textwrap.dedent(r"""
        Example Uses:
          Make sure the reranker conda env is active and cd to the base of the project.

          Export the local test checkpoint:
            python3 src/test/python/movie_lens_reranker/write_exports.py \
              --checkpoint_dir="./bin/tiny3/portable_best_hpo_checkpoint" \
              --output_dir="./bin/exports" \
              --export_fp32=true \
              --export_sq_w8a8=true \
              --export_awq_w4a16=true \
              --is_unit_test=true

          Export the trained qwen-ranker checkpoint:
            python3 src/test/python/movie_lens_reranker/write_exports.py \
              --checkpoint_dir="../TMP_RERANKER/TMP01/full/portable_best_hpo_checkpoint" \
              --output_dir="./bin/exports" \
              --export_fp32=true \
              --export_sq_w8a8=true \
              --export_awq_w4a16=true \
              --is_unit_test=false
    """)
    parser = argparse.ArgumentParser(
        description="Pipeline flags.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--is_unit_test",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="If True, uses DataSize.TINY3 and a 2-step smoke-test schedule; defaults to False.",
    )
    parser.add_argument(
        "--export_fp32",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        help="If True, exports to an fp32 ONNX model.",
    )
    parser.add_argument(
        "--export_sq_w8a8",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        help="If True, exports to a SmoothQuant W8A8 ONNX model.",
    )
    parser.add_argument(
        "--export_awq_w4a16",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        help="If True, exports to a AWQ W4A16 ONNX model.",
    )
    
    default_ckpt_dir = os.path.join(get_output_working_dir(), "portable_train_checkpoint")
    if not os.path.exists(default_ckpt_dir):
        default_ckpt_dir = os.path.join(get_output_working_dir(), "portable_best_hpo_checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        nargs="?",
        default=default_ckpt_dir,
        type=str,
        help="the path to the checkpoint directory",
    )
    
    parser.add_argument(
        "--output_dir",
        nargs="?",
        default=get_output_dir(),
        type=str,
        help="the path to a directory to write the exports into.  it will be created if doesn't exist."
             "  The default is helpers get_output_dir().",
    )
    
    args, _ = parser.parse_known_args()
    args_dict = vars(args)
    
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"checkpoint_dir not found: {args.checkpoint_dir}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline_runner = PipelineRunner(is_unit_test=args.is_unit_test,
        seed=1234)
    
    if args.export_fp32:
        fp32_onnx = pipeline_runner.make_fp32_export(
            args.checkpoint_dir, args.output_dir)
    
    if args.export_sq_w8a8:
        sq8_onnx = pipeline_runner.make_smoothquant_w8a8_export(
            args.checkpoint_dir, args.output_dir)
        
    if args.export_awq_w4a16:
        awq16_onnx = pipeline_runner.make_awq_w4a16_export(
            args.checkpoint_dir, args.output_dir)