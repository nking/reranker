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
    
    parser = argparse.ArgumentParser(description="pipeline flags")
    parser.add_argument(
        "--is_unit_test",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="If True, uses DataSize.TINY3 and a 2-step smoke-test schedule; defaults to False.",
    )
    args, _ = parser.parse_known_args()
    args_dict = vars(args)
    
    pipeline_runner = PipelineRunner(is_unit_test=args.is_unit_test)
    
    pipeline_runner.run_pipeline()