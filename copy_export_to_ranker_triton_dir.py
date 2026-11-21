import os
import shutil
import argparse
from pathlib import Path


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
    import textwrap
    
    epilog = textwrap.dedent(r"""
        Example Use:
            python3 copy_export_to_ranker_triton_dir.py \
            --exported_model_dir="./bin/exports/triton_smoothquant_w8a8" \
            --ranker_dir="../ranker" \
            --output_qwenreranker_version=1
        """)
    parser = argparse.ArgumentParser(
        description="copy exported model to ranker.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--exported_model_dir",
        nargs="?",
        type=str,
        required=True,
        help="the path to directory triton_awq_w4a16, triton_fp32, or triton_smoothquant_w8a8",
    )
    parser.add_argument(
        "--ranker_dir",
        nargs="?",
        type=str,
        required=True,
        help="the path to directory where the ranker github project is",
    )
    parser.add_argument(
        "--output_qwenreranker_version",
        nargs="?",
        type=int,
        required=True,
        help="the integer version number to store this model under.  e.g. 1",
    )
    parser.add_argument(
        "--dry_run",
        nargs="?",
        type=_str_to_bool,
        default=False,
        help="if true, prints the src and dest files, but does not copy",
    )
    args, _ = parser.parse_known_args()
    args_dict = vars(args)
    
    if not os.path.exists(args.exported_model_dir):
        raise NotADirectoryError(f"{args.exported_model_dir} does not exist")
    
    if not os.path.exists(args.ranker_dir):
        raise NotADirectoryError(f"{args.ranker_dir} does not exist")
    
    ranker_model_base_path = os.path.join(args.ranker_dir,
        "src/test/resources/model_repositories/triton_models/qwen-reranker")
    
    if not os.path.exists(ranker_model_base_path):
        raise NotADirectoryError(f"{ranker_model_base_path} does not exist")

    ranker_model_version_path = os.path.join(ranker_model_base_path, str(args.output_qwenreranker_version))
    
    if not os.path.exists(ranker_model_version_path):
        os.makedirs(ranker_model_version_path)
        
    paths = Path(args.exported_model_dir).glob("*")
    
    dry_run = args.dry_run
    if dry_run == None: #if not False or True, then it was entered as a flag without value and will be interpreted as true
        dry_run = True
    
    for path in paths:
        if path.is_file():
            if path.name == "config.pbtxt":
                if dry_run:
                    print(f"copy {path} to \n   {ranker_model_base_path}")
                else:
                    shutil.copy2(path, ranker_model_base_path)
            else:
                if dry_run:
                    print(f"copy {path} to \n   {ranker_model_version_path}")
                else:
                    shutil.copy2(path, ranker_model_version_path)
        elif path.is_dir():
            if path.name == "1":
                model_path = path / "model.onnx"
                if dry_run:
                    print(f"copy {model_path} to \n   {ranker_model_version_path}")
                else:
                    shutil.copy2(model_path, ranker_model_version_path)
        else:
            print(f"ignoring: {path}")
    
    if not dry_run:
        print(f"done writing to {ranker_model_version_path}")