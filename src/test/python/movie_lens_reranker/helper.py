import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Dict, Union
import importlib
import os

from enum import Enum
class Split(Enum):
    TRAIN = "training"
    VAL = "validation"
    TEST = "test"

import importlib.util
import os

def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    if get_kaggle():
        try:
            kaggle_secrets = importlib.import_module("kaggle_secrets")
            user_secrets = kaggle_secrets.UserSecretsClient()
            token = user_secrets.get_secret("hugging_face")
            if token:
                token = token.strip()
                os.environ["HF_TOKEN"] = token
                return token
        except Exception as e:
            print(f"Warning: Could not load 'hugging_face' from Kaggle secrets: {e}")

    # Check local project .huggingface_token file
    file_path = os.path.join(get_project_dir(), ".huggingface_token")
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                token = file.read().strip()
                if token:
                    os.environ["HF_TOKEN"] = token
                    return token
        except OSError as e:
            print(f"Warning: Error reading '{file_path}': {e}")

    # Fallback to local Hugging Face CLI cache (~/.cache/huggingface/token)
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

def get_kaggle() -> bool:
    cwd = os.getcwd()
    if "kaggle" in cwd:
        kaggle = True
    else:
        kaggle = False
    return kaggle

def get_data_dir():
    if get_kaggle():
        return "/tmp/data"
    else:
        return os.path.join(get_project_dir(), "src/test/resources/data/")

def get_project_dir() -> str:
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

def get_output_dir() -> str:
    if get_kaggle():
        return "/tmp/output"
    return os.path.join(get_project_dir(), "bin")

def get_output_working_dir():
    if get_kaggle():
        return"/kaggle/working"
    else:
        return get_output_dir()

class DataSize(Enum):
    FULL = 'full'
    SMALL = 'small'
    TINY = 'tiny'
    TINY3 = 'tiny3'  # made from recommender_systems/src/test/python/movie_lens_tfx/write_tiny2_tiered_liked_splits.py

def get_train_val_test_liked_uris(data_size:DataSize=DataSize.TINY) -> Dict[str, str]:
    base_uri = get_data_dir()
    if data_size == DataSize.TINY3:
        base_uri = os.path.join(base_uri, "tiny3")
    if data_size == DataSize.SMALL:
        base_uri = os.path.join(base_uri, "small")
    elif data_size == DataSize.TINY:
        base_uri = os.path.join(base_uri, "tiny")
    out = {}
    for key in ('train_3', 'val_3', 'test_3', 'train_liked', 'val_liked', 'test_liked', 'train_disliked', 'val_disliked', 'test_disliked'):
        out[key] = os.path.join(base_uri, f"ratings_{key}.parquet")
    return out

def get_parquet(data_size:DataSize=DataSize.TINY, split=Split) -> str:
    base_uri = get_data_dir()
    if data_size == DataSize.TINY3:
        base_uri = os.path.join(base_uri, "tiny3")
    file_path = os.path.join(base_uri, f"reranker_{split.value}_parquet_q01_r01.parquet")
    return file_path

def get_git_commit_hash() -> str :
    if os.path.exists("./git_commit_hash.txt"):
        with open("./git_commit_hash.txt", "r") as file:
            return file.readline().strip()
    else:
        return _get_git_commit_hash_if_match()

def _get_git_commit_hash_if_match() -> Union[str, None]:
    try:
        # Check if the directory is a git repository
        # 'git rev-parse --is-inside-work-tree' returns 'true' if inside a git repo
        is_git = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if is_git.returncode != 0 or is_git.stdout.strip() != "true":
            return None

        # Get the remote repository URL
        # 'git config --get remote.origin.url' retrieves the origin URL
        remote_url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if remote_url.returncode != 0 or remote_url.stdout.strip() != "git@github.com:nking/reranker.git":
            return None

        # Get the current commit hash (git rev-parse HEAD)
        rev_parse = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if rev_parse.returncode == 0:
            return rev_parse.stdout.strip()
        else:
            return None
    except FileNotFoundError:
        return None

def get_package_version() -> str:
    try:
        return version("movie_lens_reranker")
    except PackageNotFoundError:
        return ""
        
        