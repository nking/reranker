from enum import Enum
from typing import Optional, Dict
import importlib
import os

def get_hf_token() -> Optional[str]:
    # Check standard environment variables first
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()

    # Check Kaggle Secrets dynamically (avoids PyCharm static analysis errors locally)
    if get_kaggle():
        try:
            kaggle_secrets = importlib.import_module("kaggle_secrets")
            user_secrets = kaggle_secrets.UserSecretsClient()
            return user_secrets.get_secret("hugging_face")
        except Exception as e:
            print(f"Warning: Could not load 'hugging_face' from Kaggle secrets: {e}")

    # Check local project .huggingface_token file
    file_path = os.path.join(get_project_dir(), ".huggingface_token")
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
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

def get_train_parquet(data_size:DataSize=DataSize.TINY) -> str:
    base_uri = get_data_dir()
    if data_size == DataSize.TINY3:
        base_uri = os.path.join(base_uri, "tiny3")
    file_path = os.path.join(base_uri, "reranker_training_parquet_q01_r01.parquet")
    return file_path

