import os
from enum import Enum
from typing import Tuple, Dict


def get_kaggle() -> bool:
    cwd = os.getcwd()
    if "kaggle" in cwd:
        kaggle = True
    else:
        kaggle = False
    return kaggle

def get_hf_token():
    file_path = os.path.join(get_project_dir(), ".huggingface_token")
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

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

def get_bin_dir() -> str:
    return os.path.join(get_project_dir(), "bin")


class DataSize(Enum):
    FULL = 'full'
    SMALL = 'small'
    TINY = 'tiny'
    TINY3 = 'tiny3'  # made from recommender_systems/src/test/python/movie_lens_tfx/write_tiny2_tiered_liked_splits.py

def get_train_val_test_liked_uris(data_size:DataSize=DataSize.TINY, use_gcs_uri:bool=False) -> Dict[str, str]:
    if use_gcs_uri:
        base_uri = "gs://data/"
    else:
        base_uri = os.path.join(get_project_dir(), "src/test/resources/data/")
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
    base_uri = os.path.join(get_project_dir(), "src/test/resources/data/")
    if data_size == DataSize.TINY3:
        base_uri = os.path.join(base_uri, "tiny3")
    file_path = os.path.join(base_uri, "reranker_training_parquet_q01_r01.parquet")
    return file_path

