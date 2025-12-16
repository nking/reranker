import os
from typing import Tuple

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
    with open(file_path, 'r') as file:
      file_content_string = file.read()
      file_content_string = file_content_string.strip()
    return file_content_string
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

def get_data_paths(use_small_data: bool=True) -> Tuple[str, str, str]:
  s_dir = os.path.join(get_project_dir(), "src/test/resources/data/sorted_1")
  t_dir = os.path.join(get_project_dir(), "src/test/resources/data/sorted_2")
  if use_small_data:
    return (os.path.join(s_dir, "trainsmall-00000-of-00001.parquet"),
      os.path.join(s_dir, "validationsmall-00000-of-00001.parquet"),
      os.path.join(t_dir, "testsmall-00000-of-00001.parquet"))
  return (os.path.join(s_dir, "train-00000-of-00001.parquet"),
    os.path.join(s_dir, "validation-00000-of-00001.parquet"),
    os.path.join(t_dir, "test-00000-of-00001.parquet"))
  
