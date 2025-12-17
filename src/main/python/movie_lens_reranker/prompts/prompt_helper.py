import os
import enum
import glob

from packaging import version

class Prompt_Type(enum.Enum):
  QUERY = "query"
  PASSAGES = "passages"
  
def get_prompt_dir():
  return os.path.join(get_project_dir(), "src/main/python/movie_lens_reranker/prompts")

def extract_version(filepath:str, prefix:str):
  try:
    version_str = os.path.basename(filepath).replace(prefix,
      '').replace('.yaml', '')
    return version.parse(version_str)
  except (ValueError, AttributeError):
    return version.parse("0.0.0")
  
def get_yaml_prefix(prompt_type):
  if prompt_type == Prompt_Type.QUERY:
    return "query"
  elif prompt_type == Prompt_Type.PASSAGES:
    return "passages"
  else:
    raise ValueError(f"Invalid prompt type: {prompt_type}")
    
def get_yaml_prompt_path_latest(prompt_type):
  prefix = get_yaml_prefix(prompt_type)
  file_pattern = f'{get_prompt_dir()}/{prefix}-v*yaml'
  files = glob.glob(file_pattern)
  latest_file = max(files, key=lambda f: extract_version(f, prefix=prefix))
  return latest_file

def get_yaml_prompt_path(prompt_type: Prompt_Type, version: str=None) -> str:
  prefix = get_yaml_prefix(prompt_type)
  if version is None:
    return get_yaml_prompt_path_latest(prompt_type=prompt_type)
  file_path = os.path.join(get_prompt_dir(), f'{prefix}-v{version}.yaml')
  if os.path.exists(file_path):
    return file_path
  raise FileNotFoundError(f"File {file_path} not found")
  
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

