import os
from typing import List
import tensorflow as tf

def get_kaggle() -> bool:
  cwd = os.getcwd()
  if "kaggle" in cwd:
    kaggle = True
  else:
    kaggle = False
  return kaggle

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

def load_list_of_globs_into_tfrecords(list_of_globs:List[str], batch_size:int=256):
  
  def load_tfrecords(filepath):
    """Creates a TFRecordDataset, setting compression based on the file path."""
    is_compressed = tf.strings.regex_full_match(filepath, r".*\.gz$")
    compression_type = tf.where(is_compressed, tf.constant("GZIP"),tf.constant(""))
    return tf.data.TFRecordDataset(filepath,  compression_type=compression_type)
    
  files_dataset = tf.data.Dataset.list_files(list_of_globs, shuffle=False                                             )
  
  # 2. Use interleave to read from multiple files concurrently
  # This is the most efficient method for reading many files.
  records_dataset = files_dataset.interleave(
    load_tfrecords,
    cycle_length=tf.data.AUTOTUNE,
    block_length=1,
    num_parallel_calls=tf.data.AUTOTUNE
  )
  #use largest batch size possible while avoiding out-of-memory errors
  dataset = records_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return dataset

