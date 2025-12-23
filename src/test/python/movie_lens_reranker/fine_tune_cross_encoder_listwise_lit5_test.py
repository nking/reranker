
import os
import sys
import subprocess
import unittest
from typing import Dict
import shutil

from run_finetuned import run_inference

sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker"))

from helper import *
from movie_lens_reranker.run_finetuned import run_evaluation

class TestFineTuning(unittest.TestCase):

  def setUp(self):
    res_dir = os.path.join(get_project_dir(), "src/test/resources/data/sorted_1/")
    self.train_path, self.validation_path, self.test_path = get_data_paths(use_small_data=True)
    self.n_nodes = 1
    self.num_epochs = 1
    self.validation_freq = 1
    self.learning_rate = 2E-4
    
    self.model_save_dir = os.path.join(get_bin_dir(), "saved")
    self.tokenizer_save_dir = os.path.join(get_bin_dir(), "tokenizer")
    #"""
    try:
      shutil.rmtree(self.model_save_dir)
      shutil.rmtree(self.tokenizer_save_dir)
    except OSError as e:
      pass
    os.makedirs(self.model_save_dir, exist_ok=True)
    os.makedirs(self.tokenizer_save_dir, exist_ok=True)
    
    cache_dir = '/tmp/torchinductor_nichole'  # Adjust if the path in your error differs
    if os.path.exists(cache_dir):
      shutil.rmtree(cache_dir)
      print("ClearedTorchInductor cache.")
      
    #"""
    
    self.checkpoints_dir = os.path.join(get_bin_dir(), "checkpoints")
    try:
      shutil.rmtree(self.checkpoints_dir)
    except OSError as e:
      pass
    os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    self.logs_dir = os.path.join(get_bin_dir(), "logs")
    try:
      shutil.rmtree(self.logs_dir)
    except OSError as e:
      pass
    os.makedirs(self.logs_dir, exist_ok=True)

  def testTorchRun(self):
    
    script_path = os.path.join(get_project_dir(),
      "src/main/python/movie_lens_reranker/tune_train_reranker.py")
    command = [
      "torchrun",
      "--nproc_per_node", str(self.n_nodes),
      script_path,
      # model, data, and run params:
      "--train_uri", str(self.train_path),
      "--validation_uri", str(self.validation_path),
      "--test_uri", str(self.test_path),
      "--validation_freq", str(self.validation_freq),
      "--num_epochs", str(self.num_epochs),
      "--model_save_dir_uri", str(self.model_save_dir),
      "--tokenizer_save_dir_uri", str(self.tokenizer_save_dir),
      "--checkpoint_dir_uri", str(self.checkpoints_dir),
      "--logs_dir_uri", str(self.logs_dir),
      "--num_epochs", str(self.num_epochs),
      "--learning_rate", str(self.learning_rate),
      "--metrics", "ndcg@5",
    ]
    #"--metrics", "ndcg@5 map mrr precision@5 recall@5 f1@5"
    
    print(f"Executing: {' '.join(command)}")

    try:
      #"""
      result = subprocess.run(
        command,
        check=True,
        capture_output=False,
        #text=True,
        #timeout=120  # Add a timeout to prevent hanging
      )
      #print(f"result: {result.stdout}")
      #self.assertIn("Epoch 1 finished.", result.stdout)
      #"""
      #metrics = ["ndcg@5", "map", "mrr", "precision@5", "recall@5", "f1@5"]
      metrics = ["ndcg@5"]
      print(f'EVALUATION using non-distributed scripts:')
      eval_dict = run_evaluation(self.test_path, self.tokenizer_save_dir, self.model_save_dir, batch_size=4, metrics=metrics)
      print(f'eval_dict={eval_dict}')
      print(f'INFERENCE using non-distributed scripts:')
      predictions = run_inference(self.test_path,
        self.tokenizer_save_dir, self.model_save_dir, batch_size=4)
      print(f'predictions={predictions}')
      
    except subprocess.CalledProcessError as e:
      # If the script failed, print stdout/stderr for debugging
      print(f"Subprocess failed with error code {e.returncode}")
      print("STDOUT:\n", e.stdout)
      print("STDERR:\n", e.stderr)
      self.fail(f"Distributed script failed: {e}")
    
    except subprocess.TimeoutExpired:
      self.fail("Distributed script timed out.")
   
if __name__ == '__main__':
    unittest.main()
  