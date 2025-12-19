
import os
import sys
import subprocess
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silences TensorFlow/XLA noise
os.environ['PYTHONWARNINGS'] = 'ignore'  # Optional: silences noisy python warnings

package_path = '/kaggle/working/reranker'
os.chdir(package_path)
#!pip install -q /kaggle/working/reranker
if package_path not in sys.path:
    sys.path.append(package_path)
sys.path.append("/kaggle/working/reranker/src/test/python/movie_lens_reranker")

from helper import *
from movie_lens_reranker.run_finetuned import run_evaluation

class RunOnKaggle():
  def __init__(self):
    self.train_path, self.validation_path, self.test_path = get_data_paths(use_small_data=False)
    self.n_nodes = 2 #for kaggle
    self.num_epochs = 4
    self.learning_rate = 5E-5
    self.num_workers = 1
    self.accumulation_steps = 8
    
    self.model_save_dir = os.path.join(get_bin_dir(), "best_lora_weights")
    self.tokenizer_save_dir = os.path.join(get_bin_dir(), "tokenizer")
    #"""
    try:
      shutil.rmtree(self.model_save_dir)
      shutil.rmtree(self.tokenizer_save_dir)
    except OSError as e:
      pass
    os.makedirs(self.model_save_dir, exist_ok=True)
    os.makedirs(self.tokenizer_save_dir, exist_ok=True)
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

  def run(self):
    
    script_path = os.path.join(get_project_dir(),
      "src/main/python/movie_lens_reranker/tune_train_reranker.py")
    command = [
      "torchrun",
      "--nproc_per_node", str(self.n_nodes),
      script_path,
      # model, data, and run params:
      "--train_uri", str(self.train_path),
      "--validation_uri", str(self.validation_path),
      "--num_epochs", str(self.num_epochs),
      "--model_save_dir_uri", str(self.model_save_dir),
      "--tokenizer_save_dir_uri", str(self.tokenizer_save_dir),
      "--checkpoint_dir_uri", str(self.checkpoints_dir),
      "--logs_dir_uri", str(self.logs_dir),
      "--num_epochs", str(self.num_epochs),
      "--learning_rate", str(self.learning_rate),
      "--num_workers", str(self.num_workers),
      "--accumulation_steps", str(self.accumulation_steps),
      "--metrics", "ndcg@5"
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
      avg_val_loss, perplexity_val, metric_results = run_evaluation(self.test_path, self.model_save_dir, batch_size=4, metrics=metrics)
      print(f'avg_val_loss, perplexity_val, metric_results={avg_val_loss, perplexity_val, metric_results}')
      
    except subprocess.CalledProcessError as e:
      # If the script failed, print stdout/stderr for debugging
      print(f"Subprocess failed with error code {e.returncode}")
      print("STDOUT:\n", e.stdout)
      print("STDERR:\n", e.stderr)
      self.fail(f"Distributed script failed: {e}")
    
    except subprocess.TimeoutExpired:
      self.fail("Distributed script timed out.")
   
r = RunOnKaggle()
r.run()
  