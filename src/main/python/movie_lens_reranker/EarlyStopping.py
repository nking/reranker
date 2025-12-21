import os
import shutil

class EarlyStopping:
    def __init__(self, patience=3, min_val_delta=1E-3,  min_perplexity_delta=0.05,
      checkpoint_path:str=None, verbose=True):
        self.patience = patience
        self.min_delta = min_val_delta
        self.min_perplexity_delta = min_perplexity_delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.best_ppl = float('inf')
        self.early_stop = False
        if self.verbose:
          print(f'checkpoint_path = {self.checkpoint_path}')

    def __call__(self, val_loss, val_perplexity, tokenizer, model, rank) -> bool:
      if val_perplexity < (self.best_ppl - self.min_delta):
        self.best_ppl = val_perplexity
        self.save_checkpoint(tokenizer, model, rank)
        self.counter = 0
        return True
      else:
        self.counter += 1
        if self.counter >= self.patience:
          self.early_stop = True
        return False
    
    def _init_dirs(self):
      cp_back = os.path.join(self.checkpoint_path, 'previous_checkpoint')
      tok_back = os.path.join(self.checkpoint_path, 'previous_tokenizer')
      cp_dir = os.path.join(self.checkpoint_path, 'latest_checkpoint')
      tok_dir = os.path.join(self.checkpoint_path, 'latest_tokenizer')
      if os.path.exists(tok_dir):
        os.replace(cp_dir, cp_back)
        os.replace(tok_dir, tok_back)
      os.makedirs(cp_dir, exist_ok=True)
      os.makedirs(tok_dir, exist_ok=True)
      return cp_dir, tok_dir
      
    def save_checkpoint(self, tokenizer, model, rank):
      if self.checkpoint_path is None:
        return
      if rank == 0:
        if self.verbose:
          print(f"rank {rank}: Validation loss improved. Saving model to {self.checkpoint_path}...")
        cp_dir, tok_dir = self._init_dirs()
        model_to_save = model.module if hasattr(model,"module") else model
        model_to_save.save_pretrained(cp_dir)
        tokenizer.save_pretrained(tok_dir)