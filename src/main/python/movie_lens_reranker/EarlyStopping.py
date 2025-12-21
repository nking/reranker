
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

    def __call__(self, val_loss, val_perplexity, model, rank) -> bool:
      if val_perplexity < (self.best_ppl - self.min_delta):
        self.best_ppl = val_perplexity
        self.save_checkpoint(model, rank)
        self.counter = 0
        return True
      else:
        self.counter += 1
        if self.counter >= self.patience:
          self.early_stop = True
        return False
    
    def _check_val(self, val_loss, model, rank):
      if self.best_loss is None:
        self.best_loss = val_loss
        self.save_checkpoint(model, rank)
      elif val_loss > self.best_loss - self.min_delta:
        self.counter += 1
        if self.verbose and rank == 0:
          print(
            f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
          self.early_stop = True
      else:
        self.best_loss = val_loss
        self.save_checkpoint(model, rank)
        self.counter = 0
      return self.early_stop
    
    def save_checkpoint(self, model, rank):
      if self.checkpoint_path is None:
        return
      if rank == 0:
        if self.verbose:
          print(f"rank {rank}: Validation loss improved. Saving model to {self.checkpoint_path}...")
        model_to_save = model.module if hasattr(model,"module") else model
        model_to_save.save_pretrained(self.checkpoint_path)