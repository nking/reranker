"""
script for running the evaluation and inference using the fine-tuned model without
distributed configuration.
"""
from typing import Dict, Tuple, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset as hf_load_dataset
import torch
import os
from tqdm.auto import tqdm # Used for progress bar

from movie_lens_reranker.tune_train_reranker import eval as _eval, \
  load_fine_tuned_model
from movie_lens_reranker.load_datasets import build_sequential_dataloader
import torch.distributed as dist

from peft import AutoPeftModelForSeq2SeqLM
from functools import partial

MODEL_NAME = "castorini/LiT5-Distill-base-v2"

def _get_device():
  if torch.cuda.is_available():
    if 'LOCAL_RANK' in os.environ:
      local_rank = int(os.environ["LOCAL_RANK"])
    else:
      local_rank = 0
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
  else:
    device = torch.device("cpu")
  return device

def run_inference(data_uri: str, fine_tuned_tokenizer_directory:str, fine_tuned_model_directory: str,
  batch_size: int):
  
  num_workers = 1
  
  #{"fine_tuned_model", "tokenizer",  "collator_function", "base_model"
  ft_model_dict = load_fine_tuned_model(fine_tuned_tokenizer_directory, fine_tuned_model_directory)
  
  dataloader, num_rows_dict = build_sequential_dataloader(data_uri, batch_size,
    num_workers, ft_model_dict['collator_function'])

  device = _get_device()
  
  predictions = _inference(dataloader, ft_model_dict['tokenizer'], ft_model_dict['fine_tuned_model'], device)
  
  return predictions

def run_evaluation(data_uri:str, fine_tuned_tokenizer_directory:str, fine_tuned_model_directory: str,
  batch_size:int, metrics:List[str])-> Dict[str, float]:
  
  device = _get_device()
  num_workers = 1
  
  # {"fine_tuned_model", "tokenizer",  "collator_function", "base_model"
  ft_model_dict = load_fine_tuned_model(fine_tuned_tokenizer_directory, fine_tuned_model_directory)

  dataloader, num_rows_dict = build_sequential_dataloader(data_uri, batch_size, num_workers, ft_model_dict['collator_function'])
  
  val_dict = _eval(dataloader, ft_model_dict['tokenizer'], ft_model_dict['fine_tuned_model'], device, metrics)
  
  rank = dist.get_rank() if dist.is_initialized() else 0

  print(f'rank {rank}: fine-tuned model eval on validation dataset: {val_dict}')
  return val_dict
  
def _inference(dataloader, tokenizer, model, device):
  
  model.eval()
  model.to(device)
  
  all_predictions = []
  
  # Disable gradient tracking for speed and memory
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running Inference"):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      
      generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,  # Max length for the generated ranking sequence
        num_beams=4,  # Use Beam Search for higher quality
        early_stopping=True,
        # Stop when all beam hypotheses reach EOS token
        # You might need to set the pad token ID explicitly for some models
        # pad_token_id=tokenizer.pad_token_id
        do_sample=False
      )
      
      predictions = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,  # Remove PAD, EOS, BOS tokens
        clean_up_tokenization_spaces=True
      )
      all_predictions.append(predictions)
  
  return all_predictions