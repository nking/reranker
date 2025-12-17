from typing import Dict, Any, Tuple, List

from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import PeftModel,  PeftConfig
from datasets import load_dataset, arrow_dataset
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate
from collections import defaultdict
import torch
import os
import argparse
from tqdm.auto import tqdm # Used for progress bar

import torch.distributed as dist

from peft import LoraConfig, TaskType, get_peft_model, \
  AutoPeftModelForSeq2SeqLM
from functools import partial

from movie_lens_reranker.load_datasets import hf_dataset_to_torch, custom_seq2seq_collator

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from movie_lens_reranker.load_datasets import DatasetWrapper

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

def run_inference(data_uri: str, fine_tuned_model_directory: str,
 batch_size: int):
  
  device = _get_device()
  if device.type == 'cuda':
    num_workers = 4 * torch.cuda.device_count()
  elif device.type == 'cpu':
    num_workers = 2 * os.cpu_count()
  else:
    raise ValueError(
      f"modify to include Unsupported device type {device.type}")
  
  model_dict = _load_models_and_tokenizers(fine_tuned_model_directory)
  
  dataloader, num_rows_dict = _build_dataloader(data_uri, batch_size,
    num_workers, model_dict['collator_function'])

  device = _get_device()
  
  predictions = _inference(dataloader, model_dict['fine_tuned_model'], device)
  
  return predictions

def run_evaluation(data_uri:str, fine_tuned_model_directory:str, batch_size:int, metrics:List[str]):
  
  device = _get_device()
  if device.type == 'cuda':
    num_workers = 4 * torch.cuda.device_count()
  elif device.type == 'cpu':
    num_workers = 2 * os.cpu_count()
  else:
    raise ValueError(
      f"modify to include Unsupported device type {device.type}")
  
  model_dict = _load_models_and_tokenizers(fine_tuned_model_directory)
  
  dataloader, num_rows_dict = _build_dataloader(data_uri, batch_size, num_workers, model_dict['collator_function'])
  
  loss_fine_tuned = _eval(dataloader, model_dict['tokenizer'], model_dict['fine_tuned_model'], device, metrics)
  
  loss_base_model = _eval(dataloader, model_dict['tokenizer'], model_dict['base_model'], device, metrics)
  
  print(f'base_model loss={loss_base_model}\n fine-tuned model loss={loss_fine_tuned}')
  return {"loss_base_model": loss_base_model, "loss_fine_tuned": loss_fine_tuned,
    "num_rows": num_rows_dict["train"]}
  
def _load_models_and_tokenizers(fine_tuned_model_directory)\
  -> Dict[str, AutoTokenizer | AutoPeftModelForSeq2SeqLM | PeftModel | partial]:
  """Loads the base model and then loads the PEFT adapter weights and sets it to eval model"""
  
  base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    # map_location=device #fails with cpu
  )
  
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  model = PeftModel.from_pretrained(
    base_model,
    fine_tuned_model_directory,
    is_trainable=False  # Set to False for inference/evaluation
  )
  print("Model successfully reloaded with PEFT adapter applied.")
  
  # OR Load the state dict and apply (More manual)
  # adapter_weights = torch.load(os.path.join(adapter_directory, "adapter_model.bin"))
  # model.load_state_dict(adapter_weights, strict=False)
  
  collator_function = partial(custom_seq2seq_collator, tokenizer=tokenizer)
  
  return {"fine_tuned_model" : model, "tokenizer" : tokenizer, "collator_function" : collator_function,
    "base_model" : base_model}
  
def _build_dataloader(data_uri, batch_size_per_replica, num_workers, collator_function)\
  -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
  """
  Build the train and validation DataLoaders
  
  Args:
    
    data_params:  dictionary of parameters:
    
      data_uri : uri to dataset.  expected to be a parquet file with columns
        ['user_id', 'age', 'movies', 'ratings', 'genres']
        where:
          user_id and age are integers
          movies, ratings, and genres are arrays of hard-negative mining values where relevance is ratings.
          the arrays' first elements are values for the positive point, i.e. a rating of "4" or "5" and
          the remaining elements are values for the negative points, i.e., ratings of "1", or "2".
      
      batch_size: batch_size_is_per_replica, which is 1 for this non-distributed inference
       
  """
  
  # ['user_id', 'age', 'movies', 'ratings', 'genres']
  
  hf_ds = load_dataset("parquet", data_files=data_uri)
  ds =  DatasetWrapper(hf_ds["train"])
  
  sampler = SequentialSampler(ds)  # a torch util
  
  dataloader = DataLoader(
    ds,
    sampler=sampler,
    batch_size=batch_size_per_replica,
    num_workers=num_workers,
    collate_fn=collator_function
  )
  
  return dataloader, hf_ds.num_rows

def _eval(dataloader, tokenizer, model, device, metrics):
  model.eval()
  model.to(device)
  
  total_val_loss = 0
  
  all_qrels_data = defaultdict(dict)
  all_run_data = defaultdict(dict)
  
  # Disable gradient tracking for speed and memory
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running Evaluation"):
      
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
      )
      
      loss = outputs.loss
      total_val_loss += loss.item()
      
      # metrics:
      predicted_ids_tensors = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_beams=1,
        do_sample=False
      ) #shape (batch_size, 50)
      
      q_ids = batch['query_id']
      qrels_dicts_list = batch['relevance_scores_dict']
      for i in range(len(q_ids)):
        query_id_str = str(q_ids[i])
        d = qrels_dicts_list[i]
        all_qrels_data[query_id_str].update(d)
        
        predicted_ranking_str = tokenizer.decode(
          predicted_ids_tensors[i],
          skip_special_tokens=True,
          clean_up_tokenization_spaces=True
        )
        predicted_doc_ids = predicted_ranking_str.split() #list of length 28
        
        run_scores_for_query = {}
        
        for rank, doc_id in enumerate(predicted_doc_ids):
          score = 1.0 / (rank + 1)
          run_scores_for_query[doc_id] = score
        
        all_run_data[query_id_str].update(run_scores_for_query)
  
  # Calculate average loss and reset model for potential further training
  # Calculate average loss and reset model for potential further training
  avg_val_loss = total_val_loss / len(dataloader)
  
  #expects all_qrels_data is dict of {queryid_1: {docid_A:score_a, docid_B:score_B}, ...
  # Create the ranx objects
  qrels = Qrels(all_qrels_data)
  run = Run(all_run_data, name="LiT5_Distill_v2_Run")
  
  results = ranx_evaluate(qrels, run, metrics=metrics)
  
  print(f'ranx result={results}')
  
  return avg_val_loss

def _inference(dataloader, model, tokenizer, device):
  
  model.eval()
  model.to(device)
  
  total_val_loss = 0
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