from typing import Dict, Any, Tuple, List

from torch.cuda import device
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import argparse
from tqdm.auto import tqdm # Used for progress bar
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM
import torch.distributed as dist

from peft import LoraConfig, TaskType, get_peft_model, \
  AutoPeftModelForSeq2SeqLM
from functools import partial

from load_datasets import hf_dataset_to_torch, custom_seq2seq_collator

import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def build_model_lit5(model_params: Dict[str, Any]) -> Tuple[AutoTokenizer, AutoPeftModelForSeq2SeqLM, partial]:
  """
  Build a lit5 model prepared for Lora PEFT fine-tuning from the pre-trained model .castorini/LiT5-Distill-base-v2
  
  Args:
    
    model_params:  dictionary of parameters:
    
      lora_rank: (optional).  Lora attention dimension (the "rank").  By default is 4.
      
      lora_alpha: (optional) The alpha parameter for Lora scaling.  by default is 32,
        generally lora_alpha = 2 * lora_rank to 4 * lora_rank.
        Higher alpha boosts the influence of the learned low-rank matrices (A and B) on the original weights,
        essentially multiplying the effective learning rate. An alpha that is too high results in overfitting.
      
      lora_dropout: (optional)
      
  Returns:
    a tuple of the tokenizer, the PEFT adapted model, the collator function
  """
  
  MODEL_NAME = "castorini/LiT5-Distill-base-v2"
  
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  # from transformers import T5ForConditionalGeneration
  # model2 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME) is the same as:
  model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    # map_location=device #fails with cpu
  )
  
  lora_config = LoraConfig(
    r=model_params["lora_rank"],
    lora_alpha=model_params["lora_alpha"],
    target_modules=["q", "v"],
    lora_dropout=model_params["lora_dropout"],
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    init_lora_weights=False,
  )
  
  lora_model = get_peft_model(model, lora_config)
  lora_model.print_trainable_parameters()
  
  collator_function = partial(custom_seq2seq_collator, tokenizer=tokenizer)
  
  return tokenizer, lora_model, collator_function
  
def build_dataloaders(data_params: Dict[str, Any], collator_function:partial, device)\
  -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
  """
  Build the train and validation DataLoaders
  
  Args:
    
    data_params:  dictionary of parameters:
    
      train_uri : uri to training dataset.  expected to be a parquet file with columns
        ['user_id', 'age', 'movies', 'ratings', 'genres']
        where:
          user_id and age are integers
          movies, ratings, and genres are arrays of hard-negative mining values where relevance is ratings.
          the arrays' first elements are values for the positive point, i.e. a rating of "4" or "5" and
          the remaining elements are values for the negative points, i.e., ratings of "1", or "2".
      
      validation_uri : uri to validation dataset.  expected to be a parquet file with columns
        ['user_id', 'age', 'movies', 'ratings', 'genres']
        where:
          user_id and age are integers
          movies, ratings, and genres are arrays of hard-negative mining values where relevance is ratings.
          the arrays' first elements are values for the positive point, i.e. a rating of "4" or "5" and
          the remaining elements are values for the negative points, i.e., ratings of "1", or "2".
          
      device: torch.device of 'cpu', 'gpu', 'tpu'
      
      num_epochs = number of epochs to train the model
      
      batch_size_is_per_replica:  when True, the batch size given is per replica, else it is the global batch size.
      
      batch_size: depending upon batch_size_is_per_replica, this is the per replica batch size else the global batch size.
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas
        TRAIN_STEPS_PER_EPOCH = math.ceil(num_train) / GLOBAL_BATCH_SIZE)
        EVAL_STEPS_PER_EPOCH = math.ceil(num_eval) / GLOBAL_BATCH_SIZE)
       
  """
  
  """
  BATCH_SIZE_PER_REPLICA = hp.get("BATCH_SIZE")
  NUM_EPOCHS = hp.get("NUM_EPOCHS")

  n_replicas = strategy.num_replicas_in_sync
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas

  # virtual epochs:
  TRAIN_STEPS_PER_EPOCH = math.ceil(hp.get("num_train") / GLOBAL_BATCH_SIZE)
  EVAL_STEPS_PER_EPOCH = math.ceil(hp.get("num_eval") / GLOBAL_BATCH_SIZE)
  """
  
  train_dataset, validation_dataset, num_rows_dict = hf_dataset_to_torch(data_params["train_uri"],
    data_params["validation_uri"])
  
  if device.type == 'cuda':
    train_sampler = DistributedSampler(
      train_dataset,
      num_replicas=data_params["num_replicas"],
      rank=data_params["rank"],
      shuffle=True,
      set_pin=True
    )
  else:
    train_sampler = DistributedSampler(
      train_dataset,
      num_replicas=data_params["num_replicas"],
      rank=data_params["rank"],
      shuffle=True
    )
  
  train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=data_params["batch_size_per_replica"],
    num_workers=data_params["num_workers"],
    collate_fn=collator_function
  )
  
  validation_sampler = DistributedSampler(
    validation_dataset,
    num_replicas=data_params["num_replicas"],
    rank=data_params["rank"],
    shuffle=True
  )
  
  validation_dataloader = DataLoader(
    validation_dataset,
    sampler=validation_sampler,
    batch_size=data_params["batch_size_per_replica"],
    num_workers=data_params["num_workers"],
    collate_fn=collator_function
  )
  
  return train_dataloader, validation_dataloader, num_rows_dict

def train(train_dataloader, validation_dataloader, tokenizer, model, device, optimizer, scheduler, run_params:Dict[str, Any]):
  """
  
  run_params["GLOBAL_BATCH_SIZE"] = GLOBAL_BATCH_SIZE
  run_params['TRAIN_STEPS_PER_EPOCH'] = TRAIN_STEPS_PER_EPOCH
  run_params['VALIDATIOM_STEPS_PER_EPOCH'] = VALIDATIOM_STEPS_PER_EPOCH
  run_params['NUM_TRAINING_STEPS'] = NUM_TRAINING_STEPS
  
  """
  total_steps = run_params["NUM_TRAINING_STEPS"]
  
  for epoch in range(run_params["num_epochs"]):
    
    total_train_loss = 0
    
    if hasattr(train_dataloader, 'sampler') and isinstance(
      train_dataloader.sampler, DistributedSampler):
      train_dataloader.sampler.set_epoch(epoch)
      
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{run_params['num_epochs']}")
    
    model.train()
    
    for batch_idx, batch in enumerate(progress_bar):
      
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      
      optimizer.zero_grad()
      
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
        # The T5 model calculates the Sequence-to-Sequence loss internally
        # when 'labels' are provided, returning it as outputs.loss
      )
      
      #TODO: add metrics
      #TODO: add logging to run_params['logs_dir']
      
      loss = outputs.loss
      total_train_loss += loss.item()
      
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
      optimizer.step()
      scheduler.step()
      
      # Update progress bar with current loss
      progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}',
        'avg_loss': f'{total_train_loss / (batch_idx + 1):.4f}'})
    
    avg_epoch_loss = total_train_loss / len(train_dataloader)
    print(f"\nEpoch {epoch + 1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
    
    if ((batch_idx +1) % run_params['validation_freq']) == 0:
      avg_val_loss = eval(validation_dataloader, tokenizer, model, device, run_params['metrics'])
      print(f"\nAverage Validation Loss: {avg_val_loss:.4f}")
      model.train()
    
    if "checkpoint_dir_uri" in run_params:
      path = f"./{run_params['checkpoint_dir_uri']}/epoch_{epoch}"
      save_peft_model_checkpoint(model, path, run_params['rank'])
  
  save_peft_model_checkpoint(model, run_params['model_save_dir_uri'], run_params['rank'])

def eval(validation_dataloader, tokenizer, model, device, metrics):
  
  model.eval()
  
  total_val_loss = 0
  
  all_qrels_data = defaultdict(dict)
  all_run_data = defaultdict(dict)
  
  # Disable gradient tracking for speed and memory
  with (torch.no_grad()):
    
    for batch in validation_dataloader:
      
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
      )  # shape (batch_size, 50)
      
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
        predicted_doc_ids = predicted_ranking_str.split()  # list of length 28
        
        run_scores_for_query = {}
        
        for rank, doc_id in enumerate(predicted_doc_ids):
          score = 1.0 / (rank + 1)
          run_scores_for_query[doc_id] = score
        
        all_run_data[query_id_str].update(run_scores_for_query)
  
  # Calculate average loss and reset model for potential further training
  avg_val_loss = total_val_loss / len(validation_dataloader)
  
  # Create the ranx objects
  qrels = Qrels(all_qrels_data)
  run = Run(all_run_data, name="LiT5_Distill_v2_Run")
  
  results = ranx_evaluate( qrels, run, metrics=metrics)
  
  print(f'ranx result={results}')
  
  return avg_val_loss

def save_peft_model_checkpoint(ddp_model, save_directory, global_rank):
  """Saves the PEFT adapter weights on the master process (Rank 0)."""
  
  if global_rank != 0:
    # Wait for Rank 0 to finish saving if you have post-save logic
    # dist.barrier()
    return
  
  print(f"Rank {global_rank}: Saving model checkpoint to {save_directory}")
  
  # Unwrap the Model
  # DDP adds the 'module' attribute, giving access to the original model.
  unwrapped_model = ddp_model.module
  
  # Save the PEFT Adapter
  # This saves the adapter weights and the configuration (adapter_config.json).
  unwrapped_model.save_pretrained(save_directory)
  
  # Note: The base model remains untouched and is not saved here.
  print(f"Rank {global_rank}: Successfully saved PEFT adapter.")
  
def prepare_data_and_model(params, device:torch.device)\
  -> Tuple[DDP, AutoTokenizer, DataLoader, DataLoader, Dict[str, int]]:
  
  tokenizer, lora_model, collator_function = build_model_lit5(params)
  
  train_dataloader, validation_dataloader, num_rows_dict = build_dataloaders(params, collator_function=collator_function, device=device)
  
  lora_model = lora_model.to(device)
  
  # 4. Wrap Model in DDP
  # For CPU training, device_ids is often omitted.
  if device.type == 'cuda':
    model = DDP(lora_model, device_ids=[device.index])
  else:
    model = DDP(lora_model)  # DDP works on CPU using the 'gloo' backend
  
  return model, tokenizer, train_dataloader, validation_dataloader, num_rows_dict

def setup_distributed() -> Tuple[Any, int, torch.device]:
  """Initializes the distributed environment."""
  # Use environment variables set by the launcher (torchrun)
  
  # The unique identifier assigned to the current process across all machines/GPUs
  # participating in the distributed job.
  # e.g. If running on 4 GPUs on one machine, the RANK values will be $0, 1, 2, 3.
  #      If running on 2 machines, each with 4 GPUs, the RANK values will be 0 through 7.
  rank = int(os.environ["RANK"])
  
  # The total number of processes (replicas) participating in the distributed job across all machines/GPUs.
  world_size = int(os.environ["WORLD_SIZE"])
  
  """
  example:
     torchrun --nproc_per_node=4 tune_train_reranker.py

     This launches 4 processes. Each process will see:
        Process 0: RANK=0, WORLD_SIZE=4
        Process 1: RANK=1, WORLD_SIZE=4
        Process 2: RANK=2, WORLD_SIZE=4
        Process 3: RANK=3, WORLD_SIZE=4
  """
  
  # Check for CUDA availability
  if torch.cuda.is_available():
    backend = 'nccl' #handles gpu to gpu communication
    # LOCAL_RANK is the GPU index on the current node
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
  else:
    # Use 'gloo' for CPU or single-node CPU testing
    backend = 'gloo'
    device = torch.device("cpu")
  
  # Initialize the process group
  dist.init_process_group(backend=backend, rank=rank,
    world_size=world_size)
  
  return rank, world_size, device

def cleanup_distributed():
  """Tears down the distributed environment."""
  dist.destroy_process_group()

def main(params:Dict[str,Any]):
  """
  run the default trainer.
  Note that this has to be invoked by torchrun.
  For example:
      torchrun --nproc_per_node=4 tune_train_reranker.py
  if 1 CPU w/ 4 cores, this runs  "gloo"
  
  The following arguments must follow the
  
  """
  rank, world_size, device = setup_distributed()
  
  params["rank"] = rank
  params["num_replicas"] = world_size
  
  if device.type == 'cuda':
    params["num_workers"] = 4 * torch.cuda.device_count()
  elif device.type == 'cpu':
    params["num_workers"] = 2 * os.cpu_count()
  else:
    raise ValueError(f"modify to include Unsupported device type {device.type}")
  
  model, tokenizer, train_dataloader, validation_dataloader, num_rows_dict\
    = prepare_data_and_model(params, device)
  
  trainable_params = [
    p for p in model.parameters() if p.requires_grad
  ]
  
  optimizer = optim.Adam(trainable_params, lr=params["learning_rate"])
  
  BATCH_SIZE_PER_REPLICA = params["batch_size_per_replica"]
  NUM_EPOCHS = params["num_epochs"]
  n_replicas = params["num_replicas"]
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas
  
  params["num_train"] = num_rows_dict["train"]
  params["num_validation"] = num_rows_dict["validation"]
  # virtual epochs:
  TRAIN_STEPS_PER_EPOCH = int(float.__ceil__(params["num_train"] / GLOBAL_BATCH_SIZE))
  VALIDATIOM_STEPS_PER_EPOCH = int(float.__ceil__(params["num_validation"] / GLOBAL_BATCH_SIZE))
  
  from transformers import get_linear_schedule_with_warmup
  NUM_TRAINING_STEPS = TRAIN_STEPS_PER_EPOCH * NUM_EPOCHS
  # Typically 5-10% of total steps
  NUM_WARMUP_STEPS = int(NUM_TRAINING_STEPS * 0.05)
  
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=NUM_TRAINING_STEPS
  )
  
  params["GLOBAL_BATCH_SIZE"] = GLOBAL_BATCH_SIZE
  params['TRAIN_STEPS_PER_EPOCH'] = TRAIN_STEPS_PER_EPOCH
  params['VALIDATIOM_STEPS_PER_EPOCH'] = VALIDATIOM_STEPS_PER_EPOCH
  params['NUM_TRAINING_STEPS'] = NUM_TRAINING_STEPS
  
  train(train_dataloader, validation_dataloader, tokenizer, model, device, optimizer, scheduler, params)

  cleanup_distributed()
  
def parse_args():
  parser = argparse.ArgumentParser(description="Distributed PEFT Fine-Tuner of pre-trained LiT5 Reranker")

  parser.add_argument(
    "--train_uri", type=str,
    help="uri for training data parquet file containing columns 'user_id', 'age', 'movies', 'ratings', 'genres."
  )
  parser.add_argument(
    "--validation_uri", type=str,
    help="uri for validation data parquet file containing columns 'user_id', 'age', 'movies', 'ratings', 'genres."
  )
  parser.add_argument(
    "--batch_size_per_replica", type=int, default=4,
    help="batch size"
  )
  parser.add_argument(
    "--num_epochs", type=int, default=1,
    help="number of training epochs"
  )
  parser.add_argument(
    "--lora_rank", type=int, default=4,
    help="LoRA rank"
  )
  parser.add_argument(
    "--lora_alpha", type=int, default=16,
    help="LoRA alpha"
  )
  parser.add_argument(
    "--lora_dropout", type=float, default=0.05,
    help="LoRA drop-out rate"
  )
  
  parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Optimizer learning rate."
  )
  
  parser.add_argument(
    "--checkpoint_dir_uri",
    type=str,
    help="a directory to store checkpoint in.  if not provided, checkpoints will not be saved"
  )
  
  parser.add_argument(
    "--model_save_dir_uri",
    type=str,
    help="a directory to save the model to"
  )
  parser.add_argument(
    "--validation_freq",
    type=int, default=5,
    help="the number of batches in between each validation run"
  )
  
  parser.add_argument(
    '--metrics',
    nargs='+',  # one or more arguments
    help='A list of metrics (e.g., "ndcg@10" "map" "mrr")'
  )
  
  parser.add_argument(
    "--logs_dir_uri", type=str, help="directory to store logs such as metrics")
  
  # 2. Critical: Handling the Rank argument
  # PyTorch < 2.0 used --local_rank (underscore); >= 2.0 prefers --local-rank (dash).
  # It's best practice to accept both for maximum compatibility.
  parser.add_argument(
    "--local-rank",
    "--local_rank",
    type=int,
    default=-1,
    help="Local rank assigned by torchrun. DO NOT SET MANUALLY."
  )
  
  # Use parse_args() normally
  args = parser.parse_args()
  
  # Optional: Read LOCAL_RANK from environment variable if it wasn't set by command line
  # This is a good fallback for fully modern torchrun scripts.
  if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
    
  if args.train_uri is None:
    raise ValueError("train_uri must be provided")
  if args.validation_uri is None:
    raise ValueError("calidation_uri must be provided")
  if args.model_save_dir_uri is None:
    raise ValueError("model_save_dir_uri must be provided")
  if args.logs_dir_uri is None:
    raise ValueError("logs_dir_uri must be provided")
  
  params = {}
  
  params['train_uri'] = args.train_uri
  params['validation_uri'] = args.validation_uri
  params['batch_size_per_replica'] = args.batch_size_per_replica
  params['num_epochs'] = args.num_epochs
  params['learning_rate'] = args.learning_rate
  params['local_rank'] = args.local_rank
  params['model_save_dir_uri'] = args.model_save_dir_uri
  if args.checkpoint_dir_uri is not None:
    params['checkpoint_dir_uri'] = args.checkpoint_dir_uri
  params["logs_dir_uri"] = args.logs_dir_uri
  params['validation_freq'] = args.validation_freq
  params['lora_rank'] = args.lora_rank
  params['lora_alpha'] = args.lora_alpha
  params['lora_dropout'] = args.lora_dropout
  if args.metrics is None:
    params['metrics'] = ['ndcg@5', 'map', 'mrr', 'precision@5', 'recall@5', 'f1@5']
  else:
    params['metrics'] = args.metrics
  
  return params
  
if __name__ == "__main__":
  # This script must be run via the torchrun launcher!
  # e.g.
  # --nproc_per_node=4 means spawn 4 processes on the current machine
  # torchrun --nproc_per_node=4 run_lit5_peft_finetune_distr.py
  #  if 1 CPU w/ 4 cores, this runs  "gloo"
  params = parse_args()
  
  main(params)