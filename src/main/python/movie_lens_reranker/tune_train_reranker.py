from typing import Dict, Any, Tuple

from transformers import AutoTokenizer
import torch
import os
import argparse
from tqdm.auto import tqdm # Used for progress bar
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM
import torch.distributed as dist

from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForSeq2SeqLM
from functools import partial

from movie_lens_reranker.load_datasets import hf_dataset_to_torch, custom_seq2seq_collator

import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_model_lit5(model_params: Dict[str, Any]) -> Tuple[
  AutoTokenizer, AutoPeftModelForSeq2SeqLM, partial]:
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
  
  collator_function = partial(custom_seq2seq_collator,
    tokenizer=tokenizer)
  
  return tokenizer, lora_model, collator_function


def build_dataloaders(data_params: Dict[str, Any],
  collator_function: partial, device) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
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
  
  train_dataset, validation_dataset, num_rows_dict = hf_dataset_to_torch(
    data_params["train_uri"],
    data_params["validation_uri"])
  
  if device.type == 'cuda':
    train_sampler = DistributedSampler(
      train_dataset,
      num_replicas=data_params["num_replicas"],
      rank=data_params["rank"],
      shuffle=True,
      # set_pin=True
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

def train(train_dataloader, validation_dataloader, tokenizer, model,
  device, optimizer, scheduler, run_params: Dict[str, Any]):
  """
  
  run_params["GLOBAL_BATCH_SIZE"] = GLOBAL_BATCH_SIZE
  run_params['TRAIN_STEPS_PER_EPOCH'] = TRAIN_STEPS_PER_EPOCH
  run_params['VALIDATIOM_STEPS_PER_EPOCH'] = VALIDATIOM_STEPS_PER_EPOCH
  run_params['NUM_TRAINING_STEPS'] = NUM_TRAINING_STEPS
  
  """
  total_steps = run_params["NUM_TRAINING_STEPS"]
  # TODO: add logging to run_params['logs_dir']
  
  model.to(device)
  
  rank = dist.get_rank() if dist.is_initialized() else 0
  
  for epoch in range(run_params["num_epochs"]):
    
    total_train_loss = 0
    global_total_train_loss = 0
    
    if hasattr(train_dataloader, 'sampler') and isinstance(
      train_dataloader.sampler, DistributedSampler):
      train_dataloader.sampler.set_epoch(epoch)
    
    progress_bar = tqdm(train_dataloader,
      desc=f"Epoch {epoch + 1}/{run_params['num_epochs']}")
    
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
      
      loss = outputs.loss
      total_train_loss += loss.item()
      
      loss.backward()
      
      dist_loss = loss.clone().detach()
      
      if epoch == 0 and batch_idx == 0:
        trainable_grads = []
        frozen_grads = []
        for name, param in model.module.named_parameters():
          if param.requires_grad:
            if param.grad is not None:
              trainable_grads.append(param.grad.norm().item())
            else:
              print(
                f"❌ WARNING: Trainable param {name} has NO gradient!")
          else:
            if param.grad is not None:
              print(
                f"❌ WARNING (Rank {rank}): Frozen param {name} HAS a gradient (should be None)!")
        if trainable_grads:
          avg_grad = sum(trainable_grads) / len(trainable_grads)
          print(
            f"✅ Success (Rank {rank}): LoRA adapters are receiving gradients. Avg Norm: {avg_grad:.6f}")
        print("-------------------------------------------\n")
      
      if dist.is_initialized():
        dist.all_reduce(dist_loss, op=dist.ReduceOp.SUM)
        global_total_train_loss += (
            dist_loss.item() / dist.get_world_size())
      else:
        global_total_train_loss += dist_loss.item()
      
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
      optimizer.step()
      scheduler.step()
      
      # Update progress bar with current loss
      progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}',
        'avg_loss': f'{total_train_loss / (batch_idx + 1):.4f}',
        'global_avg_loss': f'{global_total_train_loss / (batch_idx + 1):.4f}'}
      )
      
      if ((batch_idx + 1) % run_params['validation_freq']) == 0:
        avg_val_loss = eval(validation_dataloader, tokenizer, model,
          device, run_params['metrics'])
        print(f"\nAverage Validation Loss: {avg_val_loss:.4f}")
        model.train()
    
    avg_epoch_loss = total_train_loss / len(train_dataloader)
    print(
      f"\nEpoch {epoch + 1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
    
    if "checkpoint_dir_uri" in run_params:
      path = f"./{run_params['checkpoint_dir_uri']}/epoch_{epoch}"
      save_peft_model_checkpoint(model, path, run_params['rank'])
  
  if run_params['rank'] == 0:
    save_peft_model_checkpoint(model, run_params['model_save_dir_uri'],
      run_params['rank'])
    tokenizer.save_pretrained(run_params['tokenizer_save_dir_uri'])

def eval(validation_dataloader, tokenizer, model, device, metrics):
  model.eval()
  model.to(device)
  
  is_distributed = dist.is_initialized()
  rank = dist.get_rank() if is_distributed else 0
  world_size = dist.get_world_size() if is_distributed else 1
  
  local_total_val_loss = 0
  local_qrels_data = defaultdict(dict)
  local_run_data = defaultdict(dict)
  local_sample_count = 0
  
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
      
      local_total_val_loss += outputs.loss.item()
      local_sample_count += 1
      
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
        local_qrels_data[query_id_str].update(qrels_dicts_list[i])
        
        predicted_ranking_str = tokenizer.decode(
          predicted_ids_tensors[i],
          skip_special_tokens=True,
          clean_up_tokenization_spaces=True
        )
        predicted_doc_ids = predicted_ranking_str.split()
        
        run_scores_for_query = {doc_id: 1.0 / (r + 1) for r, doc_id in
          enumerate(predicted_doc_ids)}
        local_run_data[query_id_str].update(run_scores_for_query)
  
  if is_distributed:
    gathered_qrels = [None] * world_size
    gathered_run = [None] * world_size
    gathered_losses = [None] * world_size
    gathered_counts = [None] * world_size
    
    dist.all_gather_object(gathered_qrels, dict(local_qrels_data))
    dist.all_gather_object(gathered_run, dict(local_run_data))
    dist.all_gather_object(gathered_losses, local_total_val_loss)
    dist.all_gather_object(gathered_counts, local_sample_count)
    
    # Merge results on all ranks (or just rank 0)
    global_qrels_data = {}
    global_run_data = {}
    total_global_loss = sum(gathered_losses)
    total_global_count = sum(gathered_counts)
    
    for d in gathered_qrels:
      global_qrels_data.update(d)
    for d in gathered_run:
      global_run_data.update(d)
    
    avg_val_loss = total_global_loss / total_global_count
  else:
    global_qrels_data = local_qrels_data
    global_run_data = local_run_data
    avg_val_loss = local_total_val_loss / local_sample_count
  
  # --- EVALUATION ---
  # Usually, we only calculate/print ranx metrics on rank 0 to avoid redundant logs
  if rank == 0:
    qrels = Qrels(global_qrels_data)
    run = Run(global_run_data, name="LiT5_Distill_v2_Run")
    results = ranx_evaluate(qrels, run, metrics=metrics)
    print(f'Global ranx result={results}')
  
  # Ensure all processes wait for rank 0 to finish printing
  if is_distributed:
    dist.barrier()
  
  return avg_val_loss
  
def save_peft_model_checkpoint(ddp_model, save_directory, global_rank):
  """Saves the PEFT adapter weights on the master process (Rank 0)."""
  
  if global_rank != 0:
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
  
  cwd = os.getcwd()
  
  tokenizer, lora_model, collator_function = build_model_lit5(params)
  
  train_dataloader, validation_dataloader, num_rows_dict = build_dataloaders(params, collator_function=collator_function, device=device)
  
  is_distributed = dist.is_initialized()
  
  lora_model.gradient_checkpointing_enable()
  lora_model.enable_input_require_grads()
  lora_model.config.use_cache = False
  lora_model = lora_model.to(device)
  
  rank = dist.get_rank() if is_distributed else 0
  
  if device.type == 'cuda':
    model = DDP(lora_model, device_ids=[rank], static_graph=True)
  else:
    model = DDP(lora_model, static_graph=True)  # DDP works on CPU using the 'gloo' backend
  
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
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
  
  if dist.is_initialized():
    dist.barrier()
  
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
  
  local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
  if local_world_size > 1:
    total_cores = os.cpu_count()
    threads_per_process = max(1, total_cores // local_world_size)
    if device.type == 'cpu':
      torch.set_num_threads(threads_per_process)
    params["num_workers"] = threads_per_process
  else:
    params["num_workers"] = 2  # Default for single-process
    
  # For hardware division (CPU cores, workers)
  local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
  cpu_threads = os.cpu_count() // local_size
  
  # For mathematical scaling (Learning rate, total batch size)
  global_size = int(os.environ.get("WORLD_SIZE", 1))
  print(f'world_size={world_size}, setting lr from {params["learning_rate"]} to {params["learning_rate"]*global_size}')
  params["learning_rate"] = params["learning_rate"] * global_size
  
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
    "--tokenizer_save_dir_uri",
    type=str,
    help="a directory to save the tokenizer to"
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
  if args.tokenizer_save_dir_uri is None:
    raise ValueError("tokenizer_save_dir_uri must be provided")
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
  params['tokenizer_save_dir_uri'] = args.tokenizer_save_dir_uri
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