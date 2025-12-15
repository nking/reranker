
"""
any use or redistribution of Gemma or derivatives needs requirements and terms of use as stated:
https://www.kaggle.com/models/google/gemma/license/consent?verifyToken=CfDJ8JvX8PYzZ5dKharEO0H57fGDgQTy_ZErH4RnMuITWlke_r6yS0TOuEzfJZgg8-DkMVLzK5QwQ5SDov5X8WdrAPj3Lj6hVop_2C2RQr9fBVM7B16ynjKlx3z8GoGQ0Lny4zOOrC3Nlluu_8tuiaz2JcMZmBmQLG1ylC6P8eoky6W4o46YMSbyC17o34krG2ECUZntuDOZ1tDyZtQSqdxI0teigQ

"""
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_reranker"))

from helper import *
from load_datasets import hf_dataset_to_torch, custom_seq2seq_collator
import tensorflow as tf
import tf_keras as keras
from peft import LoraConfig, TaskType, get_peft_model
from functools import partial
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, SequentialSampler

from tqdm.auto import tqdm # Used for progress bar

#import keras  #keras3 is not compatible w/ transformeres. so either install tf <= 2.15 or install tf_keras and import it

os.environ["KERAS_BACKEND"] = "tensorflow"
# Set data type policy for mixed precision (optional, but recommended for speed/memory)
keras.mixed_precision.set_global_policy("mixed_float16")

#encoder-decoder architecture trained for sequence generation (text-to-text)
# The v2 models can process up to 100 candidate passages simultaneously.
# Implementations of LiT5-Distill often use a default maximum of 300 tokens for ranking.
MODEL_NAME = "castorini/LiT5-Distill-base-v2"
BATCH_SIZE = 4
res_dir = os.path.join(get_project_dir(), "src/test/resources/data/sorted_1/")
train_path = os.path.join(res_dir, "train-00000-of-00001.parquet")
validation_path = os.path.join(res_dir, "validation-00000-of-00001.parquet")

#from huggingface_hub import login
#login(token=get_hf_token())

#note that the project's code uses instead:
#   tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_path, return_dict=False, legacy=False, use_fast=True)
#   model_class = src.model.FiD which is class FiD(T5ForConditionalGeneration):
#   model = model_class.from_pretrained(opt.model_path).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
  MODEL_NAME,
  torch_dtype=torch.float32,
  #map_location='cpu'
)
#print(f"Model:\n{model}")
#if have troubles w/ reloading or corruption, clear: ~/.cache/huggingface/hub

#from transformers import T5Tokenizer, T5ForConditionalGeneration
#tokenizer2 = T5Tokenizer.from_pretrained(MODEL_NAME)
#model2 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q","v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        init_lora_weights=False,
    )

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

collator_function = partial(custom_seq2seq_collator, tokenizer=tokenizer)

train_dataset, validation_dataset = hf_dataset_to_torch(train_path, validation_path)

train_sampler = SequentialSampler(train_dataset) #a torch util
train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=BATCH_SIZE,
    num_workers=4,
    collate_fn=collator_function
)

validation_sampler = SequentialSampler(train_dataset) #a torch util
validation_dataloader = DataLoader(
    validation_dataset,
    sampler=validation_sampler,
    batch_size=BATCH_SIZE,
    num_workers=4,
    collate_fn=collator_function
)

from torch.optim import AdamW
trainable_params = [
    p for p in lora_model.parameters() if p.requires_grad
]
optimizer = AdamW(trainable_params, lr=5e-5, eps=1e-8)
from transformers import get_linear_schedule_with_warmup
NUM_TRAINING_STEPS = 10000
# Typically 5-10% of total steps
NUM_WARMUP_STEPS = 1000

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=NUM_TRAINING_STEPS
)

device = "cpu"
#TODO: consider change to use early stopping
NUM_EPOCHS = 2

lora_model.to(device)

def train():
  
  total_steps = len(train_dataloader) * NUM_EPOCHS
  
  # 2. Main Training Loop
  for epoch in range(NUM_EPOCHS):
    
    total_train_loss = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    lora_model.train()
    
    for batch_idx, batch in enumerate(progress_bar):
      
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      
      optimizer.zero_grad()
      
      outputs = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
        # The T5 model calculates the Sequence-to-Sequence loss internally
        # when 'labels' are provided, returning it as outputs.loss
      )
      
      loss = outputs.loss
      total_train_loss += loss.item()
      
      #calc grad:
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
      
      optimizer.step()
      scheduler.step()
      
      # Update progress bar with current loss
      progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}',
        'avg_loss': f'{total_train_loss / (batch_idx + 1):.4f}'})
    
    avg_epoch_loss = total_train_loss / len(train_dataloader)
    print(f"\nEpoch {epoch + 1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
    
    eval()
    
    # Save checkpoint here
    #lora_model.save_pretrained(f'./checkpoints/epoch_{epoch+1}')

def eval():
  
  lora_model.eval()
  
  total_val_loss = 0
  all_predictions = []
  
  # Disable gradient tracking for speed and memory
  with torch.no_grad():
    for batch in validation_dataloader:
      
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      
      outputs = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
      )
      
      loss = outputs.loss
      total_val_loss += loss.item()
      
      # TODO: add evaluating metrics (like NDCG or MRR) beyond loss:
      # 1. Use outputs.logits to get raw prediction scores
      # 2. Convert logits to final ranked sequence predictions
      # 3. Store or calculate validation metrics
  
  # Calculate average loss and reset model for potential further training
  avg_val_loss = total_val_loss / len(validation_dataloader)
  lora_model.train()
  
  return avg_val_loss

train()