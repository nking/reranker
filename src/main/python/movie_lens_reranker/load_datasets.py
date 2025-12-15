"""
input dataset is expected to be parquet files with columns:
['user_id', 'age', 'movies', 'ratings', 'genres']
where:
user_id and age are integers
movies, ratings, and genres are arrays of hard-negative mining values based upon the ratings.
the arrays first elements are values for the positive point, i.e. a rating of "4" or "5" and
the remaining elements are values for the negative points,, i.e., ratings of "1", or "2".

each row can be reformatted for a single training query, candidates, and label.
"""
from typing import Tuple, Dict, List, Any

from datasets import load_dataset, arrow_dataset
import torch
import numpy as np
from transformers import DataCollatorForSeq2Seq, T5TokenizerFast

def hf_dataset_to_torch(train_file_path:str, validation_file_path:str) \
  -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
  
  #['user_id', 'age', 'movies', 'ratings', 'genres']
  dataset = load_dataset("parquet", data_files={'train': train_file_path,
    'validation': validation_file_path})
  
  dataset_train_torch = DatasetWrapper(dataset['train'])
  dataset_validation_torch = DatasetWrapper(dataset['validation'])
  
  return dataset_train_torch, dataset_validation_torch

class DatasetWrapper(torch.utils.data.Dataset):
  def __init__(self, hf_dataset: arrow_dataset.Dataset):
    self.data = hf_dataset
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    example = self.data[idx]
    question = f"Based on the past data for user u_{example['user_id']} rank the following movies for them:"
    n = len(example['movies'])
    #labels are the document passage ids in order of decreasing preference
    m_ratings = [(example['movies'][i], example['ratings']) for i in range(n)]
    m_ratings = sorted(m_ratings, key=lambda x: x[1], reverse=True)
    labels = [str(x[0]) for x in m_ratings]
    passages = [f"movie {example['movies'][i]} with rating {example['ratings'][i]}" for i in range(n)]
    passages = " ".join(passages)
    labels = " ".join(labels)
    return {
      'index': idx,
      'question': question,
      'passages': passages,
      'labels': labels,
    }
  
def custom_seq2seq_collator(
    features: List[Dict[str, Any]],
    tokenizer: T5TokenizerFast,
    max_input_length: int = 512,
    max_target_length: int = 64
) -> Dict[str, torch.Tensor]:
    """
    Custom collator for LiT5 ranking tasks that concatenates query and passages
    and uses Hugging Face's DataCollatorForSeq2Seq for final tensor conversion.
    """
    
    input_texts = [
        f"{feature['question']} {feature['passages']}"
        for feature in features
    ]
    
    input_batch = tokenizer(
        input_texts,
        padding="longest",
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )

    target_texts = [feature['labels'] for feature in features]
    
    label_batch = tokenizer(
        target_texts,
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # The T5/LiT5 model expects the target IDs to be called 'labels'
    input_batch['labels'] = label_batch['input_ids']
    
    # T5 models expect padding in labels to be ignored by loss function (-100)
    # The DataCollatorForSeq2Seq typically does this, but we implement it here
    # for full control over manually tokenized labels.
    input_batch['labels'][input_batch['labels'] == tokenizer.pad_token_id] = -100

    return input_batch

#adapted from LiT5_files/data.py
class SlidingWindowWrapper(torch.utils.data.Dataset):
  def __init__(self, hf_dataset: arrow_dataset.Dataset):
    self.data = hf_dataset
    self.n_passages = None
    self.start_pos = 0
  
  def __len__(self):
    return len(self.data)
  
  def set_window(self, n_passages=None, start_pos=0):
    self.n_passages = n_passages
    self.start_pos = start_pos
  
  def __getitem__(self, idx):
    """
    returns a dictionary of 'index', 'question', 'passages', 'labels'
    """
    item = self.data[idx]
    # Perform custom logic (e.g., normalization, custom casting)
    #return {
    #  "input": torch.tensor(item["input_ids"]),
    #  "input": torch.tensor(item["movie_id"]),
    #  "label": torch.tensor(item["rating"])
    #}
    """
    Example Queries:
      Sequential Prompting:
         User u_id is searching for movies to watch based on past ratings
      Recency-Focused Prompting:
         User u_id watched the following movies...
         Note that my most recently watched movie is 'Dead Presidents'.
         Rank the following candidates, prioritizing my most recent taste".
      Identity/Persona Prompting:
         You are a fan of 90s sci-fi and Christopher Nolan.  Rank this list of movies:
      Keyword Summarization:
        Based on these history interactions, rank the following items by importance:
        
    Example Passages (candidate lists and metadata):
      Candidate movie list:
        ['Postman, The (1997)', 'Liar Liar (1997)', 'Contact (1997)', 'Welcome To Sarajevo (1997)', 'I Know What You Did Last Summer (1997)']
      Item Profiles (metadata):
        genres, descriptions, case and crew, popularity metrics
        
    Ground Truth (the order):
      the document passage ids reordered by decreasing preference.
    """
    example = self.data[idx]
    question = f"Based on the past data for user u_{example['user_id']} rank the following movies for them:"
    
    #TODO: this could be vectorized:
    passages = []
    #labels are the document passage ids in order of decreasing preference
    m_ratings = []
    for i in range(self.start_pos, self.start_pos + self.n_passages):
      SEP = " " if i < (self.start_pos + self.n_passages - 1) else ""
      if i < len(example['movies']):
        passages.append(f"movie m_{example['movies'][i]} with rating {example['ratings'][i]}{SEP}")
        m_ratings.append((example['movies'][i], example['ratings'][i]))
      else :
        passages.append(f"movie m_{-1} with rating -1{SEP}")
        m_ratings.append((-1, -100))
    passages = np.array(passages)
    m_ratings = sorted(m_ratings, key=lambda x: x[1])
    labals = [x[0] for x in m_ratings]
   
    return {
      'index': idx,
      'question': question,
      'passages': passages,
      'labels': labals,
    }
