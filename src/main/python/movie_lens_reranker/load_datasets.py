"""
input dataset is expected to be parquet files with columns:
['user_id', 'age', 'movies', 'ratings', 'genres']
where:
user_id and age are integers
movies, ratings, and genres are arrays of hard-negative mining values based upon the ratings.
the arrays first elements are values for the positive point, i.e. a rating of "4" or "5" and
the remaining elements are values for the negative points, i.e., ratings of "1", or "2".

each row can be reformatted for a single training query, candidates, and label.
"""
from typing import Tuple, Dict, List, Any
from movie_lens_reranker.prompts.prompt_helper import *
from datasets import load_dataset, arrow_dataset
import torch
from transformers import T5TokenizerFast, BatchEncoding
import yaml
from jinja2 import Template

def hf_dataset_to_torch(train_file_path:str, validation_file_path:str) \
  -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[str, int]]:
  
  #['user_id', 'age', 'movies', 'ratings', 'genres']
  dataset = load_dataset("parquet", data_files={'train': train_file_path,
    'validation': validation_file_path})
  
  dataset_train_torch = DatasetWrapper(dataset['train'])
  dataset_validation_torch = DatasetWrapper(dataset['validation'])
  
  return dataset_train_torch, dataset_validation_torch, dataset.num_rows

class DatasetWrapper(torch.utils.data.Dataset):
  """
  #if need a sliding window approach, see: https://github.com/castorini/LiT5.git
  """
  def __init__(self, hf_dataset: arrow_dataset.Dataset):
    self.data = hf_dataset
    #for the prompts
    yaml_path = get_yaml_prompt_path(Prompt_Type.QUERY)
    with open(yaml_path, 'r') as f:
      config = yaml.safe_load(f)
    raw_template = config['template']
    self.query_template = Template(raw_template)
    yaml_path = get_yaml_prompt_path(Prompt_Type.PASSAGES)
    with open(yaml_path, 'r') as f:
      config = yaml.safe_load(f)
    raw_template = config['template']
    self.passages_template = Template(raw_template)
    
  def __len__(self):
    return len(self.data)
  
  def _format_question(self, example):
    return self.query_template.render(
        user_id=example["user_id"], age=example["age"]
    )
  
  def _format_passages(self, example):
    return [
      self.passages_template.render(i=i+1, movie_id=example['movies'][i], rating=example['ratings'][i], genres=" ".join(example['genres'][i].split("|")))
      for i in range(len(example['movies']))
    ]
  
  def __getitem__(self, idx):
    example = self.data[idx]
    question = self._format_question(example)
    n = len(example['movies'])
    
    if "ratings" in example:
      #labels are the document passage ids in order of decreasing preference
      m_ratings = [(f'[{i+1}]', example['ratings']) for i in range(n)]
      m_ratings = sorted(m_ratings, key=lambda x: x[1], reverse=True)
      labels = [str(x[0]) for x in m_ratings]
      labels = " ".join(labels)
      # for evaluation, we also need these
      query_id = example['user_id']
      #using string keys to match the decoded generated ids in evaluation
      relevance_scores_dict = {str(k):v for k, v in zip(example['movies'], example['ratings'])}
    else:
      labels = None
      query_id = None
      relevance_scores_dict = None
      
    passages = self._format_passages(example)
    passages = " ".join(passages)
    
    return {
      'index': idx,
      'question': question,
      'passages': passages,
      'labels': labels,
      #for evaluation:
      'query_id': query_id,
      'relevance_scores_dict': relevance_scores_dict,
    }
  
def custom_seq2seq_collator(
    features: List[Dict[str, Any]],
    tokenizer: T5TokenizerFast,
    max_input_length: int = 512,
    max_target_length: int = 64,
) -> BatchEncoding:
    """
    Custom collator for LiT5 ranking tasks that concatenates query and passages
    and uses Hugging Face's DataCollatorForSeq2Seq for final tensor conversion.
    """
    
    labels_present = 'labels' in features[0]
    
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
    
    if labels_present:
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
      #for evaluation:
      input_batch['query_id'] = torch.tensor([f['query_id'] for f in features], dtype=torch.long)
      input_batch['relevance_scores_dict'] = [f['relevance_scores_dict'] for f in features]
      
    return input_batch

