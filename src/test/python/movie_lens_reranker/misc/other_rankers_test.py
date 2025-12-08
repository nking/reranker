
import unittest
from movie_lens_retrieval.misc.other_rankers import *
import numpy as np
import polars as pl
import plotly.express as px
import pandas as pd

class BayesianAvgTest(unittest.TestCase):
  
  def setUp(self):
    self.data = {
      'title': ['loved_many', 'loved_few', 'loved_and_hated', 'hated', 'new_unrated',  'hated_few'],
      'movie_id': [1, 2, 3, 4, 5, 6],
      '1': [500,   2,  4000, 800, 0, 40],
      '2': [100,   1,  1000, 100, 0, 5],
      '3': [200,   0,  500,   50, 0, 0],
      '4': [3000,  5,  1000,  20, 0, 0],
      '5': [8000, 40,  4000,  10, 0, 0]
    }
    
  def test_wilson_score(self):
    df = pl.DataFrame(self.data)
    df = df.with_columns(
      pl.sum_horizontal("1", "2", "3", "4", "5").cast(pl.Float64).alias(
        "total_votes")
    )
    df = df.with_columns(
      pl.sum_horizontal("1", "2", "3").cast(pl.Float64).alias(
        "total_pos_votes")
    )
    df_ranked = wilson_score_df(df, 'total_pos_votes', 'total_votes')
    print(df_ranked)
    
    s1 = wilson_score(5, 5)
    s2 = wilson_score(95, 100)
    self.assertLess(s1, s2)
  
  def test_thomspon(self):
    df = pd.DataFrame(self.data)
    df_ranked = thompson_sampling_with_dirichlet_prior(df, seed=None)
    print(df_ranked)
    df_ranked = thompson_sampling_with_dirichlet_prior(df, seed=None)
    print(df_ranked)
    
    
if __name__ == '__main__':
  unittest.main()
