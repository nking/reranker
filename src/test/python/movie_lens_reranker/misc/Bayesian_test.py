
import unittest
from movie_lens_retrieval.misc.Bayesian import BayesianAvg, BayesianShrinkageEstimator
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
    
  def testUnweighted(self):
    m = 1
    b = BayesianAvg(pd.DataFrame(self.data), m=m)
    t = b.get_top(10)
    print(f"{t}")
    b = BayesianAvg(pl.DataFrame(self.data), m=m)
    t = b.get_top(10)
    print(f"\n{t}")
    
  def testWeighted(self):
    b = BayesianShrinkageEstimator(pd.DataFrame(self.data))
    t = b.get_top(10)
    print(f"{t}")
    b = BayesianShrinkageEstimator(pl.DataFrame(self.data))
    t = b.get_top(10)
    print(f"\n{t}")
    
if __name__ == '__main__':
  unittest.main()
