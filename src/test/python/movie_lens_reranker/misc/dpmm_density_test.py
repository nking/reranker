
import unittest
from movie_lens_retrieval.misc.dpmm_density import genereate_pdf_function
import numpy as np
import plotly.express as px

class DensityFuncTest(unittest.TestCase):
  
  def test_generate_density_func(self):
    y = [12, 3, 6, 5]
    func = genereate_pdf_function(y)
    arr_linspace = np.linspace(0, 15, 20).tolist()
    arr_linspace.extend(y)
    arr_linspace = np.array(arr_linspace)
    arr_linspace.sort()
    p = func(arr_linspace)
    self.assertIsNotNone(p)
    self.assertTrue(np.any(p > 0.))
    #will be renderd in browser
    #fig = px.line(x=arr_linspace, y=p, title='pdf')
    #fig.show()
    
if __name__ == '__main__':
  unittest.main()
