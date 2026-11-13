import os
from unittest import TestCase

from helper import get_data_dir
from movie_lens_reranker.util import *

class UtilTest(TestCase):
    
    def test_read_movie_tiers(self):
        data_dir = get_data_dir()
        
        movie_tiers, movie_offset, num_catalog_movies = read_movie_tiers(
            os.path.join(data_dir, "movie_tiers.json"))
        
        self.assertEqual(movie_tiers.shape[0], num_catalog_movies + 1)
        self.assertTrue(isinstance(movie_tiers[0], np.int32))
    
