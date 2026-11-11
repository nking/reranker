import unittest

import numpy as np
from helper import *

from movie_lens_reranker.UserHistory import UserHistory

class TestUserHistory_vec(unittest.TestCase):
    def setUp(self):
        
        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.FULL)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
    
    def test_load_user_history(self):
        max_history = 2048
        ratings_uri_list = [self.ratings_train_liked_uri,
            self.ratings_train_3_uri,
            self.ratings_train_disliked_uri]
        uh = UserHistory(ratings_uri_list=ratings_uri_list,
            max_history=max_history)
        for (m, r, t) in zip(uh.movie_ids, uh.ratings, uh.timestamps):
            i_t = np.where(t == uh.ts_pad_value)
            i_r = np.where(r == uh.pad_value)
            i_m = np.where(m == uh.pad_value)
            np.testing.assert_array_equal(i_t, i_r)
            np.testing.assert_array_equal(i_t, i_m)
    
    def test_user_history(self):
        max_history = 2048
        
        # the full history:
        ratings_uri_list = [self.ratings_train_liked_uri,
            self.ratings_train_3_uri,
            self.ratings_train_disliked_uri]
        
        '''
        some expected examples:
            6040::6888::4::956703932
            6040::6630::5::956703954
            6040::8356::4::956703954
            6040::6684::5::956704257
            ...
            6039::9368::4::956705581
            6039::6940::4::956705636

        choose  timestamps < 956705600
        '''
        uh = UserHistory(ratings_uri_list=ratings_uri_list,
            max_history=max_history)
        
        user_ids = np.array([6040, 6039])
        
        # ==== test scalar timestamps ========
        ts = 956705600
        fixed_length = 200
        
        (_m, _r, _t) = uh.get_history_with_timestamps_before_timestamp(
            user_ids, timestamp=ts, max_hist=fixed_length)
        movie_hist = uh.get_movieids_before_timestamp(user_ids, timestamp=ts,
            max_hist=fixed_length)
        
        # grab the 10th timestamps out of the valid for 2nd request
        ts2 = []
        movie_hist_valid = []
        for i in range(len(_m)):
            m = _m[i]
            r = _r[i]
            t = _t[i]
            i_t = np.where(t == uh.ts_pad_value)
            i_r = np.where(r == uh.pad_value)
            i_m = np.where(m == uh.pad_value)
            np.testing.assert_array_equal(i_t, i_r)
            np.testing.assert_array_equal(i_t, i_m)
            self.assertTrue(i_t[0][
                                -1] == fixed_length - 1)  # the padfilled elements are at end of arrays
            mask = np.where(t < uh.ts_pad_value)
            movie_hist_valid.append((m[mask], r[mask], t[mask]))
            ts2.append(t[mask][10])
            ## assert this because the 2nd request using ts2 checks overlap for underfilled cases
            self.assertTrue(len(mask) < fixed_length)
            print(i, len(mask))
        
        self.assertTrue((len(user_ids), fixed_length) == movie_hist.shape)
        
        # make request using ts2.
        # [.... 63 items]  first 9
        # [... 43 items]   first 9
        (_m2, _r2, _t2) = uh.get_history_with_timestamps_before_timestamp(
            user_ids, timestamp=ts2, max_hist=fixed_length)
        for i in range(len(_m2)):
            m = _m2[i]
            r = _r2[i]
            t = _t2[i]
            i_t = np.where(t == uh.ts_pad_value)
            i_r = np.where(r == uh.pad_value)
            i_m = np.where(m == uh.pad_value)
            np.testing.assert_array_equal(i_t, i_r)
            np.testing.assert_array_equal(i_t, i_m)
            self.assertTrue(i_t[0][
                                -1] == fixed_length - 1)  # the pad filled elements are at end of arrays
            mask = np.where(t < uh.ts_pad_value)
            m, r, t = m[mask], r[mask], t[mask]
            m_compare, r_compare, t_compare = movie_hist_valid[i]
            ## spot check for a known value:
            if i == 0:
                self.assertEqual(m_compare[-1], 6888)
            n = len(mask[0])
            m_compare = m_compare[-n:]
            r_compare = r_compare[-n:]
            t_compare = t_compare[-n:]
            np.testing.assert_array_equal(m, m_compare)
            np.testing.assert_array_equal(r, r_compare)
            np.testing.assert_array_equal(t, t_compare)
        
        # not an indep test, but use dictionary to check timestamps:
        lookup, max_history = uh.build_history_lookup(ratings_uri_list)
        
        for user_id, movies in zip(user_ids, movie_hist):
            self.assertEqual(fixed_length, len(movies))
            count = np.sum(movies != -1)
            
            user_ts, user_movies, user_ratings = lookup[user_id]
            end_idx = np.searchsorted(user_ts, ts)
            self.assertEqual(end_idx, count)
            
            if user_id == 6039:
                test_idx = np.searchsorted(user_ts, 956705636)
                test_movie = user_movies[test_idx]  # 6940
                self.assertEqual(6940, test_movie)
                self.assertTrue(test_movie not in movies)
            
            user_movies = user_movies[:end_idx]
            self.assertEqual(len(user_movies), count)
        
        movie_hist2, ratings_hist2 = uh.get_history_before_timestamp(user_ids,
            timestamp=ts,
            max_hist=fixed_length)
        
        np.testing.assert_array_equal(movie_hist2, movie_hist)
        count = np.sum(movie_hist != -1, axis=1)
        count2 = np.sum(movie_hist2 != -1, axis=1)
        np.testing.assert_array_equal(count, count2)
        count3 = np.sum(ratings_hist2 != -1, axis=1)
        np.testing.assert_array_equal(count, count3)


if __name__ == '__main__':
    unittest.main()
