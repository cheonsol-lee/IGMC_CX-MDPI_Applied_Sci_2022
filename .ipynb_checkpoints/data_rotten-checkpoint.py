"""Rotten Tomato dataset"""

import os
import scipy.sparse as sp

import numpy as np
import pandas as pd
import torch as th

import dgl 
from dgl.data.utils import download, extract_archive, get_download_dir
from refex import extract_refex_feature
import utils

import os
import random
import pickle as pkl
import pandas as pd
import numpy as np
import scipy.sparse as sp

# For automatic dataset downloading
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

path = './raw_data/rotten_tomato/'

class RottenTomato(object):
    def __init__(self, path, testing=False, 
                 test_ratio=0.1, valid_ratio=0.2):
        
        print("Create RottenTomato Class...")
        (
            num_user, num_movie, adj_train, train_labels, train_u_indices, train_v_indices,
            val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
            test_v_indices, class_values
        ) = load_official_trainvaltest_split(testing, None, None, 1.0)

            
        self._num_user = num_user
        self._num_movie = num_movie

        # reindex u and v, v nodes start after u
        train_v_indices += self.num_user
        val_v_indices += self.num_user
        test_v_indices += self.num_user

        self.train_rating_pairs = (th.LongTensor(train_u_indices), th.LongTensor(train_v_indices))
        self.valid_rating_pairs = (th.LongTensor(val_u_indices), th.LongTensor(val_v_indices))
        self.test_rating_pairs = (th.LongTensor(test_u_indices), th.LongTensor(test_v_indices))
        self.train_rating_values = th.FloatTensor(train_labels)
        self.valid_rating_values = th.FloatTensor(val_labels)
        self.test_rating_values = th.FloatTensor(test_labels)

        print("\tTrain rating pairs : {}".format(len(train_labels)))
        print("\tValid rating pairs : {}".format(len(val_labels)))
        print("\tTest rating pairs  : {}".format(len(test_labels)))

        # build dgl graph object, which is homogeneous and bidirectional and contains only training edges
        self.train_graph = dgl.graph((th.cat([self.train_rating_pairs[0], self.train_rating_pairs[1]]), 
                                      th.cat([self.train_rating_pairs[1], self.train_rating_pairs[0]])))
        self.train_graph.edata['etype'] = th.cat([self.train_rating_values, self.train_rating_values]).to(th.long)

    @property
    def num_rating(self):
        return self._rating.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie
    

def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    return data, id_dict, n    
 
    
# dataset 로드
def load_official_trainvaltest_split(testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    dtypes = {'u_nodes': np.int16, 'v_nodes': np.int16, 'ratings': np.float16}
    
    level = '0.5'
    
    # Small data
#     data_train = pd.read_csv(path + 's_trainset.csv', dtype=dtypes)
#     data_test  = pd.read_csv(path + 's_testset.csv', dtype=dtypes)
    
    # Large data
    data_train = pd.read_csv(path + 'l_trainset.csv', dtype=dtypes)
    data_test  = pd.read_csv(path + 'l_testset.csv', dtype=dtypes)
    
    
    data_train.rename(columns={f'user_id':'u_nodes', 'movie_id':'v_nodes', 'rating_'+level:'ratings'}, inplace=True)
    data_test.rename(columns={f'user_id':'u_nodes', 'movie_id':'v_nodes', 'rating_'+level:'ratings'}, inplace=True)
    columns = ['u_nodes','v_nodes','ratings','review_score','sentiment','emotion','review_date','origin_rating_'+level,'review_content']
    
    
    data_train = data_train[columns]
    data_test  = data_test[columns]
    
    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    # 경고무시함
#     for i in range(len(u_nodes)):
#         assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code
    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    # 경고 무시함
#     for i in range(len(ratings)):
#         assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train+num_val]
    idx_nonzero_test = idx_nonzero[num_train+num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
    pairs_nonzero_test = pairs_nonzero[num_train+num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(1234)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])
    
    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))
    
    
    return num_users, num_items, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values