"""build graph with edge features"""

import os
from networkx.classes.function import subgraph
import scipy.sparse as sp

import numpy as np
import pandas as pd
import torch as th

import dgl 
from dgl.data.utils import download, extract_archive, get_download_dir
# from refex import extract_refex_feature
# import utils

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


#######################
# Build graph
#######################

class UserItemGraph(object):
    def __init__(self, label_col, user_col, item_col, edge_feature_col, data_path, 
                #  train_data_path, test_data_path, 
                 test_ratio=0.1, valid_ratio=0.2, use_node_features=False):
        
        # # load data from csv 
        # train_df = pd.read_csv(train_data_path)
        # test_df = pd.read_csv(test_data_path)

        # # merge data, replace id
        # # TODO : all test users, items should be exisit in train set?  
        # df = pd.concat([train_df, test_df])

        df = pd.read_csv(data_path)

        df[user_col] = map_newid(df, user_col)
        df[item_col] = map_newid(df, item_col)

        self._num_user = len(df[user_col].unique())
        self._num_item = len(df[item_col].unique())
        self._num_label = len(df[label_col].unique())
        
        df[item_col]+= self._num_user

        # train_len = len(train_df)

        # train_df, test_df = df[:len(train_df)], df[len(train_df):] # TODO

        # # TODO : sort interactions by timestamp
        # train_df, valid_df = train_df[:], train_df[int(len(train_df)*valid_ratio):] 
        
        # return pairs 
        # train_u_idx, train_i_idx = train_df[user_col], train_df[item_col]
        # test_u_idx, test_i_idx = 


        u_idx, i_idx, = df[user_col].to_numpy(), df[item_col].to_numpy(), 
        e_feature = df[edge_feature_col].to_numpy() 
        labels = df[label_col].to_numpy()

        #build main graph 
        sp_mat = sp.coo_matrix((labels,(u_idx, i_idx)), shape=(self._num_user, self._num_user + self._num_item))

        self.bt_graph =dgl.bipartite_from_scipy(sp_mat=sp_mat,
                                                utype='user',
                                                etype='label',
                                                vtype='item',
                                                idtype=th.int32
                                            )

        self.user_indices = th.tensor(u_idx, dtype=th.int32)
        self.item_indices = th.tensor(i_idx, dtype=th.int32)

        self.bt_graph.edata['feature'] = th.tensor(e_feature) #TODO dtype
        self.bt_graph.edata['original_user_idx'] = self.user_indices
        self.bt_graph.edata['original_item_idx'] = self.item_indices
        self.bt_graph.edata['label'] = th.tensor(labels, dtype=th.int32)


        #TODO : node features
        # if use_node_features :
        #     pass

    @property
    def num_label(self):
        return self._num_label

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def get_user_item_pairs(self,):
        pairs = []
        for u, i in zip(self.user_indices, self.item_indices):
            pairs.append((u,i))
        return pairs
    

def map_newid(df, col):
    old_ids = df[col]
    old_id_uniq = old_ids.unique()

    id_dict = {old: new for new, old in enumerate(sorted(old_id_uniq))}
    new_ids = np.array([id_dict[x] for x in old_ids])

    return new_ids


def get_neighbor_nodes_labels(u_nodes, i_nodes, graph, 
                              hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
    
    # 1. neighbor nodes sampling
    dist = 0
    u_nodes = th.unsqueeze(u_nodes, 0)
    i_nodes = th.unsqueeze(i_nodes, 0)
    u_dist, i_dist = th.tensor([0], dtype=th.int32), th.tensor([0], dtype=th.int32)
    u_visited, i_visited = th.unique(u_nodes), th.unique(i_nodes)
    u_fringe, i_fringe = th.unique(u_nodes), th.unique(i_nodes) #인접노드

    for dist in range(1, hop+1):
        # sample neigh alternately
        u_fringe, i_fringe = graph.in_edges(i_fringe)[0], graph.out_edges(u_fringe)[1]
        u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
        i_fringe = th.from_numpy(np.setdiff1d(i_fringe.numpy(), i_visited.numpy()))
        u_visited = th.unique(th.cat([u_visited, u_fringe]))
        i_visited = th.unique(th.cat([i_visited, i_fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(u_fringe))
            u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
            shuffled_idx = th.randperm(len(i_fringe))
            i_fringe = i_fringe[shuffled_idx[:int(sample_ratio*len(i_fringe))]]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
            if max_nodes_per_hop < len(i_fringe):
                shuffled_idx = th.randperm(len(i_fringe))
                i_fringe = i_fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(i_fringe) == 0:
            break
        
        u_nodes = th.cat([u_nodes, u_fringe])
        i_nodes = th.cat([i_nodes, i_fringe])
        u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int32)])
        i_dist = th.cat([i_dist, th.full((len(i_fringe), ), dist, dtype=th.int32)])

    # nodes = th.cat([u_nodes, i_nodes])

    # 2. node labeling ????
    u_node_labels = th.stack([x*2 for x in u_dist])
    i_node_labels = th.stack([x*2+1 for x in i_dist])
    # node_labels = th.cat([u_node_labels, i_node_labels])

    return u_nodes, i_nodes #u_node_labels, i_node_labels


def subgraph_extraction_labeling(u_node_idx, i_node_idx, graph, mode="bipartite", 
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200):

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    u_nodes, i_nodes, = get_neighbor_nodes_labels(u_nodes=u_node_idx, i_nodes=i_node_idx,
                                                  graph=graph, 
                                                  hop=hop, sample_ratio=sample_ratio, max_nodes_per_hop=max_nodes_per_hop
                                                  )

    subgraph = dgl.node_subgraph(graph, {'user':u_nodes, 'item':i_nodes}, store_ids=True)
    # TODO : need node label??
    # set edge mask to zero as to remove links between target nodes in training process
    
    #switch index original
    subgraph.edges = (u_nodes, i_nodes)


    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges())
    su = subgraph.nodes('user')[subgraph.ndata[dgl.NID]['user']==u_node_idx]
    si = subgraph.nodes('item')[subgraph.ndata[dgl.NID]['item']==i_node_idx]
    _, _, target_edges = subgraph.edge_ids([su, si], [si, su], return_uv=True)
    subgraph.edata['edge_mask'][target_edges.type(th.long)] = 0

    return subgraph


#############
# preprocess subgraph data
#############

from dgl.data.utils import save_graphs
from tqdm import tqdm

def build_subgraph_data(graph, path, pairs, batch_size):
    subgraphs = []
    count=0
    for u_idx, i_idx in tqdm(pairs):
        subgraph = subgraph_extraction_labeling(u_idx, i_idx, graph)
        subgraphs.append(subgraph)
        if len(subgraphs)==batch_size:
            save_graphs(f"{path}/subgraph_batch_{count}.bin", subgraphs)
            count += 1
            subgraphs=[]

def run_preprocess(data_path, save_path):

    user_item_graph = UserItemGraph(label_col='rating_10', user_col='user_id', item_col='movie_id', edge_feature_col='sentiment', data_path=data_path)
    pairs = user_item_graph.get_user_item_pairs()
    graph = user_item_graph.bt_graph
    build_subgraph_data(graph, save_path, pairs, 1024)


if __name__=='__main__':
    run_preprocess('l_testset.csv','subgraph_data')



