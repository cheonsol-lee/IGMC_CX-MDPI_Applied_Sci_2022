import random
from collections import namedtuple

import numpy as np
import torch as th
import dgl

from utils import subgraph_extraction_labeling

class RottenTomatoDataset(th.utils.data.Dataset):
    def __init__(self, links, g_labels, graph, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
        self.links = links
        self.g_labels = g_labels
        self.graph = graph 

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.links[0])

    def __getitem__(self, idx):
        u, v = self.links[0][idx], self.links[1][idx]
        g_label = self.g_labels[idx]

        subgraph = subgraph_extraction_labeling(
            (u, v), self.graph, 
            hop=self.hop, sample_ratio=self.sample_ratio, max_nodes_per_hop=self.max_nodes_per_hop)

        return subgraph, g_label

def collate_rotten_tomato(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label



class MultiRottenTomatoDataset(th.utils.data.Dataset):
    def __init__(self, links, g_labels, graph, 
                hop, sample_ratio, max_nodes_per_hop):
        # 리스트로 입력받음
        self.links = links
        self.g_labels = g_labels
        self.graph = graph 

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.links[0][0])

    def __getitem__(self, idx):
        # rating
        u_r, v_r = self.links[0][0][idx], self.links[0][1][idx]
        g_label_r = self.g_labels[0][idx]

        subgraph_r = subgraph_extraction_labeling(
            (u_r, v_r), self.graph[0], 
            hop=self.hop[0], sample_ratio=self.sample_ratio[0], max_nodes_per_hop=self.max_nodes_per_hop[0])
        
        # sentiment
        u_s, v_s = self.links[1][0][idx], self.links[1][1][idx]
        g_label_s = self.g_labels[1][idx]
        
        subgraph_s = subgraph_extraction_labeling(
            (u_s, v_s), self.graph[1], 
            hop=self.hop[1], sample_ratio=self.sample_ratio[1], max_nodes_per_hop=self.max_nodes_per_hop[1])
        
        # emotion
        u_e, v_e = self.links[2][0][idx], self.links[2][1][idx]
        g_label_e = self.g_labels[2][idx]
        
        subgraph_e = subgraph_extraction_labeling(
            (u_e, v_e), self.graph[2], 
            hop=self.hop[2], sample_ratio=self.sample_ratio[2], max_nodes_per_hop=self.max_nodes_per_hop[2])
        
        
        # 통합
        g_label = [g_label_r, g_label_s, g_label_e]
        subgraph = [subgraph_r, subgraph_s, subgraph_e]

        return subgraph, g_label

# data : tuple 형태(subgraph, g_label)
def multi_collate_rotten_tomato(data):
    r_data    = list()
    s_data = list()
    e_data   = list()

    # batch 샘플 순서
    for i in range(len(data)):
        r_data.append((data[i][0][0], data[i][1][0])) # rating
        s_data.append((data[i][0][1], data[i][1][1])) # sentiment
        e_data.append((data[i][0][2], data[i][1][2])) # emotion

   # rating
    g_list_r, label_list_r = map(list, zip(*r_data))
    g_r = dgl.batch(g_list_r)
    g_label_r = th.stack(label_list_r)
    
    # sentiment
    g_list_s, label_list_s = map(list, zip(*s_data))
    g_s = dgl.batch(g_list_s)
    g_label_s = th.stack(label_list_s)
    
    # emotion
    g_list_e, label_list_e = map(list, zip(*e_data))
    g_e = dgl.batch(g_list_e)
    g_label_e = th.stack(label_list_e)
    
    # 리스트로 출력
    g = [g_r, g_s, g_e]
    g_label = [g_label_r, g_label_s, g_label_e]
    
    return g, g_label



