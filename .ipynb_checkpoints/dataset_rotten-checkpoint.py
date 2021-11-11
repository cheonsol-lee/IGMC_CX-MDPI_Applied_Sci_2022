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

