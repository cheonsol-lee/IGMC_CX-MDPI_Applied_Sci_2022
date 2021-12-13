"""IGMC modules"""

import math 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dgl.nn.pytorch import RelGraphConv


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    
def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph


class IGMC(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=10, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        # rating
        self.convs_r = th.nn.ModuleList()
        self.convs_r.append(gconv(in_feats, latent_dim[0], 10, num_bases=num_bases, self_loop=True, low_mem=True))
        for i in range(0, len(latent_dim)-1):
            self.convs_r.append(gconv(latent_dim[i], latent_dim[i+1], 10, num_bases=num_bases, self_loop=True, low_mem=True))
        
        # sentiment
        self.convs_s = th.nn.ModuleList()
        self.convs_s.append(gconv(in_feats, latent_dim[0], 5, num_bases=num_bases, self_loop=True, low_mem=True))
        for i in range(0, len(latent_dim)-1):
            self.convs_s.append(gconv(latent_dim[i], latent_dim[i+1], 5, num_bases=num_bases, self_loop=True, low_mem=True))
        
        # emotion
        self.convs_e = th.nn.ModuleList()
        self.convs_e.append(gconv(in_feats, latent_dim[0], 6, num_bases=num_bases, self_loop=True, low_mem=True))
        for i in range(0, len(latent_dim)-1):
            self.convs_e.append(gconv(latent_dim[i], latent_dim[i+1], 6, num_bases=num_bases, self_loop=True, low_mem=True))
        
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs_r:
            size = conv.num_bases * conv.in_feat
            uniform(size, conv.weight)
            uniform(size, conv.w_comp)
            uniform(size, conv.loop_weight)
            uniform(size, conv.h_bias)
            
        for conv in self.convs_s:
            size = conv.num_bases * conv.in_feat
            uniform(size, conv.weight)
            uniform(size, conv.w_comp)
            uniform(size, conv.loop_weight)
            uniform(size, conv.h_bias)
            
        for conv in self.convs_e:
            size = conv.num_bases * conv.in_feat
            uniform(size, conv.weight)
            uniform(size, conv.w_comp)
            uniform(size, conv.loop_weight)
            uniform(size, conv.h_bias)
            
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    # @profile
    def forward(self, block_r, block_s, block_e):
        block_r = edge_drop(block_r, self.edge_dropout, self.training)
        block_s = edge_drop(block_s, self.edge_dropout, self.training)
        block_e = edge_drop(block_e, self.edge_dropout, self.training)

        # rating
        concat_states_r = []
        x = block_r.ndata['x']
        # GCN 메시지 패싱 부분
        for conv in self.convs_r:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block_r, x, block_r.edata['etype'], 
                             norm=block_r.edata['edge_mask'].unsqueeze(1)))
            concat_states_r.append(x)
        concat_states_r = th.cat(concat_states_r, 1)
        
        # sentiment
        concat_states_s = []
        x = block_s.ndata['x']
        # GCN 메시지 패싱 부분
        for conv in self.convs_s:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block_s, x, block_s.edata['etype'], 
                             norm=block_s.edata['edge_mask'].unsqueeze(1)))
            concat_states_s.append(x)
        concat_states_s = th.cat(concat_states_s, 1)
        
        # emotion
        concat_states_e = []
        x = block_e.ndata['x']
        # GCN 메시지 패싱 부분
        for conv in self.convs_e:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block_e, x, block_e.edata['etype'], 
                             norm=block_e.edata['edge_mask'].unsqueeze(1)))
            concat_states_e.append(x)
        concat_states_e = th.cat(concat_states_e, 1)

        
        self.users = block_r.ndata['nlabel'][:, 0] == 1
        self.items = block_r.ndata['nlabel'][:, 1] == 1
        
#         # 내부 임베딩 벡터 노드별 출력
#         emb_r_users = concat_states_r[self.users]
#         emb_r_items = concat_states_r[self.items]
#         emb_s_users = concat_states_s[self.users]
#         emb_s_items = concat_states_s[self.items]
#         emb_e_users = concat_states_e[self.users]
#         emb_e_items = concat_states_e[self.items]
        
#         my_dict = {'rating': [emb_r_users.cpu(), emb_r_items.cpu()],
#                    'sentiment': [emb_s_users.cpu(), emb_s_items.cpu()],
#                    'emotion': [emb_e_users.cpu(), emb_e_items.cpu()]
#                   }
#         emb_df = pd.DataFrame(my_dict)
        
        self.x_r = th.cat([concat_states_r[self.users], concat_states_r[self.items]], 1)
        self.x_s = th.cat([concat_states_s[self.users], concat_states_s[self.items]], 1)
        self.x_e = th.cat([concat_states_e[self.users], concat_states_e[self.items]], 1)

        # aggregation 부분
        self.agg_x = (self.x_r*0.5 + self.x_s*0.25 + self.x_e*0.25) 
        
        self.x = F.relu(self.lin1(self.agg_x))
        self.x = F.dropout(self.x, p=0.5, training=self.training)
        self.x = self.lin2(self.x)
        if self.regression:
            return self.x[:, 0] * self.multiply_by
#             return self.x[:, 0] * self.multiply_by, emb_df
        else:
            assert False
            # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__