#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 22:58
# @Author  : zhangbc0315@outlook.com
# @File    : gnn_base_model.py
# @Software: PyCharm

import torch
import torch_scatter
from torch.nn import Module, Sequential, Linear, ReLU, ELU, Dropout

# ELU for selector

def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    # Ensure src and index have the same shape
    if src.shape != index.shape:
        raise ValueError("The shapes of `src` and `index` must be the same.")
    
    # Determine the output shape
    output_shape = list(src.shape)
    if dim_size is not None:
        output_shape[dim] = dim_size
    elif out is not None:
        output_shape = out.shape
    else:
        output_shape[dim] = int(index.max()) + 1
    
    # Initialize output tensor if not provided
    if out is None:
        out = torch.zeros(*output_shape, dtype=src.dtype, device=src.device)
    
    # Expand index to match src shape
    expanded_index = index.unsqueeze(-1).expand_as(src)
    
    # Use scatter_add_ to accumulate values at specified indices
    out.scatter_add_(dim, expanded_index, src)
    
    return out


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    out = scatter_add(src, index,dim, out, dim_size, fill_value)
    count = scatter_add(torch.ones_like(src), index, dim, None, out.size(dim))
    return out / count.clamp(min=1)

class EdgeModel(Module):
    def __init__(self, num_node_features, num_edge_features, out_features):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Sequential(Linear(num_node_features + num_node_features + num_edge_features, 128),
                                   ELU(),
                                   # Dropout(0.5),
                                   Linear(128, out_features))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(Module):
    def __init__(self, num_node_features, num_edge_features_out, out_features):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Sequential(Linear(num_node_features + num_edge_features_out, 256),
                                     ELU(),
                                     # Dropout(0.5),
                                     Linear(256, 256))
        self.node_mlp_2 = Sequential(Linear(num_node_features + 256, 256),
                                     ELU(),
                                     # Dropout(0.5),
                                     Linear(256, out_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(Module):
    def __init__(self, num_node_features, num_global_features, out_channels):
        super(GlobalModel, self).__init__()
        self.global_mlp = Sequential(Linear(num_global_features + num_node_features, 256),
                                     ELU(),
                                     # Dropout(0.3),
                                     Linear(256, out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        if u is None:
            out = scatter_mean(x, batch, dim=0)
        else:
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)

if __name__ == "__main__":
    pass