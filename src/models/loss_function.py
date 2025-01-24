import torch.nn.functional as F

def compute_node_loss(node_out, node_target):
    return F.cross_entropy(node_out, node_target)

def compute_edge_loss(edge_out, edge_target):
    return F.mse_loss(edge_out.squeeze(), edge_target)