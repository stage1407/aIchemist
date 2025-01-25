import torch
from sklearn.metrics import accuracy_score, f1_score

def _tanimoto_score(pred_graph, target_graph):
    pred_set = set(pred_graph.edge_index.cpu().numpy().flatten())
    target_set = set(target_graph.edge_index.cpu().numpy().flatten())

    intersection = len(pred_set & target_set)
    union = len(pred_set | target_set)
    if union != 0:
        return intersection / union
    else:
        return 0.0
    
def compute_metrics(node_pred, node_target, edge_pred, edge_target, tanimoto_enabled=False, pred_graph=None, target_graph=None):
    node_pred = node_pred.argmax(dim=1).cpu().numpy()
    node_target = node_target.cpu().numpy()
    edge_pred = edge_pred.argmax(dim=1).cpu().numpy()
    edge_target = edge_target.cpu().numpy()

    metrics = {
        "node_accuracy": accuracy_score(node_target, node_pred),
        "node_f1": f1_score(node_target, node_pred, average="weighted"),
        "edge_accuracy": accuracy_score(edge_target, edge_pred),
        "edge_f1": f1_score(edge_target, edge_pred, average="weighted")
    }

    if tanimoto_enabled and pred_graph is not None and target_graph is not None:
        metrics["tanimoto_score"] = _tanimoto_score(pred_graph, target_graph)

    return metrics


