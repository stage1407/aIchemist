import torch
import torch.nn as nn
import torch.nn.functional as F
import periodictable
from func.chem_structures import reaction_graph
from scipy.optimize import linear_sum_assignment

def compute_node_loss(node_out, node_target):
    device = node_out.device

    if node_out.shape[0] < node_target.shape[0]:
        padding = torch.zeros((node_target.shape[0] - node_out.shape[0], node_out.shape[1]), device=device)
        node_out = torch.cat([node_out, padding], dim=0)    # Pad prediction
    elif node_out.shape[0] > node_target.shape[0]:
        padding = torch.zeros((node_out.shape[0] - node_target.shape[0], node_target.shape[1]), device=device)
        node_target = torch.cat([node_target,padding],dim=0)    # Pad ground_truth

    """ Seems useless to me
    # Step 2: Match Feature Dimensions (Zero-Padding)
    if node_out.shape[1] < node_target.shape[1]:
        feature_pad = torch.zeros((node_out.shape[0], node_target.shape[1] - node_out.shape[1]), device=device)
        node_out = torch.cat([node_out, feature_pad], dim=1)  # Pad prediction features

    elif node_out.shape[1] > node_target.shape[1]:
        feature_pad = torch.zeros((node_target.shape[0], node_out.shape[1] - node_target.shape[1]), device=device)
        node_target = torch.cat([node_target, feature_pad], dim=1)  # Pad ground truth features
    """

    return F.cross_entropy(node_out, node_target)

def compute_edge_loss(edge_out, edge_target):
    return F.mse_loss(edge_out.squeeze(), edge_target)

def tanimoto_similarity(vec1, vec2):
    """
    Computes Tanimoto similarity between two feature tensors.
    Works for both node and edge features.
    """
    intersection = torch.sum(torch.min(vec1,vec2),dim=0)
    union = torch.sum(vec1 + vec2 - torch.min(vec1,vec2),dim=0)
    return intersection / (union + 1e-8)    # Avoids division by zero

def compute_tanimoto_dist(generated_reaction_graph, ground_truth_graph, node_weight=0.5, edge_weight=0.5):
    graph1, graph2 = generated_reaction_graph, ground_truth_graph
    node_features1 = graph1.x if isinstance(graph1.x, torch.Tensor) else torch.tensor(graph1.x, dtype=torch.float)
    node_features2 = graph2.x if isinstance(graph2.x, torch.Tensor) else torch.tensor(graph2.x, dtype=torch.float)

    edge_features1 = graph1.edge_attr if isinstance(graph1.edge_attr, torch.Tensor) else torch.tensor(graph1.edge_attr,dtype=torch.float)
    edge_features2 = graph2.edge_attr if isinstance(graph2.edge_attr, torch.Tensor) else torch.tensor(graph2.edge_attr,dtype=torch.float)

    node_tanimoto = tanimoto_similarity(node_features1, node_features2).mean()

    edge_tanimoto = tanimoto_similarity(edge_features1, edge_features2).mean()

    reaction_tanimoto = node_weight*node_tanimoto + edge_weight*edge_tanimoto

    return reaction_tanimoto

class Chem:
    def _check_chemical_rules(edge_index, edge_attr, node_features):
        num_nodes = node_features.size(0)
        violations = torch.zeros(num_nodes, dtype=torch.float32)

        # For each atom
        for i in range(num_nodes):
            atomic_number = int(node_features[i,0].item())
            element = periodictable.elements[atomic_number]

            # Get the maximum valence of its element
            max_val = element.max_bond_valence if hasattr(element, 'max_bond_valence') else 0

            # Count the number of electron bonds one atom is sharing
            connected_edges = (edge_index[0] == i).nonzero(as_tuple=True)[0]
            bond_orders = edge_attr[connected_edges]
            total_valence = bond_orders.sum().item()
            if total_valence > max_val:
                violations[i] = total_valence - max_val

        return violations
    
    def compute_chemical_loss(edge_index, edge_attr, node_features):
        violations = Chem._check_chemical_rules(edge_index, edge_attr, node_features)
        chemical_loss = violations.sum()    # Sum of all violations as the penalty

        return chemical_loss
    
    class ChemicalDistanceLoss(nn.Module):
        def __init__(self, weight=1.0, mode= "absolute"):
            """
            :param weight: Gewichtung der chemischen Distanz im Gesamt-Loss.
            :param mode: Vergleichsmodus ("absolute" oder "ratio").
            """
            super(Chem.ChemicalDistanceLoss, self).__init__()
            self.weight = weight
            self.mode = mode
        
        def forward(self, generated_reaction_graph : reaction_graph, ground_truth_reaction_graph : reaction_graph, eps=1e-8):
            # generate the reaction_graph by output of the model.
            cd_generated = generated_reaction_graph.chemical_distance()
            cd_ground_truth = ground_truth_reaction_graph.chemical_distance()

            # ChemDistLoss berechnen
            if self.mode == "absolute":
                chem_dist_loss = abs(cd_generated - cd_ground_truth)
            elif self.mode == "ratio":
                raw_ratio = cd_generated / (cd_ground_truth + eps)
                stabilised_ratio = torch.log(1 + raw_ratio)
                chem_dist_loss = stabilised_ratio
            else:
                raise ValueError("Unsupported mode. Use 'absolute' or 'ratio'.")
            
            return self.weight*chem_dist_loss


    #TODO: Chemical Distance as graph theoretic loss function. (Use func.reaction_graph module)

def compute_graph_loss(prediction, ground_truth, structural_weight=1, chem_rule_weight=1, chem_distance_weight=1, tanimoto_weight=1):
    alpha, beta, gamma, delta = structural_weight, chem_rule_weight, chem_distance_weight, tanimoto_weight
    print(prediction.x.shape)
    print(ground_truth.x.shape)
    node_loss = compute_node_loss(prediction.x, ground_truth.x)
    edge_loss = compute_edge_loss(prediction.edge_attr, ground_truth.edge_attr)
    chemical_loss = Chem.compute_chemical_loss(prediction.edge_index, prediction.edge_attr, prediction.x)
    CD = Chem.ChemicalDistanceLoss(weight=1.0, mode="ratio")
    chemical_distance_loss = CD(prediction, ground_truth)
    tanimoto_distance_loss = compute_tanimoto_dist(prediction, ground_truth)
    total_loss = alpha*(node_loss + edge_loss) + beta*chemical_loss + gamma*chemical_distance_loss + delta*tanimoto_distance_loss
    print("Tanimoto/CD: ", tanimoto_distance_loss, chemical_distance_loss)
    return total_loss

########################## Atom Mapping Loss ##########################

def hungarian_loss(pred_mapping, ground_truth_mapping, educt_graph, product_graph):
    """
    Hungarian loss for atom mapping with element constraints.
    """
    #TODO: Use Node Type from nx.Graph structure, if not in add it also to pred_mapping (One-Hot-Encoding of PTE)
    # Enforce element-wise constraint again in loss
    element_mask = (educt_graph.chem_elem[:, None] == product_graph.chem_elem[None, :]).float()
    
    # Zero out invalid assignments
    pred_mapping = pred_mapping * element_mask

    # Solve Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-pred_mapping.detach().cpu().numpy())  # Maximize assignment
    
    # Convert to one-hot ground truth assignment matrix
    target = torch.zeros_like(pred_mapping)
    target[row_ind, col_ind] = 1  # Optimal assignment
    
    return F.kl_div(pred_mapping.log(), target, reduction='batchmean')