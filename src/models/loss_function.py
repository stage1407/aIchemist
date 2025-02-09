import torch
import torch.nn as nn
import torch.nn.functional as F
import periodictable
from func.chem_structures import reaction_graph


def compute_node_loss(node_out, node_target):
    return F.cross_entropy(node_out, node_target)

def compute_edge_loss(edge_out, edge_target):
    return F.mse_loss(edge_out.squeeze(), edge_target)

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
