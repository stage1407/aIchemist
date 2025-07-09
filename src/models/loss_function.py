import torch
import torch.nn as nn
import torch.nn.functional as F
import periodictable
from func.chem_structures import reaction_graph
from scipy.optimize import linear_sum_assignment

def compute_node_loss(node_out, node_target):
    """ Computes the node loss between predicted and target node features. 
    This function handles cases where the number of nodes in the output and target differ by padding them with zeros. 
    It also ensures that the feature dimensions match by padding if necessary.
    This is useful for models that predict node features in a graph, such as in graph neural networks.
    Args:
        node_out (torch.Tensor): Predicted node features.
        node_target (torch.Tensor): Target node features.   
    Returns:
        torch.Tensor: Computed node loss.
    """
    device = node_out.device

    if node_out.shape[0] < node_target.shape[0]:
        padding = torch.zeros((node_target.shape[0] - node_out.shape[0], node_out.shape[1]), device=device)
        node_out = torch.cat([node_out, padding], dim=0)    # Pad prediction
    elif node_out.shape[0] > node_target.shape[0]:
        padding = torch.zeros((node_out.shape[0] - node_target.shape[0], node_target.shape[1]), device=device)
        node_target = torch.cat([node_target,padding],dim=0)    # Pad ground_truth

    return F.cross_entropy(node_out, node_target)

def compute_edge_loss(edge_out, edge_target):
    return F.mse_loss(edge_out.squeeze(), edge_target)

def tanimoto_similarity(vec1, vec2):
    """
    Computes Tanimoto similarity between two feature tensors.
    Works for both node and edge features.
    Args:
        vec1 (torch.Tensor): First feature tensor.
        vec2 (torch.Tensor): Second feature tensor.
    Returns:
        torch.Tensor: Tanimoto similarity score.
    The Tanimoto similarity is computed as the ratio of the intersection to the union of the two feature vectors.
    It is defined as:
        Tanimoto(vec1, vec2) = |vec1 ∩ vec2| / |vec1 ∪ vec2|
    where |vec1 ∩ vec2| is the sum of the minimum values of corresponding elements in vec1 and vec2,
    and |vec1 ∪ vec2| is the sum of the maximum values of corresponding elements in vec1 and vec2.
    This function avoids division by zero by adding a small constant (1e-8) to the denominator.
    The Tanimoto similarity is commonly used in cheminformatics to compare chemical structures or molecular fingerprints.
    It ranges from 0 (no similarity) to 1 (perfect similarity).
    Example:
        >>> vec1 = torch.tensor([1, 0, 1, 0])
        >>> vec2 = torch.tensor([1, 1, 0, 0])
        >>> tanimoto_similarity(vec1, vec2)
        tensor(0.3333)
    This example shows how to compute the Tanimoto similarity between two binary vectors.
    The result indicates that the two vectors have a Tanimoto similarity of approximately 0.3333, meaning they share some common features but also have distinct elements.
    Note:
        The input tensors vec1 and vec2 should be of the same shape. If they are not, the function will raise an error.
        The function assumes that the input tensors are non-negative, as Tanimoto similarity is typically used with non-negative feature vectors
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
    """ Class for chemical rules and losses. """
    def _check_chemical_rules(edge_index, edge_attr, node_features):
        """ Checks chemical rules for valence violations in a reaction graph.
        Args:
            edge_index (torch.Tensor): Edge indices of the graph.
            edge_attr (torch.Tensor): Edge attributes (bond orders).
            node_features (torch.Tensor): Node features (atomic numbers).
        Returns:
            torch.Tensor: A tensor indicating the number of valence violations for each atom.
        This function iterates over each atom in the graph, retrieves its atomic number, and checks
        the maximum valence for that element using the `periodictable` library.
        It then counts the total bond orders for each atom by summing the edge attributes of connected edges.
        If the total bond order exceeds the maximum valence, it records a violation for that atom.
        The function returns a tensor where each element corresponds to an atom in the graph,
        and the value indicates the number of valence violations for that atom.
        """
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
        """ Computes the chemical loss based on valence violations in a reaction graph.
        Args:
            edge_index (torch.Tensor): Edge indices of the graph.
            edge_attr (torch.Tensor): Edge attributes (bond orders).
            node_features (torch.Tensor): Node features (atomic numbers).
        Returns:
            torch.Tensor: A tensor representing the total chemical loss due to valence violations.
        This function uses the `_check_chemical_rules` method to identify valence violations in the graph.
        It sums the violations to compute the total chemical loss, which can be used as a penalty in training.
        The chemical loss is useful for ensuring that the generated reaction graphs adhere to chemical rules,
        particularly regarding the valence of atoms.
        """
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
        
        def forward(self, generated_reaction_graph : reaction_graph, ground_truth_reaction_graph : reaction_graph, eps=1e-8) -> torch.Tensor:
            """
            Berechnet den ChemDistLoss zwischen dem generierten und dem Ground-Truth-Reaktionsgraphen.
            :param generated_reaction_graph: Reaktionsgraph, der von dem Modell generiert wurde.
            :param ground_truth_reaction_graph: Ground-Truth-Reaktionsgraph.
            :param educts: Liste der Educts (Edukte) des Reaktionsgraphen.
            :param products: Liste der Produkte des Reaktionsgraphen.
            :param eps: Kleiner Wert zur Vermeidung von Division durch Null.
            :return: ChemDistLoss-Wert.
            """
            # generate the reaction_graph by output of the model.
            cd_generated = generated_reaction_graph.chemical_distance()
            cd_ground_truth = ground_truth_reaction_graph.chemical_distance()

            # compute the loss using ChemDistLoss
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

def compute_graph_loss(prediction, ground_truth, educts, products, structural_weight=1, chem_rule_weight=1, chem_distance_weight=1, tanimoto_weight=1):
    """ Computes the total graph loss based on structural, chemical rule, chemical distance, and Tanimoto similarity losses. 
    Args:
        prediction (reaction_graph): The predicted reaction graph.
        ground_truth (reaction_graph): The ground truth reaction graph.
        educts (list): List of educts in the reaction.
        products (list): List of products in the reaction.
        structural_weight (float): Weight for structural loss components (node and edge loss).
        chem_rule_weight (float): Weight for chemical rule loss.
        chem_distance_weight (float): Weight for chemical distance loss.
        tanimoto_weight (float): Weight for Tanimoto similarity loss.
    Returns:
        float: The total graph loss, which is a weighted sum of the individual losses.
    """
    #TODO: Add Tanimoto Distance as graph theoretic loss function.
    #TODO: Add Chemical Distance as graph theoretic loss function.
    alpha, beta, gamma, delta = structural_weight, chem_rule_weight, chem_distance_weight, tanimoto_weight
    print(prediction.x.shape)
    print(ground_truth.x.shape)
    node_loss = compute_node_loss(prediction.x, ground_truth.x) if structural_weight != 0 else 0
    edge_loss = compute_edge_loss(prediction.edge_attr, ground_truth.edge_attr) if structural_weight != 0 else 0
    chemical_loss = Chem.compute_chemical_loss(prediction.edge_index, prediction.edge_attr, prediction.x) if chem_rule_weight != 0 else 0
    CD = Chem.ChemicalDistanceLoss(weight=1.0, mode="ratio")
    chemical_distance_loss = CD(prediction, ground_truth, educts, products) if chem_distance_weight != 0 else 0
    tanimoto_distance_loss = compute_tanimoto_dist(prediction, ground_truth) if tanimoto_weight != 0 else 0
    total_loss = alpha*(node_loss + edge_loss) + beta*chemical_loss + gamma*chemical_distance_loss + delta*tanimoto_distance_loss
    print("Tanimoto/CD: ", tanimoto_distance_loss, chemical_distance_loss)
    return total_loss

########################## Atom Mapping Loss ##########################

def hungarian_loss(pred_mapping, ground_truth_mapping, educt_graph, product_graph):
    """
    Hungarian loss for atom mapping with element constraints.
    Args:
        pred_mapping (torch.Tensor): Predicted atom mapping matrix (shape: [num_educt_atoms, num_product_atoms]).
        ground_truth_mapping (torch.Tensor): Ground truth atom mapping matrix (shape: [num_educt_atoms, num_product_atoms]).
        educt_graph (reaction_graph): Educt graph containing chemical elements.
        product_graph (reaction_graph): Product graph containing chemical elements.
    Returns:
        torch.Tensor: Computed Hungarian loss.
    This function computes the Hungarian loss for atom mapping between educt and product graphs.
    It uses the Hungarian algorithm to find the optimal assignment of atoms from educts to products
    while enforcing element-wise constraints based on the chemical elements present in the graphs.
    The loss is computed as the Kullback-Leibler divergence between the predicted mapping and the target mapping.
    The element-wise constraint ensures that only valid assignments are considered based on the chemical elements of the atoms.
    The function uses the `linear_sum_assignment` from `scipy.optimize` to solve the assignment problem.
    The predicted mapping is adjusted by applying a mask based on the chemical elements of the educt and product graphs.
    The target mapping is created as a one-hot encoded matrix based on the optimal assignment found by the Hungarian algorithm.
    The final loss is computed as the Kullback-Leibler divergence between the predicted mapping and the target mapping.
    This loss function is useful in scenarios where atom mapping is required, such as in reaction prediction tasks,
    where the goal is to align atoms from reactants to products while respecting chemical constraints.
    The function assumes that the `pred_mapping` and `ground_truth_mapping` are provided as matrices where each row corresponds
    to an atom in the educt graph and each column corresponds to an atom in the product graph.
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