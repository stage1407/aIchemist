from rdkit import Chem
from rdkit.Chem import rdFMCS
#from rdkit.Chem.rdMolDescriptors import CalcCrippenContribs
from scipy.optimize import linear_sum_assignment
import mol_graph
import reaction_graph
from embedding import feature_filter, properties, get_atom_properties, get_bond_properties

def _derive_feature_vector(node=None, edge=None, subgraph=None, mol=None):
    features = []
    if node:
        feature_definition : dict = feature_filter["node_features"]
        for p in properties["node_features"]:
            if node.GetSymbol() in feature_definition[p]:
                features.append(get_atom_properties(node,p))
    elif edge and mol:
        feature_definition : dict = feature_filter["edge_features"]
        bond : Chem.Bond = edge
        for p in properties["edge_features"]:
            if feature_definition[p]:
                features.append(get_bond_properties(mol,bond,p))
    elif subgraph and not subgraph:    #! Just to skip subgraph features yet!
        #? How to implement subgraphs?
        ("mol_interpretation")
        ("mol_fragment_type")
    else:
        raise AttributeError
    return features

def _derive_isometries():
    #TODO: Geometric isometry (cis/trans), conformation isometry (rotations), diastereomerie (non-isometric isomeres, enantiomeres, diastereomeres,...)
    pass



    

def compute_chemical_distance(graph1 : mol_graph, graph2 : mol_graph):
    sum = 0
    for e in graph1.edges + graph2.edges:
        sum += e["edge_feature"]["weight"]      #TODO: feature
    max_mcs = None
    max = 1
    for mol1 in graph1.mols:
        for mol2 in graph2.mols:
            mcs : rdFMCS.MCSResult = rdFMCS.FindMCS([mol1,mol2])
            if mcs.numAtoms > max:
                max = mcs.numAtoms
                max_mcs = mcs
    mcs_mol = Chem.MolFromSmarts(max_mcs)
    mcs_sum = 0
    for e in mcs_mol.GetBonds():
        e : Chem.Bond = e
        mcs_sum += e.GetBondTypeAsDouble()
    return sum - 2*mcs_sum

def transform(G : mol_graph, omega : list, G_R : reaction_graph):
    def omega_inv(i):
        for j,p in enumerate(omega):
            if i == p:
                return j
        raise ValueError("Permutation is not applicable")
    phi_ = {}
    for u in G.nodes:
        for v in G.nodes:
            if u == v:
                continue
            if (omega[u],omega[v]) not in G_R.psi.keys():
                continue
            #? Aromatic Bond, Tiple Bond, Double Bond, Simple Bond, Hydrogen Bond
            print("Reaction between: 1. G_R.psi, 2. G.data_before, 3. G.data_after")
            print(u,v,G_R.psi[(omega[u],omega[v])])
            print(u,v,G.get_edge_data(u,v))
            x = -1
            if (u,v) in G.phi.keys():
                x = G.phi[(u,v)] + G_R.psi[(omega[u],omega[v])]
                G[u][v]["bond_type"] = x
            else:
                x = G_R.psi[(omega[u],omega[v])]
                G.add_edge(u,v,bond_type=x)
            phi_.update({(u,v): x}) if x >= 0.0 else ValueError("Transformation is not feasible")
            # G[u][v]["bond_type"] = G[u][v]["bond_type"] + G_R.psi[(omega[u],omega[v])]
            if float(x) <= 0.0: # G[u][v]["bond_type"] == 0:
                G.remove_edge(u,v)
            print(u,v,G.get_edge_data(u,v))
    G.phi = phi_
    return G
