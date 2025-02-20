import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent.parent
sources = project_dir / "src"
database = project_dir / "data"
sys.path.insert(0, str(sources))  # Nutze insert(0) statt append(), um Konflikte zu vermeiden
import json
import numpy as np
from itertools import product as times
from rdkit import Chem
import networkx as nx #type: ignore
from rdkit.Chem import AllChem 
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdmolops import AddHs
import periodictable
from scipy.optimize import linear_sum_assignment
from networkx.algorithms.isomorphism import GraphMatcher

MAX_DEPTH = 10
quantum_mechanics = False       # set True for modeling pi-electrons        #!Dummy for now
geometric_properties = False    # set True for modeling hydrogen bonds      #!Dummy for now

properties = [
    "Hybridization",            #00
    "Formal charge",            #01
    "Implicit hydrogens",       #02
    "Aromaticity",              #03
    "Chirality",                #04
    "Explicit valence",         #05
    "Implicit valence",         #06
    "Lone pairs",               #07
    #! "Bonding",                  #08   -> semantically belongs to edge features
    "No lone pairs",            #08
    "Degree",                   #09
    "Total_Degree",             #10
    "Formal_Charge",            #11
    "Lone_Pair_Count",          #12
    "Aromatic",                 #13
    "Electronegativity",        #14
    "Stereochemistry",          #15
    "Hydrophobicity"            #16
]


#*node_features = ["element","degree","charge","valence","lones","aromaticity","hydrogen","hybridization"]
#*edge_features = []

with open("./src/func/chemical_features.json","r") as file:
    feature_filter : dict = json.load(file)

properties = {
    "node_features": list(feature_filter["node_features"].keys()),
    "edge_features": list(feature_filter["edge_features"].keys()),
    "graph_features": list(feature_filter["graph_features"].keys())
}

def get_atom_properties(atom : Chem.Atom, property) -> float:
    node_properties = properties["node_features"]
    hybrid_num = {v: k for k,v in Chem.rdchem.HybridizationType.values.items()}
    chiral_num = {v: k for k,v in Chem.rdchem.ChiralType.values.items()}
    property_dict = {
        node_properties[0]: hybrid_num[atom.GetHybridization()],  # Returns the hybridization type (e.g., SP2).    #? Enhance to types
        node_properties[1]: atom.GetFormalCharge(),  # Returns the formal charge on the atom.
        node_properties[2]: atom.GetNumImplicitHs(),  # Returns the number of implicit hydrogens.
        node_properties[3]: atom.GetIsAromatic(),  # Returns True if the atom is aromatic.
        #* Note: Same as Stereochemistry 
        node_properties[4]: chiral_num[atom.GetChiralTag()],  # Returns chirality (e.g., R/S).             #? Enhance to types     
        node_properties[5]: atom.GetExplicitValence(),  # Returns the explicit valence of the atom. 
        node_properties[6]: atom.GetImplicitValence(),  # Returns the implicit valence of the atom.
        node_properties[7]: (atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2, # Calculated manually as lone pairs are not directly accessible.
        #! node_properties[8]: [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()],  # Returns bond types connected to the atom.
        node_properties[8]: int(((atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2) == 0),  # True if no lone pairs.
        node_properties[9]: atom.GetDegree(),  # Returns the number of directly bonded atoms
        node_properties[10]: atom.GetTotalDegree(),  # Returns total degree including implicit hydrogens.
        node_properties[11]: atom.GetFormalCharge(),  # Same as Formal charge.
        node_properties[12]: (atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2,  # Same as Lone pairs.
        node_properties[13]: int(atom.GetIsAromatic()),  # Same as Aromaticity.
        node_properties[14]: atom.GetAtomicNum(),  # Approximate electronegativity by atomic number.
        node_properties[15]: chiral_num[atom.GetChiralTag()] if not geometric_properties else _derive_isometries(),  # Same as Chirality/Stereochemistry.   #? Enhance to types
        node_properties[16]: int(atom.GetAtomicNum() in [6, 7, 8, 15, 16, 17]),  # An approximate way to infer hydrophobicity.
    }
    return property_dict[property]

def get_bond_properties(bond : Chem.Bond, p):
    stereo_num = {v : k for k, v in Chem.rdchem.BondStereo.values.items()}
    properties = {
        "Bond_Order": bond.GetBondTypeAsDouble(),  # Returns the bond order (e.g., 1.0, 2.0).
        "Electrophilic_Character": int(bond.GetBeginAtom().GetAtomicNum() in [6, 7] and bond.GetEndAtom().GetAtomicNum() in [8, 16]),  # Approximate electrophilic character.
        "Nucleophilic_Character": int(bond.GetBeginAtom().GetAtomicNum() in [15, 16] or bond.GetEndAtom().GetAtomicNum() in [15, 16]),  # Approximate nucleophilic character.
        #"Length": rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),  # Bond length from 3D coordinates.
        "Aromaticity": int(bond.GetIsAromatic()),  # Checks if the bond is aromatic.
        #"Orientation": "Not directly available",  # Orientation calculation requires vector math.
        "Stereo_State": stereo_num[bond.GetStereo()],  # Returns the stereochemistry of the bond.           #? Enhance to bondType?
        #"Distance": rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())  # Same as Length.
    }
    return properties[p]

class mol_graph(nx.Graph):
    def __init__(self, smilies : list[str] = None, mols : Chem.RWMol = None):
        """
        Initialize the MoleculeGraph from a SMILES string or RDKit molecule object.
        
        Parameters:
        - smiles: Optional; SMILES string representing the molecule.
        - mol: Optional; RDKit Mol object. If both smiles and mol are provided, mol will be used.
        """
        super().__init__()
        if (mols is None and smilies is None) or (mols is not None and smilies is not None):
            raise ValueError("Either a valid RDKit molecule or SMILES string must be provided.")
        self.smiles = []
        self.mols = []
        # read all smiles strings of the reaction
        if mols is None:
            for smiles in smilies:
                #TODO: SMILES -> SMARTS
                self.smiles.append(smiles)
                try:
                    mol : Chem.rdchem.Mol = Chem.MolFromSmiles(smiles)
                except:
                    smarts = smiles
                    try:
                        mol : Chem.rdchem.Mol = Chem.MolFromSmarts(smarts)
                    except:
                        print(smiles)
                self.mols.append(AddHs(mol))
        else:
            self.mols = mols
        # Make all hydrogen atoms in the molecule explicit
        # look up the config
        if geometric_properties:
            AllChem.EmbedMolecule(self.mol)
            AllChem.UFFOptimizeMolecule(self.mol)
        self._build_graph()

        """Should be obsolete but dunno
        def __getitem__(self, key):
        print("TODO this") 
        # access underlying mols as 
        return key"""

    def _build_graph(self):
        """
        Internal method to build the graph based on the RDKit molecule.
        - Nodes represent atoms.
        - Edges represent bonds with special handling for aromatic bonds.
        """
        self.phi : dict = {}
        index_shift = 0
        separated_list = []
        for mol in self.mols:
            sep_mol = nx.Graph()
            i = 0
            for atom in mol.GetAtoms():
                self.add_node(atom.GetIdx() + index_shift, element=atom.GetSymbol(), feature=_derive_feature_vector(node=atom))
                sep_mol.add_node(atom.GetIdx() + index_shift, element=atom.GetSymbol(), feature=_derive_feature_vector(node=atom))
                i += 1
            for bond in mol.GetBonds():
                self.add_edge(bond.GetBeginAtomIdx() + index_shift, bond.GetEndAtomIdx() + index_shift, bond_type=bond.GetBondTypeAsDouble(), feature=_derive_feature_vector(edge=bond,mol=mol))
                sep_mol.add_edge(bond.GetBeginAtomIdx() + index_shift, bond.GetEndAtomIdx() + index_shift, bond_type=bond.GetBondTypeAsDouble(), feature=_derive_feature_vector(edge=bond,mol=mol))
                self.phi.update({(bond.GetBeginAtomIdx() + index_shift, bond.GetEndAtomIdx() + index_shift): bond.GetBondTypeAsDouble()})
            index_shift = i
            if geometric_properties:
                self._add_hydrogen_bonds(mol)      #? feature or algorithmic retrieval
            separated_list.append(sep_mol)
        self.mol_list = separated_list

    def _add_hydrogen_bonds(self, mol : Chem.RWMol, distance_threshold=2.5):
        #TODO: Comment
        self.donor_acceptor_pairs = self._find_donor_acceptor_pairs(mol)
        conf = mol.GetConformer()
        #TODO: Comment
        for donor, acceptor in self.donor_acceptor_pairs:
            donor_hydrogen = donor['hydrogen']
            acceptor_idx = acceptor.GetIdx()
            #TODO: Comment
            donor_hydrogen_coords = conf.GetAtomPosition(donor_hydrogen.GetIdx())
            acceptor_coords = conf.GetAtomPosition(acceptor_idx)
            #TODO: Comment
            distance = donor_hydrogen_coords.Distance(acceptor_coords)
            #TODO: Comment
            if distance <= distance_threshold:
                self.add_edge(donor_hydrogen.GetIdx(), acceptor_idx, bond_type="Hydrogen Bonding", feature=[0.1,False,Chem.rdchem.BondStereo.STEREONONE, distance])
    
    def _find_donor_acceptor_pairs(self, mol : Chem.Mol):
        donors = []
        acceptors = []
        #TODO: Comment
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in (7,8,9): # N, O, F
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1: # Hydrogen
                        donors.append({'atom': atom, 'hydrogen': neighbor})
            if atom.GetAtomicNum() in (7, 8, 9) and atom.GetTotalDegree() <= 3:
                acceptors.append(atom)
        #TODO: Comment
        donor_acceptor_pairs = []
        for donor in donors:
            for acceptor in acceptors:
                if donor['atom'].GetIdx() != acceptor.GetIdx():
                    donor_acceptor_pairs.append((donor, acceptor))
        return donor_acceptor_pairs

    def get_atom_element(self, atom_idx):
        if atom_idx not in self.nodes:
            raise ValueError(f"Atom index {atom_idx} not found in the graph.")
        return self.nodes[atom_idx]["element"]

    def get_elements(self):
        elem_list = []
        for node_id in self.nodes:
            elem_list.append(self.get_atom_element(node_id))
        return elem_list

    def get_atom_features(self, atom_idx):
        """
        Retrieve atom features from the graph.
        
        Parameters:
        - atom_idx: Index of the atom node.
        
        Returns:
        - A dictionary of features for the specified atom.
        """
        if atom_idx not in self.nodes:
            raise ValueError(f"Atom index {atom_idx} not found in the graph.")
        return self.nodes[atom_idx]["feature"]
    
    def get_bond_features(self, atom_idx1, atom_idx2):
        """
        Retrieve bond features between two atoms.

        Parameters:
        - atom_idx1: Index of the first atom.
        - atom_idx2: Index of the second atom.

        Return:
        - A dictionary of features for the bond.
        """
        if not self.has_edge(atom_idx1, atom_idx2):
            raise ValueError(f"No bond exists between atom {atom_idx1} and atom {atom_idx2}.")
        return self.edges[atom_idx1, atom_idx2]["feature"]
    
    def get_substructural_features(self, subgraph : nx.Graph):
        #TODO:
        pass

    def check_valences(self):
        def valence_electrons(symbol):
            try:
                # Hole das Element aus dem Periodensystem
                element = getattr(periodictable, symbol)
                # Elektronenkonfiguration abrufen
                config = element.electrons
                # Bestimme die Valenzelektronen (letzt Schale mit Elektronen)
                valence_shell = max(config.keys())
                valence_electrons = config[valence_shell]
                return valence_electrons
            except AttributeError:
                return f"Element '{symbol}' was not found!"
            except ValueError:
                return f"Cannot determine number of valence electrons for {symbol}!"
        for v in self.nodes:
            deg = self.degree(v)
            elem = self.get_atom_element(v)
            if deg != valence_electrons(elem):
                print(f"Wrong number of valence electrons on {elem}{v}!")
                return False            
        return True

def _derive_feature_vector(node=None, edge=None, subgraph=None, mol=None):
    features = []
    if node is not None:
        feature_definition : dict = feature_filter["node_features"]
        for p in properties["node_features"]:
            if node.GetSymbol() in feature_definition[p]:
                features.append(get_atom_properties(node,p))
            else:
                features.append(0)
    elif edge is not None and mol is not None:
        feature_definition : dict = feature_filter["edge_features"]
        bond : Chem.Bond = edge
        for p in properties["edge_features"]:
            if feature_definition[p]:
                features.append(get_bond_properties(bond,p))
            else:
                features.append(0)
    
    # elif subgraph and not subgraph:    #! Just to skip subgraph features yet!
    #     #? How to implement subgraphs?
    #     ("mol_interpretation")
    #     ("mol_fragment_type")
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




class reaction_graph(nx.Graph):
    def __init__(self, mol_educts:mol_graph=None, mol_products:mol_graph=None, graph:nx.Graph=None):
        assert (mol_educts is not None and mol_products is not None) or graph is not None 

        super().__init__()

        if mol_educts is not None and mol_products is not None:
            self.educts = mol_educts
            self.products = mol_products
            print(mol_educts.smiles)
            print(mol_products.smiles)
            self.create_reaction_graph()
        elif graph is not None:
            # print("Debug1:", type(graph.nodes))
            # print("Debug2:", list(graph.nodes))
            self.add_nodes_from(graph.nodes(data=True))
            self.add_edges_from(graph.edges(data=True))
            # print("Debug3:", type(self.nodes))
            # print("Debug4:", list(self.nodes))
        else:
            raise AttributeError("Could not create ReactionGraph by Nones as arguments")
        
    def create_reaction_graph(self):
        # Bipartite maximization of MCS relation between educts and products (extended backpacking problem) #! just a heuristic approach
        _, selected_mcs, _ = self.maximize_disjoint_mcs()
        
        # Derive atom-mapping
        atom_mapping = self.compute_atom_mapping(selected_mcs)
        print("Atom Mapping:",atom_mapping)
        # Compute bond changes
        bond_changes = self.compute_bond_changes(atom_mapping)
        print("Bond Changes:",bond_changes)
        # Create nodes and edges in reaction_graph based on bond_changes between educts and products
        # print(selected_pairs,atom_mapping,bond_changes)
        self.build_graph_from_bond_changes(bond_changes)
        print("Nodes:",self.nodes)
        print("Edges:",self.edges)
        # Save atom mapping
        self.bijection = atom_mapping

    """
    def flexible_edge_match(e1,e2):
        bond1 = e1.get("bond_type",None)
        bond2 = e2.get("bond_type",None)
        assert bond1 is not None and bond2 is not None
        if bond
        print(bond1, bond2)
        return True
    """

    def maximize_disjoint_mcs(self):
        reactants = self.educts
        products = self.products

        print("Maximizing MCS...")
        print(f"Reactants count: {len(reactants.mol_list)} | Products count: {len(products.mol_list)}")

        def compute_mcs_sizes(r_list, p_list):
            """
            Computes the MCS sizes between reactants and products.
            Returns a weight matrix and the corresponding pairs.
            """
            def find_mcs_nx(graph1 : mol_graph, graph2 : mol_graph):
                GM = GraphMatcher(graph1, graph2,
                                  node_match=lambda n1, n2: n1['element'] == n2['element'], 
                                  edge_match=lambda e1, e2: True)
                common_sg = {}
                max_score = 0

                for mapping in GM.subgraph_isomorphisms_iter():
                    num_nodes = len(mapping)
                    num_edges = sum(min(graph1.get_edge_data(u,v)["bond_type"],graph2.get_edge_data(u,v)["bond_type"]) 
                                    for u,v in mapping.items() if graph1.has_edge(u,v) and graph2.has_edge(u,v))
                    score = num_nodes + num_edges   # Maximize nodes + edges

                    if score > max_score:
                        max_score = score
                        common_sg = mapping

                #print(common_sg)

                """
                mcs_graph = nx.Graph()
                for u, v in common_sg.items():
                    mcs_graph.add_node(u, **graph1.nodes[u])

                for u, v in common_sg.items():
                    for neighbor in graph1.neighbors(u):
                        if neighbor in common_sg:
                            mcs_graph.add_edge(u, neighbor, **graph1.edges[u, neighbor])
                """

                return common_sg

            pairs = []
            weights = []

            for (i, reactant), (j, product) in times(enumerate(r_list), enumerate(p_list)):
                print(f"Comparing Reactant {i} with Product {j}...")

                # Compute Maximal Common Substructure
                mcs_result = find_mcs_nx(reactant, product)
                mcs_size = len(mcs_result.items()) # if mcs_result != {} else 0

                print(f"MCS Size between Reactant {i} and Product {j}: {mcs_size}")

                # Save the pair and weight
                if mcs_size > 0:
                    pairs.append((reactant, product, mcs_result, mcs_size))
                    weights.append((i, j, mcs_size))

            return pairs, weights

        pairs, weights = compute_mcs_sizes(reactants.mol_list, products.mol_list)

        print(f"Found {len(pairs)} MCS pairs.")

        # Check if no valid pairs were found
        if not pairs:
            print("No MCS pairs found! Something is wrong.")

        # Solve assignment problem
        num_reactants = len(reactants.mol_list)
        num_products = len(products.mol_list)

        cost_matrix = np.zeros((num_reactants, num_products))
        for i, j, mcs_size in weights:
            cost_matrix[i, j] = -mcs_size  # Hungarian method (maximization by minimization)

        if num_reactants == 0 or num_products == 0:
            print("No reactants or products available! Returning empty pairs.")
            return [], [], 0

        reactant_indices, product_indices = linear_sum_assignment(cost_matrix)

        # Gathering maximal MCS and their pairs
        selected_pairs = []
        selected_mcs_results = []
        total_mcs_size = 0
        for r_idx, p_idx in zip(reactant_indices, product_indices):
            for pair in pairs:
                if reactants.mol_list[r_idx] == pair[0] and products.mol_list[p_idx] == pair[1]:
                    selected_pairs.append(pair)
                    selected_mcs_results.append(pair[2])
                    total_mcs_size += pair[3]
                    break

        print(f"Final Selected Pairs: {len(selected_pairs)}")
        return selected_pairs, selected_mcs_results, total_mcs_size

    def compute_atom_mapping(self, mcs_list):
        """
        def get_substruct_match_nx(substructure_graph,educts=True):
            GM = GraphMatcher(self.educts if educts else self.products, substructure_graph,
                              node_match=lambda n1, n2: n1['element'] == n2['element'],
                              edge_match=lambda e1, e2: True)
            matches = next(GM.isomorphisms_iter(), None)
            assert matches is not None
            substruct_graph = nx.Graph()
            for u, v in matches.items():
                substruct_graph.add_node(u, **mol_graph.nodes[u])
        
            for u, v in matches.items():
                for neighbor in mol_graph.neighbors(u):
                    if neighbor in matches:
                        substruct_graph.add_edge(u, neighbor, **mol_graph.edges[u, neighbor])
            return matches
        
        atom_mapping : dict = {}
        for pair in selected_pairs:
            mcs = pair[2]

            reactant_mol : nx.Graph = pair[0]
            product_mol : nx.Graph = pair[1]

            #mcs = Chem.MolFromSmarts(mcs_smarts)

            #reactant_match = reactant_mol.GetSubstructMatch(mcs)
            #product_match = product_mol.GetSubstructMatch(mcs)
            reactant_match = get_substruct_match_nx(mcs,educts=True)
            product_match = get_substruct_match_nx(mcs,educts=False)
            # print(type(reactant_match),type(product_match))

            for r_idx, p_idx in zip(reactant_match, product_match):
                #? Is this enough for the condition
                if reactant_mol.nodes[r_idx]['element'] == product_mol.nodes[p_idx]['element']:
                    atom_mapping[r_idx] = p_idx
        """
        print(mcs_list)
        atom_mapping = {}
        for mcs in mcs_list:
            atom_mapping |= mcs
        return atom_mapping

    def compute_bond_changes(self, atom_mapping : dict):
        bond_changes = {}
        # Compute this for each molecule
        # atom_indices = []
        # for mol in self.educts:
        #    mol : Chem.Mol = mol
        #    print(type(mol))
        #    for atom in mol.GetAtoms():
        #        atom : Chem.Atom = atom
        #        atom_indices.append(atom.GetIdx())
        if atom_mapping == {} or atom_mapping is None:
            print("Shit happens")
            return {}                                   # TODO
        print("Things happen...")
        # print([((atom_mapping[e[0]],atom_mapping[e[1]]), self.products.get_edge_data(atom_mapping[e[0]],atom_mapping[e[1]])) for e in self.educts.edges])
        # print([(e, self.educts.get_edge_data(e[0],e[1])) for e in self.educts.edges])
        # print((list(self.products.edges),list(map(lambda x: self.get_edge_data(x[0],x[1]),self.products.edges))))
        for idx1, idx2 in times(atom_mapping.keys(),atom_mapping.keys()):
            if idx1 < idx2: # Prevent (v,u) if (u,v) is already captured
                try:
                    # print(idx1, idx2)
                    # print(atom_mapping)
                    # print(self.products.edges)
                    prod_n1,prod_n2 = atom_mapping[idx1],atom_mapping[idx2]
                    ed_n1,ed_n2 = idx1,idx2
                    # print(self.products.get_edge_data(prod_n1,prod_n2))
                    # print(self.educts.get_bond_features(ed_n1,ed_n2))
                    # TODO: Debug (No balance : Just added bonds No removals)
                    diff = self.products.get_edge_data(prod_n1,prod_n2)["bond_type"] - self.educts.get_edge_data(ed_n1,ed_n2)["bond_type"]
                    print(diff)
                    if diff != 0:
                        e = (idx1,idx2)
                        bond_changes[e] = diff
                except:
                    pass
        return bond_changes

    def build_graph_from_bond_changes(self, bond_changes):
        """
        Baut den Reaktionsgraphen aus den Bindungsänderungen auf.
        """
        for (atom_idx1, atom_idx2), bond_change in bond_changes.items():
            self.add_node(atom_idx1)
            self.add_node(atom_idx2)
            self.add_edge(atom_idx1, atom_idx2, weight=bond_change)
        
    def isEmpty(self):
        return self.number_of_nodes == 0
    
    def _setPseudoEmpty(self):
        self.add_node()

    def chemical_distance(self):
        """
        Computes the chemical Distance between self and other_graph.
        """
        # Sums up
        sum_g1 = sum(data["weight"] for _,_,data in self.educts.edges(data=True))
        sum_g2 = sum(data["weight"] for _,_,data in self.products.edges(data=True))
        
        selected_pairs = self.maximize_disjoint_mcs()
        sum_mcs = 0
        for pair in selected_pairs:
            mcs_mol = Chem.MolFromSmarts(pair[2])
            for bond in mcs_mol.GetBonds():
                sum_mcs += int(bond.GetBondTypeAsDouble())
                

        chemical_distance = sum_g1 + sum_g2 - 2*sum_mcs

        return chemical_distance
    

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