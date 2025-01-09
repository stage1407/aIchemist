from data_structures import mol_graph, reaction_graph, transform
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from visualize import visualize_molecule_graph

# Step 1: Generate phenylhydrazine with two carbon atoms and a carbonyl group (R1 and R2)
# Example: R1 = C=O (aldehyde), R2 = C=O (aldehyde) as an acetone-like structure
educt_smiles = 'C1=CC=CC=C1NN=C(C=O)C(C=O)'
# reactive_substructure_smiles = 'C=CNN=CC'

IM_1 = 'C1=CC=CC=C1NNC(C=O)=C(C=O)'  # Phenylhydrazine derivative with carbon and two aldehydes
IM_2 = ''
IM_3 = ''
IM_4 = ''
# Convert SMILES to RDKit Mol object
educt_mol = Chem.MolFromSmiles(educt_smiles)
educt_mol = Chem.AddHs(educt_mol)

# reactive_sub_mol = Chem.MolFromSmiles(reactive_substructure_smiles)
# reactive_sub_mol = Chem.AddHs(reactive_sub_mol)
# Step 2: Generate 2D coordinates for better visualization
# AllChem.Compute2DCoords(educt_mol)

educt_mol

for bond in educt_mol.GetBonds():
    bond : Chem.Bond = bond
    begin : Chem.Atom = bond.GetBeginAtom()
    end : Chem.Atom = bond.GetEndAtom()
    # print(f"[({begin.GetSymbol()},{begin.GetIdx()}),{bond.GetBondType()},({end.GetSymbol()},{end.GetIdx()})]")

atoms = educt_mol.GetAtoms()
mol_indices = len(atoms)*[-1]
#mol_indices[0] = 18
#mol_indices[1] = 4       #? Geoemtry
#mol_indices[2] = 5
#mol_indices[3] = 6
#mol_indices[4] = 19
#mol_indices[5] = 7
#mol_indices[6] = 8
#mol_indices[7] = 11
#mol_indices[8] = 22
#mol_indices[9] = 23
mol_indices[18] = 0
mol_indices[4] = 1
mol_indices[5] = 2
mol_indices[6] = 3
mol_indices[19] = 4
mol_indices[7] = 5
mol_indices[8] = 6
mol_indices[11] = 7
mol_indices[21] = 8
mol_indices[22] = 9


educt_graph = mol_graph(mols=[educt_mol])



transformation1 : dict = {
    (5,6) : -1.0,
    (6,7) : 1.0,
    (7,8) : -1.0,
    (8,5) : 1.0
}

transformation2 : dict = {
    (1,2) : -1.0,
    (2,3) : 1.0,
    (3,5) : -1.0,
    (5,6) : 1.0,
    (6,7) : -1.0,
    (7,1) : 1.0
}

transformation3 : dict = {
    (0,1) : -1,
    (1,2) : 1,
    (2,3) : -1,
    (3,0) : 1
}

transformation4 : dict = {
    (0,3) : -1,
    (3,6) : 1,
    (6,5) : -1,
    (5,0) : 1
}

transformation5 : dict = {
    (5,6) : -1,
    (6,7) : 1,
    (7,9) : -1,
    (9,5) : 1
}

transformations = [transformation1,
                   transformation2,
                   transformation3,
                   transformation4,
                   transformation5]

reaction_graphs = []
intermediates = []
current_graph : mol_graph = educt_graph
visualize_molecule_graph(educt_graph)

i = 0
for t in transformations:
    current_graph = transform(current_graph,mol_indices,reaction_graph(t))
    i+=1
    visualize_molecule_graph(current_graph)
    intermediates.append(current_graph)
    print(f"Performed {i}!")



reaction_path = tuple(reaction_graphs)

