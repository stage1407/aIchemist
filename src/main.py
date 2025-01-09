import rdkit
import scipy                    #type: ignore
import sympy                    #type: ignore
import torch                    #type: ignore
import matplotlib               #type: ignore
import math
import torch_geometric as pyg   #type: ignore
import numpy

"""
What are the node features?
Are edge features required?
Are graph features useful?
"""

class GNN:
    # TODO: Implement designed architecture and test it (train on reaction networks with ORD data)
    #? Which is more performant on Reaction Networks, GAT or GraphSAGE?
    def __init__(self):
        # TODO
        pass
    
    def train(self):
        # TODO
        pass

    def apply(self):
        # TODO
        pass

class hSage(GNN):
    # TODO: Implement GraphSAGE adapted to heterogenous graphs
    def __init__(self):
        pass
    
class RGAT(GNN):
    # TODO: Implement RGAT
    def __init__(self):
        pass    


class Simulation:
    # TODO: Including Reaction Kinetics, 
    # TODO: -> first-order, second-order reaction
    # TODO: -> Arrhenius Equation, Transition State Theory 
    # TODO: Molecular Dynamics, 
    # TODO: -> Newton's Second Law, Lennard-Jones Potential, Coulomb's Law, Thermostats and Barostats (Nosé-Hoover, Berendsen)
    #? optional in uncertainty of the model Quantum Mechanics + Quantum Chemistry, too.
    #? -> Consider DFT
    # TODO: All-in-One with multiple modes
    #? How are these things connected
    #? -> Born-Oppenheimer Approximation
    #? -> Quantum Molecular Dynamics
    #? -> Monte-Carlo Method to sample configurations of quantum chemistry and molecular dynamics
    def __init__(self):
        # TODO
        pass

    def apply(self):
        # TODO
        pass

    def __solveDGL__(self):
        # TODO
        pass

def main():
    print("Program works!")

if __name__ == '__main__':
    main()