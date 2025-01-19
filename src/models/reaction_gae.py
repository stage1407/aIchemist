from torch_geometric.nn import GAE

class ReactionGAE(GAE):
    def __init__(self, encoder):
        super(ReactionGAE, self).__init__(encoder)