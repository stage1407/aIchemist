from torch_geometric.nn import VGAE

class ReactionVGAE(VGAE):
    def __init__(self, encoder):
        super(ReactionVGAE, self).__init__(encoder)