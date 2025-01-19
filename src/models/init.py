from reaction_gat import ReactionGAT
from gnn_encoder import GNNEncoder
from reaction_gae import ReactionGAE
from reaction_vgae import ReactionVGAE

# Für den AutoEncoder (GAE und VGAE) wird ein Encoder benötigt
input_dim = 16  # Anzahl der Knoteneigenschaften
hidden_dim = 32  # Größe der latenten Repräsentation
output_dim = 2   # Anzahl der Knotenklassen (z. B. Reaktionsbeteiligung)

# Initialisierung eines GAT-Modells
reaction_gat = ReactionGAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, heads=4, dropout=0.1)

# Initialisierung eines GAE-Modells
encoder = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
reaction_gae = ReactionGAE(encoder=encoder)

# Initialisierung eines VGAE-Modells
reaction_vgae = ReactionVGAE(encoder=encoder)