import algorithmic #type: ignore
import torch
import torch.nn
import torch.nn.functional as F

"""
TODO: Implement all Neural Models, with Training
"""

class Transformer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        #? Final MLP for feature transformation
        self.linear_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, node_features, mask=None):
        # Node features shape: (num_nodes, feature_dim)
        
        # Encoder step
        encoded_features = self.encoder(node_features, src_key_padding_mask=mask)

        # Decoder step
        decoded_features = self.decoder(node_features, encoded_features, tgt_key_padding_mask=mask)

        # Final feature transformation
        output_features = self.linear_out(decoded_features)

        return output_features

class RGATLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads, num_layers):
        super(RGATLayer, self).__init__()
        self.transformer = Transformer(feature_dim, hidden_dim, num_heads, num_layers)

    def forward(self, x, edge_index):
        #TODO x: Node features (num_nodes, feature_dim)
        #TODO edge_index: Graph connectivity in COO format (2, num_edges)
        
        # Collect messages from neighbors
        row, col = edge_index  # row = target node, col = source node
        messages = x[col]  # Gather neighbor features for each target node
        
        # Pass messages through Transformer
        transformed_messages = self.transformer(messages)
        
        # Aggregate messages
        x_updated = torch.zeros_like(x)
        x_updated.index_add_(0, row, transformed_messages)  # Aggregate transformed messages
        
        return x_updated

class RGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(RGAT, self).__init__()
        self.layer = RGATLayer(input_dim, hidden_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(input_dim, num_classes)

    def forward(self, x, edge_index):
        # GNN Layer with Transformer-based Message Mechanism
        x = self.layer(x, edge_index)

        # Lineare Transformation zum Output
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)

#! OPEN CHAT: https://chatgpt.com/share/67236ed2-6b44-8008-8420-68d53cadf05a
