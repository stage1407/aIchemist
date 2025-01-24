from enum import Enum
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
from func.mol_graph import properties
# from data.dataloader import DataLoader
from reaction_gae import ReactionGAE
from gnn_encoder import GNNEncoder
from reaction_gat import ReactionGAT
from reaction_vgae import ReactionVGAE
from data.extract import Extractor, DatasetType
from data.dataloader import ReactionDataset
from func.mol_graph_converter import MolGraphConverter
from loss_function import compute_node_loss, compute_edge_loss

# Beispielkonfiguration
config = {
    "hidden_dim": 32,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Trainingsfunktion
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward-Pass
        node_out, edge_out = model(data)
        
        # Loss-Berechnung
        node_loss = compute_node_loss(node_out, data.node_target)
        edge_loss = compute_edge_loss(edge_out, data.edge_target)
        loss = node_loss + edge_loss

        # Backward-Pass und Optimierung
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Speicherung für Metrikberechnung
        all_preds.append(node_out.argmax(dim=1).detach().cpu())
        all_targets.append(data.node_target.cpu())

    # Durchschnittliche Verlustfunktion
    avg_loss = total_loss / len(loader)

    # Metriken (auf allen Batch-Daten kombiniert)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")

    return avg_loss, acc, f1

# Validierungsfunktion
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Forward-Pass
            node_out, edge_out = model(data)

            # Loss-Berechnung
            node_loss = compute_node_loss(node_out, data.node_target)
            edge_loss = compute_edge_loss(edge_out, data.edge_target)
            loss = node_loss + edge_loss

            total_loss += loss.item()

            # Speicherung für Metrikberechnung
            all_preds.append(node_out.argmax(dim=1).detach().cpu())
            all_targets.append(data.node_target.cpu())

    # Durchschnittliche Verlustfunktion
    avg_loss = total_loss / len(loader)

    # Metriken
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")

    return avg_loss, acc, f1

class ModelType(Enum):
    ENC = 0
    GAE = 1
    GAT = 2
    VGAE = 3

# Haupttrainingspipeline
def main(model_type : ModelType):
    train_mol_graphs = Extractor(DatasetType.TRAINING)
    val_mol_graphs = Extractor(DatasetType.VALIDATION)

    converter = MolGraphConverter(normalize_features=True, one_hot_edges=True)

    train_dataset = ReactionDataset(train_mol_graphs, converter)
    val_dataset = ReactionDataset(val_mol_graphs, converter)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)     # Füge hier den Trainings-Loader ein
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])                       # Füge hier den Validierungs-Loader ein
    input_dim = len(properties["node_features"])
    edge_attr_dim = len(properties["edge_features"])
    model = None         # Initialisiere das Modell

    if model_type == ModelType.ENC:
        model = GNNEncoder(input_dim=input_dim, hidden_dim=config["hidden_dim"], edge_attr_dim=edge_attr_dim)
    elif model_type == ModelType.GAE:
        model = ReactionGAE(input_dim=input_dim, hidden_dim=config["hidden_dim"], edge_attr_dim=edge_attr_dim)
    elif model_type == ModelType.GAT:
        model = ReactionGAT(input_dim=input_dim, hidden_dim=config["hidden_dim"], edge_attr_dim=edge_attr_dim)
    elif model_type == ModelType.VGAE:
        model = ReactionVGAE(input_dim=input_dim, hidden_dim=config["hidden_dim"], edge_attr_dim=edge_attr_dim)

    model.to(config["device"])

    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, config["device"])
        val_loss, val_acc, val_f1 = validate(model, val_loader, config["device"])

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

if __name__ == "__main__":
    main()
