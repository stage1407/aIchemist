import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent.parent
sources = project_dir / "src"
database = project_dir / "data"
sys.path.insert(0, str(sources))  # Nutze insert(0) statt append(), um Konflikte zu vermeiden
sys.path.insert(0, str(database))
from enum import Enum
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
from src.func.chem_structures import properties
from src.func.chem_structures import reaction_graph
# from data.dataloader import DataLoader
from src.models.mock import generate_mock_dataset
from src.models.reaction_gae import ReactionGAE
from src.models.gnn_encoder import GNNEncoder
from src.models.reaction_gat import ReactionGAT
from src.models.reaction_vgae import ReactionVGAE
from data.extract import Extractor, DatasetType
from data.dataloader import ReactionDataset
from src.func.mol_graph_converter import MolGraphConverter
#from src.func.reaction_graph import reaction_graph
from src.models.loss_function import compute_graph_loss
import gc

MOCK_ON=False

# Beispielkonfiguration
config = {
    "hidden_dim": 32,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def pad_missing_features(data_x, target_dim=19):
    current_dim = data_x.shape[1]

    if current_dim < target_dim:
        padding = torch.zeros((data_x.shape[0], target_dim - current_dim), dtype=torch.float32, device=data_x.device)
        data_x = torch.cat([data_x, padding], dim=1)
    elif current_dim > target_dim:
        data_x = data_x[:, :target_dim]  # Truncate extra dimensions if necessary

    return data_x


# Trainingsfunktion
def train(model, loader, optimizer, device):
    # TODO: Maybe weight valence rule violations much higher than chemical distance loss and structural loss.
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    for educt_graph, react_graph in loader:
        educt_graph = educt_graph.to(device)
        react_graph = react_graph.to(device)
        optimizer.zero_grad()
        # print(data, data.x, data.edge_index)
        # Forward-Pass
        # data.x = pad_missing_features(data.x, target_dim=19)
        #data.x = data.x.detach()
        predicted_reaction = model(educt_graph)
        
        # Loss-Berechnung
        loss = compute_graph_loss(predicted_reaction, react_graph)

        # Backward-Pass und Optimierung
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Speicherung für Metrikberechnung
        all_preds.append(predicted_reaction.argmax(dim=1).detach().cpu())
        all_targets.append(educt_graph.node_target.cpu())

        total_loss += loss.item()

        del loss, predicted_reaction
        torch.cuda.empty_cache()
        # gc.collect()


    # Durchschnittliche Verlustfunktion
    avg_loss = total_loss / len(loader)

    # Metriken (auf allen Batch-Daten kombiniert)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")

    return avg_loss, acc, f1

# Validierungsfunktion
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, react_graph in loader:
            data = data.to(device)

            # Forward-Pass
            data.x = pad_missing_features(data.x, target_dim=19)
            data.x = data.x.detach()
            predicted_reaction = model(data.x, data.edge_index)

            # Loss-Berechnung
            loss = compute_graph_loss(predicted_reaction, react_graph)

            # Speicherung für Metrikberechnung
            all_preds.append(predicted_reaction.argmax(dim=1).detach().cpu())
            all_targets.append(data.node_target.cpu())

            total_loss += loss.item()

            del loss, predicted_reaction
            torch.cuda.empty_cache()
            # gc.collect()




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

def custom_collate(batch):
    educt_list, reaction_list = zip(*batch)
    return Batch.from_data_list(educt_list), Batch.from_data_list(reaction_list)

# Haupttrainingspipeline
def main(model_type : ModelType):
    train_mol_graphs = Extractor(DatasetType.TRAINING)
    val_mol_graphs = Extractor(DatasetType.VALIDATION)

    converter = MolGraphConverter(normalize_features=True, one_hot_edges=True)

    train_dataset = ReactionDataset(train_mol_graphs.data, converter)
    val_dataset = ReactionDataset(val_mol_graphs.data, converter)

    train_loader = DataLoader(train_dataset, collate_fn=custom_collate, batch_size=config["batch_size"], shuffle=True, num_workers=12, pin_memory=True) #TODO Local Settings   # Füge hier den Trainings-Loader ein
    val_loader = DataLoader(val_dataset, collate_fn=custom_collate, batch_size=config["batch_size"], num_workers=12, pin_memory=True) #TODO Local Settings                      # Füge hier den Validierungs-Loader ein
    if MOCK_ON:
        ####### MOCK #######
        mock_dataset = generate_mock_dataset(num_graphs=100, num_nodes=15, num_edges=30, feature_dim=8, edge_attr_dim=4)
        train_loader = DataLoader(mock_dataset[:80], collate_fn=custom_collate, batch_size=8, shuffle=True)
        val_loader = DataLoader(mock_dataset[80:], collate_fn=custom_collate, batch_size=8, shuffle=False)
        ####################
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
    for model in ModelType:
        print(f"Modell: {model}:")
        print(f"Statistik:")
        main(model)
