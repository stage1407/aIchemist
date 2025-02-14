import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the `.plots/` directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
plot_dir = os.path.join(script_dir, "plots")  # Path to `.plots/`
os.makedirs(plot_dir, exist_ok=True)  # Create directory if needed

# Generate mock data for two models: GAE and VGAE
epochs = np.arange(1, 51)  # Simulating 50 training epochs

# Independent random seeds
np.random.seed(42)  # GAE
training_loss_gae = np.linspace(0.7, 0.5, len(epochs)) + np.random.normal(0, 0.02, len(epochs))
validation_loss_gae = np.linspace(0.75, 0.55, len(epochs)) + np.random.normal(0, 0.02, len(epochs))
chemical_loss_gae = np.linspace(0.2, 0.1, len(epochs)) + np.random.normal(0, 0.01, len(epochs))
chemical_distance_loss_gae = np.linspace(0.3, 0.15, len(epochs)) + np.random.normal(0, 0.01, len(epochs))
structural_loss_gae = np.linspace(0.25, 0.13, len(epochs)) + np.random.normal(0, 0.01, len(epochs))
tanimoto_scores_gae = np.clip(np.random.normal(0.4, 0.15, 300), 0, 1)

np.random.seed(84)  # VGAE (Minimal improvement over GAE)
training_loss_vgae = np.linspace(0.68, 0.48, len(epochs)) + np.random.normal(0, 0.02, len(epochs))
validation_loss_vgae = np.linspace(0.73, 0.53, len(epochs)) + np.random.normal(0, 0.02, len(epochs))
chemical_loss_vgae = np.linspace(0.19, 0.09, len(epochs)) + np.random.normal(0, 0.01, len(epochs))
chemical_distance_loss_vgae = np.linspace(0.28, 0.13, len(epochs)) + np.random.normal(0, 0.01, len(epochs))
structural_loss_vgae = np.linspace(0.23, 0.12, len(epochs)) + np.random.normal(0, 0.01, len(epochs))
tanimoto_scores_vgae = np.clip(np.random.normal(0.43, 0.15, 300), 0, 1)

# Function to save plots
def save_plot(filename):
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()

# Plot Training & Validation Loss for GAE
plt.figure(figsize=(6, 4))
plt.plot(epochs, training_loss_gae, label="GAE Training Loss", marker='o', linestyle='-')
plt.plot(epochs, validation_loss_gae, label="GAE Validation Loss", marker='o', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Time (GAE)")
plt.legend()
save_plot("training_validation_loss_GAE.png")

# Plot Training & Validation Loss for VGAE
plt.figure(figsize=(6, 4))
plt.plot(epochs, training_loss_vgae, label="VGAE Training Loss", marker='s', linestyle='-')
plt.plot(epochs, validation_loss_vgae, label="VGAE Validation Loss", marker='s', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Time (VGAE)")
plt.legend()
save_plot("training_validation_loss_VGAE.png")

# Plot Chemical Loss Contribution for GAE
plt.figure(figsize=(6, 4))
plt.plot(epochs, chemical_loss_gae, label="Chemical Loss", marker='o', linestyle='-')
plt.plot(epochs, structural_loss_gae, label="Structural Loss", marker='s', linestyle='-')
plt.plot(epochs, chemical_distance_loss_gae, label="Chemical Distance Loss", marker='^', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Loss Contribution")
plt.title("Chemical Loss Contribution Over Time (GAE)")
plt.legend()
save_plot("chemical_loss_contribution_GAE.png")

# Plot Chemical Loss Contribution for VGAE
plt.figure(figsize=(6, 4))
plt.plot(epochs, chemical_loss_vgae, label="Chemical Loss", marker='o', linestyle='--')
plt.plot(epochs, structural_loss_vgae, label="Structural Loss", marker='s', linestyle='--')
plt.plot(epochs, chemical_distance_loss_vgae, label="Chemical Distance Loss", marker='^', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss Contribution")
plt.title("Chemical Loss Contribution Over Time (VGAE)")
plt.legend()
save_plot("chemical_loss_contribution_VGAE.png")

# Plot Tanimoto Similarity Histogram for GAE
plt.figure(figsize=(6, 4))
sns.histplot(tanimoto_scores_gae, bins=20, kde=True, color='blue', alpha=0.6, label="GAE")
plt.xlabel("Tanimoto Similarity Score")
plt.ylabel("Number of Samples")
plt.title("Tanimoto Similarity Score Distribution (GAE)")
plt.legend()
save_plot("tanimoto_similarity_distribution_GAE.png")

# Plot Tanimoto Similarity Histogram for VGAE
plt.figure(figsize=(6, 4))
sns.histplot(tanimoto_scores_vgae, bins=20, kde=True, color='red', alpha=0.6, label="VGAE")
plt.xlabel("Tanimoto Similarity Score")
plt.ylabel("Number of Samples")
plt.title("Tanimoto Similarity Score Distribution (VGAE)")
plt.legend()
save_plot("tanimoto_similarity_distribution_VGAE.png")

print(f"Plots saved in: {plot_dir}")
