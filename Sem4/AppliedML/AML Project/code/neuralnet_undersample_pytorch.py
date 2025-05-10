import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import wandb
import os
from tqdm import tqdm

# --- Configuration ---
WANDB_PROJECT = "goodreads-popularity-pytorch" # Replace with your project name
ENTITY = None # Replace with your wandb entity if needed

# --- Setup ---
sns.set_theme(style="whitegrid")
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure reproducible results
SEED = 229
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Data Loading and Preparation ---
print("Loading data...")
# Assuming data files are in a 'data' subdirectory relative to this script
data_path = "data/tokenized_reviews.csv"
if not os.path.exists(data_path):
    # If not in data/, try relative to reference_code/ location
    data_path = "reference_code/data/tokenized_reviews.csv"
    if not os.path.exists(data_path):
        print(f"Error: Cannot find {data_path}. Please ensure the data exists.")
        sys.exit(1)

dat = pd.read_csv(data_path)
dat = dat.dropna()
dat["quote"] = dat["quote"].astype(int)
# dat["tokenized_words"] = dat["tokenized_words"].apply(lambda x: x.strip("[']").replace("', '"," "))

# Define feature subsets (as column names)
subset_a_cols = ["user_reviews","days_since_review","user_rating","rating_diff"]
subset_b_cols = ["user_reviews","days_since_review","user_rating","rating_diff",
                 "num_words","avg_word_len","avg_sent_len","pct_verbs",
                 "pct_nouns","pct_adj","quote","sentiment"]

# --- Train/Test Split ---
print("Splitting data...")
X = dat.drop(columns=["popular", "tokenized_words"])
y = dat["popular"]

X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

# --- Undersampling the Training Set ---
print("Undersampling training data...")
neg_indices = y_train_pd[y_train_pd == 0].index
pos_indices = y_train_pd[y_train_pd == 1].index
pos_count = len(pos_indices)

# Randomly sample negative indices to match positive count
rng = np.random.default_rng(seed=SEED)
neg_indices_sampled = rng.choice(neg_indices, pos_count, replace=False)

# Combine positive and sampled negative indices
undersampled_indices = np.concatenate([pos_indices, neg_indices_sampled])

# Create undersampled training set
X_train_under_pd = X_train_pd.loc[undersampled_indices]
y_train_under_pd = y_train_pd.loc[undersampled_indices]

print(f"Original training size: {len(X_train_pd)}")
print(f"Undersampled training size: {len(X_train_under_pd)}")
print(f"Undersampled class distribution:\n{y_train_under_pd.value_counts()}")

# --- PyTorch Model Definition (Same as before) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_units)
        self.relu1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_units, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu1(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        return x

# --- Training Function (Modified wandb name) ---
def train_model(X_train_np, y_train_np, X_test_np, y_test_np, subset_name, subset_column_list, input_dim, hidden_units, config):
    # Note: X_train_np and y_train_np are ALREADY UNDERSAMPLED when passed here
    run_name = f"NN_Undersample_{subset_name}"
    wandb.init(project=WANDB_PROJECT, entity=ENTITY, name=run_name, config=config, reinit=True)
    print(f"\n--- Training Neural Net (Undersampled) - Subset {subset_name} ---")

    # Define model save directory
    model_save_dir = os.path.join("models", "undersample", subset_name)
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, f'best_model_under_{subset_name}.pt')

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np) # Fit only on train, transform test

    # Create Tensors and Datasets
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
    # Test set remains the original imbalanced one
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    # Further split training data for validation
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        X_train_tensor, y_train_tensor, test_size=config["validation_split"], random_state=SEED
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Initialize Model, Loss, Optimizer
    model = SimpleMLP(input_dim=input_dim, hidden_units=hidden_units).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    wandb.watch(model, log_freq=100)

    # Early Stopping and Checkpointing Setup
    best_val_loss = float('inf')
    patience_counter = 0

    # Training Loop
    print("Starting training...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(config["epochs"]):
        model.train()
        train_loss_epoch = 0.0
        correct_train = 0
        total_train = 0
        # Wrap train_loader with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Train", leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * batch_X.size(0)
            predicted = (outputs > 0.5).float()
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss_epoch / len(train_loader.dataset)
        avg_train_acc = correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # Validation Loop
        model.eval()
        val_loss_epoch = 0.0
        correct_val = 0
        total_val = 0
        # Wrap val_loader with tqdm
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Val", leave=False)
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_pbar:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)

                val_loss_epoch += loss_val.item() * batch_X_val.size(0)
                predicted_val = (outputs_val > 0.5).float()
                total_val += batch_y_val.size(0)
                correct_val += (predicted_val == batch_y_val).sum().item()

        avg_val_loss = val_loss_epoch / len(val_loader.dataset)
        avg_val_acc = correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc
        })

        # Early Stopping and Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saved model to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{config['early_stopping_patience']}")

        if patience_counter >= config['early_stopping_patience']:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    # --- Evaluation ---
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    all_labels = []
    # Wrap test_loader with tqdm
    test_pbar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch_X_test, batch_y_test in test_pbar:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            outputs_test = model(batch_X_test)
            predicted_test = (outputs_test > 0.5).float()
            all_preds.extend(predicted_test.cpu().numpy())
            all_labels.extend(batch_y_test.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    print("Test Set Performance:")
    report = classification_report(all_labels, all_preds, output_dict=True)
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    roc_auc = roc_auc_score(all_labels, all_preds)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    wandb.log({
        "test_accuracy": report['accuracy'],
        "test_precision_macro": report['macro avg']['precision'],
        "test_recall_macro": report['macro avg']['recall'],
        "test_f1_macro": report['macro avg']['f1-score'],
        "test_roc_auc": roc_auc,
        "test_confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels.astype(int), preds=all_preds.astype(int), class_names=['Unpopular', 'Popular'])
    })

    # --- Plotting History ---
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - Subset {subset_name} (Undersampled)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    acc_plot_file = f"net_pytorch_acc_{subset_name}_u.png"
    plt.savefig(acc_plot_file)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Subset {subset_name} (Undersampled)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    loss_plot_file = f"net_pytorch_loss_{subset_name}_u.png"
    plt.savefig(loss_plot_file)
    plt.close()

    wandb.finish()

# === Main Execution ===

# --- Subset A (Undersampled) ---
config_a = {
    "subset": "A",
    "epochs": 500,
    "batch_size": 5000,
    "learning_rate": 1e-3,
    "validation_split": 0.13,
    "early_stopping_patience": 20,
    "hidden_units": 4, # From original script
    "input_dim": len(subset_a_cols),
    "seed": SEED,
    "undersampled": True
}
train_model(
    X_train_under_pd[subset_a_cols].values, y_train_under_pd.values, # Use undersampled train data
    X_test_pd[subset_a_cols].values, y_test_pd.values, # Use original test data
    'A', subset_a_cols, config_a["input_dim"], config_a["hidden_units"], config_a
)

# --- Subset B (Undersampled) ---
config_b = {
    "subset": "B",
    "epochs": 500,
    "batch_size": 5000,
    "learning_rate": 1e-3,
    "validation_split": 0.13,
    "early_stopping_patience": 20,
    "hidden_units": 9, # From original script
    "input_dim": len(subset_b_cols),
    "seed": SEED,
    "undersampled": True
}
train_model(
    X_train_under_pd[subset_b_cols].values, y_train_under_pd.values, # Use undersampled train data
    X_test_pd[subset_b_cols].values, y_test_pd.values, # Use original test data
    'B', subset_b_cols, config_b["input_dim"], config_b["hidden_units"], config_b
)

print("\nScript finished.") 