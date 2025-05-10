import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import vstack # To handle sparse matrix splitting
import seaborn as sns
import wandb
import os
import pickle
from tqdm import tqdm # Added tqdm

# --- Configuration ---
WANDB_PROJECT = "goodreads-popularity-pytorch" # Replace with your project name
ENTITY = None # Replace with your wandb entity if needed

# --- Setup ---
sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 229
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Data Loading (Pre-split and Undersampled) ---
print("Loading pre-split and undersampled data...")
data_dir = "data/"
if not os.path.exists(os.path.join(data_dir, "X_train.pkl")):
     data_dir = "reference_code/data/" # Try alternative path
     if not os.path.exists(os.path.join(data_dir, "X_train.pkl")):
          print("Error: Cannot find data .pkl files. Run save_undersample.py first.")
          sys.exit(1)

try:
    with open(os.path.join(data_dir, "X_train.pkl"), 'rb') as f:
        X_train_pd = pickle.load(f)
    with open(os.path.join(data_dir, "X_test.pkl"), 'rb') as f:
        X_test_pd = pickle.load(f)
    with open(os.path.join(data_dir, "y_train.pkl"), 'rb') as f:
        y_train_pd = pickle.load(f)
    with open(os.path.join(data_dir, "y_test.pkl"), 'rb') as f:
        y_test_pd = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle files: {e}")
    sys.exit(1)

print(f"Loaded undersampled X_train shape: {X_train_pd.shape}")
print(f"Loaded X_test shape: {X_test_pd.shape}")

# --- Feature Engineering (BOW + Numerical) ---
print("Applying Feature Engineering (BOW + Scaling)... ")
numerical_cols = [col for col in X_train_pd.columns if col != 'tokenized_words']

# Use ColumnTransformer to handle text and scale numerical features separately
preprocessor = ColumnTransformer(
    transformers=[
        ('bow', CountVectorizer(max_features=10000), 'tokenized_words'),
        ('num', StandardScaler(), numerical_cols)
    ],
    remainder='drop' # Drop any columns not specified (shouldn't be any)
)

# Fit on training data, transform train and test
X_train_processed = preprocessor.fit_transform(X_train_pd)
X_test_processed = preprocessor.transform(X_test_pd)

# Get feature names after transformation (important for SHAP)
feature_names = list(preprocessor.get_feature_names_out())
input_dim = X_train_processed.shape[1]

print(f"Processed feature dimension: {input_dim}")

# Further split training data for validation (handle sparse matrix)
val_split = 0.13 # As per original script
train_indices, val_indices = train_test_split(
    np.arange(X_train_processed.shape[0]),
    test_size=val_split,
    random_state=SEED,
    stratify=y_train_pd # Stratify on the original y_train before splitting
)

X_train_final = X_train_processed[train_indices]
y_train_final = y_train_pd.iloc[train_indices].values
X_val_final = X_train_processed[val_indices]
y_val_final = y_train_pd.iloc[val_indices].values

# Convert test y to numpy array
y_test_np = y_test_pd.values

# --- PyTorch Dataset for Sparse Data ---
class SparseDataset(Dataset):
    def __init__(self, X, y):
        # X is a scipy sparse matrix
        # y is a numpy array
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert sparse row to dense numpy array
        x_item = self.X[idx].toarray().flatten()
        y_item = self.y[idx]
        # Convert to tensors
        return torch.tensor(x_item, dtype=torch.float32), torch.tensor(y_item, dtype=torch.float32)

# Create Datasets and DataLoaders
train_dataset = SparseDataset(X_train_final, y_train_final)
val_dataset = SparseDataset(X_val_final, y_val_final)
test_dataset = SparseDataset(X_test_processed, y_test_np)

# --- PyTorch Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_units)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer_2 = nn.Linear(hidden_units, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # The input x will be dense here because the Dataset converts it
        x = self.layer_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        return x

# --- Training Configuration ---
# Reduce hidden units significantly to prevent overfitting
hidden_units_fixed = 256 
print(f"Using fixed Hidden Units: {hidden_units_fixed}")

config = {
    "subset": "C_BOW",
    "epochs": 500,
    "batch_size": 5000,
    "learning_rate": 1e-3,
    "validation_split": val_split,
    "early_stopping_patience": 20,
    "hidden_units": hidden_units_fixed, # Use fixed value
    "input_dim": input_dim,
    "vectorizer": "CountVectorizer",
    "max_features": 10000,
    "seed": SEED,
    "undersampled": True # Loaded presaved undersampled data
}

# --- Define Model Save Directory ---
model_save_dir = os.path.join("models", "undersample", config["subset"])
os.makedirs(model_save_dir, exist_ok=True)
best_model_path = os.path.join(model_save_dir, f'best_model_under_{config["subset"]}.pt')

# --- Initialize Wandb ---
run_name = f"NN_Undersample_{config['subset']}"
wandb.init(project=WANDB_PROJECT, entity=ENTITY, name=run_name, config=config)

# --- Initialize Model, Loss, Optimizer ---
model = SimpleMLP(input_dim=config["input_dim"], hidden_units=config["hidden_units"]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
wandb.watch(model, log_freq=100)

# Create DataLoaders (after model init for potential CUDA pinning)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True if device=="cuda" else False)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=2, pin_memory=True if device=="cuda" else False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=2, pin_memory=True if device=="cuda" else False)

# --- Training Loop ---
print(f"\n--- Training Neural Net (Undersampled) - Subset {config['subset']} ---")
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(config["epochs"]):
    model.train()
    train_loss_epoch = 0.0
    correct_train = 0
    total_train = 0
    # Wrap train_loader with tqdm
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Train", leave=False)
    for batch_X, batch_y in train_pbar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)

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
            batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device).unsqueeze(1)
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
        batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device).unsqueeze(1)
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
plt.title(f'Model Accuracy - Subset {config["subset"]} (Undersampled)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
ac_plot_file = f"net_pytorch_acc_{config['subset']}_u.png"
plt.savefig(ac_plot_file)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss - Subset {config["subset"]} (Undersampled)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
loss_plot_file = f"net_pytorch_loss_{config['subset']}_u.png"
plt.savefig(loss_plot_file)
plt.close()

wandb.finish()
print("\nScript finished.") 