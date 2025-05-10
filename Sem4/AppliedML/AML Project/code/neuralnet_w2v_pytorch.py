import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # Use TensorDataset for dense vectors
import seaborn as sns
import wandb
import os
import pickle
from tqdm import tqdm
import gensim # Added Gensim
import time

# --- Configuration ---
ENTITY = None # Replace with your wandb entity if needed
SEED = 229
RESULTS_FILE = "results_E_W2V_Undersample.txt"
WORD2VEC_VECTOR_SIZE = 100 # Dimension for Word2Vec vectors
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 5 # Ignore words with frequency lower than this
WORD2VEC_WORKERS = max(1, os.cpu_count() - 2) # Use most cores

# --- Setup ---
sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Data Loading (Pre-split and Undersampled) ---
print("Loading pre-split and undersampled data...")
data_dir = "../data/"
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

# --- Train Word2Vec Model --- 
print("Training Word2Vec model on training data...")
# Ensure tokenized_words is a list of lists of strings
sentences = X_train_pd['tokenized_words'].apply(lambda x: x.split()).tolist()

w2v_model = gensim.models.Word2Vec(
    sentences,
    vector_size=WORD2VEC_VECTOR_SIZE,
    window=WORD2VEC_WINDOW,
    min_count=WORD2VEC_MIN_COUNT,
    workers=WORD2VEC_WORKERS,
    seed=SEED
)
print(f"Word2Vec model trained with vocabulary size: {len(w2v_model.wv.key_to_index)}")

# --- Feature Engineering (Averaged Word2Vec + Numerical) ---
print("Applying Feature Engineering (Word2Vec Averaging + Scaling)... ")
numerical_cols = [col for col in X_train_pd.columns if col != 'tokenized_words']

def get_averaged_w2v(token_list_series, w2v_model, vector_size):
    vectors = []
    for tokens_str in tqdm(token_list_series, desc="Averaging Word2Vec"): # Iterate over the Series
        tokens = tokens_str.split() # Split string back into list of tokens
        doc_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv] # Get vectors for words in vocab
        if len(doc_vectors) == 0:
            vectors.append(np.zeros(vector_size))
        else:
            vectors.append(np.mean(doc_vectors, axis=0))
    return np.vstack(vectors)

# Create averaged W2V vectors
X_train_w2v = get_averaged_w2v(X_train_pd['tokenized_words'], w2v_model, WORD2VEC_VECTOR_SIZE)
X_test_w2v = get_averaged_w2v(X_test_pd['tokenized_words'], w2v_model, WORD2VEC_VECTOR_SIZE)

# Get numerical features
X_train_num = X_train_pd[numerical_cols].values
X_test_num = X_test_pd[numerical_cols].values

# Scale numerical features
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Combine W2V and scaled numerical features
X_train_processed = np.hstack((X_train_w2v, X_train_num_scaled))
X_test_processed = np.hstack((X_test_w2v, X_test_num_scaled))

input_dim = X_train_processed.shape[1]
y_train_np = y_train_pd.values
y_test_np = y_test_pd.values

print(f"Processed feature dimension: {input_dim}")

# --- PyTorch Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.5, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_units)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(hidden_units, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        return x

# --- Hyperparameter Grid ---
param_grid = [
    {"lr": 5e-4, "hidden_units": 64, "dropout": 0.3, "batch_size": 4096},
]

# --- Training & Evaluation Loop ---
best_overall_val_loss = float('inf')
best_params = None
best_model_state = None
hyperparam_results = []

print("\n--- Starting Hyperparameter Search for Word2Vec (Undersampled) ---")

# --- Split full processed training data further for validation ---
# Note: X_train_processed is now a dense numpy array
val_split_frac = 0.13 # Use same split proportion as others
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_processed,
    y_train_np,
    test_size=val_split_frac,
    random_state=SEED,
    stratify=y_train_np
)

# Create validation dataset (TensorDataset for dense data)
val_dataset = TensorDataset(torch.tensor(X_val_final, dtype=torch.float32), torch.tensor(y_val_final, dtype=torch.float32).unsqueeze(1))
# Create test dataset
test_dataset = TensorDataset(torch.tensor(X_test_processed, dtype=torch.float32), torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1))

for i, params in enumerate(param_grid):
    print(f"\n--- Trial {i+1}/{len(param_grid)} --- Params: {params} ---")
    run_name = f"NN_E_W2V_Under_Trial_{i+1}"
    config = {
        "subset": "E_Word2Vec",
        "epochs": 300,
        "batch_size": params["batch_size"],
        "learning_rate": params["lr"],
        "validation_split": val_split_frac,
        "early_stopping_patience": 5,
        "hidden_units": params["hidden_units"],
        "dropout_rate": params["dropout"],
        "input_dim": input_dim,
        "word2vec_vector_size": WORD2VEC_VECTOR_SIZE,
        "word2vec_window": WORD2VEC_WINDOW,
        "word2vec_min_count": WORD2VEC_MIN_COUNT,
        "seed": SEED,
        "undersampled": True
    }
    wandb.init(project=WANDB_PROJECT, entity=ENTITY, name=run_name, config=config, reinit=True)

    # Create train dataset/loader for this trial
    train_dataset = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), torch.tensor(y_train_final, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True if device=="cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=2, pin_memory=True if device=="cuda" else False)

    # Initialize Model, Loss, Optimizer
    model = SimpleMLP(input_dim=config["input_dim"], hidden_units=config["hidden_units"], dropout_rate=config["dropout_rate"]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    wandb.watch(model, log_freq=100)

    best_trial_val_loss = float('inf')
    patience_counter = 0
    trial_model_state = None

    for epoch in range(config["epochs"]):
        model.train()
        train_loss_epoch = 0.0
        train_pbar = tqdm(train_loader, desc=f"Trial {i+1} Epoch {epoch+1} Train", leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * batch_X.size(0)
        avg_train_loss = train_loss_epoch / len(train_loader.dataset)

        # Validation Loop
        model.eval()
        val_loss_epoch = 0.0
        val_pbar = tqdm(val_loader, desc=f"Trial {i+1} Epoch {epoch+1} Val", leave=False)
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_pbar:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_loss_epoch += loss_val.item() * batch_X_val.size(0)
        avg_val_loss = val_loss_epoch / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Check if this is the best validation loss for THIS trial
        if avg_val_loss < best_trial_val_loss:
            best_trial_val_loss = avg_val_loss
            trial_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['early_stopping_patience']:
            print(f"Trial {i+1} early stopping.")
            break

    # End of trial
    wandb.log({"best_trial_val_loss": best_trial_val_loss})
    hyperparam_results.append({"params": params, "best_val_loss": best_trial_val_loss})

    if best_trial_val_loss < best_overall_val_loss:
        best_overall_val_loss = best_trial_val_loss
        best_params = params
        best_model_state = trial_model_state
        print(f"*** New best overall validation loss found: {best_overall_val_loss:.4f} with params: {best_params} ***")

    wandb.finish()

print("\n--- Hyperparameter Search Finished ---")
print(f"Best validation loss: {best_overall_val_loss:.4f}")
print(f"Best parameters: {best_params}")

# --- Final Evaluation with Best Model ---
if best_model_state is None:
    print("Error: No best model found.")
    sys.exit(1)

print("\n--- Evaluating Best Model on Test Set --- ")
final_model = SimpleMLP(
    input_dim=input_dim,
    hidden_units=best_params["hidden_units"],
    dropout_rate=best_params["dropout"]
).to(device)
final_model.load_state_dict(best_model_state)
final_model.eval()

test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], num_workers=2, pin_memory=True if device=="cuda" else False)

all_preds = []
all_labels = []
test_pbar = tqdm(test_loader, desc="Final Test Evaluation", leave=False)
with torch.no_grad():
    for batch_X_test, batch_y_test in test_pbar:
        batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
        outputs_test = final_model(batch_X_test)
        predicted_test = (outputs_test > 0.5).float()
        all_preds.extend(predicted_test.cpu().numpy())
        all_labels.extend(batch_y_test.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

print("\nFinal Test Set Performance (Best Hyperparameters):")
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
print(classification_report(all_labels, all_preds, zero_division=0))
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
specificity = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
roc_auc = roc_auc_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# --- Log Results to File ---
print(f"\nSaving results to {RESULTS_FILE}...")
with open(RESULTS_FILE, 'w') as f:
    f.write("Experiment: Neural Network - Word2Vec Features (Undersampled)\n")
    f.write("===\n")
    f.write("Hyperparameters Considered:\n")
    for p in param_grid:
        f.write(f" - {p}\n")
    f.write("\nBest Hyperparameters Found:\n")
    f.write(f" - {best_params}\n")
    f.write(f"   (Achieved Validation Loss: {best_overall_val_loss:.4f})\n")
    f.write("\nTest Set Metrics:\n")
    f.write(f" - Accuracy: {accuracy:.4f}\n")
    f.write(f" - Precision: {precision:.4f}\n")
    f.write(f" - Recall/Sensitivity: {recall:.4f}\n")
    f.write(f" - Specificity: {specificity:.4f}\n")
    f.write(f" - F1 Score: {f1:.4f}\n")
    f.write(f" - ROC AUC: {roc_auc:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(f"{cm}\n")

print("\nScript finished.") 