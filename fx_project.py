"""
fx_project.py: Graph Neural Networks for Foreign Exchange Market Prediction

Training and evaluation pipeline executed on the University of Liverpool's
Barkla HPC cluster. Loads pre-cleaned OHLCV data, trains LSTM and T-GCN
models across 4 walk-forward folds, and outputs results to CSV and pickle.

Author: Aisha Jirde (201386550)
Supervisor: Andrew Roxburgh
Module: COMP702 MSc Project
"""
import os

DATA_PATH  = "/mnt/scratch/users/psajirde/fx_project/data/cleaned_fx_data.csv"
OUTPUT_DIR = "/mnt/scratch/users/psajirde/fx_project/outputs_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


"""
==================== SETUP ======================================
"""
# [Central device. use GPU if available, else CPU. Everything moves to this]
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

"""
==================== DATA PREPROCESSING ==========================
"""

"""
Converting the data to percentage returns and train/val/test split
"""

# === [Loading the cleaned CSV file back into DataFrames] ===
import pandas as pd
data = pd.read_csv(DATA_PATH, header = [0, 1], index_col=0, parse_dates=True)
print(data.dtypes)


# === [Convert the closing prices to daily percentage returns] ========================
close = data["Close"]
returns = close.pct_change().dropna()
print(returns.head())


# === [Manual verification of the percentage return calculation] ======================

check_pair = "EURUSD=X"
check_date = close.index[1]
prev_date = close.index[0]

manual_return = (close.loc[check_date, check_pair] - close.loc[prev_date, check_pair]) / close.loc[prev_date, check_pair]
pandas_return = returns.loc[check_date, check_pair]

print(f"Manual calculation: {manual_return:.6f}")
print(f"Pandas pct_change(): {pandas_return:.6f}")
assert abs(manual_return - pandas_return) < 1e-9, "Return calculation mismatch!"
print("Verified: % return calculation is correct.")



# === [Threshold and labelling of the percentage returns] ============================
def label_returns(returns, threshold):
    labels = returns.copy().astype(object)
    labels[returns > threshold] = "UP"
    labels[returns < -threshold] = "DOWN"
    labels[(returns >= -threshold) & (returns <= threshold)] = "NEUTRAL"
    return labels

# Threshold sweep to justify +-0.1% threshold
# Shows how the DOWN/NEUTRAL/UP balance shifts as the threshold changes

# Using +-0.05%, +-0.1%, +-0.2% thresholds
candidate_thresholds = [0.0005, 0.001, 0.002]

for t in candidate_thresholds:
    sweep_labels = label_returns(returns, t)
    counts = sweep_labels.apply(pd.Series.value_counts).fillna(0).astype(int)
    shares = (counts / counts.sum() * 100).round(1)
    print(f"\n=== Threshold +-{t*100:.2f}% ===")
    print("Class shares (%) — DOWN / NEUTRAL / UP")
    print(shares)



# The chosen threshold used for the rest of the project is +-0.1%
CHOSEN_THRESHOLD = 0.001
labels = label_returns(returns, CHOSEN_THRESHOLD)
print(f"\nUsing chosen threshold +-{CHOSEN_THRESHOLD*100:.1f}% for all downstream labelling.")


# === [Splits the data chronologically into 4 time based Folds] ===================

# Defines 4 separate Folds using the walk-forward fold definition.
# Each Fold covers a different period in the market

FOLDS = [
    {"name": "Fold 1 - Pre-COVID (2019)",
     "train_end": "2018-11-30", "val_end": "2019-02-28", "test_end": "2019-12-31"},
    {"name": "Fold 2 - COVID shock (2020)",
     "train_end": "2019-11-30", "val_end": "2020-02-29", "test_end": "2020-12-31"},
    {"name": "Fold 3 - Recovery & Ukraine war (2021-Jun 2022)",
     "train_end": "2020-11-30", "val_end": "2021-02-28", "test_end": "2022-06-30"},
    {"name": "Fold 4 - Calmer period (Sep 2022-2023)",
     "train_end": "2022-05-31", "val_end": "2022-08-31", "test_end": "2023-12-29"},
]
DATA_START = "2018-01-01"

# This produces each fold's Train/Val/Test sets based on their respective date ranges
def get_fold_splits(returns_all, labels_all, fold, start=DATA_START):
    idx_r = returns_all.index
    train_mask = (idx_r > pd.Timestamp(start)) & (idx_r <= pd.Timestamp(fold["train_end"]))
    val_mask   = (idx_r > pd.Timestamp(fold["train_end"])) & (idx_r <= pd.Timestamp(fold["val_end"]))
    test_mask  = (idx_r > pd.Timestamp(fold["val_end"])) & (idx_r <= pd.Timestamp(fold["test_end"]))
    return (returns_all[train_mask], labels_all[train_mask],
            returns_all[val_mask],   labels_all[val_mask],
            returns_all[test_mask],  labels_all[test_mask])

# Sanity print of split sizes per fold

for fold in FOLDS:
    r_tr, l_tr, r_va, l_va, r_te, l_te = get_fold_splits(returns, labels, fold)
    print(f"{fold['name']:45s} train={len(r_tr):4d}  val={len(r_va):4d}  test={len(r_te):4d}")



# ===[Maps text labels to integer classes] =======================================
# Converts string labels to numeric class indices (DOWN=0, NEUTRAL=1, UP=2)

label_map = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}

# This only converts if labels are still strings to prevent any corruption

if labels.dtypes.eq(object).any():
    labels = labels.replace(label_map).astype("int64")

# Confirm every value is a valid class index before continuing
assert labels.isin([0, 1, 2]).all().all(), "Unexpected label values after encoding"
print("Labels encoded to 0/1/2. dtypes:")
print(labels.dtypes)



# === [Standardises the returns in each fold] =================================

from sklearn.preprocessing import StandardScaler

# This takes one fold's Train/Val/Test returns
def scale_fold(returns_train, returns_val, returns_test):

  # Creates a scaler that rescales the returns to mean (0) and spread (1)
  # Scaling is only learned from solely the training data and scales the training set
  # Then scales the validation and test sets using the training numbers
    scaler = StandardScaler()
    tr = pd.DataFrame(scaler.fit_transform(returns_train),
                       index=returns_train.index, columns=returns_train.columns)
    va = pd.DataFrame(scaler.transform(returns_val),
                       index=returns_val.index, columns=returns_val.columns)
    te = pd.DataFrame(scaler.transform(returns_test),
                       index=returns_test.index, columns=returns_test.columns)

    # Returns the scaled sets
    return tr, va, te, scaler



# === [Compute and defines the class weight function for imbalanced labels] ====== 
# So that rare direction classes (e.g. NEUTRAL) are not ignored during training
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(labels_flat, num_classes=3):

    # Finds the classes that appear in the training labels of the fold
    present = np.unique(labels_flat)

    # Creates a balanced weight for each class
    w = compute_class_weight(class_weight="balanced", classes=present, y=labels_flat)

    # Sets weight at 1.0 for each class
    weight_full = np.ones(num_classes, dtype=np.float32)

    # Fills in the computed weights for the present classes
    for cls, wt in zip(present, w):
        weight_full[int(cls)] = wt
    return torch.tensor(weight_full, dtype=torch.float32)


"""
==================== BASELINES ======================================
"""
# === [Set up the lookback window: This is a shared experimental constant] ========================================
LOOKBACK = 20




# ===== [Persistence baseline: Predicts that the direction of each day repeats the previous day's direction] ======

def evaluate_persistence_baseline(labels_series, lookback=LOOKBACK):
    vals = labels_series.values
    trues = vals[lookback:]
    preds = vals[lookback - 1:-1]
    return trues, preds



# ===== [Majority class baseline: Predicts the single most frequent class in the training data for every test day] ========================================
def evaluate_majority_baseline(labels_train_series, labels_test_series, lookback=LOOKBACK):
    train_vals = labels_train_series.values                     # Most common class in Training
    majority_class = int(pd.Series(train_vals).mode().iloc[0])
    trues = labels_test_series.values[lookback:]
    preds = [majority_class] * len(trues)
    return trues, preds



# ===== [Logistic regression baseline: Each 20 day return window is flattened into features and is fitted per pair] ========================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# Defines a function that converts returns into feature & target pairs for logistic regression
def build_lr_dataset(returns_col, labels_col, lookback=LOOKBACK):

    # Each row is the Lookback days of returns, and the target is the next day's label
    values = returns_col.values
    labs = labels_col.values
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i + lookback])    # The 20 days of returns become 20 features
        y.append(labs[i + lookback])        # The label of the next day is the target
    return np.array(X), np.array(y)

def train_eval_logreg(pair, returns_train, labels_train, returns_test, labels_test,
                       lookback=LOOKBACK):

    # Constructs a table with the flattened features for training and test
    X_tr, y_tr = build_lr_dataset(returns_train[pair], labels_train[pair], lookback)
    X_te, y_te = build_lr_dataset(returns_test[pair],  labels_test[pair],  lookback)

    # Shapes a class balanced logistic regression solely on the training data
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    # Generates predictions on the test set and score with Macro-F1 and Accuracy
    preds = clf.predict(X_te)
    f1 = f1_score(y_te, preds, average="macro")
    acc = accuracy_score(y_te, preds)
    return f1, acc


"""
==================== LSTM IMPLEMENTATION ======================================
"""

"""
LSTM Architecture Design
"""
# ===== [Defines FXDataset: Converts a single currency pair's returns and labels into training ready windows] ========================================
import torch
from torch.utils.data import Dataset, DataLoader


class FXDataset(Dataset):
    def __init__(self, returns_col, labels_col, lookback):
        # Stores each pair's window length & the returns and labels as tensors
        self.returns = torch.tensor(returns_col.values, dtype=torch.float32)
        self.labels  = torch.tensor(labels_col.values,  dtype=torch.long)
        self.lookback = lookback

    def __len__(self):
        # Instructs how many training windows can be formed from this pair's data
        return len(self.returns) - self.lookback

    def __getitem__(self, i):
      # Returns one training example: 20 days of returns as an input sequence and the following day's label
        x = self.returns[i : i + self.lookback]
        y = self.labels[i + self.lookback]
        return x.unsqueeze(-1), y



# ===== [Defining the LSTM model class] ========================================

# Hyperparameter values are placeholders in this cell
# The actual searched values come from the Grid Search

import torch.nn as nn

# Forming the LSTM model
class FXLSTM(nn.Module):

  # Placeholder Hyperparameter values, actual values decided from Grid Search method
    def __init__(self, hidden_size=32, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()

        # LSTM reads sequence of daily turns and constructs hidden summary
        self.lstm = nn.LSTM(1, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)

        # Final layer converts the sequence summary into scores (DOWN / NEUTRAL / UP)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        # Sequence is processed through the LSTM
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


   
"""
LSTM Model Training
"""
# ===== [Trains one LSTM model for a single currency pair using a given set of hyperparameters] ========================================

# The hyperparameters (hidden, dropout and batch size) are decided by the grid search
# When those values are decided this function is executed repeatedly with different values

def train_one(pair, returns_train, labels_train, returns_val, labels_val,
              lookback=LOOKBACK, hidden=32, dropout=0.2, num_layers=2,
              batch_size=32, lr=0.001, epochs=30, patience=5, verbose=True):


    # Converts this single pair's Train and Validation returns into datasets
    tr_ds = FXDataset(returns_train[pair], labels_train[pair], lookback)
    va_ds = FXDataset(returns_val[pair],   labels_val[pair],   lookback)

    # Wrap these datasets into DataLoaders to process the data into batches without shuffling
    # Setting shuffle to False preserves the sequence order
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)


    # Computes the class weights based on the distribution of the training labels
    # Therefore under-represented classes (e.g. NEUTRAL) aren't ignored
    class_weights = get_class_weights(labels_train[pair].values)


    # Builds the models with these specified architectural parameters, optimiser and loss function
    model = FXLSTM(hidden_size=hidden, num_layers=num_layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)


    # Initialises empty lists to track loss for each epoch
    # Stores the best model seen so far in its memory
    train_losses, val_losses = [], []
    best_val, wait, best_state = float("inf"), 0, None

    # Training pass: Updates the model on the training batches
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= max(len(tr_dl), 1)


        # Validation pass: Measures the loss without updating the model
        model.eval(); vloss = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                vloss += crit(model(xb), yb).item()
        vloss /= max(len(va_dl), 1)

        train_losses.append(epoch_train_loss)
        val_losses.append(vloss)

        # Early stopping: This keeps the best model and stops if the validation loss stalls
        if verbose: print(f"  {pair} epoch {epoch+1}: val_loss={vloss:.3f}")
        if vloss < best_val:
            best_val, wait, best_state = vloss, 0, model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                if verbose: print("  early stop")
                break

    # Restore the best model found during the training process and then returns it
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


"""
LSTM Hyperparameter tuning
"""
# ===== [Hyperparameter tuning for the LSTM] ========================================

import random
import numpy as np
import torch

# Defines a function to lock in randomness so a run is repeatable and reproducible

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# The 5 seeds we train and average the results over
SEEDS = [0, 1, 2, 3, 4]

# Searched LSTM hyperparameters
# The search grid for each combination of hyperparameter values
LSTM_GRID = {
    "hidden":     [16, 32, 64, 128],
    "dropout":    [0.1, 0.2, 0.3],
    "batch_size": [32, 64, 128],
}

# Fixed LSTM hyperparameters
LSTM_FIXED = {
    "num_layers": 2,
    "lr":         0.001,
    "lookback":   LOOKBACK,
    "epochs":     30,
    "patience":   5,
}

# Prints search grid size for confirmation
_n_configs = len(LSTM_GRID["hidden"]) * len(LSTM_GRID["dropout"]) * len(LSTM_GRID["batch_size"])
print(f"LSTM grid: {_n_configs} configs per pair "
      f"({LSTM_GRID['hidden']} x {LSTM_GRID['dropout']} x {LSTM_GRID['batch_size']})")
print(f"Seeds for final models: {SEEDS}")



# ===== [LSTM hyperparameter grid search: Searches each hyperparameter combination for each pair's returns] ========================================

import itertools
from sklearn.metrics import f1_score

def eval_lstm_val_f1(model, returns_val, labels_val, pair, lookback, batch_size):

    """Macro-F1 of trained LSTM on validation set for one pair"""

    # Build the dataset using the validation returns & labels for the specified currency pair
    ds = FXDataset(returns_val[pair], labels_val[pair], lookback)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds += model(xb).argmax(1).tolist()   # Predicted class per sample
            trues += yb.tolist()
    if not trues:
        return float("nan")
    return f1_score(trues, preds, average="macro")


def search_lstm_pair(pair, r_tr, l_tr, r_va, l_va,
                     grid=LSTM_GRID, fixed=LSTM_FIXED, search_seed=0, verbose=False):
    """Grid search for one pair's LSTM. Selects the configuration with the best Validation Macro-F1"""

    best = {"val_f1": -1.0, "config": None}

    # Executes every combination for each searched hyperparameters
    for hidden, dropout, batch_size in itertools.product(
            grid["hidden"], grid["dropout"], grid["batch_size"]):

        # Uses the same seed for every configuration for fair evaluation
        set_seed(search_seed)

        # Trains this configuration
        model, _, _ = train_one(
            pair, r_tr, l_tr, r_va, l_va,
            lookback=fixed["lookback"], hidden=hidden, dropout=dropout,
            num_layers=fixed["num_layers"], batch_size=batch_size,
            lr=fixed["lr"], epochs=fixed["epochs"], patience=fixed["patience"],
            verbose=False)

        # Sets a F1 score for the configuration on the validation set
        val_f1 = eval_lstm_val_f1(model, r_va, l_va, pair,
                                  fixed["lookback"], batch_size)

        if verbose:
            print(f"    hidden={hidden:3d} dropout={dropout} batch={batch_size:3d} "
                  f"-> val_f1={val_f1:.4f}")

        # Retains this if it provides the optimal validation score
        if val_f1 > best["val_f1"]:
            best = {"val_f1": val_f1,
                    "config": {"hidden": hidden, "dropout": dropout,
                               "batch_size": batch_size}}

    return best   # {"val_f1": float, "config": {...}}



# ===== [Multi-seed run of the winning LSTM configuration: Retrains the best configuration decided by the Grid search across all seeds] ========================================

# Reports the test Macro-F1 as a mean & standard deviation across those seeds

import numpy as np
from sklearn.metrics import f1_score

def eval_lstm_test_f1(model, returns_test, labels_test, pair, lookback, batch_size):
    """Macro-F1 of a trained LSTM on the TEST set for one pair."""
    ds = FXDataset(returns_test[pair], labels_test[pair], lookback)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds += model(xb).argmax(1).tolist()
            trues += yb.tolist()
    if not trues:
        return float("nan")
    return f1_score(trues, preds, average="macro")

def run_lstm_multiseed(pair, r_tr, l_tr, r_va, l_va, r_te, l_te,
                       config, fixed=LSTM_FIXED, seeds=SEEDS, verbose=False):

    """Retrain the winning LSTM configuration across all seeds and reports test macro-F1
    as a mean and standard deviation. Each seed generates a different random initialisation.
    Therefore the spread across the seeds reflect the effect the initialistaion has on the result"""

    seed_f1s = []
    for seed in seeds:

        # Sets the random seed for this specific run, then train the winning configuration
        set_seed(seed)
        model, _, _ = train_one(
            pair, r_tr, l_tr, r_va, l_va,
            lookback=fixed["lookback"], hidden=config["hidden"],
            dropout=config["dropout"], num_layers=fixed["num_layers"],
            batch_size=config["batch_size"], lr=fixed["lr"],
            epochs=fixed["epochs"], patience=fixed["patience"], verbose=False)

        test_f1 = eval_lstm_test_f1(model, r_te, l_te, pair,
                                    fixed["lookback"], config["batch_size"])
        seed_f1s.append(test_f1)
        if verbose:
            print(f"    seed={seed} -> test_f1={test_f1:.4f}")

    # Combines the results per seed into a mean, standard deviation and the individual scores
    seed_f1s = np.array(seed_f1s, dtype=float)
    return {
        "mean_f1": np.nanmean(seed_f1s),
        "std_f1":  np.nanstd(seed_f1s),
        "per_seed": seed_f1s.tolist(),
        "config":  config,
    }


"""
==================== T-GCN IMPLEMENTATION ======================================
"""

"""
T-GCN Architecture Design
"""
# ===== [Installing PyTorch Geometric and Temporal] ========================================

import torch
tv = torch.__version__.split('+')[0]          # e.g. 2.11.0
cu = torch.version.cuda.replace('.', '')      # e.g. 128
wheel_url = f"https://data.pyg.org/whl/torch-{tv}+cu{cu}.html"
print("Using prebuilt wheels from:", wheel_url)


from torch_geometric_temporal.nn.recurrent import TGCN
print("TGCN import OK")



# ===== [The T-GCN full currency graph variant construction] ========================================
# Each currency is a node and an undirected edge connects two currencies whenever a traded pair links them

import torch

# Defines 6 unique currency nodes across the Forex market graph
nodes = ["USD", "EUR", "GBP", "JPY", "CNY", "AUD"]
idx = {c: i for i, c in enumerate(nodes)}

# Defines 7 traded pairs, each mapped to its currencies
pair_currencies = {
    "GBPUSD=X": ("GBP", "USD"),
    "EURUSD=X": ("EUR", "USD"),
    "EURGBP=X": ("EUR", "GBP"),
    "EURJPY=X": ("EUR", "JPY"),
    "USDJPY=X": ("USD", "JPY"),
    "USDCNY=X": ("USD", "CNY"),
    "AUDUSD=X": ("AUD", "USD"),
}

# Constructs bidirectional edge list: Adds an edge in both directions for each pair between its two currencies
# Providing information flow in both directions during message passing

src, dst = [], []
for pair, (a, b) in pair_currencies.items():
    src += [idx[a], idx[b]]
    dst += [idx[b], idx[a]]
edge_index = torch.tensor([src, dst], dtype=torch.long)

print("nodes:", len(nodes), " edges (directed):", edge_index.shape[1])
print(edge_index)

# Confirms the graph has both corrected size and valid node indices
assert len(nodes) == 6, f"Expected 6 nodes, got {len(nodes)}"
assert edge_index.shape[1] == 14, f"Expected 14 directed edges, got {edge_index.shape[1]}"
assert edge_index.max().item() < len(nodes), "edge_index references a node index that doesn't exist"
print("Graph structure sanity checks passed: 6 nodes, 14 directed edges.")



# ===== [The T-GCN Self-loops ablation variant construction] ========================================

# Each currency node connects only to itself, therefore no information passes between currencies

n_nodes = len(nodes)

# Builds 6 self loops
self_loop_edge_index = torch.tensor(
    [list(range(n_nodes)), list(range(n_nodes))], dtype=torch.long)

print("Self-loops edge_index:")
print(self_loop_edge_index)

# Confirms there is exactly one self-loop per node
assert self_loop_edge_index.shape[1] == n_nodes, \
    f"Expected {n_nodes} self-loops, got {self_loop_edge_index.shape[1]}"
assert (self_loop_edge_index[0] == self_loop_edge_index[1]).all(), \
    "Self-loop edge_index must connect each node to itself only"
print(f"Self-loops sanity checks passed: {n_nodes} self-loops, no cross-currency edges.")



# ===== [Defines the pair ordering and builds the prediction targets for the T-GCN] ========================================

# The directional label for each pair is used for the targets

pair_list = list(pair_currencies.keys())  # The fixed order for the 7 currency pairs

def build_edge_labels(labels_df):
    # Returns the label columns for all of the pairs as a clean integer array
    # The T-GCN uses this as its prediction targets
    return labels_df[pair_list].astype("int64").values



# ===== [Constructs each currency's daily node feature by averaging the returns of each corresponding pair] ========================================

import numpy as np


# A currency can be on either side of the pair so the return is sign oriented
# Return is either +1 (when it is the base ) and -1 (when it is the quote)

node_pairs_signed = {}
for c in nodes:
    lst = []
    for p, (base, quote) in pair_currencies.items():
        if c == base:
            lst.append((p, 1.0))
        elif c == quote:
            lst.append((p, -1.0))
    node_pairs_signed[c] = lst


def build_node_features(returns_df):
  # Apply the orientation and average per currency to produce one feature per node
    feats = np.zeros((len(returns_df), len(nodes), 1), dtype=np.float32)
    for i, c in enumerate(nodes):
        signed = node_pairs_signed[c]
        cols = [p for p, s in signed]
        signs = np.array([s for p, s in signed], dtype=np.float32)
        oriented = returns_df[cols].values * signs        # flip quote-side signs
        feats[:, i, 0] = oriented.mean(axis=1)
    return feats



# ===== [Windows the graph sequence for the T-GCN: Each item brings LOOKBACK (20) days of node features as inputs] ========================================

# Paired with the next day's edge labels as the targets

from torch.utils.data import Dataset, DataLoader

class GraphSeqDataset(Dataset):

    def __init__(self, X, Y, lookback):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.lookback = lookback
    def __len__(self):
        return len(self.X) - self.lookback
    def __getitem__(self, i):
        x = self.X[i:i+self.lookback]
        y = self.Y[i+self.lookback]
        return x, y

"""
T-GCN Model training
"""

# ===== [Defines the batched T-GCN: Processes sequences of the currency graph] ========================================
# Produces a directional class prediction (UP/DOWN/NEUTRAL) for each pair

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN

# Defines the models individual components
class FXTGCNEdgeBatched(nn.Module):
    def __init__(self, in_channels=1, hidden=32, num_classes=3, dropout=0.0,
                 pair_currencies=pair_currencies, idx=idx, num_nodes=None):
        super().__init__()
        self.tgcn = TGCN(in_channels=in_channels, out_channels=hidden)  # Graph time layer component (The T-GCN layer)
        self.dropout = nn.Dropout(dropout)                              # Dropout component
        self.edge_fc = nn.Linear(hidden * 2, num_classes)               # Pair classifier: Scores each pair as UP / DOWN / NEUTRAL
        self.edge_pairs = [(idx[a], idx[b]) for a, b in pair_currencies.values()]  # Maps pair to its two nodes
        self.num_nodes = num_nodes if num_nodes is not None else len(idx)          # Number of currency nodes in the graph


    # Defines function that replicates the graph for each sample, offsetting each copy's node indices
    # So that replicas stay disconnected, permitting whole batch to be processed in one pass
    def _batched_edge_index(self, edge_index, batch_size, device):
        N = self.num_nodes
        E = edge_index.shape[1]
        offsets = (torch.arange(batch_size, device=device) * N).repeat_interleave(E)
        big = edge_index.to(device).repeat(1, batch_size) + offsets.unsqueeze(0)
        return big

    # Defines model's forward pass
    def forward(self, window, edge_index, edge_weight=None):
        # Shape of the input data: [batch, lookback, num_nodes, in_channels]
        B, T, N, Fdim = window.shape
        device = window.device

        # Combines one graph copy per sample into single big graph for processing
        big_edge_index = self._batched_edge_index(edge_index, B, device)

        # Graph is unweighted
        big_edge_weight = None
        if edge_weight is not None:
            big_edge_weight = edge_weight.to(device).repeat(B)

        # Input the LOOKBACK days into the graph time layer, one day at a time
        # Each step updates h (memory of each currency's state)
        h = None
        for t in range(T):
            xt = window[:, t].reshape(B * N, Fdim)
            h = self.tgcn(xt, big_edge_index, big_edge_weight, H=h)


        h = self.dropout(h)
        h = h.reshape(B, N, -1)

        # Conjoin the currencies' final state and score its direction for each pair
        logits = []
        for (a, b) in self.edge_pairs:
            pair_h = torch.cat([h[:, a, :], h[:, b, :]], dim=-1)
            logits.append(self.edge_fc(pair_h))
        return torch.stack(logits, dim=1)




# ===== [Batched T-GCN training function (one config): Trains single config on one fold's data] ========================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Fixed T-GCN settings;
TGCN_FIXED = {
    "lr":       0.001,
    "lookback": LOOKBACK,   # shared constant (=20), same as LSTM
    "epochs":   30,
    "patience": 5,
}


def train_tgcn_one(g_train_X, g_train_Y, g_val_X, g_val_Y,
                   edge_index, class_weights,
                   hidden, dropout, batch_size,
                   fixed=TGCN_FIXED, verbose=False):
    lookback = fixed["lookback"]

    # Builds batch loaders for the fold's training and validation graphs
    tr_dl = DataLoader(GraphSeqDataset(g_train_X, g_train_Y, lookback),
                       batch_size=batch_size, shuffle=False)
    va_dl = DataLoader(GraphSeqDataset(g_val_X, g_val_Y, lookback),
                       batch_size=batch_size, shuffle=False)


    # Builds the T-GCN model, optimiser and loss function
    # Moves it to the GPU for training from the CPU (where it was initially created)
    edge_index = edge_index.to(device)
    model = FXTGCNEdgeBatched(hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=fixed["lr"])
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))


    train_losses, val_losses = [], []
    best, wait, best_state = float("inf"), 0, None
    for epoch in range(fixed["epochs"]):


        # Training pass
        model.train()
        ep_loss = 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, edge_index, None)
            loss = crit(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            loss.backward(); opt.step()
            ep_loss += loss.item()
        ep_loss /= max(len(tr_dl), 1)


        # Validation pass
        model.eval(); vloss = 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                out = model(x, edge_index, None)
                vloss += crit(out.reshape(-1, out.shape[-1]), y.reshape(-1)).item()
        vloss /= max(len(va_dl), 1)

        train_losses.append(ep_loss); val_losses.append(vloss)
        if verbose: print(f"    epoch {epoch+1}: val_loss={vloss:.3f}")


        # Early stopping: Stops if the validation loss stalls and keeps the best model
        if vloss < best:
            best, wait, best_state = vloss, 0, model.state_dict()
        else:
            wait += 1
            if wait >= fixed["patience"]:
                if verbose: print("    early stop")
                break

    # Restores the best model before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


"""
T-GCN Hyperparameter tuning
"""

# ===== [T-GCN grid search: Uses Validation only & selects the best configuration] ========================================

import itertools
from sklearn.metrics import f1_score

# T-GCN search grid:
TGCN_GRID = {
    "hidden":     [8, 16, 32, 64, 100, 128],
    "dropout":    [0.1, 0.3, 0.5],
    "batch_size": [32, 64],
}


def eval_tgcn_val_f1(model, g_val_X, g_val_Y, edge_index, lookback, batch_size):
    """Pooled macro-F1 of a trained T-GCN on the validation set."""
    dl = DataLoader(GraphSeqDataset(g_val_X, g_val_Y, lookback),
                    batch_size=batch_size, shuffle=False)
    edge_index = edge_index.to(device)  # Moves the graph to GPU
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)            # Moves each batch to GPU
            out = model(x, edge_index, None)
            preds += out.argmax(-1).reshape(-1).tolist()
            trues += y.reshape(-1).tolist()
    if not trues:
        return float("nan")
    return f1_score(trues, preds, average="macro")


def search_tgcn(g_tr_X, g_tr_Y, g_va_X, g_va_Y, edge_index, class_weights,
                grid=TGCN_GRID, fixed=TGCN_FIXED, search_seed=0, verbose=False):
    """Grid-search the T-GCN. Selects configuration with best validation Macro-F1.
    Single fixed seed during search so configurations are compared fairly. """

    best = {"val_f1": -1.0, "config": None}

    for hidden, dropout, batch_size in itertools.product(
            grid["hidden"], grid["dropout"], grid["batch_size"]):

        # Uses the same seed for each configuration for fairness
        set_seed(search_seed)
        model, _, _ = train_tgcn_one(
            g_tr_X, g_tr_Y, g_va_X, g_va_Y, edge_index, class_weights,
            hidden=hidden, dropout=dropout, batch_size=batch_size,
            fixed=fixed, verbose=False)

        val_f1 = eval_tgcn_val_f1(model, g_va_X, g_va_Y, edge_index,
                                  fixed["lookback"], batch_size)

        if verbose:
            print(f"    hidden={hidden:3d} dropout={dropout} batch={batch_size:3d} "
                  f"-> val_f1={val_f1:.4f}")

        if val_f1 > best["val_f1"]:
            best = {"val_f1": val_f1,
                    "config": {"hidden": hidden, "dropout": dropout,
                               "batch_size": batch_size}}

    return best



# ===== [T-GCN multiseed: Runs winning configuration across 5 seeds] ========================================

# Retrains the chosen configuration across all 5 seeds &
# Reports per pair macro-F1 mean ± std for both T-GCN variants

import numpy as np
from sklearn.metrics import f1_score

def eval_tgcn_test_perpair(model, g_te_X, g_te_Y, edge_index, lookback, batch_size):
    """Per pair predictions on test set. Returns dict pair_idx -> (trues, preds)."""
    dl = DataLoader(GraphSeqDataset(g_te_X, g_te_Y, lookback),
                    batch_size=batch_size, shuffle=False)
    edge_index = edge_index.to(device)                    # Moves the graph to GPU
    model.eval()
    n_pairs = len(pair_list)
    preds = {i: [] for i in range(n_pairs)}
    trues = {i: [] for i in range(n_pairs)}
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)                              # Moves ecah batch to GPU
            out = model(x, edge_index, None).argmax(-1)   # Predicted class per pair
            out = out.cpu()                               # Back to the CPU for results collection

            # Splits the prediction and actual values
            for i in range(n_pairs):
                preds[i] += out[:, i].tolist()
                trues[i] += y[:, i].tolist()
    return trues, preds


def run_tgcn_multiseed(g_tr_X, g_tr_Y, g_va_X, g_va_Y, g_te_X, g_te_Y,
                       edge_index, class_weights, config,
                       fixed=TGCN_FIXED, seeds=SEEDS, verbose=False):
    """Retrain the winning configuration across all seeds; per pair test Macro-F1
    mean ± std. Same configuration used for whichever edge_index (variant) is passed."""
    lookback = fixed["lookback"]
    bs = config["batch_size"]
    n_pairs = len(pair_list)

    # Collect the score for each pair across the seeds and a pooled score per specific seed
    per_pair_f1 = {i: [] for i in range(n_pairs)}
    pooled_f1 = []
    last_seed_preds = None   # keep last seed's per pair (trues, preds) for confusion matrices

    for seed in seeds:
      # Train the winning configuration under this seed
        set_seed(seed)
        model, _, _ = train_tgcn_one(
            g_tr_X, g_tr_Y, g_va_X, g_va_Y, edge_index, class_weights,
            hidden=config["hidden"], dropout=config["dropout"],
            batch_size=bs, fixed=fixed, verbose=False)

        # Retrieve the seeds test predictions per pair
        trues, preds = eval_tgcn_test_perpair(
            model, g_te_X, g_te_Y, edge_index, lookback, bs)
        last_seed_preds = (trues, preds)    # Keeps the last seed

        # Give a score for each pair in this seed and then merge all of the pair scores for a combined figure
        pooled_t, pooled_p = [], []
        for i in range(n_pairs):
            if trues[i]:
                per_pair_f1[i].append(f1_score(trues[i], preds[i], average="macro"))
                pooled_t += trues[i]; pooled_p += preds[i]
        if pooled_t:
            pooled_f1.append(f1_score(pooled_t, pooled_p, average="macro"))
        if verbose:
            print(f"    seed={seed} -> pooled_f1={pooled_f1[-1]:.4f}")


    # Aggregate the scores per pair across seeds
    per_pair = {}
    for i in range(n_pairs):
        arr = np.array(per_pair_f1[i], dtype=float)
        per_pair[pair_list[i]] = {
            "mean_f1": float(np.nanmean(arr)) if arr.size else float("nan"),
            "std_f1":  float(np.nanstd(arr))  if arr.size else float("nan"),
            "per_seed": arr.tolist(),
        }

    # Return the results: For each pair and the combined value
    return {
        "config": config,
        "per_pair": per_pair,                                  # per pair mean±std
        "pooled_mean_f1": float(np.nanmean(pooled_f1)),        # Combined value
        "pooled_std_f1":  float(np.nanstd(pooled_f1)),
        "pooled_per_seed": pooled_f1,
        "last_seed_preds": last_seed_preds,                    # For the confusion matricies
    }



"""
==================== MODEL TRAINING AND EVALUATION==========================
"""

# ===== [Imports] ========================================
from sklearn.metrics import precision_score, recall_score



# ===== [Model results] ========================================

# Containers to collect the results across all folds

results_rows = []
detailed_rows = []

all_lstm_preds = {}
all_tgcn_preds = {}
loss_histories = {"LSTM": {}, "T-GCN": {}}  # Stores Training/Validation curves per fold


for fold in FOLDS:
    print("=" * 70)
    print(fold["name"])
    print("=" * 70)
    # Splits and scales this fold's training data
    r_tr, l_tr, r_va, l_va, r_te, l_te = get_fold_splits(returns, labels, fold)
    r_tr_s, r_va_s, r_te_s, scaler = scale_fold(r_tr, r_va, r_te)


    # Runs the three baselines: Persistence, Majority, Logistic Regression for every pair
    for pair in r_tr.columns:
        trues_p, preds_p = evaluate_persistence_baseline(l_te[pair])
        f1_p = f1_score(trues_p, preds_p, average="macro")
        results_rows.append({"fold": fold["name"], "pair": pair,
                              "model": "Persistence", "test_macro_f1": round(f1_p, 4)})

    for pair in r_tr.columns:
        trues_m, preds_m = evaluate_majority_baseline(l_tr[pair], l_te[pair])
        f1_m = f1_score(trues_m, preds_m, average="macro")
        results_rows.append({"fold": fold["name"], "pair": pair,
                              "model": "Majority", "test_macro_f1": round(f1_m, 4)})

    for pair in r_tr.columns:
        f1_lr, acc_lr = train_eval_logreg(pair, r_tr_s, l_tr, r_te_s, l_te)
        results_rows.append({"fold": fold["name"], "pair": pair,
                              "model": "LogReg", "test_macro_f1": round(f1_lr, 4)})


    # LSTM Grid search per pair results: Searches for best config, retrain across seeds

    fold_lstm_models = {}
    loss_histories["LSTM"][fold["name"]] = {}
    for pair in r_tr.columns:
        best = search_lstm_pair(pair, r_tr_s, l_tr, r_va_s, l_va, verbose=False)
        best_cfg = best["config"]

        ms = run_lstm_multiseed(pair, r_tr_s, l_tr, r_va_s, l_va, r_te_s, l_te,
                                config=best_cfg, verbose=False)

        # Per-pair LSTM results:
        results_rows.append({
            "fold": fold["name"], "pair": pair, "model": "LSTM",
            "test_macro_f1": round(ms["mean_f1"], 4),
            "test_macro_f1_std": round(ms["std_f1"], 4),
            "hidden": best_cfg["hidden"], "dropout": best_cfg["dropout"],
            "batch_size": best_cfg["batch_size"],
            "val_macro_f1": round(best["val_f1"], 4),
            "per_seed_f1": [round(x, 4) for x in ms["per_seed"]],
        })


        # Representative model for the confusion matrices and loss curves
        set_seed(SEEDS[-1])
        model, tr_hist, va_hist = train_one(
            pair, r_tr_s, l_tr, r_va_s, l_va,
            lookback=LSTM_FIXED["lookback"], hidden=best_cfg["hidden"],
            dropout=best_cfg["dropout"], num_layers=LSTM_FIXED["num_layers"],
            batch_size=best_cfg["batch_size"], lr=LSTM_FIXED["lr"],
            epochs=LSTM_FIXED["epochs"], patience=LSTM_FIXED["patience"], verbose=False)
        fold_lstm_models[pair] = model
        loss_histories["LSTM"][fold["name"]][pair] = (tr_hist, va_hist)

        # Prints pair's chosen configuration and validation/test scores
        print(f"  {pair}: best={best_cfg}  val_f1={best['val_f1']:.3f}  "
              f"test_f1={ms['mean_f1']:.3f}±{ms['std_f1']:.3f}")


    # T-GCN Grid search: Grid search each currency pair
    # Multi seed the winner for both T-GCN variants (Full graph and Self loops ablation)
    X_tr = build_node_features(r_tr_s); Y_tr_edge = build_edge_labels(l_tr)
    X_va = build_node_features(r_va_s); Y_va_edge = build_edge_labels(l_va)
    X_te = build_node_features(r_te_s); Y_te_edge = build_edge_labels(l_te)

    # Graph node features and edge label shape checks
    assert X_tr.shape[1:] == (len(nodes), 1), f"Unexpected node feature shape: {X_tr.shape}"
    assert Y_tr_edge.shape[1] == len(pair_list), f"Unexpected edge label shape: {Y_tr_edge.shape}"

    tgcn_class_weights = get_class_weights(Y_tr_edge.reshape(-1))

    # 1) Grid search on the T-GCN full graph (validation only)
    tgcn_best = search_tgcn(X_tr, Y_tr_edge, X_va, Y_va_edge,
                            edge_index, tgcn_class_weights, verbose=False)
    tgcn_cfg = tgcn_best["config"]
    print(f"  T-GCN best config: {tgcn_cfg}  val_f1={tgcn_best['val_f1']:.3f}")

    # 2) Multiseed the winner for both variants (Self loops variant reuses the Full graph config)
    for variant_name, variant_edges in [("T-GCN-full", edge_index),
                                        ("T-GCN-selfloops", self_loop_edge_index)]:
        ms_t = run_tgcn_multiseed(
            X_tr, Y_tr_edge, X_va, Y_va_edge, X_te, Y_te_edge,
            variant_edges, tgcn_class_weights, config=tgcn_cfg, verbose=False)

        # Record per pair T-GCN results
        for pair_name, pp in ms_t["per_pair"].items():
            results_rows.append({
                "fold": fold["name"], "pair": pair_name, "model": variant_name,
                "test_macro_f1": round(pp["mean_f1"], 4),
                "test_macro_f1_std": round(pp["std_f1"], 4),
                "hidden": tgcn_cfg["hidden"], "dropout": tgcn_cfg["dropout"],
                "batch_size": tgcn_cfg["batch_size"],
                "val_macro_f1": round(tgcn_best["val_f1"], 4),
                "per_seed_f1": [round(x, 4) for x in pp["per_seed"]],
            })

        # Record combines result for current T-GCN variant
        results_rows.append({
            "fold": fold["name"], "pair": "ALL (pooled)", "model": variant_name,
            "test_macro_f1": round(ms_t["pooled_mean_f1"], 4),
            "test_macro_f1_std": round(ms_t["pooled_std_f1"], 4),
            "hidden": tgcn_cfg["hidden"], "dropout": tgcn_cfg["dropout"],
            "batch_size": tgcn_cfg["batch_size"],
        })

        print(f"    {variant_name}: pooled test_f1={ms_t['pooled_mean_f1']:.3f}"
              f"±{ms_t['pooled_std_f1']:.3f}")

        # Store this variant's per-pair predictions for confusion matrices later.
        all_tgcn_preds.setdefault(fold["name"], {})[variant_name] = ms_t["last_seed_preds"]


    # ---- T-GCN representative loss curves & detailed metrics ----
    loss_histories["T-GCN"][fold["name"]] = {}
    for variant_name, variant_edges in [("T-GCN-full", edge_index),
                                        ("T-GCN-selfloops", self_loop_edge_index)]:

        # Retrain one representative model (last seed) and keep the loss curves
        set_seed(SEEDS[-1])
        rep_model, tr_hist, va_hist = train_tgcn_one(
            X_tr, Y_tr_edge, X_va, Y_va_edge,
            variant_edges, tgcn_class_weights,
            hidden=tgcn_cfg["hidden"], dropout=tgcn_cfg["dropout"],
            batch_size=tgcn_cfg["batch_size"], verbose=False)
        loss_histories["T-GCN"][fold["name"]][variant_name] = (tr_hist, va_hist)

        # Detailed metrics from the last-seed predictions already stored
        trues_dict, preds_dict = all_tgcn_preds[fold["name"]][variant_name]
        for i, pair_name in enumerate(pair_list):
            f1_per_class = f1_score(trues_dict[i], preds_dict[i],
                                    average=None, labels=[0, 1, 2])
            detailed_rows.append({
                "fold": fold["name"], "pair": pair_name, "model": variant_name,
                "accuracy": accuracy_score(trues_dict[i], preds_dict[i]),
                "precision_macro": precision_score(trues_dict[i], preds_dict[i],
                                                   average="macro", zero_division=0),
                "recall_macro": recall_score(trues_dict[i], preds_dict[i],
                                             average="macro", zero_division=0),
                "f1_DOWN": f1_per_class[0],
                "f1_NEUTRAL": f1_per_class[1],
                "f1_UP": f1_per_class[2],
            })

    # Stores LSTM predictions per fold for Confusion matrices and detailed metrics
    for pair in fold_lstm_models.keys():
        m = fold_lstm_models[pair]
        ds = FXDataset(r_te_s[pair], l_te[pair], LOOKBACK)
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        m.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl:
                preds += m(xb).argmax(1).tolist(); trues += yb.tolist()
        all_lstm_preds.setdefault(fold["name"], {})[pair] = (trues, preds)

        f1_per_class = f1_score(trues, preds, average=None, labels=[0, 1, 2])
        detailed_rows.append({
            "fold": fold["name"], "pair": pair, "model": "LSTM",
            "accuracy": accuracy_score(trues, preds),
            "precision_macro": precision_score(trues, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(trues, preds, average="macro", zero_division=0),
            "f1_DOWN": f1_per_class[0], "f1_NEUTRAL": f1_per_class[1], "f1_UP": f1_per_class[2],
        })

# Assembles the collected results into DataFrames and save them to CSV
results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_results.csv"), index=False)

detailed_df = pd.DataFrame(detailed_rows)
detailed_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_metrics.csv"), index=False)

# Save raw predictions and loss histories to disk so the confusion matrices and
# loss curves can be rebuilt later (e.g. downloaded from Barkla and loaded in Colab).
import pickle
with open(os.path.join(OUTPUT_DIR, "predictions.pkl"), "wb") as f:
    pickle.dump({
        "all_lstm_preds": all_lstm_preds,
        "all_tgcn_preds": all_tgcn_preds,
        "loss_histories": loss_histories,
    }, f)
print("Saved predictions.pkl")


