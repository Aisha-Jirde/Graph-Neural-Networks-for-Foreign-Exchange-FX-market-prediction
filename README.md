# Graph Neural Networks for Foreign Exchange (FX) Market Prediction

MSc dissertation project modelling the FX market as a graph with currencies as nodes, currency pairs as edges in order to to predict directional movement in the form of  (UP / NEUTRAL / DOWN). A Temporal Graph Convolutional Network (T-GCN) is trained to learn cross-currency relationships and evaluated against an Long Short Term Memory (LSTM) and three classical baselines: Persistence, Majority and Logistic Regression using walk-forward validation across four distinct market regimes from 2018 to 2023.

Author: Aisha Jirde (201386550)
Supervisor Andrew Roxburgh
Module COMP702 MSc Project, University of Liverpool

# Summary Of project

The majority of FX forecasting treats each currency pair as an isolated time series. This project instead represents the market as a graph using 6 currency nodes (USD, EUR, GBP, JPY, CNY, AUD) connected by 7 traded pairs. The goal is that a model can learn from relationships between the currencies and not just each pair's own history.

My reasearch question is:

Does modelling cross currency structure improve short term directional forecasting and is it effective across different market conditions?. To answer thsi question I use the models below


Baselines - Persistence (repeat yesterday's direction), Majority class, and Logistic Regression on a 20 dayreturn window.

LSTM - A sequence model trained independently per currency pair.
T-GCN - A temporal graph model trained jointly across all pairs, in two variants:
  `T-GCN-full`  the real currency graph (all 7 pair edges).
  `T-GCN-selfloops` an ablation where each currency only sees itself, isolating how much of the T-GCN's performance actually comes from cross currency information flow.

Files used in my project:

# Result visualistaion

Model_Analysis.ipynb for the full statistical comparison, breakdown, confusion matricies, loss curves and results tables.

# Repository contents


fx_project.py- Full training and evaluation pipeline. Loads cleaned OHLCV data, builds walk forward splits, trains baselines/LSTM/T-GCN with grid search across 5 random seeds, and writes results to CSV/pickle. Built to run on the University of Liverpool's Barkla HPC cluster


Model_Analysis.ipynb - Analysis and visualisation notebook built in Google Colab. Loads the outputs of fx_project.py, reproduces the baseline predictions, and generates confusion matrices, loss curves, summary tables, and the Wilcoxon significance tests. 


cleaned_fx_data.csv- Pre-cleaned daily OHLCV data (2018-2023) for 7 FX pairs, multi-index columns (Close/High/Low/Open/Volume). Input to fx_project.py


walk_forward_results.csv- Per-pair, per-fold macro-F1 results for every model, including the winning hyperparameter configuration and per-seed scores. Main output of fx_project.py


detailed_metrics.csv- Per-pair, per-fold accuracy, precision, recall, and per-class (DOWN/NEUTRAL/UP) F1 for the LSTM and T-GCN variants

predictions.pkl - Raw true & predicted labels and training/validation loss curves for the LSTM and T-GCN models, saved so confusion matrices and loss plots can be rebuilt without retraining.



# How to reproduce

# 1. Training (Barkla HPC / any GPU machine)

```bash
python fx_project.py
```

Requires `torch`, `torch_geometric_temporal`, `pandas`, `scikit-learn`, `numpy`. Expects `cleaned_fx_data.csv` at the path set in `DATA_PATH` (edit this at the top of the script for your environment  it currently points to a Barkla scratch path). Outputs `walk_forward_results.csv`, `detailed_metrics.csv`, and `predictions.pkl` to `OUTPUT_DIR`.

Training runs a full grid search (LSTM: 36 configs/pair; T-GCN: 36 configs/fold) × 5 seeds × 4 folds. It is computationally expensive so this is why it was run on HPC rather than locally.

# 2. Analysis (Google Colab / Jupyter)

Open `Model_Analysis.ipynb` and upload the four output files (`cleaned_fx_data.csv`, `walk_forward_results.csv`, `detailed_metrics.csv`, `predictions.pkl`). The notebook regenerates baseline predictions, then produces:

- Confusion matrices per model/fold/pair
- Training vs validation loss curves
- Summary macro-F1 tables (overall, per fold, per pair)
- Head-to-head win-count comparison between LSTM and both T-GCN variants
- Wilcoxon signed-rank significance tests

