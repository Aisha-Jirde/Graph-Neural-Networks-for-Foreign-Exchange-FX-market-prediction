# Graph Neural Networks for Foreign Exchange (FX) Market Prediction

MSc dissertation (COMP702, University of Liverpool, 2025/26) by Aisha Jirde, supervised by Andrew Roxburgh.

## Overview

Most FX forecasting models treat each currency pair as an isolated time series, ignoring the dependencies between currencies. This project models the FX market as a graph: six currencies (USD, EUR, GBP, JPY, CNY, AUD) are nodes and seven tradable pairs are edges. A Temporal Graph Convolutional Network (T-GCN) predicts the next-day direction (UP / NEUTRAL / DOWN) of every pair at once.

**Research question:** Does modelling cross-currency structure improve short-term directional forecasting, and does it hold across different market conditions?

## Models compared

- **Baselines:** Persistence (repeat yesterday's direction), Majority class, and Logistic Regression on a 20-day return window
- **LSTM:** a sequence model trained independently for each currency pair
- **T-GCN-full:** a temporal graph model trained jointly on the real currency graph (all 7 pair edges)
- **T-GCN-selfloops:** an ablation where each currency only sees itself, isolating how much of the performance comes from cross-currency information flow

## Evaluation

Walk-forward validation across four market regimes (2018-2023), five random seeds, macro-F1 as the main metric, and Wilcoxon signed-rank tests over 28 matched pair-fold comparisons.

## Key findings

- The full T-GCN did **not** significantly outperform the LSTM (p = 0.063).
- The self-loops ablation **did** significantly outperform the LSTM (p = 0.007).
- The advantage therefore appears to come from the T-GCN's temporal architecture and joint training, not from the cross-currency graph. The static, unweighted graph with single-feature nodes appears to add noise rather than signal at this prediction horizon.
- Conclusion: static, hand-designed currency graphs are not straightforwardly advantageous for short-term FX directional forecasting.

The full write-up is in [`dissertation/Dissertation.pdf`](dissertation/Dissertation.pdf).

## Repository contents

| File | Description |
|---|---|
| `fx_project.py` | Full training and evaluation pipeline. Loads cleaned OHLCV data, builds walk-forward splits, trains baselines, LSTM and T-GCN with grid search across 5 seeds, and writes results to CSV/pickle. Built for the University of Liverpool's Barkla HPC cluster. |
| `Model_Analysis.ipynb` | Analysis notebook (built in Google Colab). Reproduces baseline predictions and generates confusion matrices, loss curves, summary tables and Wilcoxon tests. |
| `cleaned_fx_data.csv` | Pre-cleaned daily OHLCV data (2018-2023) for 7 FX pairs, sourced from Yahoo Finance. Input to `fx_project.py`. |
| `walk_forward_results.csv` | Per-pair, per-fold macro-F1 for every model, including winning hyperparameters and per-seed scores. |
| `detailed_metrics.csv` | Per-pair, per-fold accuracy, precision, recall and per-class (DOWN/NEUTRAL/UP) F1 for the LSTM and T-GCN variants. |
| `predictions.pkl` | True and predicted labels plus training/validation loss curves, so plots can be rebuilt without retraining. |

## How to reproduce

### 1. Training (Barkla HPC or any GPU machine)

```bash
python fx_project.py
```

Requires `torch`, `torch_geometric_temporal`, `pandas`, `scikit-learn` and `numpy`. Edit `DATA_PATH` at the top of the script to point to `cleaned_fx_data.csv` (it currently points to a Barkla scratch path). Outputs `walk_forward_results.csv`, `detailed_metrics.csv` and `predictions.pkl` to `OUTPUT_DIR`.

The grid search (LSTM: 36 configs per pair; T-GCN: 36 configs per fold) x 5 seeds x 4 folds is computationally expensive, which is why it was run on HPC.

### 2. Analysis (Google Colab or Jupyter)

Open `Model_Analysis.ipynb` and upload `cleaned_fx_data.csv`, `walk_forward_results.csv`, `detailed_metrics.csv` and `predictions.pkl`. The notebook produces:

- Confusion matrices per model, fold and pair
- Training vs validation loss curves
- Summary macro-F1 tables (overall, per fold, per pair)
- Head-to-head win-count comparison between the LSTM and both T-GCN variants
- Wilcoxon signed-rank significance tests

## Licence

Code is released under the MIT licence (see `LICENSE`). Dissertation text © Aisha Jirde.
