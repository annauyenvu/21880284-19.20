# Blockchain Fraud Detection Toolkit

This repository contains utilities to experiment with financial fraud detection on
blockchain transaction graphs. The codebase focuses on the [Elliptic Bitcoin
transaction dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
and provides:

- **Data ingestion** helpers that parse the CSV files published by Elliptic and
  expose them as PyTorch-ready tensors.
- **Graph feature engineering** utilities such as degree-based metrics and
  PageRank to complement neural models.
- **Graph neural network (GNN) models** including Graph Convolutional Networks
  (GCN) and GraphSAGE implemented directly in PyTorch.
- **Training loops** with temporal splits that mirror the evaluation protocol
  commonly used for the dataset.
- **Command line scripts** and unit tests that demonstrate how to use the
  components on synthetic data.

## Project structure

```
blockchain_fraud/
├── __init__.py
├── data.py              # Elliptic dataset loader and temporal slicing helpers
├── features.py          # Centrality and clustering feature extraction
├── models.py            # GCN and GraphSAGE implementations
├── training.py          # Training loop with early stopping utilities
└── scripts/
    └── train_elliptic.py  # CLI to train models on the Elliptic dataset
```

The `tests/` directory contains unit tests covering the main modules. They build
small synthetic graphs so that the tests run quickly without requiring the full
Elliptic dataset.

## Getting started

1. Install the Python dependencies in a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

   The toolkit depends on `torch`, `pandas`, `numpy`, `networkx`, `scipy` and `pytest`.

2. Download the Elliptic dataset from Kaggle and extract the CSV files into a
   directory, e.g. `data/elliptic`.

3. Train a model:

   ```bash
   python -m blockchain_fraud.scripts.train_elliptic data/elliptic --model sage --epochs 100
   ```

   The script loads the dataset, creates temporal train/validation/test splits
   and trains the selected model. Metrics for each split are printed at the end
   of training.

## Extending to real blockchain data

The modules are intentionally lightweight so they can be repurposed for datasets
collected directly from public blockchains (Ethereum, Bitcoin, etc.). After
constructing a graph with node features and heuristic labels, wrap the tensors
into an `EllipticData` instance and reuse the provided GNNs and training loop.

## Running the tests

Execute the unit tests with:

```bash
pytest
```

The test suite ensures the data loader, feature computation and training loop
work on controlled toy examples.
