"""Command line utility to train GNN models on the Elliptic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data import load_elliptic_dataset
from ..models import GCNClassifier, GraphSAGEClassifier
from ..training import temporal_split_masks, train_model, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Path to the directory containing the Elliptic CSV files")
    parser.add_argument(
        "--model",
        choices=["gcn", "sage"],
        default="gcn",
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def build_model(args: argparse.Namespace, in_channels: int) -> torch.nn.Module:
    if args.model == "gcn":
        return GCNClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
        )
    return GraphSAGEClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    data = load_elliptic_dataset(args.root)

    model = build_model(args, data.features.size(1))
    masks = temporal_split_masks(data)
    result = train_model(
        model,
        data,
        masks,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    metrics = {split: evaluate_model(result.model, data, mask) for split, mask in masks.items()}
    print(f"Best epoch: {result.best_epoch}")
    for split, stats in metrics.items():
        print(f"{split.capitalize()} metrics: {stats}")


if __name__ == "__main__":
    main()
