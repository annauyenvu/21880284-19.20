"""Training utilities for graph neural networks on blockchain transaction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required to use blockchain_fraud.training; install it via pip install torch"
    ) from exc
from torch import nn
from torch.optim import Optimizer

from .data import EllipticData


@dataclass
class TrainingResult:
    model: nn.Module
    history: Dict[str, list]
    best_epoch: int


def temporal_split_masks(
    data: EllipticData,
    *,
    train_range: Tuple[int, int] = (1, 34),
    val_range: Tuple[int, int] = (35, 41),
    test_range: Tuple[int, int] = (42, 49),
) -> Dict[str, torch.BoolTensor]:
    """Create boolean masks splitting nodes by temporal ranges."""

    times = data.times
    label_mask = data.label_mask

    def _mask(time_range: Tuple[int, int]) -> torch.BoolTensor:
        start, end = time_range
        mask = (times >= start) & (times <= end) & label_mask
        return mask

    return {
        "train": _mask(train_range),
        "val": _mask(val_range),
        "test": _mask(test_range),
    }


def _binary_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_labels = (pred >= 0.5).long()
    tp = ((pred_labels == 1) & (target == 1)).sum().item()
    fp = ((pred_labels == 1) & (target == 0)).sum().item()
    fn = ((pred_labels == 0) & (target == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def evaluate_model(model: nn.Module, data: EllipticData, mask: torch.BoolTensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.features, data.edge_index)
        prob = torch.sigmoid(logits.squeeze(-1))
        labels = data.labels
        mask = mask & data.label_mask
        prob = prob[mask]
        labels = labels[mask]
        loss = nn.functional.binary_cross_entropy(prob, labels.float())
        acc = ((prob >= 0.5).long() == labels).float().mean().item()
        f1 = _binary_f1(prob, labels)
    return {"loss": float(loss.item()), "accuracy": float(acc), "f1": f1}


def train_model(
    model: nn.Module,
    data: EllipticData,
    masks: Dict[str, torch.BoolTensor],
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    optimizer_class: type[Optimizer] = torch.optim.Adam,
) -> TrainingResult:
    """Train a graph neural network for binary classification."""

    optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_state = None
    best_epoch = -1
    best_metric = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.features, data.edge_index).squeeze(-1)
        train_labels = data.labels[masks["train"]]
        train_logits = logits[masks["train"]]
        loss = criterion(train_logits, train_labels.float())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_metrics = evaluate_model(model, data, masks["val"])

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["f1"] > best_metric:
            best_metric = val_metrics["f1"]
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(model=model, history=history, best_epoch=best_epoch)
