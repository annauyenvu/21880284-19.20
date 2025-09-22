from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blockchain_fraud.data import EllipticData
from blockchain_fraud.models import GCNClassifier
from blockchain_fraud.training import evaluate_model, temporal_split_masks, train_model


def _make_dataset() -> EllipticData:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    times = torch.tensor([1, 1, 36, 45], dtype=torch.long)
    node_ids = np.arange(4)
    label_mask = torch.ones(4, dtype=torch.bool)

    return EllipticData(
        edge_index=edge_index,
        features=features,
        labels=labels,
        times=times,
        node_ids=node_ids,
        label_mask=label_mask,
    )


def test_train_model_runs():
    data = _make_dataset()
    masks = temporal_split_masks(data)
    model = GCNClassifier(in_channels=data.features.size(1), hidden_channels=4, num_layers=2)
    result = train_model(
        model,
        data,
        masks,
        epochs=30,
        learning_rate=0.05,
        weight_decay=0.0,
        patience=10,
    )

    assert result.best_epoch > 0
    assert len(result.history["train_loss"]) >= 1

    metrics = evaluate_model(result.model, data, masks["test"])
    assert set(metrics) == {"loss", "accuracy", "f1"}
