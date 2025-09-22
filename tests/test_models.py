from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from blockchain_fraud.models import GCNClassifier, GraphSAGEClassifier


def _sample_graph():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    features = torch.randn(4, 5)
    return edge_index, features


def test_gcn_forward_shapes():
    edge_index, features = _sample_graph()
    model = GCNClassifier(in_channels=5, hidden_channels=8, num_layers=2)
    logits = model(features, edge_index)
    assert logits.shape == (4, 1)


def test_sage_forward_shapes():
    edge_index, features = _sample_graph()
    model = GraphSAGEClassifier(in_channels=5, hidden_channels=8, num_layers=2)
    logits = model(features, edge_index)
    assert logits.shape == (4, 1)
