from __future__ import annotations

import networkx as nx
import pytest

from blockchain_fraud.features import compute_graph_features


def test_compute_graph_features_normalised():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

    features = compute_graph_features(graph)
    assert set(features.columns) == {
        "node",
        "degree",
        "in_degree",
        "out_degree",
        "pagerank",
        "betweenness",
        "clustering",
    }
    assert len(features) == graph.number_of_nodes()
    assert features["degree"].mean() == pytest.approx(0.0, abs=1e-6)


