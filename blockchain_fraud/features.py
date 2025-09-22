"""Graph-based feature engineering helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - optional dependency for runtime use only
    nx = None  # type: ignore


def compute_graph_features(
    graph: "nx.Graph",
    *,
    nodes: Optional[Iterable[int]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute classic graph centrality measures for the specified nodes.

    Parameters
    ----------
    graph:
        A NetworkX graph instance.
    nodes:
        Optional iterable of nodes. When ``None`` every node in ``graph`` is used.
    normalize:
        Whether to z-score normalise continuous features.
    """

    if nx is None:
        raise ImportError("NetworkX is required to compute graph features")

    if nodes is None:
        nodes = list(graph.nodes)
    else:
        nodes = list(nodes)

    if not nodes:
        raise ValueError("Graph does not contain any of the requested nodes")

    undirected = graph.to_undirected()
    degree = dict(graph.degree(nodes))
    in_degree = dict(graph.in_degree(nodes)) if graph.is_directed() else degree
    out_degree = dict(graph.out_degree(nodes)) if graph.is_directed() else degree

    pagerank = _safe_metric(nx.pagerank, graph, nodes)
    betweenness = _safe_metric(nx.betweenness_centrality, graph, nodes)
    clustering = _safe_metric(nx.clustering, undirected, nodes)

    frame = pd.DataFrame(
        {
            "node": nodes,
            "degree": [degree.get(node, 0.0) for node in nodes],
            "in_degree": [in_degree.get(node, 0.0) for node in nodes],
            "out_degree": [out_degree.get(node, 0.0) for node in nodes],
            "pagerank": [pagerank.get(node, 0.0) for node in nodes],
            "betweenness": [betweenness.get(node, 0.0) for node in nodes],
            "clustering": [clustering.get(node, 0.0) for node in nodes],
        }
    )

    if normalize:
        for column in ["degree", "in_degree", "out_degree", "pagerank", "betweenness", "clustering"]:
            values = frame[column].to_numpy(dtype=np.float32)
            if np.allclose(values.std(), 0):
                continue
            frame[column] = (values - values.mean()) / (values.std() + 1e-8)

    return frame


def _safe_metric(func, graph: "nx.Graph", nodes: Iterable[int]) -> Mapping[int, float]:
    try:
        return func(graph, nodes)
    except TypeError:
        # pagerank/betweenness do not accept ``nodes`` keyword in old NetworkX versions
        values = func(graph)
        return {node: values.get(node, 0.0) for node in nodes}
