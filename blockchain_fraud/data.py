"""Data loading utilities for blockchain fraud detection datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - handled during optional import
    raise ModuleNotFoundError(
        "PyTorch is required to use blockchain_fraud.data; install it via pip install torch"
    ) from exc

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - optional dependency for runtime use only
    nx = None  # type: ignore


@dataclass
class TemporalSlice:
    """Container describing a temporal snapshot of a dynamic transaction graph."""

    time: int
    nodes: np.ndarray
    edge_index: torch.LongTensor


@dataclass
class EllipticData:
    """Container for the Elliptic Bitcoin transaction dataset.

    Attributes
    ----------
    edge_index:
        A tensor of shape ``(2, num_edges)`` containing directed edges.
    features:
        Node feature matrix with shape ``(num_nodes, num_features)``.
    labels:
        Tensor of integer labels. Fraudulent nodes are labelled ``1`` and licit
        nodes ``0``. Unknown nodes are encoded as ``-1``.
    times:
        Tensor containing the temporal step assigned to each node.
    node_ids:
        Original node identifiers as provided by the dataset.
    label_mask:
        Boolean tensor indicating nodes with a known label.
    """

    edge_index: torch.LongTensor
    features: torch.FloatTensor
    labels: torch.LongTensor
    times: torch.LongTensor
    node_ids: np.ndarray
    label_mask: torch.BoolTensor

    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_networkx(self, create_using: Optional["nx.DiGraph"] = None):  # type: ignore[name-defined]
        """Build a NetworkX graph from the dataset.

        Parameters
        ----------
        create_using:
            The type of graph to create. Defaults to :class:`networkx.DiGraph` if
            NetworkX is installed.
        """

        if nx is None:
            raise ImportError("NetworkX is required to create a graph representation")

        graph = (create_using or nx.DiGraph)()
        mapping = {int(idx): int(node_id) for idx, node_id in enumerate(self.node_ids)}
        graph.add_nodes_from(int(node_id) for node_id in self.node_ids)
        src = self.edge_index[0].cpu().numpy()
        dst = self.edge_index[1].cpu().numpy()
        for s, d in zip(src, dst):
            graph.add_edge(int(mapping[int(s)]), int(mapping[int(d)]))
        return graph


LABEL_MAPPING: Dict[str, int] = {
    "1": 0,  # licit
    "2": 1,  # illicit
    "3": -1,  # unknown
    "unknown": -1,
    "licit": 0,
    "illicit": 1,
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return pd.read_csv(path, header=None)


def load_elliptic_dataset(root: Path | str) -> EllipticData:
    """Load the Elliptic Bitcoin transaction dataset.

    Parameters
    ----------
    root:
        Directory containing the CSV files ``elliptic_txs_features.csv``,
        ``elliptic_txs_edgelist.csv`` and ``elliptic_txs_classes.csv``.

    Returns
    -------
    :class:`EllipticData`
        Parsed dataset ready for consumption by PyTorch models.
    """

    root_path = Path(root)
    features_df = _read_csv(root_path / "elliptic_txs_features.csv")
    edges_df = _read_csv(root_path / "elliptic_txs_edgelist.csv")
    classes_df = _read_csv(root_path / "elliptic_txs_classes.csv")

    node_ids = features_df.iloc[:, 0].astype(int).to_numpy()
    times = features_df.iloc[:, 1].astype(int).to_numpy()
    feature_matrix = features_df.iloc[:, 2:].to_numpy(dtype=np.float32)

    class_map = classes_df.set_index(0)[1].astype(str)
    labels = np.array([LABEL_MAPPING.get(class_map.get(node_id, "unknown"), -1) for node_id in node_ids], dtype=np.int64)
    label_mask = labels >= 0

    # Build an index mapping for edges
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_sources = edges_df.iloc[:, 0].map(node_id_to_idx)
    edge_targets = edges_df.iloc[:, 1].map(node_id_to_idx)

    if edge_sources.isnull().any() or edge_targets.isnull().any():
        missing = set(edges_df.iloc[:, 0]).union(edges_df.iloc[:, 1]) - set(node_ids)
        raise ValueError(
            "Edge list contains node identifiers that are absent in the feature file: "
            f"{sorted(missing)[:10]}"
        )

    edge_index = torch.as_tensor(
        np.vstack([edge_sources.to_numpy(dtype=np.int64), edge_targets.to_numpy(dtype=np.int64)]),
        dtype=torch.long,
    )

    return EllipticData(
        edge_index=edge_index,
        features=torch.from_numpy(feature_matrix),
        labels=torch.from_numpy(labels),
        times=torch.from_numpy(times.astype(np.int64)),
        node_ids=node_ids,
        label_mask=torch.from_numpy(label_mask),
    )


def build_temporal_slices(data: EllipticData, *, include_unlabeled: bool = True) -> List[TemporalSlice]:
    """Split the dataset into temporal slices.

    Parameters
    ----------
    data:
        Dataset produced by :func:`load_elliptic_dataset`.
    include_unlabeled:
        Whether to retain nodes without labels. Setting this to ``False`` filters
        out nodes whose label is ``-1``.
    """

    times = data.times.cpu().numpy()
    if not include_unlabeled:
        mask = data.label_mask.cpu().numpy()
        candidate_indices = np.nonzero(mask)[0]
    else:
        candidate_indices = np.arange(data.num_nodes())

    slices: List[TemporalSlice] = []
    for time in np.unique(times[candidate_indices]):
        node_selector = np.where(times == time)[0]
        if not include_unlabeled:
            node_selector = np.intersect1d(node_selector, candidate_indices, assume_unique=True)
        if node_selector.size == 0:
            continue

        nodes = data.node_ids[node_selector]
        # Filter edges that connect nodes within the same temporal slice
        mask_edges = np.isin(data.edge_index[0].cpu().numpy(), node_selector) & np.isin(
            data.edge_index[1].cpu().numpy(), node_selector
        )
        edge_index = data.edge_index[:, mask_edges]
        slices.append(TemporalSlice(time=int(time), nodes=nodes, edge_index=edge_index))
    return slices


def elliptic_subset(
    data: EllipticData, node_indices: Sequence[int]
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
    """Return feature, label and time tensors for the provided node indices."""

    node_indices = torch.as_tensor(node_indices, dtype=torch.long)
    return (
        data.features[node_indices],
        data.labels[node_indices],
        data.times[node_indices],
    )
