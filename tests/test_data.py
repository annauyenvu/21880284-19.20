from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from blockchain_fraud.data import EllipticData, build_temporal_slices, elliptic_subset, load_elliptic_dataset


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(str(value) for value in row))
            handle.write("\n")


def test_load_elliptic_dataset(tmp_path):
    features = [
        (1, 1, 0.1, 1.0),
        (2, 1, 0.2, 0.9),
        (3, 2, 0.3, 0.8),
    ]
    edges = [(1, 2), (2, 3)]
    classes = [(1, 1), (2, 2), (3, 3)]

    _write_csv(tmp_path / "elliptic_txs_features.csv", features)
    _write_csv(tmp_path / "elliptic_txs_edgelist.csv", edges)
    _write_csv(tmp_path / "elliptic_txs_classes.csv", classes)

    data = load_elliptic_dataset(tmp_path)

    assert isinstance(data, EllipticData)
    assert data.features.shape == (3, 2)
    assert data.edge_index.shape == (2, 2)
    assert torch.equal(data.labels, torch.tensor([0, 1, -1]))
    assert torch.equal(data.label_mask, torch.tensor([True, True, False]))

    slices = build_temporal_slices(data, include_unlabeled=False)
    assert len(slices) == 1  # only time step 1 remains because node 3 is unlabeled
    assert slices[0].time == 1
    assert slices[0].edge_index.shape[1] == 1

    subset_x, subset_y, subset_t = elliptic_subset(data, [0, 2])
    assert subset_x.shape == (2, 2)
    assert torch.equal(subset_y, torch.tensor([0, -1]))
    assert torch.equal(subset_t, torch.tensor([1, 2]))
