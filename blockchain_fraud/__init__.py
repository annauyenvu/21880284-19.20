"""Utilities for blockchain-based financial fraud detection using graph learning."""

__all__ = [
    "EllipticData",
    "load_elliptic_dataset",
    "build_temporal_slices",
    "compute_graph_features",
    "GCNClassifier",
    "GraphSAGEClassifier",
    "temporal_split_masks",
    "train_model",
    "evaluate_model",
]


def __getattr__(name):
    if name == "compute_graph_features":
        from .features import compute_graph_features as _compute_graph_features

        return _compute_graph_features

    if name in {
        "EllipticData",
        "load_elliptic_dataset",
        "build_temporal_slices",
        "GCNClassifier",
        "GraphSAGEClassifier",
        "temporal_split_masks",
        "train_model",
        "evaluate_model",
    }:
        from . import data, models, training

        exports = {
            "EllipticData": data.EllipticData,
            "load_elliptic_dataset": data.load_elliptic_dataset,
            "build_temporal_slices": data.build_temporal_slices,
            "GCNClassifier": models.GCNClassifier,
            "GraphSAGEClassifier": models.GraphSAGEClassifier,
            "temporal_split_masks": training.temporal_split_masks,
            "train_model": training.train_model,
            "evaluate_model": training.evaluate_model,
        }
        return exports[name]

    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
