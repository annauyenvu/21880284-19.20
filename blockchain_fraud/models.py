"""Light-weight Graph Neural Network models used for blockchain fraud detection."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required to use blockchain_fraud.models; install it via pip install torch"
    ) from exc
from torch import nn
from torch.nn import functional as F


def _add_self_loops(edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([self_loops, self_loops], dim=0)
    return torch.cat([edge_index, loop_index], dim=1)


def _normalized_adj(edge_index: torch.LongTensor, num_nodes: int) -> torch.sparse.FloatTensor:
    edge_index = _add_self_loops(edge_index, num_nodes)
    row, col = edge_index
    values = torch.ones(row.size(0), device=edge_index.device)
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.index_add_(0, row, values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(torch.vstack([row, col]), norm_values, (num_nodes, num_nodes))
    return adj.coalesce()


def _mean_aggregate(x: torch.Tensor, edge_index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    aggregated = torch.zeros_like(x)
    aggregated.index_add_(0, dst, x[src])
    deg = torch.zeros(num_nodes, device=x.device)
    deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
    deg = deg.unsqueeze(-1).clamp_min_(1.0)
    return aggregated / deg


class GraphConvolution(nn.Module):
    """Implementation of the spectral GCN layer proposed by Kipf & Welling."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = x.size(0)
        adj = _normalized_adj(edge_index, num_nodes)
        x = self.dropout(x)
        support = self.linear(x)
        out = torch.sparse.mm(adj, support)
        return out


class SAGEConv(nn.Module):
    """Mean-aggregator GraphSAGE layer."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.linear_self = nn.Linear(in_channels, out_channels)
        self.linear_neigh = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = x.size(0)
        x = self.dropout(x)
        aggregated = _mean_aggregate(x, edge_index, num_nodes)
        out = self.linear_self(x) + self.linear_neigh(aggregated)
        return out


class _BaseClassifier(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = nn.Dropout(dropout)
        self.hidden_channels = hidden_channels
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    def _predict(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits.squeeze(-1))


class GCNClassifier(_BaseClassifier):
    """Two-headed GCN classifier for binary node classification."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout
        )

        layers = []
        last_channels = in_channels
        for layer_idx in range(num_layers - 1):
            layers.append(GraphConvolution(last_channels, hidden_channels, dropout=dropout))
            last_channels = hidden_channels
        layers.append(GraphConvolution(last_channels, 1, dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
        logits = self.layers[-1](x, edge_index)
        return logits


class GraphSAGEClassifier(_BaseClassifier):
    """GraphSAGE classifier for binary node classification."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout
        )

        layers = []
        last_channels = in_channels
        for layer_idx in range(num_layers - 1):
            layers.append(SAGEConv(last_channels, hidden_channels, dropout=dropout))
            last_channels = hidden_channels
        layers.append(SAGEConv(last_channels, 1, dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
        logits = self.layers[-1](x, edge_index)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
