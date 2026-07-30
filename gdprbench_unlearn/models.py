from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    loops = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loops], dim=1)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        edge_index = add_self_loops(edge_index, n)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        h = self.lin(x)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * norm.unsqueeze(-1))
        return out


class SAGEConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        neigh = agg / deg.clamp(min=1).unsqueeze(-1)
        return self.lin_self(x) + self.lin_neigh(neigh)


class GATLayer(nn.Module):
    """Full-batch GAT layer implemented with native torch scatter_reduce.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, concat: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        edge_index = add_self_loops(edge_index, n)
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x).view(n, self.heads, self.out_dim)
        h_src = h[src]
        h_dst = h[dst]
        e = (h_src * self.att_src.unsqueeze(0)).sum(-1) + (h_dst * self.att_dst.unsqueeze(0)).sum(-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Segment softmax over incoming edges per destination node and head.
        idx = dst.view(-1, 1).expand(-1, self.heads)
        max_per_dst = torch.full((n, self.heads), -torch.inf, device=x.device, dtype=x.dtype)
        max_per_dst.scatter_reduce_(0, idx, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_dst[dst])
        denom = torch.zeros((n, self.heads), device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / denom[dst].clamp(min=1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((n, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, h_src * alpha.unsqueeze(-1))
        if self.concat:
            out = out.reshape(n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class GNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        backbone: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.backbone = backbone.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        layers: List[nn.Module] = []

        if self.backbone == "gcn":
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(num_layers):
                layers.append(GCNLayer(dims[i], dims[i + 1]))
        elif self.backbone == "sage":
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(num_layers):
                layers.append(SAGEConvLayer(dims[i], dims[i + 1]))
        elif self.backbone == "gat":
            # Hidden GAT layers concatenate heads; final layer averages heads.
            first_hidden = max(1, hidden_dim // heads)
            layers.append(GATLayer(in_dim, first_hidden, heads=heads, concat=True, dropout=dropout))
            current_dim = first_hidden * heads
            for _ in range(num_layers - 2):
                h_per_head = max(1, hidden_dim // heads)
                layers.append(GATLayer(current_dim, h_per_head, heads=heads, concat=True, dropout=dropout))
                current_dim = h_per_head * heads
            layers.append(GATLayer(current_dim, out_dim, heads=heads, concat=False, dropout=dropout))
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Use gcn, sage, or gat.")
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def last_layer_parameters(self):
        return self.layers[-1].parameters()


def build_model(
    backbone: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    heads: int = 2,
) -> GNN:
    return GNN(in_dim, hidden_dim, out_dim, backbone=backbone, num_layers=num_layers, dropout=dropout, heads=heads)
