from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


ERROR_CODE_TO_NAME = {
    0: "clean",
    1: "wrong_fact",
    2: "misattributed",
    3: "outdated",
    4: "invalid_combination",
    5: "incorrect_derived_label",  
}
ERROR_NAME_TO_CODE = {v: k for k, v in ERROR_CODE_TO_NAME.items()}
ERROR_NAME_TO_CODE["noisy_label"] = 5
ERROR_NAME_TO_CODE["invalid"] = 4
ERROR_NAME_TO_CODE["misattr"] = 2
ERROR_NAME_TO_CODE["wrong"] = 1


@dataclass
class GraphData:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    is_corrupted: torch.Tensor
    error_code: torch.Tensor
    num_classes: int
    npz_path: str

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges_directed(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            name=self.name,
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            is_corrupted=self.is_corrupted.to(device),
            error_code=self.error_code.to(device),
            num_classes=self.num_classes,
            npz_path=self.npz_path,
        )


def infer_name_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.replace("gdpr_", "")


def load_npz(path: str | Path, name: Optional[str] = None, mmap: bool = False) -> GraphData:
    path = Path(path)
    data = np.load(path, allow_pickle=True, mmap_mode="r" if mmap else None)
    required = ["x", "y", "edge_index", "train_mask", "val_mask", "test_mask"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Missing required keys in {path}: {missing}. Available: {data.files}")

    x = torch.as_tensor(np.asarray(data["x"]), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(data["y"]), dtype=torch.long)
    edge_index = torch.as_tensor(np.asarray(data["edge_index"]), dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    train_mask = torch.as_tensor(np.asarray(data["train_mask"]), dtype=torch.bool)
    val_mask = torch.as_tensor(np.asarray(data["val_mask"]), dtype=torch.bool)
    test_mask = torch.as_tensor(np.asarray(data["test_mask"]), dtype=torch.bool)

    if "error_code" in data.files:
        error_code = torch.as_tensor(np.asarray(data["error_code"]), dtype=torch.long)
    else:
        error_code = torch.zeros_like(y, dtype=torch.long)
    if "is_corrupted" in data.files:
        is_corrupted = torch.as_tensor(np.asarray(data["is_corrupted"]), dtype=torch.bool)
    else:
        is_corrupted = error_code > 0

    n = x.shape[0]
    for k, arr in [("y", y), ("train_mask", train_mask), ("val_mask", val_mask), ("test_mask", test_mask), ("error_code", error_code)]:
        if arr.shape[0] != n:
            raise ValueError(f"{k} length {arr.shape[0]} does not match x nodes {n}")

    num_classes = int(y.max().item() + 1) if y.numel() else 0
    return GraphData(
        name=name or infer_name_from_path(path),
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        is_corrupted=is_corrupted,
        error_code=error_code,
        num_classes=num_classes,
        npz_path=str(path),
    )


def get_forget_idx(
    data: GraphData,
    forget_name: str,
    train_only: bool = True,
    file_forget_idx: Optional[str | Path] = None,
    npz_key: str = "forget_idx",
) -> torch.Tensor:
    """Return node indices for a GDPRBench forget target.

    Supported forget_name values:
      - all_corrupted
      - wrong_fact, misattributed, outdated, invalid_combination,
        incorrect_derived_label/noisy_label
      - file_forget_idx: read a vector from a .npy/.npz file
      - npz:<key>: read a key from the currently loaded .npz file
      - none/empty: no deletion
    """
    name = (forget_name or "none").strip().lower()
    if name in {"none", "clean", "original", ""}:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.y.device)
    elif name == "all_corrupted":
        mask = data.error_code > 0
    elif name == "file_forget_idx":
        if file_forget_idx is None:
            raise ValueError("file_forget_idx path is required when forget_name=file_forget_idx")
        arr_data = np.load(file_forget_idx, allow_pickle=True)
        if isinstance(arr_data, np.lib.npyio.NpzFile):
            if npz_key not in arr_data.files:
                raise ValueError(f"Key {npz_key} not found in {file_forget_idx}; available={arr_data.files}")
            arr = arr_data[npz_key]
        else:
            arr = arr_data
        idx = torch.as_tensor(arr, dtype=torch.long, device=data.y.device)
        if train_only:
            idx = idx[data.train_mask[idx]]
        return torch.unique(idx)
    elif name.startswith("npz:"):
        key = name.split(":", 1)[1]
        arr_data = np.load(data.npz_path, allow_pickle=True)
        if key not in arr_data.files:
            raise ValueError(f"Key {key} not found in {data.npz_path}; available={arr_data.files}")
        idx = torch.as_tensor(arr_data[key], dtype=torch.long, device=data.y.device)
        if train_only:
            idx = idx[data.train_mask[idx]]
        return torch.unique(idx)
    else:
        if name not in ERROR_NAME_TO_CODE:
            raise ValueError(f"Unknown forget_name '{forget_name}'. Known names: {sorted(ERROR_NAME_TO_CODE)}")
        mask = data.error_code == ERROR_NAME_TO_CODE[name]

    if train_only:
        mask = mask & data.train_mask
    return torch.where(mask)[0]


def mask_from_idx(num_nodes: int, idx: torch.Tensor, device: Optional[torch.device | str] = None) -> torch.Tensor:
    dev = torch.device(device) if device is not None else idx.device
    m = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    if idx.numel() > 0:
        m[idx.to(dev)] = True
    return m


def remove_nodes_from_edges(edge_index: torch.Tensor, forget_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if forget_idx.numel() == 0:
        return edge_index
    forget_mask = mask_from_idx(num_nodes, forget_idx, edge_index.device)
    keep = (~forget_mask[edge_index[0]]) & (~forget_mask[edge_index[1]])
    return edge_index[:, keep]


def remove_edges_by_idx(edge_index: torch.Tensor, forget_edge_idx: torch.Tensor) -> torch.Tensor:
    if forget_edge_idx.numel() == 0:
        return edge_index
    keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
    keep[forget_edge_idx.to(edge_index.device)] = False
    return edge_index[:, keep]


def retained_train_mask(data: GraphData, forget_idx: torch.Tensor) -> torch.Tensor:
    if forget_idx.numel() == 0:
        return data.train_mask.clone()
    forget_mask = mask_from_idx(data.num_nodes, forget_idx, data.train_mask.device)
    return data.train_mask & (~forget_mask)


def retained_eval_mask(mask: torch.Tensor, forget_idx: torch.Tensor) -> torch.Tensor:
    if forget_idx.numel() == 0:
        return mask.clone()
    forget_mask = mask_from_idx(mask.shape[0], forget_idx, mask.device)
    return mask & (~forget_mask)


def sample_random_edges(edge_index: torch.Tensor, num_edges: int, seed: int = 0, undirected_pairs: bool = False) -> torch.Tensor:
    """Sample edge positions for optional edge-unlearning appendix experiments.
    """
    g = torch.Generator(device=edge_index.device)
    g.manual_seed(seed)
    e = edge_index.shape[1]
    num_edges = min(int(num_edges), e)
    return torch.randperm(e, generator=g, device=edge_index.device)[:num_edges]


def graph_summary(data: GraphData) -> Dict[str, int | float | str]:
    return {
        "name": data.name,
        "num_nodes": data.num_nodes,
        "num_edges_directed": data.num_edges_directed,
        "feature_dim": data.feature_dim,
        "num_classes": data.num_classes,
        "train_nodes": int(data.train_mask.sum().item()),
        "val_nodes": int(data.val_mask.sum().item()),
        "test_nodes": int(data.test_mask.sum().item()),
        "corrupted_nodes": int((data.error_code > 0).sum().item()),
    }
