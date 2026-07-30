from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel

from .data import ERROR_CODE_TO_NAME, ERROR_NAME_TO_CODE, GraphData, get_forget_idx, mask_from_idx
from .metrics import khop_mask


def degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(edge_index[1].cpu(), minlength=num_nodes).long()


def same_label_edge_fraction(data: GraphData, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return float("nan")
    node_mask = mask_from_idx(data.num_nodes, idx.cpu(), device="cpu")
    edge = data.edge_index.cpu()
    src, dst = edge[0], edge[1]
    incident = node_mask[src] | node_mask[dst]
    if int(incident.sum().item()) == 0:
        return float("nan")
    y = data.y.cpu()
    return float((y[src[incident]] == y[dst[incident]]).float().mean().item())


def feature_ks_mmd(data: GraphData, target_idx: torch.Tensor, ref_idx: torch.Tensor, sample: int = 5000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x = data.x.cpu().numpy()
    ti = target_idx.cpu().numpy()
    ri = ref_idx.cpu().numpy()
    if len(ti) == 0 or len(ri) == 0:
        return {"ks_mean_over_features": float("nan"), "ks_max_over_features": float("nan"), "mmd_rbf_sample": float("nan")}
    if len(ti) > sample:
        ti_s = rng.choice(ti, size=sample, replace=False)
    else:
        ti_s = ti
    if len(ri) > sample:
        ri_s = rng.choice(ri, size=sample, replace=False)
    else:
        ri_s = ri
    ks_vals = []
    for j in range(x.shape[1]):
        try:
            ks_vals.append(float(ks_2samp(x[ti_s, j], x[ri_s, j]).statistic))
        except Exception:
            pass
    # RBF MMD on a smaller sample to avoid O(n^2) explosion.
    m = min(1000, len(ti_s), len(ri_s))
    if m < 2:
        mmd = float("nan")
    else:
        ti_m = rng.choice(ti_s, size=m, replace=False)
        ri_m = rng.choice(ri_s, size=m, replace=False)
        X = x[ti_m]
        Y = x[ri_m]
        # Median heuristic with fallback.
        Z = np.vstack([X[: min(len(X), 300)], Y[: min(len(Y), 300)]])
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-9)
        kxx = rbf_kernel(X, X, gamma=gamma).mean()
        kyy = rbf_kernel(Y, Y, gamma=gamma).mean()
        kxy = rbf_kernel(X, Y, gamma=gamma).mean()
        mmd = float(kxx + kyy - 2 * kxy)
    return {
        "ks_mean_over_features": float(np.mean(ks_vals)) if ks_vals else float("nan"),
        "ks_max_over_features": float(np.max(ks_vals)) if ks_vals else float("nan"),
        "mmd_rbf_sample": mmd,
    }


def validity_summary_for_group(data: GraphData, group_name: str, idx: torch.Tensor, seed: int = 0) -> Dict[str, float | int | str]:
    deg = degree(data.edge_index, data.num_nodes)
    y = data.y.cpu()
    idx = idx.cpu()
    ref_idx = torch.where(data.train_mask.cpu() & (~mask_from_idx(data.num_nodes, idx, device="cpu")))[0]
    one_hop = khop_mask(data.edge_index.cpu(), idx, data.num_nodes, k=1, exclude_start=True)
    two_hop = khop_mask(data.edge_index.cpu(), idx, data.num_nodes, k=2, exclude_start=True)
    row: Dict[str, float | int | str] = {"group": group_name, "n": int(idx.numel())}
    if idx.numel() == 0:
        for k in ["label_1_rate", "degree_mean", "degree_median", "degree_p90", "one_hop_neighbors", "two_hop_neighbors", "two_hop_coverage_pct", "same_label_edge_fraction"]:
            row[k] = float("nan")
        return row
    labels = y[idx]
    row.update({
        "label_0_count": int((labels == 0).sum().item()),
        "label_1_count": int((labels == 1).sum().item()) if int(y.max().item()) >= 1 else 0,
        "label_1_rate": float((labels == 1).float().mean().item()) if int(y.max().item()) >= 1 else float("nan"),
        "degree_mean": float(deg[idx].float().mean().item()),
        "degree_median": float(deg[idx].float().median().item()),
        "degree_p90": float(torch.quantile(deg[idx].float(), 0.9).item()),
        "one_hop_neighbors": int(one_hop.sum().item()),
        "two_hop_neighbors": int(two_hop.sum().item()),
        "two_hop_coverage_pct": float(two_hop.float().mean().item() * 100),
        "same_label_edge_fraction": same_label_edge_fraction(data, idx),
    })
    row.update(feature_ks_mmd(data, idx, ref_idx, seed=seed))
    return row


def validate_benchmark(data: GraphData, groups: Sequence[str], seed: int = 0) -> pd.DataFrame:
    rows = []
    for g in groups:
        idx = get_forget_idx(data, g, train_only=True)
        rows.append(validity_summary_for_group(data, g, idx, seed=seed))
    return pd.DataFrame(rows)


def quantile_bins(values: np.ndarray, q: Sequence[float]) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    cuts = np.unique(np.quantile(values, q))
    return np.digitize(values, cuts, right=False)


def profile_bins(x: np.ndarray) -> np.ndarray:
    # Lightweight profile signature based on row mean/std/sum and first two raw dimensions.
    cols = [x.mean(axis=1), x.std(axis=1), x.sum(axis=1)]
    if x.shape[1] >= 1:
        cols.append(x[:, 0])
    if x.shape[1] >= 2:
        cols.append(x[:, 1])
    bins = [quantile_bins(c, [0.2, 0.4, 0.6, 0.8]) for c in cols]
    out = np.zeros(x.shape[0], dtype=np.int64)
    base = 1
    for b in bins:
        out += b * base
        base *= 10
    return out


def make_group_keys(data: GraphData, mode: str) -> np.ndarray:
    mode = mode.lower()
    n = data.num_nodes
    y = data.y.cpu().numpy()
    deg_np = degree(data.edge_index, n).numpy()
    deg_bin = quantile_bins(deg_np, [0.2, 0.4, 0.6, 0.8])
    prof_bin = profile_bins(data.x.cpu().numpy())
    keys = np.zeros(n, dtype=object)
    for i in range(n):
        parts = []
        if "label" in mode:
            parts.append(f"y{int(y[i])}")
        if "degree" in mode:
            parts.append(f"d{int(deg_bin[i])}")
        if "profile" in mode or "community" in mode:
            parts.append(f"p{int(prof_bin[i])}")
        keys[i] = "|".join(parts) if parts else "all"
    return keys


def create_matched_control(
    data: GraphData,
    target_idx: torch.Tensor,
    mode: str,
    seed: int = 0,
    clean_only: bool = True,
) -> torch.Tensor:
    """Create size-matched or distribution-matched non-target control nodes.

    Modes: uniform, label, degree, label_degree, label_degree_profile.
    The default candidate pool is clean training nodes, which makes the control
    a genuine non-corrupted contrast.  If too few nodes are available inside a
    stratum, the function falls back to non-target training nodes.
    """
    rng = np.random.default_rng(seed)
    target = target_idx.cpu().numpy().astype(int)
    n = data.num_nodes
    target_mask = np.zeros(n, dtype=bool)
    target_mask[target] = True
    train = data.train_mask.cpu().numpy().astype(bool)
    clean = (data.error_code.cpu().numpy() == 0)
    if clean_only:
        cand_mask = train & clean & (~target_mask)
    else:
        cand_mask = train & (~target_mask)
    if cand_mask.sum() < len(target):
        cand_mask = train & (~target_mask)
    candidates_all = np.where(cand_mask)[0]
    if len(target) == 0:
        return torch.empty(0, dtype=torch.long)
    if len(candidates_all) < len(target):
        raise ValueError(f"Not enough candidates for control: need {len(target)}, have {len(candidates_all)}")
    mode = mode.lower()
    if mode == "uniform":
        return torch.tensor(rng.choice(candidates_all, size=len(target), replace=False), dtype=torch.long)

    keys = make_group_keys(data, mode)
    chosen: List[int] = []
    used = set()
    for key in np.unique(keys[target]):
        t_count = int((keys[target] == key).sum())
        pool = [int(i) for i in candidates_all[keys[candidates_all] == key] if int(i) not in used]
        if len(pool) >= t_count:
            sel = rng.choice(pool, size=t_count, replace=False).tolist()
        else:
            sel = pool
            remaining = t_count - len(sel)
            fallback = [int(i) for i in candidates_all if int(i) not in used and int(i) not in sel]
            sel += rng.choice(fallback, size=remaining, replace=False).tolist()
        chosen.extend(sel)
        used.update(sel)
    if len(chosen) < len(target):
        fallback = [int(i) for i in candidates_all if int(i) not in used]
        chosen.extend(rng.choice(fallback, size=len(target) - len(chosen), replace=False).tolist())
    if len(chosen) > len(target):
        chosen = rng.choice(chosen, size=len(target), replace=False).tolist()
    return torch.tensor(sorted(chosen), dtype=torch.long)


def save_control_npz(original_npz_path: str | Path, out_path: str | Path, control_idx: torch.Tensor, extra_name: str = "forget_idx_control") -> None:
    d = np.load(original_npz_path, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    arrays[extra_name] = control_idx.cpu().numpy().astype(np.int64)
    # Also set forget_idx to the control for direct compatibility with scripts.
    arrays["forget_idx"] = arrays[extra_name]
    np.savez_compressed(out_path, **arrays)
