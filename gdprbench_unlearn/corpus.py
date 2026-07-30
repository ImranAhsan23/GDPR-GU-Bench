from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def generate_profile_features(num_nodes: int, seed: int = 0, feature_dim: int = 46) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an auditable semi-synthetic personal-profile feature surface.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 85, size=num_nodes)
    income = rng.lognormal(mean=10.5, sigma=0.55, size=num_nodes)
    last_update = rng.integers(2017, 2026, size=num_nodes)
    education = rng.integers(0, 5, size=num_nodes)
    job = rng.integers(0, 10, size=num_nodes)
    country = rng.integers(0, 8, size=num_nodes)
    city = rng.integers(0, 12, size=num_nodes)

    # Risk label: stable rule-based construction.
    score = (
        0.03 * (age - 18)
        + 0.000015 * income
        + 0.25 * (education >= 3)
        + 0.15 * np.isin(job, [1, 4, 7])
        - 0.18 * (last_update >= 2023)
        + rng.normal(0, 0.35, size=num_nodes)
    )
    y = (score > np.median(score)).astype(np.int64)

    numeric = np.stack([
        (age - 18) / (85 - 18),
        (np.log(income) - np.log(income).mean()) / (np.log(income).std() + 1e-9),
        (last_update - 2017) / (2025 - 2017),
    ], axis=1)
    onehots = []
    for arr, k in [(education, 5), (job, 10), (country, 8), (city, 12)]:
        oh = np.zeros((num_nodes, k), dtype=np.float32)
        oh[np.arange(num_nodes), arr] = 1.0
        onehots.append(oh)
    x = np.concatenate([numeric.astype(np.float32)] + onehots, axis=1)
    if x.shape[1] < feature_dim:
        extra = rng.normal(size=(num_nodes, feature_dim - x.shape[1])).astype(np.float32) * 0.05
        x = np.concatenate([x, extra], axis=1)
    elif x.shape[1] > feature_dim:
        x = x[:, :feature_dim]
    return x.astype(np.float32), y.astype(np.int64)


def allocate_errors(num_nodes: int, corruption_rate: float = 0.25, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_corr = int(round(num_nodes * corruption_rate))
    corrupted = np.zeros(num_nodes, dtype=bool)
    selected = rng.choice(num_nodes, size=n_corr, replace=False)
    corrupted[selected] = True
    # 25/25/20/20/10 Hamilton-style allocation.
    weights = np.array([0.25, 0.25, 0.20, 0.20, 0.10])
    raw = weights * n_corr
    counts = np.floor(raw).astype(int)
    rem = n_corr - counts.sum()
    if rem > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:rem]] += 1
    codes = np.zeros(num_nodes, dtype=np.int64)
    rng.shuffle(selected)
    start = 0
    for code, count in enumerate(counts, start=1):
        codes[selected[start:start + count]] = code
        start += count
    return corrupted, codes


def random_split(num_nodes: int, seed: int = 0, train: float = 0.7, val: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_nodes)
    n_train = int(round(train * num_nodes))
    n_val = int(round(val * num_nodes))
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


def save_gdpr_npz_from_edge_index(edge_index: np.ndarray, num_nodes: int, out_path: str | Path, seed: int = 0, corruption_rate: float = 0.25) -> None:
    x, y = generate_profile_features(num_nodes, seed=seed)
    is_corrupted, error_code = allocate_errors(num_nodes, corruption_rate=corruption_rate, seed=seed + 1)
    train_mask, val_mask, test_mask = random_split(num_nodes, seed=seed + 2)
    forget_idx_all = np.where(is_corrupted)[0].astype(np.int64)
    forget_idx = np.where(is_corrupted & train_mask)[0].astype(np.int64)
    np.savez_compressed(
        out_path,
        x=x,
        y=y,
        edge_index=edge_index.astype(np.int64),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        forget_idx=forget_idx,
        forget_idx_all=forget_idx_all,
        is_corrupted=is_corrupted,
        error_code=error_code,
    )
