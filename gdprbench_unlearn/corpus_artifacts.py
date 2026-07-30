from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception:  # pragma: no cover
    OneHotEncoder = None
    StandardScaler = None


ERROR_CODE_TO_NAME = {
    0: "clean",
    1: "wrong_fact",
    2: "misattributed",
    3: "outdated",
    4: "invalid_combination",
    5: "incorrect_derived_label",
}
ERROR_NAME_TO_CODE = {v: k for k, v in ERROR_CODE_TO_NAME.items()}
ERROR_NAME_TO_LEGACY = {
    "wrong_fact": "WRONG_FACT",
    "misattributed": "MISATTRIBUTED",
    "outdated": "OUTDATED",
    "invalid_combination": "INVALID_COMBO",
    "incorrect_derived_label": "NOISY_LABEL",  # legacy name in earlier draft/code
}
ERROR_ORDER = ["wrong_fact", "misattributed", "outdated", "invalid_combination", "incorrect_derived_label"]

MIXTURE_PROFILES: Dict[str, Dict[str, float]] = {
    "default": {
        "wrong_fact": 0.25,
        "misattributed": 0.25,
        "outdated": 0.20,
        "invalid_combination": 0.20,
        "incorrect_derived_label": 0.10,
    },
    "balanced": {
        "wrong_fact": 0.25,
        "misattributed": 0.25,
        "outdated": 0.20,
        "invalid_combination": 0.20,
        "incorrect_derived_label": 0.10,
    },
    "noise_heavy": {
        "wrong_fact": 0.15,
        "misattributed": 0.15,
        "outdated": 0.15,
        "invalid_combination": 0.15,
        "incorrect_derived_label": 0.40,
    },
    "outdated_heavy": {
        "wrong_fact": 0.10,
        "misattributed": 0.15,
        "outdated": 0.35,
        "invalid_combination": 0.30,
        "incorrect_derived_label": 0.10,
    },
    "wrongfact_heavy": {
        "wrong_fact": 0.45,
        "misattributed": 0.20,
        "outdated": 0.15,
        "invalid_combination": 0.10,
        "incorrect_derived_label": 0.10,
    },
}

COUNTRIES = {
    "DE": ["Berlin", "Munich", "Hamburg"],
    "FR": ["Paris", "Lyon", "Marseille"],
    "IT": ["Rome", "Milan", "Naples"],
    "ES": ["Madrid", "Barcelona", "Valencia"],
    "UK": ["London", "Manchester", "Birmingham"],
    "US": ["New York", "San Francisco", "Austin"],
    "PK": ["Lahore", "Karachi", "Islamabad"],
    "IN": ["Delhi", "Bengaluru", "Mumbai"],
}
COUNTRY_LIST = list(COUNTRIES.keys())
ALL_CITIES = sorted({c for cities in COUNTRIES.values() for c in cities})
EDUCATION_LEVELS = ["none", "highschool", "bachelor", "master", "phd"]
JOBS = ["student", "unemployed", "service", "professional", "manager", "retired"]


def make_one_hot_encoder():
    if OneHotEncoder is None:
        raise RuntimeError("scikit-learn is required to encode the GDPRBench profile table")
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def hamilton_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    s = float(sum(weights.values()))
    if s <= 0:
        raise ValueError("Mixture weights must sum to a positive value")
    norm = {k: float(v) / s for k, v in weights.items()}
    raw = {k: norm[k] * total for k in ERROR_ORDER}
    counts = {k: int(math.floor(raw[k])) for k in ERROR_ORDER}
    remaining = total - sum(counts.values())
    if remaining > 0:
        order = sorted(ERROR_ORDER, key=lambda k: raw[k] - counts[k], reverse=True)
        for k in order[:remaining]:
            counts[k] += 1
    # Pairwise misattribution requires an even count. Adjust by at most one/two
    # while preserving total and deterministic allocation.
    if counts["misattributed"] % 2 == 1:
        donor_order = [k for k in ERROR_ORDER if k != "misattributed" and counts[k] > 0]
        # Prefer moving one unit to the largest non-misattributed bucket.
        if donor_order:
            kmax = max(donor_order, key=lambda k: counts[k])
            counts["misattributed"] -= 1
            counts[kmax] += 1
        else:
            counts["misattributed"] -= 1
    assert sum(counts.values()) == total, (counts, total)
    return counts


def risk_label_rule(age: np.ndarray, education: np.ndarray, job: np.ndarray, income: np.ndarray, last_update_year: np.ndarray) -> np.ndarray:
    edu_score = np.isin(education, ["master", "phd"]).astype(float)
    risky_job = np.isin(job, ["unemployed", "retired"]).astype(float)
    recency = (last_update_year >= 2022).astype(float)
    score = (
        0.030 * (age.astype(float) - 18.0)
        + 0.000015 * income.astype(float)
        + 0.35 * edu_score
        + 0.20 * risky_job
        - 0.25 * recency
    )
    thr = np.median(score)
    return (score > thr).astype(np.int64)


def generate_clean_records(num_nodes: int, seed: int, topology_name: str = "") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 85, size=num_nodes)
    country = rng.choice(COUNTRY_LIST, size=num_nodes)
    city = np.array([rng.choice(COUNTRIES[c]) for c in country], dtype=object)
    education = rng.choice(EDUCATION_LEVELS, size=num_nodes, p=[0.08, 0.30, 0.34, 0.22, 0.06])
    # Mild schema coherence in the clean table.
    job = np.empty(num_nodes, dtype=object)
    for i, a in enumerate(age):
        if a <= 24:
            probs = np.array([0.55, 0.10, 0.18, 0.12, 0.03, 0.02])
        elif a >= 65:
            probs = np.array([0.00, 0.05, 0.08, 0.12, 0.05, 0.70])
        else:
            probs = np.array([0.02, 0.08, 0.22, 0.38, 0.25, 0.05])
        job[i] = rng.choice(JOBS, p=probs / probs.sum())
    base_income = {
        "student": 12000,
        "unemployed": 10000,
        "service": 32000,
        "professional": 62000,
        "manager": 85000,
        "retired": 28000,
    }
    income = np.array([base_income[j] for j in job], dtype=float) * rng.lognormal(mean=0, sigma=0.25, size=num_nodes)
    last_update_year = rng.integers(2018, 2026, size=num_nodes)
    y = risk_label_rule(age, education, job, income, last_update_year)
    df = pd.DataFrame({
        "node_id": np.arange(num_nodes, dtype=np.int64),
        "age": age.astype(np.int64),
        "country": country,
        "city": city,
        "education": education,
        "job": job,
        "income": np.round(income, 2),
        "last_update_year": last_update_year.astype(np.int64),
        "risk_label": y.astype(np.int64),
        "y": y.astype(np.int64),
    })
    return df


def allocate_error_codes(num_nodes: int, corruption_rate: float, seed: int, mixture_profile: str = "default") -> np.ndarray:
    if mixture_profile not in MIXTURE_PROFILES:
        raise ValueError(f"Unknown mixture_profile={mixture_profile}; known={sorted(MIXTURE_PROFILES)}")
    rng = np.random.default_rng(seed)
    n_corr = int(round(corruption_rate * num_nodes))
    n_corr = min(max(n_corr, 0), num_nodes)
    selected = rng.choice(num_nodes, size=n_corr, replace=False)
    rng.shuffle(selected)
    counts = hamilton_counts(n_corr, MIXTURE_PROFILES[mixture_profile])
    code = np.zeros(num_nodes, dtype=np.int64)
    offset = 0
    for name in ERROR_ORDER:
        c = counts[name]
        code[selected[offset:offset + c]] = ERROR_NAME_TO_CODE[name]
        offset += c
    return code


def _different_country_city(rng: np.random.Generator, old_country: str) -> Tuple[str, str]:
    choices = [c for c in COUNTRY_LIST if c != old_country]
    new_country = str(rng.choice(choices))
    new_city = str(rng.choice(COUNTRIES[new_country]))
    return new_country, new_city


def apply_errors(clean: pd.DataFrame, error_code: np.ndarray, seed: int, outdated_cutoff_year: int = 2018) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    obs = clean.copy(deep=True)
    rows: List[Dict[str, object]] = []

    # Misattribution pairs are handled together.
    mis_idx = np.where(error_code == ERROR_NAME_TO_CODE["misattributed"])[0]
    rng.shuffle(mis_idx)
    if len(mis_idx) % 2 == 1:
        mis_idx = mis_idx[:-1]
    mis_pairs = set()
    for a, b in mis_idx.reshape(-1, 2):
        for col in ["education", "job", "income"]:
            obs.at[a, col], obs.at[b, col] = obs.at[b, col], obs.at[a, col]
        mis_pairs.add(int(a))
        mis_pairs.add(int(b))

    for node in range(len(clean)):
        code = int(error_code[node])
        if code == 0:
            rows.append({
                "node_id": node,
                "is_corrupted": False,
                "error_type": "",
                "error_type_legacy": "",
                "error_fields": "",
            })
            continue
        name = ERROR_CODE_TO_NAME[code]
        fields: List[str] = []
        if name == "wrong_fact":
            new_country, new_city = _different_country_city(rng, str(clean.at[node, "country"]))
            obs.at[node, "country"] = new_country
            obs.at[node, "city"] = new_city
            shift = int(rng.choice([-10, -5, 5, 10]))
            obs.at[node, "age"] = int(np.clip(int(clean.at[node, "age"]) + shift, 18, 85))
            fields = ["country", "city", "age"]
        elif name == "misattributed":
            fields = ["education", "job", "income"] if node in mis_pairs else []
        elif name == "outdated":
            obs.at[node, "last_update_year"] = int(rng.integers(2010, outdated_cutoff_year + 1))
            if rng.random() < 0.5:
                obs.at[node, "job"] = str(rng.choice([j for j in JOBS if j != clean.at[node, "job"]]))
                fields = ["last_update_year", "job"]
            else:
                new_country, new_city = _different_country_city(rng, str(clean.at[node, "country"]))
                obs.at[node, "country"] = new_country
                obs.at[node, "city"] = new_city
                fields = ["last_update_year", "country", "city"]
        elif name == "invalid_combination":
            pattern = int(rng.integers(0, 3))
            if pattern == 0:
                obs.at[node, "age"] = int(rng.integers(18, 23))
                obs.at[node, "education"] = "phd"
                fields = ["age", "education"]
            elif pattern == 1:
                obs.at[node, "age"] = int(rng.integers(18, 35))
                obs.at[node, "job"] = "retired"
                fields = ["age", "job"]
            else:
                obs.at[node, "job"] = "unemployed"
                obs.at[node, "income"] = float(rng.uniform(150000, 250000))
                fields = ["job", "income"]
        elif name == "incorrect_derived_label":
            obs.at[node, "risk_label"] = int(1 - int(clean.at[node, "risk_label"]))
            obs.at[node, "y"] = int(obs.at[node, "risk_label"])
            fields = ["risk_label", "y"]
        rows.append({
            "node_id": node,
            "is_corrupted": True,
            "error_type": name,
            "error_type_legacy": ERROR_NAME_TO_LEGACY[name],
            "error_fields": ";".join(fields),
        })
    # Keep y in sync except for intentionally corrupted derived labels.
    non_label = error_code != ERROR_NAME_TO_CODE["incorrect_derived_label"]
    obs.loc[non_label, "y"] = obs.loc[non_label, "risk_label"].astype(int)
    err = pd.DataFrame(rows)
    return obs, err


def compute_detection_scores(obs: pd.DataFrame, clean: pd.DataFrame, errors: pd.DataFrame, seed: int, outdated_cutoff_year: int = 2018) -> pd.DataFrame:
    # Simple, deterministic annotation artifacts. They are not treated as a core
    # detector in the main evaluation; they are released for future detector-selected protocols.
    n = len(obs)
    rule = np.zeros(n, dtype=float)
    invalid = (
        ((obs["age"] < 23) & (obs["education"] == "phd")) |
        ((obs["age"] < 35) & (obs["job"] == "retired")) |
        ((obs["job"] == "unemployed") & (obs["income"] > 120000))
    )
    outdated = obs["last_update_year"].astype(int).values <= outdated_cutoff_year
    city_valid = np.array([str(city) in COUNTRIES.get(str(country), []) for country, city in zip(obs["country"], obs["city"])])
    rule[invalid.values | outdated | (~city_valid)] = 1.0

    # Label inconsistency: compare stored y to rule-derived label from observed fields.
    derived = risk_label_rule(
        obs["age"].astype(int).values,
        obs["education"].astype(str).values,
        obs["job"].astype(str).values,
        obs["income"].astype(float).values,
        obs["last_update_year"].astype(int).values,
    )
    label_inconsistency = (derived != obs["y"].astype(int).values).astype(float)

    # Lightweight anomaly score over numeric attributes. Robust z-score style, clipped to [0,1].
    num = obs[["age", "income", "last_update_year"]].astype(float).values
    med = np.median(num, axis=0)
    mad = np.median(np.abs(num - med), axis=0) + 1e-9
    z = np.abs((num - med) / (1.4826 * mad))
    anomaly = np.clip(z.max(axis=1) / 5.0, 0.0, 1.0)

    combined = 0.4 * rule + 0.3 * anomaly + 0.3 * label_inconsistency
    return pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "rule_violation_score": rule,
        "anomaly_score": anomaly,
        "label_inconsistency_score": label_inconsistency,
        "combined_score": combined,
        "is_corrupted_ground_truth": errors["is_corrupted"].astype(bool).values,
        "error_type_ground_truth": errors["error_type"].values,
    })


def encode_features(obs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    num_cols = ["age", "income", "last_update_year"]
    cat_cols = ["country", "city", "education", "job"]
    X_num = obs[num_cols].astype(float).values
    enc = make_one_hot_encoder()
    X_cat = enc.fit_transform(obs[cat_cols].astype(str))
    X = np.concatenate([X_num, X_cat], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = obs["y"].astype(int).values.astype(np.int64)
    return X, y


def random_split(num_nodes: int, seed: int, train_ratio: float = 0.70, val_ratio: float = 0.10, test_ratio: float = 0.20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_nodes)
    n_train = int(round(train_ratio * num_nodes))
    n_val = int(round(val_ratio * num_nodes))
    train = np.zeros(num_nodes, dtype=bool)
    val = np.zeros(num_nodes, dtype=bool)
    test = np.zeros(num_nodes, dtype=bool)
    train[idx[:n_train]] = True
    val[idx[n_train:n_train + n_val]] = True
    test[idx[n_train + n_val:]] = True
    return train, val, test


def normalize_edge_index(edge_index: np.ndarray, num_nodes: int, symmetrize: bool = True, remove_self_loops: bool = True) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.shape[0] != 2:
        if edge_index.shape[1] == 2:
            edge_index = edge_index.T
        else:
            raise ValueError(f"edge_index must have shape [2,E], got {edge_index.shape}")
    src, dst = edge_index[0], edge_index[1]
    ok = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    if remove_self_loops:
        ok &= src != dst
    edge_index = np.vstack([src[ok], dst[ok]]).astype(np.int64)
    if symmetrize:
        rev = edge_index[[1, 0], :]
        edge_index = np.concatenate([edge_index, rev], axis=1)
    # Deduplicate directed pairs.
    pairs = np.unique(edge_index.T, axis=0)
    return pairs.T.astype(np.int64)


def unique_undirected_pairs(edge_index: np.ndarray) -> np.ndarray:
    src = edge_index[0]
    dst = edge_index[1]
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pairs.astype(np.int64)


def save_edges_csv(edge_index: np.ndarray, path: Path) -> int:
    pairs = unique_undirected_pairs(edge_index)
    pd.DataFrame({"src": pairs[:, 0], "dst": pairs[:, 1]}).to_csv(path, index=False)
    return int(pairs.shape[0])


def save_corpus_from_edge_index(
    edge_index: np.ndarray,
    num_nodes: int,
    name: str,
    out_dir: str | Path,
    seed: int = 1,
    corruption_rate: float = 0.25,
    mixture_profile: str = "default",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    symmetrize_edges: bool = True,
    write_csv: bool = True,
    write_edges_csv: bool = True,
    outdated_cutoff_year: int = 2018,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    corpus_dir = out_dir / "corpora" / name
    data_dir = out_dir / "data"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    edge_index = normalize_edge_index(edge_index, num_nodes=num_nodes, symmetrize=symmetrize_edges)
    clean = generate_clean_records(num_nodes, seed=seed, topology_name=name)
    codes = allocate_error_codes(num_nodes, corruption_rate, seed=seed + 11, mixture_profile=mixture_profile)
    observed, errors = apply_errors(clean, codes, seed=seed + 17, outdated_cutoff_year=outdated_cutoff_year)
    detection = compute_detection_scores(observed, clean, errors, seed=seed + 23, outdated_cutoff_year=outdated_cutoff_year)
    X, y = encode_features(observed)
    train_mask, val_mask, test_mask = random_split(num_nodes, seed=seed + 29, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    is_corrupted = codes > 0
    forget_idx_all = np.where(is_corrupted)[0].astype(np.int64)
    forget_idx = np.where(is_corrupted & train_mask)[0].astype(np.int64)

    npz_path = data_dir / f"{name}.npz"
    np.savez_compressed(
        npz_path,
        x=X.astype(np.float32),
        y=y.astype(np.int64),
        edge_index=edge_index.astype(np.int64),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        forget_idx=forget_idx,
        forget_idx_all=forget_idx_all,
        is_corrupted=is_corrupted,
        error_code=codes.astype(np.int64),
    )

    unique_edges_count = None
    if write_csv:
        clean.to_csv(corpus_dir / "nodes_clean.csv", index=False)
        observed.to_csv(corpus_dir / "nodes_observed.csv", index=False)
        errors.to_csv(corpus_dir / "errors_metadata.csv", index=False)
        detection.to_csv(corpus_dir / "detection_scores.csv", index=False)
        if write_edges_csv:
            unique_edges_count = save_edges_csv(edge_index, corpus_dir / "edges.csv")

    counts = {ERROR_CODE_TO_NAME[i]: int((codes == i).sum()) for i in range(0, 6)}
    stats = {
        "name": name,
        "num_nodes": int(num_nodes),
        "num_edges_directed": int(edge_index.shape[1]),
        "num_edges_unique_undirected": int(unique_edges_count) if unique_edges_count is not None else None,
        "feature_dim": int(X.shape[1]),
        "num_classes": int(y.max() + 1),
        "corruption_rate": float(corruption_rate),
        "mixture_profile": mixture_profile,
        "error_counts": counts,
        "split_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "forget_idx_all_size": int(forget_idx_all.size),
        "forget_idx_train_only_size": int(forget_idx.size),
        "npz_path": str(npz_path),
        "corpus_dir": str(corpus_dir),
        "seed": int(seed),
    }
    cfg = {
        "name": name,
        "schema_version": "gdprbench-fresh-2.0",
        "seed": int(seed),
        "corruption_rate": float(corruption_rate),
        "mixture_profile": mixture_profile,
        "mixture_weights": MIXTURE_PROFILES[mixture_profile],
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "symmetrize_edges": bool(symmetrize_edges),
        "outdated_cutoff_year": int(outdated_cutoff_year),
        "error_code_to_name": ERROR_CODE_TO_NAME,
    }
    if write_csv:
        (corpus_dir / "corpus_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (corpus_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        pd.DataFrame([stats | {f"count_{k}": v for k, v in counts.items()}]).to_csv(corpus_dir / "corpus_stats.csv", index=False)
    return stats


def synthetic_edge_index(num_nodes: int, avg_degree: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Sample roughly undirected Erdos-Renyi edges without materializing N^2.
    m = int(round(num_nodes * avg_degree / 2.0))
    src = rng.integers(0, num_nodes, size=m * 2)
    dst = rng.integers(0, num_nodes, size=m * 2)
    keep = src != dst
    pairs = np.stack([np.minimum(src[keep], dst[keep]), np.maximum(src[keep], dst[keep])], axis=1)
    pairs = np.unique(pairs, axis=0)
    if pairs.shape[0] > m:
        pairs = pairs[:m]
    # Return directed both-way; normalize_edge_index will deduplicate.
    e = np.concatenate([pairs, pairs[:, [1, 0]]], axis=0)
    return e.T.astype(np.int64)
