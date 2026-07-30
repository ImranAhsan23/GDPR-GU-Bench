from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("GDPRBENCH_DETERMINISTIC", "0") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return to_jsonable(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    return obj


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer() -> Iterable[Dict[str, float]]:
    start = time.perf_counter()
    out: Dict[str, float] = {}
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start


def cuda_peak_memory_mb(device: torch.device | str) -> float:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2))
    return 0.0


def reset_cuda_peak(device: torch.device | str) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def atomic_touch(path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8"):
        os.utime(path, None)


def readable_num(n: int | float) -> str:
    n = float(n)
    for suffix in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{suffix}" if suffix else f"{int(n)}"
        n /= 1000
    return f"{n:.1f}T"
