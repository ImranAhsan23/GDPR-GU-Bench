from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import classification_metrics
from .utils import ensure_dir, reset_cuda_peak, cuda_peak_memory_mb


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 80
    eval_every: int = 10
    valid_metric: str = "macro_f1"  # macro_f1 or loss
    grad_clip: float = 0.0
    verbose: bool = True


@dataclass
class TrainResult:
    best_epoch: int
    best_val_metric: float
    train_seconds: float
    peak_cuda_mb: float
    history: List[Dict[str, float]]


def _copy_state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def compute_logits(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x, edge_index)


def train_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device | str,
    log_csv: Optional[str | Path] = None,
) -> Tuple[torch.nn.Module, TrainResult, torch.Tensor]:
    """Full-batch GNN training with early stopping and validation logging."""
    device = torch.device(device)
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device).bool()
    val_mask = val_mask.to(device).bool()
    if int(val_mask.sum().item()) == 0:
        val_mask = train_mask

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = _copy_state_to_cpu(model)
    best_epoch = 0
    best_metric = -float("inf") if cfg.valid_metric != "loss" else float("inf")
    patience_left = cfg.patience
    history: List[Dict[str, float]] = []
    log_path = Path(log_csv) if log_csv else None
    if log_path:
        ensure_dir(log_path.parent)

    reset_cuda_peak(device)
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        if int(train_mask.sum().item()) == 0:
            raise ValueError("Training mask is empty after deletion; cannot train.")
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if epoch % cfg.eval_every == 0 or epoch == 1 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x, edge_index)
                val_loss = float(F.cross_entropy(logits_eval[val_mask], y[val_mask]).item())
                val_metrics = classification_metrics(logits_eval, y, val_mask, prefix="val_")
                train_metrics = classification_metrics(logits_eval, y, train_mask, prefix="train_")
            row = {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "val_loss": val_loss,
                **val_metrics,
                **train_metrics,
            }
            history.append(row)
            if cfg.valid_metric == "loss":
                current = val_loss
                improved = current < best_metric - 1e-12
            else:
                current = val_metrics.get("val_macro_f1", float("nan"))
                if np.isnan(current):
                    current = -float("inf")
                improved = current > best_metric + 1e-12
            if improved:
                best_metric = float(current)
                best_epoch = epoch
                best_state = _copy_state_to_cpu(model)
                patience_left = cfg.patience
            else:
                patience_left -= cfg.eval_every
            if cfg.verbose:
                print(f"epoch={epoch:04d} loss={loss.item():.4f} val_loss={val_loss:.4f} val_macro_f1={val_metrics['val_macro_f1']:.4f}")
            if patience_left <= 0:
                if cfg.verbose:
                    print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    train_seconds = time.perf_counter() - start
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    with torch.no_grad():
        best_logits = compute_logits(model, x, edge_index).detach().cpu()
    peak = cuda_peak_memory_mb(device)

    if log_path and history:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    return model, TrainResult(best_epoch=best_epoch, best_val_metric=float(best_metric), train_seconds=float(train_seconds), peak_cuda_mb=peak, history=history), best_logits


def save_checkpoint(model: torch.nn.Module, path: str | Path, extra: Optional[Dict] = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {"model_state": _copy_state_to_cpu(model), "extra": extra or {}}
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device | str = "cpu") -> Dict:
    payload = torch.load(path, map_location=device)
    state = payload["model_state"] if "model_state" in payload else payload
    model.load_state_dict(state)
    return payload.get("extra", {}) if isinstance(payload, dict) else {}
