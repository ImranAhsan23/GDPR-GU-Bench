from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import GraphData, mask_from_idx, remove_nodes_from_edges, retained_train_mask
from .metrics import khop_mask
from .models import build_model
from .train import TrainConfig, train_model, save_checkpoint, load_checkpoint, compute_logits
from .utils import ensure_dir, reset_cuda_peak, cuda_peak_memory_mb


@dataclass
class ModelConfig:
    backbone: str = "gcn"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 2


@dataclass
class UnlearnConfig:
    method: str = "finetune_retain"
    unlearn_epochs: int = 60
    unlearn_lr: float = 0.005
    unlearn_weight_decay: float = 5e-4
    ga_epochs: int = 10
    ga_lr: float = 0.001
    gif_damping: float = 1e-3
    gif_scale: float = 1.0
    gif_retain_ft_epochs: int = 10
    agu_khop: int = 2
    agu_alpha: float = 2.0
    eraser_shards: int = 10
    eraser_epochs: int = 200
    eraser_partition: str = "degree_label"
    verbose: bool = True


SUPPORTED_METHODS = [
    "original",
    "retrain",
    "no_unlearn",
    "finetune_retain",
    "last_layer",
    "gradient_ascent",
    "gif_diag",
    "agu_lite",
    "graph_eraser",
    "utu_edge",
]


def new_model(data: GraphData, cfg: ModelConfig) -> torch.nn.Module:
    return build_model(
        cfg.backbone,
        in_dim=data.feature_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=data.num_classes,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        heads=cfg.heads,
    )


def clone_model_structure(data: GraphData, cfg: ModelConfig, source_ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = new_model(data, cfg).to(device)
    load_checkpoint(model, source_ckpt, device=device)
    return model


def train_original(
    data: GraphData,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    resume: bool = True,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    ckpt = out_dir / "original_model.pt"
    logits_file = out_dir / "original_logits.pt"
    metrics_file = out_dir / "original_meta.pt"
    model = new_model(data, model_cfg).to(device)
    if resume and ckpt.exists() and logits_file.exists() and metrics_file.exists():
        load_checkpoint(model, ckpt, device=device)
        logits = torch.load(logits_file, map_location="cpu")
        meta = torch.load(metrics_file, map_location="cpu")
        return model, logits, meta
    ensure_dir(out_dir)
    model, result, logits = train_model(
        model=model,
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        cfg=train_cfg,
        device=device,
        log_csv=out_dir / "original_train_log.csv",
    )
    meta = {"train_result": result.__dict__}
    save_checkpoint(model, ckpt, extra=meta)
    torch.save(logits, logits_file)
    torch.save(meta, metrics_file)
    return model, logits, meta


def retrain_reference(
    data: GraphData,
    forget_idx: torch.Tensor,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    resume: bool = True,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    ckpt = out_dir / "retrain_model.pt"
    logits_file = out_dir / "retrain_logits.pt"
    metrics_file = out_dir / "retrain_meta.pt"
    model = new_model(data, model_cfg).to(device)
    if resume and ckpt.exists() and logits_file.exists() and metrics_file.exists():
        load_checkpoint(model, ckpt, device=device)
        logits = torch.load(logits_file, map_location="cpu")
        meta = torch.load(metrics_file, map_location="cpu")
        return model, logits, meta
    r_edge = remove_nodes_from_edges(data.edge_index.to(device), forget_idx.to(device), data.num_nodes).detach().cpu()
    r_train_mask = retained_train_mask(data, forget_idx.cpu())
    model, result, logits = train_model(
        model=model,
        x=data.x,
        y=data.y,
        edge_index=r_edge,
        train_mask=r_train_mask,
        val_mask=data.val_mask,
        cfg=train_cfg,
        device=device,
        log_csv=out_dir / "retrain_train_log.csv",
    )
    meta = {"train_result": result.__dict__, "removed_directed_edges": int(data.num_edges_directed - r_edge.shape[1])}
    save_checkpoint(model, ckpt, extra=meta)
    torch.save(logits, logits_file)
    torch.save(meta, metrics_file)
    return model, logits, meta


def _fine_tune(
    model: torch.nn.Module,
    data: GraphData,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    device: torch.device,
    cfg: UnlearnConfig,
    only_last_layer: bool = False,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    model = model.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    if only_last_layer:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.last_layer_parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.unlearn_lr, weight_decay=cfg.unlearn_weight_decay)
    reset_cuda_peak(device)
    start = time.perf_counter()
    hist: List[Dict[str, float]] = []
    weights = sample_weight.to(device) if sample_weight is not None else None
    idx = torch.where(train_mask)[0]
    for epoch in range(1, cfg.unlearn_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        per = F.cross_entropy(logits[idx], y[idx], reduction="none")
        if weights is not None:
            w = weights[idx]
            loss = (per * w).sum() / w.sum().clamp(min=1e-12)
        else:
            loss = per.mean()
        loss.backward()
        opt.step()
        if cfg.verbose and (epoch == 1 or epoch == cfg.unlearn_epochs or epoch % 20 == 0):
            print(f"unlearn_ft epoch={epoch:04d} loss={loss.item():.4f}")
        hist.append({"epoch": float(epoch), "loss": float(loss.item())})
    seconds = time.perf_counter() - start
    logits = compute_logits(model, x, edge_index).detach().cpu()
    return model, logits, {"update_seconds": seconds, "peak_cuda_mb": cuda_peak_memory_mb(device), "history": hist}


def no_unlearn(
    original_model: torch.nn.Module,
    data: GraphData,
    forget_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    reset_cuda_peak(device)
    start = time.perf_counter()
    edge = remove_nodes_from_edges(data.edge_index.to(device), forget_idx.to(device), data.num_nodes)
    logits = compute_logits(original_model.to(device), data.x.to(device), edge).detach().cpu()
    return original_model, logits, {"update_seconds": time.perf_counter() - start, "peak_cuda_mb": cuda_peak_memory_mb(device)}


def fine_tune_retain(
    original_model: torch.nn.Module,
    data: GraphData,
    forget_idx: torch.Tensor,
    device: torch.device,
    cfg: UnlearnConfig,
    only_last_layer: bool = False,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    model = copy.deepcopy(original_model).to(device)
    edge = remove_nodes_from_edges(data.edge_index.to(device), forget_idx.to(device), data.num_nodes)
    train_mask = retained_train_mask(data, forget_idx.cpu())
    return _fine_tune(model, data, edge, train_mask, device, cfg, only_last_layer=only_last_layer)


def gradient_ascent_unlearn(
    original_model: torch.nn.Module,
    data: GraphData,
    forget_idx: torch.Tensor,
    device: torch.device,
    cfg: UnlearnConfig,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    model = copy.deepcopy(original_model).to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    fidx = forget_idx.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ga_lr, weight_decay=0.0)
    reset_cuda_peak(device)
    start = time.perf_counter()
    hist = []
    if fidx.numel() > 0:
        for epoch in range(1, cfg.ga_epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(x, data.edge_index.to(device))
            loss = -F.cross_entropy(logits[fidx], y[fidx])
            loss.backward()
            opt.step()
            hist.append({"ga_epoch": float(epoch), "negative_forget_loss": float(loss.item())})
            if cfg.verbose and (epoch == 1 or epoch == cfg.ga_epochs):
                print(f"gradient_ascent epoch={epoch:04d} neg_forget_loss={loss.item():.4f}")
    edge = remove_nodes_from_edges(data.edge_index.to(device), fidx, data.num_nodes)
    train_mask = retained_train_mask(data, forget_idx.cpu())
    ft_cfg = copy.deepcopy(cfg)
    ft_model, logits, meta = _fine_tune(model, data, edge, train_mask, device, ft_cfg, only_last_layer=False)
    meta["ga_history"] = hist
    meta["update_seconds"] = time.perf_counter() - start
    meta["peak_cuda_mb"] = max(meta.get("peak_cuda_mb", 0.0), cuda_peak_memory_mb(device))
    return ft_model, logits, meta


def gif_diag_unlearn(
    original_model: torch.nn.Module,
    data: GraphData,
    forget_idx: torch.Tensor,
    device: torch.device,
    cfg: UnlearnConfig,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    """Diagonal-Fisher influence approximation inspired by GIF.
    """
    model = copy.deepcopy(original_model).to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    edge = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    fidx = forget_idx.to(device)
    reset_cuda_peak(device)
    start = time.perf_counter()
    if fidx.numel() > 0:
        rmask = retained_train_mask(data, forget_idx.cpu()).to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        logits_r = model(x, edge)
        retain_loss = F.cross_entropy(logits_r[rmask], y[rmask])
        retain_grads = torch.autograd.grad(retain_loss, [p for p in model.parameters() if p.requires_grad], create_graph=False, retain_graph=False, allow_unused=True)
        fisher = [(g.detach() ** 2 if g is not None else None) for g in retain_grads]

        model.zero_grad(set_to_none=True)
        logits_f = model(x, edge)
        forget_loss = F.cross_entropy(logits_f[fidx], y[fidx])
        forget_grads = torch.autograd.grad(forget_loss, [p for p in model.parameters() if p.requires_grad], create_graph=False, retain_graph=False, allow_unused=True)
        with torch.no_grad():
            for p, fg, fd in zip([p for p in model.parameters() if p.requires_grad], forget_grads, fisher):
                if fg is None or fd is None:
                    continue
                step = cfg.gif_scale * fg / (fd + cfg.gif_damping)
                step = torch.clamp(step, min=-1.0, max=1.0)
                p.add_(step)
    if cfg.gif_retain_ft_epochs > 0:
        ft_cfg = copy.deepcopy(cfg)
        old_epochs = ft_cfg.unlearn_epochs
        ft_cfg.unlearn_epochs = cfg.gif_retain_ft_epochs
        edge_r = remove_nodes_from_edges(data.edge_index.to(device), fidx, data.num_nodes)
        rmask_cpu = retained_train_mask(data, forget_idx.cpu())
        model, logits, meta = _fine_tune(model, data, edge_r, rmask_cpu, device, ft_cfg, only_last_layer=False)
        ft_cfg.unlearn_epochs = old_epochs
    else:
        edge_r = remove_nodes_from_edges(data.edge_index.to(device), fidx, data.num_nodes)
        logits = compute_logits(model, x, edge_r).detach().cpu()
        meta = {}
    meta["update_seconds"] = time.perf_counter() - start
    meta["peak_cuda_mb"] = max(meta.get("peak_cuda_mb", 0.0), cuda_peak_memory_mb(device))
    meta["gif_variant"] = "diagonal_fisher"
    return model, logits, meta


def agu_lite_unlearn(
    original_model: torch.nn.Module,
    data: GraphData,
    forget_idx: torch.Tensor,
    device: torch.device,
    cfg: UnlearnConfig,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    """AGU-inspired affected-neighbor retained fine-tuning.

    """
    model = copy.deepcopy(original_model).to(device)
    fidx = forget_idx.to(device)
    edge_orig = data.edge_index.to(device)
    affected = khop_mask(edge_orig, fidx, data.num_nodes, k=cfg.agu_khop, exclude_start=True)
    train_mask_cpu = retained_train_mask(data, forget_idx.cpu())
    weights = torch.ones(data.num_nodes, dtype=torch.float32)
    weights[affected.detach().cpu()] = 1.0 + float(cfg.agu_alpha)
    edge_r = remove_nodes_from_edges(edge_orig, fidx, data.num_nodes)
    model, logits, meta = _fine_tune(model, data, edge_r, train_mask_cpu, device, cfg, only_last_layer=False, sample_weight=weights)
    meta["affected_neighbors"] = int(affected.sum().item())
    meta["agu_variant"] = "khop_weighted_retained_finetune"
    return model, logits, meta


def partition_nodes(data: GraphData, train_mask: torch.Tensor, k: int, seed: int, mode: str = "degree_label") -> List[torch.Tensor]:
    idx = torch.where(train_mask.cpu())[0].numpy()
    rng = np.random.default_rng(seed)
    if len(idx) == 0:
        return []
    if mode == "random":
        rng.shuffle(idx)
    else:
        # Sort by label and degree bin then round-robin to balance shards.
        edge = data.edge_index.cpu()
        deg = torch.bincount(edge[1], minlength=data.num_nodes).numpy()
        degree_bins = np.digitize(deg[idx], np.quantile(deg[idx], [0.2, 0.4, 0.6, 0.8]))
        labels = data.y.cpu().numpy()[idx]
        order = np.lexsort((rng.random(len(idx)), degree_bins, labels))
        idx = idx[order]
    shards = [[] for _ in range(k)]
    for j, node in enumerate(idx):
        shards[j % k].append(int(node))
    return [torch.tensor(s, dtype=torch.long) for s in shards if len(s) > 0]


def graph_eraser_unlearn(
    data: GraphData,
    forget_idx: torch.Tensor,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    cfg: UnlearnConfig,
    out_dir: Path,
    seed: int,
    resume: bool = True,
) -> Tuple[None, torch.Tensor, Dict]:
    """GraphEraser-style shard retraining and mean logit aggregation.
    """
    ensure_dir(out_dir)
    aggregate_file = out_dir / "graph_eraser_logits.pt"
    meta_file = out_dir / "graph_eraser_meta.pt"
    if resume and aggregate_file.exists() and meta_file.exists():
        return None, torch.load(aggregate_file, map_location="cpu"), torch.load(meta_file, map_location="cpu")

    rmask = retained_train_mask(data, forget_idx.cpu())
    edge_r = remove_nodes_from_edges(data.edge_index.to(device), forget_idx.to(device), data.num_nodes).detach().cpu()
    shards = partition_nodes(data, rmask, cfg.eraser_shards, seed=seed, mode=cfg.eraser_partition)
    logits_sum = None
    total_seconds = 0.0
    peak = 0.0
    shard_meta = []
    local_train_cfg = copy.deepcopy(train_cfg)
    local_train_cfg.epochs = cfg.eraser_epochs
    for i, shard_idx in enumerate(shards):
        shard_dir = out_dir / f"shard_{i:03d}"
        ckpt = shard_dir / "model.pt"
        logits_file = shard_dir / "logits.pt"
        meta_file_shard = shard_dir / "meta.pt"
        model = new_model(data, model_cfg).to(device)
        if resume and ckpt.exists() and logits_file.exists() and meta_file_shard.exists():
            logits = torch.load(logits_file, map_location="cpu")
            meta_s = torch.load(meta_file_shard, map_location="cpu")
        else:
            smask = torch.zeros(data.num_nodes, dtype=torch.bool)
            smask[shard_idx] = True
            model, result, logits = train_model(
                model=model,
                x=data.x,
                y=data.y,
                edge_index=edge_r,
                train_mask=smask,
                val_mask=data.val_mask,
                cfg=local_train_cfg,
                device=device,
                log_csv=shard_dir / "train_log.csv",
            )
            meta_s = {"train_result": result.__dict__, "shard_size": int(smask.sum().item())}
            save_checkpoint(model, ckpt, extra=meta_s)
            torch.save(logits, logits_file)
            torch.save(meta_s, meta_file_shard)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        total_seconds += float(meta_s.get("train_result", {}).get("train_seconds", 0.0))
        peak = max(peak, float(meta_s.get("train_result", {}).get("peak_cuda_mb", 0.0)))
        shard_meta.append(meta_s)
    if logits_sum is None:
        raise ValueError("GraphEraser partition produced no shards.")
    agg_logits = logits_sum / len(shards)
    meta = {
        "update_seconds": total_seconds,
        "peak_cuda_mb": peak,
        "num_shards": len(shards),
        "partition": cfg.eraser_partition,
        "shards": shard_meta,
        "variant": "mean_logit_aggregation",
    }
    torch.save(agg_logits, aggregate_file)
    torch.save(meta, meta_file)
    return None, agg_logits, meta


def utu_edge_unlearn(
    original_model: torch.nn.Module,
    data: GraphData,
    forget_edge_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    from .data import remove_edges_by_idx
    reset_cuda_peak(device)
    start = time.perf_counter()
    edge = remove_edges_by_idx(data.edge_index.to(device), forget_edge_idx.to(device))
    logits = compute_logits(original_model.to(device), data.x.to(device), edge).detach().cpu()
    return original_model, logits, {
        "update_seconds": time.perf_counter() - start,
        "peak_cuda_mb": cuda_peak_memory_mb(device),
        "removed_directed_edges": int(forget_edge_idx.numel()),
        "variant": "unlink_only_edge_unlearning",
    }


def run_unlearner(
    method: str,
    original_model: torch.nn.Module,
    data: GraphData,
    forget_idx: torch.Tensor,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    unlearn_cfg: UnlearnConfig,
    device: torch.device,
    out_dir: Path,
    seed: int,
    resume: bool = True,
    forget_edge_idx: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.nn.Module], torch.Tensor, Dict]:
    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")
    if method == "no_unlearn":
        return no_unlearn(original_model, data, forget_idx, device)
    if method == "finetune_retain":
        return fine_tune_retain(original_model, data, forget_idx, device, unlearn_cfg, only_last_layer=False)
    if method == "last_layer":
        return fine_tune_retain(original_model, data, forget_idx, device, unlearn_cfg, only_last_layer=True)
    if method == "gradient_ascent":
        return gradient_ascent_unlearn(original_model, data, forget_idx, device, unlearn_cfg)
    if method == "gif_diag":
        return gif_diag_unlearn(original_model, data, forget_idx, device, unlearn_cfg)
    if method == "agu_lite":
        return agu_lite_unlearn(original_model, data, forget_idx, device, unlearn_cfg)
    if method == "graph_eraser":
        return graph_eraser_unlearn(data, forget_idx, model_cfg, train_cfg, device, unlearn_cfg, out_dir, seed=seed, resume=resume)
    if method == "utu_edge":
        if forget_edge_idx is None:
            raise ValueError("forget_edge_idx is required for method=utu_edge")
        return utu_edge_unlearn(original_model, data, forget_edge_idx, device)
    raise ValueError(f"Method {method} is handled outside run_unlearner or unsupported.")
