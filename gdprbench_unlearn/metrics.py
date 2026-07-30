from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def masked_numpy(t: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return t.detach().cpu()[mask.detach().cpu()].numpy()


def classification_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return {prefix + "accuracy": float("nan"), prefix + "macro_f1": float("nan"), prefix + "micro_f1": float("nan")}
    pred = logits.argmax(dim=-1)
    yt = masked_numpy(y, mask)
    yp = masked_numpy(pred, mask)
    return {
        prefix + "accuracy": float(accuracy_score(yt, yp)),
        prefix + "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        prefix + "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
    }


def prediction_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return float("nan")
    pa = logits_a.argmax(-1)[mask]
    pb = logits_b.argmax(-1)[mask]
    return float((pa == pb).float().mean().item())


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return float("nan")
    logpa = F.log_softmax(logits_a[mask], dim=-1)
    logpb = F.log_softmax(logits_b[mask], dim=-1)
    pa = logpa.exp()
    pb = logpb.exp()
    kl_ab = F.kl_div(logpa, pb, reduction="batchmean")
    kl_ba = F.kl_div(logpb, pa, reduction="batchmean")
    return float(0.5 * (kl_ab + kl_ba).item())


def posterior_features(logits: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=-1).values
        if top2.shape[1] == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs.clamp(min=1e-12).log() * probs).sum(dim=-1)
        loss = F.cross_entropy(logits, y, reduction="none")
        correct = (pred == y).float()
        feats = torch.stack([conf, margin, entropy, loss, correct], dim=1)
    return feats.detach().cpu().numpy()


def membership_inference_auc(
    logits: torch.Tensor,
    y: torch.Tensor,
    member_mask: torch.Tensor,
    nonmember_mask: torch.Tensor,
    seed: int = 0,
    max_points_per_class: int = 20000,
) -> Dict[str, float]:
    """Posterior-feature membership inference diagnostic.
    """
    member_idx = torch.where(member_mask.bool())[0].detach().cpu().numpy()
    non_idx = torch.where(nonmember_mask.bool())[0].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    if len(member_idx) < 5 or len(non_idx) < 5:
        return {"raw_auc": float("nan"), "auc_star": float("nan"), "attack_points": float(len(member_idx) + len(non_idx))}
    if len(member_idx) > max_points_per_class:
        member_idx = rng.choice(member_idx, size=max_points_per_class, replace=False)
    if len(non_idx) > max_points_per_class:
        non_idx = rng.choice(non_idx, size=max_points_per_class, replace=False)

    feats = posterior_features(logits, y)
    X = np.concatenate([feats[member_idx], feats[non_idx]], axis=0)
    labels = np.concatenate([np.ones(len(member_idx)), np.zeros(len(non_idx))])
    if len(np.unique(labels)) < 2:
        return {"raw_auc": float("nan"), "auc_star": float("nan"), "attack_points": float(len(labels))}
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.5, random_state=seed, stratify=labels)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, score))
        return {"raw_auc": auc, "auc_star": max(auc, 1.0 - auc), "attack_points": float(len(labels))}
    except Exception:
        return {"raw_auc": float("nan"), "auc_star": float("nan"), "attack_points": float(len(labels))}


def khop_mask(edge_index: torch.Tensor, start_idx: torch.Tensor, num_nodes: int, k: int = 2, exclude_start: bool = False) -> torch.Tensor:
    device = edge_index.device
    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if start_idx.numel() == 0:
        return visited
    frontier[start_idx.to(device)] = True
    visited |= frontier
    src, dst = edge_index
    for _ in range(k):
        # Treat graph as directed as stored; if the NPZ is undirected-as-bidirected,
        # this equals undirected k-hop expansion.
        e_mask = frontier[src]
        nxt = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if int(e_mask.sum().item()) > 0:
            nxt[dst[e_mask]] = True
        e_mask_rev = frontier[dst]
        if int(e_mask_rev.sum().item()) > 0:
            nxt[src[e_mask_rev]] = True
        nxt = nxt & (~visited)
        visited |= nxt
        frontier = nxt
        if int(frontier.sum().item()) == 0:
            break
    if exclude_start:
        visited[start_idx.to(device)] = False
    return visited


def locality_metrics(
    before_logits: Optional[torch.Tensor],
    after_logits: torch.Tensor,
    retrain_logits: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    forget_idx: torch.Tensor,
    num_nodes: int,
    k: int = 2,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    fmask = torch.zeros(num_nodes, dtype=torch.bool, device=after_logits.device)
    if forget_idx.numel() > 0:
        fmask[forget_idx.to(after_logits.device)] = True
    nmask = khop_mask(edge_index.to(after_logits.device), forget_idx.to(after_logits.device), num_nodes, k=k, exclude_start=True)
    out["forget_size"] = float(fmask.sum().item())
    out[f"{k}hop_size"] = float(nmask.sum().item())
    if before_logits is not None:
        before_logits = before_logits.to(after_logits.device)
        out["kl_forget_before_after"] = symmetric_kl(before_logits, after_logits, fmask)
        out[f"kl_{k}hop_before_after"] = symmetric_kl(before_logits, after_logits, nmask)
    if retrain_logits is not None:
        retrain_logits = retrain_logits.to(after_logits.device)
        out["kl_forget_retrain_after"] = symmetric_kl(retrain_logits, after_logits, fmask)
        out[f"kl_{k}hop_retrain_after"] = symmetric_kl(retrain_logits, after_logits, nmask)
        out["agreement_forget_retrain_after"] = prediction_agreement(retrain_logits, after_logits, fmask)
        out[f"agreement_{k}hop_retrain_after"] = prediction_agreement(retrain_logits, after_logits, nmask)
    return out
