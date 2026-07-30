from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
FORGETS = "all_corrupted,wrong_fact,misattributed,outdated,invalid_combination,incorrect_derived_label"
METHODS = "retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite,graph_eraser"
METHODS_NO_ERASER = "retrain,no_unlearn,finetune_retain,last_layer,gradient_ascent,gif_diag,agu_lite"

@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    name: str
    no_edges_csv: bool = False
    no_csv: bool = False


def dataset_specs(suite: str) -> List[DatasetSpec]:
    """Dataset suites used by the Python-only pipeline.
    """
    suite = suite.lower().strip()
    main = [
        DatasetSpec("lastfm_asia", "gdpr_lastfm_asia"),
        DatasetSpec("yelp", "gdpr_yelp", no_edges_csv=True),
    ]
    extended_extra = [
        DatasetSpec("pubmed", "gdpr_pubmed"),
        DatasetSpec("flickr", "gdpr_flickr", no_edges_csv=True),
        DatasetSpec("ogbn-arxiv", "gdpr_ogbn_arxiv", no_edges_csv=True),
    ]
    large_extra = [
        DatasetSpec("ogbn-products", "gdpr_ogbn_products", no_csv=True),
    ]
    all_public_extra = [
        DatasetSpec("cora", "gdpr_cora"),
        DatasetSpec("citeseer", "gdpr_citeseer"),
        DatasetSpec("amazon_computers", "gdpr_amazon_computers"),
        DatasetSpec("amazon_photo", "gdpr_amazon_photo"),
        DatasetSpec("coauthor_cs", "gdpr_coauthor_cs", no_edges_csv=True),
        DatasetSpec("coauthor_physics", "gdpr_coauthor_physics", no_edges_csv=True),
    ]
    if suite == "smoke":
        return [DatasetSpec("synthetic", "gdpr_synth_smoke", no_edges_csv=True)]
    if suite == "main":
        return main
    if suite == "extended":
        return main + extended_extra
    if suite == "large":
        return main + extended_extra + large_extra
    if suite in {"all", "all_public"}:
        return main + all_public_extra + extended_extra
    raise ValueError(f"Unknown suite={suite!r}. Use smoke, main, extended, large, all_public, or all.")


def npz_paths(workspace: str | Path, suite: str) -> List[Path]:
    w = Path(workspace)
    return [w / "data" / f"{spec.name}.npz" for spec in dataset_specs(suite)]


def run_command(cmd: Sequence[str], log_file: str | Path | None = None, dry_run: bool = False) -> int:
    printable = " ".join(str(c) for c in cmd)
    print("\n" + "=" * 100)
    print(printable)
    if dry_run:
        return 0
    start = time.time()
    log_handle = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.write(printable + "\n")
        log_handle.flush()
    try:
        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            if log_handle is not None:
                log_handle.write(line)
        return_code = proc.wait()
        elapsed = time.time() - start
        msg = f"\n[finished] return_code={return_code} elapsed_seconds={elapsed:.2f}\n"
        print(msg)
        if log_handle is not None:
            log_handle.write(msg)
        return return_code
    finally:
        if log_handle is not None:
            log_handle.close()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def python_cmd(script_name: str, *args: str) -> List[str]:
    return [sys.executable, str(SCRIPT_DIR / script_name), *map(str, args)]
