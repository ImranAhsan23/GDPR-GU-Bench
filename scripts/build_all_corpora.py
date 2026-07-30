#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import dataset_specs, python_cmd, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Build all GDPRBench corpora and NPZ files.")
    p.add_argument("--workspace", default="workspace", help="Output folder. Creates <workspace>/data and <workspace>/corpora.")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"], help="Dataset set to build.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--corruption_rate", type=float, default=0.25)
    p.add_argument("--mixture_profile", default="default", choices=["default", "balanced", "noise_heavy", "outdated_heavy", "wrongfact_heavy"])
    p.add_argument("--raw_root", default=None, help="Where PyG/OGB downloads are cached. Default: <workspace>/raw_public")
    p.add_argument("--npz_only", action="store_true", help="Write only NPZ files; skip CSV corpus artifacts.")
    p.add_argument("--include_large_edges_csv", action="store_true", help="Also write edges.csv for large datasets. This can be very large.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--force_redownload", action="store_true", help="Delete raw/processed cache before rebuilding each dataset.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    raw_root = Path(args.raw_root) if args.raw_root else workspace / "raw_public"
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    failed = []
    for spec in dataset_specs(args.suite):
        cmd = python_cmd(
            "build_gdpr_corpus.py",
            "--dataset", spec.dataset,
            "--name", spec.name,
            "--out_root", str(workspace),
            "--raw_root", str(raw_root),
            "--seed", str(args.seed),
            "--corruption_rate", str(args.corruption_rate),
            "--mixture_profile", args.mixture_profile,
        )
        if args.force_redownload:
            cmd.append("--force_redownload")
        if spec.dataset == "synthetic":
            cmd += ["--synthetic_nodes", "1000", "--synthetic_avg_degree", "8"]
        if args.npz_only or spec.no_csv:
            cmd.append("--no_csv")
        elif spec.no_edges_csv and not args.include_large_edges_csv:
            cmd.append("--no_edges_csv")
        rc = run_command(cmd, log_file=logs / f"build_{spec.name}.log", dry_run=args.dry_run)
        if rc != 0:
            failed.append((spec.name, rc))
            if not args.continue_on_error:
                return rc

    print("\n[done] Build step completed.")
    print(f"NPZ files:     {workspace / 'data'}")
    print(f"CSV corpora:   {workspace / 'corpora'}")
    print(f"Logs:          {logs}")
    if failed:
        print("Failed builds:", failed)
        if args.continue_on_error:
            print("[warning] Some datasets failed, but --continue_on_error was used, so the pipeline returns 0. Verify outputs with verify_all_npz.py.")
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
