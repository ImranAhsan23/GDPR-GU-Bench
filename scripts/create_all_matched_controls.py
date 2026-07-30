#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import FORGETS, npz_paths, python_cmd, require_files, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Create matched-control forget sets for all selected datasets.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--targets", default=FORGETS)
    p.add_argument("--modes", default="uniform,label,degree,label_degree,label_degree_profile")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = npz_paths(args.workspace, args.suite)
    require_files(paths)
    out = Path(args.workspace) / "matched_controls"
    logs = Path(args.workspace) / "logs"
    failed = []
    for p in paths:
        cmd = python_cmd(
            "create_matched_controls.py",
            "--npz", str(p),
            "--out_dir", str(out),
            "--targets", args.targets,
            "--modes", args.modes,
            "--seed", str(args.seed),
        )
        rc = run_command(cmd, log_file=logs / "matched_controls_all.log", dry_run=args.dry_run)
        if rc != 0:
            failed.append((str(p), rc))
            if not args.continue_on_error:
                return rc
    print(f"\n[done] Matched controls written under {out}")
    if failed:
        print("Failures:", failed)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
