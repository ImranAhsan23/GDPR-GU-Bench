#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import FORGETS, npz_paths, python_cmd, require_files, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Run construct-validity diagnostics for all corpora.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--groups", default=FORGETS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = npz_paths(args.workspace, args.suite)
    require_files(paths)
    logs = Path(args.workspace) / "logs"
    out = Path(args.workspace) / "analysis" / "validity"
    failed = []
    for p in paths:
        cmd = python_cmd("validate_benchmark.py", "--npz", str(p), "--out_dir", str(out), "--groups", args.groups, "--seed", str(args.seed))
        rc = run_command(cmd, log_file=logs / "validity_all.log", dry_run=args.dry_run)
        if rc != 0:
            failed.append((str(p), rc))
            if not args.continue_on_error:
                return rc
    print(f"\n[done] Validity diagnostics written under {out}")
    if failed:
        print("Failures:", failed)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
