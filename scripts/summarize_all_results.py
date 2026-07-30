#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import dataset_specs, python_cmd, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Summarize all experiment folders and create tables/figures.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--include_matched", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    logs = workspace / "logs"
    failed = []
    for spec in dataset_specs(args.suite):
        data_name = spec.name.replace("gdpr_", "")
        for kind in ["main"] + (["matched"] if args.include_matched else []):
            run_dir = workspace / "runs" / f"{kind}_{data_name}"
            if not run_dir.exists():
                print(f"[skip] missing run_dir={run_dir}")
                continue
            out_dir = workspace / "analysis" / f"results_{kind}_{data_name}"
            cmd = python_cmd("summarize_results.py", "--run_dir", str(run_dir), "--out_dir", str(out_dir))
            rc = run_command(cmd, log_file=logs / "summarize_all.log", dry_run=args.dry_run)
            if rc != 0:
                failed.append((str(run_dir), rc))
                if not args.continue_on_error:
                    return rc
    print(f"\n[done] Summaries are under {workspace / 'analysis'}")
    if failed:
        print("Failures:", failed)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
