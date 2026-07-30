#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline_common import python_cmd, run_command


def parse_args():
    p = argparse.ArgumentParser(description="Run the complete GDPRBench pipeline using Python only.")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--suite", default="main", choices=["smoke", "main", "extended", "large", "all", "all_public"])
    p.add_argument("--preset", default="quick", choices=["quick", "paper"])
    p.add_argument("--device", default="auto")
    p.add_argument("--with_matched_experiments", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    logs = workspace / "logs"
    steps = [
        ("check_gpu", python_cmd("check_gpu.py")),
        ("build_all_corpora", python_cmd("build_all_corpora.py", "--workspace", str(workspace), "--suite", args.suite)),
        ("verify_all_npz", python_cmd("verify_all_npz.py", "--workspace", str(workspace), "--suite", args.suite)),
        ("run_all_validity", python_cmd("run_all_validity.py", "--workspace", str(workspace), "--suite", args.suite)),
        ("create_all_matched_controls", python_cmd("create_all_matched_controls.py", "--workspace", str(workspace), "--suite", args.suite)),
        ("run_all_main_experiments", python_cmd("run_all_main_experiments.py", "--workspace", str(workspace), "--suite", args.suite, "--preset", args.preset, "--device", args.device)),
    ]
    if args.with_matched_experiments:
        steps.append(("run_all_matched_experiments", python_cmd("run_all_matched_experiments.py", "--workspace", str(workspace), "--suite", args.suite, "--preset", args.preset, "--device", args.device)))
        summary_cmd = python_cmd("summarize_all_results.py", "--workspace", str(workspace), "--suite", args.suite, "--include_matched")
    else:
        summary_cmd = python_cmd("summarize_all_results.py", "--workspace", str(workspace), "--suite", args.suite)
    steps.append(("summarize_all_results", summary_cmd))

    failures = []
    for name, cmd in steps:
        rc = run_command(cmd, log_file=logs / f"pipeline_{name}.log", dry_run=args.dry_run)
        if rc != 0:
            failures.append((name, rc))
            if not args.continue_on_error:
                return rc
    print(f"\n[done] Complete pipeline finished. Workspace: {workspace}")
    print(f"Results: {workspace / 'analysis'}")
    if failures:
        print("Failures:", failures)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
