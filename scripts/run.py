#!/usr/bin/env python3
"""
Thin wrapper around vturrisi/solo-learn main_pretrain.py.

- Works both on Kaggle and locally.
- Lets you pass ANY solo-learn Hydra config override unchanged.
- Auto-sets checkpoint and output directories if you don't specify them.
- Calls the repo's main_pretrain.py via subprocess (no import changes).

Usage examples:
  python scripts/run.py --repo_dir ./learning/solo-learn \
      -- --config-path scripts/pretrain/cifar/ --config-name simclr.yaml \
         name="simclr-cifar100" data.dataset=cifar100

Note: main_pretrain.py uses Hydra exclusively, so all overrides must be in Hydra format
(e.g., checkpoint.dir=/path, data.dataset=cifar100, not --checkpoint_dir or --dataset).
"""
import os
import sys
import shlex
import argparse
import subprocess
from pathlib import Path

# ---------- helpers ----------
def in_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

def default_data_root() -> Path:
    return Path("/kaggle/input") if in_kaggle() else Path(os.environ.get("LOCAL_DATA", "./data"))

def default_work_dir() -> Path:
    return Path("/kaggle/working") if in_kaggle() else Path(os.environ.get("LOCAL_WORK", "./runs"))

def detect_python() -> str:
    return sys.executable or "python"

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run vturrisi/solo-learn main_pretrain.py with pass-through options."
    )
    p.add_argument("--repo_dir", required=True,
                   help="Path to the root of vturrisi/solo-learn (e.g., /kaggle/input/solo-learn).")
    p.add_argument("--data_dir", default=None,
                   help="If not set, will use /kaggle/input or ./data depending on environment.")
    p.add_argument("--work_dir", default=None,
                   help="If not set, will use /kaggle/working or ./runs depending on environment.")
    p.add_argument("--print_cmd", action="store_true", help="Print the final command before running.")
    p.add_argument("--dry_run", action="store_true", help="Only print command; do not execute.")
    # Everything after `--` goes straight to main_pretrain.py (Hydra or argparse).
    p.add_argument("solo_args", nargs=argparse.REMAINDER,
                   help="Use `--` and then put any main_pretrain.py options verbatim.")
    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    entry = repo_dir / "main_pretrain.py"
    if not entry.exists():
        raise SystemExit(f"[ERR] main_pretrain.py not found at: {entry}")

    # Auto paths if user didn't specify them in the forwarded args
    data_dir = Path(args.data_dir) if args.data_dir else default_data_root()
    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build base command
    py = detect_python()
    cmd_parts = [py, str(entry)]

    # Parse forwarded args to separate Hydra flags from overrides
    if args.solo_args:
        # If the first is a literal "--", drop it (common in argparse remainder)
        cleaned = args.solo_args[1:] if args.solo_args[0] == "--" else args.solo_args
    else:
        cleaned = []

    # Separate Hydra flags (--config-path, --config-name) from overrides (key=value)
    hydra_flags = []
    overrides = []
    
    i = 0
    while i < len(cleaned):
        arg = cleaned[i]
        # Check if it's a Hydra flag (starts with --)
        if arg.startswith("--"):
            hydra_flags.append(arg)
            # Some flags like --config-path take a value
            if arg in ["--config-path", "--config-name"] and i + 1 < len(cleaned):
                hydra_flags.append(cleaned[i + 1])
                i += 2
                continue
            i += 1
        else:
            # It's an override (key=value format)
            overrides.append(arg)
            i += 1

    # Add Hydra flags first (required to come before overrides)
    cmd_parts.extend(hydra_flags)

    # Check forwarded args for existing overrides
    forwarded_overrides = " ".join(overrides)

    def not_in_forwarded(substrs):
        return all(s not in forwarded_overrides for s in substrs)

    # checkpoint dir default (Hydra-style: checkpoint.dir=...)
    if not_in_forwarded(["checkpoint.dir="]):
        overrides.insert(0, f"checkpoint.dir={work_dir / 'checkpoints'}")

    # Hydra output directory (optional, but helps organize outputs)
    if not_in_forwarded(["hydra.run.dir="]):
        overrides.insert(0, f"hydra.run.dir={work_dir / 'hydra_outputs'}")

    # Note: data.train_path is dataset-specific and usually set in config files,
    # so we don't auto-set it here. Users should override it in their config or via Hydra.

    # Add all overrides (user-provided + defaults)
    cmd_parts.extend(overrides)

    cmd = " ".join(shlex.quote(x) for x in cmd_parts)

    if args.print_cmd or args.dry_run:
        print("[solo-learn cmd]")
        print(cmd)

    if args.dry_run:
        return

    # Run in the repo directory so relative config paths (if any) resolve like upstream.
    subprocess.run(cmd, shell=True, check=True, cwd=str(repo_dir))

if __name__ == "__main__":
    main()
