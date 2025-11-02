#!/usr/bin/env python3
"""
Thin wrapper around vturrisi/solo-learn main_pretrain.py.

- Works both on Kaggle and locally.
- Lets you pass ANY solo-learn flag unchanged (Hydra or argparse).
- Auto-sets data dir and working dir if you don't specify them.
- Calls the repo's main_pretrain.py via subprocess (no import changes).

Usage examples (see bottom or README):
  python run_solo_pretrain.py --repo_dir /kaggle/input/solo-learn \
      -- extra method=simclr data.dataset=cifar10 trainer.max_epochs=100 optimizer.batch_size=256

  python run_solo_pretrain.py --repo_dir /kaggle/input/solo-learn \
      -- --dataset cifar10 --backbone resnet18 --max_epochs 100 --batch_size 256 --method simclr
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

    # We *optionally* inject defaults ONLY if user didn't already pass an explicit value.
    # Works for both legacy argparse flags (e.g., --data_dir) and Hydra keys (data.data_dir=...).
    forwarded = " ".join(args.solo_args)

    def not_in_forwarded(substrs):
        return all(s not in forwarded for s in substrs)

    # data dir default
    if not_in_forwarded(["--data_dir", "data.data_dir="]):
        cmd_parts += ["--data_dir", str(data_dir)]

    # sensible default for checkpoints/logs if not provided
    if not_in_forwarded(["--checkpoint_dir", "paths.checkpoint_dir="]):
        cmd_parts += ["--checkpoint_dir", str(work_dir / "checkpoints")]
    if not_in_forwarded(["--log_dir", "paths.log_dir="]):
        cmd_parts += ["--log_dir", str(work_dir / "logs")]

    # device/mixed precision are handled by solo-learn (Lightning/Hydra) if you pass them.
    # We don't override—keep wrapper thin and version-proof.

    # Finally add the raw solo-learn args (after a lone "--", Jupyter strips it; argparse keeps it in solo_args already)
    if args.solo_args:
        # If the first is a literal "--", drop it (common in argparse remainder)
        cleaned = args.solo_args[1:] if args.solo_args[0] == "--" else args.solo_args
        cmd_parts += cleaned

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
