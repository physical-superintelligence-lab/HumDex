#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click dataset builder using the strict two-step pipeline:
1) build_wuji_hand_policy_dataset.py
2) collect2.py

This keeps the final NPZ format identical to collect2.py output.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run build_wuji_hand_policy_dataset.py then collect2.py.")
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(_repo_root() / "deploy_real" / "humdex demonstration" / "20260306_2310_twist2_left"),
        help="Input root containing episode_*/data.json.",
    )
    parser.add_argument(
        "--hand_side",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="Which hand side to export.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_repo_root() / "wuji_policy" / "data"),
        help="Directory for final merged npz.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Final output name, with or without .npz. Default: wuji_<hand_side>.npz",
    )
    parser.add_argument(
        "--intermediate_root",
        type=str,
        default="",
        help="Optional intermediate output directory for per-segment npz files.",
    )
    return parser.parse_args()


def _resolve_output_path(output_dir: Path, output_name: str, hand_side: str) -> Path:
    if output_name:
        filename = output_name if output_name.endswith(".npz") else f"{output_name}.npz"
    else:
        filename = f"wuji_{hand_side}.npz"
    return (output_dir / filename).resolve()


def _load_collect2_module(collect2_path: Path):
    spec = importlib.util.spec_from_file_location("collect2_module", str(collect2_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec: {collect2_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    script_dir = Path(__file__).resolve().parent

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = _resolve_output_path(output_dir, args.output_name.strip(), args.hand_side)

    if args.intermediate_root.strip():
        intermediate_root = Path(args.intermediate_root).expanduser().resolve()
    else:
        intermediate_root = (repo_root / "deploy_real" / "wuji_hand_policy_dataset_tmp" / f"{output_path.stem}_{args.hand_side}").resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"[ERROR] input_root does not exist: {input_root}", file=sys.stderr)
        return 1

    build_script = (script_dir / "build_wuji_hand_policy_dataset.py").resolve()
    collect2_script = (script_dir / "collect2.py").resolve()

    print(f"[INFO] input_root    : {input_root}")
    print(f"[INFO] hand_side     : {args.hand_side}")
    print(f"[INFO] intermediate  : {intermediate_root}")
    print(f"[INFO] output_npz    : {output_path}")

    build_cmd = [
        sys.executable,
        str(build_script),
        "--input_root",
        str(input_root),
        "--hand_side",
        args.hand_side,
        "--output_root",
        str(intermediate_root),
        "--overwrite",
    ]
    subprocess.run(build_cmd, check=True)

    collect2 = _load_collect2_module(collect2_script)
    collect2.SRC_DIRS = [intermediate_root]
    collect2.OUT_PATH = output_path
    if args.hand_side == "left":
        collect2.FILE_PATTERNS = ["*left.npz", "*left*.npz"]
    elif args.hand_side == "right":
        collect2.FILE_PATTERNS = ["*right.npz", "*right*.npz"]
    else:
        collect2.FILE_PATTERNS = ["*left*.npz", "*right*.npz"]

    collect2.main()
    print(f"[DONE] Dataset written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
