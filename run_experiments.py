#!/usr/bin/env python3
"""
Ablation & Experiment Runner
==============================
Runs all experiments needed for the proposal:
  1. Main results across datasets × horizons
  2. Ablation studies (no-wavelet, no-graph, no-crossmodal, no-MoE)
  3. Multi-seed runs for confidence intervals
"""

import subprocess
import sys
import os
import json
import itertools
from pathlib import Path

PYTHON = sys.executable

# ── Experiment definitions ────────────────────────────────────────────────

# Datasets with their data paths (adjust paths as needed)
DATASETS = {
    "ETTh1": {"data_path": None},            # auto-downloads
    "ETTh2": {"data_path": None},
    "ETTm1": {"data_path": None},
    "ETTm2": {"data_path": None},
    "Weather": {"data_path": "./data/weather.csv"},
    "Exchange": {"data_path": "./data/exchange_rate.csv"},
}

HORIZONS = [96, 192, 336, 720]
SEEDS = [42, 123, 456]


def run_cmd(cmd: list[str], desc: str = ""):
    """Run a command and return exit code."""
    cmd_str = " ".join(cmd)
    print(f"\n{'='*60}")
    print(f"Running: {desc or cmd_str}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main_experiments():
    """Main results: WaveMoE across all datasets × horizons × 3 seeds."""
    print("\n" + "=" * 70)
    print("PHASE 1: Main Results")
    print("=" * 70)

    for ds_name, ds_cfg in DATASETS.items():
        for horizon in HORIZONS:
            cmd = [
                PYTHON, "train.py",
                "--dataset", ds_name,
                "--pred_len", str(horizon),
                "--seq_len", "96",
                "--d_model", "128",
                "--dwt_levels", "3",
                "--use_graph", "1",
                "--top_k", "0",
                "--epochs", "50",
                "--seeds", ",".join(str(s) for s in SEEDS),
                "--save_dir", f"./results/main/{ds_name}",
            ]
            if ds_cfg["data_path"]:
                cmd.extend(["--data_path", ds_cfg["data_path"]])

            run_cmd(cmd, f"{ds_name} pred_len={horizon}")


def ablation_experiments():
    """
    Ablation studies on ETTh1 and Weather (pred_len=96):
      A1: No Wavelet (bypass DWT, single-band processing)
      A2: No Graph Learning
      A3: No Cross-Modal Band Attention (the fix we added)
      A4: No MoE (single expert for all bands)
      A5: Sparse Top-K=1 routing vs soft routing
      A6: Different wavelets (haar, db4, sym8, coif3)
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Ablation Studies")
    print("=" * 70)

    ablation_ds = ["ETTh1", "Weather"]
    base_args = [
        "--pred_len", "96", "--seq_len", "96",
        "--d_model", "128", "--epochs", "50",
        "--seeds", ",".join(str(s) for s in SEEDS),
    ]

    ablations = {
        "A1_no_wavelet": ["--dwt_levels", "0"],
        "A2_no_graph": ["--use_graph", "0"],
        "A3_no_crossmodal_fix": [],   # requires code flag (see below)
        "A4_single_expert": [],       # requires code flag
        "A5_sparse_topk1": ["--top_k", "1"],
        "A6_wavelet_haar": ["--wavelet", "haar"],
        "A6_wavelet_sym8": ["--wavelet", "sym8"],
        "A6_wavelet_coif3": ["--wavelet", "coif3"],
    }

    for ds_name in ablation_ds:
        for abl_name, abl_args in ablations.items():
            cmd = [
                PYTHON, "train.py",
                "--dataset", ds_name,
                *base_args,
                *abl_args,
                "--save_dir", f"./results/ablation/{abl_name}/{ds_name}",
            ]
            if DATASETS[ds_name]["data_path"]:
                cmd.extend(["--data_path", DATASETS[ds_name]["data_path"]])

            run_cmd(cmd, f"Ablation {abl_name} on {ds_name}")


def collect_results():
    """Gather all results into a summary table."""
    print("\n" + "=" * 70)
    print("Collecting Results")
    print("=" * 70)

    results = []
    for root, dirs, files in os.walk("./results"):
        for f in files:
            if f == "multi_seed_results.json":
                path = os.path.join(root, f)
                with open(path) as fp:
                    data = json.load(fp)
                # Extract experiment info from path
                parts = Path(root).parts
                results.append({
                    "path": root,
                    "mse_mean": data["mse_mean"],
                    "mse_std": data["mse_std"],
                    "mae_mean": data["mae_mean"],
                    "mae_std": data["mae_std"],
                })

    if results:
        # Print table
        print(f"\n{'Experiment':<60} {'MSE':>20} {'MAE':>20}")
        print("-" * 100)
        for r in sorted(results, key=lambda x: x["path"]):
            mse_str = f"{r['mse_mean']:.6f} ± {r['mse_std']:.6f}"
            mae_str = f"{r['mae_mean']:.6f} ± {r['mae_std']:.6f}"
            print(f"{r['path']:<60} {mse_str:>20} {mae_str:>20}")

        # Save
        with open("./results/summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to ./results/summary.json")
    else:
        print("No results found yet.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["main", "ablation", "collect", "all"])
    args = parser.parse_args()

    if args.phase in ("main", "all"):
        main_experiments()
    if args.phase in ("ablation", "all"):
        ablation_experiments()
    if args.phase in ("collect", "all"):
        collect_results()
