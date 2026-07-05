#!/usr/bin/env python3
"""
Figure Generation for Proposal
================================
Generates publication-quality figures from experiment outputs:
  1. Training/validation loss curves
  2. Expert routing heatmaps (which expert handles which band)
  3. Cross-modal attention weights per frequency band
  4. Forecast vs ground truth plots
  5. Ablation comparison bar charts

Usage:
  python generate_figures.py --results_dir ./results --output_dir ./figures
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_training_curves(history_path: str, out_path: str):
    """Plot train loss + val MSE/MAE over epochs."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_mse = [h["val_mse"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, "b-", linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, val_mse, "r--", linewidth=1.5, label="Val MSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / MSE")
    ax1.set_title("Training Convergence")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(epochs, val_mae, "g-", linewidth=1.5, label="Val MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.set_title("Validation MAE")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_forecast_comparison(preds_path: str, trues_path: str, out_path: str,
                             n_samples: int = 3, n_channels: int = 3):
    """Plot predicted vs actual for a few test samples."""
    preds = np.load(preds_path)  # (N, H, C)
    trues = np.load(trues_path)  # (N, H, C)

    fig, axes = plt.subplots(
        n_samples, n_channels, figsize=(4 * n_channels, 3 * n_samples),
        sharex=True,
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    if n_channels == 1:
        axes = axes[:, np.newaxis]

    # Pick evenly spaced samples
    sample_indices = np.linspace(0, len(preds) - 1, n_samples, dtype=int)

    for i, idx in enumerate(sample_indices):
        for j in range(min(n_channels, preds.shape[2])):
            ax = axes[i, j]
            H = preds.shape[1]
            t = np.arange(H)
            ax.plot(t, trues[idx, :, j], "k-", linewidth=1.2, label="Ground Truth")
            ax.plot(t, preds[idx, :, j], "r--", linewidth=1.2, label="WaveMoE")
            if i == 0:
                ax.set_title(f"Channel {j}")
            if j == 0:
                ax.set_ylabel(f"Sample {idx}")
            if i == n_samples - 1:
                ax.set_xlabel("Horizon")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.suptitle("Forecast vs Ground Truth", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_expert_routing(checkpoint_path: str, out_path: str):
    """
    Visualise expert routing weights from a trained model.
    Loads model, runs a dummy forward pass, and plots gating weights.
    """
    import torch
    from wavemoe.model import WaveMoE

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model_cfg_path = Path(checkpoint_path).parent / "model_config.json"

    if model_cfg_path.exists():
        with open(model_cfg_path) as f:
            model_cfg = json.load(f)
    else:
        print(f"Warning: model_config.json not found at {model_cfg_path}")
        return

    model = WaveMoE(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Create dummy input
    B = 4
    T = model_cfg["seq_len"]
    x_mods = [
        torch.randn(B, T, c)
        for c in model_cfg["modality_channels"]
    ]

    with torch.no_grad():
        _ = model(x_mods)

    # Extract gating weights by running through the pipeline manually
    # For visualisation, we average over the batch
    encoded = [enc(x) for enc, x in zip(model.encoders, x_mods)]
    all_bands = [model.dwt(e) for e in encoded]
    n_bands = len(all_bands[0])

    # Get gating weights for first modality
    cross_bands = [[] for _ in range(model.n_modalities)]
    for l in range(n_bands):
        reps = [all_bands[m][l] for m in range(model.n_modalities)]
        enhanced, _ = model.band_cross_modal[l](reps)
        for m in range(model.n_modalities):
            cross_bands[m].append(enhanced[m])

    if model.use_graph:
        for m in range(model.n_modalities):
            for l in range(n_bands):
                g_out, _ = model.graph_layers[l](cross_bands[m][l])
                cross_bands[m][l] = g_out

    weights, _ = model.gating(cross_bands[0])
    w_np = weights.mean(dim=0).numpy()  # (n_bands, n_experts)

    # Plot
    expert_names = ["SSM\n(coarse)", "GRU\n(mid)", "Transformer\n(fine)", "TCN\n(finest)"]
    band_names = [f"Band {i}\n({'approx' if i == 0 else f'detail {i}'})" for i in range(n_bands)]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(w_np, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(expert_names)))
    ax.set_xticklabels(expert_names)
    ax.set_yticks(range(len(band_names)))
    ax.set_yticklabels(band_names)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Frequency Band")
    ax.set_title("Expert Routing Weights")

    # Annotate
    for i in range(w_np.shape[0]):
        for j in range(w_np.shape[1]):
            text_color = "white" if w_np[i, j] > 0.5 else "black"
            ax.text(j, i, f"{w_np[i, j]:.2f}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_ablation_comparison(results_dir: str, out_path: str):
    """Bar chart comparing ablation variants."""
    ablation_results = {}
    results_root = Path(results_dir) / "ablation"

    if not results_root.exists():
        print(f"No ablation results found at {results_root}")
        return

    for abl_dir in sorted(results_root.iterdir()):
        if abl_dir.is_dir():
            for ds_dir in abl_dir.iterdir():
                summary = ds_dir / "multi_seed_results.json"
                if summary.exists():
                    with open(summary) as f:
                        data = json.load(f)
                    key = f"{abl_dir.name}"
                    if key not in ablation_results:
                        ablation_results[key] = {}
                    ablation_results[key][ds_dir.name] = {
                        "mse": data["mse_mean"],
                        "mse_std": data["mse_std"],
                        "mae": data["mae_mean"],
                        "mae_std": data["mae_std"],
                    }

    if not ablation_results:
        print("No ablation results to plot.")
        return

    # Get datasets
    datasets = sorted(set(
        ds for v in ablation_results.values() for ds in v.keys()
    ))

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    ablation_names = sorted(ablation_results.keys())
    x = np.arange(len(ablation_names))
    width = 0.35

    for ax, ds in zip(axes, datasets):
        mses = [
            ablation_results.get(a, {}).get(ds, {}).get("mse", 0)
            for a in ablation_names
        ]
        mse_stds = [
            ablation_results.get(a, {}).get(ds, {}).get("mse_std", 0)
            for a in ablation_names
        ]
        maes = [
            ablation_results.get(a, {}).get(ds, {}).get("mae", 0)
            for a in ablation_names
        ]
        mae_stds = [
            ablation_results.get(a, {}).get(ds, {}).get("mae_std", 0)
            for a in ablation_names
        ]

        ax.bar(x - width / 2, mses, width, yerr=mse_stds,
               label="MSE", capsize=3, color="#4C78A8")
        ax.bar(x + width / 2, maes, width, yerr=mae_stds,
               label="MAE", capsize=3, color="#F58518")

        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in ablation_names],
                           fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"Ablation Study — {ds}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_horizon_comparison(results_dir: str, out_path: str):
    """Line plot: MSE/MAE across forecast horizons for each dataset."""
    main_root = Path(results_dir) / "main"
    if not main_root.exists():
        print(f"No main results at {main_root}")
        return

    dataset_results = {}
    for ds_dir in sorted(main_root.iterdir()):
        if ds_dir.is_dir():
            for sub in ds_dir.iterdir():
                summary = sub / "multi_seed_results.json"
                if summary.exists():
                    with open(summary) as f:
                        data = json.load(f)
                    # Extract horizon from per_seed results
                    if data.get("per_seed"):
                        horizon = data["per_seed"][0].get("pred_len", 96)
                    else:
                        continue
                    ds_name = ds_dir.name
                    if ds_name not in dataset_results:
                        dataset_results[ds_name] = []
                    dataset_results[ds_name].append({
                        "horizon": horizon,
                        "mse": data["mse_mean"],
                        "mse_std": data["mse_std"],
                        "mae": data["mae_mean"],
                        "mae_std": data["mae_std"],
                    })

    if not dataset_results:
        print("No horizon results to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_results)))
    for (ds_name, results), color in zip(sorted(dataset_results.items()), colors):
        results.sort(key=lambda x: x["horizon"])
        horizons = [r["horizon"] for r in results]
        mses = [r["mse"] for r in results]
        mse_stds = [r["mse_std"] for r in results]
        maes = [r["mae"] for r in results]
        mae_stds = [r["mae_std"] for r in results]

        ax1.errorbar(horizons, mses, yerr=mse_stds, marker="o",
                     linewidth=1.5, label=ds_name, color=color, capsize=3)
        ax2.errorbar(horizons, maes, yerr=mae_stds, marker="s",
                     linewidth=1.5, label=ds_name, color=color, capsize=3)

    ax1.set_xlabel("Forecast Horizon")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE across Horizons")
    ax1.legend()
    ax2.set_xlabel("Forecast Horizon")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE across Horizons")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for expert routing plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Training curves — find all train_history.json files
    for root, dirs, files in os.walk(args.results_dir):
        if "train_history.json" in files:
            name = Path(root).relative_to(args.results_dir)
            name_clean = str(name).replace("/", "_").replace("\\", "_")
            out = os.path.join(args.output_dir, f"train_curve_{name_clean}.pdf")
            plot_training_curves(os.path.join(root, "train_history.json"), out)

        if "test_preds.npy" in files and "test_trues.npy" in files:
            name = Path(root).relative_to(args.results_dir)
            name_clean = str(name).replace("/", "_").replace("\\", "_")
            out = os.path.join(args.output_dir, f"forecast_{name_clean}.pdf")
            plot_forecast_comparison(
                os.path.join(root, "test_preds.npy"),
                os.path.join(root, "test_trues.npy"),
                out,
            )

    # 2. Expert routing
    if args.checkpoint:
        plot_expert_routing(
            args.checkpoint,
            os.path.join(args.output_dir, "expert_routing.pdf"),
        )

    # 3. Ablation bar chart
    plot_ablation_comparison(
        args.results_dir,
        os.path.join(args.output_dir, "ablation_comparison.pdf"),
    )

    # 4. Horizon comparison
    plot_horizon_comparison(
        args.results_dir,
        os.path.join(args.output_dir, "horizon_comparison.pdf"),
    )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
