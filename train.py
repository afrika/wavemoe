#!/usr/bin/env python3
"""
WaveMoE Training Script
========================
End-to-end training with:
  - Mixed precision (AMP fp16)
  - Gradient clipping
  - OneCycleLR scheduler
  - Early stopping
  - Checkpoint save/resume
  - TensorBoard logging
  - Per-epoch evaluation (MSE, MAE)
  - Interpretability export (routing weights, cross-modal attention)

Usage:
  python train.py --config configs/etth1.yaml
  python train.py --dataset ETTh1 --seq_len 96 --pred_len 96
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# AMP: compatible with PyTorch 2.x
try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast(enabled=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return _autocast(device_type=device, enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from wavemoe.model import WaveMoE
from wavemoe.data import build_dataloaders, build_timemmd_dataloaders, collate_multimodal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────
def mse_metric(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae_metric(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


# ── Seed ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Training loop ─────────────────────────────────────────────────────────
def train_one_epoch(
    model: WaveMoE,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    criterion,
    cfg: dict,
    device: torch.device,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    lambda_aux = cfg.get("lambda_aux", 0.01)
    lambda_graph = cfg.get("lambda_graph", 0.001)

    for batch_idx, (x_mods, y) in enumerate(loader):
        x_mods = [x.to(device, non_blocking=True) for x in x_mods]
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=bool(cfg.get("use_amp", True))):
            pred = model(x_mods)
            loss_main = criterion(pred, y)

            # Auxiliary losses
            aux = model.get_loss_components()
            loss = (
                loss_main
                + lambda_aux * aux["aux_loss"]
                + lambda_graph * aux["graph_reg"]
            )

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), cfg.get("max_grad_norm", 1.0),
        )

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss_main.item()
        n_batches += 1

        if (batch_idx + 1) % cfg.get("log_every", 50) == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={loss_main.item():.6f} aux={aux['aux_loss']:.4f} "
                f"grad={grad_norm:.3f} lr={lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: WaveMoE,
    loader,
    criterion,
    device: torch.device,
    cfg: dict,
):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    n_batches = 0

    for x_mods, y in loader:
        x_mods = [x.to(device, non_blocking=True) for x in x_mods]
        y = y.to(device, non_blocking=True)

        with autocast(enabled=bool(cfg.get("use_amp", True))):
            pred = model(x_mods)
            loss = criterion(pred, y)

        total_loss += loss.item()
        n_batches += 1
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mse_metric(preds, trues)
    mae = mae_metric(preds, trues)
    avg_loss = total_loss / max(n_batches, 1)

    return {"loss": avg_loss, "mse": mse, "mae": mae, "preds": preds, "trues": trues}


# ── Checkpoint ────────────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, cfg):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict(),
        "best_val": best_val,
        "cfg": cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


# ── Main ──────────────────────────────────────────────────────────────────
def build_config_from_args(args) -> dict:
    """Merge CLI args with optional YAML config."""
    cfg = {
        # Data
        "dataset": args.dataset,
        "data_path": args.data_path,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "target_mode": args.target_mode,
        "timemmd_root": args.timemmd_root,
        "domain": args.domain,
        # Model
        "d_model": args.d_model,
        "wavelet": args.wavelet,
        "dwt_levels": args.dwt_levels,
        "dwt_trainable": args.dwt_trainable,
        "top_k": args.top_k,
        "use_graph": args.use_graph,
        "d_state": args.d_state,
        "dropout": args.dropout,
        # Training
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.wd,
        "lambda_aux": args.lambda_aux,
        "lambda_graph": args.lambda_graph,
        "max_grad_norm": args.max_grad_norm,
        "use_amp": args.use_amp,
        "patience": args.patience,
        "log_every": args.log_every,
        "seed": args.seed,
        # Paths
        "save_dir": args.save_dir,
    }

    # Override with YAML if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            ycfg = yaml.safe_load(f)
        for k, v in ycfg.items():
            cfg[k] = v

    return cfg


def main():
    parser = argparse.ArgumentParser(description="WaveMoE Training")
    # Data
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_mode", type=str, default="all",
                        choices=["all", "last"])
    # Time-MMD
    parser.add_argument("--timemmd_root", type=str, default=None,
                        help="Path to Time-MMD repo (enables multimodal mode)")
    parser.add_argument("--domain", type=str, default="Health_US",
                        help="Time-MMD domain name")
    # Model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--dwt_levels", type=int, default=3)
    parser.add_argument("--dwt_trainable", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0,
                        help="0=soft routing, >0=sparse top-k")
    parser.add_argument("--use_graph", type=int, default=1)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lambda_aux", type=float, default=0.01)
    parser.add_argument("--lambda_graph", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    # Paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides CLI args)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    # Multi-seed
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-run, e.g. '42,123,456'")

    args = parser.parse_args()
    cfg = build_config_from_args(args)

    # Multi-seed wrapper
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        all_results = []
        for s in seeds:
            logger.info(f"\n{'='*60}\nSeed {s}\n{'='*60}")
            cfg["seed"] = s
            result = run_training(cfg, resume_path=args.resume)
            all_results.append(result)

        # Aggregate
        mses = [r["test_mse"] for r in all_results]
        maes = [r["test_mae"] for r in all_results]
        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-seed results ({len(seeds)} runs):")
        logger.info(f"  MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
        logger.info(f"  MAE: {np.mean(maes):.6f} ± {np.std(maes):.6f}")

        # Save summary
        summary_path = os.path.join(cfg["save_dir"], "multi_seed_results.json")
        os.makedirs(cfg["save_dir"], exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "seeds": seeds,
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "per_seed": all_results,
            }, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
    else:
        run_training(cfg, resume_path=args.resume)


def run_training(cfg: dict, resume_path=None) -> dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──
    if cfg.get("timemmd_root"):
        data = build_timemmd_dataloaders(
            timemmd_root=cfg["timemmd_root"],
            domain=cfg.get("domain", "Health_US"),
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 4),
        )
    else:
        data = build_dataloaders(
            dataset_name=cfg["dataset"],
            data_path=cfg.get("data_path"),
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 4),
            target_mode=cfg.get("target_mode", "all"),
        )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    # ── Model ──
    model_cfg = {
        "modality_channels": data["modality_channels"],
        "d_model": cfg["d_model"],
        "seq_len": cfg["seq_len"],
        "pred_len": cfg["pred_len"],
        "n_targets": data["n_targets"],
        "wavelet": cfg.get("wavelet", "db4"),
        "dwt_levels": cfg.get("dwt_levels", 3),
        "dwt_trainable": bool(cfg.get("dwt_trainable", True)),
        "top_k": cfg.get("top_k", 0),
        "use_graph": bool(cfg.get("use_graph", True)),
        "d_state": cfg.get("d_state", 64),
        "dropout": cfg.get("dropout", 0.1),
    }
    model = WaveMoE(model_cfg).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_train_params:,} trainable")
    logger.info(f"Modality channels: {data['modality_channels']}")
    logger.info(f"Modality groups: {list(data['modality_map_idx'].keys())}")
    logger.info(f"Targets: {data['n_targets']} channels")

    # ── Optimiser / Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", cfg.get("wd", 1e-4)),
    )
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    try:
        scaler = GradScaler("cuda", enabled=bool(cfg.get("use_amp", True)))
    except TypeError:
        scaler = GradScaler(enabled=bool(cfg.get("use_amp", True)))
    criterion = nn.MSELoss()

    # ── Resume ──
    start_epoch = 0
    best_val = float("inf")
    if resume_path:
        start_epoch, best_val = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler,
        )
        logger.info(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    # ── Save dir ──
    exp_name = f"{cfg['dataset']}_s{cfg['seq_len']}_p{cfg['pred_len']}_d{cfg['d_model']}_seed{cfg['seed']}"
    save_dir = Path(cfg.get("save_dir", "./checkpoints")) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    # ── TensorBoard (optional) ──
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(save_dir / "tb"))
    except ImportError:
        logger.info("TensorBoard not available, skipping.")

    # ── Training ──
    patience_counter = 0
    train_history = []

    logger.info(f"\nStarting training: {cfg['epochs']} epochs, save_dir={save_dir}")
    logger.info(f"Config: seq_len={cfg['seq_len']}, pred_len={cfg['pred_len']}, "
                f"d_model={cfg['d_model']}, wavelet={cfg.get('wavelet','db4')}, "
                f"levels={cfg.get('dwt_levels',3)}, graph={cfg.get('use_graph',True)}")

    t_start = time.time()
    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        t_epoch = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, cfg, device, epoch,
        )

        # Validate
        val_result = evaluate(model, val_loader, criterion, device, cfg)
        val_mse = val_result["mse"]
        val_mae = val_result["mae"]

        elapsed = time.time() - t_epoch
        logger.info(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # TensorBoard
        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_result["loss"], epoch)
            tb_writer.add_scalar("metric/val_mse", val_mse, epoch)
            tb_writer.add_scalar("metric/val_mae", val_mae, epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # History
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        })

        # Best model
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
            save_checkpoint(
                save_dir / "best.pt", model, optimizer, scheduler,
                scaler, epoch, best_val, cfg,
            )
            logger.info(f"  ★ New best val_mse={best_val:.6f}, saved.")
        else:
            patience_counter += 1

        # Latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                save_dir / "latest.pt", model, optimizer, scheduler,
                scaler, epoch, best_val, cfg,
            )

        # Early stopping
        if patience_counter >= cfg.get("patience", 10):
            logger.info(f"Early stopping at epoch {epoch} (patience={cfg['patience']})")
            break

    total_time = time.time() - t_start
    logger.info(f"\nTraining complete in {total_time/60:.1f} minutes.")

    # ── Test ──
    logger.info("\nLoading best model for testing...")
    best_path = save_dir / "best.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), model)

    test_result = evaluate(model, test_loader, criterion, device, cfg)
    logger.info(
        f"\n{'='*60}\n"
        f"TEST RESULTS ({cfg['dataset']}, seed={cfg['seed']})\n"
        f"  seq_len={cfg['seq_len']}, pred_len={cfg['pred_len']}\n"
        f"  MSE = {test_result['mse']:.6f}\n"
        f"  MAE = {test_result['mae']:.6f}\n"
        f"{'='*60}"
    )

    # Save history and results
    with open(save_dir / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=2)

    results = {
        "dataset": cfg["dataset"],
        "seq_len": cfg["seq_len"],
        "pred_len": cfg["pred_len"],
        "seed": cfg["seed"],
        "test_mse": test_result["mse"],
        "test_mae": test_result["mae"],
        "best_val_mse": best_val,
        "total_time_min": total_time / 60,
        "n_params": n_params,
    }
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save test predictions for analysis
    np.save(str(save_dir / "test_preds.npy"), test_result["preds"])
    np.save(str(save_dir / "test_trues.npy"), test_result["trues"])

    if tb_writer:
        tb_writer.add_scalar("metric/test_mse", test_result["mse"], 0)
        tb_writer.add_scalar("metric/test_mae", test_result["mae"], 0)
        tb_writer.close()

    return results


if __name__ == "__main__":
    main()