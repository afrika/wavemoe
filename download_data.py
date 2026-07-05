#!/usr/bin/env python3
"""
Download all datasets needed for WaveMoE experiments.
======================================================
  1. ETTh1/h2, ETTm1/m2 — auto-download from GitHub
  2. Weather — from thuml Time-Series-Library
  3. Time-MMD — real multimodal (numerical + text) from AdityaLab

Usage:
  python download_data.py --all
  python download_data.py --ett --weather
  python download_data.py --timemmd
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "./data"

# ── ETT datasets ──────────────────────────────────────────────────────────
ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
ETT_FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]

# ── Weather / Exchange (from thuml TSLib) ─────────────────────────────────
TSLIB_BASE = "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset"
TSLIB_FILES = {
    "weather.csv": f"{TSLIB_BASE}/weather/weather.csv",
    "exchange_rate.csv": f"{TSLIB_BASE}/exchange_rate/exchange_rate.csv",
}

# ── Time-MMD (multimodal) ────────────────────────────────────────────────
TIMEMMD_REPO = "https://github.com/AdityaLab/Time-MMD.git"
TIMEMMD_HF = "https://huggingface.co/datasets/Maple728/Time-MMD"


def download_file(url: str, dest: str):
    """Download a file if it doesn't exist."""
    if os.path.exists(dest):
        logger.info(f"  Already exists: {dest}")
        return True
    try:
        logger.info(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        logger.info(f"  Saved: {dest}")
        return True
    except Exception as e:
        logger.error(f"  Failed to download {url}: {e}")
        return False


def download_ett():
    """Download ETT datasets."""
    logger.info("\n=== Downloading ETT datasets ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname in ETT_FILES:
        url = f"{ETT_BASE}/{fname}"
        dest = os.path.join(DATA_DIR, fname)
        download_file(url, dest)


def download_weather():
    """Download Weather and Exchange datasets."""
    logger.info("\n=== Downloading Weather & Exchange ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, url in TSLIB_FILES.items():
        dest = os.path.join(DATA_DIR, fname)
        if not download_file(url, dest):
            logger.warning(
                f"\n  Direct download failed for {fname}."
                f"\n  This may be due to network restrictions (GFW)."
                f"\n  Manual download options:"
                f"\n    1. From GitHub: https://github.com/thuml/Time-Series-Library"
                f"\n       Navigate to dataset/{fname.replace('.csv','')}/"
                f"\n    2. Via proxy/VPN"
                f"\n    3. Copy from another machine: scp weather.csv {DATA_DIR}/"
            )


def download_timemmd():
    """Download Time-MMD multimodal dataset."""
    logger.info("\n=== Downloading Time-MMD (multimodal) ===")
    timemmd_dir = os.path.join(DATA_DIR, "timemmd")
    os.makedirs(timemmd_dir, exist_ok=True)

    # Try git clone first
    logger.info(f"  Cloning Time-MMD repo...")
    ret = os.system(f"git clone --depth 1 {TIMEMMD_REPO} {timemmd_dir}/repo 2>&1")
    if ret == 0:
        logger.info(f"  Cloned to {timemmd_dir}/repo")
        logger.info(f"\n  Time-MMD has 9 domains: agriculture, climate, economy,")
        logger.info(f"  energy, environment, health, social, traffic, weather_timemmd")
        logger.info(f"\n  Data files are in the repo. For each domain you want to use:")
        logger.info(f"    1. Find the numerical CSV and text CSV in the domain folder")
        logger.info(f"    2. Run: python encode_text.py --domain_dir {timemmd_dir}/repo/<domain> --model bert")
    else:
        logger.warning(
            f"\n  Git clone failed (possibly network restricted)."
            f"\n  Alternative: Download Time-MMD from HuggingFace:"
            f"\n    {TIMEMMD_HF}"
            f"\n"
            f"\n  If HuggingFace is also blocked, you can use a mirror:"
            f"\n    pip install huggingface_hub"
            f"\n    huggingface-cli download Maple728/Time-MMD --local-dir {timemmd_dir}"
            f"\n"
            f"\n  Or use hf-mirror.com (accessible in China):"
            f"\n    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Maple728/Time-MMD --local-dir {timemmd_dir}"
        )

    # Also try MM-TSFlib (the benchmark code)
    mmtsf_dir = os.path.join(timemmd_dir, "MM-TSFlib")
    logger.info(f"\n  Cloning MM-TSFlib benchmark code...")
    ret2 = os.system(f"git clone --depth 1 https://github.com/AdityaLab/MM-TSFlib.git {mmtsf_dir} 2>&1")
    if ret2 == 0:
        logger.info(f"  Cloned MM-TSFlib to {mmtsf_dir}")
        logger.info(f"  This contains baseline implementations (iTransformer, PatchTST, etc.)")
    else:
        logger.warning(f"  MM-TSFlib clone failed. Download manually from:")
        logger.warning(f"    https://github.com/AdityaLab/MM-TSFlib")


def verify_datasets():
    """Check what's available."""
    logger.info("\n=== Dataset Status ===")
    checks = {
        "ETTh1": os.path.join(DATA_DIR, "ETTh1.csv"),
        "ETTh2": os.path.join(DATA_DIR, "ETTh2.csv"),
        "ETTm1": os.path.join(DATA_DIR, "ETTm1.csv"),
        "ETTm2": os.path.join(DATA_DIR, "ETTm2.csv"),
        "Weather": os.path.join(DATA_DIR, "weather.csv"),
        "Exchange": os.path.join(DATA_DIR, "exchange_rate.csv"),
        "Time-MMD": os.path.join(DATA_DIR, "timemmd"),
    }
    for name, path in checks.items():
        exists = os.path.exists(path)
        if exists and os.path.isfile(path):
            size = os.path.getsize(path) / 1024
            status = f"✓  ({size:.0f} KB)"
        elif exists and os.path.isdir(path):
            status = "✓  (directory)"
        else:
            status = "✗  MISSING"
        logger.info(f"  {name:<12} {status}")


def main():
    parser = argparse.ArgumentParser(description="Download WaveMoE datasets")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--ett", action="store_true", help="Download ETT datasets")
    parser.add_argument("--weather", action="store_true", help="Download Weather/Exchange")
    parser.add_argument("--timemmd", action="store_true", help="Download Time-MMD")
    parser.add_argument("--verify", action="store_true", help="Check what's available")
    args = parser.parse_args()

    if not any([args.all, args.ett, args.weather, args.timemmd, args.verify]):
        parser.print_help()
        print("\nExample: python download_data.py --all")
        return

    if args.ett or args.all:
        download_ett()
    if args.weather or args.all:
        download_weather()
    if args.timemmd or args.all:
        download_timemmd()

    # Always verify at the end
    verify_datasets()


if __name__ == "__main__":
    main()