"""
Data pipeline for WaveMoE.
===========================
Supports:
  1. Standard benchmarks (ETTh1/h2, ETTm1/m2, Weather, Exchange, ECL, Traffic)
     — variates are grouped into synthetic "modality" clusters by physical type
  2. Time-MMD format (numerical + text columns, text encoded via BERT)
  3. Custom CSV with user-defined modality mapping
"""

import os
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ── Download URLs for standard benchmarks ─────────────────────────────────
_ETT_BASE = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
)
_DATASET_URLS = {
    "ETTh1": f"{_ETT_BASE}/ETTh1.csv",
    "ETTh2": f"{_ETT_BASE}/ETTh2.csv",
    "ETTm1": f"{_ETT_BASE}/ETTm1.csv",
    "ETTm2": f"{_ETT_BASE}/ETTm2.csv",
}

# ── Default modality groupings ────────────────────────────────────────────
# Each group becomes one "modality" in the multimodal pipeline.
MODALITY_MAPS = {
    "ETTh1": {
        "load_high": ["HUFL", "HULL"],
        "load_mid": ["MUFL", "MULL"],
        "load_low": ["LUFL", "LULL"],
        "target": ["OT"],
    },
    "ETTh2": {
        "load_high": ["HUFL", "HULL"],
        "load_mid": ["MUFL", "MULL"],
        "load_low": ["LUFL", "LULL"],
        "target": ["OT"],
    },
    "ETTm1": {
        "load_high": ["HUFL", "HULL"],
        "load_mid": ["MUFL", "MULL"],
        "load_low": ["LUFL", "LULL"],
        "target": ["OT"],
    },
    "ETTm2": {
        "load_high": ["HUFL", "HULL"],
        "load_mid": ["MUFL", "MULL"],
        "load_low": ["LUFL", "LULL"],
        "target": ["OT"],
    },
    "Weather": {
        # 21 variates grouped by physical type
        "temperature": ["T (degC)", "Tpot (K)", "Tdew (degC)"],
        "humidity": ["rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)"],
        "pressure": ["p (mbar)", "sh (g/kg)"],
        "wind": ["wv (m/s)", "max. wv (m/s)", "wd (deg)"],
        "radiation": ["rain (mm)", "raining (s)", "SWDR (W/m²)", "PAR (µmol/m²/s)"],
        "other": ["Tlog (degC)", "OT"],
    },
    "Exchange": {
        "americas": ["0", "1"],
        "europe": ["2", "3", "4"],
        "asia": ["5", "6", "7"],
    },
}


def download_dataset(name: str, data_dir: str = "./data") -> str:
    """Download a standard benchmark CSV. Returns path to CSV."""
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{name}.csv")
    if os.path.exists(out_path):
        logger.info(f"{name} already downloaded: {out_path}")
        return out_path
    url = _DATASET_URLS.get(name)
    if url is None:
        raise ValueError(
            f"No download URL for '{name}'. Available: {list(_DATASET_URLS)}\n"
            f"For Weather/Exchange/ECL/Traffic, download manually from:\n"
            f"  https://github.com/thuml/Time-Series-Library/tree/main/dataset"
        )
    logger.info(f"Downloading {name} from {url}")
    urllib.request.urlretrieve(url, out_path)
    logger.info(f"Saved to {out_path}")
    return out_path


# ── Normalization ─────────────────────────────────────────────────────────
class StandardScaler:
    """Per-channel z-score normalization fitted on training set."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """data: (N, C)"""
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.std = d["std"]


# ── Dataset ───────────────────────────────────────────────────────────────
class MultimodalTSDataset(Dataset):
    """
    Sliding-window dataset that returns modality-grouped inputs and targets.

    Args:
        data_array:  (N, C_total) numpy array, all channels
        modality_map: dict[str, list[int]]  — group_name → list of column indices
        target_cols:  list[int]             — column indices for prediction target
        seq_len:      int                   — input lookback window
        pred_len:     int                   — forecast horizon
        stride:       int                   — sliding window stride
    """

    def __init__(
        self,
        data_array: np.ndarray,
        modality_map: dict[str, list[int]],
        target_cols: list[int],
        seq_len: int = 96,
        pred_len: int = 96,
        stride: int = 1,
    ):
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.modality_map = modality_map
        self.modality_names = list(modality_map.keys())
        self.modality_indices = [modality_map[k] for k in self.modality_names]
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.n_samples = max(
            0, (len(data_array) - seq_len - pred_len) // stride + 1
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end_x = start + self.seq_len
        end_y = end_x + self.pred_len

        # Modality inputs: list of tensors
        x_mods = [
            self.data[start:end_x, cols] for cols in self.modality_indices
        ]
        # Target
        y = self.data[end_x:end_y, self.target_cols]

        return x_mods, y

    @property
    def modality_channels(self) -> list[int]:
        return [len(cols) for cols in self.modality_indices]

    @property
    def n_targets(self) -> int:
        return len(self.target_cols)


def collate_multimodal(batch):
    """Custom collate: stack each modality separately."""
    x_mods_list, y_list = zip(*batch)
    n_mods = len(x_mods_list[0])
    x_mods = [
        torch.stack([sample[m] for sample in x_mods_list])
        for m in range(n_mods)
    ]
    y = torch.stack(y_list)
    return x_mods, y


# ── Builder ───────────────────────────────────────────────────────────────
def build_dataloaders(
    dataset_name: str,
    data_path: Optional[str] = None,
    modality_map: Optional[dict] = None,
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    target_mode: str = "all",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict:
    """
    Build train/val/test DataLoaders.

    Args:
        dataset_name: name key for predefined datasets, or 'custom'
        data_path:    path to CSV file (required for 'custom', optional for known)
        modality_map: dict[str, list[str]] col-name grouping (auto for known datasets)
        seq_len:      input window length
        pred_len:     forecast horizon
        batch_size:   batch size
        target_mode:  'all' (predict all channels) or 'last' (last column only)
        train_ratio:  fraction for training
        val_ratio:    fraction for validation

    Returns:
        dict with keys: train_loader, val_loader, test_loader, scaler,
                        modality_channels, n_targets, modality_map_idx
    """
    # ── Load CSV ──
    if data_path is None:
        data_path = download_dataset(dataset_name)
    df = pd.read_csv(data_path)

    # Drop date column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    elif df.columns[0].lower() in ("date", "datetime", "time", "timestamp"):
        df = df.drop(columns=[df.columns[0]])

    col_names = list(df.columns)
    data = df.values.astype(np.float32)
    N, C = data.shape
    logger.info(f"Loaded {dataset_name}: {N} rows, {C} columns: {col_names[:10]}...")

    # ── Modality mapping → column indices ──
    if modality_map is None:
        modality_map = MODALITY_MAPS.get(dataset_name)

    if modality_map is not None:
        # Convert column names to indices
        mod_idx = {}
        for group, cols in modality_map.items():
            indices = []
            for c in cols:
                if isinstance(c, int):
                    indices.append(c)
                elif c in col_names:
                    indices.append(col_names.index(c))
                else:
                    logger.warning(f"Column '{c}' not found in {dataset_name}, skipping")
            if indices:
                mod_idx[group] = indices
    else:
        # Auto-split into roughly equal groups of 2-4 modalities
        n_groups = max(2, C // 3)
        n_groups = min(n_groups, 8)
        per_group = C // n_groups
        mod_idx = {}
        for g in range(n_groups):
            start_c = g * per_group
            end_c = start_c + per_group if g < n_groups - 1 else C
            mod_idx[f"group_{g}"] = list(range(start_c, end_c))

    # ── Target columns ──
    if target_mode == "all":
        # All unique columns across all modalities
        all_cols = sorted(set(c for cols in mod_idx.values() for c in cols))
        target_cols = all_cols
    elif target_mode == "last":
        target_cols = [C - 1]
    else:
        target_cols = list(range(C))

    # ── Train / val / test split ──
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_data = data[:n_train]
    val_data = data[n_train: n_train + n_val]
    test_data = data[n_train + n_val:]

    # ── Normalise ──
    scaler = StandardScaler().fit(train_data)
    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)

    # ── Datasets ──
    ds_train = MultimodalTSDataset(
        train_norm, mod_idx, target_cols, seq_len, pred_len, stride=1,
    )
    ds_val = MultimodalTSDataset(
        val_norm, mod_idx, target_cols, seq_len, pred_len, stride=1,
    )
    ds_test = MultimodalTSDataset(
        test_norm, mod_idx, target_cols, seq_len, pred_len, stride=1,
    )

    logger.info(
        f"Splits: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}"
    )
    logger.info(f"Modalities: {list(mod_idx.keys())}")
    logger.info(f"Channels per modality: {ds_train.modality_channels}")
    logger.info(f"Targets: {len(target_cols)} columns")

    # ── DataLoaders ──
    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_multimodal, drop_last=True,
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_multimodal, drop_last=False,
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_multimodal, drop_last=False,
    )

    return {
        "train_loader": dl_train,
        "val_loader": dl_val,
        "test_loader": dl_test,
        "scaler": scaler,
        "modality_channels": ds_train.modality_channels,
        "n_targets": ds_train.n_targets,
        "modality_map_idx": mod_idx,
        "col_names": col_names,
    }


# ── Time-MMD loader (text + numerical) ────────────────────────────────────

def _load_timemmd_numerical(domain_dir: str, domain_name: str):
    """
    Load and clean Time-MMD numerical CSV.
    Handles: 'X' missing values, string columns, date columns.
    Returns: (cleaned_df, numeric_col_names, target_col_name)
    """
    domain_dir = Path(domain_dir)
    csv_path = domain_dir / f"{domain_name}.csv"
    if not csv_path.exists():
        # Try finding any CSV
        csvs = list(domain_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {domain_dir}")
        csv_path = csvs[0]

    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {csv_path.name}: {len(df)} rows, {len(df.columns)} cols")

    # Drop date/string columns
    drop_cols = []
    for col in df.columns:
        if col.lower() in ("date", "start_date", "end_date", "year_week",
                           "region type", "region", "year", "week"):
            drop_cols.append(col)
    df = df.drop(columns=drop_cols, errors="ignore")

    # Replace 'X' and non-numeric with NaN, then forward-fill
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill().bfill().fillna(0)

    # Target = 'OT' column if present, else last column
    target_col = "OT" if "OT" in df.columns else df.columns[-1]
    numeric_cols = list(df.columns)
    logger.info(f"  Numeric columns: {len(numeric_cols)}, target: {target_col}")
    logger.info(f"  Columns: {numeric_cols[:8]}{'...' if len(numeric_cols) > 8 else ''}")

    return df, numeric_cols, target_col


def _load_timemmd_text(domain_dir: str, domain_name: str):
    """
    Load Time-MMD textual data (report + search).
    Returns: dict mapping start_date → concatenated text string
    """
    domain_dir = Path(domain_dir)
    texts_by_date = {}

    for suffix in ["_report.csv", "_search.csv"]:
        csv_path = domain_dir / f"{domain_name}{suffix}"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"  Could not read {csv_path.name}: {e}")
            continue

        for _, row in df.iterrows():
            date_key = str(row.get("start_date", ""))
            fact = str(row.get("fact", ""))
            preds = str(row.get("preds", ""))

            # Skip NA entries
            if fact.startswith("NA") or fact == "nan":
                fact = ""
            if preds.startswith("NA") or preds == "nan":
                preds = ""

            combined = f"{fact} {preds}".strip()
            if combined and date_key:
                if date_key in texts_by_date:
                    texts_by_date[date_key] += " " + combined
                else:
                    texts_by_date[date_key] = combined

    n_non_empty = sum(1 for v in texts_by_date.values() if v)
    logger.info(f"  Text entries: {len(texts_by_date)} dates, {n_non_empty} non-empty")
    return texts_by_date


class TimeMMDDataset(Dataset):
    """
    Time-MMD dataset: numerical time series + aligned text embeddings.

    Actual repo structure:
        numerical/<Domain>/<Domain>.csv
            columns: date, start_date, end_date, ..., OT, feature1, feature2, ...
            'X' values = missing, mixed string/numeric columns
        textual/<Domain>/<Domain>_report.csv
            columns: index, start_date, end_date, fact, preds
        textual/<Domain>/<Domain>_search.csv
            same format as report

    Two modalities:
        - Modality 0: numerical features (after cleaning)
        - Modality 1: text embeddings (pre-encoded via encode_text.py, or zeros)
    """

    def __init__(
        self,
        data_array: np.ndarray,
        text_array: np.ndarray,
        target_idx: int,
        n_numeric: int,
        seq_len: int = 96,
        pred_len: int = 96,
    ):
        self.data_num = torch.tensor(data_array, dtype=torch.float32)
        self.data_txt = torch.tensor(text_array, dtype=torch.float32)
        self.target_idx = target_idx
        self.n_numeric = n_numeric
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = max(0, len(data_array) - seq_len - pred_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = idx
        e_x = s + self.seq_len
        e_y = e_x + self.pred_len

        x_num = self.data_num[s:e_x]       # (T, C_num)
        x_txt = self.data_txt[s:e_x]       # (T, D_text)
        y = self.data_num[e_x:e_y, self.target_idx:self.target_idx + 1]  # (H, 1)
        return [x_num, x_txt], y

    @property
    def modality_channels(self):
        return [self.n_numeric, self.data_txt.shape[1]]

    @property
    def n_targets(self):
        return 1


def build_timemmd_dataloaders(
    timemmd_root: str,
    domain: str = "Health_US",
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    text_embed_dim: int = 768,
) -> dict:
    """
    Build train/val/test DataLoaders for a Time-MMD domain.

    Args:
        timemmd_root: path to cloned Time-MMD repo (e.g. './data/timemmd/repo')
        domain:       domain name (Health_US, Energy, Economy, etc.)
        text_embed_dim: dimension of pre-encoded text embeddings
    """
    root = Path(timemmd_root)
    num_dir = root / "numerical" / domain
    txt_dir = root / "textual" / domain

    logger.info(f"\n=== Loading Time-MMD: {domain} ===")

    # ── Load numerical (keeping dates for alignment) ──
    num_csv = num_dir / f"{domain}.csv"
    if not num_csv.exists():
        csvs = list(num_dir.glob("*.csv"))
        num_csv = csvs[0] if csvs else num_csv
    raw_df = pd.read_csv(num_csv)

    # Extract dates for alignment before dropping
    num_dates = None
    for date_col in ["start_date", "date"]:
        if date_col in raw_df.columns:
            num_dates = raw_df[date_col].astype(str).tolist()
            break

    num_df, num_cols, target_col = _load_timemmd_numerical(str(num_dir), domain)
    num_data = num_df.values.astype(np.float32)
    N, C = num_data.shape
    target_idx = num_cols.index(target_col) if target_col in num_cols else C - 1

    # ── Load text embeddings with date alignment ──
    embed_path = num_dir / "text_embeddings.npy"
    if not embed_path.exists():
        embed_path = txt_dir / "text_embeddings.npy"
    if not embed_path.exists():
        embed_path = root / f"{domain}_text_embeddings.npy"

    if embed_path.exists() and num_dates is not None:
        raw_embeds = np.load(str(embed_path)).astype(np.float32)
        embed_dim = raw_embeds.shape[1]
        logger.info(f"  Raw text embeddings: {raw_embeds.shape} from {embed_path}")

        # Load text CSV dates for alignment
        text_dates = []
        for suffix in ["_report.csv", "_search.csv"]:
            txt_csv = txt_dir / f"{domain}{suffix}"
            if txt_csv.exists():
                try:
                    tdf = pd.read_csv(txt_csv)
                    if "start_date" in tdf.columns:
                        text_dates = tdf["start_date"].astype(str).tolist()
                        break
                except Exception:
                    pass

        # Date-based alignment: create full N×D array, fill where dates match
        text_embeds = np.zeros((N, embed_dim), dtype=np.float32)
        if text_dates and len(text_dates) == len(raw_embeds):
            num_date_to_idx = {d: i for i, d in enumerate(num_dates)}
            matched = 0
            for t_idx, t_date in enumerate(text_dates):
                if t_date in num_date_to_idx and t_idx < len(raw_embeds):
                    n_idx = num_date_to_idx[t_date]
                    if n_idx < N:
                        text_embeds[n_idx] = raw_embeds[t_idx]
                        matched += 1
            logger.info(
                f"  Date alignment: {matched}/{len(text_dates)} text entries matched "
                f"to {N} numerical rows ({matched/N*100:.1f}% coverage)"
            )
        else:
            # Fallback: can't align, use zero vectors
            logger.warning(
                f"  Could not align text dates ({len(text_dates)}) "
                f"with numerical ({N}). Using zero vectors."
            )
    elif embed_path.exists():
        # No dates available, truncate (fallback)
        raw_embeds = np.load(str(embed_path)).astype(np.float32)
        embed_dim = raw_embeds.shape[1]
        min_len = min(N, len(raw_embeds))
        text_embeds = np.zeros((N, embed_dim), dtype=np.float32)
        text_embeds[:min_len] = raw_embeds[:min_len]
        logger.warning(f"  No date columns for alignment, used first {min_len} embeddings")
    else:
        text_embeds = np.zeros((N, text_embed_dim), dtype=np.float32)
        logger.warning(
            f"  No text embeddings found. Using zeros ({N} × {text_embed_dim}).\n"
            f"  To encode text, run:\n"
            f"    python encode_text.py --domain_dir {txt_dir} --model bert\n"
            f"  Then copy .npy to {num_dir}/text_embeddings.npy"
        )

    # ── Split ──
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    # ── Normalise numerical (fit on train) ──
    scaler = StandardScaler().fit(num_data[:n_train])
    num_norm = scaler.transform(num_data)

    # ── Build datasets ──
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, N),
    }

    datasets = {}
    for name, (s, e) in splits.items():
        datasets[name] = TimeMMDDataset(
            num_norm[s:e], text_embeds[s:e], target_idx,
            n_numeric=C, seq_len=seq_len, pred_len=pred_len,
        )

    for name, ds in datasets.items():
        logger.info(f"  {name}: {len(ds)} samples")

    logger.info(f"  Modality channels: {datasets['train'].modality_channels}")
    logger.info(f"  Target: {target_col} (idx={target_idx})")

    # ── DataLoaders ──
    loaders = {}
    for name, ds in datasets.items():
        loaders[f"{name}_loader"] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_multimodal,
            drop_last=(name == "train"),
        )

    return {
        **loaders,
        "scaler": scaler,
        "modality_channels": datasets["train"].modality_channels,
        "n_targets": datasets["train"].n_targets,
        "modality_map_idx": {"numerical": list(range(C)), "text": [0]},
        "col_names": num_cols,
        "domain": domain,
    }