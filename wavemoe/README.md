# WaveMoE — Explainable Multimodal Time Series Forecasting in Frequency Domain

Complete implementation for proposal experiments.

## Architecture Overview

```
Per modality:  Encoder → Learnable DWT → L+1 frequency bands
Per band:      ★ Cross-Modal Band Attention (all modalities interact per band)
               → Graph Learning → Frequency-Aware Gating → Expert Dispatch
                 ├── SSM Expert    (Band 0, coarsest — long-range trends)
                 ├── GRU Expert    (Bands 1-2, mid — sequential patterns)
                 ├── Transformer   (Band L-1, fine — local attention)
                 └── TCN Expert    (Band L, finest — high-freq patterns)
Per modality:  Cross-Band Fusion (upsample + cross-attention)
Global:        Cross-Modal Fusion → Prediction Head → Forecast
```

**Key fix over original proposal:** Cross-modal attention now happens *inside* 
band-level processing (between DWT and expert routing), giving frequency-specific 
cross-modal alignment — the actual contribution that justifies "multimodal."

## Quick Start (R740)

```bash
# 1. Setup
conda activate Tunde   # or your conda env
pip install -r requirements.txt

# 2. Quick end-to-end test (< 5 min on GPU)
bash run_quick_test.sh

# 3. Full experiment on ETTh1
python train.py --config configs/etth1.yaml --seeds 42,123,456

# 4. Full experiment on Weather (download weather.csv first)
python train.py --config configs/weather.yaml --seeds 42,123,456

# 5. Run all experiments + ablations
python run_experiments.py --phase main       # main results
python run_experiments.py --phase ablation   # ablation studies
python run_experiments.py --phase collect    # gather results table

# 6. Generate figures for proposal
python generate_figures.py --results_dir ./results --output_dir ./figures
```

## Data Setup

**Auto-download:** ETTh1, ETTh2, ETTm1, ETTm2 download automatically.

**Manual download** (Weather, Exchange, ECL, Traffic):
```bash
# From Time-Series-Library:
# https://github.com/thuml/Time-Series-Library/tree/main/dataset
# Place CSV files in ./data/
mkdir -p data
# Copy weather.csv, exchange_rate.csv, etc. to ./data/
```

**Time-MMD (real multimodal — numerical + text):**
```bash
# Clone Time-MMD dataset:
git clone https://github.com/AdityaLab/Time-MMD.git ./data/timemmd

# Pre-encode text for each domain:
python encode_text.py --domain_dir ./data/timemmd/health --model bert
python encode_text.py --domain_dir ./data/timemmd/energy --model bert
```

## File Structure

```
wavemoe/
├── wavemoe/
│   ├── model.py           # Full WaveMoE architecture (all components)
│   └── data.py            # Dataset, DataLoader, normalization
├── train.py               # Training loop (AMP, early stopping, checkpoints)
├── run_experiments.py      # Batch experiment runner (main + ablation)
├── generate_figures.py     # Publication figures from results
├── encode_text.py          # Text embedding for Time-MMD
├── configs/
│   ├── etth1.yaml
│   └── weather.yaml
├── run_quick_test.sh       # One-command E2E test
└── requirements.txt
```

## Experiment Matrix for Proposal

### Main Results (Table 2 in revised proposal)
| Dataset | Horizons | Seeds | Total runs |
|---------|----------|-------|------------|
| ETTh1   | 96, 192, 336, 720 | 42, 123, 456 | 12 |
| ETTh2   | 96, 192, 336, 720 | 42, 123, 456 | 12 |
| Weather | 96, 192, 336, 720 | 42, 123, 456 | 12 |
| Exchange| 96, 192, 336, 720 | 42, 123, 456 | 12 |

### Ablation Studies (Table 3)
- A1: No Wavelet (dwt_levels=0)
- A2: No Graph Learning (use_graph=0)
- A3: No Cross-Modal Band Attention (bypass the fix)
- A4: Sparse Top-K=1 routing vs soft routing
- A5: Wavelet comparison: haar / db4 / sym8 / coif3

### Key Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model   | 128   | Latent dimension |
| dwt_levels| 3     | → 4 frequency bands |
| wavelet   | db4   | Daubechies-4 |
| d_state   | 64    | SSM state dimension |
| lr        | 1e-3  | AdamW |
| wd        | 1e-4  | Weight decay |
| scheduler | OneCycleLR | 10% warmup, cosine |
| batch_size| 32    | Per GPU |
| epochs    | 50    | Early stop patience=10 |
| AMP       | fp16  | Mixed precision |
| grad_clip | 1.0   | Max gradient norm |

## GPU Notes (P40)

- Tesla P40 = Pascal SM_61, 24 GB VRAM
- With d_model=128, batch_size=32, AMP fp16: ~6-8 GB per GPU
- Use `CUDA_VISIBLE_DEVICES=0` or `=1` for single-GPU runs
- Dual GPU: wrap in DataParallel (add `--dp` flag to train.py)
- Estimated time per experiment (50 epochs, ETTh1): ~15-25 min
- Full experiment suite: ~6-8 hours
