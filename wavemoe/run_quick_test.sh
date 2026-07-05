#!/bin/bash
# ============================================================
# WaveMoE — Setup & Quick Test
# Run on R740: ssh tundecode@<server> then execute this script
# ============================================================
set -e

echo "========================================"
echo "WaveMoE Setup"
echo "========================================"

# ── 1. Environment ──
# Activate conda env (adjust name as needed)
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "Tunde"; then
        echo "Activating conda env: Tunde"
        conda activate Tunde
    else
        echo "Creating conda env: wavemoe"
        conda create -n wavemoe python=3.10 -y
        conda activate wavemoe
    fi
fi

# ── 2. Install dependencies ──
echo "Installing requirements..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || \
pip install -r requirements.txt

# ── 3. Download ETTh1 for quick test ──
echo "Downloading ETTh1..."
mkdir -p data
python -c "from wavemoe.data import download_dataset; download_dataset('ETTh1', './data')"
echo "Done."

# ── 4. Quick end-to-end test (5 epochs, small model) ──
echo ""
echo "========================================"
echo "Quick E2E Test (5 epochs, d_model=64)"
echo "========================================"

python train.py \
    --dataset ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 64 \
    --dwt_levels 3 \
    --use_graph 1 \
    --epochs 5 \
    --batch_size 16 \
    --num_workers 2 \
    --lr 0.001 \
    --seed 42 \
    --save_dir ./checkpoints/quick_test \
    --log_every 10

echo ""
echo "========================================"
echo "Quick test complete!"
echo "If you see MSE/MAE numbers above, the pipeline works end-to-end."
echo ""
echo "Next steps:"
echo "  1. Full ETTh1 run:   python train.py --config configs/etth1.yaml --seeds 42,123,456"
echo "  2. Weather run:       python train.py --config configs/weather.yaml --seeds 42,123,456"
echo "  3. All experiments:   python run_experiments.py --phase main"
echo "  4. Ablation studies:  python run_experiments.py --phase ablation"
echo "  5. Generate figures:  python generate_figures.py --results_dir ./results"
echo "========================================"
