#!/bin/bash
# ============================================================
# WaveMoE - Setup & Quick Test
# One-command end-to-end verification
# ============================================================
set -e

echo "========================================"
echo "WaveMoE Setup"
echo "========================================"

if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "wavemoe"; then
        echo "Activating conda env: wavemoe"
        conda activate wavemoe
    else
        echo "Creating conda env: wavemoe"
        conda create -n wavemoe python=3.10 -y
        conda activate wavemoe
    fi
fi

echo "Installing requirements..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || \
pip install -r requirements.txt

echo "Downloading ETTh1..."
mkdir -p data
python -c "from wavemoe.data import download_dataset; download_dataset('ETTh1', './data')"
echo "Done."

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
echo "Quick test complete!"
echo "Next: python train.py --config configs/etth1.yaml --seeds 42,123,456"
