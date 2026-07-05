#!/usr/bin/env python3
"""
WaveMoE Explainability & Verification Suite
=============================================
  1. Expert routing heatmap        (from TRAINED model)
  2. Cross-modal band attention    (text <-> numerical coupling per band)
  3. Verification: perturbation    (zero/shuffle a modality -> MSE must rise)
  4. Verification: synthetic probe (injected freq must localise to right band)
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavemoe.model import WaveMoE
from wavemoe.data import build_timemmd_dataloaders, build_dataloaders

plt.rcParams.update({"font.family":"serif","font.size":10,"figure.dpi":300,
    "savefig.dpi":300,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.25,
    "axes.spines.top":False,"axes.spines.right":False})

def load_model_and_data(args, device):
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    with open(ckpt_path.parent / "model_config.json") as f:
        model_cfg = json.load(f)
    model = WaveMoE(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch','?')})")
    if args.timemmd_root:
        data = build_timemmd_dataloaders(timemmd_root=args.timemmd_root,
            domain=args.domain, seq_len=args.seq_len, pred_len=args.pred_len,
            batch_size=args.batch_size, num_workers=0)
    else:
        data = build_dataloaders(dataset_name=args.dataset, seq_len=args.seq_len,
            pred_len=args.pred_len, batch_size=args.batch_size, num_workers=0)
    return model, model_cfg, data

@torch.no_grad()
def eval_mse(model, loader, device, transform_mods=None, max_batches=None):
    criterion = nn.MSELoss(); total, n = 0.0, 0
    for i, (x_mods, y) in enumerate(loader):
        x_mods = [x.to(device) for x in x_mods]; y = y.to(device)
        if transform_mods is not None: x_mods = transform_mods(x_mods)
        total += criterion(model(x_mods), y).item(); n += 1
        if max_batches and i + 1 >= max_batches: break
    return total / max(n, 1)

@torch.no_grad()
def extract_routing_weights(model, loader, device, max_batches=20):
    all_w = []
    for i, (x_mods, _) in enumerate(loader):
        x_mods = [x.to(device) for x in x_mods]
        encoded = [enc(x) for enc, x in zip(model.encoders, x_mods)]
        all_bands = [model.dwt(e) for e in encoded]
        n_bands = len(all_bands[0])
        cross = [[] for _ in range(model.n_modalities)]
        for l in range(n_bands):
            reps = [all_bands[m][l] for m in range(model.n_modalities)]
            enh, _ = model.band_cross_modal[l](reps)
            for m in range(model.n_modalities): cross[m].append(enh[m])
        if model.use_graph:
            for m in range(model.n_modalities):
                for l in range(n_bands):
                    g, _ = model.graph_layers[l](cross[m][l]); cross[m][l] = g
        w, _ = model.gating(cross[0]); all_w.append(w.cpu())
        if i + 1 >= max_batches: break
    return torch.cat(all_w, 0).mean(0).numpy()

def plot_routing(W, out):
    nb, ne = W.shape
    en = ["SSM","GRU","Transformer","TCN"][:ne]
    bn = [f"Band {i}" for i in range(nb)]
    fig, ax = plt.subplots(figsize=(5,3.2))
    im = ax.imshow(W, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.5,W.max()))
    ax.set_xticks(range(ne)); ax.set_xticklabels(en)
    ax.set_yticks(range(nb)); ax.set_yticklabels(bn)
    ax.set_xlabel("Expert"); ax.set_ylabel("Frequency Band")
    for i in range(nb):
        for j in range(ne):
            c = "white" if W[i,j] > 0.5*W.max() else "black"
            ax.text(j,i,f"{W[i,j]:.2f}",ha="center",va="center",fontsize=9,fontweight="bold",color=c)
    plt.colorbar(im,ax=ax,shrink=0.8,label="Routing Weight (trained)")
    plt.tight_layout(); plt.savefig(out); plt.close(); print(f"  Saved: {out}")

@torch.no_grad()
def extract_cm_attn(model, loader, device, max_batches=20):
    nb = model.cfg["dwt_levels"] + 1; M = model.n_modalities
    acc = [np.zeros((M,M)) for _ in range(nb)]; count = 0
    for i, (x_mods, _) in enumerate(loader):
        x_mods = [x.to(device) for x in x_mods]; _ = model(x_mods)
        attn = model.get_interpretability()["band_cross_modal_attn"]
        for l, w in enumerate(attn):
            if w is not None: acc[l] += w.mean(dim=(0,1)).cpu().numpy()
        count += 1
        if i + 1 >= max_batches: break
    return [a / max(count,1) for a in acc]

def plot_cm_attn(attns, names, out):
    nb = len(attns)
    fig, axes = plt.subplots(1, nb, figsize=(3.2*nb, 3))
    if nb == 1: axes = [axes]
    for l, (ax, A) in enumerate(zip(axes, attns)):
        ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"Band {l}", fontsize=9)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                c = "white" if A[i,j] > 0.5 else "black"
                ax.text(j,i,f"{A[i,j]:.2f}",ha="center",va="center",fontsize=8,color=c)
    fig.suptitle("Cross-Modal Attention per Band", fontsize=10, y=1.05)
    plt.tight_layout(); plt.savefig(out); plt.close(); print(f"  Saved: {out}")

def perturbation_test(model, loader, device, names):
    res = {"baseline": eval_mse(model, loader, device)}
    for m in range(len(names)):
        def zero_m(x, m=m):
            o = [t.clone() for t in x]; o[m] = torch.zeros_like(o[m]); return o
        def shuf_m(x, m=m):
            o = [t.clone() for t in x]
            o[m] = o[m][torch.randperm(o[m].size(0), device=o[m].device)]; return o
        res[f"zero_{names[m]}"] = eval_mse(model, loader, device, zero_m)
        res[f"shuffle_{names[m]}"] = eval_mse(model, loader, device, shuf_m)
    return res

def plot_pert(res, out):
    labels = list(res.keys()); vals = [res[k] for k in labels]
    colors = ["#4472C4"] + ["#E06666" if "zero" in k else "#ED7D31" for k in labels[1:]]
    fig, ax = plt.subplots(figsize=(6.5,3.2))
    bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_","\n") for l in labels], fontsize=8)
    ax.set_ylabel("Test MSE")
    ax.axhline(y=vals[0], color="gray", linestyle=":", alpha=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout(); plt.savefig(out); plt.close(); print(f"  Saved: {out}")

@torch.no_grad()
def synthetic_probe(model, cfg, device):
    T = cfg["seq_len"]; mc = cfg["modality_channels"]; nb = cfg["dwt_levels"] + 1
    freqs = {"low (period=T)": 1.0/T, "mid (period=T/4)": 4.0/T, "high (period=T/16)": 16.0/T}
    table = {}; t = torch.arange(T, dtype=torch.float32)
    for name, f in freqs.items():
        sig = torch.sin(2*np.pi*f*t)
        x = sig.unsqueeze(0).unsqueeze(-1).expand(1, T, mc[0]).to(device)
        bands = model.dwt(model.encoders[0](x))
        e = np.array([b.pow(2).mean().item() for b in bands])
        table[name] = e / (e.sum() + 1e-9)
    return table, nb

def plot_probe(table, nb, out):
    fig, ax = plt.subplots(figsize=(6,3.2))
    x = np.arange(nb); w = 0.25
    colors = ["#4472C4","#ED7D31","#E06666"]
    for i, (name, e) in enumerate(table.items()):
        ax.bar(x+(i-1)*w, e, w, label=name, color=colors[i%3], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Band {i}" for i in range(nb)], fontsize=8)
    ax.set_ylabel("Normalised Band Energy")
    ax.legend(fontsize=8, title="Injected sinusoid", title_fontsize=8)
    plt.tight_layout(); plt.savefig(out); plt.close(); print(f"  Saved: {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--timemmd_root", default=None)
    p.add_argument("--domain", default="Energy")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--seq_len", type=int, default=52)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", default="./explain_out")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg, data = load_model_and_data(args, device)
    loader = data["test_loader"]; names = list(data["modality_map_idx"].keys())
    print(f"Modalities: {names}\n")

    print("1. Expert routing (trained)...")
    W = extract_routing_weights(model, loader, device)
    plot_routing(W, os.path.join(args.output_dir, "routing_heatmap_trained.png"))
    np.save(os.path.join(args.output_dir, "routing_weights.npy"), W)

    print("\n2. Cross-modal band attention...")
    attn = extract_cm_attn(model, loader, device)
    plot_cm_attn(attn, names, os.path.join(args.output_dir, "cross_modal_attention.png"))

    print("\n3. Perturbation verification...")
    pert = perturbation_test(model, loader, device, names)
    plot_pert(pert, os.path.join(args.output_dir, "perturbation_test.png"))
    for k, v in pert.items():
        d = (v - pert["baseline"]) / pert["baseline"] * 100
        print(f"    {k:<25} MSE={v:.4f}  ({d:+.1f}%)")

    print("\n4. Synthetic frequency probe...")
    table, nb = synthetic_probe(model, cfg, device)
    plot_probe(table, nb, os.path.join(args.output_dir, "synthetic_probe.png"))
    for name, e in table.items():
        print(f"    {name:<22} {np.array2string(e, precision=2)}")

    with open(os.path.join(args.output_dir, "explainability_summary.json"), "w") as f:
        json.dump({"routing_weights": W.tolist(), "perturbation_mse": pert,
                   "synthetic_probe": {k: v.tolist() for k, v in table.items()}}, f, indent=2)
    print(f"\nAll outputs in {args.output_dir}/")

if __name__ == "__main__":
    main()
