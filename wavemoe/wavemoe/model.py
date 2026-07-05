"""
WaveMoE: Explainable Multimodal Time Series Forecasting in Frequency Domain
=============================================================================
Full architecture with frequency-level cross-modal attention.

Key fix over original proposal: cross-modal attention happens INSIDE band-level
processing (between DWT and expert routing), not just at the very end. This gives
frequency-specific cross-modal alignment — the actual selling point.

Architecture flow:
  Per modality: Encode → DWT → bands
  Per band: CrossModalBandAttention(band_l across all modalities)  ← THE FIX
  Per band per modality: GraphLearning → FreqGating → Expert
  Per modality: CrossBandFusion
  Global: CrossModalFusion → PredictionHead → Forecast
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Wavelet filter coefficients (no pywt dependency) ────────────────────────
WAVELET_FILTERS = {
    "haar": {
        "dec_lo": [0.7071067811865476, 0.7071067811865476],
        "dec_hi": [-0.7071067811865476, 0.7071067811865476],
    },
    "db4": {
        "dec_lo": [
            -0.010597401784997278, 0.032883011666982945,
            0.030841381835986965, -0.18703481171888114,
            -0.02798376941698385, 0.6308807679295904,
            0.7148465705525415, 0.23037781330885523,
        ],
        "dec_hi": [
            -0.23037781330885523, 0.7148465705525415,
            -0.6308807679295904, -0.02798376941698385,
            0.18703481171888114, 0.030841381835986965,
            -0.032883011666982945, -0.010597401784997278,
        ],
    },
    "sym8": {
        "dec_lo": [
            -0.0033824159510061256, -0.0005421323317911481,
            0.03169508781149298, 0.007607487324917605,
            -0.1432942383508097, -0.061273359067658524,
            0.4813596512583722, 0.7771857517005235,
            0.3644418948353314, -0.05194583810770904,
            -0.027219029917056003, 0.049137179673607506,
            0.003808752013890615, -0.01495225833704823,
            -0.0003029205147213668, 0.0018899503327594609,
        ],
        "dec_hi": [
            -0.0018899503327594609, -0.0003029205147213668,
            0.01495225833704823, 0.003808752013890615,
            -0.049137179673607506, -0.027219029917056003,
            0.05194583810770904, 0.3644418948353314,
            -0.7771857517005235, 0.4813596512583722,
            0.061273359067658524, -0.1432942383508097,
            -0.007607487324917605, 0.03169508781149298,
            0.0005421323317911481, -0.0033824159510061256,
        ],
    },
    "coif3": {
        "dec_lo": [
            -0.0034968250946298457, -0.011131187735279018,
            0.04757245554614399, 0.027321011594512505,
            -0.08230192478013955, -0.07173556556806395,
            0.22414386804201338, 0.8272820370578301,
            0.42970917175182404, -0.05827731595289542,
            -0.059434418646466836, 0.07612029282550975,
            0.01685856930708416, -0.023423788461710085,
            -0.001117518771401685, 0.006029128638487337,
            -0.0001797530818700529, -0.0006545396855498098,
        ],
        "dec_hi": [
            0.0006545396855498098, -0.0001797530818700529,
            -0.006029128638487337, -0.001117518771401685,
            0.023423788461710085, 0.01685856930708416,
            -0.07612029282550975, -0.059434418646466836,
            0.05827731595289542, 0.42970917175182404,
            -0.8272820370578301, 0.22414386804201338,
            0.07173556556806395, -0.08230192478013955,
            -0.027321011594512505, 0.04757245554614399,
            0.011131187735279018, -0.0034968250946298457,
        ],
    },
}


def get_wavelet_filters(name: str):
    """Return (dec_lo, dec_hi) tensors for a named wavelet."""
    try:
        import pywt
        w = pywt.Wavelet(name)
        return (
            torch.tensor(w.dec_lo, dtype=torch.float32),
            torch.tensor(w.dec_hi, dtype=torch.float32),
        )
    except ImportError:
        pass
    if name not in WAVELET_FILTERS:
        raise ValueError(f"Unknown wavelet '{name}'. Available: {list(WAVELET_FILTERS)}")
    f = WAVELET_FILTERS[name]
    return (
        torch.tensor(f["dec_lo"], dtype=torch.float32),
        torch.tensor(f["dec_hi"], dtype=torch.float32),
    )


# ════════════════════════════════════════════════════════════════════════════
# 1. Modality Encoder
# ════════════════════════════════════════════════════════════════════════════
class ModalityEncoder(nn.Module):
    """
    Projects raw modality features into shared latent space.
    (B, T, C_m) → Conv1d(k=3) → Linear → LayerNorm → Dropout → (B, T, d_model)
    """

    def __init__(self, in_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_m)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T, d_model)
        h = self.linear(h)
        h = self.norm(h)
        h = self.dropout(h)
        return h


# ════════════════════════════════════════════════════════════════════════════
# 2. Learnable Discrete Wavelet Transform
# ════════════════════════════════════════════════════════════════════════════
class LearnableDWT(nn.Module):
    """
    Multi-level DWT via grouped depthwise convolutions.
    Filters initialised from a named wavelet and optionally fine-tuned.

    Input:  (B, T, d_model)
    Output: list of L+1 bands [band_0 … band_L], coarsest to finest.
            band_0 = final approximation  (B, T//2^L, d_model)
            band_l = detail at level L-l+1 (B, T//2^(L-l+1), d_model)  for l≥1
    """

    def __init__(
        self,
        d_model: int,
        wavelet: str = "db4",
        levels: int = 3,
        trainable: bool = True,
    ):
        super().__init__()
        self.levels = levels
        dec_lo, dec_hi = get_wavelet_filters(wavelet)
        filt_len = len(dec_lo)

        self.low_pass = nn.ModuleList()
        self.high_pass = nn.ModuleList()
        for _ in range(levels):
            lo = nn.Conv1d(
                d_model, d_model, filt_len,
                stride=2, padding=(filt_len - 1) // 2,
                groups=d_model, bias=False,
            )
            hi = nn.Conv1d(
                d_model, d_model, filt_len,
                stride=2, padding=(filt_len - 1) // 2,
                groups=d_model, bias=False,
            )
            with torch.no_grad():
                w_lo = dec_lo.flip(0).unsqueeze(0).unsqueeze(0).expand(d_model, 1, -1)
                w_hi = dec_hi.flip(0).unsqueeze(0).unsqueeze(0).expand(d_model, 1, -1)
                lo.weight.copy_(w_lo)
                hi.weight.copy_(w_hi)
            if not trainable:
                lo.weight.requires_grad_(False)
                hi.weight.requires_grad_(False)
            self.low_pass.append(lo)
            self.high_pass.append(hi)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model)
        h = x.transpose(1, 2)                              # (B, D, T)
        details = []
        for lvl in range(self.levels):
            d = self.high_pass[lvl](h)                      # (B, D, T/2^(l+1))
            h = self.low_pass[lvl](h)                       # (B, D, T/2^(l+1))
            details.append(d.transpose(1, 2))               # (B, T_l, D)
        # bands: [approx, detail_L, detail_{L-1}, … detail_1]  (coarse→fine)
        bands = [h.transpose(1, 2)] + details[::-1]
        return bands


# ════════════════════════════════════════════════════════════════════════════
# 3. Band-Level Cross-Modal Attention  (★ THE FIX ★)
# ════════════════════════════════════════════════════════════════════════════
class BandCrossModalAttention(nn.Module):
    """
    At each frequency band, each modality attends to all other modalities
    at each timestep. This gives frequency-specific cross-modal alignment:
    temperature–humidity coupling is strong at low freq, wind–precipitation
    coupling at high freq.

    Input:  list of M tensors, each (B, T_l, d_model)
    Output: list of M enhanced tensors, attention weights for interpretability
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, band_reps: list[torch.Tensor]):
        """
        band_reps: list of M tensors, each (B, T_l, D)
        """
        if len(band_reps) == 1:
            return band_reps, None          # single modality — skip

        B, T_l, D = band_reps[0].shape
        M = len(band_reps)

        # (B, M, T_l, D) → (B*T_l, M, D)  — at each timestep, M modality tokens
        stacked = torch.stack(band_reps, dim=1)                 # (B, M, T_l, D)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B * T_l, M, D)

        attn_out, attn_w = self.mha(stacked, stacked, stacked)  # (B*T_l, M, D)
        attn_out = self.norm(stacked + self.dropout(attn_out))

        attn_out = attn_out.reshape(B, T_l, M, D).permute(0, 2, 1, 3)  # (B, M, T_l, D)

        enhanced = [attn_out[:, m] for m in range(M)]
        # attn_w: (B*T_l, M, M) → (B, T_l, n_heads?, M, M) — keep compact
        return enhanced, attn_w.reshape(B, T_l, M, M)


# ════════════════════════════════════════════════════════════════════════════
# 4. Per-Band Graph Learning
# ════════════════════════════════════════════════════════════════════════════
class BandGraphLearning(nn.Module):
    """
    For each frequency band, learn a soft adjacency matrix among variates
    via node-embedding similarity, then run 2-layer GAT-style message passing.

    A_l = softmax(E_l · E_l^T / √d)
    H^(k+1) = σ(A_l H^(k) W^(k))

    Input:  (B, T_l, d_model)   — all variates stacked in d_model
    Output: (B, T_l, d_model),  adjacency matrix for interpretability
    """

    def __init__(self, d_model: int, n_nodes: int = 0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # Node embeddings (used when n_nodes > 0 — otherwise treat d_model channels)
        self.embed_q = nn.Linear(d_model, d_model)
        self.embed_k = nn.Linear(d_model, d_model)
        # 2-layer message passing
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.sparsity_reg = 0.0  # updated during forward, read by loss

    def forward(self, x: torch.Tensor):
        # x: (B, T_l, D)
        B_sz, T_l, D = x.shape

        # Compute adjacency: treat temporal mean as node representation
        node_rep = x.mean(dim=1)                         # (B, D)
        Q = self.embed_q(node_rep)                       # (B, D)
        K = self.embed_k(node_rep)                       # (B, D)
        # Feature-wise adjacency: (B, T_l, T_l) via temporal correlation
        # Reshape to per-timestep adjacency for richer structure
        Q_t = self.embed_q(x)                            # (B, T_l, D)
        K_t = self.embed_k(x)                            # (B, T_l, D)
        A = torch.bmm(Q_t, K_t.transpose(1, 2))         # (B, T_l, T_l)
        A = A / math.sqrt(D)
        A = F.softmax(A, dim=-1)

        self.sparsity_reg = A.norm(p=1, dim=(-2, -1)).mean()

        # Layer 1
        h = torch.bmm(A, x)                              # (B, T_l, D)
        h = self.W1(h)
        h = F.gelu(h)
        h = self.norm1(x + self.dropout(h))

        # Layer 2
        h2 = torch.bmm(A, h)
        h2 = self.W2(h2)
        h2 = F.gelu(h2)
        h2 = self.norm2(h + self.dropout(h2))

        return h2, A


# ════════════════════════════════════════════════════════════════════════════
# 5. Expert Modules
# ════════════════════════════════════════════════════════════════════════════

class SSMExpert(nn.Module):
    """
    S4D-style state-space model for coarsest bands (long-range dependencies).
    Uses convolutional mode via FFT — no custom CUDA kernels needed.
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A: log-parameterised for stability, complex
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(d_model, d_state))
        )
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(d_state).float().unsqueeze(0).expand(d_model, -1).clone()
        )
        self.B_param = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_param = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.zeros(d_model) - 2.0)      # log(dt)

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _ssm_kernel(self, T: int, device: torch.device):
        dt = F.softplus(self.log_dt)                                  # (D,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag           # (D, N)
        dtA = (dt.unsqueeze(-1) * A).to(torch.complex64)             # (D, N)

        # Vandermonde: exp(dtA * t) for t = 0..T-1
        t_idx = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
        V = torch.exp(dtA.unsqueeze(-1) * t_idx)                     # (D, N, T)

        CB = (self.C_param * self.B_param * dt.unsqueeze(-1)).to(torch.complex64)  # (D, N)
        K = torch.einsum("dn,dnt->dt", CB, V).real                   # (D, T)
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        B_sz, T, D = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        K = self._ssm_kernel(T, x.device)                            # (D, T)

        # Causal conv via FFT — use float32 + power-of-2 for cuFFT/AMP compat
        u = x_proj.transpose(1, 2).float()                           # (B, D, T)
        K_f32 = K.float()
        n_fft = 1
        while n_fft < 2 * T:
            n_fft *= 2
        U_f = torch.fft.rfft(u, n=n_fft)                             # (B, D, n_fft//2+1)
        K_f = torch.fft.rfft(K_f32, n=n_fft)                         # (D, n_fft//2+1)
        Y_f = U_f * K_f.unsqueeze(0)
        y = torch.fft.irfft(Y_f, n=n_fft)[..., :T]                  # (B, D, T)
        y = y.to(x_proj.dtype).transpose(1, 2)                       # (B, T, D)

        y = y + x_proj * self.D_param
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.norm(residual + self.drop(y))


class GRUExpert(nn.Module):
    """Bidirectional GRU expert for mid-frequency bands."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h, _ = self.gru(x)                                           # (B, T, D)
        h = self.proj(h)
        return self.norm(residual + self.drop(h))


class TransformerExpert(nn.Module):
    """Transformer encoder expert for fine-detail bands."""

    def __init__(
        self, d_model: int, n_heads: int = 4, d_ff: int = 0,
        n_layers: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff == 0:
            d_ff = d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TCNExpert(nn.Module):
    """Temporal Convolutional Network expert for finest bands."""

    def __init__(
        self, d_model: int, kernel_size: int = 3,
        n_layers: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation
            layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size,
                          padding=pad, dilation=dilation, groups=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, 1),
                nn.Dropout(dropout),
            ))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                                        # (B, D, T)
        for conv_block, norm in zip(self.layers, self.norms):
            residual = h
            out = conv_block(h)
            out = out[..., :h.size(-1)]                               # trim causal padding
            h = norm((residual + out).transpose(1, 2)).transpose(1, 2)
        return h.transpose(1, 2)


# ════════════════════════════════════════════════════════════════════════════
# 6. Frequency-Aware Gating
# ════════════════════════════════════════════════════════════════════════════
class FrequencyGating(nn.Module):
    """
    Routes each frequency band to specialised experts.
    Supports soft routing (full distribution) and Top-K sparse routing.

    Input:  list of bands, each (B, T_l, D)
    Output: routing weights (B, n_bands, n_experts), aux load-balance loss
    """

    def __init__(
        self, d_model: int, n_bands: int, n_experts: int = 4,
        top_k: int = 0, dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_experts = n_experts
        self.top_k = top_k                       # 0 = soft, >0 = sparse

        self.band_embed = nn.Parameter(torch.randn(n_bands, d_model) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_experts),
        )

    def forward(self, bands: list[torch.Tensor]):
        """
        Returns:
            weights: (B, n_bands, n_experts) — routing distribution
            aux_loss: scalar — KL from uniform for load balance
        """
        B = bands[0].size(0)
        weights_list = []
        for i, band in enumerate(bands):
            pooled = band.mean(dim=1)                                # (B, D)
            be = self.band_embed[i].unsqueeze(0).expand(B, -1)      # (B, D)
            gate_in = torch.cat([pooled, be], dim=-1)                # (B, 2D)
            logits = self.gate(gate_in)                              # (B, n_experts)

            if self.top_k > 0:
                topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
                mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
                logits = logits.masked_fill(mask == 0, float("-inf"))

            w = F.softmax(logits, dim=-1)                            # (B, n_experts)
            weights_list.append(w)

        weights = torch.stack(weights_list, dim=1)                   # (B, n_bands, n_experts)

        # Load-balance loss: KL(mean_weights || uniform)
        avg = weights.mean(dim=(0, 1))                               # (n_experts,)
        uniform = torch.ones_like(avg) / self.n_experts
        aux_loss = F.kl_div(
            (avg + 1e-8).log(), uniform, reduction="batchmean",
        )
        return weights, aux_loss


class ExpertRegistry(nn.Module):
    """
    Houses all expert types and dispatches bands via gating weights.
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            SSMExpert(d_model, d_state=d_state, dropout=dropout),
            GRUExpert(d_model, dropout=dropout),
            TransformerExpert(d_model, dropout=dropout),
            TCNExpert(d_model, dropout=dropout),
        ])

    def forward(
        self,
        bands: list[torch.Tensor],
        weights: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        bands:   list of n_bands tensors, each (B, T_l, D)
        weights: (B, n_bands, n_experts)
        Returns: list of n_bands tensors, expert-processed
        """
        outputs = []
        for i, band in enumerate(bands):
            w = weights[:, i, :]                                    # (B, n_experts)
            # Weighted sum of expert outputs
            expert_outs = []
            for j, expert in enumerate(self.experts):
                out_j = expert(band)                                # (B, T_l, D)
                expert_outs.append(out_j * w[:, j].unsqueeze(-1).unsqueeze(-1))
            merged = sum(expert_outs)                               # (B, T_l, D)
            outputs.append(merged)
        return outputs


# ════════════════════════════════════════════════════════════════════════════
# 7. Cross-Band Fusion
# ════════════════════════════════════════════════════════════════════════════
class CrossBandFusion(nn.Module):
    """
    Fuses expert outputs across frequency bands via cross-attention.
    Upsamples all bands to the finest resolution, adds band-position
    embeddings, and uses cross-attention with coarsest band as query.
    """

    def __init__(self, d_model: int, n_bands: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.band_pos = nn.Parameter(torch.randn(n_bands, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, bands: list[torch.Tensor]) -> torch.Tensor:
        """
        bands: list of n_bands tensors, each (B, T_l, D), lengths may differ
        Returns: (B, T_max, D)
        """
        T_max = max(b.size(1) for b in bands)
        B, _, D = bands[0].shape

        # Upsample all bands to T_max and add band-position embeddings
        upsampled = []
        for i, band in enumerate(bands):
            if band.size(1) < T_max:
                band = F.interpolate(
                    band.transpose(1, 2), size=T_max, mode="linear", align_corners=False,
                ).transpose(1, 2)
            band = band + self.band_pos[i]
            upsampled.append(band)

        # Concatenate: all bands as KV sequence
        kv = torch.cat(upsampled, dim=1)                           # (B, n_bands*T_max, D)
        query = upsampled[0]                                       # coarsest as query

        # Cross-attention
        attn_out, _ = self.cross_attn(query, kv, kv)               # (B, T_max, D)
        h = self.norm1(query + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h


# ════════════════════════════════════════════════════════════════════════════
# 8. Cross-Modal Fusion (final stage)
# ════════════════════════════════════════════════════════════════════════════
class CrossModalFusion(nn.Module):
    """
    Final aggregation across modalities via self-attention over modality tokens.
    (B, M, D) self-attention → (B, D)
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mod_features: list[torch.Tensor]) -> torch.Tensor:
        """
        mod_features: list of M tensors, each (B, T, D)
        Returns: (B, T, D)
        """
        if len(mod_features) == 1:
            return mod_features[0]

        # Temporal pool → modality tokens
        tokens = torch.stack(
            [f.mean(dim=1) for f in mod_features], dim=1,
        )                                                            # (B, M, D)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm(tokens + self.dropout(attn_out))          # (B, M, D)

        # Combine: mean of modality tokens, then broadcast-add to finest
        fused = tokens.mean(dim=1)                                   # (B, D)

        # Also produce a temporal output: weight each modality's temporal
        # representation by the attention-derived importance
        importance = F.softmax(tokens.mean(dim=-1), dim=-1)          # (B, M)
        temporal = torch.stack(mod_features, dim=1)                  # (B, M, T, D)
        T = mod_features[0].size(1)
        # Align temporal lengths (take min)
        T_min = min(f.size(1) for f in mod_features)
        temporal = torch.stack(
            [f[:, :T_min, :] for f in mod_features], dim=1,
        )                                                            # (B, M, T_min, D)
        weighted = torch.einsum("bm,bmtd->btd", importance, temporal)
        return weighted                                              # (B, T_min, D)


# ════════════════════════════════════════════════════════════════════════════
# 9. Prediction Head
# ════════════════════════════════════════════════════════════════════════════
class PredictionHead(nn.Module):
    """Flatten → Linear → GELU → Dropout → Linear → (B, H, C_target)"""

    def __init__(
        self, d_model: int, seq_len: int, pred_len: int,
        n_targets: int, dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.flatten_dim = d_model * seq_len
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len * n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  — T may be shorter than expected; pad or trim
        B, T, D = x.shape
        expected_T = self.flatten_dim // D
        if T < expected_T:
            x = F.pad(x, (0, 0, 0, expected_T - T))
        elif T > expected_T:
            x = x[:, :expected_T, :]
        out = self.head(x)                                           # (B, H * C)
        return out.view(B, self.pred_len, self.n_targets)


# ════════════════════════════════════════════════════════════════════════════
# 10. Full WaveMoE Model
# ════════════════════════════════════════════════════════════════════════════
class WaveMoE(nn.Module):
    """
    Complete WaveMoE architecture.

    Config dict keys:
        modality_channels: list[int]  — #features per modality  e.g. [4, 3]
        d_model:           int        — latent dim              e.g. 128
        seq_len:           int        — input window            e.g. 96
        pred_len:          int        — forecast horizon        e.g. 96
        n_targets:         int        — #target channels        e.g. 7
        wavelet:           str        — wavelet family          e.g. 'db4'
        dwt_levels:        int        — decomposition depth     e.g. 3
        dwt_trainable:     bool       — fine-tune filters       e.g. True
        top_k:             int        — 0=soft, >0=sparse       e.g. 0
        use_graph:         bool       — enable graph learning   e.g. True
        d_state:           int        — SSM state dim           e.g. 64
        dropout:           float      — global dropout          e.g. 0.1
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        mc = cfg["modality_channels"]
        D = cfg["d_model"]
        n_bands = cfg["dwt_levels"] + 1
        dropout = cfg.get("dropout", 0.1)
        self.n_modalities = len(mc)

        # Per-modality encoders
        self.encoders = nn.ModuleList([
            ModalityEncoder(c, D, dropout) for c in mc
        ])

        # Shared DWT (same filters for all modalities)
        self.dwt = LearnableDWT(
            D, wavelet=cfg.get("wavelet", "db4"),
            levels=cfg.get("dwt_levels", 3),
            trainable=cfg.get("dwt_trainable", True),
        )

        # ★ Band-level cross-modal attention (one per band)
        self.band_cross_modal = nn.ModuleList([
            BandCrossModalAttention(D, n_heads=4, dropout=dropout)
            for _ in range(n_bands)
        ])

        # Optional graph learning (one per band)
        self.use_graph = cfg.get("use_graph", True)
        if self.use_graph:
            self.graph_layers = nn.ModuleList([
                BandGraphLearning(D, dropout=dropout) for _ in range(n_bands)
            ])

        # Shared gating and experts (applied per modality)
        self.gating = FrequencyGating(
            D, n_bands, n_experts=4,
            top_k=cfg.get("top_k", 0), dropout=dropout,
        )
        self.experts = ExpertRegistry(
            D, d_state=cfg.get("d_state", 64), dropout=dropout,
        )

        # Per-modality cross-band fusion
        self.band_fusion = CrossBandFusion(D, n_bands, n_heads=4, dropout=dropout)

        # Final cross-modal fusion
        self.modal_fusion = CrossModalFusion(D, n_heads=4, dropout=dropout)

        # Prediction head
        # After band fusion, temporal length = T // 2^L (coarsest query len)
        head_T = max(1, cfg["seq_len"] // (2 ** cfg.get("dwt_levels", 3)))
        self.pred_head = PredictionHead(
            D, head_T, cfg["pred_len"], cfg["n_targets"], dropout,
        )

        self._aux_loss = 0.0
        self._graph_reg = 0.0

    def forward(
        self,
        x_modalities: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        x_modalities: list of M tensors, each (B, T, C_m)
        Returns: (B, pred_len, n_targets)
        """
        # ── 1. Encode each modality ──
        encoded = [enc(x) for enc, x in zip(self.encoders, x_modalities)]
        # each: (B, T, D)

        # ── 2. DWT decomposition per modality ──
        all_bands = [self.dwt(e) for e in encoded]
        # all_bands[m][l]: (B, T_l, D)

        n_bands = len(all_bands[0])

        # ── 3. ★ Band-level cross-modal attention ──
        # For each band, gather that band from all modalities, do cross-modal attn
        cross_modal_bands = [[] for _ in range(self.n_modalities)]
        self._band_attn_weights = []
        for l in range(n_bands):
            band_reps = [all_bands[m][l] for m in range(self.n_modalities)]
            enhanced, attn_w = self.band_cross_modal[l](band_reps)
            self._band_attn_weights.append(attn_w)
            for m in range(self.n_modalities):
                cross_modal_bands[m].append(enhanced[m])

        # ── 4. Per-band graph learning (optional) ──
        self._graph_reg = 0.0
        if self.use_graph:
            for m in range(self.n_modalities):
                for l in range(n_bands):
                    g_out, adj = self.graph_layers[l](cross_modal_bands[m][l])
                    cross_modal_bands[m][l] = g_out
                    self._graph_reg += self.graph_layers[l].sparsity_reg

        # ── 5. Gating + Expert processing per modality ──
        self._aux_loss = 0.0
        expert_outputs = []   # list of M, each list of n_bands tensors
        for m in range(self.n_modalities):
            weights, aux = self.gating(cross_modal_bands[m])
            self._aux_loss += aux
            exp_out = self.experts(cross_modal_bands[m], weights)
            expert_outputs.append(exp_out)

        # ── 6. Cross-band fusion per modality ──
        fused_modalities = []
        for m in range(self.n_modalities):
            fused = self.band_fusion(expert_outputs[m])             # (B, T_coarse, D)
            fused_modalities.append(fused)

        # ── 7. Cross-modal fusion ──
        combined = self.modal_fusion(fused_modalities)              # (B, T', D)

        # ── 8. Prediction head ──
        forecast = self.pred_head(combined)                         # (B, H, C_target)
        return forecast

    def get_loss_components(self):
        """Return auxiliary losses for training."""
        return {
            "aux_loss": self._aux_loss,
            "graph_reg": self._graph_reg,
        }

    def get_interpretability(self):
        """Return attention weights for visualisation."""
        return {
            "band_cross_modal_attn": self._band_attn_weights,
        }