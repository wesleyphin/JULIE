"""Bar-sequence encoder (Path 1 of smartness roadmap).

Self-supervised TCN (Temporal Convolutional Network) that ingests the
last N bars of OHLCV and produces a fixed-dim embedding capturing
pattern / momentum / volatility structure GBTs can't see from flat
feature rows.

The encoder is pre-trained on next-bar-direction prediction (plus a
small reconstruction head for regularization) across the full ES
parquet. Downstream ML layers can then treat the frozen encoder's
embedding as additional features, or fine-tune through it.

Architecture: small 1D dilated convolutions — compact, fast,
inference-cheap on CPU. 5 OHLCV channels → dilated conv stack → mean
pool → 32-dim embedding.

Public API:
  BarEncoder(seq_len=60, embed_dim=32)      — nn.Module
  build_sequences(bars_df) -> Tensor        — training-data prep
  train_encoder(...)                         — self-supervised trainer
  encode(bars_df, end_idx) -> ndarray       — inference wrapper
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ES_PARQUET = ROOT / "es_master_outrights.parquet"
OUT_ENCODER = ROOT / "artifacts" / "signal_gate_2025" / "bar_encoder.pt"


SEQ_LEN = 60              # bars of history fed to encoder
EMBED_DIM = 32            # output embedding size
BAR_CHANNELS = 5          # open, high, low, close, volume


class BarEncoder(nn.Module):
    """Compact temporal conv encoder.

    Input:  (batch, BAR_CHANNELS, SEQ_LEN)  — NCHW-style 1D
    Output: (batch, EMBED_DIM)              — fixed-dim embedding
    """
    def __init__(self, seq_len: int = SEQ_LEN, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # Dilated 1D convs — each block halves the effective receptive
        # field requirement; 3 blocks give us ~15-bar receptive field at
        # dilation 4, covering most pattern scales in 60 bars.
        self.blocks = nn.ModuleList([
            # Block 1 — dilation 1
            nn.Sequential(
                nn.Conv1d(BAR_CHANNELS, 16, kernel_size=3, padding=1, dilation=1),
                nn.GELU(),
                nn.Conv1d(16, 16, kernel_size=3, padding=1, dilation=1),
                nn.GELU(),
            ),
            # Block 2 — dilation 2
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
            ),
            # Block 3 — dilation 4
            nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4),
                nn.GELU(),
            ),
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(32, embed_dim)
        # Self-supervised heads (used during pre-training only)
        self.direction_head = nn.Linear(embed_dim, 3)  # up / flat / down
        self.return_head = nn.Linear(embed_dim, 1)     # next-bar return magnitude

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, BAR_CHANNELS, SEQ_LEN) → (batch, EMBED_DIM)."""
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).squeeze(-1)   # (batch, 32)
        return self.proj(x)            # (batch, EMBED_DIM)

    def forward(self, x: torch.Tensor):
        emb = self.encode(x)
        return emb, self.direction_head(emb), self.return_head(emb)


def _normalize_bar_window(window: np.ndarray) -> np.ndarray:
    """Normalize a (SEQ_LEN, 5) window:
       - OHLC: pct change from the window's first close
       - volume: z-score within the window
    Returns (5, SEQ_LEN) for Conv1d's NCHW convention.
    """
    ohlc = window[:, :4].astype(np.float32)
    vol = window[:, 4].astype(np.float32)
    anchor = float(ohlc[0, 3])  # first bar's close
    if anchor <= 0:
        anchor = 1.0
    ohlc_norm = (ohlc - anchor) / anchor * 100.0  # pct
    vol_mean = vol.mean(); vol_std = vol.std() + 1e-6
    vol_norm = (vol - vol_mean) / vol_std
    out = np.concatenate([ohlc_norm, vol_norm[:, None]], axis=1)  # (SEQ_LEN, 5)
    out = out.T  # (5, SEQ_LEN)
    return out


def build_sequences(bars_df, *, seq_len: int = SEQ_LEN, stride: int = 10,
                    max_samples: int = 400_000):
    """Build (X, y_dir, y_ret) from a DataFrame of OHLCV bars.

    Target labels:
      y_dir: 0 = next bar down by >0.05%, 1 = flat, 2 = up by >0.05%
      y_ret: actual next-bar pct return (continuous)

    Subsamples to keep training tractable.
    """
    if "volume" not in bars_df.columns:
        bars_df = bars_df.assign(volume=1.0)
    arr = bars_df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    n = len(arr)
    xs, y_dir, y_ret = [], [], []
    i = seq_len
    count = 0
    while i < n - 1 and count < max_samples:
        window = arr[i - seq_len: i]  # (seq_len, 5)
        cur_close = arr[i - 1, 3]
        next_close = arr[i, 3]
        if cur_close <= 0:
            i += stride; continue
        pct = (next_close - cur_close) / cur_close * 100.0
        if pct >= 0.05:
            label = 2
        elif pct <= -0.05:
            label = 0
        else:
            label = 1
        xs.append(_normalize_bar_window(window))
        y_dir.append(label)
        y_ret.append(pct)
        i += stride
        count += 1
    X = np.stack(xs).astype(np.float32)         # (n, 5, seq_len)
    y_dir = np.array(y_dir, dtype=np.int64)
    y_ret = np.array(y_ret, dtype=np.float32)
    return X, y_dir, y_ret


def train_encoder(
    bars_df,
    *,
    seq_len: int = SEQ_LEN,
    embed_dim: int = EMBED_DIM,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: Optional[str] = None,
    max_samples: int = 400_000,
) -> BarEncoder:
    """Self-supervised training loop."""
    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bar_encoder] device={device}  seq_len={seq_len}  embed={embed_dim}")
    print(f"[bar_encoder] building sequences...")
    X, y_dir, y_ret = build_sequences(bars_df, seq_len=seq_len, max_samples=max_samples)
    n = len(X)
    print(f"  sequences: {n:,}   y_dir balance: "
          f"{np.bincount(y_dir, minlength=3)/n}")
    # Train/val split (90/10 random; for self-supervised pretraining random
    # split is fine since we're learning representations, not claiming
    # out-of-time generalization of the downstream task)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr = torch.from_numpy(X[tr_idx]); y_dir_tr = torch.from_numpy(y_dir[tr_idx]); y_ret_tr = torch.from_numpy(y_ret[tr_idx])
    X_va = torch.from_numpy(X[va_idx]); y_dir_va = torch.from_numpy(y_dir[va_idx]); y_ret_va = torch.from_numpy(y_ret[va_idx])

    model = BarEncoder(seq_len=seq_len, embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        losses = []; corrects = 0; total = 0
        for b in range(0, len(X_tr), batch_size):
            bi = perm[b: b + batch_size]
            x = X_tr[bi].to(device); yd = y_dir_tr[bi].to(device); yr = y_ret_tr[bi].to(device)
            opt.zero_grad()
            emb, d_pred, r_pred = model(x)
            loss_dir = F.cross_entropy(d_pred, yd)
            loss_ret = F.mse_loss(r_pred.squeeze(-1), yr)
            loss = loss_dir + 0.1 * loss_ret   # direction is primary; return is regularizer
            loss.backward()
            opt.step()
            losses.append(loss.item())
            corrects += (d_pred.argmax(dim=-1) == yd).sum().item()
            total += len(yd)
        train_acc = corrects / total
        # Val — chunk through to keep memory down on MPS
        model.eval()
        with torch.no_grad():
            val_corrects = 0; val_total = 0; val_ret_err_sum = 0.0
            for b in range(0, len(X_va), batch_size):
                x = X_va[b: b + batch_size].to(device)
                yd = y_dir_va[b: b + batch_size].to(device)
                yr = y_ret_va[b: b + batch_size].to(device)
                emb, d_pred, r_pred = model(x)
                val_corrects += (d_pred.argmax(dim=-1) == yd).sum().item()
                val_total += len(yd)
                val_ret_err_sum += (r_pred.squeeze(-1) - yr).abs().sum().item()
            val_dir_acc = val_corrects / max(1, val_total)
            val_ret_mae = val_ret_err_sum / max(1, val_total)
        print(f"  epoch {epoch+1}/{epochs}  train_loss={np.mean(losses):.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_dir_acc:.3f}  val_ret_mae={val_ret_mae:.4f}")
    model.eval()
    return model


def encode(model: BarEncoder, bars_df, end_idx: int) -> np.ndarray:
    """Inference: produce a 32-dim embedding for the bar sequence ending
    at `end_idx` (inclusive) in `bars_df`. Pads at front if < SEQ_LEN bars
    of history are available."""
    n = len(bars_df)
    if end_idx >= n:
        end_idx = n - 1
    start = max(0, end_idx - model.seq_len + 1)
    window = bars_df.iloc[start: end_idx + 1][["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    if len(window) < model.seq_len:
        pad = np.zeros((model.seq_len - len(window), 5), dtype=np.float64)
        window = np.concatenate([pad, window], axis=0)
    x = _normalize_bar_window(window)            # (5, seq_len)
    # Move input to model's device (model may be on CPU/MPS/CUDA)
    device = next(model.parameters()).device
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode(x_t).cpu().numpy()
    return emb[0]   # (EMBED_DIM,)


def _main():
    import argparse, pandas as pd, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    print(f"[parquet] loading {ES_PARQUET}")
    df = pd.read_parquet(ES_PARQUET)
    df = df[df.index.year >= 2020].sort_index()
    if "symbol" in df.columns and "volume" in df.columns:
        df = df.sort_values("volume", ascending=False).groupby(df.index).first().sort_index()
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    print(f"  {len(df):,} bars")

    t0 = time.time()
    model = train_encoder(
        df, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, max_samples=args.max_samples,
    )
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed/60:.1f} min")

    OUT_ENCODER.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "seq_len": model.seq_len,
        "embed_dim": model.embed_dim,
        "bar_channels": BAR_CHANNELS,
        "architecture": "dilated_conv_tcn",
    }, OUT_ENCODER)
    print(f"[write] {OUT_ENCODER}")

    # Smoke-test inference
    print("\n[inference smoke test]")
    emb = encode(model, df.tail(100), end_idx=99)
    print(f"  embedding shape: {emb.shape}  "
          f"norm: {float(np.linalg.norm(emb)):.3f}  "
          f"range: [{emb.min():.3f}, {emb.max():.3f}]")


if __name__ == "__main__":
    _main()
