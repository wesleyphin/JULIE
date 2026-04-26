"""Kronos smoke test — verifies install + basic inference on synthetic 1-min OHLCV.

Run with:
    cd /Users/wes/Downloads/JULIE001
    source .kronos_venv/bin/activate
    OMP_NUM_THREADS=1 python3 tools/kronos_smoke_test.py
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kronos_external"))

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosTokenizer, KronosPredictor


def main():
    print("=" * 60)
    print("Kronos smoke test")
    print("=" * 60)

    # Pick device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"torch: {torch.__version__}")

    print("\nLoading tokenizer...")
    t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    print(f"  tokenizer loaded in {time.time()-t0:.1f}s")

    print("\nLoading model (Kronos-small)...")
    t0 = time.time()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print(f"  model loaded in {time.time()-t0:.1f}s")

    predictor = KronosPredictor(model, tokenizer, max_context=512, device=device)

    # Build synthetic 1-min ES-style series, 600 bars
    rng = np.random.default_rng(42)
    n = 600
    ts = pd.date_range("2025-01-01 09:30", periods=n, freq="1min")
    base = 6000.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = base + np.abs(rng.normal(0.0, 0.4, n))
    low = base - np.abs(rng.normal(0.0, 0.4, n))
    open_ = base + rng.normal(0.0, 0.1, n)
    close = base + rng.normal(0.0, 0.1, n)
    vol = rng.integers(100, 1000, n).astype(float)
    amount = vol * base

    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum.reduce([high, open_, close]),
        "low": np.minimum.reduce([low, open_, close]),
        "close": close,
        "volume": vol,
        "amount": amount,
    })
    df.index = ts

    x_df = df.iloc[:512]
    x_ts = pd.Series(df.index[:512])
    y_ts = pd.Series(df.index[512:512 + 30])

    print(f"\nInference: pred_len=30, sample_count=10, context=512")
    t0 = time.time()
    pred = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=30,
        T=1.0,
        top_p=0.9,
        sample_count=10,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  inference time: {elapsed:.2f}s")
    print(f"  forecast shape: {pred.shape}")
    print(f"  forecast head:\n{pred.head()}")
    print(f"  forecast tail:\n{pred.tail()}")

    last_close = float(x_df["close"].iloc[-1])
    final_pred_close = float(pred["close"].iloc[-1])
    pred_atr = float((pred["high"] - pred["low"]).mean())
    print(
        f"\n  last context close: {last_close:.2f}  "
        f"final pred close: {final_pred_close:.2f}  "
        f"signed move: {final_pred_close-last_close:+.2f}"
    )
    print(f"  pred ATR (mean H-L over 30 bars): {pred_atr:.3f}")
    print(f"\nSmoke test PASS — total ~{time.time()-t0:.1f}s post-load")


if __name__ == "__main__":
    main()
