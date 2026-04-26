"""Quick timing test: how fast is Kronos predict_batch on real ES bars at varying batch sizes?"""
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
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print("Loading corpus + ES master...")
    corpus = pd.read_parquet("artifacts/v12_training_corpus.parquet")
    de3 = corpus[corpus["strategy"] == "DynamicEngine3"].copy()
    es = pd.read_parquet("es_master_outrights.parquet")
    print(f"  DE3 candidates: {len(de3)}")
    print(f"  ES master rows: {len(es)}")

    print("Loading Kronos-small...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512, device=device)

    # Pre-index ES master by symbol
    es_by_sym = {sym: g.copy() for sym, g in es.groupby("symbol")}
    for sym, g in es_by_sym.items():
        g.sort_index(inplace=True)
        es_by_sym[sym] = g

    # Pick K candidates and assemble inputs
    def make_input(row):
        sym = row["contract"]
        ts = pd.Timestamp(row["ts"])
        if sym not in es_by_sym:
            return None
        g = es_by_sym[sym]
        # Find <= ts (use UTC tz-naive comparison)
        try:
            cutoff = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
        except Exception:
            cutoff = ts
        # ES master index is tz-aware (-05:00); align
        idx = g.index
        if idx.tz is None:
            ts_lookup = cutoff
        else:
            ts_lookup = ts.tz_convert(idx.tz) if ts.tzinfo else ts.tz_localize(idx.tz)
        # Use searchsorted for efficiency
        loc = idx.searchsorted(ts_lookup, side="right")
        start = max(0, loc - 512)
        if loc - start < 512:
            return None
        sl = g.iloc[start:loc]
        if len(sl) < 512:
            return None
        df = pd.DataFrame({
            "open": sl["open"].astype(float).values,
            "high": sl["high"].astype(float).values,
            "low": sl["low"].astype(float).values,
            "close": sl["close"].astype(float).values,
            "volume": sl["volume"].astype(float).values,
            "amount": (sl["volume"].astype(float) * sl["close"].astype(float)).values,
        }, index=sl.index)
        x_ts = pd.Series(sl.index)
        # Forecast 30 future minutes
        last_ts = sl.index[-1]
        y_ts = pd.Series(pd.date_range(last_ts + pd.Timedelta(minutes=1), periods=30, freq="1min", tz=last_ts.tz))
        return df, x_ts, y_ts

    samples = []
    for i, row in de3.head(40).iterrows():
        s = make_input(row)
        if s:
            samples.append(s)
        if len(samples) >= 32:
            break
    print(f"Built {len(samples)} usable samples for timing test")

    for batch_size in [1, 4, 8, 16, 32]:
        if batch_size > len(samples):
            continue
        chunk = samples[:batch_size]
        df_list = [c[0] for c in chunk]
        x_ts_list = [c[1] for c in chunk]
        y_ts_list = [c[2] for c in chunk]
        t0 = time.time()
        out = predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=30,
            T=1.0,
            top_p=0.9,
            sample_count=10,
            verbose=False,
        )
        el = time.time() - t0
        print(f"  batch_size={batch_size}: {el:.2f}s  ({el/batch_size*1000:.0f} ms/series)")


if __name__ == "__main__":
    main()
