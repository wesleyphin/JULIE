"""Kronos Phase B-RA — extract Kronos forecast features for RA + friend-rule
candidates. Mirror of `kronos_phase_b_extract.py` but filtered to
family == 'regimeadaptive' AND allowed_by_friend_rule == True.

Outputs:
  artifacts/v18_kronos_features_ra.parquet
"""
from __future__ import annotations

import os
import sys
import gc
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = "/Users/wes/Downloads/JULIE001"
sys.path.insert(0, os.path.join(ROOT, "kronos_external"))

import psutil
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from model import Kronos, KronosTokenizer, KronosPredictor


CORPUS_PATH = os.path.join(ROOT, "artifacts", "v12_training_corpus.parquet")
BARS_PATH = os.path.join(ROOT, "es_master_outrights.parquet")
OUT_PARQUET = os.path.join(ROOT, "artifacts", "v18_kronos_features_ra.parquet")

CTX_BARS = 512
PRED_LEN = 30
RAM_FLOOR_MB = 1500


def ram_mb():
    return psutil.virtual_memory().available // 1024 // 1024


def used_mb():
    return psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024


def get_window(bars, end_ts, n):
    sub = bars.loc[:end_ts]
    if len(sub) and sub.index[-1] == end_ts:
        sub = sub.iloc[:-1]
    if len(sub) < 100:
        return None
    return sub.iloc[-n:].copy()


def normalize_bars(window):
    df = window[["open", "high", "low", "close", "volume"]].copy().astype(float)
    df["amount"] = df["volume"] * df["close"]
    df = df.dropna()
    return df


def build_y_timestamps(last_ts, n):
    idx = pd.date_range(start=last_ts + pd.Timedelta(minutes=1), periods=n, freq="1min")
    return pd.Series(idx)


def kronos_features(pred_df, entry_price, side, inf_time_s):
    high_max = float(pred_df["high"].max())
    low_min = float(pred_df["low"].min())
    final_close = float(pred_df["close"].iloc[-1])
    atr_30 = float((pred_df["high"] - pred_df["low"]).mean())
    max_high_above = high_max - entry_price
    min_low_below = entry_price - low_min
    side_up = (side or "").upper()
    if side_up in ("LONG", "BUY", "L"):
        favorable = max_high_above
    elif side_up in ("SHORT", "SELL", "S"):
        favorable = min_low_below
    else:
        favorable = float("nan")
    return {
        "kronos_pred_atr_30bar": atr_30,
        "kronos_dir_move": final_close - entry_price,
        "kronos_max_high_above": max_high_above,
        "kronos_min_low_below": min_low_below,
        "kronos_close_vs_entry": final_close - entry_price,
        "kronos_pred_favorable": favorable,
        "kronos_inf_time_s": inf_time_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-count", type=int, default=5)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (0 = all)")
    ap.add_argument("--resume", action="store_true", help="skip already-processed idxs from existing OUT_PARQUET")
    args = ap.parse_args()

    print(f"[init] free={ram_mb()} MB, used={used_mb()} MB")
    print(f"  sample_count={args.sample_count}  batch={args.batch}  limit={args.limit}  resume={args.resume}")

    corpus = pd.read_parquet(CORPUS_PATH)
    ra_fr = corpus[(corpus["family"] == "regimeadaptive") & (corpus["allowed_by_friend_rule"] == True)].copy()
    ra_fr = ra_fr.reset_index(drop=True)
    ra_fr["row_idx"] = ra_fr.index
    print(f"  RA+friend candidates: {len(ra_fr)}")

    if args.limit > 0:
        ra_fr = ra_fr.iloc[:args.limit].copy()
    ra_fr["is_winner"] = (ra_fr["net_pnl_after_haircut"] > 0).astype(int)

    done_idxs = set()
    existing_rows = []
    if args.resume and os.path.exists(OUT_PARQUET):
        existing = pd.read_parquet(OUT_PARQUET)
        done_idxs = set(existing["row_idx"].astype(int).tolist())
        existing_rows = existing.to_dict("records")
        print(f"  resume: {len(done_idxs)} already done")

    todo = ra_fr[~ra_fr["row_idx"].isin(done_idxs)].copy()
    print(f"  todo: {len(todo)}")

    symbols_needed = sorted(ra_fr["contract"].dropna().unique().tolist())
    bars_df = pd.read_parquet(BARS_PATH, columns=["open", "high", "low", "close", "volume", "symbol"])
    bars_df = bars_df[bars_df["symbol"].isin(symbols_needed)].copy()
    bars_by_sym = {}
    for sym in symbols_needed:
        sub = bars_df[bars_df["symbol"] == sym].drop(columns=["symbol"]).sort_index()
        bars_by_sym[sym] = sub
    del bars_df
    gc.collect()
    print(f"  bars indexed for {len(bars_by_sym)} symbols, free={ram_mb()} MB")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=CTX_BARS, device="cpu")
    print(f"  model loaded, free={ram_mb()} MB")

    rows = list(existing_rows)
    skipped = 0
    t_start = time.time()
    aborted = False

    for batch_start in range(0, len(todo), args.batch):
        free = ram_mb()
        if free < RAM_FLOOR_MB:
            print(f"  pre-batch free={free} MB < {RAM_FLOOR_MB}; pausing 5s")
            time.sleep(5)
            gc.collect()
            free = ram_mb()
            if free < RAM_FLOOR_MB:
                print(f"  still low; ABORT after {len(rows)-len(existing_rows)} processed in this run")
                aborted = True
                break
        batch = todo.iloc[batch_start: batch_start + args.batch]
        for _, row in batch.iterrows():
            try:
                contract = row["contract"]
                ts = row["ts"]
                entry_price = float(row["entry_price"])
                side = row.get("side", None)
                bars = bars_by_sym.get(contract)
                if bars is None or len(bars) == 0:
                    skipped += 1
                    continue
                window = get_window(bars, ts, CTX_BARS)
                if window is None:
                    skipped += 1
                    continue
                df_win = normalize_bars(window)
                if len(df_win) < 100:
                    skipped += 1
                    continue
                x_ts = pd.Series(df_win.index)
                y_ts = build_y_timestamps(df_win.index[-1], PRED_LEN)
                t0 = time.time()
                with torch.no_grad():
                    pred_df = predictor.predict(
                        df=df_win,
                        x_timestamp=x_ts,
                        y_timestamp=y_ts,
                        pred_len=PRED_LEN,
                        T=1.0,
                        top_p=0.9,
                        sample_count=args.sample_count,
                        verbose=False,
                    )
                inf_time = time.time() - t0
                feats = kronos_features(pred_df, entry_price, side, inf_time)
                feats["row_idx"] = int(row["row_idx"])
                feats["ts"] = ts
                feats["contract"] = contract
                feats["entry_price"] = entry_price
                feats["side"] = side
                feats["is_winner"] = int(row["is_winner"])
                feats["net_pnl_after_haircut"] = float(row["net_pnl_after_haircut"])
                rows.append(feats)
                del pred_df
            except Exception as e:
                skipped += 1
                print(f"    ! exc idx={row['row_idx']}: {e}")

        gc.collect()
        try:
            pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
        except Exception as e:
            print(f"  ! checkpoint failed: {e}")
        free = ram_mb()
        elapsed = time.time() - t_start
        nproc = len(rows) - len(existing_rows)
        per = (elapsed / max(1, nproc)) if nproc else 0.0
        remaining = (len(todo) - (batch_start + len(batch))) * per
        print(
            f"  end batch start={batch_start} cum_proc={nproc} skipped={skipped} "
            f"free={free} MB used={used_mb()} MB elapsed={elapsed/60:.1f}m per={per:.2f}s ETA~{remaining/60:.1f}m"
        )

    print(f"\nDone. total rows in out: {len(rows)}, skipped this run: {skipped}, aborted={aborted}")
    pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
