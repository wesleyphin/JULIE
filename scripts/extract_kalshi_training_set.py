#!/usr/bin/env python3
"""Extract a Kalshi training dataset from a replay log."""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/regime_ml"))
from _common import load_continuous_bars

RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_BAR = re.compile(r"Bar: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price:")
RE_KALSHI_VIEW = re.compile(
    r"\[KALSHI_ENTRY_VIEW\].*?strategy=(?P<strategy>\S+) \| side=(?P<side>LONG|SHORT) "
    r"\| entry_price=(?P<entry>[-\d.]+) \| role=(?P<role>\S+) "
    r"\| decision=(?P<decision>PASS|BLOCK) \| entry_probability=(?P<ep>[-\d.]+) "
    r"\| probe_price=(?P<pp>[-\d.]+) \| probe_probability=(?P<pprob>[-\d.]+) "
    r"\| momentum_delta=(?P<md>[-\d.]+) \| momentum_retention=(?P<mr>[-\d.]+) "
    r"\| support_score=(?P<ss>[-\d.]+) \| threshold=(?P<thr>[-\d.]+)"
)


def parse_log(log_path):
    rows = []
    last_bar_mts = None
    with log_path.open(errors="ignore") as f:
        for line in f:
            m = RE_HEADER.match(line)
            if not m: continue
            bm = RE_BAR.search(line)
            if bm: last_bar_mts = bm.group(1); continue
            ev = RE_KALSHI_VIEW.search(line)
            if not ev or last_bar_mts is None: continue
            rows.append({
                "market_ts": last_bar_mts, "strategy": ev.group("strategy"),
                "side": ev.group("side"), "entry_price": float(ev.group("entry")),
                "role": ev.group("role"), "rule_decision": ev.group("decision"),
                "k_entry_probability": float(ev.group("ep")),
                "k_probe_price": float(ev.group("pp")),
                "k_probe_probability": float(ev.group("pprob")),
                "k_momentum_delta": float(ev.group("md")),
                "k_momentum_retention": float(ev.group("mr")),
                "k_support_score": float(ev.group("ss")),
                "k_threshold": float(ev.group("thr")),
            })
    return rows


def label_with_forward_pnl(events, bars, horizons=(15, 30, 60), tp=6.0, sl=4.0, pt_value=5.0):
    bars_idx = {ts.strftime("%Y-%m-%d %H:%M"): i for i, ts in enumerate(bars.index)}
    H = bars["high"].to_numpy(); L = bars["low"].to_numpy(); C = bars["close"].to_numpy()
    n_bars = len(C)
    rows = []
    dropped = 0
    for e in events:
        key = e["market_ts"][:16]
        pos = bars_idx.get(key)
        if pos is None: dropped += 1; continue
        side_sign = 1 if e["side"] == "LONG" else -1
        entry = e["entry_price"]
        out = dict(e)
        for hz in horizons:
            end_idx = min(pos + 1 + hz, n_bars)
            if pos + 1 >= end_idx:
                out[f"forward_pnl_{hz}m"] = np.nan
                out[f"label_{hz}m"] = "no_data"
                continue
            hs = H[pos+1:end_idx]; ls = L[pos+1:end_idx]
            if side_sign > 0:
                tp_h = np.where(hs >= entry + tp)[0]
                sl_h = np.where(ls <= entry - sl)[0]
            else:
                tp_h = np.where(ls <= entry - tp)[0]
                sl_h = np.where(hs >= entry + sl)[0]
            tp_i = tp_h[0] if len(tp_h) else 1<<30
            sl_i = sl_h[0] if len(sl_h) else 1<<30
            if tp_i == 1<<30 and sl_i == 1<<30:
                last = C[end_idx-1]
                pnl = (last - entry) * side_sign * pt_value
            elif tp_i < sl_i: pnl = tp * pt_value
            else: pnl = -sl * pt_value
            out[f"forward_pnl_{hz}m"] = round(float(pnl), 2)
            margin = 15.0
            if pnl > margin: out[f"label_{hz}m"] = "pass"
            elif pnl < -margin: out[f"label_{hz}m"] = "block"
            else: out[f"label_{hz}m"] = "ambiguous"
        rows.append(out)
    return pd.DataFrame(rows), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}"); return 1
    print(f"parsing log: {log_path}")
    events = parse_log(log_path)
    print(f"  KALSHI_ENTRY_VIEW events: {len(events)}")
    if not events: print("no events"); return 1
    bars = load_continuous_bars(args.start, args.end)
    print(f"  bars: {len(bars):,}")
    df, dropped = label_with_forward_pnl(events, bars)
    print(f"  labeled: {len(df)}  dropped: {dropped}")
    if len(df) > 0:
        for h in [15, 30, 60]:
            print(f"  label_{h}m: {dict(df[f'label_{h}m'].value_counts())}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\n[write] {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
