#!/usr/bin/env python3
"""Full-window per-year stability check for H9 Gap-Fade rule.

Reads daily_table.parquet from artifacts/h9_gapfade_ml/, applies the rule
WITHOUT ML overlay across the full 2011-2026 window, and reports PnL by
calendar year and by political-era period."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "h9_gapfade_ml"
DOLLAR_PER_PT_MES = 5.0
COMMISSION = 1.50
SIZE = 10
GAP_THRESHOLD = 0.005
TP_PCT = 0.0030
SL_PCT = 0.0040


def label_period(ts):
    if ts < pd.Timestamp("2017-01-20", tz=ts.tz):
        return "pre_t1"
    if ts < pd.Timestamp("2021-01-20", tz=ts.tz):
        return "trump1"
    if ts < pd.Timestamp("2025-01-20", tz=ts.tz):
        return "biden"
    return "trump2"


def main():
    daily = pd.read_parquet(ART / "daily_table.parquet")
    # Build trades: every day a signal fires
    trades = []
    for ts, row in daily.iterrows():
        if row["fire_long"]:
            side, outcome, pnl_pts = "LONG", int(row["long_lbl"]), float(row["long_pnl_pts"])
        elif row["fire_short"]:
            side, outcome, pnl_pts = "SHORT", int(row["short_lbl"]), float(row["short_pnl_pts"])
        else:
            continue
        gross = pnl_pts * SIZE * DOLLAR_PER_PT_MES
        net = gross - COMMISSION * SIZE
        trades.append({
            "ts": ts, "side": side, "gap_pct": row["gap_pct"],
            "outcome": outcome, "pnl_pts": pnl_pts,
            "pnl_dollars_net": net, "year": ts.year,
            "period": label_period(ts),
        })
    tr = pd.DataFrame(trades)
    print(f"=== FULL-WINDOW backtest (2011-01 → 2026-04, RULE-ONLY) ===")
    print(f"Total trades: {len(tr)}")
    print(f"Total PnL:    ${tr['pnl_dollars_net'].sum():,.0f}")
    print(f"WR:           {(tr['pnl_dollars_net']>0).mean()*100:.1f}%")
    print(f"Avg trade:    ${tr['pnl_dollars_net'].mean():,.2f}")
    print(f"Avg win:      ${tr.loc[tr['pnl_dollars_net']>0,'pnl_dollars_net'].mean():,.2f}")
    print(f"Avg loss:     ${tr.loc[tr['pnl_dollars_net']<0,'pnl_dollars_net'].mean():,.2f}")
    cum = tr['pnl_dollars_net'].cumsum()
    dd = (cum - cum.cummax()).min()
    print(f"Max DD:       ${abs(dd):,.0f}")
    print()
    print("=== By calendar year ===")
    by_year = tr.groupby("year").agg(
        n_trades=("pnl_dollars_net", "count"),
        net_pnl=("pnl_dollars_net", "sum"),
        wr=("pnl_dollars_net", lambda s: (s > 0).mean() * 100),
    )
    print(by_year.to_string())
    print()
    print("=== By political era ===")
    by_period = tr.groupby("period").agg(
        n_trades=("pnl_dollars_net", "count"),
        net_pnl=("pnl_dollars_net", "sum"),
        wr=("pnl_dollars_net", lambda s: (s > 0).mean() * 100),
    )
    print(by_period.to_string())
    print()
    print("=== By side ===")
    by_side = tr.groupby("side").agg(
        n_trades=("pnl_dollars_net", "count"),
        net_pnl=("pnl_dollars_net", "sum"),
        wr=("pnl_dollars_net", lambda s: (s > 0).mean() * 100),
    )
    print(by_side.to_string())
    print()
    # Save
    tr.to_csv(ART / "per_trade_full_window.csv", index=False)
    full_summary = {
        "total_trades": int(len(tr)),
        "total_pnl_dollars": float(tr["pnl_dollars_net"].sum()),
        "win_rate_pct": float((tr["pnl_dollars_net"] > 0).mean() * 100),
        "avg_win_dollars": float(tr.loc[tr['pnl_dollars_net']>0, 'pnl_dollars_net'].mean()),
        "avg_loss_dollars": float(tr.loc[tr['pnl_dollars_net']<0, 'pnl_dollars_net'].mean()),
        "max_drawdown_dollars": float(abs(dd)),
        "by_year": by_year.to_dict(),
        "by_period": by_period.to_dict(),
        "by_side": by_side.to_dict(),
        "config": {"size_mes": SIZE, "gap_threshold": GAP_THRESHOLD,
                   "tp_pct": TP_PCT, "sl_pct": SL_PCT,
                   "dollar_per_pt": DOLLAR_PER_PT_MES,
                   "commission_per_round_trip": COMMISSION},
    }
    (ART / "full_window_summary.json").write_text(
        json.dumps(full_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
