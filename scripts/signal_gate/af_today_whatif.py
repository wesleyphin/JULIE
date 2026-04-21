"""For each of today's 56 AF signals, simulate the full pipeline:

  1. AF internal gate (hypothetically passed — we take the signal even though
     the live bot blocked all 56 below-threshold)
  2. Kalshi blocker (only active 12-16 ET; otherwise auto-pass)
  3. G gate (model_aetherflow.joblib at thr=0.275)
  4. Forward bracket simulation using today's bars (close-only, conservative)

Report per-signal PnL + what each layer would have done.

Bars today are log-only (close prices, no H/L). We synthesize bar ranges
from local close-to-close volatility to approximate TP/SL hits. Note: this
is CLOSE-FIDELITY (conservative vs reality), not true OHLC hit detection.
"""
from __future__ import annotations

import json, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))

from aetherflow_features import resolve_setup_params
from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS

NY = ZoneInfo("America/New_York")
LOG = ROOT / "topstep_live_bot.log"
POINT_VALUE = 5.0  # MES $5/pt
SIZE = 5            # AF default size per live config
TODAY = "2026-04-21"

# ---- Kalshi approximation ----
KALSHI_GATING_HOURS_ET = {12, 13, 14, 15, 16}
KALSHI_ENTRY_THRESHOLD = 0.45
KALSHI_DAILY = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"

# --------------- Log parsing ---------------

PAT_SIG = re.compile(
    r"^(?P<host_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),"
    r".*\[STRATEGY_SIGNAL\].*AetherFlow.*\| side=(?P<side>\w+) "
    r"\| price=(?P<price>[\d.]+) "
    r"\| tp_dist=(?P<tp>[\d.]+) "
    r"\| sl_dist=(?P<sl>[\d.]+) "
    r"\| status=(?P<status>\w+) "
    r"\| decision=(?P<decision>\w+) "
    r"\| reason=(?P<reason>\w+) "
    r"\| combo_key=(?P<combo>\w+) "
    r"\| vol_regime=(?P<regime>\w+) "
    r"\| gate_prob=(?P<prob>[\d.]+) "
    r"\| gate_threshold=(?P<thr>[\d.]+) "
    r"\| session_id=(?P<sess>\d+)"
)

PAT_BAR = re.compile(
    r"\[INFO\] Bar: (?P<bar_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET "
    r"\| Price: (?P<price>[\d.]+)"
)


def host_ts_to_et(host_str: str) -> datetime:
    """Host is PDT (UTC-7); ET = PDT + 3h."""
    hd = datetime.strptime(host_str, "%Y-%m-%d %H:%M:%S")
    from datetime import timedelta
    et_naive = hd + timedelta(hours=3)
    return et_naive.replace(tzinfo=NY)


def main():
    # --- Parse signals ---
    signals = []
    bars = []
    with LOG.open(errors="replace") as fh:
        for line in fh:
            if TODAY not in line:
                continue
            m_sig = PAT_SIG.search(line)
            if m_sig:
                d = m_sig.groupdict()
                # timestamp: use the embedded [YYYY-MM-DD HH:MM:SS] inside line if present, else host
                inner_ts_m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+\] \[STRATEGY_SIGNAL\]", line)
                if inner_ts_m:
                    d["et_ts"] = inner_ts_m.group(1)  # host-local; convert
                    d["et_dt"] = host_ts_to_et(d["et_ts"])
                else:
                    d["et_dt"] = host_ts_to_et(d["host_ts"])
                signals.append(d)
                continue
            m_bar = PAT_BAR.search(line)
            if m_bar:
                bar_ts = pd.Timestamp(m_bar.group("bar_ts"), tz=NY)
                bars.append((bar_ts, float(m_bar.group("price"))))

    # Dedupe bars on timestamp (log has 3x per minute due to retries)
    bar_df = pd.DataFrame(bars, columns=["ts", "price"]).drop_duplicates("ts").sort_values("ts")
    bar_df = bar_df[bar_df["ts"].dt.date == pd.Timestamp(TODAY).date()]
    bar_df = bar_df.set_index("ts")

    print(f"[parse] {len(signals)} AF signals, {len(bar_df)} unique bars today")
    if not signals or bar_df.empty:
        return

    # Synthesize OHLC from close series — H/L via local rolling abs-diff
    closes = bar_df["price"].astype(float)
    # Use a short-window max(abs diff) as half-range proxy
    roll_range = closes.diff().abs().rolling(5, min_periods=1).max().fillna(0.25) * 0.5
    bar_df["open"] = closes.shift(1).fillna(closes.iloc[0])
    bar_df["close"] = closes
    bar_df["high"] = np.maximum(bar_df["open"], closes) + roll_range
    bar_df["low"]  = np.minimum(bar_df["open"], closes) - roll_range
    bar_df["volume"] = np.nan

    # ATR14 for bracket sizing — use built-in feature frame
    feats_today = _compute_feature_frame(bar_df[["open","high","low","close","volume"]])

    # --- Kalshi snapshot (daily) ---
    kalshi_path = KALSHI_DAILY / f"{TODAY}.parquet"
    if kalshi_path.exists():
        kalshi_df = pd.read_parquet(kalshi_path)
        print(f"[kalshi] loaded {len(kalshi_df)} rows for {TODAY}")
    else:
        kalshi_df = None
        # try yesterday's snapshot as fallback
        yesterday = (pd.Timestamp(TODAY) - pd.Timedelta(days=1)).date().isoformat()
        ky = KALSHI_DAILY / f"{yesterday}.parquet"
        if ky.exists():
            kalshi_df = pd.read_parquet(ky)
            print(f"[kalshi] {TODAY} missing, using fallback from {yesterday}: {len(kalshi_df)} rows")

    # --- G gate ---
    g_model = joblib.load(ROOT / "artifacts" / "signal_gate_2025" / "model_aetherflow.joblib")
    g_thr = float(g_model["veto_threshold"])
    print(f"[G] model loaded, thr={g_thr}")

    def next_settlement_hour(et_hour: int):
        for h in (10, 11, 12, 13, 14, 15, 16):
            if h > et_hour:
                return h
        return None

    def kalshi_decision(et_hour, entry_price, side):
        if et_hour not in KALSHI_GATING_HOURS_ET:
            return ("auto_pass", None, "outside_gating_hours")
        if kalshi_df is None:
            return ("auto_pass", None, "no_data")
        next_set = next_settlement_hour(et_hour)
        if next_set is None:
            return ("auto_pass", None, "no_settlement")
        sub = kalshi_df[(kalshi_df["settlement_hour_et"] == next_set) &
                        (kalshi_df["event_date"] == TODAY)]
        if sub.empty:
            return ("auto_pass", None, "no_strike_data")
        sub = sub.copy()
        sub["dist"] = (sub["strike"] - entry_price).abs()
        row = sub.nsmallest(1, "dist").iloc[0]
        yes_prob = (float(row["high"]) + float(row["low"])) / 200.0
        aligned = yes_prob if side == "LONG" else 1.0 - yes_prob
        if aligned < KALSHI_ENTRY_THRESHOLD:
            return ("block", aligned, f"aligned_prob={aligned:.2f}<{KALSHI_ENTRY_THRESHOLD}")
        return ("pass", aligned, f"aligned_prob={aligned:.2f}")

    def simulate_bracket(entry_idx, side, tp_dist, sl_dist, horizon):
        """Close-only bracket sim, conservative."""
        if entry_idx + 1 >= len(bar_df):
            return None
        entry_price = float(bar_df.iloc[entry_idx + 1]["open"])
        if side == "LONG":
            tp = entry_price + tp_dist
            sl = entry_price - sl_dist
        else:
            tp = entry_price - tp_dist
            sl = entry_price + sl_dist
        end = min(len(bar_df), entry_idx + 1 + horizon)
        for j in range(entry_idx + 1, end):
            h = float(bar_df.iloc[j]["high"])
            l = float(bar_df.iloc[j]["low"])
            sl_hit = (side == "LONG" and l <= sl) or (side == "SHORT" and h >= sl)
            tp_hit = (side == "LONG" and h >= tp) or (side == "SHORT" and l <= tp)
            if sl_hit and tp_hit:
                return {"exit_idx": j, "pnl_points": -sl_dist, "source": "stop_pess",
                        "entry_price": entry_price, "exit_price": sl, "bars_held": j - entry_idx}
            if sl_hit:
                return {"exit_idx": j, "pnl_points": -sl_dist, "source": "stop",
                        "entry_price": entry_price, "exit_price": sl, "bars_held": j - entry_idx}
            if tp_hit:
                return {"exit_idx": j, "pnl_points": tp_dist, "source": "take",
                        "entry_price": entry_price, "exit_price": tp, "bars_held": j - entry_idx}
        # Time stop
        j = end - 1
        exit_price = float(bar_df.iloc[j]["close"])
        pts = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        return {"exit_idx": j, "pnl_points": pts, "source": "horizon",
                "entry_price": entry_price, "exit_price": exit_price, "bars_held": j - entry_idx}

    # --- Walk every AF signal ---
    results = []
    for sig in signals:
        et_dt = sig["et_dt"]
        side = sig["side"]
        combo = sig["combo"]
        # Find the entry bar in today's bar_df
        idx = int(bar_df.index.searchsorted(et_dt))
        if idx >= len(bar_df) - 1:
            continue
        # Resolve bracket from AF defaults + local ATR
        feat_row = feats_today.iloc[min(idx, len(feats_today) - 1)] if len(feats_today) > 0 else pd.Series()
        atr14 = float(feat_row.get("atr14", 2.5)) if len(feats_today) > 0 else 2.5
        if not np.isfinite(atr14) or atr14 <= 0:
            atr14 = 2.5
        params = resolve_setup_params(pd.Series({"setup_family": combo, "atr14": atr14}))
        tp_dist = float(params["tp_points"])
        sl_dist = float(params["sl_points"])
        horizon = int(params["horizon_bars"])

        sim = simulate_bracket(idx, side, tp_dist, sl_dist, horizon)
        if sim is None:
            continue
        gross_pts = sim["pnl_points"]
        pnl = gross_pts * POINT_VALUE * SIZE

        # Kalshi
        k_decision, k_prob, k_reason = kalshi_decision(et_dt.hour, sim["entry_price"], side)

        # G gate — build feature row using parquet/feats_today proxy
        feat_names = g_model["feature_names"]
        numeric = g_model["numeric_features"]
        cat_maps = g_model.get("categorical_maps", {})
        row = {c: 0.0 for c in feat_names}
        for c in numeric:
            v = feat_row.get(c, 0.0) if len(feats_today) > 0 else 0.0
            try:
                fv = float(v)
                if not np.isfinite(fv): fv = 0.0
            except: fv = 0.0
            if c in row: row[c] = fv
        # categorical: side + session
        def sess_of(h):
            if 18 <= h or h < 3: return "ASIA"
            if 3 <= h < 7: return "LONDON"
            if 7 <= h < 9: return "NY_PRE"
            if 9 <= h < 16: return "NY"
            return "POST"
        cat_vals = {"side": side.upper(), "session": sess_of(et_dt.hour)}
        for cc, kvs in cat_maps.items():
            val = cat_vals.get(cc, "")
            for kv in kvs:
                nm = f"{cc}__{kv}"
                if nm in row and val == kv: row[nm] = 1
        if "et_hour" in row:
            row["et_hour"] = float(et_dt.hour)
        X = np.array([[row[c] for c in feat_names]])
        p_big_loss = float(g_model["model"].predict_proba(X)[0, 1])
        g_veto = p_big_loss >= g_thr

        results.append({
            "time_et": et_dt.strftime("%H:%M"),
            "side": side,
            "combo": combo,
            "entry_price": sim["entry_price"],
            "exit_price": sim["exit_price"],
            "exit_source": sim["source"],
            "bars_held": sim["bars_held"],
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "pnl_points": gross_pts,
            "pnl_dollars": pnl,
            "af_prob": float(sig["prob"]),
            "af_thr": float(sig["thr"]),
            "af_pass": float(sig["prob"]) >= float(sig["thr"]),
            "kalshi_decision": k_decision,
            "kalshi_aligned_prob": k_prob,
            "kalshi_reason": k_reason,
            "g_p_big_loss": p_big_loss,
            "g_veto": g_veto,
            "regime": sig["regime"],
            "sess_id": sig["sess"],
            "would_take_all": True,  # hypothetical: AF loose
            "would_take_af_strict": float(sig["prob"]) >= float(sig["thr"]),
        })

    # --- Report ---
    print("\n" + "=" * 120)
    print(f"Per-signal simulation ({len(results)} signals forward-walked through today's bars)")
    print("=" * 120)
    hdr = f"{'time':<6} {'side':<5} {'combo':<20} {'entry':>8} {'exit':>8} {'src':<10} {'tp':>5} {'sl':>5} " \
          f"{'bars':>4} {'pnl$':>9} | AF {'prob':>5}/{'thr':>5} {'pass?':<5} | Kal {'dec':<8} {'ap':>5} | G {'pLoss':>5} {'veto':<4}"
    print(hdr)
    print("-" * 120)
    for r in results:
        afp = f"{r['af_prob']:.3f}"; aft = f"{r['af_thr']:.2f}"
        afpass = "Y" if r["af_pass"] else "N"
        kp = f"{r['kalshi_aligned_prob']:.2f}" if r["kalshi_aligned_prob"] is not None else "  —"
        gveto = "Y" if r["g_veto"] else "N"
        print(f"{r['time_et']:<6} {r['side']:<5} {r['combo']:<20} {r['entry_price']:>8.2f} {r['exit_price']:>8.2f} "
              f"{r['exit_source']:<10} {r['tp_dist']:>5.2f} {r['sl_dist']:>5.2f} {r['bars_held']:>4} "
              f"{r['pnl_dollars']:>+9.2f} | AF {afp}/{aft} {afpass:<5} | Kal {r['kalshi_decision']:<8} {kp} "
              f"| G {r['g_p_big_loss']:.3f} {gveto:<4}")

    # --- Scenario aggregates ---
    print("\n" + "=" * 120)
    print("PnL under different gating scenarios (all 56 AF signals forward-walked)")
    print("=" * 120)

    def agg(selector, label):
        picked = [r for r in results if selector(r)]
        if not picked:
            print(f"  {label:<50}  n=0")
            return
        pnl = sum(r["pnl_dollars"] for r in picked)
        wins = sum(1 for r in picked if r["pnl_dollars"] > 0)
        stops = sum(1 for r in picked if "stop" in r["exit_source"])
        takes = sum(1 for r in picked if r["exit_source"] == "take")
        horis = sum(1 for r in picked if r["exit_source"] == "horizon")
        print(f"  {label:<50}  n={len(picked):<3}  pnl=${pnl:>+9.2f}  "
              f"wr={wins/len(picked):.1%}  (stops={stops} takes={takes} horizon={horis})")

    agg(lambda r: True, "BASELINE: take every signal (hypothetical)")
    agg(lambda r: r["af_pass"], "AF strict gate only (live threshold)")
    agg(lambda r: r["kalshi_decision"] != "block", "Kalshi pass only (AF gate bypassed)")
    agg(lambda r: not r["g_veto"], "G gate pass only (AF gate bypassed)")
    agg(lambda r: r["af_pass"] and r["kalshi_decision"] != "block", "AF + Kalshi")
    agg(lambda r: r["af_pass"] and not r["g_veto"], "AF + G")
    agg(lambda r: r["kalshi_decision"] != "block" and not r["g_veto"], "Kalshi + G (AF bypassed)")
    agg(lambda r: r["af_pass"] and r["kalshi_decision"] != "block" and not r["g_veto"], "AF + Kalshi + G (FULL LIVE STACK)")

    # With loose AF (>= 0.50 instead of 0.55+)
    agg(lambda r: r["af_prob"] >= 0.50, "AF loose ≥0.50 (hypothetical threshold drop)")
    agg(lambda r: r["af_prob"] >= 0.50 and r["kalshi_decision"] != "block" and not r["g_veto"],
        "AF loose ≥0.50 + Kalshi + G")
    agg(lambda r: r["af_prob"] >= 0.45, "AF looser ≥0.45")
    agg(lambda r: r["af_prob"] >= 0.45 and r["kalshi_decision"] != "block" and not r["g_veto"],
        "AF looser ≥0.45 + Kalshi + G")

    # Export
    out = Path("/tmp/af_today_whatif.json")
    out.write_text(json.dumps(results, default=str, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
