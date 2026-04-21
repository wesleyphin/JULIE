"""Train per-strategy gate models.

Reads existing 2025 replay closed_trades, splits by strategy, computes
features per trade using REAL OHLCV from the parquet, trains one
GradientBoostingClassifier per strategy that has enough data.

Output: artifacts/signal_gate_2025/model_<strategy>.joblib

Currently trainable:
  - DynamicEngine3 (already shipped as model_de3.joblib — re-trains for parity)
  - RegimeAdaptive (587 trades across 2025)

Not yet trainable (no live trade data):
  - AetherFlow (0 trades — needs separate backtest)
  - MLPhysics (0 trades)
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/wes/Downloads/JULIE001")
PARQUET = Path("/Users/wes/Downloads/es_master_outrights-2.parquet")
OUT_DIR = ROOT / "artifacts" / "signal_gate_2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS  # noqa: E402
from regime_classifier import RegimeClassifier, WINDOW_BARS as _REG_WINDOW_BARS  # noqa: E402

# Front-month roll map for picking active contract per timestamp
ROLL_MAP = [
    (pd.Timestamp("2025-01-01", tz="America/New_York"), "ESH5"),
    (pd.Timestamp("2025-03-17", tz="America/New_York"), "ESM5"),
    (pd.Timestamp("2025-06-16", tz="America/New_York"), "ESU5"),
    (pd.Timestamp("2025-09-15", tz="America/New_York"), "ESZ5"),
    (pd.Timestamp("2025-12-15", tz="America/New_York"), "ESH6"),
    (pd.Timestamp("2026-03-16", tz="America/New_York"), "ESM6"),
]


def active_symbol(ts) -> str:
    if ts.tzinfo is None:
        from zoneinfo import ZoneInfo
        ts = ts.tz_localize("America/New_York")
    best = ROLL_MAP[0][1]
    for d, s in ROLL_MAP:
        if ts >= d:
            best = s
        else:
            break
    return best


# Numeric features (real OHLCV available, so use the full set)
NUMERIC_FEATURES = [
    "de3_entry_ret1_atr",
    "de3_entry_body_pos1",
    "de3_entry_body1_ratio",
    "de3_entry_close_pos1",
    "de3_entry_lower_wick_ratio",
    "de3_entry_upper_wick_ratio",
    "de3_entry_down3",
    "de3_entry_flips5",
    "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20",
    "de3_entry_atr14",
    # Trend-alignment numeric (derived below in compute_features_for_trades):
    #   +1 when signal_side agrees with the sign of ret1 (momentum-aligned),
    #   -1 when opposed, 0 when ambiguous. Lets G distinguish "high ATR +
    #   with trend = winner" from "high ATR + against trend = loser."
    "trend_align_ret1",
]
CATEGORICAL_FEATURES = ["side", "session", "regime", "mkt_regime"]
# regime     = AF manifold label (CHOP_SPIRAL | DISPERSED | TREND_GEODESIC | ROTATIONAL_TURBULENCE)
#              — only populated on AF trades; empty for DE3/RA
# mkt_regime = global bot regime classifier (neutral | whipsaw | calm_trend)
#              — computed per-trade from 120 bars of recent closes; populated
#              for ALL strategies. Matches what LFG/CB see in live.
ORDINAL_FEATURES = ["et_hour"]

# Regime values we expect in AF trades (from aetherflow_strategy):
REGIME_VALUES = ["DISPERSED", "TREND_GEODESIC", "CHOP_SPIRAL", "ROTATIONAL_TURBULENCE", ""]
MKT_REGIME_VALUES = ["neutral", "whipsaw", "calm_trend", ""]


def _mkt_regime_for(master_df: pd.DataFrame, symbol: str,
                    end_ts: pd.Timestamp) -> str:
    """Replay the bot's RegimeClassifier on 120 bars up to end_ts and return
    the current label. Stateless helper — builds a fresh classifier, feeds
    bars in order. Empty string if not enough data."""
    try:
        start = pd.Timestamp(end_ts).tz_convert("UTC") - pd.Timedelta(minutes=_REG_WINDOW_BARS * 2)
        end_utc = pd.Timestamp(end_ts).tz_convert("UTC")
    except Exception:
        return ""
    sub = master_df.loc[(master_df.index >= start) & (master_df.index <= end_utc) &
                        (master_df["symbol"] == symbol), "close"]
    if len(sub) < _REG_WINDOW_BARS:
        return ""
    clf = RegimeClassifier()
    # Feed closes in chronological order
    last = "warmup"
    for ts, c in sub.iloc[-_REG_WINDOW_BARS * 2:].items():
        try:
            r = clf.update(ts, float(c))
            if r:
                last = r
        except Exception:
            continue
    return "" if last == "warmup" else str(last)


def _session_of(h):
    if 18 <= h or h < 3: return "ASIA"
    if 3 <= h < 7: return "LONDON"
    if 7 <= h < 9: return "NY_PRE"
    if 9 <= h < 16: return "NY"
    return "POST"


def collect_strategy_trades(target_strategy: str, extra_globs=None):
    """Walk every existing 2025 replay folder + any extra glob locations,
    return list of trades for the specified strategy with deduplication on
    (entry_time, side, entry_price).

    extra_globs: list of (glob_pattern, source_kind) tuples where source_kind is
      - 'closed_trades' = expects f/closed_trades.json (live-replay style)
      - 'backtest_json' = expects f to be a JSON with "trade_log" array
      - 'backtest_dir'  = expects f/backtest_*.json files w/ trade_log
    """
    base = ROOT / "backtest_reports" / "full_live_replay"
    seen = set()
    out = []
    sources = []
    # Default: 2025_* live-replay folders
    for f in sorted(base.glob("2025_*")):
        sources.append((f / "closed_trades.json", "closed_trades"))
    # Also include outrageous_* folders
    for f in sorted(base.glob("outrageous_*")):
        sources.append((f / "closed_trades.json", "closed_trades"))
    # Also include 2026 replays
    for f in sorted(base.glob("2026_*")):
        sources.append((f / "closed_trades.json", "closed_trades"))
    # Replay loops
    for f in sorted((ROOT / "backtest_reports").glob("replay_*/live_loop_*")):
        sources.append((f / "closed_trades.json", "closed_trades"))
    # Fast AF replay outputs (closed_trades.json shape)
    # Use Kalshi-passed subset if present (user's directive), else fall back to raw
    for f in sorted((ROOT / "backtest_reports").glob("af_fast_replay/*")):
        if f.is_dir():
            kalshi_passed = f / "closed_trades_kalshi_passed.json"
            raw = f / "closed_trades.json"
            chosen = kalshi_passed if kalshi_passed.exists() else raw
            sources.append((chosen, "closed_trades"))
    # Plus any extras
    for pat, kind in (extra_globs or []):
        for p in sorted(Path(pat[0] if isinstance(pat, tuple) else pat).parent.glob(Path(pat).name) if "*" in str(pat) else [Path(pat)]):
            sources.append((p, kind))

    def _accept(t):
        s = str(t.get("strategy", "")).strip()
        if s != target_strategy and s != f"{target_strategy}Strategy":
            return
        key = (str(t.get("entry_time")), t.get("side"), float(t.get("entry_price") or 0))
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    for path, kind in sources:
        if not path.exists():
            continue
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if kind == "closed_trades":
            if isinstance(data, list):
                for t in data:
                    _accept(t)
        elif kind == "backtest_json":
            for t in (data.get("trade_log") or []):
                _accept(t)
    return out


def collect_af_backtest_trades(target_strategy: str, dirs):
    """Pull trade_log entries from AetherFlow-only backtest output JSONs.

    Args:
        target_strategy: e.g. 'AetherFlow'
        dirs: list of directory paths under backtest_reports/ to scan
    """
    seen = set()
    out = []
    for d in dirs:
        p = Path(d) if Path(d).is_absolute() else (ROOT / "backtest_reports" / d)
        if not p.is_dir():
            continue
        for fp in sorted(p.glob("backtest_*.json")):
            # skip _monte_carlo, _baseline, _gemini sub-reports
            if any(x in fp.name for x in ("_monte_carlo", "_baseline_comparison", "_gemini_recommendation")):
                continue
            try:
                data = json.load(open(fp))
            except Exception:
                continue
            for t in (data.get("trade_log") or []):
                s = str(t.get("strategy", "")).strip()
                if s != target_strategy and s != f"{target_strategy}Strategy":
                    continue
                key = (str(t.get("entry_time")), t.get("side"), float(t.get("entry_price") or 0))
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
    return out


def compute_features_for_trades(trades, master_df):
    """For each trade, look up REAL OHLCV bars from master_df at entry_time
    and compute features. Returns a DataFrame ready for training."""
    rows = []
    feature_cache = {}  # (symbol, day) → feature DataFrame
    for t in trades:
        try:
            from zoneinfo import ZoneInfo
            et = datetime.fromisoformat(t["entry_time"]).astimezone(ZoneInfo("America/New_York"))
        except Exception:
            continue
        symbol = active_symbol(et)
        # Pull a 4-hour window of bars before entry
        start = pd.Timestamp(et).tz_convert("UTC") - pd.Timedelta(hours=4)
        end = pd.Timestamp(et).tz_convert("UTC")
        cache_key = (symbol, et.replace(hour=0, minute=0, second=0))
        if cache_key in feature_cache:
            feats_day = feature_cache[cache_key]
        else:
            sub = master_df.loc[(master_df.index >= start - pd.Timedelta(hours=2)) &
                                (master_df.index <= end + pd.Timedelta(hours=4)) &
                                (master_df["symbol"] == symbol),
                                ["open", "high", "low", "close", "volume"]]
            if len(sub) < 50:
                continue
            feats_day = _compute_feature_frame(sub)
            feature_cache[cache_key] = feats_day
        # Find the bar at or before et
        try:
            idx = feats_day.index.searchsorted(end)
        except Exception:
            continue
        if idx <= 0 or idx > len(feats_day):
            continue
        feat_row = feats_day.iloc[idx - 1]
        if feat_row.isna().all():
            continue
        pnl = float(t.get("pnl_dollars", 0))
        size = int(t.get("size", 1) or 1)
        per_contract = pnl / size if size > 0 else pnl
        side_str = str(t.get("side", "")).upper()
        # Trend alignment: +1 if signal side agrees with ret1 sign, else -1; 0 when ambiguous.
        ret1_raw = feat_row.get("de3_entry_ret1_atr", 0.0)
        try:
            ret1 = float(ret1_raw)
            if not np.isfinite(ret1):
                ret1 = 0.0
        except Exception:
            ret1 = 0.0
        if ret1 > 0:
            trend_align = 1.0 if side_str == "LONG" else -1.0
        elif ret1 < 0:
            trend_align = 1.0 if side_str == "SHORT" else -1.0
        else:
            trend_align = 0.0
        # AF manifold regime: prefer explicit 'regime' field (AF fast-replay
        # format), fall back to 'aetherflow_regime' (live bot format), else
        # empty (DE3/RA trades don't carry it).
        regime_raw = t.get("regime") or t.get("aetherflow_regime") or ""
        regime = str(regime_raw or "").strip().upper()
        if regime and regime not in {"DISPERSED", "TREND_GEODESIC", "CHOP_SPIRAL", "ROTATIONAL_TURBULENCE"}:
            regime = ""  # unknown value → empty one-hot
        # Global bot market regime (neutral / whipsaw / calm_trend) — compute
        # by replaying the RegimeClassifier on 120 bars up to entry.
        try:
            mkt_regime = _mkt_regime_for(master_df, symbol, pd.Timestamp(et))
        except Exception:
            mkt_regime = ""
        row = {
            "entry_time": et.isoformat(),
            "et_hour": et.hour,
            "session": _session_of(et.hour),
            "side": side_str,
            "regime": regime,
            "mkt_regime": mkt_regime,
            "size": size,
            "entry_price": float(t.get("entry_price", 0)),
            "pnl_dollars": pnl,
            "per_contract_pnl": per_contract,
            "win": 1 if pnl > 0 else 0,
            "big_loss": 1 if pnl <= -100 else 0,
            "big_loss_per_contract": 1 if per_contract <= -45 else 0,  # ~10pt SL on 1 contract
            "sub_strategy": str(t.get("sub_strategy", "")),
            "trend_align_ret1": trend_align,
        }
        for c in ENTRY_SHAPE_COLUMNS:
            row[c] = float(feat_row.get(c, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def assemble_X(df, cat_maps=None):
    updated = dict(cat_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        if known is None:
            known = sorted(df[col].dropna().unique().tolist())
            updated[col] = known
        encoded = pd.DataFrame({f"{col}__{v}": (df[col] == v).astype(int) for v in known}, index=df.index)
        parts.append(encoded)
    parts.append(df[ORDINAL_FEATURES].astype(float))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def train_one(strategy: str, target_col: str, master_df: pd.DataFrame,
              extra_af_dirs=None):
    print(f"\n{'='*78}\nTRAINING {strategy}  (target={target_col})\n{'='*78}")
    trades = collect_strategy_trades(strategy)
    if extra_af_dirs:
        af_trades = collect_af_backtest_trades(strategy, extra_af_dirs)
        print(f"  +{len(af_trades)} {strategy} trades from AF backtest dirs")
        trades.extend(af_trades)
    print(f"  collected {len(trades)} {strategy} trades total")
    if len(trades) < 50:
        print(f"  [SKIP] insufficient trade data (need ≥50, have {len(trades)})")
        return None

    df = compute_features_for_trades(trades, master_df)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES + [target_col])
    df = df.reset_index(drop=True)
    print(f"  {len(df)} feature rows after dropping NaN")
    if len(df) < 50:
        print(f"  [SKIP] too few valid rows after feature extraction")
        return None

    print(f"  win rate: {df['win'].mean():.1%}")
    print(f"  big_loss rate: {df['big_loss'].mean():.1%}")
    print(f"  per-contract big_loss rate: {df['big_loss_per_contract'].mean():.1%}")
    print(f"  total P&L in training: ${df['pnl_dollars'].sum():+.2f}")

    X, cat_maps = assemble_X(df)
    y = df[target_col].astype(int).values
    print(f"  features: {X.shape[1]}  pos rate: {y.mean():.1%}")

    if y.sum() < 10 or (1 - y).sum() < 10:
        print(f"  [SKIP] target imbalance too severe (pos={y.sum()}, neg={(1-y).sum()})")
        return None

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    print(f"  5-fold CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Temporal holdout for threshold tuning
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    X_sorted, _ = assemble_X(df_sorted, cat_maps=cat_maps)
    y_sorted = df_sorted[target_col].astype(int).values
    split = int(0.85 * len(df_sorted))
    if len(df_sorted) - split < 20:
        split = max(1, len(df_sorted) - 20)
    X_tr, X_te = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_tr, y_te = y_sorted[:split], y_sorted[split:]
    df_te = df_sorted.iloc[split:]

    clf_te = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    clf_te.fit(X_tr, y_tr)
    p_te = clf_te.predict_proba(X_te)[:, 1]
    if len(set(y_te)) >= 2:
        tail_auc = roc_auc_score(y_te, p_te)
    else:
        tail_auc = float('nan')
    print(f"  temporal-tail AUC (last 15%): {tail_auc:.3f}")

    # EV-best threshold on holdout — but clamped to a minimum floor.
    # The raw auto-tuner can pick thr=0.20 on a small holdout, which then
    # over-vetoes on trend days (we saw G veto winners during 2026-04-21's
    # aggressive NY drop because it flagged high-ATR bars as danger even
    # when they were trend-continuation wins). Walk-forward across the full
    # 13.5-month set agrees that 0.275 is a better operating point.
    # Search the top of [floor, 0.80) and pick the best delta there.
    THR_FLOOR = 0.275
    best_thr, best_delta = THR_FLOOR, -1e9
    base = df_te["pnl_dollars"].sum()
    for thr in np.arange(THR_FLOOR, 0.80, 0.025):
        veto = p_te >= thr
        kept = df_te.loc[~veto, "pnl_dollars"].sum()
        if kept - base > best_delta:
            best_delta = kept - base
            best_thr = float(thr)
    print(f"  EV-best veto threshold (floor={THR_FLOOR}): {best_thr:.3f}  "
          f"(held-out tail delta: ${best_delta:+.2f})")

    # Retrain on full data
    final = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    final.fit(X, y)

    importances = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])
    print(f"  top 5 features:")
    for n, imp in importances[:5]:
        print(f"    {n:<32} {imp:.4f}")

    # Filename family mapping: runtime (signal_gate_2025._STRATEGY_MODEL_MAP)
    # expects de3→model_de3.joblib, so normalize the strategy name to match.
    _family_map = {
        "dynamicengine3": "de3",
        "aetherflow": "aetherflow",
        "regimeadaptive": "regimeadaptive",
        "mlphysics": "mlphysics",
    }
    family = _family_map.get(strategy.lower(), strategy.lower())
    out_path = OUT_DIR / f"model_{family}.joblib"
    joblib.dump({
        "model": final,
        "model_kind": "GBT_d3_per_strategy",
        "target": target_col,
        "veto_threshold": best_thr,
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_date_utc": datetime.now().isoformat() + "Z",
        "training_rows": int(len(df)),
        "training_date_range": [df["entry_time"].min(), df["entry_time"].max()],
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "tail_auc": float(tail_auc),
        "tail_delta": float(best_delta),
        "tail_baseline_pnl": float(base),
        "strategy": strategy,
        "notes": (
            f"Per-strategy G model for {strategy}, target={target_col}. "
            f"Trained on {len(df)} trades from existing 2025 replays. "
            f"Features computed from REAL OHLCV (parquet)."
        ),
    }, out_path)
    print(f"  [write] {out_path}  ({out_path.stat().st_size // 1024} KB)")
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+",
                   default=["RegimeAdaptive", "AetherFlow"],
                   help="Strategies to train (default: RegimeAdaptive AetherFlow)")
    p.add_argument("--af-backtest-dirs", nargs="*", default=["af_smoke_apr2025", "af_full_2025_2026"],
                   help="Extra dirs under backtest_reports/ to scan for AetherFlow trade_log")
    args = p.parse_args()

    print(f"[load] {PARQUET}")
    master = pd.read_parquet(PARQUET)
    master = master[master.index >= "2025-01-01"]
    print(f"  2025+ rows: {len(master):,}")

    for strat in args.strategies:
        extras = args.af_backtest_dirs if strat.lower().startswith("aether") else None
        train_one(strat, "big_loss", master, extra_af_dirs=extras)


if __name__ == "__main__":
    main()
