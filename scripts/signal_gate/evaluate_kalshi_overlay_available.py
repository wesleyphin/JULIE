"""Evaluate the currently implemented Kalshi overlay on the broadest local data.

This script does two related things:

1. Entry-overlay replay on the *real* cached Kalshi daily ladders available in
   ``.tmp_upstream_julie/data/kalshi/kxinxu_2025_daily`` using the current
   ``kalshi_trade_overlay.build_trade_plan`` logic. Because we do not have the
   full live execution path here, this replay evaluates the entry-side effect
   only:
     - hard blocks -> trade removed
     - size trims  -> pnl scaled by trimmed_size / original_size
     - TP adjustment / trailing are reported, but not re-simulated

2. Coverage + rolling-origin summaries for the saved Kalshi ML datasets/artifacts:
     - model_kalshi_gate.joblib / kalshi_training_data.parquet
     - model_kalshi_tp_gate.joblib / kalshi_tp_training_data.parquet

Output:
  artifacts/signal_gate_2025/kalshi_overlay_available_eval_YYYYMMDD.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = ROOT / ".tmp_upstream_julie"
ART_DIR = ROOT / "artifacts" / "signal_gate_2025"
OUT_PATH = ART_DIR / f"kalshi_overlay_available_eval_{datetime.now().strftime('%Y%m%d')}.json"
PARQUET = ROOT / "es_master_outrights.parquet"
LOCAL_KALSHI_DAILY_DIR = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
UPSTREAM_KALSHI_DAILY_DIR = UPSTREAM_ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
DE3_RA_SOURCE = UPSTREAM_ROOT / "backtest_reports" / "full_live_replay" / "2026_jan_apr" / "closed_trades.json"
AF_SOURCE = UPSTREAM_ROOT / "backtest_reports" / "af_fast_replay" / "af_full_2025_2026" / "closed_trades.json"

sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from kalshi_trade_overlay import analyze_recent_price_action, build_trade_plan  # noqa: E402

NY = ZoneInfo("America/New_York")
KALSHI_GATING_HOURS_ET = {12, 13, 14, 15, 16}
TP_RGX = re.compile(r"_TP(?P<tp>\d+(?:\.\d+)?)")


def compute_dd(pnls: List[float]) -> float:
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for pnl in pnls:
        cum += float(pnl)
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return float(dd)


def evaluate_rows(rows: List[Dict[str, Any]], pnl_key: str = "realized_pnl_dol") -> Dict[str, Any]:
    pnls = [float(r.get(pnl_key, 0.0) or 0.0) for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": int(len(rows)),
        "pnl": float(sum(pnls)),
        "dd": compute_dd(pnls),
        "wr": float(wins / len(rows)) if rows else 0.0,
        "avg_trade": float(sum(pnls) / len(rows)) if rows else 0.0,
    }


def load_trades(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_strategy(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if raw.startswith("DynamicEngine3"):
        return "DynamicEngine3"
    if raw.startswith("RegimeAdaptive"):
        return "RegimeAdaptive"
    if raw.startswith("AetherFlow"):
        return "AetherFlow"
    return raw or "Unknown"


def parse_tp_dist(trade: Dict[str, Any]) -> Optional[float]:
    direct = trade.get("tp_dist")
    try:
        if direct is not None and float(direct) > 0:
            return float(direct)
    except Exception:
        pass
    sub = str(trade.get("sub_strategy", "") or "")
    m = TP_RGX.search(sub)
    if not m:
        return None
    try:
        return float(m.group("tp"))
    except Exception:
        return None


def load_market_df(start_et: pd.Timestamp, end_et: pd.Timestamp) -> pd.DataFrame:
    start_utc = start_et.tz_convert("UTC")
    end_utc = end_et.tz_convert("UTC")
    df = pd.read_parquet(PARQUET)
    df = df[(df.index >= start_utc) & (df.index <= end_utc)].copy()
    if "symbol" in df.columns and "volume" in df.columns:
        df = (
            df.sort_values("volume", ascending=False)
            .groupby(df.index)
            .first()
            .sort_index()
        )
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY)
    else:
        df.index = df.index.tz_convert(NY)
    return df.sort_index()


class HistoricalKalshiProvider:
    _SETTLEMENT_HOURS = [10, 11, 12, 13, 14, 15, 16]

    def __init__(self, daily_dirs: List[Path]):
        self.daily_dirs = [Path(d) for d in daily_dirs]
        self.enabled = True
        self.is_healthy = True
        self.basis_offset = 0.0
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._sentiment_history: List[tuple[pd.Timestamp, float]] = []
        self._context_time: Optional[pd.Timestamp] = None

    def set_context_time(self, ts_et: pd.Timestamp) -> None:
        self._context_time = pd.Timestamp(ts_et).tz_convert(NY)

    def es_to_spx(self, es_price: float) -> float:
        return float(es_price) - float(self.basis_offset)

    def spx_to_es(self, spx_price: float) -> float:
        return float(spx_price) + float(self.basis_offset)

    def active_settlement_hour_et(self, ref_time: Optional[datetime] = None, rollover_minute: int = 5) -> Optional[int]:
        if ref_time is None:
            if self._context_time is None:
                return None
            now = self._context_time.to_pydatetime()
        else:
            now = ref_time if ref_time.tzinfo is not None else ref_time.replace(tzinfo=NY)
            now = now.astimezone(NY)
        for hour in self._SETTLEMENT_HOURS:
            if hour > now.hour or (hour == now.hour and now.minute < int(rollover_minute)):
                return hour
        return None

    def _load_day(self, date_str: str) -> Optional[pd.DataFrame]:
        if date_str in self._cache:
            return self._cache[date_str]
        for daily_dir in self.daily_dirs:
            path = daily_dir / f"{date_str}.parquet"
            if path.exists():
                df = pd.read_parquet(path).copy()
                self._cache[date_str] = df
                return df
        self._cache[date_str] = None
        return None

    def _markets_for_context(self) -> List[Dict[str, Any]]:
        if self._context_time is None:
            return []
        date_str = self._context_time.date().isoformat()
        df = self._load_day(date_str)
        if df is None or df.empty:
            return []
        settlement_hour = self.active_settlement_hour_et(self._context_time.to_pydatetime(), rollover_minute=5)
        if settlement_hour is None:
            return []
        sub = df[
            (df["event_date"] == date_str)
            & (df["settlement_hour_et"].astype(int) == int(settlement_hour))
        ].copy()
        if sub.empty:
            return []
        markets: List[Dict[str, Any]] = []
        for _, row in sub.iterrows():
            hi = float(row.get("high") or 0.0)
            lo = float(row.get("low") or 0.0)
            prob = (hi + lo) / 200.0
            markets.append(
                {
                    "strike": float(row["strike"]),
                    "probability": float(prob),
                    "status": str(row.get("status", "") or ""),
                    "open_interest": int(row.get("open_interest") or 0),
                    "daily_volume": int(row.get("daily_volume") or 0),
                    "strike_es": float(row["strike"]),
                }
            )
        markets.sort(key=lambda r: r["strike"])
        return markets

    def get_relative_markets_for_ui(self, es_prices: Optional[List[float]] = None, window_size: int = 30) -> List[Dict]:
        markets = self._markets_for_context()
        if not markets:
            return []
        if len(markets) <= int(window_size):
            return markets
        ref_prices = [self.es_to_spx(float(p)) for p in (es_prices or []) if p is not None]
        if not ref_prices:
            midpoint = len(markets) // 2
            start = max(0, midpoint - (window_size // 2))
            end = min(len(markets), start + window_size)
            return markets[max(0, end - window_size):end]
        nearest = [
            min(range(len(markets)), key=lambda idx: abs(float(markets[idx]["strike"]) - ref))
            for ref in ref_prices
        ]
        lo = min(nearest)
        hi = max(nearest)
        span = (hi - lo) + 1
        if span >= window_size:
            midpoint = (lo + hi) // 2
            start = max(0, midpoint - (window_size // 2))
            end = min(len(markets), start + window_size)
            return markets[max(0, end - window_size):end]
        padding = window_size - span
        start = max(0, lo - (padding // 2))
        end = min(len(markets), hi + 1 + (padding - (padding // 2)))
        if (end - start) < window_size:
            if start == 0:
                end = min(len(markets), window_size)
            else:
                start = max(0, end - window_size)
        return markets[start:end]

    def get_probability(self, strike_price: float) -> Optional[float]:
        markets = self._markets_for_context()
        if not markets:
            return None
        exact = [m for m in markets if abs(float(m["strike"]) - float(strike_price)) < 0.01]
        if exact:
            return float(exact[0]["probability"])
        below = [m for m in markets if float(m["strike"]) <= float(strike_price)]
        above = [m for m in markets if float(m["strike"]) > float(strike_price)]
        if below and above:
            lo = below[-1]
            hi = above[0]
            frac = (float(strike_price) - float(lo["strike"])) / max(1e-9, float(hi["strike"]) - float(lo["strike"]))
            return float(lo["probability"]) * (1.0 - frac) + float(hi["probability"]) * frac
        if below:
            return float(below[-1]["probability"])
        if above:
            return float(above[0]["probability"])
        return None

    def get_implied_level(self) -> Optional[float]:
        markets = self._markets_for_context()
        if len(markets) < 2:
            return None
        for idx in range(len(markets) - 1):
            p1 = float(markets[idx]["probability"])
            p2 = float(markets[idx + 1]["probability"])
            if p1 >= 0.5 > p2:
                s1 = float(markets[idx]["strike"])
                s2 = float(markets[idx + 1]["strike"])
                frac = (0.5 - p1) / max(1e-9, p2 - p1)
                return float(s1 + frac * (s2 - s1))
        return None

    def get_sentiment(self, es_price: float) -> Dict[str, Any]:
        probability = self.get_probability(self.es_to_spx(es_price))
        implied_level_spx = self.get_implied_level()
        implied_level_es = self.spx_to_es(implied_level_spx) if implied_level_spx is not None else None
        distance_es = implied_level_es - float(es_price) if implied_level_es is not None else None
        return {
            "probability": round(float(probability), 4) if probability is not None else None,
            "distance_es": round(float(distance_es), 2) if distance_es is not None else None,
            "implied_level_es": round(float(implied_level_es), 2) if implied_level_es is not None else None,
            "healthy": True,
        }

    def get_sentiment_momentum(self, es_price: float, lookback: int = 3) -> Optional[float]:
        probability = self.get_probability(self.es_to_spx(es_price))
        if probability is None or self._context_time is None:
            return None
        self._sentiment_history.append((self._context_time, float(probability)))
        self._sentiment_history = self._sentiment_history[-40:]
        if len(self._sentiment_history) < int(lookback) + 1:
            return None
        prior = self._sentiment_history[-(int(lookback) + 1)][1]
        return round(float(probability - prior), 4)


def collect_overlap_trades(min_date: str, max_date: str) -> List[Dict[str, Any]]:
    min_ts = pd.Timestamp(min_date, tz=NY)
    max_ts = pd.Timestamp(max_date, tz=NY) + pd.Timedelta(hours=23, minutes=59)
    trades: List[Dict[str, Any]] = []
    for raw in load_trades(DE3_RA_SOURCE):
        strategy = normalize_strategy(raw.get("strategy"))
        if strategy not in {"DynamicEngine3", "RegimeAdaptive"}:
            continue
        try:
            et = pd.Timestamp(raw["entry_time"]).tz_convert(NY)
        except Exception:
            continue
        if not (min_ts <= et <= max_ts):
            continue
        trades.append({**raw, "strategy": strategy})
    for raw in load_trades(AF_SOURCE):
        strategy = normalize_strategy(raw.get("strategy"))
        if strategy != "AetherFlow":
            continue
        try:
            et = pd.Timestamp(raw["entry_time"]).tz_convert(NY)
        except Exception:
            continue
        if not (min_ts <= et <= max_ts):
            continue
        trades.append({**raw, "strategy": strategy})
    trades.sort(key=lambda t: str(t.get("entry_time", "")))
    return trades


def summarize_dataset(path: Path, payload_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    payload = joblib.load(payload_path)
    out: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "payload": {
            "model_kind": payload.get("model_kind"),
            "training_rows": payload.get("training_rows"),
            "rolling_origin_mean_auc": payload.get("rolling_origin_mean_auc"),
            "pass_threshold": payload.get("pass_threshold"),
            "regressor_gate_threshold": payload.get("regressor_gate_threshold"),
            "gate_mode": payload.get("gate_mode"),
        },
    }
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        out["window"] = {
            "min": str(ts.min()),
            "max": str(ts.max()),
        }
    for col in ["source_dir", "source", "role", "regime", "side", "hit_tp", "label"]:
        if col in df.columns:
            counts = df[col].astype(str).value_counts().head(20).to_dict()
            out[f"{col}_counts"] = {str(k): int(v) for k, v in counts.items()}
    if "pnl_dollars" in df.columns:
        pnl = df["pnl_dollars"].astype(float)
        out["pnl"] = {
            "sum": float(pnl.sum()),
            "mean": float(pnl.mean()),
            "median": float(pnl.median()),
        }
    if "rolling_origin_cv" in payload:
        out["rolling_origin_cv_summary"] = {}
        cv = payload.get("rolling_origin_cv") or []
        if cv:
            if "rule_pnl" in cv[0]:
                out["rolling_origin_cv_summary"]["cumulative_rule_pnl"] = float(
                    sum(float(r.get("rule_pnl", 0.0) or 0.0) for r in cv)
                )
            if "reg0_delta" in cv[0]:
                out["rolling_origin_cv_summary"]["cumulative_reg0_delta"] = float(
                    sum(float(r.get("reg0_delta", 0.0) or 0.0) for r in cv)
                )
                out["rolling_origin_cv_summary"]["cumulative_clf50_delta"] = float(
                    sum(float(r.get("clf50_delta", 0.0) or 0.0) for r in cv)
                )
            if "gate_deltas" in cv[0]:
                gate_totals: Dict[str, float] = defaultdict(float)
                for row in cv:
                    for thr, stats in (row.get("gate_deltas") or {}).items():
                        gate_totals[str(thr)] += float((stats or {}).get("delta", 0.0) or 0.0)
                out["rolling_origin_cv_summary"]["cumulative_gate_deltas"] = dict(gate_totals)
    return out


def resolve_daily_dirs() -> List[Path]:
    dirs: List[Path] = []
    for candidate in [LOCAL_KALSHI_DAILY_DIR, UPSTREAM_KALSHI_DAILY_DIR]:
        if candidate.exists() and any(candidate.glob("*.parquet")):
            dirs.append(candidate)
    return dirs


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)

    kalshi_daily_dirs = resolve_daily_dirs()
    if not kalshi_daily_dirs:
        raise SystemExit(
            f"No Kalshi daily files found under {LOCAL_KALSHI_DAILY_DIR} or {UPSTREAM_KALSHI_DAILY_DIR}"
        )
    kalshi_file_map: Dict[str, Path] = {}
    for daily_dir in reversed(kalshi_daily_dirs):
        for path in daily_dir.glob("*.parquet"):
            kalshi_file_map[path.stem] = path
    kalshi_files = sorted(kalshi_file_map.values(), key=lambda p: p.stem)
    if not kalshi_files:
        raise SystemExit("No Kalshi daily files found after resolving local/upstream Kalshi caches")
    min_day = kalshi_files[0].stem
    max_day = kalshi_files[-1].stem

    trades = collect_overlap_trades(min_day, max_day)
    if not trades:
        raise SystemExit("No overlapping trades found for Kalshi daily coverage")

    start_et = pd.Timestamp(min_day, tz=NY) - pd.Timedelta(days=15)
    end_et = pd.Timestamp(max_day, tz=NY) + pd.Timedelta(days=1)
    market_df = load_market_df(start_et, end_et)

    provider = HistoricalKalshiProvider(kalshi_daily_dirs)
    overlay_cfg = dict(CONFIG.get("KALSHI_TRADE_OVERLAY", {}) or {})

    evaluated: List[Dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    for trade in trades:
        try:
            et = pd.Timestamp(trade["entry_time"]).tz_convert(NY)
        except Exception:
            skipped_counts["bad_entry_time"] += 1
            continue
        tp_dist = parse_tp_dist(trade)
        if tp_dist is None or tp_dist <= 0:
            skipped_counts["missing_tp"] += 1
            continue
        base_pnl = float(trade.get("pnl_dollars", 0.0) or 0.0)
        base_size = max(1, int(trade.get("size", 1) or 1))
        signal = {
            "strategy": trade["strategy"],
            "side": str(trade.get("side", "")).upper(),
            "entry_price": float(trade.get("entry_price", 0.0) or 0.0),
            "tp_dist": float(tp_dist),
            "size": int(base_size),
        }
        current_price = float(signal["entry_price"])
        gating_hour = et.hour in KALSHI_GATING_HOURS_ET
        hist = market_df.loc[:et].tail(int(overlay_cfg.get("lookback_bars", 20000)))
        profile = analyze_recent_price_action(hist, overlay_cfg)
        provider.set_context_time(et)

        realized = base_pnl
        plan = None
        if gating_hour:
            plan = build_trade_plan(
                signal,
                current_price,
                provider,
                price_action_profile=profile,
                overlay_cfg=overlay_cfg,
                tick_size=0.25,
            )
            if bool(plan.get("applied", False)):
                if bool(plan.get("entry_blocked", False)):
                    realized = 0.0
                else:
                    size_multiplier = float(plan.get("size_multiplier", 1.0) or 1.0)
                    trimmed_size = max(1, int(math.floor((float(base_size) * float(size_multiplier)) + 1e-9)))
                    realized = base_pnl * (float(trimmed_size) / float(base_size))

        evaluated.append(
            {
                "entry_time": str(trade["entry_time"]),
                "strategy": trade["strategy"],
                "side": signal["side"],
                "baseline_pnl_dol": base_pnl,
                "realized_pnl_dol": float(realized),
                "gating_hour": bool(gating_hour),
                "kalshi_plan_applied": bool(plan.get("applied", False)) if isinstance(plan, dict) else False,
                "kalshi_entry_blocked": bool(plan.get("entry_blocked", False)) if isinstance(plan, dict) else False,
                "kalshi_role": str(plan.get("role", "")) if isinstance(plan, dict) else "",
                "kalshi_mode": str(plan.get("mode", "")) if isinstance(plan, dict) else "",
                "kalshi_curve_informative": bool(plan.get("curve_informative", False)) if isinstance(plan, dict) else False,
                "kalshi_size_multiplier": float(plan.get("size_multiplier", 1.0) or 1.0) if isinstance(plan, dict) else 1.0,
                "kalshi_tp_adjusted": bool(plan.get("tp_adjusted", False)) if isinstance(plan, dict) else False,
                "kalshi_tp_trail_enabled": bool(plan.get("trail_enabled", False)) if isinstance(plan, dict) else False,
                "kalshi_entry_probability": plan.get("entry_probability") if isinstance(plan, dict) else None,
                "kalshi_probe_probability": plan.get("probe_probability") if isinstance(plan, dict) else None,
                "kalshi_support_score": plan.get("entry_support_score") if isinstance(plan, dict) else None,
                "kalshi_entry_threshold": plan.get("entry_threshold") if isinstance(plan, dict) else None,
                "kalshi_reason": str(plan.get("reason", "")) if isinstance(plan, dict) else ("outside_gating_hours" if not gating_hour else "missing_plan"),
            }
        )

    baseline_rows = [{**row, "realized_pnl_dol": row["baseline_pnl_dol"]} for row in evaluated]
    by_strategy_current: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_strategy_baseline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_strategy_current[row["strategy"]].append(row)
        by_strategy_baseline[row["strategy"]].append({**row, "realized_pnl_dol": row["baseline_pnl_dol"]})

    overlay_summary = {
        "coverage": {
            "kalshi_daily_dirs": [str(p) for p in kalshi_daily_dirs],
            "kalshi_daily_files": int(len(kalshi_files)),
            "kalshi_min_day": min_day,
            "kalshi_max_day": max_day,
            "overlap_trade_count": int(len(evaluated)),
            "skipped_counts": {str(k): int(v) for k, v in skipped_counts.items()},
            "trade_counts_by_strategy": dict(Counter(row["strategy"] for row in evaluated)),
            "gating_hour_trade_count": int(sum(1 for row in evaluated if row["gating_hour"])),
            "plan_applied_count": int(sum(1 for row in evaluated if row["kalshi_plan_applied"])),
        },
        "overall": {
            "baseline": evaluate_rows(baseline_rows),
            "entry_overlay_estimate": evaluate_rows(evaluated),
            "delta_pnl": float(sum(r["realized_pnl_dol"] - r["baseline_pnl_dol"] for r in evaluated)),
            "blocked_count": int(sum(1 for row in evaluated if row["kalshi_entry_blocked"])),
            "size_trim_count": int(sum(1 for row in evaluated if (not row["kalshi_entry_blocked"]) and row["kalshi_size_multiplier"] < 0.999)),
            "tp_adjusted_count": int(sum(1 for row in evaluated if row["kalshi_tp_adjusted"])),
            "tp_trail_enabled_count": int(sum(1 for row in evaluated if row["kalshi_tp_trail_enabled"])),
            "role_counts": dict(Counter(row["kalshi_role"] for row in evaluated if row["kalshi_role"])),
            "mode_counts": dict(Counter(row["kalshi_mode"] for row in evaluated if row["kalshi_mode"])),
            "reason_counts": dict(Counter(row["kalshi_reason"] for row in evaluated)),
        },
        "by_strategy": {},
    }
    for strategy in sorted(by_strategy_current):
        overlay_summary["by_strategy"][strategy] = {
            "baseline": evaluate_rows(by_strategy_baseline[strategy]),
            "entry_overlay_estimate": evaluate_rows(by_strategy_current[strategy]),
            "delta_pnl": float(
                sum(row["realized_pnl_dol"] - row["baseline_pnl_dol"] for row in by_strategy_current[strategy])
            ),
            "blocked_count": int(sum(1 for row in by_strategy_current[strategy] if row["kalshi_entry_blocked"])),
            "size_trim_count": int(
                sum(1 for row in by_strategy_current[strategy] if (not row["kalshi_entry_blocked"]) and row["kalshi_size_multiplier"] < 0.999)
            ),
            "tp_adjusted_count": int(sum(1 for row in by_strategy_current[strategy] if row["kalshi_tp_adjusted"])),
            "tp_trail_enabled_count": int(sum(1 for row in by_strategy_current[strategy] if row["kalshi_tp_trail_enabled"])),
            "role_counts": dict(Counter(row["kalshi_role"] for row in by_strategy_current[strategy] if row["kalshi_role"])),
            "reason_counts": dict(Counter(row["kalshi_reason"] for row in by_strategy_current[strategy])),
        }

    ml_summary = {
        "entry_gate": summarize_dataset(
            ART_DIR / "kalshi_training_data.parquet",
            ART_DIR / "model_kalshi_gate.joblib",
        ),
        "tp_gate": summarize_dataset(
            ART_DIR / "kalshi_tp_training_data.parquet",
            ART_DIR / "model_kalshi_tp_gate.joblib",
        ),
    }

    result = {
        "entry_overlay_replay": overlay_summary,
        "ml_datasets": ml_summary,
        "notes": [
            "Entry-overlay replay uses the current kalshi_trade_overlay.build_trade_plan logic against cached daily ladders.",
            "Realized PnL estimate captures only entry blocking and size trimming.",
            "TP adjustment and TP-trailing enablement are counted but not path-replayed here.",
            "ML dataset summaries come from saved parquets + joblib payload rolling-origin metadata.",
        ],
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    print(f"[write] {OUT_PATH}")
    print(json.dumps(result["entry_overlay_replay"]["coverage"], indent=2))
    print(json.dumps(result["entry_overlay_replay"]["overall"], indent=2))


if __name__ == "__main__":
    main()
