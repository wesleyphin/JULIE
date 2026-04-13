import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest_mes_et import (
    CONFIG,
    NY_TZ,
    apply_symbol_mode,
    choose_symbol,
    load_csv_cached,
    parse_user_datetime,
    run_backtest,
)


TARGET_BUCKET_SWEEPS = {
    "15min_03-06_Long_Rev_T5_SL10_TP25": {
        "sl": [9.0, 10.0, 11.0],
        "tp": [20.0, 22.5, 25.0, 27.5, 30.0],
    },
    "5min_09-12_Long_Rev_T6_SL10_TP25": {
        "sl": [9.0, 10.0, 11.0],
        "tp": [20.0, 22.5, 25.0, 27.5, 30.0, 32.5],
    },
    "5min_21-24_Long_Rev_T2_SL10_TP12.5": {
        "sl": [8.0, 9.0, 10.0, 11.0],
        "tp": [10.0, 12.5, 15.0, 17.5],
    },
}


def _fmt_hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(seconds + 0.5)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _baseline_from_bucket_id(bucket_id: str) -> tuple[Optional[float], Optional[float]]:
    sl = None
    tp = None
    if "_SL" in bucket_id:
        sl_text = bucket_id.split("_SL", 1)[1].split("_", 1)[0]
        try:
            sl = float(sl_text)
        except Exception:
            sl = None
    if "_TP" in bucket_id:
        tp_text = bucket_id.split("_TP", 1)[1]
        try:
            tp = float(tp_text)
        except Exception:
            tp = None
    return sl, tp


def _parse_bucket_id(bucket_id: str) -> Optional[dict]:
    text = str(bucket_id or "").strip()
    if not text:
        return None
    try:
        left, right = text.rsplit("_T", 1)
        thresh_text, right = right.split("_SL", 1)
        sl_text, tp_text = right.split("_TP", 1)
        parts = left.split("_")
        if len(parts) < 3:
            return None
        tf = parts[0]
        session = parts[1]
        strategy_type = "_".join(parts[2:])
        return {
            "tf": tf,
            "session": session,
            "strategy_type": strategy_type,
            "thresh": float(thresh_text),
            "sl": float(sl_text),
            "tp": float(tp_text),
        }
    except Exception:
        return None


def _resolve_target_bucket_id(target_id: str, available_ids: Iterable[str]) -> str:
    avail = [str(x or "").strip() for x in available_ids if str(x or "").strip()]
    if target_id in avail:
        return target_id

    tgt = _parse_bucket_id(target_id)
    if tgt is None:
        return target_id

    candidates: list[tuple[float, str]] = []
    for cand_id in avail:
        parsed = _parse_bucket_id(cand_id)
        if parsed is None:
            continue
        if (
            str(parsed["tf"]) == str(tgt["tf"])
            and str(parsed["session"]) == str(tgt["session"])
            and str(parsed["strategy_type"]) == str(tgt["strategy_type"])
            and abs(float(parsed["thresh"]) - float(tgt["thresh"])) < 1e-9
        ):
            dist = abs(float(parsed["sl"]) - float(tgt["sl"])) + abs(float(parsed["tp"]) - float(tgt["tp"]))
            candidates.append((float(dist), cand_id))
    if not candidates:
        return target_id
    candidates.sort(key=lambda x: (x[0], x[1]))
    return str(candidates[0][1])


def _parse_csv_set(value: str) -> Optional[set[str]]:
    text = str(value or "").strip()
    if not text:
        return None
    items = {s.strip() for s in text.split(",") if s.strip()}
    return items or None


def _compute_drawdown(pnl_seq: list[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_seq:
        cum += float(p)
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _profit_factor(pnl_seq: list[float]) -> float:
    gross_win = sum(p for p in pnl_seq if p > 0.0)
    gross_loss = -sum(p for p in pnl_seq if p < 0.0)
    if gross_loss <= 0.0:
        return float("inf") if gross_win > 0.0 else 0.0
    return float(gross_win / gross_loss)


def _bucket_metrics(trades: list[dict], bucket_id: str) -> dict:
    rows = [
        t
        for t in trades
        if str(t.get("strategy", "")).lower() == "dynamicengine3"
        and str(t.get("sub_strategy", "") or "") == bucket_id
    ]
    pnl = [float(t.get("pnl_net", 0.0) or 0.0) for t in rows]
    trade_count = len(rows)
    wins = sum(1 for p in pnl if p > 0.0)
    stop_count = sum(1 for t in rows if str(t.get("exit_reason", "") or "") in {"stop", "stop_gap"})
    stop_gap_count = sum(1 for t in rows if str(t.get("exit_reason", "") or "") == "stop_gap")
    avg_mae = float(sum(float(t.get("mae_points", 0.0) or 0.0) for t in rows) / trade_count) if trade_count else 0.0
    avg_mfe = float(sum(float(t.get("mfe_points", 0.0) or 0.0) for t in rows) / trade_count) if trade_count else 0.0
    net_pnl = float(sum(pnl))
    return {
        "bucket_id": bucket_id,
        "trades": int(trade_count),
        "net_pnl": net_pnl,
        "profit_factor": _profit_factor(pnl),
        "win_rate": (float(wins) / float(trade_count)) if trade_count else 0.0,
        "avg_pnl": (net_pnl / float(trade_count)) if trade_count else 0.0,
        "stop_count": int(stop_count),
        "stop_gap_count": int(stop_gap_count),
        "avg_mae": float(avg_mae),
        "avg_mfe": float(avg_mfe),
        "max_drawdown": _compute_drawdown(pnl),
    }


def _recommend_variant(rows: list[dict]) -> dict:
    baseline = next((r for r in rows if bool(r.get("baseline_flag"))), None)
    if baseline is None:
        baseline = rows[0]
    b_trades = max(1, int(baseline.get("trades", 0) or 0))
    b_dd = float(baseline.get("max_drawdown", 0.0) or 0.0)
    b_pf = float(baseline.get("profit_factor", 0.0) or 0.0)
    b_net = float(baseline.get("net_pnl", 0.0) or 0.0)

    def _eligible(r: dict) -> bool:
        trades = int(r.get("trades", 0) or 0)
        dd = float(r.get("max_drawdown", 0.0) or 0.0)
        if trades < max(5, int(math.floor(0.85 * b_trades))):
            return False
        if b_dd > 0.0 and dd > (1.15 * b_dd):
            return False
        return True

    pool = [r for r in rows if _eligible(r)]
    if not pool:
        return baseline

    def _pf_sort(v: float) -> float:
        if not math.isfinite(v):
            return 1e9
        return v

    pool_sorted = sorted(
        pool,
        key=lambda r: (
            _pf_sort(float(r.get("profit_factor", 0.0) or 0.0)),
            float(r.get("net_pnl", 0.0) or 0.0),
            -abs(int(r.get("trades", 0) or 0) - b_trades),
            -float(r.get("max_drawdown", 0.0) or 0.0),
        ),
        reverse=True,
    )
    best = pool_sorted[0]
    best_pf = float(best.get("profit_factor", 0.0) or 0.0)
    best_net = float(best.get("net_pnl", 0.0) or 0.0)
    if best_pf <= (b_pf + 0.03) and best_net <= (b_net + 25.0):
        return baseline
    return best


def _load_range_df(
    data_path: Path,
    start_time: dt.datetime,
    end_time: dt.datetime,
    symbol_mode: str,
    symbol_method: str,
    symbol: Optional[str],
) -> tuple[pd.DataFrame, str]:
    base_dir = Path(__file__).resolve().parents[1]
    df = load_csv_cached(data_path, cache_dir=base_dir / "cache", use_cache=True)
    range_df = df.loc[start_time:end_time]
    if range_df.empty:
        raise RuntimeError("No rows in selected range.")

    active_symbol = symbol or ""
    if "symbol" in range_df.columns:
        if symbol_mode == "single":
            if range_df["symbol"].nunique(dropna=True) > 1:
                preferred = active_symbol or CONFIG.get("TARGET_SYMBOL")
                chosen_symbol = choose_symbol(range_df, preferred)
                range_df = range_df[range_df["symbol"] == chosen_symbol]
                active_symbol = str(chosen_symbol)
            else:
                active_symbol = str(range_df["symbol"].dropna().iloc[0]) if not range_df.empty else "AUTO"
        else:
            range_df, auto_label, _ = apply_symbol_mode(range_df, symbol_mode, symbol_method)
            active_symbol = str(auto_label)
        if symbol_mode != "single":
            range_df = range_df.drop(columns=["symbol"], errors="ignore")

    if range_df.empty:
        raise RuntimeError("No rows after symbol mode selection.")
    if not active_symbol:
        active_symbol = "AUTO"
    return range_df, active_symbol


def _run_once(
    df: pd.DataFrame,
    start_time: dt.datetime,
    end_time: dt.datetime,
    *,
    enabled_strategies: Optional[set[str]],
    enabled_filters: Optional[set[str]],
    overrides: Optional[dict] = None,
) -> dict:
    return run_backtest(
        df,
        start_time,
        end_time,
        enabled_strategies=enabled_strategies,
        enabled_filters=enabled_filters,
        de3_bucket_bracket_overrides=overrides,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Narrow DE3 v2 bucket bracket sweep for selected high-impact buckets.",
    )
    parser.add_argument("--data", required=True, help="Path to source parquet/csv dataset.")
    parser.add_argument("--start", required=True, help="Start ET datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM).")
    parser.add_argument("--end", required=True, help="End ET datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM).")
    parser.add_argument(
        "--symbol-mode",
        default=str(CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"),
        choices=["single", "auto", "auto_by_day", "roll"],
        help="Symbol selection mode (default from config).",
    )
    parser.add_argument(
        "--symbol-method",
        default=str(CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"),
        choices=["volume", "rows"],
        help="AUTO_BY_DAY symbol scoring method.",
    )
    parser.add_argument("--symbol", default="", help="Optional symbol override for single mode.")
    parser.add_argument(
        "--enabled-strategies",
        default="DynamicEngine3Strategy",
        help="Comma-separated strategy allowlist (default: DynamicEngine3Strategy).",
    )
    parser.add_argument(
        "--enabled-filters",
        default="",
        help="Comma-separated filter allowlist (default: all backtest filters).",
    )
    parser.add_argument(
        "--out-dir",
        default="reports",
        help="Output directory for sweep reports.",
    )
    parser.add_argument(
        "--buckets",
        default="",
        help="Optional comma-separated subset of target bucket IDs.",
    )
    parser.add_argument(
        "--show-backtest-eta",
        action="store_true",
        help="Enable inner backtest console ETA/progress during each sweep run.",
    )
    parser.add_argument(
        "--backtest-eta-every-sec",
        type=float,
        default=15.0,
        help="Inner backtest ETA log cadence in seconds when --show-backtest-eta is set.",
    )
    args = parser.parse_args()

    data_path = Path(str(args.data)).expanduser()
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parents[1] / data_path
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    start_time = parse_user_datetime(str(args.start), NY_TZ, is_end=False)
    end_time = parse_user_datetime(str(args.end), NY_TZ, is_end=True)
    if start_time > end_time:
        raise SystemExit("start must be <= end")

    enabled_strategies = _parse_csv_set(args.enabled_strategies)
    enabled_filters = _parse_csv_set(args.enabled_filters)

    bucket_subset = _parse_csv_set(args.buckets)
    if bucket_subset:
        sweeps = {k: v for k, v in TARGET_BUCKET_SWEEPS.items() if k in bucket_subset}
    else:
        sweeps = dict(TARGET_BUCKET_SWEEPS)
    if not sweeps:
        raise SystemExit("No matching target buckets selected for sweep.")

    out_dir = Path(str(args.out_dir)).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol_mode = str(args.symbol_mode or "single").lower()
    if symbol_mode == "auto":
        symbol_mode = "auto_by_day"
    if symbol_mode == "roll":
        symbol_mode = "auto_by_day"

    # Speed/IO hygiene for multi-run sweep; no strategy logic impact.
    prev_console_progress = CONFIG.get("BACKTEST_CONSOLE_PROGRESS", True)
    prev_console_status = CONFIG.get("BACKTEST_CONSOLE_STATUS", True)
    prev_live_report = dict(CONFIG.get("BACKTEST_LIVE_REPORT", {}) or {})
    show_backtest_eta = bool(args.show_backtest_eta)
    if show_backtest_eta:
        CONFIG["BACKTEST_CONSOLE_PROGRESS"] = True
        CONFIG["BACKTEST_CONSOLE_STATUS"] = True
        try:
            CONFIG["BACKTEST_CONSOLE_PROGRESS_EVERY_SEC"] = max(
                1.0, float(args.backtest_eta_every_sec or 15.0)
            )
        except Exception:
            CONFIG["BACKTEST_CONSOLE_PROGRESS_EVERY_SEC"] = 15.0
    else:
        CONFIG["BACKTEST_CONSOLE_PROGRESS"] = False
        CONFIG["BACKTEST_CONSOLE_STATUS"] = False
    CONFIG["BACKTEST_LIVE_REPORT"] = {"enabled": False}

    try:
        range_df, active_symbol = _load_range_df(
            data_path=data_path,
            start_time=start_time,
            end_time=end_time,
            symbol_mode=symbol_mode,
            symbol_method=str(args.symbol_method or "volume"),
            symbol=str(args.symbol or "").strip() or None,
        )

        total_bucket_runs = sum(
            len(spec.get("sl", [])) * len(spec.get("tp", []))
            for spec in sweeps.values()
        )
        total_runs = 1 + total_bucket_runs + 1  # baseline + variants + combined
        run_idx = 0
        started = dt.datetime.now(dt.timezone.utc)

        def _log_progress(tag: str) -> None:
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            if run_idx > 0:
                eta = (elapsed / float(run_idx)) * float(max(0, total_runs - run_idx))
            else:
                eta = float("nan")
            pct = (float(run_idx) / float(total_runs)) * 100.0 if total_runs else 0.0
            print(
                f"[de3-sweep] {run_idx}/{total_runs} ({pct:.1f}%) elapsed={_fmt_hms(elapsed)} "
                f"eta={_fmt_hms(eta)} | {tag}",
                flush=True,
            )

        # Baseline run (no overrides)
        run_idx += 1
        _log_progress("baseline")
        baseline_stats = _run_once(
            range_df,
            start_time,
            end_time,
            enabled_strategies=enabled_strategies,
            enabled_filters=enabled_filters,
            overrides=None,
        )
        baseline_trades = baseline_stats.get("trade_log", []) or []
        baseline_de3_ids = sorted(
            {
                str(t.get("sub_strategy", "") or "").strip()
                for t in baseline_trades
                if str(t.get("strategy", "")).lower() == "dynamicengine3"
                and str(t.get("sub_strategy", "") or "").strip()
            }
        )
        resolved_sweeps: dict[str, dict] = {}
        resolved_id_map: dict[str, str] = {}
        for target_id, spec in sweeps.items():
            resolved_id = _resolve_target_bucket_id(target_id, baseline_de3_ids)
            resolved_sweeps[resolved_id] = dict(spec)
            resolved_id_map[target_id] = resolved_id
            if resolved_id != target_id:
                print(
                    f"[de3-sweep] target bucket remapped: {target_id} -> {resolved_id}",
                    flush=True,
                )
        sweeps = resolved_sweeps

        all_rows: list[dict] = []
        recommendations: dict[str, dict] = {}

        for bucket_id, spec in sweeps.items():
            sl_vals = [float(v) for v in (spec.get("sl") or [])]
            tp_vals = [float(v) for v in (spec.get("tp") or [])]
            base_sl, base_tp = _baseline_from_bucket_id(bucket_id)
            if base_sl is None or base_tp is None:
                raise RuntimeError(f"Could not parse baseline SL/TP from bucket id: {bucket_id}")

            for sl in sl_vals:
                for tp in tp_vals:
                    run_idx += 1
                    _log_progress(f"{bucket_id} SL={sl:g} TP={tp:g}")
                    stats = _run_once(
                        range_df,
                        start_time,
                        end_time,
                        enabled_strategies=enabled_strategies,
                        enabled_filters=enabled_filters,
                        overrides={bucket_id: {"sl": float(sl), "tp": float(tp)}},
                    )
                    metrics = _bucket_metrics(stats.get("trade_log", []) or [], bucket_id)
                    metrics.update(
                        {
                            "tested_sl": float(sl),
                            "tested_tp": float(tp),
                            "baseline_flag": bool(abs(sl - base_sl) < 1e-9 and abs(tp - base_tp) < 1e-9),
                            "overall_net_pnl": float(stats.get("equity", 0.0) or 0.0),
                            "overall_max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
                            "overall_trades": int(stats.get("trades", 0) or 0),
                        }
                    )
                    all_rows.append(metrics)

            bucket_rows = [r for r in all_rows if str(r.get("bucket_id")) == bucket_id]
            best = _recommend_variant(bucket_rows)
            baseline_row = next((r for r in bucket_rows if bool(r.get("baseline_flag"))), bucket_rows[0])
            recommendations[bucket_id] = {
                "baseline": {
                    "sl": float(baseline_row.get("tested_sl", base_sl)),
                    "tp": float(baseline_row.get("tested_tp", base_tp)),
                    "metrics": baseline_row,
                },
                "recommended": {
                    "sl": float(best.get("tested_sl", base_sl)),
                    "tp": float(best.get("tested_tp", base_tp)),
                    "metrics": best,
                    "changed": bool(
                        abs(float(best.get("tested_sl", base_sl)) - base_sl) > 1e-9
                        or abs(float(best.get("tested_tp", base_tp)) - base_tp) > 1e-9
                    ),
                },
            }

        combined_overrides = {}
        for bucket_id, payload in recommendations.items():
            rec = payload.get("recommended", {}) or {}
            if bool(rec.get("changed")):
                combined_overrides[bucket_id] = {
                    "sl": float(rec.get("sl")),
                    "tp": float(rec.get("tp")),
                }

        run_idx += 1
        _log_progress("combined_recommended_overrides")
        combined_stats = _run_once(
            range_df,
            start_time,
            end_time,
            enabled_strategies=enabled_strategies,
            enabled_filters=enabled_filters,
            overrides=combined_overrides if combined_overrides else None,
        )

        combined_bucket_metrics = {
            bucket_id: _bucket_metrics(combined_stats.get("trade_log", []) or [], bucket_id)
            for bucket_id in sweeps.keys()
        }

        rows_df = pd.DataFrame(all_rows)
        rows_df = rows_df.sort_values(
            ["bucket_id", "baseline_flag", "profit_factor", "net_pnl"],
            ascending=[True, False, False, False],
        )
        csv_path = out_dir / "de3_v2_bucket_bracket_sweep_results.csv"
        rows_df.to_csv(csv_path, index=False)

        summary = {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "data_path": str(data_path),
            "range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "symbol_mode": symbol_mode,
            "symbol": active_symbol,
            "enabled_strategies": sorted(enabled_strategies) if enabled_strategies is not None else [],
            "enabled_filters": sorted(enabled_filters) if enabled_filters is not None else [],
            "target_buckets": list(sweeps.keys()),
            "target_bucket_resolution": resolved_id_map,
            "recommendations": recommendations,
            "combined_overrides": combined_overrides,
            "baseline_overall": {
                "net_pnl": float(baseline_stats.get("equity", 0.0) or 0.0),
                "max_drawdown": float(baseline_stats.get("max_drawdown", 0.0) or 0.0),
                "trades": int(baseline_stats.get("trades", 0) or 0),
                "winrate": float(baseline_stats.get("winrate", 0.0) or 0.0),
            },
            "combined_overall": {
                "net_pnl": float(combined_stats.get("equity", 0.0) or 0.0),
                "max_drawdown": float(combined_stats.get("max_drawdown", 0.0) or 0.0),
                "trades": int(combined_stats.get("trades", 0) or 0),
                "winrate": float(combined_stats.get("winrate", 0.0) or 0.0),
            },
            "combined_bucket_metrics": combined_bucket_metrics,
            "output_csv": str(csv_path),
        }
        json_path = out_dir / "de3_v2_bucket_bracket_sweep_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

        print(f"[de3-sweep] wrote {csv_path}", flush=True)
        print(f"[de3-sweep] wrote {json_path}", flush=True)
        print(
            "[de3-sweep] combined overrides: "
            + (json.dumps(combined_overrides, ensure_ascii=True) if combined_overrides else "{}"),
            flush=True,
        )
        print(
            (
                f"[de3-sweep] baseline net=${summary['baseline_overall']['net_pnl']:.2f} "
                f"dd=${summary['baseline_overall']['max_drawdown']:.2f} "
                f"-> combined net=${summary['combined_overall']['net_pnl']:.2f} "
                f"dd=${summary['combined_overall']['max_drawdown']:.2f}"
            ),
            flush=True,
        )
    finally:
        CONFIG["BACKTEST_CONSOLE_PROGRESS"] = prev_console_progress
        CONFIG["BACKTEST_CONSOLE_STATUS"] = prev_console_status
        CONFIG["BACKTEST_LIVE_REPORT"] = prev_live_report


if __name__ == "__main__":
    main()
