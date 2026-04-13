import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from de3_v4_book_gate_trainer import train_de3_v4_book_gate


def _resolve_path(path_text: str) -> Path:
    raw = str(path_text or "").strip()
    if not raw:
        raise SystemExit("Missing required path.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_profiles() -> dict:
    return {
        "multibook_svchop_guarded": {
            "name": "multibook_svchop_guarded",
            "scope_priority": [
                {
                    "name": "session_vol_chop",
                    "fields": ["session", "ctx_volatility_regime", "ctx_chop_trend_regime"],
                    "min_decisions": 240,
                },
                {
                    "name": "session_vol",
                    "fields": ["session", "ctx_volatility_regime"],
                    "min_decisions": 200,
                },
                {
                    "name": "vol_chop",
                    "fields": ["ctx_volatility_regime", "ctx_chop_trend_regime"],
                    "min_decisions": 220,
                },
                {"name": "session", "fields": ["session"], "min_decisions": 180},
                {"name": "vol", "fields": ["ctx_volatility_regime"], "min_decisions": 180},
            ],
            "override_rules": {
                "min_train_decisions": 180,
                "min_alt_trades": 26,
                "min_avg_pnl_delta": 1.35,
                "min_alt_profit_factor": 1.05,
                "min_profit_factor_delta": 0.015,
                "score_weight_avg_pnl_delta": 1.0,
                "score_weight_profit_factor_delta": 20.0,
                "score_weight_daily_sharpe_delta": 3.0,
                "score_weight_long_share_reduction": 4.0,
            },
            "tune_objective": {
                "weight_profit_factor": 4600.0,
                "weight_daily_sharpe": 950.0,
                "weight_daily_sortino": 450.0,
                "weight_sqn": 250.0,
                "weight_max_drawdown": 0.90,
                "weight_long_share_reduction": 1600.0,
                "weight_override_rate": 950.0,
            },
        },
        "multibook_sv_guarded": {
            "name": "multibook_sv_guarded",
            "scope_priority": [
                {
                    "name": "session_vol",
                    "fields": ["session", "ctx_volatility_regime"],
                    "min_decisions": 210,
                },
                {"name": "session", "fields": ["session"], "min_decisions": 180},
                {"name": "vol", "fields": ["ctx_volatility_regime"], "min_decisions": 180},
                {"name": "chop", "fields": ["ctx_chop_trend_regime"], "min_decisions": 180},
            ],
            "override_rules": {
                "min_train_decisions": 180,
                "min_alt_trades": 24,
                "min_avg_pnl_delta": 1.10,
                "min_alt_profit_factor": 1.04,
                "min_profit_factor_delta": 0.010,
                "score_weight_avg_pnl_delta": 1.0,
                "score_weight_profit_factor_delta": 18.0,
                "score_weight_daily_sharpe_delta": 2.4,
                "score_weight_long_share_reduction": 3.2,
            },
            "tune_objective": {
                "weight_profit_factor": 4300.0,
                "weight_daily_sharpe": 900.0,
                "weight_daily_sortino": 430.0,
                "weight_sqn": 230.0,
                "weight_max_drawdown": 0.86,
                "weight_long_share_reduction": 1450.0,
                "weight_override_rate": 820.0,
            },
        },
        "multibook_vchop_guarded": {
            "name": "multibook_vchop_guarded",
            "scope_priority": [
                {
                    "name": "vol_chop",
                    "fields": ["ctx_volatility_regime", "ctx_chop_trend_regime"],
                    "min_decisions": 240,
                },
                {"name": "vol", "fields": ["ctx_volatility_regime"], "min_decisions": 180},
                {"name": "chop", "fields": ["ctx_chop_trend_regime"], "min_decisions": 180},
                {"name": "session", "fields": ["session"], "min_decisions": 180},
            ],
            "override_rules": {
                "min_train_decisions": 180,
                "min_alt_trades": 24,
                "min_avg_pnl_delta": 1.05,
                "min_alt_profit_factor": 1.04,
                "min_profit_factor_delta": 0.010,
                "score_weight_avg_pnl_delta": 1.0,
                "score_weight_profit_factor_delta": 19.0,
                "score_weight_daily_sharpe_delta": 2.6,
                "score_weight_long_share_reduction": 3.6,
            },
            "tune_objective": {
                "weight_profit_factor": 4400.0,
                "weight_daily_sharpe": 920.0,
                "weight_daily_sortino": 430.0,
                "weight_sqn": 240.0,
                "weight_max_drawdown": 0.88,
                "weight_long_share_reduction": 1500.0,
                "weight_override_rate": 860.0,
            },
        },
        "multibook_svc_conf_guarded": {
            "name": "multibook_svc_conf_guarded",
            "scope_priority": [
                {
                    "name": "session_vol_conf",
                    "fields": ["session", "ctx_volatility_regime", "ctx_confidence_band"],
                    "min_decisions": 240,
                },
                {
                    "name": "session_vol",
                    "fields": ["session", "ctx_volatility_regime"],
                    "min_decisions": 210,
                },
                {"name": "session", "fields": ["session"], "min_decisions": 180},
                {"name": "vol", "fields": ["ctx_volatility_regime"], "min_decisions": 180},
                {"name": "confidence", "fields": ["ctx_confidence_band"], "min_decisions": 180},
            ],
            "override_rules": {
                "min_train_decisions": 180,
                "min_alt_trades": 26,
                "min_avg_pnl_delta": 1.20,
                "min_alt_profit_factor": 1.05,
                "min_profit_factor_delta": 0.015,
                "score_weight_avg_pnl_delta": 1.0,
                "score_weight_profit_factor_delta": 19.0,
                "score_weight_daily_sharpe_delta": 2.6,
                "score_weight_long_share_reduction": 3.8,
            },
            "tune_objective": {
                "weight_profit_factor": 4450.0,
                "weight_daily_sharpe": 940.0,
                "weight_daily_sortino": 440.0,
                "weight_sqn": 240.0,
                "weight_max_drawdown": 0.88,
                "weight_long_share_reduction": 1500.0,
                "weight_override_rate": 880.0,
            },
        },
        "multibook_svprice_conservative": {
            "name": "multibook_svprice_conservative",
            "scope_priority": [
                {
                    "name": "session_vol_price",
                    "fields": ["session", "ctx_volatility_regime", "ctx_price_location"],
                    "min_decisions": 260,
                },
                {
                    "name": "vol_price",
                    "fields": ["ctx_volatility_regime", "ctx_price_location"],
                    "min_decisions": 240,
                },
                {
                    "name": "session_vol",
                    "fields": ["session", "ctx_volatility_regime"],
                    "min_decisions": 220,
                },
                {"name": "price", "fields": ["ctx_price_location"], "min_decisions": 210},
                {"name": "session", "fields": ["session"], "min_decisions": 190},
            ],
            "override_rules": {
                "min_train_decisions": 200,
                "min_alt_trades": 30,
                "min_avg_pnl_delta": 1.40,
                "min_alt_profit_factor": 1.08,
                "min_profit_factor_delta": 0.020,
                "score_weight_avg_pnl_delta": 1.1,
                "score_weight_profit_factor_delta": 24.0,
                "score_weight_daily_sharpe_delta": 3.2,
                "score_weight_long_share_reduction": 4.8,
            },
            "tune_objective": {
                "weight_profit_factor": 5000.0,
                "weight_daily_sharpe": 1020.0,
                "weight_daily_sortino": 500.0,
                "weight_sqn": 280.0,
                "weight_max_drawdown": 0.96,
                "weight_long_share_reduction": 1700.0,
                "weight_override_rate": 1125.0,
            },
        },
        "multibook_vchopexp_conservative": {
            "name": "multibook_vchopexp_conservative",
            "scope_priority": [
                {
                    "name": "vol_chop_expansion",
                    "fields": ["ctx_volatility_regime", "ctx_chop_trend_regime", "ctx_compression_expansion_regime"],
                    "min_decisions": 280,
                },
                {
                    "name": "vol_expansion",
                    "fields": ["ctx_volatility_regime", "ctx_compression_expansion_regime"],
                    "min_decisions": 240,
                },
                {
                    "name": "chop_expansion",
                    "fields": ["ctx_chop_trend_regime", "ctx_compression_expansion_regime"],
                    "min_decisions": 240,
                },
                {"name": "vol", "fields": ["ctx_volatility_regime"], "min_decisions": 190},
                {"name": "expansion", "fields": ["ctx_compression_expansion_regime"], "min_decisions": 190},
            ],
            "override_rules": {
                "min_train_decisions": 200,
                "min_alt_trades": 28,
                "min_avg_pnl_delta": 1.30,
                "min_alt_profit_factor": 1.07,
                "min_profit_factor_delta": 0.018,
                "score_weight_avg_pnl_delta": 1.0,
                "score_weight_profit_factor_delta": 23.0,
                "score_weight_daily_sharpe_delta": 3.0,
                "score_weight_long_share_reduction": 4.4,
            },
            "tune_objective": {
                "weight_profit_factor": 4850.0,
                "weight_daily_sharpe": 980.0,
                "weight_daily_sortino": 470.0,
                "weight_sqn": 270.0,
                "weight_max_drawdown": 0.95,
                "weight_long_share_reduction": 1600.0,
                "weight_override_rate": 1080.0,
            },
        },
        "multibook_min_override_pf": {
            "name": "multibook_min_override_pf",
            "scope_priority": [
                {
                    "name": "session_vol",
                    "fields": ["session", "ctx_volatility_regime"],
                    "min_decisions": 260,
                },
                {
                    "name": "vol_chop",
                    "fields": ["ctx_volatility_regime", "ctx_chop_trend_regime"],
                    "min_decisions": 260,
                },
                {"name": "session", "fields": ["session"], "min_decisions": 220},
                {"name": "vol", "fields": ["ctx_volatility_regime"], "min_decisions": 220},
            ],
            "override_rules": {
                "min_train_decisions": 220,
                "min_alt_trades": 34,
                "min_avg_pnl_delta": 1.55,
                "min_alt_profit_factor": 1.10,
                "min_profit_factor_delta": 0.025,
                "score_weight_avg_pnl_delta": 1.0,
                "score_weight_profit_factor_delta": 26.0,
                "score_weight_daily_sharpe_delta": 3.4,
                "score_weight_long_share_reduction": 4.2,
            },
            "tune_objective": {
                "weight_profit_factor": 5200.0,
                "weight_daily_sharpe": 1040.0,
                "weight_daily_sortino": 510.0,
                "weight_sqn": 290.0,
                "weight_max_drawdown": 1.00,
                "weight_long_share_reduction": 1500.0,
                "weight_override_rate": 1350.0,
            },
        },
        "multibook_session_side_pf": {
            "name": "multibook_session_side_pf",
            "scope_priority": [
                {
                    "name": "session_side",
                    "fields": ["session", "side_considered"],
                    "min_decisions": 250,
                },
                {
                    "name": "session_side_tf",
                    "fields": ["session", "side_considered", "timeframe"],
                    "min_decisions": 220,
                },
                {
                    "name": "side",
                    "fields": ["side_considered"],
                    "min_decisions": 280,
                },
                {"name": "session", "fields": ["session"], "min_decisions": 240},
            ],
            "override_rules": {
                "min_train_decisions": 220,
                "min_alt_trades": 35,
                "min_avg_pnl_delta": -0.10,
                "min_alt_profit_factor": 1.02,
                "min_profit_factor_delta": 0.020,
                "score_weight_avg_pnl_delta": 0.4,
                "score_weight_profit_factor_delta": 30.0,
                "score_weight_daily_sharpe_delta": 5.0,
                "score_weight_long_share_reduction": 3.0,
            },
            "tune_objective": {
                "weight_profit_factor": 5200.0,
                "weight_daily_sharpe": 980.0,
                "weight_daily_sortino": 470.0,
                "weight_sqn": 250.0,
                "weight_max_drawdown": 0.95,
                "weight_long_share_reduction": 1200.0,
                "weight_override_rate": 780.0,
            },
        },
        "multibook_session_side_risk": {
            "name": "multibook_session_side_risk",
            "scope_priority": [
                {
                    "name": "session_side",
                    "fields": ["session", "side_considered"],
                    "min_decisions": 240,
                },
                {
                    "name": "session_side_stype",
                    "fields": ["session", "side_considered", "strategy_type"],
                    "min_decisions": 220,
                },
                {
                    "name": "side_stype",
                    "fields": ["side_considered", "strategy_type"],
                    "min_decisions": 240,
                },
                {"name": "side", "fields": ["side_considered"], "min_decisions": 260},
            ],
            "override_rules": {
                "min_train_decisions": 220,
                "min_alt_trades": 35,
                "min_avg_pnl_delta": -0.14,
                "min_alt_profit_factor": 1.01,
                "min_profit_factor_delta": 0.015,
                "score_weight_avg_pnl_delta": 0.2,
                "score_weight_profit_factor_delta": 28.0,
                "score_weight_daily_sharpe_delta": 5.5,
                "score_weight_long_share_reduction": 2.0,
            },
            "tune_objective": {
                "weight_profit_factor": 5000.0,
                "weight_daily_sharpe": 1000.0,
                "weight_daily_sortino": 500.0,
                "weight_sqn": 250.0,
                "weight_max_drawdown": 1.00,
                "weight_long_share_reduction": 900.0,
                "weight_override_rate": 700.0,
            },
        },
        "multibook_session_subvar_risk": {
            "name": "multibook_session_subvar_risk",
            "scope_priority": [
                {
                    "name": "session_side_sub_strategy",
                    "fields": ["session", "side_considered", "sub_strategy"],
                    "min_decisions": 120,
                },
                {
                    "name": "side_sub_strategy",
                    "fields": ["side_considered", "sub_strategy"],
                    "min_decisions": 160,
                },
                {
                    "name": "session_sub_strategy",
                    "fields": ["session", "sub_strategy"],
                    "min_decisions": 140,
                },
                {
                    "name": "sub_strategy",
                    "fields": ["sub_strategy"],
                    "min_decisions": 220,
                },
            ],
            "override_rules": {
                "min_train_decisions": 120,
                "min_alt_trades": 20,
                "min_avg_pnl_delta": 0.20,
                "min_alt_profit_factor": 1.03,
                "min_profit_factor_delta": 0.010,
                "score_weight_avg_pnl_delta": 0.8,
                "score_weight_profit_factor_delta": 22.0,
                "score_weight_daily_sharpe_delta": 3.0,
                "score_weight_long_share_reduction": 2.8,
            },
            "tune_objective": {
                "weight_profit_factor": 4700.0,
                "weight_daily_sharpe": 980.0,
                "weight_daily_sortino": 470.0,
                "weight_sqn": 250.0,
                "weight_max_drawdown": 1.05,
                "weight_long_share_reduction": 1200.0,
                "weight_override_rate": 700.0,
            },
        },
        "multibook_session_subvar_pf": {
            "name": "multibook_session_subvar_pf",
            "scope_priority": [
                {
                    "name": "session_side_sub_strategy",
                    "fields": ["session", "side_considered", "sub_strategy"],
                    "min_decisions": 120,
                },
                {
                    "name": "side_sub_strategy",
                    "fields": ["side_considered", "sub_strategy"],
                    "min_decisions": 160,
                },
                {
                    "name": "session_sub_strategy",
                    "fields": ["session", "sub_strategy"],
                    "min_decisions": 140,
                },
                {
                    "name": "sub_strategy",
                    "fields": ["sub_strategy"],
                    "min_decisions": 220,
                },
            ],
            "override_rules": {
                "min_train_decisions": 120,
                "min_alt_trades": 20,
                "min_avg_pnl_delta": 0.35,
                "min_alt_profit_factor": 1.05,
                "min_profit_factor_delta": 0.015,
                "score_weight_avg_pnl_delta": 0.9,
                "score_weight_profit_factor_delta": 26.0,
                "score_weight_daily_sharpe_delta": 3.2,
                "score_weight_long_share_reduction": 2.4,
            },
            "tune_objective": {
                "weight_profit_factor": 5000.0,
                "weight_daily_sharpe": 1020.0,
                "weight_daily_sortino": 500.0,
                "weight_sqn": 270.0,
                "weight_max_drawdown": 0.98,
                "weight_long_share_reduction": 1050.0,
                "weight_override_rate": 760.0,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DE3 multi-book gate bundles from subgroup exports.")
    parser.add_argument(
        "--base-bundle",
        default=str(((CONFIG.get("DE3_V4") or {}).get("bundle_path") or "").strip()),
        help="Base broad DE3 bundle path.",
    )
    parser.add_argument("--base-decisions-csv", required=True, help="Broad-book decision export CSV.")
    parser.add_argument("--base-trade-attribution-csv", required=True, help="Broad-book trade attribution CSV.")
    parser.add_argument("--balanced-bundle", required=True, help="Balanced subgroup bundle path.")
    parser.add_argument("--balanced-decisions-csv", required=True, help="Balanced subgroup decision export CSV.")
    parser.add_argument("--balanced-trade-attribution-csv", required=True, help="Balanced subgroup trade attribution CSV.")
    parser.add_argument(
        "--balanced-name",
        default="balanced2",
        help="Logical name for the first alternate subgroup book.",
    )
    parser.add_argument(
        "--balanced-label",
        default="Balanced Core",
        help="Display label for the first alternate subgroup book.",
    )
    parser.add_argument("--shortfloor-bundle", required=True, help="Shortfloor subgroup bundle path.")
    parser.add_argument("--shortfloor-decisions-csv", required=True, help="Shortfloor subgroup decision export CSV.")
    parser.add_argument("--shortfloor-trade-attribution-csv", required=True, help="Shortfloor subgroup trade attribution CSV.")
    parser.add_argument(
        "--shortfloor-name",
        default="shortfloor",
        help="Logical name for the second alternate subgroup book.",
    )
    parser.add_argument(
        "--shortfloor-label",
        default="Short Efficiency Core",
        help="Display label for the second alternate subgroup book.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/de3_multibook_gate",
        help="Directory for trained multi-book gate bundle candidates.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated candidate profile names to run.",
    )
    args = parser.parse_args()

    base_bundle_path = _resolve_path(str(args.base_bundle))
    balanced_bundle_path = _resolve_path(str(args.balanced_bundle))
    shortfloor_bundle_path = _resolve_path(str(args.shortfloor_bundle))
    output_dir = _resolve_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_bundle = _load_json(base_bundle_path)
    balanced_bundle = _load_json(balanced_bundle_path)
    shortfloor_bundle = _load_json(shortfloor_bundle_path)

    books = [
        {
            "name": "broad",
            "label": "Broad Side-Aware",
            "is_default": True,
            "bundle_path": str(base_bundle_path),
            "decision_policy_model": (
                dict(base_bundle.get("decision_policy_model", {}))
                if isinstance(base_bundle.get("decision_policy_model"), dict)
                else {}
            ),
            "candidate_variant_filter": (
                (base_bundle.get("decision_policy_model", {}) or {}).get("candidate_variant_filter", {})
                if isinstance(base_bundle.get("decision_policy_model", {}), dict)
                else {}
            ),
            "decisions_csv_path": str(_resolve_path(str(args.base_decisions_csv))),
            "trade_attribution_csv_path": str(_resolve_path(str(args.base_trade_attribution_csv))),
        },
        {
            "name": str(args.balanced_name or "balanced2").strip() or "balanced2",
            "label": str(args.balanced_label or "Balanced Core").strip() or "Balanced Core",
            "bundle_path": str(balanced_bundle_path),
            "decision_policy_model": (
                dict(balanced_bundle.get("decision_policy_model", {}))
                if isinstance(balanced_bundle.get("decision_policy_model"), dict)
                else {}
            ),
            "candidate_variant_filter": (
                (balanced_bundle.get("decision_policy_model", {}) or {}).get("candidate_variant_filter", {})
                if isinstance(balanced_bundle.get("decision_policy_model", {}), dict)
                else {}
            ),
            "decisions_csv_path": str(_resolve_path(str(args.balanced_decisions_csv))),
            "trade_attribution_csv_path": str(_resolve_path(str(args.balanced_trade_attribution_csv))),
        },
        {
            "name": str(args.shortfloor_name or "shortfloor").strip() or "shortfloor",
            "label": str(args.shortfloor_label or "Short Efficiency Core").strip() or "Short Efficiency Core",
            "bundle_path": str(shortfloor_bundle_path),
            "decision_policy_model": (
                dict(shortfloor_bundle.get("decision_policy_model", {}))
                if isinstance(shortfloor_bundle.get("decision_policy_model"), dict)
                else {}
            ),
            "candidate_variant_filter": (
                (shortfloor_bundle.get("decision_policy_model", {}) or {}).get("candidate_variant_filter", {})
                if isinstance(shortfloor_bundle.get("decision_policy_model", {}), dict)
                else {}
            ),
            "decisions_csv_path": str(_resolve_path(str(args.shortfloor_decisions_csv))),
            "trade_attribution_csv_path": str(_resolve_path(str(args.shortfloor_trade_attribution_csv))),
        },
    ]

    profiles = _candidate_profiles()
    if str(args.only or "").strip():
        allowed = {item.strip() for item in str(args.only).split(",") if item.strip()}
        profiles = {name: cfg for name, cfg in profiles.items() if name in allowed}
    if not profiles:
        raise SystemExit("No candidate profiles selected.")

    summary_rows = []
    for name, cfg in profiles.items():
        result = train_de3_v4_book_gate(
            base_bundle=base_bundle,
            books=books,
            cfg=cfg,
        )
        book_gate_model = result["book_gate_model"]
        training_report = result["book_gate_training_report"]
        bundle_payload = dict(base_bundle)
        bundle_payload["book_gate_model"] = book_gate_model
        bundle_payload["book_gate_training_report"] = training_report
        meta = bundle_payload.get("metadata", {}) if isinstance(bundle_payload.get("metadata"), dict) else {}
        meta = dict(meta)
        meta["book_gate_retrained_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta["book_gate_retrained_base_bundle"] = str(base_bundle_path)
        bundle_payload["metadata"] = meta
        bundle_path = output_dir / f"dynamic_engine3_v4_bundle.{name}.json"
        bundle_path.write_text(json.dumps(bundle_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        report_path = output_dir / f"{name}.training_report.json"
        report_path.write_text(json.dumps(training_report, indent=2, ensure_ascii=True), encoding="utf-8")
        summary_rows.append(
            {
                "name": name,
                "bundle_path": str(bundle_path),
                "training_report_path": str(report_path),
                "tune_objective_score": float(training_report.get("tune_objective_score", 0.0)),
                "tune_net_pnl": float(((training_report.get("tune_gate_metrics", {}) or {}).get("net_pnl", 0.0))),
                "tune_profit_factor": float(((training_report.get("tune_gate_metrics", {}) or {}).get("profit_factor", 0.0))),
                "tune_daily_sharpe": float(((training_report.get("tune_gate_metrics", {}) or {}).get("daily_sharpe", 0.0))),
                "tune_daily_sortino": float(((training_report.get("tune_gate_metrics", {}) or {}).get("daily_sortino", 0.0))),
                "tune_max_drawdown": float(((training_report.get("tune_gate_metrics", {}) or {}).get("max_drawdown", 0.0))),
                "tune_long_share": float(((training_report.get("tune_gate_metrics", {}) or {}).get("long_share", 0.0))),
                "tune_override_rate": float(training_report.get("tune_override_rate", 0.0)),
                "fit_override_count": int(training_report.get("fit_bucket_override_count", 0)),
            }
        )

    summary_rows.sort(key=lambda row: float(row["tune_objective_score"]), reverse=True)
    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"candidate_summary={summary_path}")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=True))


if __name__ == "__main__":
    main()
