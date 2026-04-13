import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _base_decision_cfg(base_bundle: dict) -> dict:
    decision_cfg = deepcopy(base_bundle.get("decision_policy_model", {}) or {})
    if not isinstance(decision_cfg, dict):
        decision_cfg = {}
    if not isinstance(decision_cfg.get("score_components"), dict):
        decision_cfg["score_components"] = {}
    if not isinstance(decision_cfg.get("router_model_or_router_rules"), dict):
        router_payload = deepcopy(base_bundle.get("router_model_or_router_rules", {}) or {})
        if isinstance(router_payload, dict) and router_payload:
            decision_cfg["router_model_or_router_rules"] = router_payload
    return decision_cfg


def _variant_bundle(base_bundle: dict, *, name: str) -> tuple[dict, dict]:
    bundle = deepcopy(base_bundle)
    decision_cfg = _base_decision_cfg(base_bundle)
    bundle["decision_policy_model"] = decision_cfg
    bundle.setdefault("metadata", {})
    bundle["metadata"] = dict(bundle["metadata"])
    bundle["metadata"]["multibook_variant_source"] = "current_broad"
    bundle["metadata"]["multibook_variant_name"] = str(name)
    return bundle, decision_cfg


def _set_side_bias(decision_cfg: dict, *, long_bias: float, short_bias: float) -> None:
    score_cfg = dict(decision_cfg.get("score_components", {}) or {})
    side_bias = dict(score_cfg.get("side_score_bias", {}) or {})
    side_bias.update({"long": float(long_bias), "short": float(short_bias), "default": 0.0})
    score_cfg["side_score_bias"] = side_bias
    decision_cfg["score_components"] = score_cfg


def _set_router_bias(
    decision_cfg: dict,
    *,
    global_bias: dict[str, float] | None = None,
    session_bias: dict[str, dict[str, float]] | None = None,
    timeframe_bias: dict[str, dict[str, float]] | None = None,
) -> None:
    router_payload = dict(decision_cfg.get("router_model_or_router_rules", {}) or {})
    if global_bias:
        global_map = dict(router_payload.get("lane_priors_global", {}) or {})
        global_map.update({str(k): float(v) for k, v in global_bias.items()})
        router_payload["lane_priors_global"] = global_map
    if session_bias:
        session_map = dict(router_payload.get("lane_priors_by_session", {}) or {})
        for session_name, lane_map in session_bias.items():
            merged = dict(session_map.get(session_name, {}) or {})
            merged.update({str(k): float(v) for k, v in lane_map.items()})
            session_map[str(session_name)] = merged
        router_payload["lane_priors_by_session"] = session_map
    if timeframe_bias:
        timeframe_map = dict(router_payload.get("lane_priors_by_timeframe", {}) or {})
        for timeframe_name, lane_map in timeframe_bias.items():
            merged = dict(timeframe_map.get(timeframe_name, {}) or {})
            merged.update({str(k): float(v) for k, v in lane_map.items()})
            timeframe_map[str(timeframe_name)] = merged
        router_payload["lane_priors_by_timeframe"] = timeframe_map
    if router_payload:
        decision_cfg["router_model_or_router_rules"] = router_payload


def _variant_books(base_bundle: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}

    mild_balanced, decision_cfg = _variant_bundle(base_bundle, name="mild_balanced")
    _set_side_bias(decision_cfg, long_bias=-0.12, short_bias=0.06)
    decision_cfg["candidate_variant_filter"] = {
        "enabled": True,
        "allow_on_missing_variant_stats": True,
        "min_year_coverage": 6,
        "min_n_trades": 50,
        "side_overrides": {
            "long": {
                "min_profit_factor": 1.02,
                "max_loss_share": 0.70,
                "max_drawdown_norm": 5.5,
            },
            "short": {
                "min_profit_factor": 0.98,
                "max_loss_share": 0.70,
            },
        },
    }
    out["mild_balanced"] = mild_balanced

    short_direct_soft, decision_cfg = _variant_bundle(base_bundle, name="short_direct_soft")
    decision_cfg["selection_mode"] = "replace_router_lane"
    decision_cfg["selected_threshold"] = -1.200262
    decision_cfg["candidate_variant_filter"] = {"enabled": False}
    _set_side_bias(decision_cfg, long_bias=-0.30, short_bias=0.12)
    _set_router_bias(
        decision_cfg,
        global_bias={"Long_Rev": 0.08, "Short_Rev": 0.20, "Long_Mom": 0.08, "Short_Mom": 0.22},
        session_bias={
            "09-12": {"Long_Rev": 0.10, "Short_Rev": 0.22, "Long_Mom": 0.08, "Short_Mom": 0.12},
            "12-15": {"Long_Rev": 0.10, "Short_Rev": 0.22, "Long_Mom": 0.10, "Short_Mom": 0.22},
            "15-18": {"Long_Rev": 0.00, "Short_Rev": 0.08, "Long_Mom": 0.08, "Short_Mom": 0.24},
        },
        timeframe_bias={
            "5min": {"Long_Rev": 0.12, "Short_Rev": 0.20, "Long_Mom": 0.10, "Short_Mom": 0.18},
            "15min": {"Long_Rev": 0.08, "Short_Rev": 0.12, "Long_Mom": 0.10, "Short_Mom": 0.22},
        },
    )
    out["short_direct_soft"] = short_direct_soft

    short_direct_guarded, decision_cfg = _variant_bundle(base_bundle, name="short_direct_guarded")
    decision_cfg["selection_mode"] = "replace_router_lane"
    decision_cfg["selected_threshold"] = -1.150262
    decision_cfg["candidate_variant_filter"] = {
        "enabled": True,
        "allow_on_missing_variant_stats": True,
        "min_year_coverage": 4,
        "min_n_trades": 25,
        "side_overrides": {
            "long": {
                "min_profit_factor": 0.96,
                "max_loss_share": 0.72,
                "max_drawdown_norm": 8.0,
            },
            "short": {
                "min_profit_factor": 0.90,
                "max_loss_share": 0.76,
            },
        },
    }
    _set_side_bias(decision_cfg, long_bias=-0.26, short_bias=0.10)
    _set_router_bias(
        decision_cfg,
        global_bias={"Long_Rev": 0.10, "Short_Rev": 0.18, "Long_Mom": 0.10, "Short_Mom": 0.20},
        session_bias={
            "09-12": {"Long_Rev": 0.10, "Short_Rev": 0.20, "Long_Mom": 0.10, "Short_Mom": 0.10},
            "12-15": {"Long_Rev": 0.12, "Short_Rev": 0.20, "Long_Mom": 0.12, "Short_Mom": 0.20},
            "15-18": {"Long_Rev": 0.00, "Short_Rev": 0.06, "Long_Mom": 0.08, "Short_Mom": 0.22},
        },
        timeframe_bias={
            "5min": {"Long_Rev": 0.12, "Short_Rev": 0.18, "Long_Mom": 0.12, "Short_Mom": 0.16},
            "15min": {"Long_Rev": 0.10, "Short_Rev": 0.10, "Long_Mom": 0.12, "Short_Mom": 0.20},
        },
    )
    out["short_direct_guarded"] = short_direct_guarded

    short_direct_routerflex, decision_cfg = _variant_bundle(base_bundle, name="short_direct_routerflex")
    decision_cfg["selection_mode"] = "replace_router_lane"
    decision_cfg["selected_threshold"] = -1.260262
    decision_cfg["candidate_variant_filter"] = {
        "enabled": True,
        "allow_on_missing_variant_stats": True,
        "min_year_coverage": 3,
        "min_n_trades": 20,
        "side_overrides": {
            "long": {
                "min_profit_factor": 0.92,
                "max_loss_share": 0.74,
                "max_drawdown_norm": 9.0,
            },
            "short": {
                "min_profit_factor": 0.88,
                "max_loss_share": 0.78,
            },
        },
    }
    _set_side_bias(decision_cfg, long_bias=-0.34, short_bias=0.16)
    _set_router_bias(
        decision_cfg,
        global_bias={"Long_Rev": 0.06, "Short_Rev": 0.22, "Long_Mom": 0.06, "Short_Mom": 0.24},
        session_bias={
            "03-06": {"Long_Rev": 0.08, "Short_Rev": 0.06, "Long_Mom": 0.08, "Short_Mom": 0.06},
            "06-09": {"Long_Rev": 0.08, "Short_Rev": 0.08, "Long_Mom": 0.04, "Short_Mom": 0.08},
            "09-12": {"Long_Rev": 0.08, "Short_Rev": 0.24, "Long_Mom": 0.06, "Short_Mom": 0.16},
            "12-15": {"Long_Rev": 0.08, "Short_Rev": 0.24, "Long_Mom": 0.08, "Short_Mom": 0.24},
            "15-18": {"Long_Rev": 0.00, "Short_Rev": 0.10, "Long_Mom": 0.06, "Short_Mom": 0.26},
        },
        timeframe_bias={
            "5min": {"Long_Rev": 0.10, "Short_Rev": 0.22, "Long_Mom": 0.08, "Short_Mom": 0.18},
            "15min": {"Long_Rev": 0.06, "Short_Rev": 0.14, "Long_Mom": 0.08, "Short_Mom": 0.24},
        },
    )
    out["short_direct_routerflex"] = short_direct_routerflex

    return out


def main() -> None:
    base_path = ROOT / "artifacts" / "de3_v4_live" / "latest.json"
    out_dir = ROOT / "artifacts" / "de3_multibook_variant_books_20260331"
    base_bundle = _load_json(base_path)
    variants = _variant_books(base_bundle)
    summary: dict[str, dict] = {}
    for name, payload in variants.items():
        _write_json(out_dir / f"dynamic_engine3_v4_bundle.{name}.json", payload)
        decision_cfg = payload.get("decision_policy_model", {}) or {}
        score_cfg = decision_cfg.get("score_components", {}) or {}
        summary[name] = {
            "selection_mode": decision_cfg.get("selection_mode", ""),
            "selected_threshold": decision_cfg.get("selected_threshold"),
            "side_score_bias": dict(score_cfg.get("side_score_bias", {}) or {}),
            "candidate_variant_filter": dict(decision_cfg.get("candidate_variant_filter", {}) or {}),
        }
    _write_json(out_dir / "variant_book_summary.json", summary)
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
