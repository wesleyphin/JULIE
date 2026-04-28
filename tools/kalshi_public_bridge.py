from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from services.kalshi_provider import KalshiProvider  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT / "kalshi_public_bridge_snapshot.json"


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp_path.write_text(text, encoding="utf-8")
    try:
        tmp_path.replace(path)
    except PermissionError:
        # Some Windows/sandbox combinations permit writes but deny os.replace.
        path.write_text(text, encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass


def build_public_provider() -> KalshiProvider:
    cfg = dict(CONFIG.get("KALSHI", {}) or {})
    cfg.update(
        {
            "enabled": True,
            "public_read_only": True,
            "key_id": "",
            "private_key_path": "",
            "bridge_cache_path": "",
            "bridge_cache_only": False,
        }
    )
    return KalshiProvider(cfg)


def summarize_curve(markets: list[Dict[str, Any]]) -> Dict[str, Any]:
    probabilities = []
    for row in markets:
        try:
            probabilities.append(float(row.get("probability")))
        except (AttributeError, TypeError, ValueError):
            continue
    nonzero = [prob for prob in probabilities if prob > 0.0]
    prob_min = min(probabilities) if probabilities else None
    prob_max = max(probabilities) if probabilities else None
    prob_range = (prob_max - prob_min) if prob_min is not None and prob_max is not None else None
    unique_count = len({round(prob, 4) for prob in probabilities})
    return {
        "probability_count": len(probabilities),
        "nonzero_probability_count": len(nonzero),
        "probability_min": prob_min,
        "probability_max": prob_max,
        "probability_range": prob_range,
        "unique_probability_count": unique_count,
        "informative_curve": (
            len(probabilities) >= 8
            and unique_count >= 4
            and prob_range is not None
            and prob_range >= 0.08
        ),
    }


def build_snapshot(
    provider: KalshiProvider,
    *,
    es_price: Optional[float] = None,
    target_price: Optional[float] = None,
    target_side: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    markets = provider.refresh()
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    curve_summary = summarize_curve(markets)

    snapshot: Dict[str, Any] = {
        "source": "kalshi_public_bridge",
        "mode": "public_read_only",
        "ts": time.time(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": elapsed_ms,
        "enabled": bool(getattr(provider, "enabled", False)),
        "healthy": bool(getattr(provider, "is_healthy", False)),
        "event_ticker": getattr(provider, "last_resolved_ticker", None),
        "active_settlement_hour_et": provider.active_settlement_hour_et(),
        "basis_offset": float(getattr(provider, "basis_offset", 0.0) or 0.0),
        "market_count": len(markets),
        **curve_summary,
        "markets": markets,
    }

    if es_price is not None:
        snapshot["es_price"] = float(es_price)
        snapshot["sentiment"] = provider.get_sentiment(float(es_price))
        snapshot["relative_markets"] = provider.get_relative_markets_for_ui(
            [float(es_price)],
            window_size=30,
        )

    if target_price is not None:
        snapshot["target_price"] = float(target_price)
        snapshot["target_side"] = str(target_side or "")
        snapshot["target_probability"] = provider.get_target_probability(
            float(target_price),
            str(target_side or "") or None,
        )

    return snapshot


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch public Kalshi KXINXU data once and publish a local bridge snapshot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CONFIG.get("KALSHI", {}).get("bridge_cache_path") or DEFAULT_OUTPUT_PATH),
    )
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--es-price", type=float, default=None)
    parser.add_argument("--target-price", type=float, default=None)
    parser.add_argument("--target-side", type=str, default=None)
    parser.add_argument(
        "--require-informative",
        action="store_true",
        help="Return non-zero from --once unless the fetched curve is trade-overlay informative.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    provider = build_public_provider()

    while True:
        snapshot = build_snapshot(
            provider,
            es_price=args.es_price,
            target_price=args.target_price,
            target_side=args.target_side,
        )
        write_json_atomic(args.output, snapshot)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "event_ticker": snapshot.get("event_ticker"),
                    "market_count": snapshot.get("market_count"),
                    "informative_curve": snapshot.get("informative_curve"),
                    "nonzero_probability_count": snapshot.get("nonzero_probability_count"),
                    "elapsed_ms": snapshot.get("elapsed_ms"),
                    "healthy": snapshot.get("healthy"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.once:
            has_markets = bool(snapshot.get("market_count", 0))
            if args.require_informative and not snapshot.get("informative_curve"):
                return 1
            return 0 if has_markets else 1
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
