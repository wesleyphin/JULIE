from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_mlphysics_topstep_combo import (  # noqa: E402
    bootstrap_monte_carlo,
    summarize_best_areas,
    summarize_best_combos,
    trade_metrics,
)


def _resolve_path(path_arg: str) -> Path:
    candidate = Path(path_arg).expanduser()
    if candidate.exists():
        return candidate
    fallback = ROOT / path_arg
    if fallback.exists():
        return fallback
    raise SystemExit(f"Path not found: {path_arg}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _select_area_only(trades: pd.DataFrame, *, min_trades: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    areas = summarize_best_areas(trades, min_trades=int(min_trades))
    areas = areas.loc[(areas["net_points"] > 0.0) & (areas["avg_points"] > 0.0)].reset_index(drop=True)
    if areas.empty:
        return areas, trades.iloc[0:0].copy()
    filtered = trades.merge(
        areas[["macro_name", "session_window", "open_ref_name", "side"]],
        on=["macro_name", "session_window", "open_ref_name", "side"],
        how="inner",
    )
    return areas, filtered.reset_index(drop=True)


def _select_combo_only(trades: pd.DataFrame, *, min_trades: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    combos = summarize_best_combos(trades, min_trades=int(min_trades))
    combos = combos.loc[(combos["net_points"] > 0.0) & (combos["avg_points"] > 0.0)].reset_index(drop=True)
    if combos.empty:
        return combos, trades.iloc[0:0].copy()
    filtered = trades.merge(
        combos[["combo_key", "side"]],
        on=["combo_key", "side"],
        how="inner",
    )
    return combos, filtered.reset_index(drop=True)


def _variant_payload(
    *,
    name: str,
    selection_type: str,
    threshold_config: dict[str, Any],
    selection_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    sims: int,
    seed: int,
    note: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "selection_type": selection_type,
        "config": threshold_config,
        "selection_count": int(len(selection_df)),
        "summary": trade_metrics(trades_df),
        "monte_carlo": bootstrap_monte_carlo(trades_df, simulations=int(sims), seed=int(seed)),
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build higher-volume bank-confluence engine variants from an existing OOS run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--monte-carlo-sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=290042)
    args = parser.parse_args()

    run_dir = _resolve_path(args.run_dir)
    oos_trades_path = run_dir / "oos_trades.csv"
    if not oos_trades_path.exists():
        raise SystemExit(f"Missing OOS trades file: {oos_trades_path}")
    trades = pd.read_csv(oos_trades_path)
    if trades.empty:
        raise SystemExit("OOS trades file is empty.")

    area_only_min1_df, area_only_min1_trades = _select_area_only(trades, min_trades=1)
    combo_only_min1_df, combo_only_min1_trades = _select_combo_only(trades, min_trades=1)
    combo_only_min2_df, combo_only_min2_trades = _select_combo_only(trades, min_trades=2)

    variants = [
        _variant_payload(
            name="balanced_existing",
            selection_type="existing_stability_shortlist",
            threshold_config={"source": "existing_stability_output"},
            selection_df=pd.DataFrame(),
            trades_df=pd.read_csv(run_dir / "stability_pruned_oos_trades.csv"),
            sims=int(args.monte_carlo_sims),
            seed=int(args.seed),
            note="Current balanced shortlist from the trainer's stability selector.",
        ),
        _variant_payload(
            name="volume_area_only_min1",
            selection_type="area_only",
            threshold_config={"area_min_trades": 1},
            selection_df=area_only_min1_df,
            trades_df=area_only_min1_trades,
            sims=int(args.monte_carlo_sims),
            seed=int(args.seed) + 101,
            note="Safer higher-volume variant. Keeps only positive OOS areas with at least one trade.",
        ),
        _variant_payload(
            name="volume_combo_only_min1",
            selection_type="combo_only",
            threshold_config={"combo_min_trades": 1},
            selection_df=combo_only_min1_df,
            trades_df=combo_only_min1_trades,
            sims=int(args.monte_carlo_sims),
            seed=int(args.seed) + 201,
            note="Highest-flow aggressive variant. Includes positive OOS combos with at least one trade, so it is less robust than area-level selection.",
        ),
        _variant_payload(
            name="volume_combo_only_min2",
            selection_type="combo_only",
            threshold_config={"combo_min_trades": 2},
            selection_df=combo_only_min2_df,
            trades_df=combo_only_min2_trades,
            sims=int(args.monte_carlo_sims),
            seed=int(args.seed) + 301,
            note="Moderately robust combo-only variant. Requires at least two OOS trades per combo.",
        ),
    ]

    payload = {
        "engine_name": "ml_physics_topstep_bank_confluence_variants",
        "generated_from_run": str(run_dir),
        "generated_on": pd.Timestamp.now(tz="UTC").isoformat(),
        "primary_variant": "volume_combo_only_min1",
        "safer_variant": "volume_area_only_min1",
        "variants": variants,
    }

    (run_dir / "bank_confluence_volume_variants.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    area_only_min1_df.to_csv(run_dir / "volume_area_only_min1_areas.csv", index=False)
    area_only_min1_trades.to_csv(run_dir / "volume_area_only_min1_trades.csv", index=False)
    combo_only_min1_df.to_csv(run_dir / "volume_combo_only_min1_combos.csv", index=False)
    combo_only_min1_trades.to_csv(run_dir / "volume_combo_only_min1_trades.csv", index=False)
    combo_only_min2_df.to_csv(run_dir / "volume_combo_only_min2_combos.csv", index=False)
    combo_only_min2_trades.to_csv(run_dir / "volume_combo_only_min2_trades.csv", index=False)

    latest_path = run_dir.parent / "latest_volume.json"
    latest_path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"Wrote variants to {run_dir / 'bank_confluence_volume_variants.json'}")
    print(f"Wrote latest volume manifest to {latest_path}")


if __name__ == "__main__":
    main()
