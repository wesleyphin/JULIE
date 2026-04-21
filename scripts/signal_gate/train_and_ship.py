#!/usr/bin/env python3
"""Train the final big_loss gate on ALL 2025 data and save joblib.

Final config (picked from test_on_april2026.py):
  - target: big_loss (pnl <= -$100)
  - model: GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                       learning_rate=0.03, min_samples_leaf=30)
  - threshold: 0.35 (veto if P(big_loss) >= 0.35)
  - OOS result on April 2026: AUC 0.590, +$1,406 P&L delta, DD 4→2.
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))
from train_gate import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES,
    assemble_X,
)

OUT = ROOT / "artifacts" / "signal_gate_2025" / "model.joblib"


def main():
    df = pd.read_parquet(ROOT / "artifacts" / "signal_gate_2025" / "training_rows.parquet")
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    df = df.dropna(subset=[c for c in required if c in df.columns]).reset_index(drop=True)
    X, cat_maps = assemble_X(df)
    y = df["big_loss"].astype(int).values
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        min_samples_leaf=30, random_state=42,
    )
    clf.fit(X, y)

    payload = {
        "model": clf,
        "target": "big_loss",  # predict P(pnl <= -$100)
        "veto_threshold": 0.35,  # veto if P(big_loss) >= threshold
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
        "training_date_range": [df["day"].min(), df["day"].max()],
        "oos_april2026_result": {
            "auc": 0.590,
            "vetoed": 58,
            "vetoed_wins": 23,
            "vetoed_losses": 35,
            "vetoed_pnl_aggregate": -1406.28,
            "baseline_pnl": 511.79,
            "kept_pnl": 1918.07,
            "delta": 1406.28,
            "dd_violations_base": 4,
            "dd_violations_kept": 2,
            "oos_trades": 210,
        },
        "notes": (
            "2025 signal-gate curve-fit (filter G): separate joblib artifact "
            "for bandaid protection during the current tariff/high-vol regime. "
            "Toggle via JULIE_SIGNAL_GATE_2025=1 (off by default). "
            "Trained on 1669 iter-11 trades across 9 2025 folders; features "
            "are close-based so compatible with live ProjectX OHLCV. "
            "Target=big_loss because win/loss binary was noise (AUC 0.49) "
            "but predicting WHICH trades blow the -$100 floor was learnable. "
            "Revert path: delete this file OR set JULIE_SIGNAL_GATE_2025=0."
        ),
    }
    joblib.dump(payload, OUT)
    print(f"[write] {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print(f"  target: {payload['target']}")
    print(f"  veto_threshold: {payload['veto_threshold']}")
    print(f"  n_features: {len(payload['feature_names'])}")
    print(f"  training rows: {payload['training_rows']}")
    print(f"  OOS Apr 2026 delta: ${payload['oos_april2026_result']['delta']:+.2f}")


if __name__ == "__main__":
    main()
