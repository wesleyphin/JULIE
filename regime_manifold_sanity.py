import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CONFIG
from regime_manifold_engine import RegimeManifoldEngine


TIME_COLUMNS = [
    "ts_event",
    "ts_recv",
    "ts",
    "timestamp",
    "datetime",
    "time",
    "date",
    "Timestamp",
    "Datetime",
    "Time",
    "Date",
]


def get_session_name(ts: pd.Timestamp) -> str:
    hour = ts.hour
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


def load_bars(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    time_col: Optional[str] = None
    for name in TIME_COLUMNS:
        if name in df.columns:
            time_col = name
            break

    if time_col is None:
        raise ValueError(
            "Could not find a timestamp column. Expected one of: "
            + ", ".join(TIME_COLUMNS)
        )

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    df.columns = [str(c).lower() for c in df.columns]

    required = {"open", "high", "low", "close"}
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    return df


def build_engine() -> RegimeManifoldEngine:
    cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
    bt_override = CONFIG.get("BACKTEST_REGIME_MANIFOLD", {}) or {}
    if isinstance(bt_override, dict):
        cfg.update(bt_override)
    cfg["enabled"] = True
    return RegimeManifoldEngine(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick regime manifold sanity runner.")
    parser.add_argument("--csv", default="es_master.csv", help="Path to 1-minute OHLC CSV.")
    parser.add_argument("--rows", type=int, default=1200, help="Bars from the tail to process.")
    parser.add_argument("--tail", type=int, default=60, help="Last N regime rows to print.")
    args = parser.parse_args()

    path = Path(args.csv).expanduser().resolve()
    df = load_bars(path)
    if args.rows > 0 and len(df) > args.rows:
        df = df.iloc[-int(args.rows):]

    engine = build_engine()
    rows = []
    for idx in range(len(df)):
        history = df.iloc[: idx + 1]
        ts = history.index[-1]
        meta = engine.update(history, ts=ts, session=get_session_name(ts))
        rows.append(
            {
                "ts": ts,
                "regime": meta.get("regime"),
                "R": round(float(meta.get("R", 0.0) or 0.0), 4),
                "stress": round(float(meta.get("stress", 0.0) or 0.0), 4),
                "risk_mult": round(float(meta.get("risk_mult", 1.0) or 1.0), 3),
                "side_bias": int(meta.get("side_bias", 0) or 0),
                "no_trade": bool(meta.get("no_trade", False)),
            }
        )

    out = pd.DataFrame(rows)
    tail_n = max(1, int(args.tail))
    print(out.tail(tail_n).to_string(index=False))


if __name__ == "__main__":
    main()
