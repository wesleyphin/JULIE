import argparse
from pathlib import Path

import pandas as pd


def _default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_outrights{source.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an outright-only parquet by removing spread symbols like ESM4-ESU4."
    )
    parser.add_argument("--source", default="es_master.parquet", help="Input parquet path.")
    parser.add_argument("--out", default="", help="Output parquet path.")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")
    out = Path(args.out).expanduser() if str(args.out or "").strip() else _default_output_path(source)

    df = pd.read_parquet(source)
    if "symbol" not in df.columns:
        raise SystemExit("Parquet has no 'symbol' column.")

    symbol_text = df["symbol"].astype(str)
    spread_mask = symbol_text.str.contains(r"[-+/]", regex=True, na=False)
    cleaned = df.loc[~spread_mask].copy()
    spread_symbols = sorted(symbol_text.loc[spread_mask].dropna().unique().tolist())

    cleaned.to_parquet(out)

    print(f"source={source}")
    print(f"out={out}")
    print(f"rows_total={len(df)}")
    print(f"rows_removed={int(spread_mask.sum())}")
    print(f"rows_kept={len(cleaned)}")
    print(f"spread_symbol_count={len(spread_symbols)}")
    if spread_symbols:
        print("spread_symbols=" + ",".join(spread_symbols[:50]))


if __name__ == "__main__":
    main()
