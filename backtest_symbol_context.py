from __future__ import annotations

from typing import Optional

import pandas as pd


def choose_symbol(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if "symbol" not in df.columns:
        raise KeyError("choose_symbol requires a 'symbol' column")
    work = _filter_backtest_outright_symbols(df)
    if preferred:
        symbols = set(work["symbol"].dropna().unique())
        if preferred in symbols:
            return preferred
    counts = work["symbol"].value_counts()
    return str(counts.index[0])


def _filter_backtest_outright_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return df
    symbol_text = df["symbol"].astype(str)
    # Ignore calendar spreads like ESM4-ESU4 and keep outright futures only.
    outright_mask = ~symbol_text.str.contains(r"[-+/]", regex=True, na=False)
    if bool(outright_mask.any()):
        filtered = df.loc[outright_mask]
        if not filtered.empty:
            return filtered
    return df


def _auto_select_symbol_by_day(
    df: pd.DataFrame,
    method: str = "volume",
) -> tuple[pd.DataFrame, dict]:
    if df.empty or "symbol" not in df.columns:
        return df, {}
    filtered_df = _filter_backtest_outright_symbols(df)
    work = filtered_df[["symbol", "volume"]].copy()
    work["date"] = work.index.date
    method_key = str(method or "volume").lower()
    if method_key == "rows" or work["volume"].isna().all():
        stats = work.groupby(["date", "symbol"]).size().rename("score").reset_index()
    else:
        stats = (
            work.groupby(["date", "symbol"])["volume"]
            .sum(min_count=1)
            .fillna(0.0)
            .rename("score")
            .reset_index()
    )
    stats = stats.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
    best = stats.drop_duplicates("date")
    day_to_symbol = dict(zip(best["date"], best["symbol"]))
    date_series = pd.Series(filtered_df.index.date, index=filtered_df.index)
    chosen = date_series.map(day_to_symbol)
    mask = filtered_df["symbol"].astype(str) == chosen.astype(str)
    return filtered_df.loc[mask], day_to_symbol


def apply_symbol_mode(
    df: pd.DataFrame,
    mode: str,
    method: str,
) -> tuple[pd.DataFrame, str, dict]:
    if df.empty or "symbol" not in df.columns:
        return df, "AUTO", {}
    unique_symbols = df["symbol"].nunique(dropna=True)
    if unique_symbols <= 1:
        symbol = str(df["symbol"].dropna().iloc[0]) if unique_symbols else "AUTO"
        return df, symbol, {}
    mode_key = str(mode or "single").lower()
    if mode_key in ("auto", "auto_by_day", "roll"):
        filtered, mapping = _auto_select_symbol_by_day(df, method=method)
        return filtered, "AUTO_BY_DAY", mapping
    return df, "AUTO", {}


def attach_backtest_symbol_context(
    df: pd.DataFrame,
    selected_symbol: Optional[str],
    symbol_mode: Optional[str],
    *,
    source_key: Optional[str] = None,
    source_label: Optional[str] = None,
    source_path: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    try:
        if selected_symbol:
            df.attrs["selected_symbol"] = str(selected_symbol).strip()
        else:
            df.attrs.pop("selected_symbol", None)
        if symbol_mode:
            df.attrs["selected_symbol_mode"] = str(symbol_mode).strip().lower()
        else:
            df.attrs.pop("selected_symbol_mode", None)
        if source_key is not None:
            source_key_text = str(source_key).strip()
            if source_key_text:
                df.attrs["source_cache_key"] = source_key_text
            else:
                df.attrs.pop("source_cache_key", None)
        if source_label is not None:
            source_label_text = str(source_label).strip()
            if source_label_text:
                df.attrs["source_label"] = source_label_text
            else:
                df.attrs.pop("source_label", None)
        if source_path is not None:
            source_path_text = str(source_path).strip()
            if source_path_text:
                df.attrs["source_path"] = source_path_text
            else:
                df.attrs.pop("source_path", None)
    except Exception:
        pass
    return df
