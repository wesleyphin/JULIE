"""Cross-market feature extraction (Path 4 of the smartness roadmap).

Pulls and aligns companion instruments to enrich ES/MES-only signals:
  - MNQ (NQ futures): correlation + divergence vs MES
  - VIX: vol regime indicator (daily close)
  - DXY (optional): USD strength proxy

The features this module produces are intended to be consumed by
existing ML layers during retraining. The production-path integration
lives in ml_overlay_shadow.py::get_cross_market_features() which
returns an empty dict if no supporting data is available.

Current status (April 2026):
  - MNQ historical: NOT cached locally. Production pulls happen via
    the live ProjectX client; at training time we stub to zeros.
  - VIX historical: NOT cached locally; live bot does not depend on
    VIX. We ship the feature schema so future retrains can slot it in.

Public surface:
  CrossMarketFeatures.extract_at(ts_et) -> dict
      Returns a stable-schema dict of ~8 features. All keys always
      present; values default to 0.0 when supporting data is missing.

Why ship a stub: deploys the schema + integration points end-to-end so
the downstream ML layers can already train on "the feature is a
constant 0.0" without breaking, and the moment data lands the same
trainer produces an improved model with no code change.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
MNQ_PARQUET = ROOT / "data" / "mnq_master_outrights.parquet"
VIX_PARQUET = ROOT / "data" / "vix_daily.parquet"

# Stable feature schema — keys are guaranteed to be present even when
# supporting data is missing. Defaults reflect "neutral" readings.
CROSS_MARKET_FEATURE_KEYS = (
    "mnq_ret_5min_pct",       # MNQ pct return over last 5 min
    "mnq_ret_30min_pct",
    "mes_mnq_corr_30bar",     # rolling 30-bar correlation of log returns
    "mes_mnq_divergence_pct", # pct-difference in 30-min returns (MES-MNQ)
    "vix_level",              # absolute VIX level; 15=normal, 30=high fear
    "vix_regime_code",        # 0=calm(<14), 1=normal(14-20), 2=high(20-30), 3=extreme(>30)
    "vix_rate_of_change_5d",  # VIX 5-day pct change
    "dxy_level",              # DXY (currently stub; defaults to 100)
)
CROSS_MARKET_FEATURE_DEFAULTS = {
    "mnq_ret_5min_pct": 0.0,
    "mnq_ret_30min_pct": 0.0,
    "mes_mnq_corr_30bar": 0.5,       # neutral correlation ~0.5 if unknown
    "mes_mnq_divergence_pct": 0.0,
    "vix_level": 16.0,                # ~long-run VIX mean
    "vix_regime_code": 1.0,           # "normal"
    "vix_rate_of_change_5d": 0.0,
    "dxy_level": 100.0,
}


class CrossMarketFeatures:
    """Lazy-loads available cross-market data; returns a stable-schema
    feature dict for any ET timestamp."""

    def __init__(self):
        self._mnq: Optional[pd.DataFrame] = None
        self._vix: Optional[pd.DataFrame] = None
        self._tried_load = False

    def _ensure_loaded(self):
        if self._tried_load:
            return
        self._tried_load = True
        if MNQ_PARQUET.exists():
            try:
                df = pd.read_parquet(MNQ_PARQUET)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(NY)
                else:
                    df.index = df.index.tz_convert(NY)
                if "symbol" in df.columns and "volume" in df.columns:
                    df["_d"] = df.index.date
                    day_sym_vol = df.groupby(["_d", "symbol"])["volume"].sum().reset_index()
                    day_sym_vol = day_sym_vol.sort_values(["_d", "volume"], ascending=[True, False])
                    front = day_sym_vol.drop_duplicates("_d", keep="first").set_index("_d")["symbol"]
                    df["_front"] = df["_d"].map(front)
                    df = df[df["symbol"] == df["_front"]].drop(columns=["_d", "_front"])
                self._mnq = df[["close"]].sort_index()
            except Exception:
                self._mnq = None
        if VIX_PARQUET.exists():
            try:
                df = pd.read_parquet(VIX_PARQUET)
                self._vix = df[["close"]].sort_index()
            except Exception:
                self._vix = None

    def extract_at(self, ts_et, *, mes_bars: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Return a CROSS_MARKET_FEATURE_KEYS-keyed dict for timestamp ts_et.

        mes_bars (optional): DataFrame of MES bars covering ts_et-30min..ts_et.
        Used for correlation/divergence computations. If None, those features
        return defaults.
        """
        self._ensure_loaded()
        out = dict(CROSS_MARKET_FEATURE_DEFAULTS)
        # Normalize ts (accept str / datetime / Timestamp)
        if isinstance(ts_et, str):
            ts_et = pd.Timestamp(ts_et)
        elif isinstance(ts_et, datetime) and not isinstance(ts_et, pd.Timestamp):
            ts_et = pd.Timestamp(ts_et)
        if ts_et.tzinfo is None:
            ts_et = ts_et.tz_localize(NY)
        else:
            ts_et = ts_et.tz_convert(NY)

        # --- MNQ features ---
        if self._mnq is not None and len(self._mnq) > 30 and mes_bars is not None:
            try:
                pos_mnq = self._mnq.index.searchsorted(ts_et, side="right") - 1
                pos_mes = mes_bars.index.searchsorted(ts_et, side="right") - 1
                if pos_mnq >= 30 and pos_mes >= 30:
                    mnq_closes = self._mnq["close"].iloc[pos_mnq - 29: pos_mnq + 1].to_numpy()
                    mes_closes = mes_bars["close"].iloc[pos_mes - 29: pos_mes + 1].to_numpy()
                    if len(mnq_closes) == 30 and len(mes_closes) == 30:
                        mnq_rets = np.diff(mnq_closes) / mnq_closes[:-1]
                        mes_rets = np.diff(mes_closes) / mes_closes[:-1]
                        out["mnq_ret_5min_pct"] = float(
                            (mnq_closes[-1] - mnq_closes[-6]) / mnq_closes[-6] * 100.0
                        ) if mnq_closes[-6] > 0 else 0.0
                        out["mnq_ret_30min_pct"] = float(
                            (mnq_closes[-1] - mnq_closes[0]) / mnq_closes[0] * 100.0
                        )
                        if mnq_rets.std() > 1e-9 and mes_rets.std() > 1e-9:
                            corr = float(np.corrcoef(mnq_rets, mes_rets)[0, 1])
                            out["mes_mnq_corr_30bar"] = corr if np.isfinite(corr) else 0.5
                        mes_30min_pct = float(
                            (mes_closes[-1] - mes_closes[0]) / mes_closes[0] * 100.0
                        )
                        out["mes_mnq_divergence_pct"] = mes_30min_pct - out["mnq_ret_30min_pct"]
            except Exception:
                pass

        # --- VIX features ---
        if self._vix is not None and len(self._vix) > 5:
            try:
                pos = self._vix.index.searchsorted(ts_et, side="right") - 1
                if pos >= 5:
                    vix_now = float(self._vix["close"].iloc[pos])
                    vix_5d_ago = float(self._vix["close"].iloc[pos - 5])
                    out["vix_level"] = vix_now
                    if vix_now < 14:
                        out["vix_regime_code"] = 0.0
                    elif vix_now < 20:
                        out["vix_regime_code"] = 1.0
                    elif vix_now < 30:
                        out["vix_regime_code"] = 2.0
                    else:
                        out["vix_regime_code"] = 3.0
                    if vix_5d_ago > 0:
                        out["vix_rate_of_change_5d"] = (vix_now - vix_5d_ago) / vix_5d_ago * 100.0
            except Exception:
                pass

        return out


# Module-level singleton for convenient reuse
_CROSS_MARKET: Optional[CrossMarketFeatures] = None


def get_cross_market_features(ts_et, *, mes_bars=None) -> Dict[str, float]:
    """Module-level helper — reuse a singleton CrossMarketFeatures instance."""
    global _CROSS_MARKET
    if _CROSS_MARKET is None:
        _CROSS_MARKET = CrossMarketFeatures()
    return _CROSS_MARKET.extract_at(ts_et, mes_bars=mes_bars)


if __name__ == "__main__":
    # Smoke test
    cm = CrossMarketFeatures()
    import datetime as dt
    ts = dt.datetime(2025, 4, 9, 11, 0, tzinfo=NY)
    feats = cm.extract_at(ts)
    print(f"Cross-market features at {ts}:")
    for k, v in feats.items():
        print(f"  {k:<30} {v:>8.3f}")
