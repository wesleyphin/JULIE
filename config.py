import datetime
import os
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

try:
    from config_secrets import SECRETS
except Exception:
    SECRETS = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# Core runtime configuration for the bot. This was pulled directly from the
# original julie001.py entrypoint so other modules can import it without
# bringing in the full bot runtime.
CONFIG = {
    # --- CREDENTIALS ---
    "USERNAME": str(SECRETS.get("USERNAME", "") or ""),
    "API_KEY": str(SECRETS.get("API_KEY", "") or ""),

    # --- ACCOUNT/CONTRACT (will be fetched dynamically) ---
    "ACCOUNT_ID": os.environ.get("JULIE_ACCOUNT_ID"),  # Can be set via env var or fetched via /Account/search
    "CONTRACT_ID": None,  # Fetched via /Contract/available (e.g., "CON.F.US.MES.H25")
    "CONTRACT_ROOT": "MES",  # Symbol root used to determine current MES contract (e.g., CON.F.US.MES.Z25)
    "TARGET_SYMBOL": None,  # Determined dynamically from date and CONTRACT_ROOT

    # --- API ENDPOINTS (ProjectX Gateway LIVE) ---
    # Switched from 'gateway-api-demo' to 'gateway-api'
    "REST_BASE_URL": "https://api.topstepx.com",
    "RTC_USER_HUB": "https://rtc.topstepx.com/hubs/user",
    "RTC_MARKET_HUB": "https://rtc.topstepx.com/hubs/market",
    # ProjectX user hub: event-driven account/position/trade updates.
    "PROJECTX_USER_STREAM_ENABLED": True,
    "PROJECTX_USER_STREAM_TRADE_CACHE": 256,
    "PROJECTX_USER_STREAM_MAX_POSITION_AGE_SEC": 15.0,
    "PROJECTX_USER_STREAM_MAX_ACCOUNT_AGE_SEC": 300.0,
    # Retry after external/browser session conflicts instead of yielding forever.
    "PROJECTX_EXTERNAL_SESSION_RETRY_SEC": 20.0,

    # --- SYSTEM SETTINGS ---
    "MAX_DAILY_LOSS": 1000.0,
    "TIMEZONE": "US/Eastern",
    # Live data logging:
    # Disable the old MES 1m CSV appender by default (large, low-signal logs).
    "LIVE_MES_CSV_APPENDER_ENABLED": False,
    "LIVE_MES_CSV_PATH": "ml_mes_et.csv",
    # Trade-factor logger (live): writes one row per filled trade with rich context.
    "LIVE_TRADE_FACTORS_LOGGER_ENABLED": True,
    "LIVE_TRADE_FACTORS_CSV_PATH": "live_trade_factors.csv",
    # Context window in bars for context_box rollups (240 bars ~= 4h on 1-minute bars).
    "LIVE_TRADE_FACTORS_CONTEXT_BARS": 240,
    # Safety cap for numeric columns summarized into context_box.
    "LIVE_TRADE_FACTORS_MAX_NUMERIC_COLS": 300,

    # --- RISK & FEE MANAGEMENT ---
    "RISK": {
        "POINT_VALUE": 5.0,      # $ value per point per contract (MES=$5, ES=$50)
        # Per-side fees PER CONTRACT. Example: if your total (open+close) fees are $3.70
        # for 5 contracts, that's $3.70 / (5*2) = $0.37 per side per contract.
        "FEES_PER_SIDE": 0.37,
        # Reporting-only add-on for Topstep commission. This is kept separate from
        # the base fee proxy so live PnL/blotter math can reflect broker charges
        # without changing strategy gating or EV calculations unless we choose to.
        "TOPSTEP_COMMISSION_ROUND_TURN_PER_CONTRACT": 0.50,
        "MIN_NET_PROFIT": 10.0,  # Minimum expected profit (after round-trip fees) to take a trade
        "ENFORCE_MIN_NET_PROFIT": False,  # If False, don't block candidates based on fees
        "CONTRACTS": 1           # Number of contracts traded
    },
    # --- SL/TP MINIMUMS (points) ---
    "SLTP_MIN": {
        "sl": 1.25,  # Avoid micro-noise stops on MES 1m
        "tp": 1.50,
    },
    # --- STRATEGY EXECUTION DISABLES ---
    # Strategies can still emit signals (for analysis/diagnostics), but execution is blocked.
    "STRATEGY_EXECUTION_DISABLED": [
        "LiquiditySweep",
    ],
    # Per-strategy session disables (execution blocked only in these sessions).
    "STRATEGY_EXECUTION_DISABLED_BY_SESSION": {},
    # --- FILTER/STRATEGY TOGGLES ---
    "HTF_FVG_FILTER": {
        "enabled_live": False,
        "enabled_backtest": False,
    },
    "ICT_MODEL": {
        "enabled_live": False,
        "enabled_backtest": False,
    },
    # Keep UI payload schema unchanged (lists/strings), but cap strategy slot fill.
    "UI_STRATEGY_SLOT_LIMIT": 8,
    # --- STRATEGY PROFILE MAP FOR PRE-CANDIDATE GATING ---
    # Profiles are used by ASIA viability + DynamicChop pre-candidate policies.
    # Unknown strategies fall back to default_profile.
    "STRATEGY_GATE_PROFILES": {
        "default_profile": "momentum_breakout",
        "map": {
            "DynamicEngine": "momentum_breakout",
            "DynamicEngine3": "momentum_breakout",
            "ImpulseBreakout": "momentum_breakout",
            "ORB": "momentum_breakout",
            "Confluence": "momentum_breakout",
            "ICTModel": "momentum_breakout",
            "RegimeAdaptive": "mean_reversion",
            "AuctionReversion": "mean_reversion",
            "IntradayDip": "mean_reversion",
            "SMTAnalyzer": "mean_reversion",
            "LiquiditySweep": "mean_reversion",
            "VIXMeanReversion": "mean_reversion",
            "ValueAreaBreakout": "momentum_breakout",
            "SmoothTrendAsia": "momentum_breakout",
            "MLPhysics": "ml_adaptive",
            "ManifoldStrategy": "ml_adaptive",
            "AetherFlowStrategy": "ml_adaptive",
            "Continuation": "momentum_breakout",
        },
    },
    # --- DYNAMIC CHOP PRE-CANDIDATE GATE ---
    # global: old behavior (hard chop blocks all candidates)
    # per_strategy: allow configured profiles during hard chop
    # off: disable pre-candidate chop gating (legacy downstream filters still run)
    "DYNAMIC_CHOP_GATE": {
        "mode": "per_strategy",
        "allow_profiles_in_hard_chop": ["mean_reversion"],
        # Keep false to avoid double-blocking with existing downstream range-bias checks.
        "enforce_range_bias_pre_candidate": False,
    },
    # --- PENALTY BOX BLOCKER ---
    "PENALTY_BOX": {
        "enabled": True,
        "lookback": 50,
        "tolerance": 1.0,
        "penalty_bars": 3,
    },
    # --- FIXED SL/TP FRAMEWORK (regime-based brackets) ---
    "FIXED_SLTP_FRAMEWORK": {
        "enabled": True,
        "tick_size": 0.25,
        "default_bracket": "NORMAL_TREND",
        "session_overrides": {
            "ASIA": "ASIA_SMOOTH",
        },
        "vol_regime_brackets": {
            "high": "IMPULSE",
        },
        "brackets": {
            "ASIA_SMOOTH": {"SL": 1.75, "TP": 2.00},
            "NORMAL_TREND": {"SL": 2.25, "TP": 2.75},
            "IMPULSE": {"SL": 3.00, "TP": 4.50},
        },
        "viability": {
            "enabled": True,
            "atr_window": 20,
            "atr_floor": 0.70,
            "lookback_bars": 60,
            "room_to_target_factor": 0.8,
            "room_to_target_min_points": 1.5,
            "room_to_target_k": 1.2,
            "room_to_target_atr_window": 20,
            "disable_room_to_target": False,
            "disable_room_to_target_strategies": ["ValueAreaBreakout"],
            "strategy_overrides": {
                "DynamicEngine3": {
                    "use_signal_sltp": True,
                },
                "ValueAreaBreakout": {
                    "room_reference": "continuation",
                    "room_lookback_minutes": 60,
                    "room_profile_lookback_bars": 390,
                    "room_profile_value_area_pct": 0.70,
                    "room_fallback_atr_mult": 0.8,
                    "room_to_target_k": 0.9,
                    "use_sl_override": True,
                    # Disable the continuation-room gate for now (it filtered too many
                    # profitable VAB trades); VAB quality is handled by the ASS system.
                    "enable_room_to_target_sessions": [],
                },
                "ORB": {
                    "room_to_target_factor": 0.6,
                    "room_to_target_min_points": 0.75,
                    "room_to_target_k": 0.9,
                },
            },
            "session_overrides": {
                "ASIA": {
                    "room_to_target_min_points": 0.75,
                    "room_to_target_k": 0.8,
                    "room_to_target_atr_window": 20,
                },
            },
            "vol_regime_overrides": {},
            "runtime_overrides": {},
        },
    },
    # --- ASIA VIABILITY GATE (must pass to trade in ASIA) ---
    "ASIA_VIABILITY_GATE": {
        "enabled": True,
        # global: old behavior (bar-level kill switch)
        # per_strategy: only block configured profiles when not viable
        # off: disable ASIA viability pre-candidate gating
        "mode": "per_strategy",
        "block_profiles_when_not_viable": ["momentum_breakout", "ml_adaptive"],
        # Enforce ASIA trend-bias direction for specified profiles.
        "enforce_trend_bias": True,
        "enforce_bias_profiles": ["momentum_breakout"],
        "min_bars": 120,
        # Option A: ATR expansion
        "atr_ratio_fast": 5,
        "atr_ratio_slow": 60,
        "atr_ratio_min": 1.25,
        # Option B: Compression -> Release
        "compression_atr_window": 30,
        "compression_percentile": 20,
        "compression_lookback": 200,
        "compression_release_atr_window": 20,
        "compression_range_mult": 1.2,
        # Option C: Structural interaction
        "interaction_tol_points": 0.25,
        "ny_close_hour": 16,
        "ny_close_minute": 0,
        "vp_lookback_bars": 390,
        "vp_value_area_pct": 0.70,
        "tick_size": 0.25,
        "max_history_bars": 3000,
        "asia_session_start_hour": 18,
        "asia_session_end_hour": 3,
        "use_ny_close": True,
        "use_value_area": True,
        "use_asia_sweep": True,
    },
    # --- ASIA EXTENSION FILTER SOFT PENALTY ---
    "ASIA_SOFT_EXTENSION_FILTER": {
        "enabled": True,
        "base_score": 1.0,
        "penalty": 0.35,
        "score_threshold": 0.65,
    },
    # --- NEW STRATEGIES (Impulse/Auction/LIQ Sweep/Value Area) ---
    "IMPULSE_BREAKOUT": {
        "enabled": True,
        "lookback": 20,
        "range_mult": 1.5,
        "volume_mult": 1.2,
        "breakout_buffer_atr": 0.10,
        "atr_window": 20,
        "min_range": 0.75,
        "require_trend": True,
        "ema_fast": 20,
        "ema_slow": 50,
        "ema_slope_bars": 5,
        "min_ema_separation": 0.0,
        "min_ema_separation_atr": 0.10,
        "min_body_ratio": 0.60,
        "close_position_ratio": 0.70,
        "atr_range_mult": 1.0,
        "sessions": ["NY_AM", "NY_PM"],
    },
    "AUCTION_REVERSION": {
        "enabled": True,
        "lookback": 120,
        "value_area_pct": 0.70,
        "touch_buffer": 0.35,
        "touch_buffer_atr": 0.10,
        "require_rejection": True,
        "rejection_close_buffer": 0.0,
        "rejection_close_buffer_atr": 0.10,
        "er_window": 30,
        "er_max": 0.14,
        "min_range": 5.0,
        "sessions": ["NY_AM", "NY_PM", "LONDON"],
        "skip_high_vol": False,
        "tick_size": 0.25,
        "atr_window": 20,
        # Quality guards to keep reversion trades out of one-way moves.
        "rejection_wick_min": 0.25,
        "rejection_wick_atr": 0.30,
        "volume_mult": 1.15,
        "trend_ema_period": 50,
        "long_only_above_ema": True,
        "short_only_below_ema": True,
        "cooldown_bars": 8,
    },
    "LIQUIDITY_SWEEP": {
        "enabled": True,
        "lookback": 20,
        "atr_window": 20,
        "sweep_buffer_atr": 0.10,
        "reclaim_buffer_atr": 0.05,
        "min_wick_atr": 0.20,
        "volume_mult": 1.10,
        "sessions": ["NY_AM", "NY_PM", "LONDON", "ASIA"],
        "use_pivots": True,
        "pivot_window": 2,
        "pivot_max_age": 80,
        "pivot_fallback_to_lookback": False,
        "confirm_followthrough": True,
        "confirm_bars": 1,
        "min_sweep_points": 0.50,
        "min_reclaim_points": 0.0,
        "min_wick_points": 0.25,
        "max_bar_range_atr": 1.50,
        "cooldown_bars": 5,
        "require_new_pivot": True,
        "allowed_regimes": ["low", "normal"],
    },
    "VALUE_AREA_BREAKOUT": {
        "enabled": True,
        "lookback": 120,
        "value_area_pct": 0.70,
        "accept_bars": 2,
        # Side-specific acceptance strictness (SHORT needs stronger proof).
        "accept_bars_long": 2,
        "accept_bars_short": 3,
        "buffer": 0.10,
        "buffer_atr": 0.10,
        "er_window": 30,
        "er_min": 0.25,
        "min_range": 4.0,
        "atr_window": 20,
        "close_position_ratio": 0.60,
        "close_position_ratio_long": 0.62,
        "close_position_ratio_short": 0.72,
        "volume_mult": 1.0,
        "volume_mult_long": 1.0,
        "volume_mult_short": 1.15,
        # London has been the weakest VAB pocket; keep NY-focused deployment.
        "sessions": ["NY_AM", "NY_PM"],
        "trend_ema_period": 50,
        "long_require_above_ema": False,
        "short_require_below_ema": True,
        "cooldown_bars": 4,
        "trigger_on_transition": True,
        "tick_size": 0.25,
        "allowed_regimes": ["normal"],
        "sl_tighten_mult": 0.75,
        "min_stop_ticks": 5,
        "max_stop_ticks": 0,
        # VAB Acceptance Strength Score (ASS) - compute a pre-entry quality score and
        # (optionally) choose between bracket templates or block. Default is
        # instrumentation-only so backtests stay comparable until you flip to trade mode.
        "ass": {
            "enabled": True,
            # "instrument": compute score + emit fields; do not block/alter SL/TP.
            # "trade": enforce thresholds; choose DIAG vs SURV; block if below survival.
            "mode": "trade",
            "window_closes": 4,
             "thresholds": {
                 # Stronger sessions can tolerate lower thresholds; weaker sessions demand
                 # more evidence to justify a survival bracket.
                 "NY_PM": {"diagnostic": 70, "survival": 45},
                 # NY_AM: survival brackets have been a major loss driver; require
                 # "diagnostic-level" strength or block.
                 "NY_AM": {"diagnostic": 80, "survival": 80, "force": "DIAG"},
                 # LONDON: tight DIAG stops get chopped; force SURV if it passes.
                 "LONDON": {"diagnostic": 80, "survival": 55, "force": "SURV"},
                 "default": {"diagnostic": 75, "survival": 50},
             },
            # If we see deep retest pressure at entry, downgrade DIAG->SURV.
            "retest_downgrade_atr": 0.45,
            # Multiplier applied to baseline SL when template is SURV (trade mode only).
            "survival_sl_mult": 1.0,
            # Make SHORT breakouts pass a meaningfully higher quality bar.
            "short_penalty": 8.0,
            "log_decisions": False,
            # Optional list of time-of-day windows to penalize (no hardcoding here).
            "risk_windows": [],
        },
    },

    # --- ML SESSION-BASED STRATEGY SETTINGS ---
    "WINDOW_SIZE": 15,
    "ML_PHYSICS_TIMEFRAME_MINUTES": 5,
    # How to align MLPhysics evaluation with higher-timeframe bars:
    # "open"  -> evaluate on the first 1m bar of each tf bucket (uses previous full tf bar)
    # "close" -> evaluate on the last 1m bar of each tf bucket
    "ML_PHYSICS_BAR_ALIGNMENT": "open",
    # Backtest optimization pipeline for MLPhysics:
    # - precompute features once
    # - batch predictions
    # - parquet cache
    # Live mode remains unchanged unless explicitly set to "live".
    "ML_PHYSICS_OPT": {
        "enabled": True,
        "mode": "backtest",  # backtest | live
        "feature_cache_dir": "cache/ml_physics/",
        "prediction_cache": True,
        "overwrite_cache": False,
        # Optional explicit dist cache file. Leave blank by default and rely on
        # deterministic window-specific cache files to avoid stale cross-window mismatches.
        "dist_precomputed_file": "",
        # If True and explicit file is configured, do not fall back to broad cache search.
        "dist_precomputed_strict": False,
        # Allow cache attach when index is a high-overlap near-match, filling gaps as no-signal.
        "dist_precomputed_allow_partial": True,
        "dist_precomputed_allow_partial_max_missing": 2000,
        "dist_precomputed_allow_partial_max_ratio": 0.01,
        "dist_precomputed_allow_partial_min_coverage": 0.98,
    },
    # Backtest startup optimization: defer dist model loading until needed.
    # With cached dist signals, startup can skip loading the full dist model bundle.
    "ML_PHYSICS_DIST_LAZY_LOAD_BACKTEST": True,
    # Optional: use ML training output files to override session thresholds
    "ML_PHYSICS_THRESHOLDS_FILE": "ml_physics_thresholds.json",
    "ML_PHYSICS_METRICS_FILE": "ml_physics_metrics.json",
    # Isolated pre-dist MLPhysics experiment stack. This keeps the older
    # joblib-based workflow separate from the live dist runtime.
    "ML_PHYSICS_LEGACY_EXPERIMENT": {
        "artifact_dir": "artifacts/ml_physics_legacy_experiment",
        "thresholds_file": "ml_physics_thresholds.json",
        "metrics_file": "ml_physics_metrics.json",
        "timeframe_minutes": 5,
        "min_history_bars": 200,
        "sltp_strategy_name": "Generic",
        "min_sl_points": 4.0,
        "min_tp_points": 6.0,
    },
    # Use dist_bracket_ml inference artifacts for MLPhysics (live + backtest).
    "ML_PHYSICS_REPLACE_WITH_DIST": True,
    "ML_PHYSICS_DIST_RUN_BASE_DIR": "dist_bracket_ml_runs",
    # Pin runtime to the validated run with trained gate thresholds/models.
    "ML_PHYSICS_DIST_RUN_DIR": "runpod_results/restored2_20260225_210635/dist_bracket_ml_runs/ml_physics_cut20241231/dist_bracket_20260224_041706_unknown",
    # Dist inference efficiency: cap bars passed to runtime inference per evaluation.
    # 3000 bars preserves full intraday context while avoiding quadratic feature rebuild cost.
    "ML_PHYSICS_DIST_MAX_BARS": 3000,
    # Backtest/live dist XGBoost inference tuning.
    # Note: exact GPU utilization percentage cannot be hard-guaranteed from Python.
    # target_fraction controls host-side feed threads aimed at roughly 75% load balance.
    "ML_PHYSICS_DIST_XGB_GPU_ENABLED": True,
    "ML_PHYSICS_DIST_XGB_GPU_TARGET_FRACTION": 0.75,
    "ML_PHYSICS_DIST_XGB_DEVICE": "cuda",
    "ML_PHYSICS_DIST_XGB_PREDICTOR": "gpu_predictor",
    # Runtime clamp for dist gate thresholds.
    # Purpose: keep tradeability gate usable under inference-time distribution shift.
    # This does not retrain models; it only bounds loaded gate thresholds.
    "ML_PHYSICS_DIST_GATE_THRESHOLD_CLAMP": {
        "enabled": False,
        "default": {"min": 0.35, "max": 0.50},
        "sessions": {
            "ASIA": {
                "LONG": {"max": 0.44},
                "SHORT": {"max": 0.44},
            },
            "LONDON": {
                "LONG": {"max": 0.45},
                "SHORT": {"max": 0.45},
            },
            "NY_AM": {
                "LONG": {"max": 0.49},
                "SHORT": {"max": 0.44},
            },
            "NY_PM": {
                "LONG": {"max": 0.45},
                "SHORT": {"max": 0.44},
            },
        },
    },
    # Dist-mode additional gate strictness: require p_take - gate_threshold >= margin.
    # Supports {"default": float, "sessions": {"NY_PM": float, ...}}.
    "ML_PHYSICS_DIST_MIN_GATE_MARGIN": {
        "default": 0.0,
        "sessions": {
            "NY_PM": 0.03,
        },
    },
    # Runtime dist filter: require strictly better than this RR after bracket conversion.
    # Keep this available as a safety valve, but do not enforce it globally. The broad
    # March 17, 2026 runs showed the blanket RR gate was the main reason trade count
    # collapsed without materially helping the stronger profitable run.
    "ML_PHYSICS_DIST_RUNTIME_MIN_RR": {
        "enabled": False,
        "default": 1.5,
        "sessions": {},
    },
    # Runtime dist filter: when a signal leans on the configured minimum bracket leg,
    # require a stronger gate/confidence score instead of auto-accepting floor-sized trades.
    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {
        "enabled": True,
        "default": 0.0,
        "sessions": {
            "ASIA": 0.46,
        },
    },
    # Runtime dist filter: require a minimum absolute EV before the signal is tradable.
    # This trims low-edge churn while still keeping session coverage across the day.
    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {
        "enabled": True,
        "default": 0.25,
        "sessions": {
            "ASIA": 1.0,
            "LONDON": 1.0,
            "NY_AM": 2.0,
            "NY_PM": 0.25,
        },
    },
    # Oversized dist brackets behave like overnight runner trades rather than normal
    # intraday TP/SL brackets. Keep them constrained to the broader ASIA high-vol
    # regime, but avoid exact-hour fitting so the model can adapt within session.
    "ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER": {
        "enabled": True,
        "wide_tp_min": 30.0,
        "wide_sl_min": 20.0,
        "sessions": ["ASIA"],
        "require_high_vol": True,
        "allowed_regimes": ["high"],
        "min_confidence": 0.44,
        "min_ev_abs": 50.0,
    },
    # Structural split for non-runner intraday trades. Keep the wide ASIA runner regime
    # separate, and use a simpler session/side policy for the normal book so we avoid
    # tuning a large stack of brittle numeric thresholds. Regime gating is only used
    # where the loser is broad and persistent across windows.
    "ML_PHYSICS_DIST_NORMAL_PROFILE": {
        "enabled": True,
        "default_allowed_sides": ["LONG", "SHORT"],
        "sessions": {
            "ASIA": {
                "allowed_sides": ["LONG"],
                "blocked_regimes": ["low", "normal"],
            },
            "LONDON": {
                "allowed_sides": ["LONG", "SHORT"],
                "blocked_regimes": ["normal"],
            },
            "NY_AM": {"allowed_sides": []},
            "NY_PM": {"allowed_sides": ["LONG"]},
        },
    },
    # Optional late-stage execution policy for MLPhysics dist signals. Unlike the
    # normal-profile rules, this applies after the underlying signal is generated,
    # so research variants can isolate session/side/EV/gate-prob subsets without
    # changing the upstream signal stream.
    "ML_PHYSICS_DIST_ENTRY_POLICY": {
        "enabled": False,
        "default_allowed_sides": None,
        "default_allowed_regimes": None,
        "default_allow_runner": None,
        "default_allowed_weekdays": None,
        "default_min_ev_abs": None,
        "default_min_gate_prob": None,
        "default_max_gate_prob": None,
        "sessions": {},
    },
    # For non-runner trades, use MLPhysics dist primarily as a direction model and keep
    # bracket selection on a small, coarse session/regime grid. This is intentionally
    # simple to improve stability without turning the normal book into another
    # high-dimensional tuning problem.
    "ML_PHYSICS_DIST_NORMAL_BRACKET_POLICY": {
        "enabled": True,
        "default": {"sl": 1.25, "tp": 1.5},
        "sessions": {
            "ASIA": {
                "default": {"sl": 1.25, "tp": 1.5},
                "regimes": {
                    "high": {"sl": 1.75, "tp": 3.0},
                    "normal": {"sl": 1.25, "tp": 1.5},
                },
            },
            "LONDON": {
                "default": {"sl": 1.25, "tp": 1.5},
                "regimes": {
                    "high": {"sl": 1.5, "tp": 3.0},
                    "normal": {"sl": 1.25, "tp": 1.5},
                    "low": {"sl": 1.25, "tp": 1.5},
                },
            },
            "NY_PM": {
                "default": {"sl": 1.25, "tp": 2.0},
                "regimes": {
                    "high": {"sl": 1.25, "tp": 2.5},
                },
            },
        },
    },
    # Backtest-only alias for dist input cap.
    "BACKTEST_ML_DIST_INPUT_BARS": 3000,
    # Backtest target for host-side GPU feed in dist XGBoost inference (best effort).
    "BACKTEST_GPU_TARGET_FRACTION": 0.75,
    # Filterless live roster overrides consumed by the live runtime and dashboard.
    # Canonical values: dynamic_engine3, regime_adaptive, ml_physics, aetherflow.
    # ml_physics is now hour-10-11 gated by ML model (see julie001.py
    # _mlphysics_h1011_decide). Set JULIE_ML_PHYSICS_H1011_ML=0 to disable.
    "FILTERLESS_LIVE_DISABLED_STRATEGIES": [],
    # Optional alternate artifact profile used to train/evaluate a fixed historical window.
    "EXPERIMENTAL_TRAINING": {
        # Runtime remains on full-data artifacts unless explicitly enabled.
        "enabled_runtime": False,
        # Experimental train window (inclusive calendar dates).
        "start": "2011-01-01",
        "end": "2017-12-31",
        # Suffix appended before extension, e.g. model.joblib -> model_exp2011_2017.joblib
        "artifact_suffix": "_exp2011_2017",
    },
    # ManifoldStrategy artifacts (trained by train_manifold_strategy.py).
    # This uses manifold as predictive context, not as a hard execution gate.
    "MANIFOLD_STRATEGY": {
        "enabled_live": False,
        "enabled_backtest": True,
        # Backtest execution policy:
        # True -> apply only hard filters (FixedSLTP, TargetFeasibility, VolatilityGuardrail).
        # False -> run through the full shared filter stack.
        "backtest_hard_filters_only": True,
        "model_file": "model_manifold_strategy_clean_full_robust_v2.joblib",
        "thresholds_file": "manifold_strategy_thresholds_clean_full_robust_v2.json",
        "confluence_file": "manifold_strategy_confluence_clean_full_robust_v2.json",
        "min_bars": 250,
        "min_confidence": 0.60,
        "size": 5,
        "sl_points": 4.0,
        "tp_points": 8.0,
        "sl_atr_mult": 1.25,
        "tp_atr_mult": 2.0,
        "respect_no_trade": True,
        # no_trade handling:
        # - hard: block signals whenever manifold meta says no_trade.
        # - soft: keep signals with reduced size; still hard-blocks extreme stress/regimes.
        # - off: ignore manifold no_trade flag entirely.
        "no_trade_policy": "soft",
        "no_trade_conf_boost": 0.00,
        "no_trade_size_mult": 0.50,
        "no_trade_stress_hard": 0.92,
        "no_trade_block_regimes": ["ROTATIONAL_TURBULENCE"],
        "log_evals": True,
        # Optional formula-style confluence scalar applied to probability/size.
        "confluence_enabled": False,
        "allowed_session_ids": [1, 2, 3],  # LONDON, NY_AM, NY_PM
        "use_confluence_for_size": True,
        "size_scale_min": 0.5,
        "size_scale_max": 2.0,
        # Optional overrides passed into RegimeManifoldEngine used by the strategy.
        "manifold_params": {},
    },
    # AetherFlow live runtime is pinned to the 2026-04-24 handoff:
    # corrected full manifold base plus the promoted radical AF NY_AM trend
    # routed ensemble. Keep this aligned with LIVE_RUNTIME_HANDOFF_20260424.md
    # and configs/aetherflow_current_live_policy_20260421.json.
    "AETHERFLOW_STRATEGY": {
        "enabled_live": True,
        "enabled_backtest": False,
        "backtest_hard_filters_only": True,
        "model_file": "artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/model.pkl",
        "thresholds_file": "artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/thresholds.json",
        "metrics_file": "artifacts/aetherflow_routed_ensemble_candidates_20260422/radical_af_nyam_trend_v1/metrics.json",
        "backtest_base_features_file": "artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet",
        "live_base_features_file": "artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet",
        "min_bars": 320,
        "threshold_override": 0.55,
        "min_confidence": 0.50,
        "size": 5,
        # Allow one same-direction AetherFlow add-on live, with its own bracket.
        # Research favored a 2-leg cap over 3 legs for a cleaner risk/robustness balance.
        "live_same_side_parallel_max_legs": 2,
        # Let AetherFlow press when it is entering alone, but keep it at
        # base size when it is stacking same-side alongside another live leg.
        # This is applied in the live bot after strategy sizing and before the
        # live drawdown cap, so existing risk guardrails still get the final say.
        "conditional_live_sizing": {
            "enabled": True,
            "solo_multiplier": 2.0,
            "stacked_multiplier": 1.0,
            "max_contracts": 10,
        },
        # AetherFlow-specific backtest execution override. The promoted runtime
        # reports used the 15:00-18:00 no-entry window with 16:00 force-flat.
        "direct_backtest_execution": {
            "enforce_no_new_entries_window": True,
            "no_new_entries_start_hour_et": 15,
            "no_new_entries_end_hour_et": 18,
            "force_flat_at_time": True,
            "force_flat_hour_et": 16,
            "force_flat_minute_et": 0,
        },
        # AetherFlow-specific realized-drawdown size cap. The live stack has a
        # generic cap; this keeps AF's direct replay and live execution aligned
        # without changing DE3 or RegimeAdaptive sizing.
        "drawdown_size_scaling": {
            "enabled": True,
            "start_usd": 250.0,
            "max_usd": 800.0,
            "base_contracts": 5,
            "min_contracts": 1,
        },
        "post_policy_size_rules": {
            "enabled": True,
            "rules": [
                {
                    "name": "boost_tb_london_highconf_22x",
                    "match_setup_families": ["transition_burst"],
                    "match_session_ids": [1],
                    "match_min_aetherflow_confidence": 0.70,
                    "size_multiplier": 2.2,
                },
                {
                    "name": "boost_tb_nyam_short_highconf_22x",
                    "match_setup_families": ["transition_burst"],
                    "match_session_ids": [2],
                    "match_sides": ["SHORT"],
                    "match_min_aetherflow_confidence": 0.70,
                    "size_multiplier": 2.2,
                },
                {
                    "name": "boost_er_asia_highconf_22x",
                    "match_setup_families": ["exhaustion_reversal"],
                    "match_session_ids": [0],
                    "match_min_aetherflow_confidence": 0.70,
                    "size_multiplier": 2.2,
                },
            ],
        },
        "max_feature_bars": 900,
        "allowed_session_ids": [],
        "allowed_setup_families": ["aligned_flow", "transition_burst", "exhaustion_reversal"],
        "risk_governor": {
            "enabled": True,
            "daily_loss_stop_usd": 300.0,
            "min_loss_trades": 2,
            "block_new_entries_rest_of_day": True,
        },
        "family_policies": {
            "transition_burst": {
                "threshold": 0.555,
                "allowed_session_ids": [1, 2, 3],
                "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                "use_horizon_time_stop": True,
                "policy_rules": [
                    {
                        "name": "asia_tb_disp_block_target",
                        "match_session_ids": [0],
                        "match_regimes": ["DISPERSED"],
                        "blocked_regimes": ["DISPERSED", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "asia_disp_long_054_maxslow060",
                        "match_session_ids": [0],
                        "match_regimes": ["DISPERSED"],
                        "match_sides": ["LONG"],
                        "threshold": 0.54,
                        "allowed_session_ids": [0],
                        "allowed_regimes": ["DISPERSED"],
                        "allowed_sides": ["LONG"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "max_flow_mag_slow": 0.6,
                        "use_horizon_time_stop": True,
                    },
                    {
                        "name": "asia_disp_short_054",
                        "match_session_ids": [0],
                        "match_regimes": ["DISPERSED"],
                        "match_sides": ["SHORT"],
                        "threshold": 0.54,
                        "allowed_session_ids": [0],
                        "allowed_regimes": ["DISPERSED"],
                        "allowed_sides": ["SHORT"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "min_d_alignment_3": 0.1,
                        "min_phase_d_alignment_mean_5": 0.05,
                        "use_horizon_time_stop": True,
                    },
                    {
                        "name": "nyam_tb_chop_long_quality_050",
                        "match_session_ids": [2],
                        "match_regimes": ["CHOP_SPIRAL"],
                        "match_sides": ["LONG"],
                        "threshold": 0.50,
                        "allowed_session_ids": [2],
                        "allowed_regimes": ["CHOP_SPIRAL"],
                        "allowed_sides": ["LONG"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "min_flow_agreement": 0.7,
                        "min_signed_d_alignment_3": 0.06,
                        "use_horizon_time_stop": True,
                    },
                    {
                        "name": "nyam_chop_053",
                        "match_session_ids": [2],
                        "match_regimes": ["CHOP_SPIRAL"],
                        "threshold": 0.53,
                        "allowed_session_ids": [2],
                        "allowed_regimes": ["CHOP_SPIRAL"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "use_horizon_time_stop": True,
                    },
                    {
                        "name": "nyam_tb_tg_short_block",
                        "match_session_ids": [2],
                        "match_regimes": ["TREND_GEODESIC"],
                        "match_sides": ["SHORT"],
                        "blocked_regimes": ["TREND_GEODESIC", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "nyam_tb_tg_long_block_target",
                        "match_session_ids": [2],
                        "match_regimes": ["TREND_GEODESIC"],
                        "match_sides": ["LONG"],
                        "blocked_regimes": ["TREND_GEODESIC", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "london_any_054",
                        "match_session_ids": [1],
                        "threshold": 0.54,
                        "allowed_session_ids": [1],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "use_horizon_time_stop": True,
                    },
                    {
                        "name": "nypm_tb_disp_block",
                        "match_session_ids": [3],
                        "match_regimes": ["DISPERSED"],
                        "blocked_regimes": ["DISPERSED", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "nypm_tb_long_block",
                        "match_session_ids": [3],
                        "match_sides": ["LONG"],
                        "blocked_regimes": [
                            "CHOP_SPIRAL",
                            "DISPERSED",
                            "TREND_GEODESIC",
                            "ROTATIONAL_TURBULENCE",
                        ],
                    },
                    {
                        "name": "nypm_tb_tg_short_phase_run2",
                        "match_session_ids": [3],
                        "match_regimes": ["TREND_GEODESIC"],
                        "match_sides": ["SHORT"],
                        "allowed_session_ids": [3],
                        "allowed_regimes": ["TREND_GEODESIC"],
                        "allowed_sides": ["SHORT"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "min_phase_regime_run_bars": 2,
                        "use_horizon_time_stop": True,
                    },
                ],
            },
            "aligned_flow": {
                "threshold": 0.56,
                "allowed_session_ids": [2],
                "allowed_regimes": ["DISPERSED"],
                "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                "min_d_alignment_3": 0.02,
                "min_setup_strength": 0.0,
                "max_directional_vwap_dist_atr": 5.0,
                "max_flow_mag_slow": 1.0,
                "selection_score_bias": 0.01,
                "entry_mode": "market_next_bar",
                "policy_rules": [
                    {
                        "name": "nyam_af_disp_short_block_target",
                        "match_session_ids": [2],
                        "match_regimes": ["DISPERSED"],
                        "match_sides": ["SHORT"],
                        "blocked_regimes": ["DISPERSED", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "nyam_disp_quality_054",
                        "match_session_ids": [2],
                        "match_regimes": ["DISPERSED"],
                        "threshold": 0.54,
                        "allowed_session_ids": [2],
                        "allowed_regimes": ["DISPERSED"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "entry_mode": "market_next_bar",
                        "min_flow_agreement": 0.96,
                        "max_stress_pct": 0.45,
                        "min_flow_mag_slow": 0.7,
                    },
                    {
                        "name": "nypm_disp_block",
                        "match_session_ids": [3],
                        "match_regimes": ["DISPERSED"],
                        "blocked_regimes": ["DISPERSED", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "nypm_disp",
                        "match_session_ids": [3],
                        "match_regimes": ["DISPERSED"],
                        "threshold": 0.57,
                        "allowed_session_ids": [3],
                        "allowed_regimes": ["DISPERSED"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "entry_mode": "market_next_bar",
                        "min_setup_strength": 0.0,
                        "max_directional_vwap_dist_atr": 6.0,
                        "max_flow_mag_slow": 1.0,
                    },
                    {
                        "name": "nypm_af_tg_short_quality_0526",
                        "match_session_ids": [3],
                        "match_regimes": ["TREND_GEODESIC"],
                        "match_sides": ["SHORT"],
                        "threshold": 0.526,
                        "allowed_session_ids": [3],
                        "allowed_regimes": ["TREND_GEODESIC"],
                        "allowed_sides": ["SHORT"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "entry_mode": "market_next_bar",
                        "min_setup_strength": 0.75,
                        "max_directional_vwap_dist_atr": 6.0,
                        "max_flow_mag_slow": 1.0,
                        "min_pressure_imbalance_30": -0.5,
                    },
                    {
                        "name": "nypm_af_tg_long_quality_050_mid",
                        "match_session_ids": [3],
                        "match_regimes": ["TREND_GEODESIC"],
                        "match_sides": ["LONG"],
                        "threshold": 0.50,
                        "allowed_session_ids": [3],
                        "allowed_regimes": ["TREND_GEODESIC"],
                        "allowed_sides": ["LONG"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "entry_mode": "market_next_bar",
                        "min_setup_strength": 0.0,
                        "max_directional_vwap_dist_atr": 6.0,
                        "max_flow_mag_slow": 1.0,
                        "min_flow_agreement": 0.55,
                        "min_signed_d_alignment_3": 0.04,
                        "size_multiplier": 0.5,
                    },
                    {
                        "name": "nypm_tg",
                        "match_session_ids": [3],
                        "match_regimes": ["TREND_GEODESIC"],
                        "threshold": 0.58,
                        "allowed_session_ids": [3],
                        "allowed_regimes": ["TREND_GEODESIC"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "entry_mode": "market_next_bar",
                        "min_setup_strength": 0.0,
                        "max_directional_vwap_dist_atr": 6.0,
                        "max_flow_mag_slow": 1.0,
                    },
                ],
            },
            "exhaustion_reversal": {
                "threshold": 0.58,
                "allowed_session_ids": [0, 1, 2],
                "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                "policy_rules": [
                    {
                        "name": "nyam_er_disp_long_block",
                        "match_session_ids": [2],
                        "match_regimes": ["DISPERSED"],
                        "match_sides": ["LONG"],
                        "blocked_regimes": ["DISPERSED", "ROTATIONAL_TURBULENCE"],
                    },
                    {
                        "name": "london_disp_short_flow050",
                        "match_session_ids": [1],
                        "match_regimes": ["DISPERSED"],
                        "match_sides": ["SHORT"],
                        "allowed_session_ids": [1],
                        "allowed_regimes": ["DISPERSED"],
                        "allowed_sides": ["SHORT"],
                        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                        "min_flow_agreement": 0.5,
                    },
                ],
            },
        },
        "hazard_block_regimes": ["ROTATIONAL_TURBULENCE"],
        "log_evals": True,
    },
    "KALSHI": {
        "key_id": str(SECRETS.get("KALSHI_KEY_ID", "") or ""),
        "private_key_path": str(SECRETS.get("KALSHI_PRIVATE_KEY_PATH", "") or ""),
        "base_url": "https://api.elections.kalshi.com/trade-api/v2",
        "series": "KXINXU",
        "polling_interval": 300,
        "cache_ttl": 120,
        "rate_limit_delay": 0.4,
        "request_timeout": 15,
        "max_retries": 3,
        "enabled": True,
        "basis_offset": 0.0,
        "sentiment_thresholds": {
            "strong_bull": 0.70,
            "mild_bull": 0.60,
            "neutral_low": 0.45,
            "neutral_high": 0.55,
            "mild_bear": 0.40,
            "strong_bear": 0.30,
        },
        "veto_mode": "soft",
        "extreme_hard_veto_low": 0.10,
        "extreme_hard_veto_high": 0.90,
    },
    "PCT_LEVEL_OVERLAY": {
        "enabled": True,
        "levels_pct": [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00],
        "proximity_pct": 0.05,
        "breakout_extension_pct": 0.10,
        "pivot_retrace_pct": 0.15,
        "horizon_minutes": 30,
        "atr_window_bars": 30,
        "atr_q1_pct": 0.020,
        "atr_q3_pct": 0.070,
        "range_q1_pct": 0.30,
        "range_q3_pct": 1.00,
        "scout_levels": [0.25, 0.50],
        "primary_levels": [0.75, 1.00, 1.25],
        "exhaustion_levels": [1.50, 1.75, 2.00, 2.50, 3.00],
        "size_tilt_conf_threshold": 0.30,
        "max_size_tilt_pct": 0.20,
        "min_size_tilt_pct": -0.30,
        "trail_tp_extend_pct": 0.20,
        "trail_tp_tighten_pct": 0.30,
        "dead_hours_et": [1, 7, 23],
    },
    "ENABLE_PCT_LEVEL_OVERLAY": True,
    "KALSHI_TRADE_OVERLAY": {
        "enabled": True,
        # Live handoff keeps the rule overlay scoped to DE3 only.
        "apply_strategy_prefixes": ["DynamicEngine3"],
        "lookback_bars": 20000,
        "lookback_trade_days": 10,
        "min_trade_days": 6,
        "strike_window_size": 120,
        "min_curve_points": 8,
        "min_curve_range": 0.08,
        "min_unique_probabilities": 4,
        "max_target_window_points": 32.0,
        "max_target_window_tp_mult": 3.5,
        "momentum_probe_points": 5.0,
        "fade_absolute_threshold": {
            "background": 0.48,
            "balanced": 0.50,
            "forward_primary": 0.52,
        },
        "fade_adjacent_delta_threshold": {
            "background": 0.20,
            "balanced": 0.18,
            "forward_primary": 0.16,
        },
        "support_probability_floor": {
            "background": 0.46,
            "balanced": 0.52,
            "forward_primary": 0.57,
        },
        "entry_threshold": {
            "background": 0.45,
            "balanced": 0.50,
            "forward_primary": 0.55,
        },
        "momentum_retention_floor": {
            "background": 0.72,
            "balanced": 0.76,
            "forward_primary": 0.80,
        },
        # Background mode trims confidence only lightly; non-background modes can block weak entries.
        "entry_block_buffer": {
            "background": 1.0,
            "balanced": 0.10,
            "forward_primary": 0.12,
        },
        "entry_size_floor": {
            "background": 0.85,
            "balanced": 0.60,
            "forward_primary": 0.45,
        },
        "forward_weight": {
            "background": 0.20,
            "balanced": 0.48,
            "forward_primary": 0.78,
        },
        "min_tp_multiplier": float(os.environ.get("JULIE_KALSHI_MIN_TP_MULT", "0.55")),
        "max_tp_multiplier": float(os.environ.get("JULIE_KALSHI_MAX_TP_MULT", "1.75")),
        "trail_enabled_roles": ["balanced", "forward_primary"],
        "trail_buffer_ticks": {
            "background": 6,
            "balanced": 4,
            "forward_primary": 4,
        },
        "recent_price_action": {
            "uncertain_mean_day_range": 85.0,
            "outrageous_mean_day_range": 100.0,
            "uncertain_max_day_range": 130.0,
            "outrageous_max_day_range": 150.0,
            "uncertain_flip_rate": 0.39,
            "outrageous_flip_rate": 0.42,
            "uncertain_large_bar_share": 0.14,
            "outrageous_large_bar_share": 0.17,
            "uncertain_mean_true_range": 1.55,
            "outrageous_mean_true_range": 1.85,
            "uncertain_min_score": 3,
            "outrageous_min_score": 5,
            "today_breakout_min_range_points": 70.0,
            "today_breakout_min_net_ratio": 0.60,
            "today_breakout_level_lookback_days": 3,
            "today_breakout_level_tolerance_points": 0.75,
            "today_chop_min_range_points": 30.0,
            "today_chop_min_flip_rate": 0.45,
        },
        "breakout_max_tp_multiplier": 2.5,
    },
    "TRUTH_SOCIAL_SENTIMENT": {
        "enabled": True,
        "poll_interval": 120,
        "pump_threshold": 0.85,
        "emergency_exit_threshold": -0.75,
        "finbert_local_path": "./models/finbert",
        "target_handle": str(
            SECRETS.get("TRUTHSOCIAL_TARGET_HANDLE", os.environ.get("TRUTH_SOCIAL_TARGET_HANDLE", "realDonaldTrump"))
            or "realDonaldTrump"
        ),
        "signal_max_age_seconds": 1800,
        "emergency_exit_max_age_seconds": 3600,
        "quick_pump_tp_points": 4.0,
        "quick_pump_sl_points": 2.0,
    },
    # Opposite-direction reversal confirmation (env-var tunable).
    # required_confirmations: how many consecutive opposite signals within window_bars
    # are needed to close the active position and flip direction.
    # Higher = fewer, slower reversals. Lower = more aggressive direction flipping.
    "LIVE_OPPOSITE_REVERSAL": {
        "required_confirmations": int(os.environ.get("JULIE_REVERSAL_CONFIRM", "3")),
        "window_bars": int(os.environ.get("JULIE_REVERSAL_WINDOW", "3")),
        "require_same_strategy_family": True,
        "require_same_active_trade_family": False,
        "require_same_sub_strategy": False,
    },
    # Optional macro-regime feature (disabled by default for robustness)
    "ML_PHYSICS_USE_MACRO_REGIME": False,
    "ML_PHYSICS_REGIMES_FILE": "regimes.json",
    # Guardrails to auto-disable weak sessions based on training metrics
    "ML_PHYSICS_GUARD": {
        # Rework: disable aggregate train-metric guard; rely on walk-forward stability guard.
        "enabled": False,
        "min_trades": 300,     # Require enough samples for stability
        "min_win_rate": 0.45,  # Avoid disabling viable sessions on noisy holdouts
        "min_avg_pnl": 0.0,    # Keep weak sessions visible; rely on EV gate for trade-level quality
    },
    # Walk-forward stability guard (requires multi-fold positive expectancy)
    "ML_PHYSICS_WALK_FORWARD_GUARD": {
        "enabled": False,
        "require": True,            # Disable if walk-forward data missing
        "min_folds": 2,
        "min_positive_folds": 1,
        "min_positive_ratio": 0.50,
        "min_fold_avg_pnl": 0.0,    # Avg PnL per fold must be >= this
        "min_fold_trades": 20,      # Require fold to have enough trades
        # Robust EV objective (mean + worst fold, penalize instability).
        "objective_mean_weight": 0.65,
        "objective_worst_weight": 0.35,
        "objective_std_penalty": 0.10,
        "objective_min_fold_trades": 20,
        # Hard gates for robust objective.
        "min_mean_fold_ev": 0.0,
        "min_worst_fold_ev": -0.10,
        "min_objective_score": 0.0,
        "sessions": {
            # Tighten weaker sessions with stricter robustness requirements.
            "ASIA": {
                # Legacy-ASIA experiment: bypass WF hard disable so we can compare behavior.
                "enabled": False,
            },
            "NY_PM": {
                "min_positive_ratio": 0.75,
                "min_worst_fold_ev": 0.20,
                "min_objective_score": 0.35,
            },
        },
    },
    # Optional per-session/regime overrides for ML physics guard thresholds
    "ML_PHYSICS_GUARD_OVERRIDES": {
        # ASIA experiment: allow runtime evaluation even when aggregate train metrics are weak.
        "ASIA": {
            "min_trades": 0,
            "min_win_rate": 0.0,
            "min_avg_pnl": -999.0,
        },
    },
    # Disable MLPhysics sessions in live bot (backtest still evaluates them)
    "ML_PHYSICS_LIVE_DISABLED_SESSIONS": [],
    # Disable MLPhysics sessions in backtest (default empty to allow evaluation)
    "ML_PHYSICS_BACKTEST_DISABLED_SESSIONS": [],
    # Backtest-only: disable ML-specific post-signal gating so MLPhysics uses only
    # the shared/global filter stack in backtest execution.
    "BACKTEST_ML_PHYSICS_GLOBAL_FILTERS_ONLY": True,
    # Backtest-only test mode: let MLPhysics bypass filter blocks while tagging
    # `ml_would_blocked_filters` / `bypassed_filters` on executed trades.
    "BACKTEST_ML_PHYSICS_FILTER_BYPASS": {
        "enabled": False,
        # Empty => all sessions. Example: ["NY_AM", "NY_PM"].
        "sessions": [],
    },
    # Disable specific MLPhysics regimes per session (applies to live + backtest)
    "ML_PHYSICS_DISABLED_REGIMES": {
        "NY_PM": ["low"],
    },
    # MLPhysics confidence-based priority boost (elevate to FAST)
    "ML_PHYSICS_PRIORITY_BOOST": {
        "enabled": False,
        "min_confidence": 0.93,
        "boost_priority": 1,  # 1 = FAST, 2 = STANDARD
        "sessions": ["NY_AM", "NY_PM"],
    },
    # Mixed-strategy backtests can crowd out the oversized ASIA runner trades that
    # carry a distinct holding profile. Give those wide-bracket MLPhysics signals a
    # deterministic priority bump without changing ordinary intraday ML signals.
    "ML_PHYSICS_RUNNER_PRIORITY_BOOST": {
        "enabled": True,
        "boost_priority": 1,  # 1 = FAST, 2 = STANDARD
    },
    # MLPhysics soft gating: suppress opposite signals when confidence is high
    "ML_PHYSICS_SOFT_GATING": {
        "enabled": False,
        "min_confidence": 0.93,
        "block_standard": True,
        "block_fast": False,  # keep FAST strategies unless explicitly allowed
        "sessions": ["NY_AM", "NY_PM"],
    },
    # EV-based decisioning: thresholds act as exposure control; EV gates decide take/skip.
    "ML_PHYSICS_EV_DECISION": {
        "enabled": True,
        # Rework: EV decides side first; thresholds are optional exposure control.
        "ev_first": True,
        # If EV regressors exist, use them first; otherwise fallback to prob/bracket EV.
        "use_model_predictions": True,
        # Uncertainty gate: block model-EV signals when model-vs-fallback disagreement is too high.
        "max_ev_disagreement_points": 1.0,
        "require_ev_sign_agreement": True,
        # Minimum expected edge in points after round-trip fees/slippage.
        "min_ev_points": 0.15,
        # Minimum |P(up)-P(down)| margin for EV decisions.
        "min_prob_edge": 0.03,
        # Threshold gate optional under EV-first mode; keep disabled by default.
        "require_threshold_gate": False,
        # Keep tradeability gate as a hard risk control.
        "require_trade_gate": True,
        # Gate policy: hard | soft | off. Soft converts gate shortfall into tougher EV requirement.
        "trade_gate_policy": "soft",
        "trade_gate_penalty_points_per_prob": 2.0,
        "trade_gate_penalty_cap_points": 1.0,
        # Under EV-first, uncertainty is handled as EV penalty (not immediate hard block).
        "ev_uncertainty_penalty_points": 0.35,
        # Optional floor on tradeability-gate probability (combined with learned gate threshold).
        "min_trade_gate_prob": 0.72,
        # Optional cap on tradeability-gate probability to avoid over-tight/no-trade sessions.
        "max_trade_gate_prob": 0.90,
        # When model EV is active, avoid conflicting probability-only blockers by default.
        "apply_prob_margin_with_model_ev": False,
        "apply_confidence_gates_with_model_ev": False,
        # Demand stronger EV under high volatility.
        "high_vol_min_ev_points": 0.25,
    },
    # Optional per-session overrides for EV decisioning.
    # Useful for controlled experiments (e.g., keep EV globally but revert one session).
    "ML_PHYSICS_EV_DECISION_SESSION_OVERRIDES": {
        "ASIA": {
            # Revert ASIA to legacy threshold/probability path.
            "enabled": False,
            "use_model_predictions": False,
            # Keep risk controls from threshold + tradeability gates.
            "require_trade_gate": True,
            "require_threshold_gate": True,
        },
        "LONDON": {
            # Keep EV logic, but relax hard gate to improve usable coverage.
            "min_trade_gate_prob": 0.72,
            "max_trade_gate_prob": 0.88,
        },
        "NY_AM": {
            # NY_AM keeps a stricter floor than ASIA/LONDON.
            "min_trade_gate_prob": 0.75,
            "max_trade_gate_prob": 0.90,
        },
        "NY_PM": {
            # Moderate relax to reduce no-trade stretches.
            "min_trade_gate_prob": 0.72,
            "max_trade_gate_prob": 0.88,
        },
    },
    # Payoff-model training (EV regression) used by MLPhysics runtime selection.
    "ML_PHYSICS_EV_MODELS": {
        "enabled": True,
        # Minimum train samples required per side EV model.
        "min_train_samples": 300,
        # Gate label target: mark bars tradeable when best realized side EV >= this value.
        # Slightly lower target yields less brittle gate labels and better runtime coverage.
        "gate_min_ev_points": 0.05,
        # Runtime abstain calibration from holdout predictions.
        "target_coverage_default": 0.12,
        "calibration_min_trades": 100,
        "calibration_quantile_start": 0.55,
        "calibration_quantile_end": 0.995,
        "calibration_grid_size": 61,
        "min_ev_floor_points": 0.10,
        "min_ev_cap_points": 4.0,
        "disagreement_quantile": 0.70,
        "disagreement_scale": 0.85,
        "disagreement_floor": 0.25,
        "disagreement_cap": 1.50,
        "max_sign_disagree_rate": 0.15,
        "sign_disagree_tight_gap": 0.75,
        "prob_edge_quantile": 0.25,
        "min_prob_edge_cap": 0.20,
        "high_vol_min_ev_boost": 0.10,
    },
    # Hard limits for learned tradeability gate thresholds (applies in train + runtime).
    "ML_PHYSICS_GATE_HARD_LIMITS": {
        "enabled": True,
        "default": {"min": 0.65, "max": 0.90},
        "sessions": {
            # Keep ASIA legacy path bounded, but avoid forcing high gate thresholds.
            "ASIA": {"min": 0.65, "max": 0.86},
            "LONDON": {"min": 0.68, "max": 0.88},
            "NY_AM": {"min": 0.70, "max": 0.90},
            "NY_PM": {"min": 0.68, "max": 0.88},
        },
    },
    # Backtest-only: learned continuation allowlist from walk-forward reports
    "BACKTEST_CONTINUATION_ALLOWLIST": {
        "enabled": True,
        # modes: "reports" (walk-forward backtest files) or "csv_fast" (single CSV pass)
        "mode": "csv_fast",
        # When False, backtests will NOT rebuild the allowlist from CSV.
        "runtime_train": False,
        # Key granularity for allowlist entries: full | session_day | session
        "key_granularity": "session_day",
        "reports_glob": "backtest_reports/backtest_*.json",
        "min_total_trades": 8,
        "min_fold_trades": 2,
        "min_avg_pnl_points": 0.1,
        "min_fold_expectancy_points": 0.0,
        "min_folds": 2,
        "min_positive_fold_ratio": 0.60,
        "cache_file": "backtest_reports/continuation_allowlist.json",
        "fast": {
            "folds": 4,
            "max_horizon_bars": 120,
            "exit_at_horizon": "close",
            "assume_sl_first": True,
            "use_dynamic_sltp": True,
            "default_tp": 6.0,
            "default_sl": 4.0,
            "min_win_rate": 0.45,
            "symbol_contains": ["MES", "ES"],
        },
    },
    # Backtest-only: flip-confidence allowlist for blocked-signal flips
    "BACKTEST_FLIP_CONFIDENCE": {
        "enabled": True,
        "cache_file": "backtest_reports/flip_confidence.json",
        "allowed_filters": [
            "RejectionFilter",
            "ChopRangeBias",
            "ImpulseFilter",
            "ExtensionFilter",
            "TrendFilter",
            "StructureBlocker",
            "BankLevelQuarterFilter",
            "FilterArbitrator",
            "LegacyTrend",
        ],
        # Key fields used to build flip-confidence keys
        "key_fields": ["filter", "session", "side"],
        # Flip simulation settings
        "max_horizon_bars": 120,
        "exit_at_horizon": "close",
        "assume_sl_first": True,
        # Minimum viability thresholds
        "min_total_trades": 25,
        "min_fold_trades": 10,
        "min_avg_pnl_points": 0.10,
        "min_win_rate": 0.55,
        "min_fold_expectancy_points": 0.0,
        "min_folds": 2,
        "min_positive_fold_ratio": 0.60,
        # Fold mode: regime (LORO), time, regime_time, or none
        "fold_mode": "regime_time",
        "folds": 4,
        "loro_regimes": ["low", "normal", "high"],
    },
    # Backtest-only: require market confirmation for continuation
    "BACKTEST_CONTINUATION_CONFIRM": {
        "enabled": True,
        "use_adx": True,
        "use_trend_alt": True,
        "use_vwap": True,
        "use_structure_break": True,
        "vwap_sigma_min": 1.0,
        "require_any": True,
    },
    # Backtest-only: continuation signal generation mode ("calendar" or "structure")
    "BACKTEST_CONTINUATION_SIGNAL_MODE": "structure",
    # Backtest-only: allow continuation only in proven regimes
    "BACKTEST_CONTINUATION_ALLOWED_REGIMES": ["high"],
    # Backtest-only: continuation rescues do not bypass core filters
    "BACKTEST_CONTINUATION_NO_BYPASS": True,
    # Continuation SL/TP trainer (ATR brackets)
    "CONTINUATION_SLTP_TRAIN": {
        "key_fields": ["session"],
        "sl_mults": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        "tp_mults": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "atr_window": 14,
        "max_horizon_bars": 120,
        "exit_at_horizon": "close",
        "assume_sl_first": True,
        "tick_size": 0.25,
        "min_sl": 0.0,
        "min_tp": 0.0,
        "min_total_trades": 50,
        "min_fold_trades": 10,
        "min_avg_pnl_points": 0.10,
        "min_win_rate": 0.50,
        "min_fold_expectancy_points": 0.0,
        "min_folds": 2,
        "min_positive_fold_ratio": 0.60,
        "fold_mode": "regime_time",
        "folds": 4,
        "loro_regimes": ["low", "normal", "high"],
    },
    "CONTINUATION_SLTP_FILE": "backtest_reports/continuation_sltp.json",
    # Live: flip-confidence guard (blocked-signal flips)
    "FLIP_CONFIDENCE": {
        "enabled": True,
        "allowlist_file": "backtest_reports/flip_confidence.json",
        "allowed_filters": [
            "RejectionFilter",
            "ChopRangeBias",
            "ImpulseFilter",
            "ExtensionFilter",
            "TrendFilter",
            "StructureBlocker",
            "BankLevelQuarterFilter",
            "FilterArbitrator",
            "LegacyTrend",
        ],
        "key_fields": ["filter", "session", "side"],
        "prefer_continuation": True,
        "allow_direct_flip": True,
    },
    # Live continuation guardrails (allowlist + confirmation + regime gating)
    "CONTINUATION_ENABLED": False,
    "CONTINUATION_GUARD": {
        "enabled": True,
        # Signal generation mode for live continuation ("calendar" or "structure")
        "signal_mode": "structure",
        "allowlist_file": "backtest_reports/continuation_allowlist.json",
        # Key granularity for allowlist entries: full | session_day | session
        "key_granularity": "session_day",
        "allowed_regimes": ["high"],
        "confirm": {
            "enabled": True,
            "use_adx": True,
            "use_trend_alt": True,
            "use_vwap": True,
            "use_structure_break": True,
            "vwap_sigma_min": 1.0,
            "require_any": True,
        },
        "no_bypass": True,
    },
    # Backtest-only: require stronger MLPhysics confidence before it can support consensus
    "BACKTEST_CONSENSUS_ML_MIN_CONF": 0.65,
    "BACKTEST_CONSENSUS_ML_EXTRA_MARGIN": 0.05,
    # Backtest-only: ML diagnostics capture can be memory/CPU heavy on long ranges.
    "BACKTEST_ML_DIAGNOSTICS": {
        "enabled": False,
        "include_no_signal": False,
        # 0 => unlimited (only used when enabled=True)
        "max_records": 0,
    },
    # Backtest-only: when running MLPhysicsStrategy *alone*, switch to a lightweight
    # diagnostic profile so you can see true ML signal flow instead of broad stack blocks.
    "BACKTEST_ML_ONLY_DIAGNOSTIC_PROFILE": {
        # Keep this opt-in. When enabled it records rich per-bar ML diagnostics,
        # which is useful for debugging but materially slows long ML-only backtests.
        "enabled": False,
        # Keep only core execution/risk filters for ML diagnostics.
        "filters": [
            "FixedSLTP",
            "TargetFeasibility",
            "VolatilityGuardrail",
            "MLVolRegimeGuard",
        ],
        # Force rich ML diagnostics and include no-signal rows/reasons.
        "force_ml_diagnostics": True,
        "include_no_signal": True,
        "max_diag_records": 0,
        # Disable speed-profile throttles to avoid hiding ML eval detail.
        "disable_speed_profile": True,
        "force_full_ml_eval": True,
    },
    # Backtest-only: rich market-condition snapshots/reporting (analysis aid, not execution).
    "BACKTEST_MARKET_SNAPSHOTS": {
        "enabled": True,
        "summary_enabled": True,
    },
    # Backtest-only: speed profile for long-range runs.
    # The defaults below avoid changing trade decisions while removing expensive diagnostics.
    "BACKTEST_SPEED_PROFILE": {
        "enabled": True,
        "suppress_warnings": True,
        "disable_ml_diagnostics": True,
        "disable_flip_confidence": True,
        "disable_market_snapshots": True,
        "disable_market_condition_summary": True,
        # Optional ML throttle (in bars) on top of TF alignment checks.
        "ml_eval_stride": 5,
        # Optional tighter dist context window for speed (can change decisions if too low).
        "dist_input_bars": 1500,
    },
    # Backtest-only: ensure OOS fidelity by disabling stride-based MLPhysics skips.
    # When enabled, MLPhysics still obeys its timeframe alignment, but it is checked every bar.
    "BACKTEST_OOS_FORCE_FULL_ML_EVAL": True,
    # Backtest-only: cap history fed through the main loop during ML-only runs.
    # Dist MLPhysics already uses a bounded context window, so carrying years of history
    # through every bar just adds pandas slicing overhead.
    "BACKTEST_ML_ONLY_HISTORY_MAX_BARS": 3000,
    # Backtest-only: fast/approximate mode controls
    "BACKTEST_FAST_MODE": {
        "enabled": False,
        "bar_stride": 1,
        "skip_mfe_mae": False,
    },
    # Backtest-only: strategy evaluation worker pool (threaded).
    # NOTE: Core execution is still bar-sequential; workers parallelize per-bar strategy eval.
    "BACKTEST_WORKERS": 6,
    "BACKTEST_PARALLEL_STRATEGY_EVAL": True,
    # Backtest-only: TrendDay console prints are expensive on long runs.
    "BACKTEST_TRENDDAY_VERBOSE": False,
    # Backtest-only: suppress DE3 warning spam (safety/fixed-sltp/fees blocks).
    "BACKTEST_DE3_VERBOSE_WARNINGS": False,
    # Backtest-only: select DE3 DB version without changing live defaults.
    # Set to "v1", "v2", "v3", "v4", or "" to keep CONFIG["DE3_VERSION"] as-is.
    "BACKTEST_DE3_VERSION_OVERRIDE": "v4",
    # Backtest-only: promote the DE3v4 learned side model from a soft prior to a
    # guarded hard override, but only when both long and short candidates exist.
    # Keeps live behavior unchanged while we test whether the learned side chooser
    # can stop obvious counter-bias longs.
    "BACKTEST_DE3_V4_DECISION_SIDE_MODEL": {
        "enabled": True,
        "application_mode": "hard_override",
        "apply_side_patterns": ["both"],
        # Keep the side prior inert unless the model made an actual call.
        "apply_prior_only_when_predicted": True,
    },
    # Backtest-only: write a live-updating partial report while simulation runs.
    # Useful for long runs so progress/results are inspectable before completion.
    "BACKTEST_LIVE_REPORT": {
        "enabled": True,
        "write_every_sec": 15,
        # When true, partial file includes the full serialized trade log each write.
        # Keep false for speed on long runs.
        "include_trade_log": False,
        # Optional custom output dir. None => <repo>/backtest_reports
        "output_dir": None,
    },
    # Backtest-only: throttle expensive progress text report rebuilds pushed to UI callbacks.
    # Lower => fresher report text but more overhead. Has no effect on final saved report.
    "BACKTEST_PROGRESS_REPORT_EVERY_SEC": 8.0,
    # Backtest-only: suppress PenaltyBox warning spam.
    "BACKTEST_PENALTY_BOX_VERBOSE_WARNINGS": False,
    # Backtest-only: execution assumptions for bar data
    "BACKTEST_EXECUTION": {
        # stop | take | ohlc | best | worst
        "sl_tp_conflict": "ohlc",
        # Fill at bar_open when it gaps beyond stop/target.
        "gap_fills": True,
        # Global drawdown-aware sizing (backtest-only):
        # linearly step size down from base_contracts to min_contracts as
        # realized drawdown moves from start_usd to max_usd.
        "drawdown_size_scaling_enabled": True,
        "drawdown_size_scaling_start_usd": 0.0,
        "drawdown_size_scaling_max_usd": 2000.0,
        "drawdown_size_scaling_base_contracts": 5,
        "drawdown_size_scaling_min_contracts": 1,
        # RegimeAdaptive allocation model:
        # - Assume MLPhysics and DE3 each reserve 5 contracts.
        # - RegimeAdaptive can scale from its 5-contract base up to 10 total
        #   as new realized equity highs are achieved, then scale back down
        #   under drawdown pressure using the generic drawdown ladder above.
        "regimeadaptive_growth_size_scaling_enabled": True,
        "regimeadaptive_growth_profit_step_usd": 1500.0,
        "regimeadaptive_growth_size_scaling_max_contracts": 10,
        "regimeadaptive_growth_anchor": "peak",
        # Hard cap on executed stop distance in points (None/<=0 disables).
        # Effective per-trade SL becomes: min(requested_sl_dist, max_stoploss_points).
        "max_stoploss_points": None,
        # Backtest-only: keep the global SL cap, but bypass it for MLPhysics entries.
        "disable_max_stoploss_for_mlphysics": False,
        # Backtest-only: bypass global SL cap for DynamicEngine3 when runtime DB is v2.
        "disable_max_stoploss_for_de3_v2": False,
        # Backtest-only session guard (US/Eastern):
        # no new entries during [start_hour, end_hour), force-flat open positions at force_flat time.
        "no_new_entries_start_hour_et": 16,
        "no_new_entries_end_hour_et": 18,
        "enforce_no_new_entries_window": True,
        "force_flat_at_time": True,
        "force_flat_hour_et": 16,
        "force_flat_minute_et": 0,
        # Backtest-only holiday closure (ET calendar date based).
        # Uses US federal holiday dates (e.g., Jan 1) + optional explicit extra dates.
        # Blocking is session-aware; default blocks only NY sessions so ASIA/LONDON can still trade.
        "enforce_us_holiday_closure": True,
        # Session labels: NY_AM, NY_PM, LONDON, ASIA, OFF, or ALL.
        "holiday_closure_sessions_et": ["NY_AM", "NY_PM"],
        # Optional ISO dates like ["2025-07-03", "2025-11-28"].
        "extra_closed_dates_et": [],
    },
    # Backtest-only: relax DynamicChop sensitivity (lower = fewer chop blocks)
    "BACKTEST_DYNAMIC_CHOP_MULTIPLIER": 1.0,
    # Backtest-only: replay Gemini-style multipliers from a prebuilt dataset.
    # Dataset columns (minimum): timestamp + sl_multiplier + tp_multiplier.
    # Optional: chop_multiplier.
    "BACKTEST_GEMINI_MULTIPLIERS": {
        "enabled": True,
        "path": "backtest_gemini_multipliers.csv",
        "timestamp_column": "timestamp",
        "sl_column": "sl_multiplier",
        "tp_column": "tp_multiplier",
        "chop_column": "chop_multiplier",
        # Do not apply multiplier replay to these strategy prefixes.
        # Keeps DE3 and MLPhysics brackets faithful to their native runtime selection.
        "disabled_strategy_prefixes": ["DynamicEngine3", "MLPhysics"],
        # If timestamps are naive in CSV/parquet, localize using this timezone.
        "assume_timezone": "America/New_York",
        "default_sl_multiplier": 1.0,
        "default_tp_multiplier": 1.0,
        "default_chop_multiplier": 1.0,
    },
    # Backtest-only: post-run diagnostics/recommendation exports.
    # Writes sidecar files next to each backtest JSON:
    #   *_baseline_comparison.json
    #   *_gemini_recommendation.json
    "BACKTEST_POST_RUN_RECOMMENDER": {
        "enable_baseline_comparison": True,
        "baseline_auto_lookback_runs": 20,
        # Optional explicit baseline report path override.
        # Leave blank to auto-select best recent run for same symbol/date range.
        "baseline_report_path": "",
        "enable_gemini_recommendation": True,
        # Allow post-run recommendation even when runtime backtest disables GEMINI features.
        "allow_when_gemini_disabled": True,
        "gemini_timeout_sec": 60,
        "max_code_context_chars": 1500,
        "de3_context_files": [
            "dynamic_engine3_strategy.py",
            "de3_v4_runtime.py",
            "de3_v4_router.py",
            "de3_v4_lane_selector.py",
            "de3_v4_bracket_module.py",
            "config.py",
        ],
    },
    "BACKTEST_SYMBOL_MODE": "auto_by_day",  # single | auto_by_day
    "BACKTEST_SYMBOL_AUTO_METHOD": "volume",  # volume | rows
    # Backtest-only: ASIA calibrations to better handle smooth trends
    "BACKTEST_ASIA_CALIBRATIONS": {
        "enabled": True,
        "trend_bias": {
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_slope_bars": 20,
            "min_ema_separation": 0.1,
        },
        "penalty_box": {
            "enabled": True,
            "lookback": 50,
            "tolerance": 1.5,
            "penalty_bars": 3,
        },
        "target_feasibility": {
            "enabled": True,
            "lookback": 20,
            "min_box_range": 1.0,
            "max_tp_box_mult": 1.8,
            "allow_trend_override": False,
        },
        "chop_filter": {
            "enabled": True,
            "allow_trend_override": True,
        },
    },
    # Live: ASIA calibrations to better handle smooth trends
    "ASIA_CALIBRATIONS": {
        "enabled": True,
        "trend_bias": {
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_slope_bars": 20,
            "min_ema_separation": 0.1,
        },
        "penalty_box": {
            "enabled": True,
            "lookback": 50,
            "tolerance": 1.5,
            "penalty_bars": 3,
        },
        "target_feasibility": {
            "enabled": True,
            "lookback": 20,
            "min_box_range": 1.0,
            "max_tp_box_mult": 1.8,
            "allow_trend_override": False,
        },
        "chop_filter": {
            "enabled": True,
            "allow_trend_override": True,
        },
    },
    # Smooth Trend Asia strategy (Trigger A)
    "SMOOTH_TREND_ASIA": {
        "enabled": True,
        "ema_fast": 20,
        "ema_slow": 50,
        "ema_slope_bars": 20,
        "min_ema_separation": 0.1,
        "er_window": 60,
        "er_min": 0.55,
        "persistence_window": 60,
        "persistence_min": 0.65,
        "closes_side_window": 60,
        "closes_side_min": 0.80,
        "atr_window": 20,
        "atr_long_window": 120,
        "atr_ratio_max": 1.15,
        "max_tr_mult": 2.2,
        "regime_min_passes": 3,
        "pullback_lookback": 20,
        "pullback_touch_atr_mult": 0.2,
        "pullback_max_drawdown_mult": 0.8,
        "pullback_ema50_buffer_mult": 0.2,
        "stop_ema50_buffer_mult": 0.3,
        "max_stop_points": 2.5,
        "tp_mult": 1.5,
        "min_tp_points": 1.0,
        "tick_size": 0.25,
        "cooldown_bars": 12,
    },
    # Backtest-only: Smooth Trend Asia strategy (Trigger A)
    "BACKTEST_SMOOTH_TREND_ASIA": {
        "enabled": True,
        "ema_fast": 20,
        "ema_slow": 50,
        "ema_slope_bars": 20,
        "min_ema_separation": 0.1,
        "er_window": 60,
        "er_min": 0.55,
        "persistence_window": 60,
        "persistence_min": 0.65,
        "closes_side_window": 60,
        "closes_side_min": 0.80,
        "atr_window": 20,
        "atr_long_window": 120,
        "atr_ratio_max": 1.15,
        "max_tr_mult": 2.2,
        "regime_min_passes": 3,
        "pullback_lookback": 20,
        "pullback_touch_atr_mult": 0.2,
        "pullback_max_drawdown_mult": 0.8,
        "pullback_ema50_buffer_mult": 0.2,
        "stop_ema50_buffer_mult": 0.3,
        "max_stop_points": 2.5,
        "tp_mult": 1.5,
        "min_tp_points": 1.0,
        "tick_size": 0.25,
        "cooldown_bars": 12,
    },
    # Backtest-only: extend vol-split ML sessions without affecting live defaults
    "ML_PHYSICS_VOL_SPLIT_BACKTEST_SESSIONS": [],
    # Backtest-only: disable ML vol-split for specific sessions
    "ML_PHYSICS_VOL_UNSPLIT_BACKTEST_SESSIONS": [],
    # Volatility guard: skip MLPhysics in selected sessions during high vol
    "ML_PHYSICS_VOL_GUARD": {
        "enabled": True,
        "sessions": [],
        "feature": "High_Volatility",
    },
    # Volatility regime labeling configuration
    "VOLATILITY_HIERARCHY_MODE": {
        "mode": "coarse",          # "full" or "coarse"
        "include_quarter": True,   # include yearly quarter in coarse key
    },
    # Optional file-based volatility thresholds (for train/backtest/live alignment)
    "VOLATILITY_THRESHOLDS_FILE": "volatility_thresholds.json",
    "VOLATILITY_THRESHOLDS_MIN_SAMPLES": 50,
    "VOLATILITY_STD_WINDOWS": {
        "default": 20,
        "sessions": {
            "NY_AM": 60,
            "NY_PM": 60,
        },
    },
    # Scale threshold bands when session std window differs from default
    "VOLATILITY_STD_WINDOW_SCALING": {
        "enabled": True,
        "min": 0.5,
        "max": 2.0,
        "lookback": 200,
    },
    # NY normal-vol structure filter (Phase 2)
    "ML_PHYSICS_NY_NORMAL_FILTER": {
        "enabled": True,
        "er_window": 30,
        "er_min": 0.25,
        "vwap_cross_window": 60,
        "vwap_cross_max": 3,
        "margin": 0.08,
        "block_chop": True,
    },
    # Normal-vol SL/TP adjustments (applied in volatility_filter)
    "VOLATILITY_NORMAL_ADJUSTMENTS": {
        "enabled": True,
        "default": {"sl_mult": 0.95, "tp_mult": 0.90},
        "sessions": {
            "NY_AM": {"sl_mult": 0.95, "tp_mult": 0.88},
            "NY_PM": {"sl_mult": 0.95, "tp_mult": 0.88},
        },
    },
    # MLPhysics: confidence gating by volatility regime (low/normal/high)
    "ML_PHYSICS_VOL_REGIME_GUARD": {
        "enabled": False,
        "default": {
            "ultra_low": {"block": True},
            "low": {"min_conf_delta": 0.02},
            "normal": {"min_conf_delta": 0.00},
            "high": {"min_conf_delta": 0.04},
        },
        "sessions": {
            "ASIA": {
                "low": {"min_conf_delta": 0.03},
                "high": {"min_conf_delta": 0.05},
            },
            "NY_AM": {
                "low": {"min_conf_delta": 0.03},
                "high": {"min_conf_delta": 0.06},
            },
        },
    },
    # High-vol regime tightening for MLPhysics (all runtimes; tweak per session)
    "ML_PHYSICS_HIGH_VOL_THRESHOLD_BUMP": {
        "enabled": True,
        "bump": 0.05,
        "max_threshold": 0.90,
        "sessions": ["NY_AM", "NY_PM", "ASIA"],
    },
    "ML_PHYSICS_HIGH_VOL_DIRECTIONAL_GATE": {
        "enabled": True,
        "feature": "High_Volatility",
        "min_conf_delta": 0.07,
        "max_conf": 0.95,
        "overrides": {},
    },
    # Robust NY_AM fix: split ML models by volatility regime
    "ML_PHYSICS_VOL_SPLIT": {
        # Disable hard splits for robustness; rely on pooled context features instead.
        "enabled": False,
        # ASIA is intentionally NOT split: splitting fragments an already-small sample and
        # makes LORO holdouts fail (insufficient trades per holdout). Keep ASIA as a single
        # model for robustness.
        "sessions": [],
        "feature": "High_Volatility",
    },
    # 3-way vol split (low/normal/high) using volatility_filter regimes
    "ML_PHYSICS_VOL_SPLIT_3WAY": {
        # Disabled for robustness; hard splits fragment the 5m sample.
        "enabled": False,
        "sessions": [],
    },
    # DynamicEngine: tighten confidence in NY sessions (09-18 ET buckets)
    "DYNAMIC_ENGINE_NY_CONF": {
        "enabled": True,
        "sessions": ["09-12", "12-15", "15-18"],
        "min_opt_wr": 0.25,
        "min_final_score": None,
    },
    # DynamicEngine2 (DE2) adaptive sharpe policy:
    # Keep DE2 tradable across regimes/sessions/timeframes, but apply context-sensitive
    # quality, EV, and drift constraints instead of blanket hard blocks.
    "DYNAMIC_ENGINE2_POLICY": {
        "enabled": True,
        "log_decisions": False,
        # Global floor; context profiles raise/lower this by market setting.
        "min_rr": 1.20,
        "regime_allow": {
            "enabled": True,
            # Soft mode: never fully disable a context; penalize weak buckets instead.
            "mode": "soft",
            "soft_penalty_ev_points": 0.06,
            "blocked_regimes": [],
            "sessions": {
                "default": [],
            },
            "timeframes": {
                "default": [],
            },
        },
        # Base trigger-candle quality and shock guards.
        "quality_filters": {
            "enabled": True,
            "atr_period": 14,
            "min_body_atr": 0.20,
            "max_range_atr": 2.60,
            "max_gap_atr": 0.75,
            "max_shock_range_atr": 2.90,
            "apply_close_pos_for": ["Follow_Color", "Inside_Break"],
            "long_min_close_pos": 0.55,
            "short_max_close_pos": 0.45,
            "per_strategy": {
                "Gap_Reversal": {
                    "min_body_atr": 0.12,
                    "max_range_atr": 2.20,
                    "max_gap_atr": 0.80,
                },
                "Follow_Color": {
                    "min_body_atr": 0.26,
                    "max_gap_atr": 0.60,
                },
                "Inside_Break": {
                    "min_body_atr": 0.24,
                    "max_gap_atr": 0.55,
                },
            },
        },
        # EV ranking model (edge after fees) for candidate selection.
        "ev_ranking": {
            "assumed_win_rate": 0.49,
            "min_ev_points": 0.12,
            "use_ev_per_risk": True,
            "log_rerank": True,
            "regime_win_rate": {
                "normal": 0.47,
                "high": 0.43,
                "low": 0.44,
            },
            "quarter_win_rate": {
                "Q1": 0.45,
                "Q2": 0.42,
                "Q3": 0.43,
                "Q4": 0.46,
            },
            "day_win_rate": {
                "Mon": 0.45,
                "Tue": 0.47,
                "Wed": 0.42,
                "Thu": 0.44,
                "Fri": 0.44,
            },
            "timeframe_win_rate": {
                "15min": 0.45,
                "5min": 0.43,
            },
            "session_win_rate": {
                "Asia": 0.46,
                "London": 0.43,
                "NY_AM": 0.43,
                "NY_PM": 0.46,
            },
            # Additional priors to stabilize DE2 at edge-granularity without hard disables.
            "day_session_win_rate": {
                "Wed|NY_AM": 0.36,
                "Wed|NY_PM": 0.40,
                "Wed|London": 0.43,
                "Thu|London": 0.37,
                "Fri|London": 0.36,
                "Tue|NY_AM": 0.38,
                "Mon|Asia": 0.44,
            },
            "day_timeframe_win_rate": {
                "Wed|15min": 0.41,
                "Thu|5min": 0.41,
                "Wed|5min": 0.43,
                "Mon|15min": 0.44,
                "Fri|5min": 0.43,
                "Tue|15min": 0.47,
            },
            "strategy_win_rate": {
                "Gap_Reversal": 0.44,
                "Engulfing": 0.46,
                "Wick_Rejection": 0.50,
                "Follow_Color": 0.49,
                "Fade_Color": 0.45,
                "Inside_Break": 0.45,
            },
        },
        # Preserve enough RR from the originating DE2 edge to avoid fee-dominated brackets.
        "edge_bracket_alignment": {
            "enabled": True,
            "edge_rr_blend": 0.80,
            "min_edge_rr": 1.25,
            "min_target_rr": 1.55,
            "target_rr_cap": 2.80,
            "tp_raise_only": True,
            "allow_sl_tighten": True,
            "max_tp_points": 10.0,
        },
        # Non-blocking stability controls:
        # - require a minimum probabilistic edge over break-even
        # - apply rolling trade-density penalties instead of hard context disable
        "stability": {
            "enabled": True,
            "window_minutes": 120,
            "context_soft_cap": 10,
            "strategy_soft_cap": 3,
            "timeframe_strategy_soft_cap": {
                "5min": 1,
                "15min": 3,
            },
            "context_penalty_per_extra": 0.055,
            "strategy_penalty_per_extra": 0.16,
            "min_edge_prob": 0.035,
            "session_penalty_mult": {
                "ASIA": 1.05,
                "LONDON": 1.35,
                "NY_AM": 1.35,
                "NY_PM": 1.05,
            },
            "regime_penalty_mult": {
                "normal": 1.00,
                "high": 1.40,
                "low": 1.25,
            },
            "timeframe_penalty_mult": {
                "5min": 1.45,
                "15min": 1.00,
            },
        },
        # Hierarchical adaptive profile (default -> regime -> session -> timeframe -> strategy -> side).
        "context_profiles": {
            "default": {
                "min_rr": 1.20,
                "min_ev_points": 0.12,
                "drift_max_atr": 0.80,
                "min_edge_prob": 0.035,
            },
            "regime": {
                "high": {
                    "min_rr": 1.35,
                    "min_ev_points": 0.24,
                    "drift_max_atr": 0.58,
                    "min_edge_prob": 0.060,
                    "ev_winrate_bias": -0.06,
                    "rank_boost": -0.10,
                    "density_penalty_mult": 1.30,
                    "context_soft_cap": 8,
                    "strategy_soft_cap": 1,
                    "quality_overrides": {
                        "min_body_atr": 0.32,
                        "max_range_atr": 1.95,
                        "max_gap_atr": 0.45,
                        "max_shock_range_atr": 2.40,
                    },
                },
                "normal": {
                    "min_rr": 1.20,
                    "min_ev_points": 0.12,
                    "drift_max_atr": 0.85,
                    "min_edge_prob": 0.030,
                    "rank_boost": 0.00,
                    "density_penalty_mult": 0.90,
                },
                "low": {
                    "min_rr": 1.30,
                    "min_ev_points": 0.20,
                    "drift_max_atr": 0.62,
                    "min_edge_prob": 0.055,
                    "ev_winrate_bias": -0.05,
                    "rank_boost": -0.08,
                    "density_penalty_mult": 1.15,
                    "context_soft_cap": 7,
                    "strategy_soft_cap": 1,
                    "quality_overrides": {
                        "min_body_atr": 0.22,
                        "max_range_atr": 1.85,
                        "max_gap_atr": 0.40,
                        "max_shock_range_atr": 1.95,
                    },
                },
            },
            "session": {
                "ASIA": {
                    "drift_max_atr": 0.95,
                    "min_edge_prob": 0.030,
                    "rank_boost": -0.01,
                    "density_penalty_mult": 1.05,
                },
                "LONDON": {
                    "drift_max_atr": 0.85,
                    "min_edge_prob": 0.045,
                    "rank_boost": -0.06,
                    "density_penalty_mult": 1.30,
                },
                "NY_AM": {
                    "min_rr": 1.30,
                    "min_ev_points": 0.22,
                    "drift_max_atr": 0.65,
                    "min_edge_prob": 0.055,
                    "ev_winrate_bias": -0.06,
                    "density_penalty_mult": 1.35,
                    "context_soft_cap": 7,
                    "strategy_soft_cap": 1,
                    "quality_overrides": {
                        "min_body_atr": 0.30,
                        "max_gap_atr": 0.45,
                    },
                },
                "NY_PM": {
                    "min_rr": 1.24,
                    "min_ev_points": 0.14,
                    "drift_max_atr": 0.58,
                    "min_edge_prob": 0.035,
                    "ev_winrate_bias": -0.01,
                    "density_penalty_mult": 1.05,
                    "context_soft_cap": 10,
                    "strategy_soft_cap": 2,
                    "quality_overrides": {
                        "min_body_atr": 0.24,
                        "max_gap_atr": 0.50,
                    },
                },
            },
            "timeframe": {
                "5min": {
                    "min_rr": 1.40,
                    "min_ev_points": 0.24,
                    "drift_max_atr": 0.60,
                    "min_edge_prob": 0.065,
                    "density_penalty_mult": 1.45,
                    "context_soft_cap": 6,
                    "strategy_soft_cap": 1,
                    "quality_overrides": {
                        "min_body_atr": 0.34,
                        "max_range_atr": 1.90,
                        "max_gap_atr": 0.42,
                    },
                },
                "15min": {
                    "min_rr": 1.20,
                    "min_ev_points": 0.12,
                    "drift_max_atr": 0.90,
                    "min_edge_prob": 0.030,
                    "rank_boost": 0.03,
                    "density_penalty_mult": 0.90,
                    "context_soft_cap": 12,
                    "strategy_soft_cap": 3,
                },
            },
            "strategy": {
                "Gap_Reversal": {
                    "min_rr": 1.28,
                    "min_ev_points": 0.20,
                    "drift_max_atr": 0.90,
                    "min_edge_prob": 0.050,
                    "ev_winrate_bias": -0.04,
                    "density_penalty_mult": 1.15,
                    "strategy_soft_cap": 1,
                    "quality_overrides": {
                        "min_body_atr": 0.20,
                        "max_range_atr": 2.00,
                        "max_gap_atr": 0.55,
                    },
                },
                "Engulfing": {
                    "min_rr": 1.18,
                    "min_ev_points": 0.12,
                    "min_edge_prob": 0.025,
                    "rank_boost": 0.02,
                },
                "Wick_Rejection": {
                    "min_rr": 1.16,
                    "min_ev_points": 0.10,
                    "min_edge_prob": 0.022,
                    "rank_boost": 0.01,
                },
                "Follow_Color": {
                    "min_rr": 1.20,
                    "min_ev_points": 0.12,
                    "min_edge_prob": 0.030,
                    "quality_overrides": {
                        "long_min_close_pos": 0.60,
                        "short_max_close_pos": 0.40,
                    },
                },
                "Fade_Color": {
                    "min_rr": 1.18,
                    "min_ev_points": 0.12,
                    "min_edge_prob": 0.028,
                },
                "Inside_Break": {
                    "min_rr": 1.24,
                    "min_ev_points": 0.16,
                    "min_edge_prob": 0.035,
                    "quality_overrides": {
                        "long_min_close_pos": 0.62,
                        "short_max_close_pos": 0.38,
                    },
                },
            },
            "side": {},
        },
    },
    "DYNAMIC_ENGINE2_DRIFT": {
        "enabled": True,
        "max_atr": 0.90,
        "atr_period": 14,
        "fallback_points": 0.0,
    },
    # DynamicEngine (DE1) identity policy:
    # Deterministic, Sharpe-first quality gates to keep DE1 distinct from DE3's
    # model/veto-heavy adaptive selection flow.
    "DYNAMIC_ENGINE_DE1_POLICY": {
        "enabled": True,
        # Base quality gates from the winning legacy candidate.
        "min_opt_wr": 0.22,
        "min_final_score": 8.0,
        "min_body_thresh_ratio": 1.10,
        # Reversion entries with oversized impulse bars are usually continuation traps.
        "max_reversion_body_thresh_ratio": 3.00,
        # Regime identity: DE1 is reversion-first in low/normal vol, momentum in normal/high.
        "momentum_allowed_regimes": ["normal", "high"],
        "reversion_allowed_regimes": ["low", "normal"],
        # Trigger-candle shape filters.
        "momentum_min_body_range_ratio": 0.55,
        "momentum_long_min_close_pos": 0.65,
        "momentum_short_max_close_pos": 0.35,
        "reversion_long_max_close_pos": 0.35,
        "reversion_short_min_close_pos": 0.65,
        "max_trigger_range_atr": 2.80,
        "momentum_min_body_atr": 0.40,
        "atr_period": 14,
        # Anti-cluster guard to reduce local serial correlation in entries.
        "cooldown_bars": 1,
        # Post-bracket quality floor.
        "min_rr": 1.20,
    },
    # DynamicEngine3: override NY gate (disabled by default here)
    "DYNAMIC_ENGINE3_NY_CONF": {
        "enabled": False,
        "sessions": ["09-12", "12-15", "15-18"],
        "min_opt_wr": 0.25,
        "min_final_score": None,
    },
    "DYNAMIC_ENGINE3_DB_FILE": "dynamic_engine3_strategies.json",
    # Runtime DB selector for DynamicEngine3.
    # - "v1": always use DYNAMIC_ENGINE3_DB_FILE
    # - "v2": use DE3_V2.db_path only when DE3_V2.enabled is True
    # - "v3": use DE3_V3.member_db_path + family-first DE3v3 runtime flow
    # - "v4": use DE3_V4.member_db_path + hierarchical router/lane/bracket runtime
    "DE3_VERSION": "v1",
    "DE3_V2": {
        "enabled": False,
        "db_path": "dynamic_engine3_strategies_v2.json",
        "mode": "fixed_split",  # fixed_split | rolling
        # Keep 2025 fully out-of-sample: train through 2023, validate on 2024 only.
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
        "purge_bars": 200,
        "plateau": {
            "enabled": True,
            "min_neighbors": 4,
            "neighbor_def": "adjacent_grid",
            "min_plateau_score": 0.0,
        },
        "scoring": {
            "lambda_std": 0.95,
            "gamma_dd": 0.50,
            "min_oos_trades": 80,
            "min_profitable_blocks": 3,
            "min_train_trades": 50,
            # Keep fixed-split OOS gating coarse; avoid over-constraining candidate diversity.
            "min_oos_profitable_blocks": None,
            "min_oos_profitable_block_ratio": None,
            "min_oos_profit_factor": None,
            "min_oos_win_rate": None,
            "max_oos_stop_share": None,
            "max_oos_tail_p10_abs": None,
            "max_oos_drawdown_norm": 0.90,
            "gamma_stop_share": None,
            "gamma_loss_share": None,
            "gamma_tail_p10": None,
            # Structural robustness: prune higher-threshold near-duplicates within
            # the same TF/session/type + bracket when a lower-threshold variant has
            # similar edge but better shape/coverage.
            "dominance_pruning_enabled": True,
            "dominance_avg_pnl_tolerance": 0.10,
            "dominance_score_tolerance": 0.50,
            "dominance_dd_tolerance": 0.05,
            "dominance_require_lower_or_equal_thresh": True,
        },
        "robust_ranking": {
            "enabled": True,
            # hard trust gates
            "min_oos_trades": 80,
            "min_profitable_block_ratio": 0.60,
            "min_worst_block_avg_pnl": -0.25,
            "min_worst_block_pf": 0.90,
            "max_oos_drawdown_norm": 0.80,
            # Shape quality stays penalized in StructuralScore, but these are
            # advisory thresholds (not runtime hard vetoes).
            "max_stop_like_share": 0.50,
            "max_loss_share": 0.65,
            "max_tail_p10_abs_sl_mult": 1.00,
            # support scaling
            "trade_conf_tau": 100,
            # structural score weights
            "weights": {
                "avg_pnl": 1.50,
                "profit_factor": 1.00,
                "win_rate": 0.50,
                "trade_confidence": 0.60,
                "profitable_block_ratio": 1.00,
                "worst_block_avg_pnl": 0.85,
                "worst_block_pf": 0.35,
                "drawdown_norm": -1.10,
                "stop_like_share": -0.90,
                "loss_share": -0.70,
                "tail_p10": -0.80,
                "block_std": -0.80,
                "sharpe_like": 0.20,
            },
            # runtime score weights
            "runtime_weights": {
                "edge_points": 0.35,
                "edge_gap": 0.20,
                "structural_score": 0.30,
                "bucket_score": 0.10,
                "confidence": 0.05,
                "ambiguity_penalty": -0.15,
                "concentration_penalty": -0.10,
            },
            # runtime abstain
            "runtime_abstain": {
                "enabled": True,
                "min_edge_points": 0.16,
                "min_edge_gap_points": 0.12,
                "min_structural_score": -1.00,
                "min_runtime_rank_score": 0.02,
            },
            # diagnostics
            "log_top_k": 3,
            "log_decisions": True,
        },
        "search_space": {
            "thresholds": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            "sl_list": [3, 4, 5, 6, 8, 10, 12, 15],
            "rr_list": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
            "max_per_bucket": 4,
        },
        "diversity": {
            "enabled": True,
            # Soft score penalties for repeated bracket usage (no hard blocks).
            "combo_penalty": 0.08,
            "sl_penalty": 0.02,
            "tp_penalty": 0.02,
            "global_scale": 0.50,
            # Fixed-split: keep a small set of high-quality alternate plateau brackets
            # per candidate so soft diversity has real choices.
            "candidates_per_task": 2,
            "max_cluster_drop": 0.30,
        },
        "rolling": {
            "train_years": 5,
            "valid_years": 1,
            "step_years": 1,
        },
        "workers": 1,
        "acceleration": "cpu",  # cpu | gpu | auto
        "symbol_mode": "auto_by_day",
        "symbol_method": "volume",
        "trade_resolution": "1m",
        "max_horizon": 180,
        "limit_to_session": True,
        "exit_at_horizon": "close",
        "assume_sl_first": False,
        # Align trainer tie-break behavior with backtest same-bar SL/TP conflict logic.
        "sl_tp_conflict": "ohlc",
        "execution": {
            "enforce_no_new_entries_window": True,
            "no_new_entries_start_hour_et": 16,
            "no_new_entries_end_hour_et": 18,
            "force_flat_at_hour_enabled": True,
            "force_flat_hour_et": 16,
        },
        "min_tp": 4.0,
        "max_tp": 30.0,
    },
    # DynamicEngine3 v3 (family-first runtime over DE3 v2 member DB).
    "DE3_V3": {
        "enabled": True,
        # Exact member candidates still come from DE3 v2 DB (family members).
        "member_db_path": "dynamic_engine3_strategies_v2.json",
        # Primary consolidated DE3v3 artifact bundle (runtime can also read legacy family JSON).
        "family_db_path": "dynamic_engine3_v3_bundle.json",
        # Optional compatibility output path for legacy family inventory writes.
        "family_inventory_legacy_path": "dynamic_engine3_families_v3.json",
        # If family artifact is missing, build it from member_db_path on startup.
        "auto_build_family_db": True,
        # Build-time context-profile sources and support thresholds.
        "context_profiles": {
            "enabled": True,
            "decision_csv_path": "reports/de3_decisions.csv",
            "trade_attribution_csv_path": "reports/de3_decisions_trade_attribution.csv",
            "decision_summary_json_path": "reports/de3_decisions_summary.json",
            "threshold_sensitivity_json_path": "reports/de3_threshold_sensitivity.json",
            "bucket_attribution_csv_path": "reports/de3_bucket_attribution.csv",
            "export_raw_context_fields_in_decision_journal": True,
            "min_bucket_samples": 12,
            "strong_bucket_samples": 40,
            # If required raw context fields are missing in decision export,
            # keep building with available fallbacks but mark the schema audit as needing enrichment.
            "require_enriched_export_for_runtime": False,
            "allow_parse_legacy_context_inputs": True,
        },
        # Refined DE3v3 family/member universe controls (primary runtime input for v3).
        "refined_universe": {
            "enabled": True,
            "runtime_use_refined": True,
            "allow_runtime_raw_universe_override": True,
            "max_retained_families": 12,
            "max_retained_members_per_family": 2,
            "min_family_quality_strong": 1.10,
            "min_family_quality_keep": 0.35,
            "min_family_quality_weak": 0.05,
            "min_member_quality_anchor": 0.95,
            "min_member_quality_keep": 0.30,
            "min_member_quality_weak": 0.00,
            "enable_cluster_distinctiveness_filter": True,
            "distinctiveness_margin": 0.30,
            "allow_weak_family_if_universe_too_thin": True,
            "min_retained_families": 3,
            "allow_weak_member_for_diversity": True,
            "anchor_member_selection_mode": "best_quality",
            "member_diversity_rr_min_separation": 0.20,
            "member_diversity_sl_min_separation": 1.0,
            "require_meaningful_context_support_for_context_weight": True,
            "low_support_context_weight_cap": 0.02,
        },
        # Explicit DE3v3 core anchor architecture.
        "de3v3_core": {
            "enabled": True,
            "anchor_family_ids": [
                "5min|09-12|long|Long_Rev|T6",
            ],
            # Supported: core_only | core_plus_satellites | satellites_only
            "default_runtime_mode": "core_plus_satellites",
            # Backward-compatible alias used in some reports/logs.
            "core_mode": "anchor_plus_satellites",
            "force_anchor_when_eligible": True,
        },
        # Portfolio-aware satellite discovery around the core anchor.
        "de3v3_satellites": {
            "enabled": True,
            "discovery_enabled": True,
            "min_standalone_viability": 0.20,
            "min_incremental_value_over_core": 0.05,
            "max_retained_satellites": 6,
            "require_orthogonality": True,
            "max_overlap_with_core": 0.80,
            "max_bad_overlap_with_core": 0.55,
            "allow_near_core_variants_if_incremental": True,
        },
        # Disable anti-T6 runtime complexity by default; keep behind explicit flags.
        "bloat_control": {
            "enable_family_competition_balancing": False,
            "enable_exploration_bonus": False,
            "enable_dominance_penalty": False,
            "enable_monopoly_canonical_force": False,
            "enable_compatibility_tier_slot_pressure": False,
        },
        # V3-native realized family usability state derived from DE3v3 decision history.
        "usable_family_universe": {
            "enabled": True,
            "runtime_state_json_path": "reports/de3_family_runtime_state.json",
            # LOW/NONE support is an evidence tier, not exclusion.
            "low_support_fully_competitive": True,
            # When true, only explicitly suppressed families are excluded from competition.
            "exclude_only_suppressed_families": True,
            # Orthogonal evidence-support tier thresholds (none/low/mid/strong).
            "evidence_support": {
                "min_mid_samples": 8,
                "strong_samples": 20,
            },
            # Reserve fallback_only for prior-eligible families with weak realized behavior.
            "fallback_only": {
                "min_trades": 10,
                "if_profit_factor_below": 0.90,
                "if_avg_pnl_below": -0.20,
                "if_stop_rate_above": 0.70,
                "if_stop_gap_rate_above": 0.30,
                "if_fallback_to_prior_rate_above": 0.90,
            },
            # Explicit suppression thresholds (hard exclusion state).
            "suppression": {
                "min_trades": 20,
                "if_profit_factor_below": 0.70,
                "if_stop_rate_above": 0.78,
                "if_stop_gap_rate_above": 0.35,
                "if_avg_pnl_below": -0.80,
            },
            # Bounded evidence adjustment by support tier.
            "evidence_adjustment": {
                "support_tier_base": {
                    "none": 0.00,
                    "low": -0.005,
                    "mid": 0.03,
                    "strong": 0.08,
                },
                "support_tier_quality_scale": {
                    "none": 0.00,
                    "low": 0.01,
                    "mid": 0.08,
                    "strong": 0.15,
                },
                "fallback_only_penalty": -0.05,
                "suppressed_adjustment": -0.20,
                "low_tier_min_adjustment": -0.02,
                "max_abs_adjustment": 0.20,
                "quality_confidence_trades": 40,
                # Cap context influence when evidence support is weak.
                "context_scale_by_evidence_tier": {
                    "none": 0.00,
                    "low": 0.02,
                    "mid": 0.60,
                    "strong": 1.00,
                },
            },
            "reject_suppressed_families": True,
            "fallback_only_requires_weak_or_thin_competition": True,
            "min_active_competitors_for_fallback_restriction": 1,
            # Legacy keys retained for backward compatibility with older artifacts/scripts.
            "min_trades_for_state": 8,
            "min_trades_for_active": 20,
            "min_pf_for_fallback": 0.85,
            "quality_confidence_trades": 40,
        },
        # Conservative prior-eligibility pass (coarse prior sanity filter).
        # Layer 1: determines whether family is allowed to compete at all.
        "prior_eligibility": {
            "enabled": True,
            "min_total_support_trades": 60,
            "min_best_member_profit_factor": 1.00,
            "min_best_member_profitable_block_ratio": 0.50,
            "min_best_member_worst_block_pf": 0.70,
            "min_best_member_worst_block_avg_pnl": -0.50,
            "max_median_drawdown_norm": 1.20,
            "max_median_loss_share": 0.75,
            "min_median_member_structural_score": -2.0,
            # Catastrophic-only hard exclusions (intentionally strict).
            "catastrophic_min_best_member_profit_factor": 0.50,
            "catastrophic_min_best_member_worst_block_pf": 0.25,
            "catastrophic_max_median_drawdown_norm": 2.30,
            "catastrophic_max_median_loss_share": 0.95,
            "log_rejections": True,
        },
        # Backward-compatible alias for older tooling.
        "family_eligibility": {
            "enabled": True,
            "min_total_support_trades": 60,
            "min_best_member_profit_factor": 1.00,
            "min_best_member_profitable_block_ratio": 0.50,
            "min_best_member_worst_block_pf": 0.70,
            "min_best_member_worst_block_avg_pnl": -0.50,
            "max_median_drawdown_norm": 1.20,
            "max_median_loss_share": 0.75,
            "min_median_member_structural_score": -2.0,
            "catastrophic_min_best_member_profit_factor": 0.50,
            "catastrophic_min_best_member_worst_block_pf": 0.25,
            "catastrophic_max_median_drawdown_norm": 2.30,
            "catastrophic_max_median_loss_share": 0.95,
            "log_rejections": True,
        },
        # Bootstrap competition floor to prevent early one-family collapse.
        "family_competition": {
            "use_bootstrap_family_competition_floor": True,
            "bootstrap_min_competing_families": 3,
            # Temporary DE3v3 runtime blocklist for experiments.
            "temporary_excluded_thresholds": [],
            "temporary_excluded_family_ids": [],
            # Coarse compatibility-band eligibility (pre-score candidate construction).
            "include_exact_and_compatible_only": True,
            # Legacy direct cap fields kept for backward compatibility.
            "max_family_candidates_per_decision": 6,
            "compatible_family_max_count": 4,
            "compatible_family_penalty": -0.06,
            # Post-eligibility candidate-cap policy.
            "family_candidate_cap": {
                "enabled": True,
                "max_total_candidates": 6,
                "min_exact_match_candidates": 2,
                "min_compatible_band_candidates": 2,
                "max_exact_match_candidates": 4,
                "max_compatible_band_candidates": 4,
                "use_preliminary_score_for_cap": True,
                "compatibility_penalty_exact": 0.0,
                "compatibility_penalty_compatible": -0.06,
                "log_pre_cap_post_cap": True,
            },
            "compatibility_bands": {
                # Session bands: exact match is primary; nearby windows are compatible.
                "session_nearby_max_hour_distance": 6.0,
                # Timeframe bands: same = primary; nearby bucket = compatible.
                "timeframe_nearby_max_minutes_delta": 10,
                "timeframe_nearby_max_ratio": 3.0,
                # Strategy type: allow related long/short subtype compatibility.
                "strategy_type_allow_related": True,
            },
            "family_competition_balance": {
                "enabled": False,
                "dominance_window_size": 160,
                "dominance_penalty_start_share": 0.55,
                "dominance_penalty_max": 0.12,
                "max_dominance_penalty": 0.12,
                "low_support_exploration_bonus": 0.08,
                "max_exploration_bonus": 0.08,
                "exploration_bonus_decay_threshold": 20,
                "competition_margin_points": 0.22,
                "cap_context_advantage_in_close_competition": True,
                "max_context_advantage_cap": 0.12,
            },
        },
        # Runtime family selection scoring (compact, interpretable).
        "family_scoring": {
            "weights": {
                "context_profile_expectancy": 0.45,
                "context_profile_confidence": 0.20,
                "family_prior": 0.15,
                "v3_realized_usability": 0.15,
                "adaptive_policy": 0.10,
            },
            # Runtime concentration controls (v3 scoring-path correction).
            "normalize_prior_component": True,
            "cap_context_advantage_when_single_strong_family": True,
            "single_strong_family_context_cap": 0.10,
            "compatible_band_penalty": -0.03,
            "close_competition_margin": 0.24,
            "max_competition_adjustment_close": 0.12,
            "max_competition_adjustment_far": 0.03,
            "dominance_penalty_curve": "quadratic",
            "exploration_bonus_curve": "quadratic_decay",
            "log_score_delta_ladder": True,
            "active_context_dimensions": [
                "volatility_regime",
                "compression_expansion_regime",
                "confidence_band",
            ],
            "use_context_profiles": True,
            "fallback_to_priors_when_profile_weak": True,
            # Keep scoring context compact in this iteration.
            "use_joint_context_profiles": False,
            "min_context_bucket_samples": 12,
            "strong_context_bucket_samples": 40,
            # Trust context strongly only with strong support; otherwise blend back to priors.
            "context_profile_weight_strong": 1.00,
            "context_profile_weight_mid": 0.45,
            "context_profile_weight_low_or_none": 0.02,
            "gates": {
                "enabled": True,
                "min_family_score": 0.05,
                "min_adaptive_component": -2.0,
            },
            "log_decisions": True,
            "log_top_k": 3,
        },
        # Local member/bracket selection only within the chosen family.
        "local_member_selection": {
            "weights": {
                "edge_points": 0.55,
                "structural_score": 0.30,
                "payoff": 0.10,
                "context_bracket_suitability": 0.10,
                "confidence": 0.05,
            },
            "target_rr": 1.50,
            "rr_tolerance": 1.50,
            "target_rr_expanding": 2.00,
            "target_rr_compressed": 1.15,
            "target_sl_atr_expanding": 1.40,
            "target_sl_atr_neutral": 1.10,
            "target_sl_atr_compressed": 0.85,
            "sl_atr_tolerance": 0.90,
            # Local bracket adaptation thresholds by canonical support tier.
            "full_adaptation_min_support_tier": "strong",
            "conservative_adaptation_min_support_tier": "mid",
            # Strong support: full adaptation; mid support: conservative; low support: freeze.
            "context_bracket_weight_scale_mid": 0.45,
            "context_bracket_weight_scale_low": 0.00,
            "allow_context_adaptation_mid_support": True,
            "mid_support_noncanonical_penalty": 0.05,
            "freeze_to_canonical_when_low_support": True,
            "force_canonical_when_family_monopoly": False,
            "monopoly_share_threshold": 0.75,
            "monopoly_lookback_window": 140,
            "min_local_score": -999.0,
        },
        # Full-spectrum DE3v3 observability controls.
        "observability": {
            "enabled": True,
            "emit_family_score_trace": True,
            "emit_member_resolution_audit": True,
            "emit_choice_path_audit": True,
            "emit_score_path_audit": True,
            "strict_score_path_assertions": False,
            "family_score_trace_max_rows": 300000,
            "member_resolution_trace_max_rows": 300000,
        },
    },
    # DynamicEngine3 v4 (clean hierarchical runtime: router -> lane selector -> bracket).
    "DE3_V4": {
        "enabled": True,
        # Exact member candidates still come from DE3 member DB; v4 changes runtime decision architecture.
        # Live promotion 2026-04-07:
        # use the exact-tested longfix balanced DE3 research DB directly.
        "member_db_path": "dynamic_engine3_strategies_v2_research_london_shortfix_20260407.json",
        # Canonical DE3v4 historical source (ES 1-minute parquet).
        "training_data": {
            "parquet_path": "es_master_outrights.parquet",
            "source_data_format": "parquet",
            # If this column is missing, trainer falls back to datetime index.
            "timestamp_column": "timestamp",
            # Never assume local-machine timezone for naive timestamps.
            "assume_timezone_if_naive": "UTC",
            "required_columns": ["open", "high", "low", "close", "volume"],
            # Strict anti-leakage default: treat DE3 source DB as inventory/identity input,
            # not as a source of fit-time performance targets.
            "allow_source_db_performance_metrics_for_training": False,
            # Strict DE3v4 split policy:
            # train: 2011-2023, tune: 2024, true OOS: 2025, future holdout: 2026+.
            "split": {
                "train_start": "2011-01-01",
                "train_end": "2023-12-31",
                "tune_start": "2024-01-01",
                "tune_end": "2024-12-31",
                "oos_start": "2025-01-01",
                "oos_end": "2025-12-31",
                "future_start": "2026-01-01",
            },
            # Keep DE3v4 label/outcome generation aligned with executable backtest rules.
            "execution_rules": {
                "enforce_no_new_entries_window": True,
                "no_new_entries_start_hour_et": 16,
                "no_new_entries_end_hour_et": 18,
                "force_flat_at_time": True,
                "force_flat_hour_et": 16,
                "force_flat_minute_et": 0,
            },
        },
        # Primary DE3v4 runtime/training artifact.
        # Live promotion 2026-04-24: pin the exact promoted daytype-soft bundle.
        "bundle_path": "artifacts/de3_v4_live/dynamic_engine3_v4_bundle.decision_side_daytype_soft_v1_20260422_promoted.json",
        "reports_dir": "reports",
        # If bundle is missing, runtime stays safe and falls back to candidate-level defaults.
        "auto_build_bundle": False,
        "core": {
            "enabled": False,
            "anchor_family_ids": [
                "5min|09-12|long|Long_Rev|T6",
            ],
            # Supported: core_only | core_plus_satellites | satellites_only
            "default_runtime_mode": "satellites_only",
            "force_anchor_when_eligible": False,
        },
        "satellites": {
            "enabled": True,
            "discovery_enabled": True,
            "min_standalone_viability": 0.20,
            "min_incremental_value_over_core": 0.05,
            # Used when strict anti-leak mode disables source DB performance metrics.
            "min_standalone_viability_fallback": 0.20,
            "min_incremental_value_over_core_fallback": 0.00,
            "max_retained_satellites": 8,
            "require_orthogonality": True,
            "max_overlap_with_core": 0.85,
            "max_bad_overlap_with_core": 0.60,
            "allow_near_core_variants_if_incremental": True,
        },
        "runtime": {
            # Keep runtime trace buffers bounded for long backtests to prevent
            # gradual slowdown from unbounded in-memory row growth.
            # 0 disables the cap.
            "trace_max_rows": 250000,
            "family_profile_veto": {
                "enabled": True,
                "rules": [
                    {
                        "name": "frv_06_09_long_rev_t2_normal_grind_down_distributed_normal_15m",
                        "enabled": True,
                    },
                ],
            },
            # Legacy pre-router gates can be selectively disabled for v4 experiments.
            "disable_context_policy_gate": True,
            "disable_context_veto_gate": True,
            "disable_ny_conf_gate": True,
            # Runtime-level exclusion controls (used by DE3v4 runtime candidate construction).
            # Keep the dead 09-12 Long_Rev T6 family explicitly excluded so future
            # exports / research paths do not depend on core-disabled safety to drop it.
            "excluded_family_ids": [
                "5min|09-12|long|Long_Rev|T6",
            ],
            # Baseline parity: no hard variant-pattern suppression.
            "excluded_variant_patterns": [
                # Persistent cross-year losers in the locked DE3v4 baseline:
                # negative in both 2024 tune and 2025 OOS with meaningful sample size.
                "5min_09-12_Long_Rev_T3_SL10_TP25",
                # 2025 OOS + 2011-2024 current-pool drag (validated in March 2026 sweeps).
                "15min_09-12_Long_Rev_T3_SL10_TP12.5",
                "15min_09-12_Long_Rev_T3_SL10_TP25",
                "5min_15-18_Short_Mom_T2_SL10_TP25",
                "15min_12-15_Long_Mom_T2_SL10_TP25",
                "5min_03-06_Long_Rev_T2_SL10_TP25",
                # Robust post-lock exclusion:
                # near-flat in 2024, negative in 2025, and materially negative again
                # on the Jan 2026 forward holdout when replayed on the live bundle.
                "15min_18-21_Long_Rev_T2_SL10_TP25",
                # Live promotion 2026-04-07:
                # remove the two exact-tested weak 12-15 short-reversion pockets.
                "5min_12-15_Short_Rev_T6_SL8_TP12",
                "5min_12-15_Short_Rev_T5_SL10_TP12.5",
            ],
            # v4-only pre-candidate signal gate override (applied inside DynamicSignalEngine3
            # when DE3_VERSION=v4) so non-core lanes are not hard-pruned by legacy v2/v3
            # runtime constraints before router/lane selection.
            "signal_gate": {
                "enabled": True,
                "use_runtime_gate": False,
                "use_db_settings": False,
                "db_recent_gate": False,
                "min_trades": 0,
                "min_score": -999.0,
                "min_avg_pnl": -999.0,
                "min_win_rate": -999.0,
                "recent_mode": "none",
                # Keep v4 from being filtered by legacy global runtime abstain thresholds.
                "disable_runtime_abstain": True,
            },
            # Legacy v4 hard-threshold filter block (kept for backward compatibility
            # and migration; canonical runtime path now uses execution_policy below).
            "execution_filters": {
                "enabled": False,
                "min_route_confidence": 0.00,
                "min_edge_points": 0.10,
                "min_structural_score": -0.10,
                "min_lane_score": 0.12,
                "min_variant_quality_prior": 0.08,
                "max_loss_share": 0.75,
                "max_stop_like_share": 0.85,
            },
            # Canonical v4 post-lane execution policy.
            # - hard_limits block obviously weak setups
            # - quality score tiers keep moderate setups tradeable while avoiding
            #   all-or-nothing over-pruning.
            "execution_policy": {
                "enabled": True,
                # Binding post-lane veto: reject low-quality setups outright.
                "enforce_veto": False,
                "soft_tier_on_reject": "conservative",
                "hard_limits": {
                    "min_route_confidence": 0.00,
                    "min_edge_points": 0.05,
                    "min_structural_score": -0.25,
                    "min_lane_score": 0.08,
                    "min_variant_quality_prior": 0.00,
                    "max_loss_share": 0.88,
                    "max_stop_like_share": 0.94,
                },
                # Keep non-binding by default; pre-router gate provides the
                # calibrated filtering signal.
                "reject_quality_score_below": 0.24,
                "conservative_quality_score_below": 0.40,
                "weights": {
                    "route_confidence": 0.22,
                    "edge_points": 0.28,
                    "lane_score": 0.22,
                    "structural_score": 0.12,
                    "variant_quality_prior": 0.08,
                    "loss_quality": 0.04,
                    "stop_quality": 0.04,
                },
                "ranges": {
                    "route_confidence": {"min": 0.00, "max": 0.40},
                    "edge_points": {"min": -0.18, "max": 0.78},
                    "lane_score": {"min": -0.18, "max": 0.78},
                    "structural_score": {"min": -0.75, "max": 1.75},
                    "variant_quality_prior": {"min": 0.00, "max": 1.00},
                    "loss_share": {"min": 0.25, "max": 0.90},
                    "stop_like_share": {"min": 0.25, "max": 0.94},
                },
                # Calibrated DE3v4 entry model (trained on train split, tuned on 2024).
                "calibrated_entry_model": {
                    "enabled": True,
                    # Keep live/backtest entry gating aligned with the promoted
                    # DE3 bundle instead of relying on a separate manual sync
                    # step each time the bundle changes.
                    "use_bundle_model": True,
                    "enforce_veto": True,
                    "allow_on_missing_stats": True,
                    "min_variant_trades": 25,
                    "min_lane_trades": 120,
                    # Baseline-parity fix: allow lane scope so lane-fallback candidates
                    # are evaluated by score instead of being auto-rejected.
                    "allowed_scopes": ["variant", "lane"],
                    # Runtime override for risk-first entry filtering.
                    "selected_threshold": -0.422708,
                    # Guard lane/global fallback so weak broad stats do not
                    # admit low-quality variants.
                    "fallback_scope_guard_enabled": True,
                    "fallback_guard_scopes": ["lane", "global"],
                    "fallback_allow_global_scope": False,
                    "fallback_min_ev_lcb_points": -0.05,
                    "fallback_min_quality_lcb_score": -0.02,
                    "fallback_min_p_win_lcb": 0.30,
                    "fallback_min_worst_block_avg_pnl": -60.0,
                    "fallback_min_year_coverage": 5,
                    "fallback_min_variant_quality_prior": 0.0,
                    # Scope-aware gating: require stronger score when falling
                    # back from variant stats to lane/global stats.
                    "scope_threshold_offsets": {
                        "variant": 0.00,
                        "lane": 0.06,
                        "global": 0.12,
                        "missing": 0.15,
                        "default": 0.00,
                    },
                    "conservative_buffer": 0.035,
                    "weight_quality_lcb": 0.65,
                    "weight_route_confidence": 0.20,
                    "weight_edge_points": 0.10,
                    "weight_structural_score": 0.05,
                    "weight_loss_share_penalty": 0.12,
                    "weight_stop_like_share_penalty": 0.08,
                    "weight_drawdown_penalty": 0.06,
                    "weight_worst_block_penalty": 0.08,
                    "route_confidence_center": 0.05,
                    "edge_scale_points": 0.40,
                    "structural_scale": 0.80,
                    "loss_share_center": 0.52,
                    "loss_share_scale": 0.22,
                    "stop_like_share_center": 0.62,
                    "stop_like_share_scale": 0.25,
                    "drawdown_scale": 6.0,
                    "worst_block_scale_points": 3.0,
                },
            },
            # Profit-gate v2: v4-only pre-router policy adaptation.
            # Uses legacy DE3 adaptive-policy model signals with v4 lane/session
            # overrides; configured here to be binding (not advisory).
            "pre_router_profit_gate_v2": {
                "enabled": False,
                "policy_mode": "block",  # block | shadow
                # Preserve 04:21 behavior: do not hard-block entire bar on top reject.
                "disable_block_all_on_top": True,
                "min_samples": 100,
                # Calibrated from 2024 good run + 2025 OOS sweep:
                # ~0.0024 removed a persistent loss-heavy tail while preserving
                # most trades in both periods.
                "max_p_loss_std": 0.0024,
                # Keep original permissive EV floor; binding comes from block behavior.
                "min_ev_lcb_points": -0.10,
                "min_ev_mean_points": None,
                # Preserve 04:21 behavior: soft-pass non-catastrophic rejects.
                "soft_pass_non_catastrophic_blocks": True,
                "soft_pass_risk_mult_cap": 0.85,
                "catastrophic": {
                    "enabled": True,
                    "min_samples": 160,
                    "max_ev_lcb_points": -0.25,
                    "min_p_loss": 0.60,
                    "max_p_loss_std": 0.0026,
                },
                # v4-native calibration: lane-aware thresholds (router lane universe).
                "lane_overrides": {
                    "long_rev": {
                        "min_samples": 80,
                        "max_p_loss_std": 0.0024,
                        "min_ev_lcb_points": -0.08,
                        "soft_pass_risk_mult_cap": 0.90,
                    },
                    "long_mom": {
                        "min_samples": 140,
                        "max_p_loss_std": 0.0022,
                        "min_ev_lcb_points": 0.00,
                        "soft_pass_risk_mult_cap": 0.80,
                    },
                    "short_rev": {
                        "min_samples": 130,
                        "max_p_loss_std": 0.0023,
                        "min_ev_lcb_points": -0.02,
                        "soft_pass_risk_mult_cap": 0.82,
                    },
                    "short_mom": {
                        "min_samples": 160,
                        "max_p_loss_std": 0.0022,
                        "min_ev_lcb_points": 0.02,
                        "soft_pass_risk_mult_cap": 0.78,
                    },
                },
                # Session-aware policy pressure (optional overlay).
                "session_overrides": {
                    "asia": {
                        "max_p_loss_std": 0.0022,
                        "min_ev_lcb_points": 0.00,
                        "soft_pass_risk_mult_cap": 0.78,
                    },
                    "london": {
                        "max_p_loss_std": 0.0024,
                        "min_ev_lcb_points": -0.08,
                        "soft_pass_risk_mult_cap": 0.86,
                    },
                    "ny_am": {
                        "max_p_loss_std": 0.0025,
                        "min_ev_lcb_points": -0.12,
                        "soft_pass_risk_mult_cap": 0.92,
                    },
                    "ny_pm": {
                        "max_p_loss_std": 0.0024,
                        "min_ev_lcb_points": -0.10,
                        "soft_pass_risk_mult_cap": 0.88,
                    },
                },
            },
            "router": {
                "min_route_confidence": 0.04,
                "min_lane_score_to_trade": -0.10,
                # Additional abstain guard: require sufficient best-vs-runner-up
                # route margin before taking a trade.
                "min_score_margin_to_trade": 0.0,
                # If only one lane has candidates, allow routing without artificial
                # best-vs-runner-up confidence collapse.
                "allow_single_lane_trade": True,
                # fixed | internal_margin
                "single_lane_confidence_mode": "fixed",
                # Requires lane-local top-vs-runner-up margin in single-lane cases.
                "min_single_lane_internal_margin": 0.0,
                "single_lane_internal_margin_scale": 0.05,
                "single_lane_confidence_min": 0.01,
                "single_lane_confidence_max": 0.20,
                # Used when single_lane_confidence_mode == fixed.
                "single_lane_confidence": 0.04,
            },
            "lane_selector": {
                "weights": {
                    "edge_points": 0.60,
                    "structural_score": 0.20,
                    "variant_quality_prior": 0.20,
                },
            },
            "confidence_tier_sizing": {
                "enabled": True,
                # Ordered fallback fields; first valid [0,1] confidence value is used.
                "confidence_field_priority": [
                    "de3_policy_confidence",
                    "de3_edge_confidence",
                    "de3_v4_route_confidence",
                ],
                "high_threshold": 0.86,
                "mid_threshold": 0.78,
                "high_multiplier": 1.00,
                "mid_multiplier": 0.80,
                "low_multiplier": 0.60,
                "min_contracts": 1,
                # Hard cap after confidence scaling (before drawdown scaling).
                "max_contracts": 5,
                # Robust cross-regime sizing: continuous quality adjustment
                # (not tied to a specific hour/session/variant).
                "quality_adjustment": {
                    "enabled": True,
                    "ev_lcb_field": "de3_policy_ev_lcb_points",
                    "ev_lcb_center": 3.20,
                    "ev_lcb_scale": 1.20,
                    "p_loss_std_field": "de3_policy_p_loss_std",
                    "p_loss_std_ref": 0.01,
                    "min_quality_multiplier": 0.65,
                    "max_quality_multiplier": 1.00,
                },
                # Variant-level risk shaping from long-horizon DE3 analysis:
                # keep the tradeable book intact, but de-emphasize variants whose
                # 2011-2023 history was materially weaker than the rest.
                "variant_size_multipliers": {
                    "5min_03-06_Long_Rev_T3_SL10_TP25": 0.25,
                    "5min_09-12_Short_Rev_T6_SL10_TP12.5": 0.80,
                    "15min_12-15_Long_Rev_T2_SL10_TP25": 0.80,
                    "15min_15-18_Short_Mom_T6_SL10_TP25": 0.80,
                    # Live promotion 2026-04-07:
                    # keep the weak but still-positive 15-18 long momentum pocket on
                    # a short leash instead of removing it outright.
                    "15min_15-18_Long_Mom_T5_SL10_TP25": 0.25,
                },
            },
            # Live-portable DE3v4 conditional size shaping:
            # keep the trade path intact, but trim specific weak live slices
            # that exact filterless validation showed were materially draggy.
            "signal_size_rules": {
                "enabled": True,
                "log_applies": False,
                "rules": [
                    {
                        "name": "live_09_12_long_mom_t3_close040_dist050_defensive_050",
                        "enabled": True,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_close_pos1": 0.40,
                        "min_dist_low5_atr": 0.50,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "live_09_12_long_mom_t3_body025_vol120_defensive_050",
                        "enabled": True,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_body1_ratio": 0.25,
                        "max_vol1_rel20": 1.20,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "live_06_09_long_rev_t2_size3_vol090_defensive_067",
                        "enabled": True,
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_vol1_rel20": 0.90,
                        # With the DE3 sizing stack this cleanly trims 3 -> 2
                        # while leaving 2-contract paths unchanged.
                        "size_multiplier": 0.67,
                        "min_contracts": 2,
                    },
                    {
                        "name": "live_18_21_long_rev_t2_distlow020_range565_defensive_050",
                        "enabled": True,
                        "apply_variants": ["5min_18-21_Long_Rev_T2_SL10_TP25"],
                        "max_dist_low5_atr": 0.20,
                        "max_range10_atr": 5.645,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "live_12_15_long_rev_t2_distlow050_range400_defensive_050",
                        "enabled": True,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_dist_low5_atr": 0.50,
                        "min_range10_atr": 4.0,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                ],
            },
            # Optional DE3v5-lite path: direct candidate chooser that can
            # replace router + lane selection when a bundle provides a trained
            # decision_policy_model. Safe by default for current live because
            # the promoted bundle does not include one yet.
            "direct_decision_model": {
                "enabled": True,
                "use_bundle_model": True,
                "selection_mode": "replace_router_lane",
                "score_margin_scale": 0.12,
                "confidence_floor": 0.05,
                "confidence_cap": 0.95,
                "min_confidence_to_override": 0.0,
                "min_score_delta_to_override": 0.0,
                "min_score_margin_to_override": 0.0,
                "allow_override_when_baseline_no_trade": False,
                "min_confidence_to_override_when_baseline_no_trade": 0.0,
                "min_score_delta_to_override_when_baseline_no_trade": 0.0,
                "min_score_margin_to_override_when_baseline_no_trade": 0.0,
                "min_baseline_score_advantage_to_override": 0.0,
                "min_baseline_score_delta_advantage_to_override": 0.0,
            },
            # Backtest-only DE3v4 realized-performance sizing:
            # use only prior closed trades for the same selected variant and
            # cut size when a lane is persistently cold, instead of permanently
            # blacklisting it across all regimes.
            # Current exact check status:
            # 2023 improved, 2025 improved, 2024 regressed slightly, so keep
            # disabled until a retuned threshold set clears all recent years.
            "backtest_variant_adaptation": {
                "enabled": False,
                "history_window_trades": 16,
                "warmup_trades": 6,
                "cold_avg_net_per_contract_usd": -10.0,
                "cold_max_winrate": 0.40,
                "cold_size_multiplier": 0.50,
                # Only downsize if the lane is recently cold and its lifetime
                # realized average is not clearly strong.
                "max_lifetime_avg_net_per_contract_usd": 2.0,
                "deep_cold_avg_net_per_contract_usd": None,
                "deep_cold_size_multiplier": 0.34,
                "min_contracts": 1,
                "reduce_only": True,
            },
            # Backtest-only DE3v4 admission controller:
            # maintain realized lane-context memory and defensively downsize
            # trades when that context is persistently cold. Exact annual
            # validation on the current break-even-only book improved 2022,
            # 2023, 2024, and 2025 without touching live execution.
            "backtest_admission_controller": {
                "enabled": True,
                # variant | lane | lane_context
                "key_granularity": "lane_context",
                "history_window_trades": 20,
                "warmup_trades": 10,
                "cold_avg_net_per_contract_usd": -10.0,
                "cold_max_winrate": 0.38,
                "defensive_size_multiplier": 0.60,
                # Optional deeper-cold hard block. Keep off until exact validation.
                "block_avg_net_per_contract_usd": None,
                "block_max_winrate": None,
                "min_contracts": 1,
                "reduce_only": True,
                # Keep this off for the validated default. The lane-context state
                # itself was sufficient; adding a second weakness gate made the
                # controller too inert in exact annual checks.
                "require_signal_weakness": False,
                "max_execution_quality_score": 0.60,
                "max_entry_model_margin": 0.10,
                "max_route_confidence": None,
                "max_edge_points": None,
            },
            # Backtest-only DE3v4 intraday regime controller:
            # use same-bar-only context to suppress weak countertrend DE3 trades
            # on strong directional days. This is intentionally isolated from
            # live until exact validation proves it adds value.
            "backtest_intraday_regime_controller": {
                "enabled": False,
                "mode": "block_defensive",
                "apply_sessions": ["LONDON", "NY_AM", "NY_PM"],
                "apply_lanes": [],
                "enable_bullish_mirror": True,
                "defensive_size_multiplier": 0.50,
                "min_contracts": 1,
                "reduce_only": True,
                "defensive_score_threshold": 2.80,
                "block_score_threshold": 4.10,
                "dominance_threshold": 0.70,
                "block_dominance_threshold": 1.10,
                "require_signal_weakness_for_block": True,
                "require_signal_weakness_for_defensive": False,
                "max_execution_quality_score": 0.78,
                "max_entry_model_margin": 0.16,
                "max_route_confidence": 0.18,
                "max_edge_points": None,
                "strong_execution_quality_score": 0.86,
                "strong_entry_model_margin": 0.22,
                "strong_route_confidence": 0.22,
                "strong_signal_relief": 0.65,
                "pressure_lookback_bars": 12,
                "pressure_balance_min": 0.16,
                "pressure_balance_weight": 0.90,
                "net_return_min_atr": 0.45,
                "net_return_weight": 0.85,
                "session_move_min_atr": 0.55,
                "session_move_weight": 0.95,
                "vwap_dist_min_atr": 0.18,
                "vwap_dist_scale_atr": 0.65,
                "vwap_dist_weight": 0.90,
                "vwap_slope_lookback_bars": 8,
                "vwap_slope_min_atr": 0.05,
                "vwap_slope_scale_atr": 0.16,
                "vwap_slope_weight": 0.75,
                "gap_location_weight": 0.60,
                "gap_location_low": 0.35,
                "gap_location_high": 0.65,
                "gap_outside_scale_atr": 0.70,
                "gap_outside_weight": 0.70,
                "route_bias_min": 0.12,
                "route_bias_weight": 0.55,
                "opening_range_minutes": 15,
                "opening_range_break_scale_atr": 0.70,
                "opening_range_weight": 1.10,
            },
            # Backtest-only DE3v4 walk-forward gate:
            # learned skip/defensive controller trained on prior realized DE3
            # trades. It is explicitly isolated from live until exact OOS
            # validation proves it is worth porting.
            "backtest_walkforward_gate": {
                "enabled": False,
                "artifact_path": "artifacts/de3_walkforward_gate/de3_walkforward_gate.json",
                # block | defensive | block_defensive
                "mode": "block_defensive",
                "defensive_size_multiplier": 0.50,
                "min_contracts": 1,
                "reduce_only": True,
            },
            # Backtest-only DE3v4 entry-model margin controller:
            # use the calibrated realized-value margin of admitted DE3v4 trades
            # to size marginal passes down and strong variant-scope passes up.
            # Keep disabled until exact multi-year validation proves it.
            "backtest_entry_model_margin_controller": {
                "enabled": False,
                "min_contracts": 1,
                "max_contracts": 5,
                "reduce_only": False,
                "defensive_max_margin": 0.08,
                "defensive_size_multiplier": 0.60,
                "lane_scope_size_multiplier": 0.80,
                "conservative_tier_size_multiplier": 0.80,
                "aggressive_min_margin": 0.22,
                "aggressive_size_multiplier": 1.25,
                "aggressive_variant_only": True,
            },
            # Backtest-only DE3v4 signal-conditioned size shaping:
            # preserve the trade path, but push known weak signal slices down to
            # a defensive size so drawdown scaling and occupancy stay intact.
            "backtest_signal_size_rules": {
                "enabled": False,
                "log_applies": False,
                "rules": [
                    {
                        "name": "robust_long_rev_upper_wick_080_defensive",
                        "enabled": False,
                        "apply_lanes": ["Long_Rev"],
                        "apply_sides": ["LONG"],
                        "min_upper_wick_ratio": 0.80,
                        "size_multiplier": 0.25,
                        "min_contracts": 1,
                    },
                    {
                        "name": "robust_5min_short_rev_low_close_pos_020_defensive",
                        "enabled": False,
                        "apply_timeframes": ["5min"],
                        "apply_lanes": ["Short_Rev"],
                        "apply_sides": ["SHORT"],
                        "max_close_pos1": 0.20,
                        "size_multiplier": 0.25,
                        "min_contracts": 1,
                    },
                    {
                        "name": "candidate_5min_short_rev_low_close_pos_015_defensive",
                        "enabled": False,
                        "apply_timeframes": ["5min"],
                        "apply_lanes": ["Short_Rev"],
                        "apply_sides": ["SHORT"],
                        "max_close_pos1": 0.15,
                        "size_multiplier": 0.25,
                        "min_contracts": 1,
                    },
                    {
                        "name": "hist_06_09_long_rev_t2_upper_wick_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_upper_wick_ratio": 0.29,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "hist_12_15_long_rev_t2_range10_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "min_range10_atr": 4.0,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "hist_06_09_long_rev_t2_size3_vol090_defensive_067",
                        "enabled": False,
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_vol1_rel20": 0.90,
                        "size_multiplier": 0.67,
                        # With floor sizing, this trims 3 -> 2 and leaves 2 -> 2.
                        "min_contracts": 2,
                    },
                    {
                        "name": "hist_06_09_long_rev_t2_size3_vol095_defensive_067",
                        "enabled": False,
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_vol1_rel20": 0.95,
                        "size_multiplier": 0.67,
                        "min_contracts": 2,
                    },
                    {
                        "name": "hist_06_09_long_rev_t2_size3_vol090_upper022_defensive_067",
                        "enabled": False,
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_vol1_rel20": 0.90,
                        "min_upper_wick_ratio": 0.22,
                        "size_multiplier": 0.67,
                        "min_contracts": 2,
                    },
                    {
                        "name": "hist_09_12_long_mom_t3_body028_upper035_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "max_body1_ratio": 0.28,
                        "min_upper_wick_ratio": 0.35,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "hist_12_15_long_rev_t2_vol125_close070_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "min_vol1_rel20": 1.25,
                        "min_close_pos1": 0.70,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_12_15_long_rev_t2_close030_vol090_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_close_pos1": 0.30,
                        "max_vol1_rel20": 0.90,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_12_15_long_rev_t2_close040_lower010_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_close_pos1": 0.40,
                        "max_lower_wick_ratio": 0.10,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_09_12_long_mom_t3_close040_dist050_defensive_067",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_close_pos1": 0.40,
                        "min_dist_low5_atr": 0.50,
                        "size_multiplier": 0.67,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_09_12_long_mom_t3_close040_dist050_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_close_pos1": 0.40,
                        "min_dist_low5_atr": 0.50,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_09_12_long_mom_t3_range10_le_400_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "max_range10_atr": 4.0,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_09_12_long_mom_t3_range10_le_400_defensive_067_min2",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "max_range10_atr": 4.0,
                        "size_multiplier": 0.67,
                        # Preserve 1-lot and 2-lot paths; only trim 3 -> 2.
                        "min_contracts": 2,
                    },
                    {
                        "name": "current_09_12_long_mom_t3_range10_le_350_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "max_range10_atr": 3.5,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_09_12_long_mom_t3_body050_range10_le_300_defensive_067_min2",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_body1_ratio": 0.50,
                        "max_range10_atr": 3.0,
                        "size_multiplier": 0.67,
                        # Trim 3 -> 2 while leaving 1- and 2-lot paths intact.
                        "min_contracts": 2,
                    },
                    {
                        "name": "current_09_12_long_mom_t3_body025_vol120_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_body1_ratio": 0.25,
                        "max_vol1_rel20": 1.20,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_18_21_long_rev_t2_flips0_defensive_050",
                        "enabled": False,
                        "apply_variants": ["5min_18-21_Long_Rev_T2_SL10_TP25"],
                        "max_flips5": 0.0,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_12_15_long_rev_t2_flips1_upper020_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_flips5": 1.0,
                        "max_upper_wick_ratio": 0.20,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_18_21_long_rev_t2_distlow020_disthigh400_defensive_050",
                        "enabled": False,
                        "apply_variants": ["5min_18-21_Long_Rev_T2_SL10_TP25"],
                        "max_dist_low5_atr": 0.20,
                        "max_dist_high5_atr": 4.0,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_12_15_long_rev_t2_close085_vol080_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_close_pos1": 0.85,
                        "max_vol1_rel20": 0.80,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_18_21_long_rev_t2_distlow020_range565_defensive_050",
                        "enabled": False,
                        "apply_variants": ["5min_18-21_Long_Rev_T2_SL10_TP25"],
                        "max_dist_low5_atr": 0.20,
                        "max_range10_atr": 5.645,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_21_24_long_rev_t2_distlow020_defensive_050",
                        "enabled": False,
                        # Exact live-style bearish-tape repair candidate:
                        # the active 21-24 Long_Rev T2 slice weakened materially
                        # in the current-pool trade book once the trigger printed
                        # very near the recent low, which matches the recent live
                        # filterless failures during downside pressure.
                        "apply_variants": ["5min_21-24_Long_Rev_T2_SL10_TP12.5"],
                        "max_dist_low5_atr": 0.20,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_21_24_long_rev_t2_distlow020_vol090_defensive_050",
                        "enabled": False,
                        # Narrower live-style bearish-tape repair candidate:
                        # same near-recent-low 21-24 Long_Rev T2 pocket, but only
                        # when the setup bar also comes on elevated relative volume.
                        "apply_variants": ["5min_21-24_Long_Rev_T2_SL10_TP12.5"],
                        "max_dist_low5_atr": 0.20,
                        "min_vol1_rel20": 0.90,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_21_24_long_rev_t2_close020_defensive_050",
                        "enabled": False,
                        # Chosen-shape research shows this exact 21-24 Long_Rev T2
                        # variant degrades sharply in recent history when the bar
                        # closes in the lower fifth of its range, even without an
                        # extra trend-day label.
                        "apply_variants": ["5min_21-24_Long_Rev_T2_SL10_TP12.5"],
                        "max_close_pos1": 0.20,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_21_24_long_rev_t2_close015_dist020_vol120_defensive_050",
                        "enabled": False,
                        # Extra narrow live-style repair candidate matching the
                        # exact weak shape seen in the recent live offender:
                        # close near the bar low, still near recent lows, with
                        # elevated relative volume.
                        "apply_variants": ["5min_21-24_Long_Rev_T2_SL10_TP12.5"],
                        "max_close_pos1": 0.15,
                        "max_dist_low5_atr": 0.20,
                        "min_vol1_rel20": 1.20,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "current_12_15_long_rev_t2_distlow050_range400_defensive_050",
                        "enabled": False,
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_dist_low5_atr": 0.50,
                        "min_range10_atr": 4.0,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_03_06_long_rev_t3_body050_upper020_defensive_067",
                        "enabled": False,
                        "apply_variants": ["5min_03-06_Long_Rev_T3_SL10_TP25"],
                        "min_body1_ratio": 0.50,
                        "min_upper_wick_ratio": 0.20,
                        "size_multiplier": 0.67,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_03_06_long_rev_t3_lower015_defensive_050",
                        "enabled": False,
                        "apply_variants": ["5min_03-06_Long_Rev_T3_SL10_TP25"],
                        "min_lower_wick_ratio": 0.15,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_03_06_long_rev_t3_lower015_defensive_067",
                        "enabled": False,
                        "apply_variants": ["5min_03-06_Long_Rev_T3_SL10_TP25"],
                        "min_lower_wick_ratio": 0.15,
                        "size_multiplier": 0.67,
                        "min_contracts": 1,
                    },
                    {
                        "name": "exact_03_06_long_rev_t3_lower015_range35_defensive_050",
                        "enabled": False,
                        "apply_variants": ["5min_03-06_Long_Rev_T3_SL10_TP25"],
                        "min_lower_wick_ratio": 0.15,
                        "min_range10_atr": 3.50,
                        "size_multiplier": 0.50,
                        "min_contracts": 1,
                    },
                ],
            },
            # Backtest-only DE3 policy overlay:
            # keep DE3 trade selection unchanged, but apply the learned policy
            # risk multiplier only when the chosen signal is weak on current-bar
            # quality metrics. This isolates the model's caution to marginal
            # trades instead of bluntly shrinking strong recent signals.
            "backtest_policy_context_overlay": {
                "enabled": False,
                "reduce_only": True,
                "min_contracts": 1,
                "min_policy_confidence": None,
                "min_policy_bucket_samples": 120,
                "require_signal_weakness": True,
                "max_execution_quality_score": 0.60,
                "max_entry_model_margin": 0.10,
                "max_route_confidence": None,
                "max_edge_points": None,
            },
            # DE3 v4 backtest-only trade management aligned with live bracket handling:
            # ratchet stops conservatively (next bar) once profit reaches a TP fraction,
            # and cut persistently non-working trades without enabling the legacy global path.
            "trade_management": {
                "enabled": True,
                "break_even": {
                    "enabled": True,
                    "trigger_pct": 0.40,
                    "buffer_ticks": 1,
                    "trail_pct": 0.25,
                    # Optional tighter trail once break-even has actually moved the stop.
                    "post_activation_trail_pct": 0.25,
                    # Optional extra-tight trail after a partial TP fill in research mode.
                    "post_partial_trail_pct": 0.25,
                    # Use next-bar activation to avoid assuming an intrabar high-before-low sequence.
                    "activate_on_next_bar": True,
                },
                "tiered_take_profit": {
                    # Research-only by default: trim part of the position at a first target
                    # and leave the rest working toward the original bracket.
                    "enabled": False,
                    "trigger_pct": 0.50,
                    "close_fraction": 0.50,
                    "min_entry_contracts": 2,
                    "min_remaining_contracts": 1,
                    # After the trim, stage at least a break-even style stop update.
                    "arm_break_even_after_fill": True,
                },
                "early_exit": {
                    # Best validated DE3 v4 iteration is break-even only.
                    "enabled": False,
                    "exit_if_not_green_by": 30,
                    "max_profit_crosses": 4,
                },
                "profit_milestone_stop": {
                    # Validated 2026-04-13 on es_master_outrights.parquet:
                    # promote the same late-profit ratchet to every active DE3
                    # bracket size, but keep the trigger / trail shape aware of
                    # the bracket width instead of using one blunt profile.
                    "enabled": True,
                    "profiles": [
                        {
                            "name": "validated_tp25_lock_75_60",
                            "enabled": True,
                            "apply_tp_dists": [25.0],
                            "trigger_pct": 0.75,
                            "post_reach_trail_pct": 0.60,
                            "force_break_even_on_reach": False,
                        },
                        {
                            "name": "validated_tp20_lock_75_60",
                            "enabled": True,
                            "apply_tp_dists": [20.0],
                            "trigger_pct": 0.75,
                            "post_reach_trail_pct": 0.60,
                            "force_break_even_on_reach": False,
                        },
                        {
                            "name": "validated_tp12p5_lock_80_65",
                            "enabled": True,
                            "apply_tp_dists": [12.5],
                            "trigger_pct": 0.80,
                            "post_reach_trail_pct": 0.65,
                            "force_break_even_on_reach": False,
                        },
                        {
                            "name": "validated_tp10_and_under_lock_85_70",
                            "enabled": True,
                            "max_tp_dist": 10.0,
                            "trigger_pct": 0.85,
                            "post_reach_trail_pct": 0.70,
                            "force_break_even_on_reach": False,
                        },
                    ],
                },
                "entry_trade_day_extreme_size_adjustment": {
                    # Validated 2026-04-11 as a layered follow-up to the TP25
                    # milestone rule:
                    # a few beyond-trade-day-extreme pockets still fail before
                    # the later stop-management logic can help. Trim those
                    # exact entry contexts instead of blocking them outright.
                    "enabled": True,
                    # Use the same futures trade day as the extreme-stop logic.
                    "trade_day_roll_hour_et": 18,
                    "profiles": [
                        {
                            "name": "beyond_entry_day_extreme_short_mom_t5_defensive_050",
                            "enabled": True,
                            # Full-range + recent-sample review both said this
                            # short-momentum pocket behaves better with a
                            # straight risk haircut when the original TP is
                            # already beyond the known trade-day low.
                            "apply_variants": [
                                "15min_15-18_Short_Mom_T5_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "size_multiplier": 0.50,
                            "min_contracts": 1,
                        },
                        {
                            "name": "beyond_entry_day_extreme_late_long_rev_t2_defensive_067",
                            "enabled": True,
                            # Keep the late-session Long_Rev T2 lane alive, but
                            # shave it when the take-profit is already reaching
                            # beyond the entry trade-day high.
                            "apply_variants": [
                                "5min_21-24_Long_Rev_T2_SL10_TP12.5",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "size_multiplier": 0.67,
                            "min_contracts": 1,
                        },
                    ],
                },
                "entry_trade_day_extreme_early_exit": {
                    # Research hook only.
                    # 2026-04-11 exact outrights revalidation note:
                    # the narrow bundle looked promising in the per-group study,
                    # but a fresh DE3-only exact holdout replay on
                    # es_master_outrights.parquet was worse than leaving the
                    # current stack alone, so keep this disabled until a future
                    # candidate survives exact validation.
                    "enabled": False,
                    "trade_day_roll_hour_et": 18,
                    "profiles": [
                        {
                            "name": "beyond_entry_day_extreme_lunch_long_mom_t3_not_green_10",
                            "enabled": True,
                            "apply_variants": [
                                "15min_12-15_Long_Mom_T3_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "min_progress_by_bars": 10,
                            "min_progress_pct": 0.0,
                        },
                        {
                            "name": "beyond_entry_day_extreme_morning_long_mom_t3_progress_30_005",
                            "enabled": False,
                            "apply_variants": [
                                "15min_09-12_Long_Mom_T3_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "min_progress_by_bars": 30,
                            "min_progress_pct": 0.05,
                        },
                        {
                            "name": "beyond_entry_day_extreme_open_long_rev_t3_crosses_1",
                            "enabled": True,
                            "apply_variants": [
                                "5min_03-06_Long_Rev_T3_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "max_profit_crosses": 1,
                        },
                        {
                            "name": "beyond_entry_day_extreme_lunch_long_rev_t2_progress_90_005",
                            "enabled": True,
                            "apply_variants": [
                                "15min_12-15_Long_Rev_T2_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "min_progress_by_bars": 90,
                            "min_progress_pct": 0.05,
                        },
                        {
                            "name": "beyond_entry_day_extreme_late_long_rev_t2_not_green_60",
                            "enabled": False,
                            "apply_variants": [
                                "5min_21-24_Long_Rev_T2_SL10_TP12.5",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "min_progress_by_bars": 60,
                            "min_progress_pct": 0.0,
                        },
                        {
                            "name": "beyond_entry_day_extreme_evening_long_rev_t2_not_green_90",
                            "enabled": False,
                            "apply_variants": [
                                "5min_18-21_Long_Rev_T2_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                            "min_progress_by_bars": 90,
                            "min_progress_pct": 0.0,
                        },
                    ],
                },
                "entry_trade_day_extreme_admission_block": {
                    # Research hook only.
                    # 2026-04-11 validation note:
                    # a narrow Long_Rev_T3 beyond-extreme entry block improved
                    # the January 2026 holdout, but failed the full-range
                    # robustness + Monte Carlo check, so keep this disabled for
                    # live/backtest runtime until a stronger candidate shows up.
                    "enabled": False,
                    # Same futures trade day used by the stop-management path.
                    "trade_day_roll_hour_et": 18,
                    "profiles": [
                        {
                            "name": "block_beyond_entry_day_extreme_long_rev_t3",
                            "enabled": True,
                            # Audit 2026-04-11:
                            # this variant stayed weak enough that the
                            # beyond-entry-trade-day-extreme subset is worth
                            # cutting, but the broader full-variant block
                            # proved too aggressive on the January 2026
                            # holdout.
                            "apply_variants": [
                                "5min_03-06_Long_Rev_T3_SL10_TP25",
                            ],
                            "require_target_beyond_trade_day_extreme": True,
                        },
                    ],
                },
                "entry_trade_day_extreme_stop": {
                    "enabled": True,
                    # DE3 uses the futures trade day, not midnight ET.
                    "trade_day_roll_hour_et": 18,
                    "profiles": [
                        {
                            "name": "weak_reversion_lock_after_entry_day_extreme",
                            "enabled": True,
                            # Only touch the weak reversion variants that
                            # repeatedly ran beyond the current trade-day
                            # extreme without reliably finishing the full TP.
                            "apply_variants": [
                                "5min_03-06_Long_Rev_T3_SL10_TP25",
                                "5min_09-12_Long_Rev_T5_SL10_TP25",
                                "5min_09-12_Short_Rev_T6_SL10_TP12.5",
                            ],
                            # Restrict the rule to the exact complaint:
                            # targets that are already beyond the known
                            # trade-day high/low at entry.
                            "require_target_beyond_trade_day_extreme": True,
                            # Once price actually tags that entry-time
                            # trade-day extreme, force the break-even engine
                            # on and tighten the trail materially.
                            "force_break_even_on_reach": True,
                            "post_reach_trail_pct": 0.50,
                        },
                    ],
                },
            },
            # Backtest-only DE3v4 manifold adaptation:
            # calibrated manifold scans on the current 2024-2025 DE3v4 trade book
            # showed that ROTATIONAL_TURBULENCE was the only natural manifold rule
            # that improved both years with minimal trade removal.
            # Keep disabled until exact multi-year backtest reruns finish.
            "manifold_adaptation": {
                "enabled": False,
                "mode": "block",  # shadow | block
                "log_decisions": False,
                "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
                "block_no_trade": False,
                "require_allow_style": False,
            },
            "prune_rules": {
                "enabled": True,
                "log_blocks": False,
                "rules": [
                    {
                        "name": "final_score_dead_zone",
                        "enabled": False,
                        # Cross-year weak pocket: this absolute score band was
                        # negative in both 2024 and 2025 when sampled from the
                        # live DE3v4 selected-trade book.
                        "min_final_score": 2.60,
                        "max_final_score": 2.70,
                    },
                    {
                        "name": "final_score_dead_zone_06_09_long_rev_260_265",
                        "enabled": True,
                        # Validated robust keep-on rule for break-even-only DE3v4:
                        # 2022 = no effect, 2023 = modest improvement, 2024-2025 = strong improvement.
                        # Keep this as the default prune until any replacement survives the same check.
                        "apply_sessions": ["06-09"],
                        "apply_lanes": ["Long_Rev"],
                        "min_final_score": 2.60,
                        "max_final_score": 2.65,
                    },
                    {
                        "name": "final_score_dead_zone_09_12_long_mom_260_265",
                        "enabled": False,
                        "apply_sessions": ["09-12"],
                        "apply_lanes": ["Long_Mom"],
                        "min_final_score": 2.60,
                        "max_final_score": 2.65,
                    },
                    {
                        "name": "final_score_dead_zone_09_12_long_mom_225_230",
                        "enabled": False,
                        "apply_sessions": ["09-12"],
                        "apply_lanes": ["Long_Mom"],
                        "min_final_score": 2.25,
                        "max_final_score": 2.30,
                    },
                    {
                        "name": "final_score_dead_zone_15_18_short_mom_t6_125_150",
                        "enabled": False,
                        "apply_sessions": ["15-18"],
                        "apply_lanes": ["Short_Mom"],
                        "apply_variants": ["15min_15-18_Short_Mom_T6_SL10_TP25"],
                        "min_final_score": 1.25,
                        "max_final_score": 1.50,
                    },
                    {
                        "name": "low_conf_high_upper_wick",
                        "enabled": False,
                        # Block low-confidence setups that follow a strong upper wick.
                        "max_route_confidence": 0.055,
                        "min_upper_wick_ratio": 0.343,
                        "apply_sides": ["LONG"],
                        "apply_lanes": ["Long_Rev", "Long_Mom"],
                    },
                    {
                        "name": "low_conf_upper_wick_medium",
                        "enabled": False,
                        # Cross-run robust candidate: prune medium/high upper wick in
                        # low-confidence LONG routes.
                        "max_route_confidence": 0.05,
                        "min_upper_wick_ratio": 0.33,
                        "apply_sides": ["LONG"],
                        "apply_lanes": ["Long_Rev", "Long_Mom"],
                    },
                    {
                        "name": "low_conf_upper_wick_soft",
                        "enabled": False,
                        # Broader robust candidate from recent scans.
                        "max_route_confidence": 0.05,
                        "min_upper_wick_ratio": 0.30,
                        "apply_sides": ["LONG"],
                        "apply_lanes": ["Long_Rev", "Long_Mom"],
                    },
                    {
                        "name": "live_12_15_long_rev_t2_body040_upper030",
                        "enabled": False,
                        # Live filterless March 2026 audit:
                        # 12-15 Long_Rev T2 degraded most on weak-body bars that
                        # also left a meaningful upper wick, matching the
                        # 2024/2025/2026-Jan current-version trade CSV weakness.
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_body1_ratio": 0.40,
                        "min_upper_wick_ratio": 0.30,
                    },
                    {
                        "name": "live_12_15_long_rev_t2_body040_down3_2",
                        "enabled": False,
                        # Alternate live-derived candidate:
                        # weak-body midday Long_Rev entries after multi-bar
                        # downside pressure were another recurring losing slice.
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_body1_ratio": 0.40,
                        "min_down3": 2,
                    },
                    {
                        "name": "live_06_09_long_rev_t2_dist050_range350",
                        "enabled": False,
                        # Live filterless March 2026 audit:
                        # early Long_Rev T2 deteriorated when the trigger bar was
                        # already extended above the recent low while the 10-bar
                        # ATR-normalized range was broad, a slice that also
                        # replayed poorly in the 2024/2025/2026-Jan trade book.
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_dist_low5_atr": 0.50,
                        "min_range10_atr": 3.50,
                    },
                    {
                        "name": "exact_12_15_long_rev_t2_close030_vol090",
                        "enabled": False,
                        # Exact live-style baseline scan:
                        # midday Long_Rev T2 degraded most when the trigger bar
                        # failed to close strongly while also coming on soft volume.
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_close_pos1": 0.30,
                        "max_vol1_rel20": 0.90,
                    },
                    {
                        "name": "exact_09_12_long_mom_t3_close040_dist050",
                        "enabled": False,
                        # Exact live-style baseline scan:
                        # 09-12 Long_Mom T3 was most loss-heavy when it chased
                        # extension already far off the recent low while still
                        # closing in the upper portion of the bar.
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "min_close_pos1": 0.40,
                        "min_dist_low5_atr": 0.50,
                    },
                    {
                        "name": "exact_03_06_long_rev_t3_body050_upper020",
                        "enabled": False,
                        # Exact live-style baseline scan:
                        # 03-06 Long_Rev T3 weakened when a large-body trigger bar
                        # still left a notable upper wick, indicating rejection.
                        "apply_variants": ["5min_03-06_Long_Rev_T3_SL10_TP25"],
                        "min_body1_ratio": 0.50,
                        "min_upper_wick_ratio": 0.20,
                    },
                    {
                        "name": "pf_probe_12_15_long_rev_t2_body040_cap",
                        "enabled": False,
                        # PF-focused probe:
                        # the research winner still leaks most of its midday
                        # Long_Rev T2 losses through weak-body entries. This
                        # rule isolates that pocket without changing the rest
                        # of the variant's shape.
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "max_body1_ratio": 0.40,
                    },
                    {
                        "name": "pf_probe_03_06_long_rev_t3_body080_range335",
                        "enabled": False,
                        # PF-focused probe:
                        # the remaining 03-06 Long_Rev T3 losses cluster in
                        # weak-body entries that still arrive during already
                        # expanded local range. This is meant to catch that
                        # live-style bearish-pressure pocket without removing
                        # the whole variant.
                        "apply_variants": ["5min_03-06_Long_Rev_T3_SL10_TP25"],
                        "max_body1_ratio": 0.80,
                        "min_range10_atr": 3.35,
                    },
                    {
                        "name": "hist_06_09_long_rev_t2_upper_wick_floor",
                        "enabled": False,
                        "backtest_only": True,
                        # Historical-vs-modern robustness experiment only.
                        # Keep disabled by default: the broader six-rule pack improved weak
                        # historical pockets, but failed isolated 2024 and 2025 validation.
                        # 06-09 Long_Rev T2 improved materially when the setup bar
                        # already showed a meaningful upper wick.
                        "apply_variants": ["15min_06-09_Long_Rev_T2_SL10_TP25"],
                        "min_upper_wick_ratio": 0.29,
                    },
                    {
                        "name": "hist_09_12_long_mom_t3_upper_wick_cap",
                        "enabled": False,
                        "backtest_only": True,
                        # Avoid the noisiest extension bars in the still-active
                        # 09-12 Long_Mom T3 profile.
                        "apply_variants": ["15min_09-12_Long_Mom_T3_SL10_TP25"],
                        "max_upper_wick_ratio": 0.57,
                    },
                    {
                        "name": "hist_12_15_long_rev_t2_range10_cap",
                        "enabled": False,
                        "backtest_only": True,
                        # Midday Long_Rev T2 was historically weakest on expanded
                        # 10-bar ATR-normalized range.
                        "apply_variants": ["15min_12-15_Long_Rev_T2_SL10_TP25"],
                        "min_range10_atr": 4.0,
                    },
                    {
                        "name": "hist_15_18_long_mom_t6_final_score_floor",
                        "enabled": False,
                        "backtest_only": True,
                        # Late-session Long_Mom T6 strengthened once final score
                        # cleared a modest floor.
                        "apply_variants": ["15min_15-18_Long_Mom_T6_SL10_TP25"],
                        "min_final_score": 2.43,
                    },
                    {
                        "name": "hist_09_12_short_rev_t6_upper_wick_floor",
                        "enabled": False,
                        "backtest_only": True,
                        # Short_Rev T6 improves when the trigger bar already shows
                        # a clear rejection wick instead of a flat-top rotation.
                        "apply_variants": ["5min_09-12_Short_Rev_T6_SL10_TP12.5"],
                        "min_upper_wick_ratio": 0.26,
                    },
                    {
                        "name": "hist_18_21_long_rev_t2_upper_wick_cap",
                        "enabled": False,
                        "backtest_only": True,
                        # 18-21 Long_Rev T2 degraded on extreme upper-wick setups.
                        "apply_variants": ["5min_18-21_Long_Rev_T2_SL10_TP25"],
                        "max_upper_wick_ratio": 0.60,
                    },
                    {
                        "name": "robust_long_rev_upper_wick_cap_080",
                        "enabled": False,
                        "backtest_only": True,
                        # Cross-year trade-attribution candidate:
                        # Long_Rev entries with an already-extended upper wick
                        # were persistently weak in 2022, 2023, 2024, and 2025.
                        "apply_lanes": ["Long_Rev"],
                        "apply_sides": ["LONG"],
                        "min_upper_wick_ratio": 0.80,
                    },
                    {
                        "name": "robust_5min_short_rev_low_close_pos_cap_020",
                        "enabled": False,
                        "backtest_only": True,
                        # Cross-year trade-attribution candidate:
                        # 5min Short_Rev entries closing too near the local low
                        # were consistently weak, especially in the older years.
                        "apply_timeframes": ["5min"],
                        "apply_lanes": ["Short_Rev"],
                        "apply_sides": ["SHORT"],
                        "max_close_pos1": 0.20,
                    },
                    {
                        "name": "robust_long_thu_15_18",
                        "enabled": False,
                        "backtest_only": True,
                        # Cross-year calendar-context candidate from exact-trade
                        # attribution: LONG trades in the 15-18 bucket on
                        # Thursdays were negative in 2022, 2023, 2024, and 2025.
                        "match_scope_only": True,
                        "apply_sides": ["LONG"],
                        "apply_weekdays": ["THU"],
                        "apply_hour_buckets": ["15-18"],
                    },
                    {
                        "name": "robust_mon_w1",
                        "enabled": False,
                        "backtest_only": True,
                        # First Monday-of-month context was negative across the
                        # 2022-2025 DE3v4 break-even trade book.
                        "match_scope_only": True,
                        "apply_weekdays": ["MON"],
                        "apply_weeks_of_month": ["W1"],
                    },
                    {
                        "name": "robust_long_q3_15_18",
                        "enabled": False,
                        "backtest_only": True,
                        # Late-session LONG trades in Q3 were persistently weak
                        # across the same cross-year point-level scan.
                        "match_scope_only": True,
                        "apply_sides": ["LONG"],
                        "apply_quarters": ["Q3"],
                        "apply_hour_buckets": ["15-18"],
                    },
                    {
                        "name": "robust_q4_w3_12_15",
                        "enabled": False,
                        "backtest_only": True,
                        # Midday trades in the third week of Q4 repeatedly
                        # underperformed across 2022-2025.
                        "match_scope_only": True,
                        "apply_quarters": ["Q4"],
                        "apply_weeks_of_month": ["W3"],
                        "apply_hour_buckets": ["12-15"],
                    },
                    {
                        "name": "robust_mon_w5",
                        "enabled": False,
                        "backtest_only": True,
                        # Fifth-Monday trades were a smaller but still
                        # consistently negative pocket across the scan years.
                        "match_scope_only": True,
                        "apply_weekdays": ["MON"],
                        "apply_weeks_of_month": ["W5"],
                    },
                    {
                        "name": "robust_nyam_q1_w4",
                        "enabled": False,
                        "backtest_only": True,
                        # NY AM in Q1 week 4 was negative in every scan year and
                        # remained negative after removing the broader Monday/Thursday pockets.
                        "match_scope_only": True,
                        "apply_sessions": ["NY_AM"],
                        "apply_quarters": ["Q1"],
                        "apply_weeks_of_month": ["W4"],
                    },
                    {
                        "name": "research_18_21_bearimpulse_t3_block_high_range362",
                        "enabled": False,
                        # Balance-forward Asia short candidate:
                        # this new variant only held edge when the local 10-bar
                        # ATR-normalized range stayed contained. Expanded range
                        # entries repeatedly degraded both the variant itself
                        # and the broader filterless stack.
                        "apply_variants": ["15min_18-21_Short_Mom_BearImpulse_T3_SL6_TP6"],
                        "min_range10_atr": 3.62,
                    },
                    {
                        "name": "research_18_21_bearimpulse_t3_block_high_range408",
                        "enabled": False,
                        # Softer version of the same containment rule.
                        "apply_variants": ["15min_18-21_Short_Mom_BearImpulse_T3_SL6_TP6"],
                        "min_range10_atr": 4.08,
                    },
                    {
                        "name": "research_18_21_bearimpulse_t3_block_low_close050",
                        "enabled": False,
                        # Short continuation entries that already close too near
                        # the bar low look more like overextended chase entries
                        # than fresh downside continuation.
                        "apply_variants": ["15min_18-21_Short_Mom_BearImpulse_T3_SL6_TP6"],
                        "max_close_pos1": 0.50,
                    },
                    {
                        "name": "research_18_21_bearimpulse_t3_block_low_close030",
                        "enabled": True,
                        # Stricter anti-chase guard for the same family:
                        # entries that already close very near the bar low
                        # stayed negative across most years even after the
                        # broader high-range filter was applied.
                        "apply_variants": ["15min_18-21_Short_Mom_BearImpulse_T3_SL6_TP6"],
                        "max_close_pos1": 0.30,
                    },
                    {
                        "name": "research_18_21_bearimpulse_t3_block_chop_flips1",
                        "enabled": False,
                        # The same bearish family weakened materially once local
                        # flip count rose above the cleaner trend states.
                        "apply_variants": ["15min_18-21_Short_Mom_BearImpulse_T3_SL6_TP6"],
                        "min_flips5": 1.0,
                    },
                    {
                        "name": "research_18_21_bearimpulse_t3_block_high_range362_low_close050",
                        "enabled": True,
                        # Combined containment + anti-chase guard for the new
                        # Asia short family. Narrower than blocking either
                        # condition on its own.
                        "apply_variants": ["15min_18-21_Short_Mom_BearImpulse_T3_SL6_TP6"],
                        "min_range10_atr": 3.62,
                        "max_close_pos1": 0.50,
                    },
                ],
            },
            "bracket_module": {
                # The legacy multiplier-based adaptive modes slightly reduced
                # PF versus canonical/context-selected brackets in the
                # 2026-04-08 bracket validation sweep.
                "enable_adaptive_modes": False,
                "enable_family_bracket_selector": True,
                "min_support_for_adaptive_modes": 80,
                "min_support_for_conservative_modes": 30,
                "min_support_for_aggressive_modes": 80,
                "aggressive_requires_expanding": True,
                "allow_aggressive_in_high_vol": False,
                # Preserve DE3v4-selected bracket through downstream execution filters;
                # filters can still veto or adjust size, but bracket edits are explicit.
                "preserve_selected_bracket_through_fixed_sltp": True,
                "preserve_selected_bracket_through_vol_guard": True,
            },
        },
        # DE3v4 build-time settings for router/lane/bracket modules.
        "training": {
            "router": {
                "lane_prior_top_k": 5,
                "no_trade_bias": 0.15,
                "min_route_confidence": 0.05,
                "min_lane_score_to_trade": -0.10,
                "min_score_margin_to_trade": 0.0,
                "min_single_lane_internal_margin": 0.0,
                "single_lane_confidence_mode": "fixed",
                "single_lane_internal_margin_scale": 0.05,
                "single_lane_confidence": 0.05,
                "route_score_weights": {
                    "lane_prior": 0.55,
                    "lane_max_edge": 0.30,
                    "lane_mean_edge": 0.15,
                },
            },
            "lane_selector": {
                "max_variants_per_lane": 12,
            },
            "bracket_module": {
                "enable_alternative_modes": True,
            },
            # Calibrated entry policy trainer for DE3v4 (no trade-management changes).
            "entry_policy": {
                "enabled": True,
                "wilson_z": 1.96,
                "reliability_full_samples": 60,
                "win_lcb_center": 0.50,
                "weight_win_lcb": 0.60,
                "weight_ev_lcb": 0.30,
                "weight_reliability": 0.10,
                "ev_lcb_scale_points": 2.0,
                "min_variant_trades": 25,
                "min_lane_trades": 120,
                "allow_on_missing_stats": True,
                "default_threshold": 0.0,
                "conservative_buffer": 0.035,
                "scope_threshold_offsets": {
                    "variant": 0.00,
                    "lane": 0.06,
                    "global": 0.12,
                    "missing": 0.15,
                    "default": 0.00,
                },
                "score_components": {
                    "weight_quality_lcb": 0.65,
                    "weight_route_confidence": 0.20,
                    "weight_edge_points": 0.10,
                    "weight_structural_score": 0.05,
                    "weight_loss_share_penalty": 0.12,
                    "weight_stop_like_share_penalty": 0.08,
                    "weight_drawdown_penalty": 0.06,
                    "weight_worst_block_penalty": 0.08,
                    "route_confidence_center": 0.05,
                    "edge_scale_points": 0.40,
                    "structural_scale": 0.80,
                    "loss_share_center": 0.52,
                    "loss_share_scale": 0.22,
                    "stop_like_share_center": 0.62,
                    "stop_like_share_scale": 0.25,
                    "drawdown_scale": 6.0,
                    "worst_block_scale_points": 3.0,
                },
                "threshold_tuning": {
                    "min_keep_trades": 80,
                    "min_keep_rate": 0.40,
                    "objective_weight_max_drawdown": 0.55,
                    "objective_weight_profit_factor": 140.0,
                    "objective_weight_keep_rate": 220.0,
                    "objective_profit_factor_cap": 3.5,
                    # Optional explicit grid override:
                    # "threshold_candidates": [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10],
                    "default_threshold": 0.0,
                },
            },
            # Direct candidate chooser for DE3v5-lite experiments. This is
            # trained from the same current-pool exports as entry_policy, but
            # is intended to select across all feasible candidates rather than
            # only vetoing the already-chosen trade.
            "decision_policy": {
                "enabled": False,
                "selection_mode": "replace_router_lane",
                "min_variant_trades": 25,
                "min_lane_trades": 120,
                "allow_on_missing_stats": True,
                "default_threshold": 0.0,
                "conservative_buffer": 0.035,
                "min_confidence_to_override": 0.0,
                "min_score_delta_to_override": 0.0,
                "min_score_margin_to_override": 0.0,
                "allow_override_when_baseline_no_trade": False,
                "min_confidence_to_override_when_baseline_no_trade": 0.0,
                "min_score_delta_to_override_when_baseline_no_trade": 0.0,
                "min_score_margin_to_override_when_baseline_no_trade": 0.0,
                "min_baseline_score_advantage_to_override": 0.0,
                "min_baseline_score_delta_advantage_to_override": 0.0,
                "scope_threshold_offsets": {
                    "variant": 0.00,
                    "lane": 0.05,
                    "global": 0.10,
                    "missing": 0.15,
                    "default": 0.00,
                },
                "score_components": {
                    "weight_quality_lcb": 0.56,
                    "weight_lane_prior": 0.10,
                    "weight_variant_quality_prior": 0.10,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.06,
                    "weight_profit_factor_component": 0.10,
                    "weight_year_coverage_component": 0.06,
                    "weight_loss_share_penalty": 0.16,
                    "weight_stop_like_share_penalty": 0.10,
                    "weight_drawdown_penalty": 0.10,
                    "weight_worst_block_penalty": 0.12,
                    "weight_shape_penalty_component": 0.18,
                    "lane_prior_center": 0.15,
                    "lane_prior_scale": 0.08,
                    "variant_quality_prior_center": 0.27,
                    "variant_quality_prior_scale": 0.12,
                    "edge_scale_points": 0.40,
                    "structural_scale": 0.80,
                    "profit_factor_center": 1.10,
                    "profit_factor_scale": 0.35,
                    "year_coverage_full_years": 8.0,
                    "loss_share_center": 0.52,
                    "loss_share_scale": 0.22,
                    "stop_like_share_center": 0.62,
                    "stop_like_share_scale": 0.25,
                    "drawdown_scale": 6.0,
                    "shape_penalty_scale": 1.0,
                    "shape_penalty_cap": 2.0,
                    "worst_block_scale_points": 3.0,
                },
                "threshold_tuning": {
                    "min_keep_trades": 80,
                    "min_keep_rate": 0.35,
                    "objective_weight_max_drawdown": 0.60,
                    "objective_weight_profit_factor": 165.0,
                    "objective_weight_keep_rate": 180.0,
                    "objective_profit_factor_cap": 3.5,
                    "default_threshold": 0.0,
                },
                # Reuse entry-policy shape learning rather than requiring a
                # separate direct-chooser shape config block in candidates.
                "shape_penalty_model": {
                    "enabled": False,
                },
                "wilson_z": 1.96,
                "reliability_full_samples": 60,
                "win_lcb_center": 0.50,
                "weight_win_lcb": 0.60,
                "weight_ev_lcb": 0.30,
                "weight_reliability": 0.10,
                "ev_lcb_scale_points": 2.0,
                "worst_block_trade_block_size": 40,
            },
            # Shared satellite scoring controls used by lane trainer.
            "satellites": {
                "min_standalone_viability": 0.20,
                "min_incremental_value_over_core": 0.05,
                "min_standalone_viability_fallback": 0.20,
                "min_incremental_value_over_core_fallback": 0.00,
                "max_retained_satellites": 8,
                "require_orthogonality": True,
                "max_overlap_with_core": 0.85,
                "max_bad_overlap_with_core": 0.60,
                "allow_near_core_variants_if_incremental": True,
            },
        },
    },
    # DynamicEngine3 v2 hard entry-bar blocks (1m bar-shape).
    # These are intentionally hard blocks for clear negative-EV bar structures.
    "DE3_V2_ENTRY_BAR_HARD_BLOCKS": {
        "enabled": False,
        "log_decisions": False,
        "apply_sides": ["LONG"],
        "apply_timeframes": ["5min", "15min"],
        "min_bars": 2,
        "rules": {
            # Data-driven from best-run bar neighborhood analysis:
            # keep only the strongest trap rule active.
            "body_non_positive": {
                "enabled": False,
                "body_pos_max": 0.0,
            },
            "body_strong_bear": {
                "enabled": False,
                "body_pos_max": -0.40,
            },
            "close_near_low": {
                "enabled": False,
                "close_pos_max": 0.20,
            },
            "bear_no_lower_wick": {
                "enabled": False,
                "body_pos_max": -0.30,
                "lower_wick_max": 0.10,
            },
            "long_lower_wick": {
                "enabled": True,
                "lower_wick_min": 0.55,
            },
        },
    },
    # DynamicEngine3 runtime reliability controls.
    # Uses trainer outputs (Score/Trades/Avg_PnL) for ranking and allows abstaining on ambiguous bars.
    "DYNAMIC_ENGINE3_RUNTIME": {
        "enabled": True,
        # When enabled, consume training constraints from the DE3 JSON settings block
        # (e.g., min_trades/min_win_rate/min_avg_pnl and recent_mode via per-row Recent stats).
        "use_db_settings": True,
        # Apply DB recent gating semantics (intersect/union/recent_only) when available.
        "db_recent_gate": True,
        # Hard minimums before a DB strategy can be considered at runtime.
        "min_trades": 40,
        "min_score": 0.0,
        "min_avg_pnl": 0.0,
        # Normalization caps for robust ranking.
        "score_cap": 3.0,
        "avg_pnl_cap": 3.0,
        "trades_cap": 1000,
        "win_rate_floor": 0.45,
        "win_rate_ceil": 0.75,
        "bucket_score_cap": 2.0,
        "weights": {
            "score": 0.45,
            "win_rate": 0.25,
            "avg_pnl": 0.20,
            "trades": 0.10,
            "bucket": 0.15,
            "location": 0.05,
        },
        "abstain": {
            "enabled": True,
            "min_best_score": 4.0,
            "min_side_edge": 0.30,
            "min_top2_gap_opposite": 0.20,
        },
    },
    # DE3 v2 runtime constraints (low-DOF risk controls).
    # Applies only when DE3 runtime DB version is v2.
    "DE3_V2_RUNTIME_CONSTRAINTS": {
        "enabled": False,
        # Require enough historical support for a candidate profile.
        "min_trades": 40,
        # Cap threshold aggressiveness to reduce weak high-thresh entries.
        "max_thresh": 6.0,
        # Conservative quality floor from latest 2024 OOS diagnostics.
        "min_final_score": 5.6,
        "log_decisions": False,
    },
    # DynamicEngine3 candidate selection:
    # 1) Apply feasibility gates first (policy/veto/SLTP/fees/quality filters),
    # 2) then select the feasible candidate with highest conservative edge score.
    "DE3_CANDIDATE_SELECTION": {
        "enabled": True,
        # Prefer adaptive-policy EV LCB when available; fallback to TP/SL EV proxy.
        "prefer_policy_ev_lcb": True,
        # When False, adaptive policy can still shape size/risk, but candidate
        # ranking stays on the native DE3 edge proxy instead of the policy EV.
        "use_policy_edge_in_ranking": True,
        # Optional no-trade floor on best feasible edge (points, after fee proxy).
        "min_edge_points": 0.06,
        # Optional no-trade if best-vs-second score gap is too small.
        "min_score_gap_points": 0.09,
        # Emit logs when re-ranking changes selected candidate.
        "log_rerank": True,
    },
    # DynamicEngine3 safety guards (backtest-derived blocks)
    "DE3_SAFETY_GUARDS": {
        "enabled": False,
        # Block mean-reversion entries that are persistent drawdown drivers in high-vol.
        "block_high_vol_reversals": True,
        "block_long_rev_timeframes": ["5min", "15min"],
        "block_short_rev_timeframes": ["5min", "15min"],
    },
    # DynamicEngine3 Long_Mom quality filters (root-cause fixes for late breakouts)
    "DE3_LONG_MOM_FILTERS": {
        "enabled": False,
        # Pre-breakout compression: last N bars range must be <= compression_atr_mult * ATR
        "compression_lookback": 4,
        "compression_atr_mult": 0.7,
        # If price is already > X*ATR beyond the prior range high, treat as late/exhausted.
        "late_breakout_atr_mult": 1.0,
        # Block specific threshold bands in high-vol when they are persistent loss drivers.
        "block_thresh_values_in_high_vol": [4.0, 15.0],
        # Shock filters: skip momentum entries after discontinuous opens / oversized bars.
        "shock_gap_atr_mult": 0.7,
        "shock_range_atr_mult": 2.0,
        # Breakout bar impulse quality
        "impulse_body_atr_min": 0.6,
        "impulse_close_pos_min": 0.8,
        "require_bull_close": True,
        # ATR period if atr_5m not available
        "atr_period": 14,
    },
    # DynamicEngine3 candle-structure entry guards (short-side):
    # N1: weak prior impulse + stretched range + elevated relative volume
    # N2: weak prior impulse + large body + weak close location
    # N3: 5m structure trap (close too near local high while printing upper wick)
    "DE3_ENTRY_CANDLE_FILTERS": {
        "enabled": False,
        "log_decisions": False,
        "apply_sides": ["SHORT"],
        "atr_period": 14,
        "flip_window": 5,
        "range_window": 10,
        "dist_window": 5,
        "min_bars": 40,
        "n1": {
            "ret1_atr_max": 0.017,
            "range10_atr_min": 2.612,
            "vol1_rel20_min": 0.014,
        },
        "n2": {
            "ret1_atr_max": 0.017,
            "body1_ratio_min": 0.55,
            "close_pos1_max": 0.35,
        },
        "n3": {
            "enabled": True,
            "timeframes": ["5min"],
            "dist_high5_atr_max": 1.53343,
            "upper1_ratio_min": 0.0125,
        },
    },
    # Long_Mom bracket override in high-vol (use ATR-based bracket)
    "DE3_LONG_MOM_BRACKETS": {
        "enabled": False,
        "atr_period": 14,
        "sl_atr": 1.2,
        "tp_atr": 0.9,
    },
    # DynamicEngine3 context-loss veto (trained on DE3 candidates)
    "DE3_CONTEXT_VETO": {
        "enabled": False,
        # mode: "block" (default) or "shadow" (log p_loss without blocking)
        "mode": "block",
        "model_path": "de3_context_veto_models.json",
        # Override model threshold if set; otherwise use threshold from JSON.
        # Lower threshold = stricter veto.
        "threshold": 0.55,
        # Uncertainty-aware veto: block only when loss-probability lower bound exceeds threshold.
        # Smaller z raises LCB and makes veto stricter.
        "uncertainty_z": 0.4,
        # If ensemble disagreement exceeds this std-dev, skip veto (insufficient confidence).
        "max_std": 0.30,
        # Require a minimum bucket sample size before enabling live veto blocks.
        "min_bucket_samples": 150,
        # Keep fallback behavior robust: block candidate and try next, rather than kill entire DE3 vote.
        "block_all_on_top_veto": True,
        # Feature params must match build_de3_context_dataset.py defaults.
        "atr_period": 20,
        "atr_median_window": 390,
        "price_location_window": 20,
        "log_decisions": True,
    },
    # DynamicEngine3 adaptive context policy (non-hardcoded):
    # - Uses the learned DE3 context model to estimate P(loss) + uncertainty.
    # - Converts that into expected value (EV) in points from candidate TP/SL.
    # - Gates candidates on EV with confidence, and scales position size by confidence.
    "DE3_ADAPTIVE_POLICY": {
        "enabled": True,
        # mode:
        # - "block": enforce EV/confidence gate.
        # - "shadow": annotate would-block signals without blocking.
        "mode": "block",
        "model_path": "de3_context_veto_models.json",
        "log_decisions": False,
        # Confidence controls
        "min_samples": 120,
        "uncertainty_z": 1.0,
        "max_p_loss_std": 0.20,
        "blend_empirical": True,
        "prior_strength": 300.0,
        # EV controls (points)
        "min_ev_lcb_points": 0.0,
        # Optional stricter mean EV floor (None disables)
        "min_ev_mean_points": None,
        # Confidence shaping for sizing
        "confidence_sample_target": 400,
        "confidence_std_ref": 0.20,
        # If top-ranked DE3 candidate fails adaptive gate, block all DE3 candidates for this bar.
        "block_all_on_top": True,
        "risk": {
            "enabled": True,
            "apply_to_size": True,
            # Baseline DE3 contracts before adaptive multiplier (matches prior DE3 runtime behavior).
            "base_size": 5,
            "min_mult": 0.60,
            "max_mult": 1.40,
            "confidence_weight": 0.60,
            "ev_lcb_scale_points": 6.0,
            "min_contracts": 1,
            "max_contracts": 8,
        },
    },
    # Backtest-only DE3 meta policy (low-DOF context risk control).
    # mode:
    #   - "shadow": keep trades, but annotate would-block candidates in trade log/report.
    #   - "block": block candidates when policy score/rules fail.
    "DE3_META_POLICY": {
        "enabled": False,
        "mode": "block",
        "log_decisions": True,
        "min_score": 75.0,
        "er_lookback": 10,
        "mom_min_er": 0.28,
        "rev_max_er": 0.45,
        "shock_gap_atr_mult": 0.70,
        "shock_range_atr_mult": 2.00,
        "high_vol_long_mom_max_vwap_atr": 1.15,
        "long_mom_max_close_pos": 0.88,
        "short_mom_min_close_pos": 0.12,
    },
    # Regime manifold meta-gate:
    # - Live default: disabled (safety-first).
    # - Backtest override below: enabled for evaluation.
    "REGIME_MANIFOLD": {
        "enabled": False,
        "mode": "enforce",  # enforce | shadow
        "persist_state": True,
        "seed": 42,
        "n_probes": 12,
        "mom_window": 60,
        "sigma_ewm_span": 45,
        "atr_window": 20,
        "vol_ewm_span": 30,
        "vol_z_window": 390,
        "omega_bars": 390,
        "A": 0.02,
        "beta_mean": 0.10,
        "beta_std": 0.03,
        "gamma_mean": 0.06,
        "gamma_std": 0.02,
        "c_drift": 0.01,
        "noise_theta": 0.01,
        "noise_phi": 0.01,
        "hotspot_a": 1.0,
        "risk_mult_min": 0.25,
        "risk_mult_max": 1.50,
        "side_bias_min_abs_m": 0.10,
        "side_bias_min_alignment": 0.45,
        "rotational_phi_threshold": 0.08,
        "calibration_file": "manifold_regime_calibration_clean_full.json",
        "enforce_side_bias": True,
        "min_bars": 80,
    },
    # Backtest override: disabled; manifold gate path is deprecated in favor of ManifoldStrategy model.
    "BACKTEST_REGIME_MANIFOLD": {
        "enabled": False,
        "mode": "enforce",
    },
    # MLPhysics rework defaults: coverage-aware thresholds + hierarchical fallbacks.
    "ML_PHYSICS_REWORK": {
        "enabled": True,
        "sessions": ["ASIA", "LONDON", "NY_AM", "NY_PM"],
        # Enforce two-sided validation thresholds; reject one-sided solutions.
        "require_both_sides": True,
        "coverage": {
            # Keep only upper caps; avoid forcing minimum trade coverage.
            "ASIA": {"coverage_min": None, "coverage_max": 0.08, "coverage_target": 0.03, "coverage_penalty": 0.18},
            "LONDON": {"coverage_min": None, "coverage_max": 0.20, "coverage_target": 0.08, "coverage_penalty": 0.15},
            "NY_AM": {"coverage_min": None, "coverage_max": 0.25, "coverage_target": 0.12, "coverage_penalty": 0.12},
            "NY_PM": {"coverage_min": None, "coverage_max": 0.12, "coverage_target": 0.05, "coverage_penalty": 0.18},
        },
    },
    # Hard threshold-safety gates for training artifacts.
    "ML_PHYSICS_THRESHOLD_SAFETY": {
        "require_two_sided": True,
        "max_side_share": 0.70,
    },
    # Live trade-budget nudge: adapt threshold margin when realized coverage drifts.
    "ML_PHYSICS_TRADE_BUDGET_LIVE": {
        "enabled": True,
        "window_evals": 120,
        "min_coverage": 0.01,
        "max_coverage": 0.20,
        "nudge_step": 0.01,
        "sessions": {
            "ASIA": {"min_coverage": 0.01, "max_coverage": 0.12, "window_evals": 150},
            "LONDON": {"min_coverage": 0.02, "max_coverage": 0.20, "window_evals": 120},
            "NY_AM": {"min_coverage": 0.03, "max_coverage": 0.25, "window_evals": 120},
            "NY_PM": {"min_coverage": 0.01, "max_coverage": 0.14, "window_evals": 120},
        },
    },
    # Session-specific training presets (used by ml_train_physics.py)
    "ML_PHYSICS_TRAINING_PRESETS": {
        "ASIA": {
            "timeframe_minutes": 5,
            # ~50 minutes horizon (was 16 bars @ 3m ≈ 48m)
            "horizon_bars": 10,
            # Barrier labels align with how the bot actually realizes PnL (fixed SL/TP brackets).
            "label_mode": "barrier",
            "drop_neutral": True,
            # Lowered to avoid "0 trades" in older macro holdouts under LORO.
            "thr_min": 0.50,
            "thr_max": 0.80,
            "thr_step": 0.01,
            # Prevent threshold overfitting to a tiny number of extreme-probability bars.
            "min_val_trades": 80,
            "require_both_sides": True,
            "min_long_trades": 15,
            "min_short_trades": 15,
            "max_side_share": 0.65,
            # Stability penalty for threshold selection (avg_pnl - penalty/sqrt(trades)).
            "thr_score_penalty": 0.8,
            "coverage_min": None,
            "coverage_max": 0.08,
            "coverage_target": 0.03,
            "coverage_penalty": 0.18,
            "drop_gap_minutes": 75.0,
        },
        "LONDON": {
            "label_mode": "barrier",
            "drop_neutral": True,
            # ~20 minutes horizon (was 20 bars @ 1m)
            "horizon_bars": 4,
            "timeframe_minutes": 5,
            "thr_min": 0.50,
            "thr_max": 0.80,
            "thr_step": 0.01,
            "min_val_trades": 100,
            "require_both_sides": True,
            "min_long_trades": 20,
            "min_short_trades": 20,
            "max_side_share": 0.70,
            "thr_score_penalty": 0.5,
            "coverage_min": None,
            "coverage_max": 0.20,
            "coverage_target": 0.08,
            "coverage_penalty": 0.15,
            "drop_gap_minutes": 75.0,
        },
        "NY_AM": {
            "timeframe_minutes": 5,
            # ~25 minutes horizon (was 25 bars @ 1m)
            "horizon_bars": 5,
            "label_mode": "barrier",
            "drop_neutral": True,
            "thr_min": 0.50,
            "thr_max": 0.80,
            "thr_step": 0.01,
            "min_val_trades": 100,
            "require_both_sides": True,
            "min_long_trades": 20,
            "min_short_trades": 20,
            "max_side_share": 0.70,
            "thr_score_penalty": 0.5,
            "coverage_min": None,
            "coverage_max": 0.25,
            "coverage_target": 0.12,
            "coverage_penalty": 0.12,
            "drop_gap_minutes": 75.0,
        },
        "NY_PM": {
            "label_mode": "barrier",
            "drop_neutral": True,
            "timeframe_minutes": 5,
            # ~30 minutes horizon (was 30 bars @ 1m)
            "horizon_bars": 6,
            "thr_min": 0.50,
            "thr_max": 0.82,
            "thr_step": 0.01,
            "min_val_trades": 120,
            "require_both_sides": True,
            "min_long_trades": 24,
            "min_short_trades": 24,
            "max_side_share": 0.65,
            "thr_score_penalty": 0.65,
            "coverage_min": None,
            "coverage_max": 0.12,
            "coverage_target": 0.05,
            "coverage_penalty": 0.18,
            "drop_gap_minutes": 75.0,
        },
    },

    # Training-time maintenance window to ignore for gap filtering (ET)
    "TRAINING_MAINTENANCE_WINDOW": {
        "start": "17:00",
        "end": "18:00",
        "tolerance_minutes": 5,
    },

    # --- EARLY EXIT OPTIMIZATION (from 2023-2025 backtest analysis) ---
    # Combined early exit rules:
    # 1. exit_if_not_green_by: Exit if not profitable within X bars
    # 2. max_profit_crosses: Exit if price crosses profit/loss threshold X times
    "EARLY_EXIT": {
        "Confluence": {
            "enabled": True,
            "exit_if_not_green_by": 5,  # Exit if not profitable within 5 bars)
            "max_profit_crosses": 1,     # Exit if crosses > 1
        },
        "ICT_Model": {
            "enabled": True,
            "exit_if_not_green_by": 1,   # Exit if not profitable within 1 bar
            "max_profit_crosses": 0,     # Exit if ANY profit cross
        },
        "MLPhysics_ASIA": {
            "enabled": False,
            # ~30 minutes at 5m bars
            "exit_if_not_green_by": 6,
            "max_profit_crosses": 4,      # Allow some chop
        },
        "MLPhysics_LONDON": {
            "enabled": False,
            "exit_if_not_green_by": 6,
            "max_profit_crosses": 4,      # Allow some chop
        },
        "MLPhysics_NY_AM": {
            "enabled": False,
            "exit_if_not_green_by": 6,
            "max_profit_crosses": 4,      # Allow some chop
        },
        "MLPhysics_NY_PM": {
            "enabled": False,
            "exit_if_not_green_by": 6,
            "max_profit_crosses": 4,      # Allow some chop
        },
        # These strategies don't benefit from early exit (wins happen faster than losses)
        "RegimeAdaptive": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # If still red after 30 bars, bail
            "max_profit_crosses": 8       # Allow up to 8 profit/loss crosses before we exit as chop
        },
        "IntradayDip": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "DynamicEngine": {
            "enabled": True,
            "exit_if_not_green_by": 30,   # Give trade room to develop
            "max_profit_crosses": 4,      # Allow some chop
        },
        "ORB_Long": {"enabled": False},
    },

    # --- RegimeAdaptive Filterless Defaults ---
    "REGIME_ADAPTIVE_TUNING": {
        "mode": "filterless",
        "artifact_path": "artifacts/regimeadaptive_v19_live/latest.json",
        "sma_fast": 20,
        "sma_slow": 200,
        "atr_period": 20,
        "range_window": 20,
        "range_spike_mult": 1.3,
        "range_atr_mult": 0.8,
        "cross_atr_mult": 9.0,
        "eq_atr_mult": 0.25,
        "eq_lookback": 20,
        "eq_tolerance": 0.5,
        "vol_window": 30,
        "vol_median_window": 120,
        "high_vol_mult": 1.5,
        "high_vol_block_sessions": ["NY_PM"],
        "block_hours_by_session": {"NY_PM": [12, 13, 14]},
        "use_eq_filter": False,
        "enable_high_vol_gate": False,
        "enable_time_block": False,
        "require_low_vol_trend": False,
        "require_range_spike": False,
        "enable_signal_reversion": False,
    },

    # --- BREAK-EVEN LOGIC ---
    # Move stop to entry when profit reaches X% of TP distance
    "BREAK_EVEN": {
        "enabled": True,
        "trigger_pct": 0.40,  # Trigger at 40% of TP distance
        "buffer_ticks": 1,    # Add 1 tick buffer above entry for longs (below for shorts)
        "trail_pct": 0.25,    # Lock in 25% of profit as trailing stop
    },

    # --- GEMINI 3.0 PRO OPTIMIZATION ---
    "GEMINI": {
        "enabled": True,
        "api_key": str(SECRETS.get("GEMINI_API_KEY", "") or ""),
        "model": "gemini-3-pro-preview",
        "min_interval_minutes": 45,
    },

    # Dynamic Multipliers (Updated by Bot at runtime)
    "DYNAMIC_SL_MULTIPLIER": 1.0,
    "DYNAMIC_TP_MULTIPLIER": 1.0,

    # --- SESSION DEFINITIONS (From Optimization Results) ---
    "SESSIONS": {
        "ASIA": {
            # 6:00 PM - 3:00 AM ET
            "HOURS": [18, 19, 20, 21, 22, 23, 0, 1, 2],
            "MODEL_FILE": "model_asia.joblib",
            "MODEL_FILE_LOW": "model_asia_low.joblib",
            "MODEL_FILE_HIGH": "model_asia_high.joblib",
            # Keep runtime feature timeframe aligned with training presets.
            "TIMEFRAME_MINUTES": 5,
            "THRESHOLD": 0.65,  # Strict Entry
            "SL": 4.0,          # Tight Stop
            "TP": 6.0           # Moderate Target
        },
        "LONDON": {
            # 3:00 AM - 8:00 AM ET
            "HOURS": [3, 4, 5, 6, 7],
            "MODEL_FILE": "model_london.joblib",
            "MODEL_FILE_LOW": "model_london_low.joblib",
            "MODEL_FILE_HIGH": "model_london_high.joblib",
            "THRESHOLD": 0.55,  # Standard Entry
            "SL": 4.0,          # Scalper Stop
            "TP": 4.0           # Scalper Target
        },
        "NY_AM": {
            # 8:00 AM - 12:00 PM ET
            "HOURS": [8, 9, 10, 11],
            "MODEL_FILE": "model_ny_am.joblib",
            "MODEL_FILE_LOW": "model_ny_am_low.joblib",
            "MODEL_FILE_NORMAL": "model_ny_am_normal.joblib",
            "MODEL_FILE_HIGH": "model_ny_am_high.joblib",
            "THRESHOLD": 0.55,
            "SL": 10.0,         # Wide Stop (Breathing Room)
            "TP": 4.0           # High Probability Target (80% WR)
        },
        "NY_PM": {
            # 12:00 PM - 5:00 PM ET
            "HOURS": [12, 13, 14, 15, 16],
            "MODEL_FILE": "model_ny_pm.joblib",
            "MODEL_FILE_LOW": "model_ny_pm_low.joblib",
            "MODEL_FILE_NORMAL": "model_ny_pm_normal.joblib",
            "MODEL_FILE_HIGH": "model_ny_pm_high.joblib",
            "THRESHOLD": 0.55,
            "SL": 10.0,         # Wide Stop
            "TP": 8.0           # Trend Target (Highest PnL)
        }
    }
}


def get_experimental_training_config() -> dict:
    cfg = CONFIG.get("EXPERIMENTAL_TRAINING", {})
    return cfg if isinstance(cfg, dict) else {}


def get_experimental_training_window() -> tuple[Optional[str], Optional[str]]:
    cfg = get_experimental_training_config()
    start = str(cfg.get("start", "") or "").strip() or None
    end = str(cfg.get("end", "") or "").strip() or None
    return start, end


def get_experimental_artifact_suffix(enabled: bool = True) -> str:
    if not enabled:
        return ""
    cfg = get_experimental_training_config()
    return str(cfg.get("artifact_suffix", "") or "").strip()


def resolve_artifact_suffix(override: Optional[str], experimental_enabled: bool) -> str:
    value = str(override or "").strip() if override is not None else ""
    if value:
        return value
    if experimental_enabled:
        fallback = get_experimental_artifact_suffix(enabled=True)
        return fallback or "_exp"
    return ""


def append_artifact_suffix(path_value: str, suffix: str) -> str:
    raw = str(path_value or "").strip()
    tag = str(suffix or "").strip()
    if not raw or not tag:
        return raw
    p = Path(raw)
    name = p.name
    if p.suffix:
        stem = p.stem
        if stem.endswith(tag):
            return raw
        new_name = f"{stem}{tag}{p.suffix}"
    else:
        if name.endswith(tag):
            return raw
        new_name = f"{name}{tag}"
    return str(p.with_name(new_name))


def is_experimental_runtime_enabled() -> bool:
    cfg = get_experimental_training_config()
    return bool(cfg.get("enabled_runtime", False))


def _apply_runtime_experimental_artifacts() -> None:
    if not is_experimental_runtime_enabled():
        return
    suffix = get_experimental_artifact_suffix(enabled=True)
    if not suffix:
        return

    top_level_keys = [
        "ML_PHYSICS_THRESHOLDS_FILE",
        "ML_PHYSICS_METRICS_FILE",
        "DYNAMIC_ENGINE3_DB_FILE",
        "CONTINUATION_SLTP_FILE",
        "VOLATILITY_THRESHOLDS_FILE",
    ]
    for key in top_level_keys:
        value = CONFIG.get(key)
        if isinstance(value, str) and value:
            CONFIG[key] = append_artifact_suffix(value, suffix)

    nested_path_keys = [
        ("MANIFOLD_STRATEGY", ["model_file", "thresholds_file", "confluence_file", "metrics_file"]),
        ("BACKTEST_CONTINUATION_ALLOWLIST", ["cache_file"]),
        ("BACKTEST_FLIP_CONFIDENCE", ["cache_file"]),
        ("CONTINUATION_GUARD", ["allowlist_file"]),
        ("FLIP_CONFIDENCE", ["allowlist_file"]),
        ("DE3_CONTEXT_VETO", ["model_path"]),
        ("DE3_V2", ["db_path"]),
        ("DE3_V3", ["member_db_path", "family_db_path"]),
    ]
    for cfg_key, path_keys in nested_path_keys:
        node = CONFIG.get(cfg_key)
        if not isinstance(node, dict):
            continue
        for path_key in path_keys:
            value = node.get(path_key)
            if isinstance(value, str) and value:
                node[path_key] = append_artifact_suffix(value, suffix)

    sessions = CONFIG.get("SESSIONS", {})
    if isinstance(sessions, dict):
        model_keys = [
            "MODEL_FILE",
            "MODEL_FILE_LOW",
            "MODEL_FILE_NORMAL",
            "MODEL_FILE_HIGH",
            "MODEL_FILE_GATE",
            "MODEL_FILE_EV_LONG",
            "MODEL_FILE_EV_SHORT",
        ]
        for _, settings in sessions.items():
            if not isinstance(settings, dict):
                continue
            for model_key in model_keys:
                value = settings.get(model_key)
                if isinstance(value, str) and value:
                    settings[model_key] = append_artifact_suffix(value, suffix)


CONTRACT_MONTH_CODES = {
    1: "H",  # March
    2: "H",  # March
    3: "H",  # March
    4: "M",  # June
    5: "M",  # June
    6: "M",  # June
    7: "U",  # September
    8: "U",  # September
    9: "U",  # September
    10: "Z",  # December
    11: "Z",  # December
    12: "Z",  # December
}


def determine_current_contract_symbol(
    root: str = "MES",
    tz_name: str = "US/Eastern",
    today: Optional[datetime.date] = None,
) -> str:
    """Return the active contract symbol, handling rollover logic."""
    tz = ZoneInfo(tz_name.replace("US/Eastern", "America/New_York"))
    current_date = today or datetime.datetime.now(tz).date()

    year = current_date.year
    month = current_date.month

    # Contract Months: H (Mar), M (Jun), U (Sep), Z (Dec)
    # Rollover logic: If past the 10th of an expiration month, move to next
    if month == 3 and current_date.day > 10: target_code = "M"
    elif month == 6 and current_date.day > 10: target_code = "U"
    elif month == 9 and current_date.day > 10: target_code = "Z"
    elif month == 12 and current_date.day > 10:
        target_code = "H"
        year += 1  # Roll to next year (March 2026)
    else:
        # Standard Mapping
        if month <= 3: target_code = "H"
        elif month <= 6: target_code = "M"
        elif month <= 9: target_code = "U"
        else: target_code = "Z"

    year_code = str(year % 100).zfill(2)
    return f"{root}.{target_code}{year_code}"


def refresh_target_symbol():
    """Update CONFIG['TARGET_SYMBOL'] based on today's date and configured root."""

    CONFIG["TARGET_SYMBOL"] = determine_current_contract_symbol(
        root=CONFIG.get("CONTRACT_ROOT", "MES"),
        tz_name=CONFIG.get("TIMEZONE", "US/Eastern"),
    )


_apply_runtime_experimental_artifacts()


# Initialize TARGET_SYMBOL at import time
refresh_target_symbol()


SENTIMENT_ENABLED = bool((CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {}).get("enabled", True))
SENTIMENT_POLL_INTERVAL = int((CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {}).get("poll_interval", 30) or 30)
SENTIMENT_PUMP_THRESHOLD = float((CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {}).get("pump_threshold", 0.85) or 0.85)
EMERGENCY_EXIT_THRESHOLD = float(
    (CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {}).get("emergency_exit_threshold", -0.75) or -0.75
)
FINBERT_LOCAL_PATH = str(
    (CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {}).get("finbert_local_path", "./models/finbert")
    or "./models/finbert"
)

# --- Env-var tuning overrides for Kalshi entry block buffer ---
# JULIE_KALSHI_BLOCK_BUF_BALANCED and JULIE_KALSHI_BLOCK_BUF_FP override the
# entry_block_buffer for balanced and forward_primary roles respectively.
# Higher value = fewer blocks (trades need to be more strongly opposed to be blocked).
_kalshi_overlay_cfg = CONFIG.get("KALSHI_TRADE_OVERLAY") or {}
_kalshi_block_buf = _kalshi_overlay_cfg.get("entry_block_buffer") or {}
_bb_balanced = os.environ.get("JULIE_KALSHI_BLOCK_BUF_BALANCED")
_bb_fp = os.environ.get("JULIE_KALSHI_BLOCK_BUF_FP")
if _bb_balanced is not None:
    _kalshi_block_buf["balanced"] = float(_bb_balanced)
if _bb_fp is not None:
    _kalshi_block_buf["forward_primary"] = float(_bb_fp)
if _bb_balanced is not None or _bb_fp is not None:
    _kalshi_overlay_cfg["entry_block_buffer"] = _kalshi_block_buf
    CONFIG["KALSHI_TRADE_OVERLAY"] = _kalshi_overlay_cfg

# === LOCAL OVERRIDE 2026-04-25 — NY-only + RA-off-in-NY (do not commit) ===
# Wired in julie001.py:execution_disabled_filter. Both default ON.
# Toggle via env if needed (set to "0" to disable an override).
CONFIG["LOCAL_NY_ONLY_OVERRIDE"] = _env_flag("JULIE_LOCAL_NY_ONLY_OVERRIDE", True)
# DEPRECATED 2026-04-25 — superseded by LOCAL_RA_V17_GATE_ENABLED below.
# Kept for back-compat only; no longer enforced in execution_disabled_filter.
CONFIG["LOCAL_RA_DISABLED_IN_NY"] = _env_flag("JULIE_LOCAL_RA_DISABLED_IN_NY", True)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-04-25 — V17 RA NY ML gate (replaces LOCAL_RA_DISABLED_IN_NY) ===
# When True, RegimeAdaptive (and the previously-blunt-disabled siblings
# AuctionReversion / SmoothTrendAsia) inside NY hours are gated by the V17
# quality classifier instead of being blanket-blocked. V17 model:
#   artifacts/regime_ml_ra_ny_rule_v17/ra/model.joblib
# Threshold = 0.40 (KEEP if proba >= 0.40, BLOCK otherwise).
# Conservative fallback: BLOCK when bundle missing or features unavailable —
# strictly improves on LOCAL_RA_DISABLED_IN_NY (only allows trades V17 endorses).
# Wired in julie001.py:execution_disabled_filter via v17_should_keep_ra_ny().
CONFIG["LOCAL_RA_V17_GATE_ENABLED"] = _env_flag("JULIE_LOCAL_RA_V17_GATE", True)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-04-25 — AF regime allowlist (do not commit) ===
# Backtest-justified: AF NY-only by manifold regime —
#   TREND_GEODESIC: 14 trades / 50% WR / +$820 / $154 DD  (KEEP)
#   DISPERSED:     47 trades / 44.68% WR / +$788 / $960 DD (KEEP, expanded 2026-04-26)
#   CHOP_SPIRAL:   87 trades / 37.93% WR / -$558 / $1,811 DD  (loss leader — BLOCK)
# Combined TG+DISPERSED: 61 trades / 46.4% WR / +$1,608 / ~$960 DD over 14mo.
# UPDATED 2026-04-26: expanded from TG-only to TG+DISPERSED. Conservative
# behavior preserved: if regime cannot be determined (UNKNOWN/missing) AF is
# BLOCKED. Env override JULIE_LOCAL_AF_ALLOWED_REGIMES accepts a comma-separated
# list (case-insensitive). Set to empty string to disable filter entirely.
# Wired in julie001.py:execution_disabled_filter via the `manifold_regime`
# parameter passed at each call site.
_af_allowed_raw = os.environ.get("JULIE_LOCAL_AF_ALLOWED_REGIMES", "TREND_GEODESIC,DISPERSED")
CONFIG["LOCAL_AF_REGIME_ALLOWED"] = [
    r.strip().upper() for r in _af_allowed_raw.split(",") if r.strip()
]
# Back-compat: the old TG-only flag still works. When set explicitly to True
# (env JULIE_LOCAL_AF_REGIME_TG_ONLY=1), it forces TG-only (overrides
# LOCAL_AF_REGIME_ALLOWED). Default OFF since superseded by the allowlist.
CONFIG["LOCAL_AF_REGIME_TREND_GEODESIC_ONLY"] = _env_flag("JULIE_LOCAL_AF_REGIME_TG_ONLY", False)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-04-26 — V18-DE3 toggle (do not commit) ===
# Default ON (flipped 2026-04-26): live bot uses V18-DE3 (V15's 6 probas
# + 5 Kronos features). Kronos now runs as a long-lived daemon subprocess
# in .kronos_venv (model loaded once, ~0.5-1.5s per inference). On any
# Kronos failure (daemon spawn fails, per-call timeout=15s, model error,
# feature build error) the gate falls back to V15 — V18 cannot make
# behavior worse than V15.
# Wired in julie001.py:_apply_kalshi_trade_overlay_to_signal via
# v18_should_keep_de3() / v15_should_keep_de3().
CONFIG["LOCAL_DE3_USE_V18"] = _env_flag("JULIE_LOCAL_DE3_USE_V18", True)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-04-26 — Recipe B tiered sizing for DE3 V18 (do not commit) ===
# When V18 approves a DE3 candidate, override the bot's default DE3 size with
# a V18-confidence-tier-derived size: >=0.85 -> 10, 0.65-0.85 -> 4,
# 0.60-0.65 -> 1. Tiers are descending (proba_threshold, size).
#
# Toggle off via JULIE_LOCAL_DE3_TIERED_SIZE=0 to restore the bot's existing
# DE3 sizing chain. When V18 is not used (V15 fallback path) or proba is
# unavailable, this helper returns None and the existing logic handles size.
#
# Override tiers via JULIE_LOCAL_DE3_SIZE_TIERS env var, e.g.:
#   JULIE_LOCAL_DE3_SIZE_TIERS="0.90:5,0.70:2,0.60:1"
#
# Wired in julie001.py:de3_size_from_v18_proba() called from
# _apply_live_execution_size when signal['v18_proba'] is present.
CONFIG["LOCAL_DE3_USE_TIERED_SIZING"] = _env_flag("JULIE_LOCAL_DE3_TIERED_SIZE", True)
CONFIG["LOCAL_DE3_SIZE_TIERS"] = [(0.85, 10), (0.65, 4), (0.60, 1)]
_tiers_override = os.environ.get("JULIE_LOCAL_DE3_SIZE_TIERS")
if _tiers_override:
    try:
        CONFIG["LOCAL_DE3_SIZE_TIERS"] = [
            (float(t.split(":")[0]), int(t.split(":")[1]))
            for t in _tiers_override.split(",")
            if ":" in t
        ]
    except Exception:
        pass
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-04-26 — Option 4b: regime-aware tier-4 + whipsaw skip ===
# §8.33.11 holdout analysis: Recipe B's flat tier-4 (proba 0.65-0.85 → size=4)
# is the entire DD source ($1,200 holdout DD breaches $870 ship gate while
# contributing -$360 PnL). Per-regime EV split shows:
#   calm_trend tier-4 (n=20, 55% WR, +$2.19 avg/trade) → keep size=4
#   neutral    tier-4 (n=28, 43% WR, -$4.02 avg/trade) → demote to size=1
#   whipsaw    tier-4 (n=16, defensive cut)            → SKIP (size=0)
# Result: $16,707 PnL / -$594 DD on holdout — passes both ship gates.
#
# When LOCAL_DE3_RECIPE_B_REGIME_AWARE=1 (default), tier-4 sizing branches by
# regime_classifier.current_regime() at signal-birth. When 0, falls back to
# the flat-tier behavior above (size=4 for any regime).
#
# When LOCAL_DE3_TIER4_SKIP_WHIPSAW=1 (default), whipsaw tier-4 returns
# size=0 (skip). When 0, whipsaw tier-4 demotes to size=1 (Option 4 from
# §8.33.11) instead of skipping.
#
# Both flags require LOCAL_DE3_USE_TIERED_SIZING=1 to take effect.
# Both flags require regime_classifier active (JULIE_REGIME_CLASSIFIER=1)
# to actually return a regime label; otherwise current_regime() returns
# "disabled" and the regime-aware path falls back to flat tier-4.
CONFIG["LOCAL_DE3_RECIPE_B_REGIME_AWARE"] = _env_flag(
    "JULIE_LOCAL_DE3_RECIPE_B_REGIME_AWARE", True
)
CONFIG["LOCAL_DE3_TIER4_SKIP_WHIPSAW"] = _env_flag(
    "JULIE_LOCAL_DE3_TIER4_SKIP_WHIPSAW", True
)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-04-30 — Recipe B call-order fix + bracket revert ===
# Bug discovered 2026-04-30: _apply_de3_v18_tiered_size_live (Recipe B) was
# called from _apply_live_execution_size BEFORE the Kalshi overlay stamped
# v18_proba on the signal. Audit of live_trade_factors.csv: 0/143 DE3 trades
# ever had de3_v18_tiered_size stamped — Recipe B never fired in production.
#
# Fix A (LOCAL_DE3_RECIPE_B_KALSHI_FIX): re-apply Recipe B sizing inside
# _apply_kalshi_trade_overlay_to_signal AFTER V18 stamps v18_proba.
#
# Fix C (LOCAL_DE3_RECIPE_B_BRACKET_REVERT): when Fix A lifts size to tier
# >= 4 AND dead-tape clipper had clipped brackets to TP=3/SL=5, restore
# default DE3 brackets. The +$16k Q1 backtest assumes default brackets;
# scalp brackets cap tier-10 wins at 1/16th of potential (per julie001.py
# lines 44-48 / §8.33.16). Per-month replay 2026-01..04 shows trifecta
# (Fix A + B + C) beats deployed by 5× ($16,794 vs $3,309) with better DD
# (-$625 vs -$785).
#
# Both default ON. Disable via env for safe rollback without redeploy.
CONFIG["LOCAL_DE3_RECIPE_B_KALSHI_FIX"] = _env_flag(
    "JULIE_RECIPE_B_KALSHI_FIX", True
)
CONFIG["LOCAL_DE3_RECIPE_B_BRACKET_REVERT"] = _env_flag(
    "JULIE_RECIPE_B_BRACKET_REVERT", True
)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-05-01 — Fix D: Kalshi bypass for DE3 LONG hr 13+14 ET ===
# Walk-forward audit on 206 historical Kalshi-blocked DE3 LONG candidates
# in ET hours 13-14 (10-11am PT) over ~3 weeks of bot history:
#   default brackets:  WR=80% / +$4,951 PnL / +$24/trade
#   T2 brackets:       WR=70% / +$3,828 PnL / +$19/trade
# Hour 12 ET deliberately EXCLUDED — Kalshi correctly filters losers there
# (40% WR / -$1,440 default brackets if unblocked).
#
# Bypass: when strategy=DynamicEngine3, side=LONG, et_hour in {13,14},
# override the Kalshi entry-blocked flag so the trade fires. The Kalshi
# forward-primary 0.55 score gate is too strict in this window — over-blocks
# mean-reversion winners after the first NY hour clears.
#
# Default ON. Disable via env var for safe rollback without redeploy.
CONFIG["LOCAL_DE3_KALSHI_HR13_14_BYPASS"] = _env_flag(
    "JULIE_KALSHI_DE3_HR13_14_BYPASS", True
)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-05-01 — Fix E: dual-path native LONG at hr 13-14 ET ===
# When V18 blocks a DE3 LONG candidate in ET hours 13-14, fall through to
# V15's keep decision (native pipeline). Effectively runs native + V18 DE3
# in parallel in the post-cash-open mean-reversion window.
#
# Walk-forward audit on 28 V18-blocked DE3 LONG candidates (4-day window):
#   default brackets (TP=8.25/SL=10): 100% WR, +$1,144 / 28 trades
#   T2 brackets (TP=25/SL=10):        100% WR, +$3,121 / 28 trades
#
# Default ON. SHORT side NOT included (per user — native fires LONGs only).
# Disable via env for safe rollback. 100% WR has likely survivor bias from
# short data window — monitor [V18_DE3 DUAL_PATH] log lines after deploy.
CONFIG["LOCAL_DE3_DUAL_PATH_HR13_14"] = _env_flag(
    "JULIE_LOCAL_DE3_DUAL_PATH_HR13_14", True
)
# === END LOCAL OVERRIDE ===

# === LOCAL OVERRIDE 2026-05-01 — FibH1214 exempt from Kalshi hour-turn exit ===
# Today's FibH1214_fib_1000 LONG (size 5) was early-exited at 15:01:05 ET via
# Kalshi crowd-flip rule (prob=0.05). Within 60s, price reached TP (7265.25)
# and continued to 7266.25. Money left on table: ~$40 unrealized.
#
# FibH1214 setups have explicit fib-retracement entry validation (8-bar swing
# extremum + counter-bar close) and designed short brackets. Mid-trade Kalshi
# noise doesn't add value to this strategy's exit decision.
#
# Hour-turn rule still active for DE3 / AetherFlow / RegimeAdaptive.
# Default ON (FibH1214 exempt). Disable via env to restore prior behavior.
CONFIG["LOCAL_KALSHI_HOUR_TURN_FIB_DISABLED"] = _env_flag(
    "JULIE_KALSHI_HOUR_TURN_FIB_DISABLED", True
)
# === END LOCAL OVERRIDE ===
