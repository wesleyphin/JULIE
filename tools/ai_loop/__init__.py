"""Julie AI-loop: nightly journal + ML-analyzer + backtest-gated auto-adjust.

Five layers:
    1. journal.py    — reads yesterday's logs + trades, emits daily journal
    2. analyzer.py   — heuristic pattern-matcher, proposes adjustments
    3. validator.py  — runs a short backtest per proposal, rubber-stamps or rejects
    4. applier.py    — for green-lit proposals inside the whitelist, writes a
                       git commit that patches the launcher env-defaults
    5. monitor.py    — tracks live PnL trajectory vs backtest forecast; if live
                       under-delivers, auto-reverts the most recent change

Safety — see config.py:
    AUTO_ADJUSTABLE_PARAMS  whitelist of env vars + file-based numeric configs
    BOUNDS                  absolute clamps + per-step delta limits
    COOLDOWN_DAYS           a given param can't auto-adjust more than 1× / this many days
    STOP_LOSS_DOLLARS       if live drawdown exceeds this in last 48 hrs, freeze
    KILL_SWITCH_ENV         set JULIE_FREEZE_AUTO_CONFIG=1 to halt all auto-apply

Orchestrator:
    run_daily.py  — runs all 5 layers in sequence, intended to be cron'd nightly

Every applied change is:
    - Git-committed with "[AUTO_APPLIED]" tag for grep-ability
    - Recorded in ai_loop_data/audit.jsonl append-only
    - Reversible via `git revert <sha>`
"""
