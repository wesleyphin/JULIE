import copy
import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from de3_v2_generator import _default_v2_config, generate_de3_v2
from dynamic_signal_engine3 import DynamicSignalEngine3


class DE3V2SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cfg_backup = copy.deepcopy(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._cfg_backup)

    @staticmethod
    def _build_signal_df() -> pd.DataFrame:
        idx = pd.date_range("2025-01-01 00:00:00", periods=2, freq="5min", tz="US/Eastern")
        return pd.DataFrame(
            {
                "open": [100.0, 106.0],
                "high": [107.0, 108.0],
                "low": [99.5, 105.5],
                "close": [106.0, 107.0],
                "volume": [1000.0, 1000.0],
            },
            index=idx,
        )

    def test_v1_mode_signal_unchanged(self) -> None:
        td_path = Path.cwd() / ".tmp_de3_v2_tests_case1"
        shutil.rmtree(td_path, ignore_errors=True)
        td_path.mkdir(parents=True, exist_ok=True)
        try:
            v1_path = td_path / "v1.json"
            payload = {
                "settings": {"min_trades": 0, "min_win_rate": 0.0, "min_avg_pnl": 0.0},
                "strategies": [
                    {
                        "TF": "5min",
                        "Session": "00-03",
                        "Type": "Long_Mom",
                        "Thresh": 2.0,
                        "Best_SL": 8.0,
                        "Best_TP": 8.0,
                        "Opt_WR": 0.55,
                        "Trades": 100,
                        "Avg_PnL": 1.0,
                        "Score": 1.0,
                    }
                ],
            }
            v1_path.write_text(json.dumps(payload), encoding="utf-8")

            CONFIG["DYNAMIC_ENGINE3_DB_FILE"] = str(v1_path)
            CONFIG["DE3_VERSION"] = "v1"
            CONFIG["DE3_V2"] = {"enabled": False, "db_path": str(td_path / "v2.json")}
            CONFIG["DYNAMIC_ENGINE3_RUNTIME"] = {
                "enabled": True,
                "use_db_settings": False,
                "min_trades": 0,
                "min_score": -1e9,
                "min_avg_pnl": -1e9,
                "abstain": {"enabled": False},
                "weights": {"score": 0.45, "win_rate": 0.25, "avg_pnl": 0.2, "trades": 0.1, "bucket": 0.0, "location": 0.0},
            }

            df = self._build_signal_df()
            now = df.index[-1].to_pydatetime()

            engine_config = DynamicSignalEngine3()
            sig_a = engine_config.check_signal(now, df, df)

            engine_direct = DynamicSignalEngine3(db_path=str(v1_path))
            sig_b = engine_direct.check_signal(now, df, df)

            self.assertIsNotNone(sig_a)
            self.assertIsNotNone(sig_b)
            self.assertEqual(sig_a["signal"], sig_b["signal"])
            self.assertEqual(sig_a["strategy_id"], sig_b["strategy_id"])
        finally:
            shutil.rmtree(td_path, ignore_errors=True)

    def test_v2_mode_loads_v2_and_signals(self) -> None:
        td_path = Path.cwd() / ".tmp_de3_v2_tests_case2"
        shutil.rmtree(td_path, ignore_errors=True)
        td_path.mkdir(parents=True, exist_ok=True)
        try:
            v1_path = td_path / "v1.json"
            v2_path = td_path / "v2.json"
            base_strategy = {
                "TF": "5min",
                "Session": "00-03",
                "Type": "Long_Mom",
                "Thresh": 2.0,
                "Best_SL": 8.0,
                "Best_TP": 8.0,
                "Opt_WR": 0.55,
                "Trades": 100,
                "Avg_PnL": 1.0,
                "Score": 1.0,
            }
            v1_path.write_text(json.dumps({"strategies": [base_strategy]}), encoding="utf-8")
            v2_path.write_text(json.dumps({"version": "v2", "settings": {}, "strategies": [base_strategy]}), encoding="utf-8")

            CONFIG["DYNAMIC_ENGINE3_DB_FILE"] = str(v1_path)
            CONFIG["DE3_VERSION"] = "v2"
            CONFIG["DE3_V2"] = {"enabled": True, "db_path": str(v2_path)}
            CONFIG["DYNAMIC_ENGINE3_RUNTIME"] = {
                "enabled": True,
                "use_db_settings": False,
                "min_trades": 0,
                "min_score": -1e9,
                "min_avg_pnl": -1e9,
                "abstain": {"enabled": False},
                "weights": {"score": 0.45, "win_rate": 0.25, "avg_pnl": 0.2, "trades": 0.1, "bucket": 0.0, "location": 0.0},
            }

            df = self._build_signal_df()
            now = df.index[-1].to_pydatetime()
            engine = DynamicSignalEngine3()
            sig = engine.check_signal(now, df, df)
            self.assertIsNotNone(sig)
            self.assertIn("v2", str(engine.db_version).lower())
        finally:
            shutil.rmtree(td_path, ignore_errors=True)

    def test_generator_is_deterministic(self) -> None:
        td_path = Path.cwd() / ".tmp_de3_v2_tests_case3"
        shutil.rmtree(td_path, ignore_errors=True)
        td_path.mkdir(parents=True, exist_ok=True)
        try:
            source = td_path / "tiny.csv"
            out1 = td_path / "out1.json"
            out2 = td_path / "out2.json"

            np.random.seed(42)
            idx = pd.date_range("2018-01-01 00:00:00", periods=60 * 24 * 20, freq="1min", tz="US/Eastern")
            base = 2700.0 + np.cumsum(np.random.normal(0.0, 0.25, len(idx)))
            close = base
            open_ = np.concatenate(([base[0]], base[:-1]))
            high = np.maximum(open_, close) + 0.2
            low = np.minimum(open_, close) - 0.2
            vol = np.full(len(idx), 1000.0)
            df = pd.DataFrame(
                {
                    "timestamp": idx.tz_convert("UTC").astype(str),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                }
            )
            df.to_csv(source, index=False)

            cfg = _default_v2_config()
            cfg["mode"] = "fixed_split"
            cfg["train_end"] = "2018-01-10"
            cfg["valid_start"] = "2018-01-11"
            cfg["valid_end"] = "2018-01-20"
            cfg["symbol_mode"] = "single"
            cfg["search_space"] = {
                "thresholds": [0.25, 0.5],
                "sl_list": [1.0, 2.0],
                "rr_list": [1.0, 1.5],
                "max_per_bucket": 2,
            }
            cfg["scoring"]["min_train_trades"] = 5
            cfg["scoring"]["min_oos_trades"] = 5
            cfg["min_tp"] = 0.25
            cfg["max_tp"] = 8.0
            cfg["max_horizon"] = 30

            payload1 = generate_de3_v2(
                source_path=source,
                out_path=out1,
                cfg=cfg,
                cache_dir=td_path / "cache",
                use_cache=False,
            )
            payload2 = generate_de3_v2(
                source_path=source,
                out_path=out2,
                cfg=cfg,
                cache_dir=td_path / "cache",
                use_cache=False,
            )

            self.assertEqual(payload1.get("strategies"), payload2.get("strategies"))
        finally:
            shutil.rmtree(td_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
