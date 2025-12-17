from abc import ABC, abstractmethod
from typing import Dict, Optional
from config import CONFIG

class Strategy(ABC):
    @abstractmethod
    def on_bar(self, df) -> Optional[Dict]:
        """Return a signal dict or None for no action."""
        raise NotImplementedError

    def calculate_dynamic_exit(self, base_sl: float, base_tp: float) -> tuple:
        """
        Order of Operations:
        1. Apply Gemini Volatility Multiplier (Macro Adjustment)
        2. Divide by 4 (Quarter Theory Execution)
        """
        # 1. Fetch Multipliers (Default 1.0)
        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

        # 2. Apply Gemini Multiplier FIRST (Adjust for Market Regime)
        regime_adjusted_sl = base_sl * sl_mult
        regime_adjusted_tp = base_tp * tp_mult

        # 3. Apply Division by 4 SECOND (Quartering)
        final_sl = regime_adjusted_sl / 4.0
        final_tp = regime_adjusted_tp / 4.0

        # 4. Round and Safety Cap (Max 15 points SL)
        final_sl = round(min(final_sl, 15.0), 2)
        final_tp = round(final_tp, 2)

        return final_sl, final_tp
