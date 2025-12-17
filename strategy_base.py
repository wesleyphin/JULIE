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
        Applies Gemini 3.0 Pro's optimization multipliers to the strategy's base parameters.
        """
        # Retrieve multipliers set by gemini_optimizer (default to 1.0)
        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

        # Calculate optimized values
        final_sl = round(base_sl * sl_mult, 2)
        final_tp = round(base_tp * tp_mult, 2)

        # Safety Hard Caps
        final_sl = min(final_sl, 12.0)

        return final_sl, final_tp
