import logging


_GLOBAL_CB = None


class CircuitBreaker:
    """Stops trading on Daily Max Loss, Max Consecutive Losses, OR Trailing
    Drawdown (peak-to-trough of intraday P&L).

    Trailing DD was added to address the "up $400 then gave back to -$100 =
    $500 DD" pattern — the existing daily-loss cap doesn't protect against
    give-back-of-gains. Backtest on 27 outrageous 2025 days showed 19 of
    them had DD>$350 because P&L peaked positive then drew down. The
    trailing-DD check catches those cases.
    """

    def __init__(
        self,
        max_daily_loss: float = 500.0,
        max_consecutive_losses: int = 3,
        max_trailing_dd: float = 0.0,
    ):
        self.max_daily_loss = abs(max_daily_loss)
        self.max_consecutive_losses = max_consecutive_losses
        # 0 (or negative) disables trailing-DD check for backward compat.
        self.max_trailing_dd = abs(max_trailing_dd)

        self.daily_pnl = 0.0
        self.peak_daily_pnl = 0.0  # highest intraday P&L seen this session
        self.consecutive_losses = 0
        self.is_tripped = False

    def update_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        if self.daily_pnl > self.peak_daily_pnl:
            self.peak_daily_pnl = self.daily_pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check triggers
        if self.daily_pnl <= -self.max_daily_loss:
            self.is_tripped = True
            logging.critical(
                f"🛑 CIRCUIT BREAKER: Max Daily Loss Hit (${self.daily_pnl:.2f})"
            )

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_tripped = True
            logging.critical(
                f"🛑 CIRCUIT BREAKER: Max Consecutive Losses Hit ({self.consecutive_losses})"
            )

        # NEW: trailing-DD trip. Only active when max_trailing_dd > 0.
        if self.max_trailing_dd > 0:
            trailing_dd = self.peak_daily_pnl - self.daily_pnl
            if trailing_dd >= self.max_trailing_dd:
                self.is_tripped = True
                logging.critical(
                    f"🛑 CIRCUIT BREAKER: Trailing DD Hit "
                    f"(peak=${self.peak_daily_pnl:.2f} cur=${self.daily_pnl:.2f} "
                    f"dd=${trailing_dd:.2f} / limit=${self.max_trailing_dd:.2f})"
                )

    def should_block_trade(self) -> tuple[bool, str]:
        if self.is_tripped:
            return True, "Circuit Breaker Tripped (Risk Limit Reached)"
        return False, ""

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.peak_daily_pnl = 0.0
        self.consecutive_losses = 0
        self.is_tripped = False
        logging.info("🔄 Circuit Breaker reset for new day")

    def get_state(self) -> dict:
        return {
            "daily_pnl": self.daily_pnl,
            "peak_daily_pnl": self.peak_daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "is_tripped": self.is_tripped,
        }

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self.daily_pnl = float(state.get("daily_pnl", self.daily_pnl))
        self.peak_daily_pnl = float(state.get("peak_daily_pnl", self.peak_daily_pnl))
        self.consecutive_losses = int(state.get("consecutive_losses", self.consecutive_losses))
        self.is_tripped = bool(state.get("is_tripped", self.is_tripped))
