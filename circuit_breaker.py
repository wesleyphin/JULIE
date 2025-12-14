import logging


class CircuitBreaker:
    """
    Stops trading if Daily Max Loss or Max Consecutive Losses are hit.
    """

    def __init__(self, max_daily_loss=500.0, max_consecutive_losses=3):
        self.max_daily_loss = abs(max_daily_loss)
        self.max_consecutive_losses = max_consecutive_losses

        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.is_tripped = False

    def update_trade_result(self, pnl: float):
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check triggers
        if self.daily_pnl <= -self.max_daily_loss:
            self.is_tripped = True
            logging.critical(f"🛑 CIRCUIT BREAKER: Max Daily Loss Hit (${self.daily_pnl:.2f})")

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_tripped = True
            logging.critical(f"🛑 CIRCUIT BREAKER: Max Consecutive Losses Hit ({self.consecutive_losses})")

    def should_block_trade(self) -> tuple[bool, str]:
        if self.is_tripped:
            return True, "Circuit Breaker Tripped (Risk Limit Reached)"
        return False, ""

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.is_tripped = False
        logging.info("🔄 Circuit Breaker reset for new day")
