"""
Risk Engine Module

Contains the OptimizedTPEngine for dynamic take profit calculation based on
volatility and market regime detection.
"""
import numpy as np


class OptimizedTPEngine:
    """
    Optimized take profit calculation that scales around 2.0pt base.
    Uses volatility and regime detection to adjust multiplier.
    From esbacktest002.py - proven to work with Confluence strategy.
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def estimate_volatility(self, prices: np.ndarray) -> float:
        """Quick volatility estimate (std of returns)"""
        if len(prices) < 2:
            return 0.5
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.5

    def estimate_garch_volatility(self, returns: np.ndarray) -> float:
        """GARCH(1,1) volatility estimate"""
        if len(returns) < 5:
            return np.std(returns) if len(returns) > 0 else 1.0
        sigma2 = np.var(returns)
        omega, alpha, beta = 0.00001, 0.1, 0.85
        for r in returns[-20:]:
            sigma2 = omega + alpha * r**2 + beta * sigma2
        return np.sqrt(sigma2)

    def calculate_entropy(self, returns: np.ndarray) -> float:
        """Market regime detection: trending (low) vs choppy (high)"""
        if len(returns) < 10:
            return 0.5
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) if len(hist) > 0 else 0.5
        max_entropy = np.log2(10)
        return entropy / max_entropy

    def calculate_optimized_tp(self, prices: np.ndarray) -> float:
        """
        Calculate TP as a multiplier around 2.0pt base.
        Returns DISTANCE in points.
        """
        BASE_TP = 2.0
        if len(prices) < 20:
            return BASE_TP

        vol = self.estimate_volatility(prices[-60:])
        returns = np.diff(prices) / prices[:-1]
        garch_vol = self.estimate_garch_volatility(returns[-60:] if len(returns) >= 60 else returns)
        entropy = self.calculate_entropy(returns[-60:] if len(returns) >= 60 else returns)

        volatility_signal = vol + (garch_vol * 0.5)
        regime_signal = entropy
        multiplier = 1.0 + (volatility_signal * 0.12) + (regime_signal * 0.08)
        multiplier = np.clip(multiplier, 0.95, 1.35)

        return float(np.round(BASE_TP * multiplier, 2))
