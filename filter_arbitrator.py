"""
Filter Arbitrator - Dual-Filter Decision System

When the Legacy (Dec 17th) filter system ALLOWS a trade but the
Upgraded filter system BLOCKS it, this arbitrator analyzes market
conditions to make an intelligent override decision.

Decision Logic:
1. Both BLOCK → BLOCK (no override possible)
2. Both ALLOW → ALLOW (unanimous agreement)
3. Legacy ALLOWS + Upgraded BLOCKS → ANALYZE and decide

Analysis considers:
- Current market location (session, structure)
- Recent candle patterns
- Momentum alignment
- Risk/reward context
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ArbitrationResult:
    """Result of filter arbitration analysis."""
    allow_trade: bool
    reason: str
    confidence: float  # 0.0 to 1.0
    legacy_decision: str
    upgraded_decision: str
    analysis_factors: Dict


class FilterArbitrator:
    """
    Intelligent arbitrator between Legacy and Upgraded filter systems.

    Philosophy:
    - If legacy (simpler, battle-tested) would allow the trade
    - But upgraded (more complex, stricter) blocks it
    - Analyze WHY upgraded blocked and if the concern is valid
    - Consider market context to make final decision
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Args:
            confidence_threshold: Minimum confidence to override upgraded block (0.6 = 60%)
        """
        self.confidence_threshold = confidence_threshold
        self.override_count = 0
        self.block_count = 0

        logging.info(f"Filter Arbitrator initialized (confidence threshold: {confidence_threshold})")

    def arbitrate(self,
                  df: pd.DataFrame,
                  side: str,
                  legacy_blocked: bool,
                  legacy_reason: str,
                  upgraded_blocked: bool,
                  upgraded_reason: str,
                  current_price: float,
                  tp_dist: float = None,
                  sl_dist: float = None) -> ArbitrationResult:
        """
        Main arbitration method.

        Args:
            df: OHLCV DataFrame
            side: 'LONG' or 'SHORT'
            legacy_blocked: Whether legacy system blocked
            legacy_reason: Reason from legacy system
            upgraded_blocked: Whether upgraded system blocked
            upgraded_reason: Reason from upgraded system
            current_price: Current market price
            tp_dist: Target distance (optional)
            sl_dist: Stop loss distance (optional)

        Returns:
            ArbitrationResult with final decision
        """
        # Case 1: Both agree to BLOCK
        if legacy_blocked and upgraded_blocked:
            return ArbitrationResult(
                allow_trade=False,
                reason=f"Unanimous BLOCK: Legacy='{legacy_reason}' | Upgraded='{upgraded_reason}'",
                confidence=1.0,
                legacy_decision="BLOCK",
                upgraded_decision="BLOCK",
                analysis_factors={}
            )

        # Case 2: Both agree to ALLOW
        if not legacy_blocked and not upgraded_blocked:
            return ArbitrationResult(
                allow_trade=True,
                reason="Unanimous ALLOW: Both filter systems approve",
                confidence=1.0,
                legacy_decision="ALLOW",
                upgraded_decision="ALLOW",
                analysis_factors={}
            )

        # Case 3: Legacy BLOCKS but Upgraded ALLOWS (rare - upgraded is usually stricter)
        if legacy_blocked and not upgraded_blocked:
            # Trust the upgraded system's more sophisticated analysis
            return ArbitrationResult(
                allow_trade=True,
                reason=f"Upgraded Override: Legacy blocked ({legacy_reason}) but upgraded approves",
                confidence=0.8,
                legacy_decision="BLOCK",
                upgraded_decision="ALLOW",
                analysis_factors={"upgraded_override": True}
            )

        # Case 4: Legacy ALLOWS but Upgraded BLOCKS → ANALYZE
        # This is the key case - simpler system would trade, complex system says no
        return self._analyze_override(
            df=df,
            side=side,
            upgraded_reason=upgraded_reason,
            current_price=current_price,
            tp_dist=tp_dist,
            sl_dist=sl_dist
        )

    def _analyze_override(self,
                         df: pd.DataFrame,
                         side: str,
                         upgraded_reason: str,
                         current_price: float,
                         tp_dist: float = None,
                         sl_dist: float = None) -> ArbitrationResult:
        """
        Analyze whether to override the upgraded system's block.

        Uses multiple factors to determine if the trade should proceed.
        """
        factors = {}
        confidence_score = 0.5  # Start neutral

        # === FACTOR 1: Candle Pattern Analysis ===
        candle_score = self._analyze_candle_patterns(df, side)
        factors['candle_pattern'] = candle_score
        confidence_score += candle_score * 0.15

        # === FACTOR 2: Momentum Alignment ===
        momentum_score = self._analyze_momentum(df, side)
        factors['momentum'] = momentum_score
        confidence_score += momentum_score * 0.15

        # === FACTOR 3: Support/Resistance Location ===
        location_score = self._analyze_market_location(df, side, current_price)
        factors['market_location'] = location_score
        confidence_score += location_score * 0.15

        # === FACTOR 4: Volatility Context ===
        volatility_score = self._analyze_volatility_context(df)
        factors['volatility'] = volatility_score
        confidence_score += volatility_score * 0.1

        # === FACTOR 5: Block Reason Severity ===
        severity_penalty = self._assess_block_severity(upgraded_reason)
        factors['block_severity'] = severity_penalty
        confidence_score -= severity_penalty * 0.2

        # === FACTOR 6: Risk/Reward Check ===
        if tp_dist and sl_dist:
            rr_score = self._analyze_risk_reward(tp_dist, sl_dist)
            factors['risk_reward'] = rr_score
            confidence_score += rr_score * 0.1

        # Clamp confidence to [0, 1]
        confidence_score = max(0.0, min(1.0, confidence_score))
        factors['final_confidence'] = confidence_score

        # Make decision
        allow_trade = confidence_score >= self.confidence_threshold

        if allow_trade:
            self.override_count += 1
            reason = f"OVERRIDE APPROVED (conf={confidence_score:.2f}): Legacy would trade, analysis supports"
            logging.info(f"🔓 ARBITRATOR: {reason}")
        else:
            self.block_count += 1
            reason = f"OVERRIDE DENIED (conf={confidence_score:.2f}): Upgraded block upheld - {upgraded_reason}"
            logging.info(f"🚫 ARBITRATOR: {reason}")

        return ArbitrationResult(
            allow_trade=allow_trade,
            reason=reason,
            confidence=confidence_score,
            legacy_decision="ALLOW",
            upgraded_decision="BLOCK",
            analysis_factors=factors
        )

    def _analyze_candle_patterns(self, df: pd.DataFrame, side: str) -> float:
        """
        Analyze recent candle patterns for trade direction support.
        Returns score from -1.0 (against) to +1.0 (supporting).
        """
        if len(df) < 5:
            return 0.0

        last_5 = df.iloc[-5:]
        score = 0.0

        # Check for rejection wicks
        for i, bar in last_5.iterrows():
            body = abs(bar['close'] - bar['open'])
            upper_wick = bar['high'] - max(bar['open'], bar['close'])
            lower_wick = min(bar['open'], bar['close']) - bar['low']

            if body > 0:
                if side == 'LONG' and lower_wick > body * 0.5:
                    score += 0.2  # Hammer-like rejection
                elif side == 'SHORT' and upper_wick > body * 0.5:
                    score += 0.2  # Shooting star rejection

        # Check last bar direction alignment
        last_bar = df.iloc[-1]
        if side == 'LONG' and last_bar['close'] > last_bar['open']:
            score += 0.1  # Green candle for long
        elif side == 'SHORT' and last_bar['close'] < last_bar['open']:
            score += 0.1  # Red candle for short

        return max(-1.0, min(1.0, score))

    def _analyze_momentum(self, df: pd.DataFrame, side: str) -> float:
        """
        Analyze short-term momentum alignment.
        Returns score from -1.0 to +1.0.
        """
        if len(df) < 20:
            return 0.0

        # Simple momentum: Compare current price to 10-bar and 20-bar SMAs
        closes = df['close']
        sma_10 = closes.iloc[-10:].mean()
        sma_20 = closes.iloc[-20:].mean()
        current = closes.iloc[-1]

        score = 0.0

        if side == 'LONG':
            if current > sma_10:
                score += 0.3
            if sma_10 > sma_20:
                score += 0.3
            if current > sma_20:
                score += 0.2
        else:  # SHORT
            if current < sma_10:
                score += 0.3
            if sma_10 < sma_20:
                score += 0.3
            if current < sma_20:
                score += 0.2

        # Penalize if momentum is strongly against
        if side == 'LONG' and current < sma_20 and sma_10 < sma_20:
            score -= 0.5
        elif side == 'SHORT' and current > sma_20 and sma_10 > sma_20:
            score -= 0.5

        return max(-1.0, min(1.0, score))

    def _analyze_market_location(self, df: pd.DataFrame, side: str, current_price: float) -> float:
        """
        Analyze price location within recent range.
        Favor trades at range extremes (support/resistance).
        """
        if len(df) < 50:
            return 0.0

        recent_high = df['high'].iloc[-50:].max()
        recent_low = df['low'].iloc[-50:].min()
        range_size = recent_high - recent_low

        if range_size == 0:
            return 0.0

        # Position in range (0 = low, 1 = high)
        position = (current_price - recent_low) / range_size

        score = 0.0

        if side == 'LONG':
            # Favor longs near the bottom of range
            if position < 0.3:
                score = 0.8  # Near support
            elif position < 0.5:
                score = 0.3  # Lower half
            elif position > 0.8:
                score = -0.5  # Chasing at highs
        else:  # SHORT
            # Favor shorts near the top of range
            if position > 0.7:
                score = 0.8  # Near resistance
            elif position > 0.5:
                score = 0.3  # Upper half
            elif position < 0.2:
                score = -0.5  # Chasing at lows

        return score

    def _analyze_volatility_context(self, df: pd.DataFrame) -> float:
        """
        Analyze current volatility vs recent average.
        Returns score favoring normal volatility conditions.
        """
        if len(df) < 20:
            return 0.0

        # Calculate ATR-like measure
        recent_ranges = df['high'].iloc[-20:] - df['low'].iloc[-20:]
        avg_range = recent_ranges.mean()
        current_range = df['high'].iloc[-1] - df['low'].iloc[-1]

        if avg_range == 0:
            return 0.0

        vol_ratio = current_range / avg_range

        # Favor normal volatility (0.8x - 1.5x average)
        if 0.8 <= vol_ratio <= 1.5:
            return 0.5  # Normal conditions
        elif vol_ratio < 0.5:
            return 0.2  # Very low vol - less conviction
        elif vol_ratio > 2.5:
            return -0.3  # Very high vol - caution
        else:
            return 0.0

    def _assess_block_severity(self, upgraded_reason: str) -> float:
        """
        Assess how severe the upgraded block reason is.
        Returns penalty from 0.0 (minor) to 1.0 (critical).
        """
        reason_lower = upgraded_reason.lower()

        # Critical blocks - never override
        if 'tier 3' in reason_lower or 'nuke' in reason_lower or 'capitulation' in reason_lower:
            return 1.0

        # Serious blocks - high penalty
        if 'tier 4' in reason_lower or 'macro trend' in reason_lower:
            return 0.7

        # Moderate blocks
        if 'tier 2' in reason_lower or 'tier 1' in reason_lower:
            return 0.4

        # Minor blocks
        if 'shockwave' in reason_lower:
            return 0.5

        # Unknown - moderate penalty
        return 0.3

    def _analyze_risk_reward(self, tp_dist: float, sl_dist: float) -> float:
        """
        Analyze risk/reward ratio quality.
        Returns score from -0.5 to +0.5.
        """
        if sl_dist <= 0:
            return 0.0

        rr = tp_dist / sl_dist

        if rr >= 2.0:
            return 0.5  # Excellent R:R
        elif rr >= 1.5:
            return 0.3  # Good R:R
        elif rr >= 1.0:
            return 0.0  # Neutral
        else:
            return -0.3  # Poor R:R

    def get_stats(self) -> Dict:
        """Get arbitrator statistics."""
        total = self.override_count + self.block_count
        override_rate = self.override_count / total if total > 0 else 0

        return {
            'total_arbitrations': total,
            'overrides': self.override_count,
            'blocks_upheld': self.block_count,
            'override_rate': f"{override_rate:.1%}"
        }
