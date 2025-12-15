import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class HTFFVGFilter:
    def __init__(self, expiration_bars=141):
        """
        Stateful FVG Filter.
        Remembers valid structures until they are:
        1. Invalidated by price (broken)
        2. Expired by time (older than expiration_bars)
        """
        self.memory = []  # List of active FVG dicts
        self.expiration_bars = expiration_bars 
        
    def _normalize_cols(self, df):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _scan_for_new_fvgs(self, df, timeframe_label):
        """Scan dataframe for valid FVGs and return them."""
        if df is None or len(df) < 3:
            return []

        df = self._normalize_cols(df)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        times = df.index
        
        found_fvgs = []

        # Iterate through history
        # Stop 2 bars before end to allow FVG formation logic
        for i in range(len(df) - 2):
            c1_h = highs[i]
            c1_l = lows[i]
            c3_h = highs[i+2]
            c3_l = lows[i+2]
            timestamp = times[i+2]
            
            fvg = None
            
            # BULLISH FVG: Gap between C1 High and C3 Low
            if c3_l > c1_h:
                fvg = {
                    'id': f"{timeframe_label}_{timestamp}", # Unique ID
                    'type': 'bullish',
                    'tf': timeframe_label,
                    'top': c3_l,      # Entry
                    'bottom': c1_h,   # Support/Invalidation
                    'created_at': timestamp,
                    'bar_index': i+2
                }
                
            # BEARISH FVG: Gap between C1 Low and C3 High
            elif c3_h < c1_l:
                fvg = {
                    'id': f"{timeframe_label}_{timestamp}",
                    'type': 'bearish',
                    'tf': timeframe_label,
                    'top': c1_l,      # Resistance/Invalidation
                    'bottom': c3_h,   # Entry
                    'created_at': timestamp,
                    'bar_index': i+2
                }

            if fvg:
                # Check if it was invalidated LATER in the same history
                is_broken = False
                for j in range(fvg['bar_index'] + 1, len(df)):
                    if fvg['type'] == 'bullish' and lows[j] < fvg['bottom']:
                        is_broken = True; break
                    elif fvg['type'] == 'bearish' and highs[j] > fvg['top']:
                        is_broken = True; break
                
                if not is_broken:
                    found_fvgs.append(fvg)
        
        return found_fvgs

    def _update_memory(self, new_fvgs):
        """Merge new scans into memory, avoiding duplicates."""
        existing_ids = {f['id'] for f in self.memory}
        
        count = 0
        for f in new_fvgs:
            if f['id'] not in existing_ids:
                self.memory.append(f)
                count += 1
        
        if count > 0:
            logging.info(f"🧠 HTF FVG Memory: Added {count} new structures")

    def _clean_memory(self, current_price, current_time=None):
        """
        Remove FVGs that are:
        1. Broken by current price (LIVE invalidation)
        2. Too old (Expired)
        """
        valid_fvgs = []
        broken_count = 0
        expired_count = 0
        
        for f in self.memory:
            # 1. Check Price Invalidation
            if f['type'] == 'bullish':
                # Support broken if price drops below bottom
                if current_price < f['bottom']:
                    broken_count += 1
                    continue 
            elif f['type'] == 'bearish':
                # Resistance broken if price goes above top
                if current_price > f['top']:
                    broken_count += 1
                    continue

            # 2. Check Time Expiration (Optional logic)
            # Rough approximation: 1H bar = 1 hour, 4H bar = 4 hours
            if current_time and f.get('created_at'):
                age = current_time - f['created_at']
                # Determine max age based on timeframe
                if f['tf'] == '1H':
                    max_age = timedelta(hours=self.expiration_bars) 
                else: # 4H
                    max_age = timedelta(hours=self.expiration_bars * 4)
                
                if age > max_age:
                    expired_count += 1
                    continue

            valid_fvgs.append(f)
        
        if broken_count > 0 or expired_count > 0:
            # logging.info(f"🧹 HTF Memory Cleaned: {broken_count} broken, {expired_count} expired")
            pass
            
        self.memory = valid_fvgs

 def check_signal_blocked(self, signal, current_price, df_1h=None, df_4h=None, tp_dist=None):
        """
        Check if signal is blocked using MEMORY.
        Updates:
        1. Ignore FVGs we have already pierced (dist < 0).
        2. Reduced required room to 40% of TP (was 50%) for better execution.
        """
        # 1. Refresh Memory (if data provided)
        if df_1h is not None and not df_1h.empty:
            fvgs_1h = self._scan_for_new_fvgs(df_1h, '1H')
            self._update_memory(fvgs_1h)
            
        if df_4h is not None and not df_4h.empty:
            fvgs_4h = self._scan_for_new_fvgs(df_4h, '4H')
            self._update_memory(fvgs_4h)

        # 2. Clean Memory (Live Invalidation)
        current_time = datetime.now(df_1h.index.tz) if (df_1h is not None and not df_1h.empty) else datetime.now().astimezone()
        self._clean_memory(current_price, current_time)

        # 3. Check Signal against Active Memory
        if not self.memory:
            return False, None
            
        signal = signal.upper()
        
        # --- DYNAMIC ROOM CALCULATION ---
        # Relaxed: Require only 40% of TP distance (was 50%)
        # Default to 10.0 pts if no TP provided
        min_room_needed = (tp_dist * 0.40) if tp_dist else 10.0
        
        if signal in ['BUY', 'LONG']:
            # Block if Bearish FVG overhead (Price < Resistance)
            for f in self.memory:
                if f['type'] == 'bearish':
                    if current_price < f['top']: 
                        dist = f['bottom'] - current_price
                        
                        # [FIX] Ignore if we are already inside/above the entry (we pierced the wall)
                        if dist < 0:
                            continue
                        
                        # Block only if wall is ahead AND too close
                        if dist < min_room_needed:
                            return True, f"Blocked LONG: Bearish {f['tf']} FVG overhead @ {f['bottom']:.2f} (Dist: {dist:.2f} < {min_room_needed:.2f})"

        elif signal in ['SELL', 'SHORT']:
            # Block if Bullish FVG support below (Price > Support)
            for f in self.memory:
                if f['type'] == 'bullish':
                    if current_price > f['bottom']: 
                        dist = current_price - f['top']
                        
                        # [FIX] Ignore if we are already inside/below the entry (we pierced the wall)
                        if dist < 0:
                            continue
                            
                        # Block only if wall is ahead AND too close
                        if dist < min_room_needed:
                            return True, f"Blocked SHORT: Bullish {f['tf']} FVG support @ {f['top']:.2f} (Dist: {dist:.2f} < {min_room_needed:.2f})"
                        
        return False, None
