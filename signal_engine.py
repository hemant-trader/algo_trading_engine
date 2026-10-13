"""
signal_engine.py - Cooldown State Machine & Multi-Timeframe Signal Logic
"""

import datetime
import zoneinfo
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

class CooldownStateMachine:
    def __init__(self, cooldown_bars: int = 2):
        self.cooldown_bars = cooldown_bars
        self.cooldown_remaining_bars = 0
        self.last_emitted_signal = "NO TRADE"
        self.last_signal_bar_epoch: Optional[float] = None
        self.last_processed_bar_epoch: Optional[float] = None
        self.epoch_committed_cache: Dict[float, Tuple[str, Dict]] = {}

    def reset_intraday_state(self):
        self.cooldown_remaining_bars = 0
        self.last_emitted_signal = "NO TRADE"
        self.last_signal_bar_epoch = None
        self.last_processed_bar_epoch = None
        self.epoch_committed_cache.clear()

    def process_candidate(self, candidate_signal: str, bar_epoch: float) -> Tuple[str, Dict]:
        # Strict Monotonic Epoch Check & Idempotency
        if self.last_processed_bar_epoch is not None:
            if bar_epoch < self.last_processed_bar_epoch:
                return "NO TRADE", {
                    "status": "OUT_OF_ORDER_BAR_REJECTED",
                    "is_new_signal_event": False,
                    "cooldown_remaining": self.cooldown_remaining_bars
                }
            if bar_epoch == self.last_processed_bar_epoch:
                cached_action, cached_diag = self.epoch_committed_cache.get(
                    bar_epoch, 
                    ("NO TRADE", {"status": "STATIC_RERUN_PRESERVED", "is_new_signal_event": False, "cooldown_remaining": self.cooldown_remaining_bars})
                )
                diag_copy = dict(cached_diag)
                diag_copy["status"] = "STATIC_RERUN_PRESERVED"
                diag_copy["is_new_signal_event"] = False
                return cached_action, diag_copy

        starting_cooldown = self.cooldown_remaining_bars
        action = "NO TRADE"
        is_new_event = False
        status = "HOLD"

        if candidate_signal == "NO TRADE":
            action = "NO TRADE"
            status = "NO_SIGNAL_CANDIDATE"
        elif candidate_signal == self.last_emitted_signal:
            action = self.last_emitted_signal
            status = "SAME_SIGNAL_HOLDING"
        else:
            if starting_cooldown > 0:
                action = "NO TRADE"
                status = f"COOLDOWN_ACTIVE ({starting_cooldown} bars left)"
            else:
                action = candidate_signal
                self.last_emitted_signal = candidate_signal
                self.last_signal_bar_epoch = bar_epoch
                self.cooldown_remaining_bars = self.cooldown_bars
                is_new_event = True
                status = "TRANSITION_EMITTED"

        # Decrement cooldown strictly once per unique completed bar
        if starting_cooldown > 0 and not is_new_event:
            self.cooldown_remaining_bars -= 1

        self.last_processed_bar_epoch = bar_epoch

        result_diag = {
            "status": status,
            "is_new_signal_event": is_new_event,
            "cooldown_remaining": self.cooldown_remaining_bars
        }

        self.epoch_committed_cache[bar_epoch] = (action, result_diag)
        if len(self.epoch_committed_cache) > 250:
            oldest_key = next(iter(self.epoch_committed_cache))
            del self.epoch_committed_cache[oldest_key]

        return action, result_diag


class SignalEngine:
    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        cooldown_bars_5m: int = 2
    ):
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.cooldown_sm = CooldownStateMachine(cooldown_bars=cooldown_bars_5m)

    def reset_intraday_state(self):
        self.cooldown_sm.reset_intraday_state()

    @staticmethod
    def _validate_candle(bar: Dict) -> bool:
        for k in ("open", "high", "low", "close"):
            v = bar.get(k)
            if v is None or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or v <= 0.0:
                return False
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        if h < max(o, c) or l > min(o, c):
            return False
        return True

    @classmethod
    def _validate_full_series(cls, bars: List[Dict]) -> bool:
        if not bars:
            return False
        for i, b in enumerate(bars):
            if not cls._validate_candle(b):
                return False
            if i > 0 and b["time"] <= bars[i - 1]["time"]:
                return False
        return True

    @staticmethod
    def calculate_wilder_rsi(closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 15:
            return None

        diffs = np.diff(closes)
        gains = np.where(diffs > 0, diffs, 0.0)
        losses = np.where(diffs < 0, -diffs, 0.0)

        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        for i in range(period, len(diffs)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0 and avg_gain == 0.0:
            return 50.0
        if avg_loss == 0.0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @classmethod
    def calculate_atr(cls, bars: List[Dict], period: int = 14) -> Optional[float]:
        if len(bars) < period + 5:
            return None

        trs = []
        for i in range(1, len(bars)):
            h = bars[i]["high"]
            l = bars[i]["low"]
            prev_c = bars[i - 1]["close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)

        atr = float(np.mean(trs[:period]))
        for i in range(period, len(trs)):
            atr = (atr * (period - 1) + trs[i]) / period

        return float(atr)

    def evaluate(
        self,
        bars_1m: List[Dict],
        bars_5m: List[Dict],
        bars_15m: List[Dict],
        is_gap_active: bool,
        gap_reason: str
    ) -> Tuple[str, Dict]:
        diag = {
            "macro_trend": "INSUFFICIENT_DATA",
            "rsi_1m": None,
            "atr_5m": None,
            "status": "AWAITING_WARMUP",
            "is_new_signal_event": False,
            "cooldown_remaining": self.cooldown_sm.cooldown_remaining_bars
        }

        if not bars_5m:
            diag["status"] = "NO_BARS_AVAILABLE"
            return "NO TRADE", diag

        latest_5m = bars_5m[-1]
        latest_5m_epoch = latest_5m["time"].timestamp()

        # Warm-up check: does not consume epoch
        if len(bars_5m) < 25 or len(bars_15m) < 22 or len(bars_1m) < 30:
            diag["status"] = f"WARMING_UP (5m:{len(bars_5m)}/25, 15m:{len(bars_15m)}/22, 1m:{len(bars_1m)}/30)"
            return "NO TRADE", diag

        # Universal Full-Series Validation
        if (not self._validate_full_series(bars_5m) or 
            not self._validate_full_series(bars_15m) or 
            not self._validate_full_series(bars_1m)):
            diag["status"] = "MALFORMED_CANDLE_DATA"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        if not latest_5m.get("is_clean_valid", False):
            diag["status"] = "LATEST_5M_BAR_INVALID"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        if is_gap_active:
            diag["status"] = f"BLOCKED: {gap_reason}"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        closes_5m = pd.Series([b["close"] for b in bars_5m], dtype=float)
        closes_15m = pd.Series([b["close"] for b in bars_15m], dtype=float)
        closes_1m = [float(b["close"]) for b in bars_1m]

        ema9_5m = float(closes_5m.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21_5m = float(closes_5m.ewm(span=21, adjust=False).mean().iloc[-1])
        ema9_15m = float(closes_15m.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21_15m = float(closes_15m.ewm(span=21, adjust=False).mean().iloc[-1])

        atr_5m = self.calculate_atr(bars_5m, self.atr_period)
        rsi_1m = self.calculate_wilder_rsi(closes_1m, self.rsi_period)

        diag["atr_5m"] = round(atr_5m, 2) if atr_5m is not None else None
        diag["rsi_1m"] = round(rsi_1m, 2) if rsi_1m is not None else None

        if atr_5m is None or rsi_1m is None:
            diag["status"] = "INDICATORS_INCOMPLETE"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        # Sideways compression filter
        spread_ema_5m = abs(ema9_5m - ema21_5m)
        if spread_ema_5m < (0.22 * atr_5m):
            diag["status"] = "SIDEWAYS_COMPRESSION"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        # Multi-timeframe trend confluence
        bullish = (ema9_5m > ema21_5m) and (ema9_15m > ema21_15m)
        bearish = (ema9_5m < ema21_5m) and (ema9_15m < ema21_15m)

        if not (bullish or bearish):
            diag["status"] = "TIMEFRAME_CONFLICT"
            diag["macro_trend"] = "MIXED"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        diag["macro_trend"] = "BULLISH" if bullish else "BEARISH"

        candidate = "NO TRADE"
        if bullish and rsi_1m >= 56.0:
            candidate = "BUY CE"
            diag["status"] = "CONFIRMED_BULLISH_EXPANSION"
        elif bearish and rsi_1m <= 44.0:
            candidate = "BUY PE"
            diag["status"] = "CONFIRMED_BEARISH_EXPANSION"
        else:
            diag["status"] = "MOMENTUM_LAG"

        action, sm_d = self.cooldown_sm.process_candidate(candidate, latest_5m_epoch)
        diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
        diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
        if sm_d["status"] != "TRANSITION_EMITTED":
            diag["status"] = sm_d["status"]

        return action, diag
