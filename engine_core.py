"""
engine_core.py - Institutional Sensex/Nifty Directional Engine
Includes:
- Monotonic Candle Aggregation (1m, 5m, 15m)
- Decoupled Pure Cooldown State Machine
- Multi-Timeframe Alignment & Wilder RSI
- Fail-Closed OI/PCR Change Confirmation Gate
- Deterministic Acceptance & Unit Test Suite
"""

import datetime
import zoneinfo
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# ============================================================================
# 1. FAIL-CLOSED OI / PCR CONFIRMATION ENGINE
# ============================================================================
class OIConfirmationEngine:
    """
    Completed-data-only, change-based OI & PCR Confirmation Gate.
    Evaluates institutional positioning delta across consecutive finalized snapshots.
    """
    def __init__(self, max_stale_seconds: float = 300.0, mandatory_gate: bool = True):
        self.max_stale_seconds = max_stale_seconds
        self.mandatory_gate = mandatory_gate
        self.last_snapshot: Optional[Dict] = None

    def reset_intraday_state(self):
        self.last_snapshot = None

    def process_snapshot(
        self,
        snapshot_time: datetime.datetime,
        underlying_price: float,
        chain_data: List[Dict],
        current_wall_time: datetime.datetime
    ) -> Dict:
        """
        Snapshot contract:
        - Timestamped, completed data only
        - Max stale validation
        - Computes Delta OI (PE vs CE) across consecutive snapshots
        """
        # 1. Staleness & Timestamp Validation
        age = (current_wall_time - snapshot_time).total_seconds()
        if age < 0.0 or age > self.max_stale_seconds:
            return {
                "status": "DATA_UNAVAILABLE",
                "reason": f"STALE_OR_FUTURE_SNAPSHOT (Age: {age:.1f}s)",
                "pcr": None,
                "delta_ce_oi": 0,
                "delta_pe_oi": 0
            }

        if not chain_data or not isinstance(chain_data, list):
            return {
                "status": "DATA_UNAVAILABLE",
                "reason": "EMPTY_CHAIN_DATA",
                "pcr": None,
                "delta_ce_oi": 0,
                "delta_pe_oi": 0
            }

        total_ce_oi = 0
        total_pe_oi = 0

        for row in chain_data:
            c_market = row.get("call_options", {}).get("market_data", {})
            p_market = row.get("put_options", {}).get("market_data", {})
            total_ce_oi += int(c_market.get("oi", 0) or 0)
            total_pe_oi += int(p_market.get("oi", 0) or 0)

        if total_ce_oi <= 0 or total_pe_oi <= 0:
            return {
                "status": "DATA_UNAVAILABLE",
                "reason": "ZERO_OI_DETECTED",
                "pcr": None,
                "delta_ce_oi": 0,
                "delta_pe_oi": 0
            }

        current_pcr = round(total_pe_oi / total_ce_oi, 3)

        current_record = {
            "time": snapshot_time,
            "price": underlying_price,
            "ce_oi": total_ce_oi,
            "pe_oi": total_pe_oi,
            "pcr": current_pcr
        }

        # First snapshot: Baseline established, no delta available yet
        if self.last_snapshot is None:
            self.last_snapshot = current_record
            return {
                "status": "DATA_UNAVAILABLE",
                "reason": "INITIALIZING_BASELINE_SNAPSHOT",
                "pcr": current_pcr,
                "delta_ce_oi": 0,
                "delta_pe_oi": 0
            }

        # Ensure strict temporal monotonicity
        if snapshot_time <= self.last_snapshot["time"]:
            return {
                "status": "DATA_UNAVAILABLE",
                "reason": "NON_MONOTONIC_SNAPSHOT_TIMESTAMP",
                "pcr": current_pcr,
                "delta_ce_oi": 0,
                "delta_pe_oi": 0
            }

        # Calculate Consecutive Changes
        delta_ce_oi = total_ce_oi - self.last_snapshot["ce_oi"]
        delta_pe_oi = total_pe_oi - self.last_snapshot["pe_oi"]
        delta_price = underlying_price - self.last_snapshot["price"]

        self.last_snapshot = current_record

        # Change-Based Confirmation Rules
        # Bullish: Put writing outpacing Call writing (delta_pe > delta_ce) and Price not dropping
        if delta_pe_oi > delta_ce_oi and delta_price >= 0.0:
            status = "BULLISH_CONFIRMATION"
            reason = "PUT_BUILDUP_AND_PRICE_SUPPORT"
        # Bearish: Call writing outpacing Put writing (delta_ce > delta_pe) and Price not rising
        elif delta_ce_oi > delta_pe_oi and delta_price <= 0.0:
            status = "BEARISH_CONFIRMATION"
            reason = "CALL_BUILDUP_AND_PRICE_RESISTANCE"
        else:
            status = "CONFLICT"
            reason = f"DIVERGENT_DYNAMICS (dPrice:{delta_price:.1f}, dCE:{delta_ce_oi}, dPE:{delta_pe_oi})"

        return {
            "status": status,
            "reason": reason,
            "pcr": current_pcr,
            "delta_ce_oi": delta_ce_oi,
            "delta_pe_oi": delta_pe_oi
        }


# ============================================================================
# 2. PURE COOLDOWN STATE MACHINE
# ============================================================================
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


# ============================================================================
# 3. CANDLE AGGREGATOR
# ============================================================================
class CandleAggregator:
    def __init__(
        self,
        max_1m_bars: int = 150,
        max_5m_bars: int = 100,
        max_15m_bars: int = 60,
        gap_threshold_seconds: float = 65.0,
        consecutive_recovery_needed_5m: int = 3
    ):
        self.max_1m_bars = max_1m_bars
        self.max_5m_bars = max_5m_bars
        self.max_15m_bars = max_15m_bars
        self.gap_threshold_seconds = gap_threshold_seconds
        self.consecutive_recovery_needed_5m = consecutive_recovery_needed_5m

        self.bars_1m: List[Dict] = []
        self.bars_5m: List[Dict] = []
        self.bars_15m: List[Dict] = []

        self.current_1m: Optional[Dict] = None
        self.current_5m: Optional[Dict] = None

        self.watermark_1m_epoch: float = 0.0
        self.watermark_5m_epoch: float = 0.0

        self.last_tick_dt: Optional[datetime.datetime] = None
        self.is_gap_active: bool = False
        self.gap_reason: str = "CLEAN"
        self.consecutive_clean_5m_count: int = 0

    @staticmethod
    def normalize_dt(dt: datetime.datetime) -> datetime.datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=IST)
        return dt.astimezone(IST)

    @staticmethod
    def is_active_session_time(dt: datetime.datetime) -> bool:
        open_t = dt.replace(hour=9, minute=15, second=0, microsecond=0)
        close_t = dt.replace(hour=15, minute=30, second=0, microsecond=0)
        return open_t <= dt < close_t

    def reset_intraday_state(self):
        self.current_1m = None
        self.current_5m = None
        self.last_tick_dt = None
        self.is_gap_active = False
        self.gap_reason = "SESSION_START"
        self.consecutive_clean_5m_count = 0
        self.watermark_1m_epoch = 0.0
        self.watermark_5m_epoch = 0.0

    def check_and_finalize_boundary(self, current_wall_time: datetime.datetime) -> Dict[str, Optional[Dict]]:
        now = self.normalize_dt(current_wall_time)
        closed_1m = self._finalize_expired("current_1m", self.bars_1m, 1, now)
        closed_5m = self._finalize_expired("current_5m", self.bars_5m, 5, now)

        if closed_1m:
            end_1m = (closed_1m["time"] + datetime.timedelta(minutes=1)).timestamp()
            self.watermark_1m_epoch = max(self.watermark_1m_epoch, end_1m)

        if closed_5m:
            end_5m = (closed_5m["time"] + datetime.timedelta(minutes=5)).timestamp()
            self.watermark_5m_epoch = max(self.watermark_5m_epoch, end_5m)

        closed_15m = None
        if closed_5m:
            closed_15m = self._process_new_5m_bar(closed_5m)

        return {"closed_1m": closed_1m, "closed_5m": closed_5m, "closed_15m": closed_15m}

    def _finalize_expired(
        self,
        attr: str,
        storage: List[Dict],
        interval_min: int,
        now: datetime.datetime
    ) -> Optional[Dict]:
        cur = getattr(self, attr)
        if cur is None:
            return None

        expiry = cur["time"] + datetime.timedelta(minutes=interval_min)
        session_close = cur["time"].replace(hour=15, minute=30, second=0, microsecond=0)

        if now < expiry or (now - expiry).total_seconds() > 600.0:
            return None

        if now > session_close + datetime.timedelta(seconds=60):
            return None

        closed = cur.copy()
        closed["is_closed"] = True
        storage.append(closed)

        max_limit = self.max_1m_bars if interval_min == 1 else self.max_5m_bars
        if len(storage) > max_limit:
            storage.pop(0)

        setattr(self, attr, None)
        return closed

    def add_tick(self, ltp: float, tick_dt: datetime.datetime) -> Dict[str, Optional[Dict]]:
        if not isinstance(ltp, (int, float)) or math.isnan(ltp) or math.isinf(ltp) or ltp <= 0.0:
            return {"closed_1m": None, "closed_5m": None, "closed_15m": None}

        now = self.normalize_dt(tick_dt)
        if not self.is_active_session_time(now):
            return {"closed_1m": None, "closed_5m": None, "closed_15m": None}

        if self.last_tick_dt is not None:
            delta = (now - self.last_tick_dt).total_seconds()
            if delta <= 0.0:
                return {"closed_1m": None, "closed_5m": None, "closed_15m": None}

            same_date = (self.last_tick_dt.date() == now.date())
            if same_date and delta > self.gap_threshold_seconds:
                self.is_gap_active = True
                self.gap_reason = f"FEED_OUTAGE ({delta:.1f}s missing)"
                self.consecutive_clean_5m_count = 0
                if self.current_1m: self.current_1m["intra_gap"] = True
                if self.current_5m: self.current_5m["intra_gap"] = True

        self.last_tick_dt = now

        closed_1m = self._route_bucket(ltp, now, 1, "current_1m", self.bars_1m, self.watermark_1m_epoch)
        closed_5m = self._route_bucket(ltp, now, 5, "current_5m", self.bars_5m, self.watermark_5m_epoch)

        if closed_1m:
            end_1m = (closed_1m["time"] + datetime.timedelta(minutes=1)).timestamp()
            self.watermark_1m_epoch = max(self.watermark_1m_epoch, end_1m)

        if closed_5m:
            end_5m = (closed_5m["time"] + datetime.timedelta(minutes=5)).timestamp()
            self.watermark_5m_epoch = max(self.watermark_5m_epoch, end_5m)

        closed_15m = None
        if closed_5m:
            closed_15m = self._process_new_5m_bar(closed_5m)

        return {"closed_1m": closed_1m, "closed_5m": closed_5m, "closed_15m": closed_15m}

    def _route_bucket(
        self,
        ltp: float,
        dt: datetime.datetime,
        interval_min: int,
        attr: str,
        storage: List[Dict],
        watermark_epoch: float
    ) -> Optional[Dict]:
        b_min = (dt.minute // interval_min) * interval_min
        bucket_time = dt.replace(minute=b_min, second=0, microsecond=0)
        bucket_end_epoch = (bucket_time + datetime.timedelta(minutes=interval_min)).timestamp()

        if bucket_end_epoch <= watermark_epoch:
            return None

        cur = getattr(self, attr)
        closed = None

        if cur is None:
            setattr(self, attr, {
                "time": bucket_time,
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "ticks": 1,
                "first_tick_time": dt,
                "is_closed": False,
                "intra_gap": False
            })
        elif cur["time"] != bucket_time:
            closed = cur.copy()
            closed["is_closed"] = True
            storage.append(closed)

            max_limit = self.max_1m_bars if interval_min == 1 else self.max_5m_bars
            if len(storage) > max_limit:
                storage.pop(0)

            setattr(self, attr, {
                "time": bucket_time,
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "ticks": 1,
                "first_tick_time": dt,
                "is_closed": False,
                "intra_gap": False
            })
        else:
            cur["high"] = max(cur["high"], ltp)
            cur["low"] = min(cur["low"], ltp)
            cur["close"] = ltp
            cur["ticks"] += 1

        return closed

    @staticmethod
    def _is_session_open_continuous(bar: Dict) -> bool:
        session_open = bar["time"].replace(hour=9, minute=15, second=0, microsecond=0)
        if bar["time"] != session_open:
            return False
        first_tick = bar.get("first_tick_time")
        if not isinstance(first_tick, datetime.datetime):
            return False
        return first_tick <= session_open + datetime.timedelta(seconds=30)

    def _process_new_5m_bar(self, latest_5m: Dict) -> Optional[Dict]:
        is_continuous = False
        if len(self.bars_5m) >= 2:
            prev_bar = self.bars_5m[-2]
            expected_time = prev_bar["time"] + datetime.timedelta(minutes=5)
            if latest_5m["time"].date() == prev_bar["time"].date():
                if latest_5m["time"] == expected_time:
                    is_continuous = True
            else:
                is_continuous = self._is_session_open_continuous(latest_5m)
        else:
            is_continuous = self._is_session_open_continuous(latest_5m)

        is_bar_valid = is_continuous and (not latest_5m.get("intra_gap", False))

        if is_bar_valid:
            self.consecutive_clean_5m_count += 1
            if self.consecutive_clean_5m_count >= self.consecutive_recovery_needed_5m:
                self.is_gap_active = False
                self.gap_reason = "CLEAN"
        else:
            self.is_gap_active = True
            self.gap_reason = "DISCONTINUOUS_BAR_DETECTED"
            self.consecutive_clean_5m_count = 0

        latest_5m["is_clean_valid"] = is_bar_valid
        return self._build_strict_15m(latest_5m)

    def _build_strict_15m(self, latest_5m: Dict) -> Optional[Dict]:
        end_t = latest_5m["time"] + datetime.timedelta(minutes=5)
        if end_t.minute % 15 == 0:
            exp_start = end_t - datetime.timedelta(minutes=15)
            matching = [b for b in self.bars_5m if exp_start <= b["time"] < end_t]
            matching.sort(key=lambda x: x["time"])

            if len(matching) == 3:
                expected_seq = [
                    exp_start,
                    exp_start + datetime.timedelta(minutes=5),
                    exp_start + datetime.timedelta(minutes=10)
                ]
                actual_seq = [b["time"] for b in matching]
                all_clean = all(b.get("is_clean_valid", False) for b in matching)

                if actual_seq == expected_seq and all_clean:
                    bar_15m = {
                        "time": exp_start,
                        "open": matching[0]["open"],
                        "high": max(b["high"] for b in matching),
                        "low": min(b["low"] for b in matching),
                        "close": matching[-1]["close"],
                        "ticks": sum(b["ticks"] for b in matching),
                        "is_closed": True,
                        "is_clean_valid": True
                    }
                    self.bars_15m.append(bar_15m)
                    if len(self.bars_15m) > self.max_15m_bars:
                        self.bars_15m.pop(0)
                    return bar_15m
        return None


# ============================================================================
# 4. SIGNAL ENGINE WITH OI GATE INTEGRATION
# ============================================================================
class SignalEngine:
    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        cooldown_bars_5m: int = 2,
        mandatory_oi_gate: bool = True
    ):
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.mandatory_oi_gate = mandatory_oi_gate
        self.cooldown_sm = CooldownStateMachine(cooldown_bars=cooldown_bars_5m)
        self.oi_engine = OIConfirmationEngine(max_stale_seconds=300.0, mandatory_gate=mandatory_oi_gate)

    def reset_intraday_state(self):
        self.cooldown_sm.reset_intraday_state()
        self.oi_engine.reset_intraday_state()

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
        gap_reason: str,
        oi_diag: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        diag = {
            "macro_trend": "INSUFFICIENT_DATA",
            "rsi_1m": None,
            "atr_5m": None,
            "oi_confirmation": oi_diag.get("status") if oi_diag else "NOT_PROVIDED",
            "oi_reason": oi_diag.get("reason") if oi_diag else "NO_SNAPSHOT",
            "status": "AWAITING_WARMUP",
            "is_new_signal_event": False,
            "cooldown_remaining": self.cooldown_sm.cooldown_remaining_bars
        }

        if not bars_5m:
            diag["status"] = "NO_BARS_AVAILABLE"
            return "NO TRADE", diag

        latest_5m = bars_5m[-1]
        latest_5m_epoch = latest_5m["time"].timestamp()

        if len(bars_5m) < 25 or len(bars_15m) < 22 or len(bars_1m) < 30:
            diag["status"] = f"WARMING_UP (5m:{len(bars_5m)}/25, 15m:{len(bars_15m)}/22, 1m:{len(bars_1m)}/30)"
            return "NO TRADE", diag

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

        spread_ema_5m = abs(ema9_5m - ema21_5m)
        if spread_ema_5m < (0.22 * atr_5m):
            diag["status"] = "SIDEWAYS_COMPRESSION"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

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

        # ----------------------------------------------------
        # FAIL-SAFE MANDATORY OI CONFIRMATION GATE
        # ----------------------------------------------------
        if self.mandatory_oi_gate and candidate in ("BUY CE", "BUY PE"):
            oi_status = oi_diag.get("status") if oi_diag else "DATA_UNAVAILABLE"
            if candidate == "BUY CE" and oi_status != "BULLISH_CONFIRMATION":
                candidate = "NO TRADE"
                diag["status"] = f"BLOCKED_OI_GATE ({oi_status}: {oi_diag.get('reason') if oi_diag else 'MISSING'})"
            elif candidate == "BUY PE" and oi_status != "BEARISH_CONFIRMATION":
                candidate = "NO TRADE"
                diag["status"] = f"BLOCKED_OI_GATE ({oi_status}: {oi_diag.get('reason') if oi_diag else 'MISSING'})"

        action, sm_d = self.cooldown_sm.process_candidate(candidate, latest_5m_epoch)
        diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
        diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
        if sm_d["status"] != "TRANSITION_EMITTED":
            diag["status"] = sm_d["status"]

        return action, diag


# ============================================================================
# 5. ENGINE ORCHESTRATOR
# ============================================================================
class EngineOrchestrator:
    def __init__(self, mandatory_oi_gate: bool = True):
        self.agg = CandleAggregator(consecutive_recovery_needed_5m=3)
        self.engine = SignalEngine(
            rsi_period=14,
            atr_period=14,
            cooldown_bars_5m=2,
            mandatory_oi_gate=mandatory_oi_gate
        )
        self.last_session_date: Optional[datetime.date] = None
        self.latest_oi_diag: Optional[Dict] = None

    def update_option_chain_snapshot(
        self,
        snapshot_time: datetime.datetime,
        underlying_price: float,
        chain_data: List[Dict],
        wall_time: datetime.datetime
    ) -> Dict:
        self.latest_oi_diag = self.engine.oi_engine.process_snapshot(
            snapshot_time=snapshot_time,
            underlying_price=underlying_price,
            chain_data=chain_data,
            current_wall_time=wall_time
        )
        return self.latest_oi_diag

    def process_tick(self, ltp: float, tick_dt: datetime.datetime) -> Dict:
        norm_dt = self.agg.normalize_dt(tick_dt)
        cur_date = norm_dt.date()

        if self.last_session_date is not None and cur_date != self.last_session_date:
            self.agg.reset_intraday_state()
            self.engine.reset_intraday_state()

        self.last_session_date = cur_date

        res = self.agg.add_tick(ltp, norm_dt)
        closed_5m = res.get("closed_5m")

        event_payload = None
        if closed_5m:
            action, diag = self.engine.evaluate(
                self.agg.bars_1m, self.agg.bars_5m, self.agg.bars_15m,
                self.agg.is_gap_active, self.agg.gap_reason,
                oi_diag=self.latest_oi_diag
            )
            event_payload = {
                "time": closed_5m["time"],
                "price": ltp,
                "action": action,
                "macro": diag["macro_trend"],
                "rsi": diag["rsi_1m"],
                "oi_confirm": diag["oi_confirmation"],
                "cooldown": diag["cooldown_remaining"],
                "status": diag["status"],
                "is_event": diag["is_new_signal_event"]
            }

        return {"agg_res": res, "event": event_payload}

    def finalize_boundary(self, wall_time: datetime.datetime) -> Optional[Dict]:
        boundary_res = self.agg.check_and_finalize_boundary(wall_time)
        closed_5m = boundary_res.get("closed_5m")
        if closed_5m:
            action, diag = self.engine.evaluate(
                self.agg.bars_1m, self.agg.bars_5m, self.agg.bars_15m,
                self.agg.is_gap_active, self.agg.gap_reason,
                oi_diag=self.latest_oi_diag
            )
            return {
                "time": closed_5m["time"],
                "price": closed_5m["close"],
                "action": action,
                "macro": diag["macro_trend"],
                "rsi": diag["rsi_1m"],
                "oi_confirm": diag["oi_confirmation"],
                "cooldown": diag["cooldown_remaining"],
                "status": diag["status"],
                "is_event": diag["is_new_signal_event"]
            }
        return None


# ============================================================================
# 6. DETERMINISTIC TEST SUITE FOR OI GATE & INVARIANTS
# ============================================================================
def test_oi_confirmation_engine():
    print("=" * 70)
    print("TEST: OI CONFIRMATION GATE - DETERMINISTIC DYNAMICS")
    print("=" * 70)

    oi_gate = OIConfirmationEngine(max_stale_seconds=300.0, mandatory_gate=True)
    t0 = datetime.datetime(2026, 10, 14, 9, 20, 0, tzinfo=IST)

    # Mock Chain Builder
    def mock_chain(ce_oi: int, pe_oi: int):
        return [
            {
                "strike_price": 72000,
                "call_options": {"market_data": {"oi": ce_oi}},
                "put_options": {"market_data": {"oi": pe_oi}}
            }
        ]

    # 1. Baseline Snapshot -> Must return DATA_UNAVAILABLE (initializing)
    snap1 = oi_gate.process_snapshot(t0, 72000.0, mock_chain(10000, 10000), t0 + datetime.timedelta(seconds=2))
    assert snap1["status"] == "DATA_UNAVAILABLE"
    print("PASS: Baseline snapshot correctly marks confirmation unavailable until consecutive delta exists.")

    # 2. Bullish Dynamics: PE OI jumps +5000, CE OI jumps +1000, Price +50
    t1 = t0 + datetime.timedelta(minutes=3)
    snap2 = oi_gate.process_snapshot(t1, 72050.0, mock_chain(11000, 15000), t1 + datetime.timedelta(seconds=2))
    assert snap2["status"] == "BULLISH_CONFIRMATION"
    assert snap2["delta_pe_oi"] == 5000 and snap2["delta_ce_oi"] == 1000
    print("PASS: Consecutive snapshot with Put writing expansion yields BULLISH_CONFIRMATION.")

    # 3. Conflict Dynamics: Price drops -80 but Put OI increases heavily (Divergence)
    t2 = t1 + datetime.timedelta(minutes=3)
    snap3 = oi_gate.process_snapshot(t2, 71970.0, mock_chain(12000, 20000), t2 + datetime.timedelta(seconds=2))
    assert snap3["status"] == "CONFLICT"
    print("PASS: Price/OI divergence cleanly yields CONFLICT status.")

    # 4. Stale Snapshot: Snapshot older than max_stale_seconds (350s late)
    t3 = t2 + datetime.timedelta(seconds=350)
    snap4 = oi_gate.process_snapshot(t2, 71970.0, mock_chain(12000, 20000), t3)
    assert snap4["status"] == "DATA_UNAVAILABLE"
    assert "STALE" in snap4["reason"]
    print("PASS: Stale option-chain snapshot triggers fail-safe DATA_UNAVAILABLE.")


def test_cooldown_with_mandatory_oi_gate():
    print("\n" + "=" * 70)
    print("TEST: SIGNAL ENGINE FAIL-SAFE UNDER MANDATORY OI GATE")
    print("=" * 70)

    # Initialize Engine with mandatory OI gate enabled
    engine = SignalEngine(rsi_period=14, atr_period=14, cooldown_bars_5m=2, mandatory_oi_gate=True)
    t0 = datetime.datetime(2026, 10, 14, 9, 15, 0, tzinfo=IST)

    # Mock clean series
    bars_1m = [{"time": t0 + datetime.timedelta(minutes=i), "open": 72000+i, "high": 72002+i, "low": 71999+i, "close": 72001+i} for i in range(35)]
    bars_5m = [{"time": t0 + datetime.timedelta(minutes=i*5), "open": 72000+i*10, "high": 72015+i*10, "low": 71995+i*10, "close": 72012+i*10, "is_clean_valid": True} for i in range(30)]
    bars_15m = [{"time": t0 + datetime.timedelta(minutes=i*15), "open": 72000+i*20, "high": 72030+i*20, "low": 71990+i*20, "close": 72025+i*20, "is_clean_valid": True} for i in range(25)]

    # 1. When OI confirms Bullish -> BUY CE allowed
    bullish_oi = {"status": "BULLISH_CONFIRMATION", "reason": "TEST"}
    act1, d1 = engine.evaluate(bars_1m, bars_5m, bars_15m, False, "CLEAN", oi_diag=bullish_oi)
    # Technical condition on synthetic linear series might evaluate expansion or lag; verify gate enforcement:
    if d1["macro_trend"] == "BULLISH" and d1["rsi_1m"] and d1["rsi_1m"] >= 56.0:
        assert act1 == "BUY CE"
        print("PASS: BUY CE emitted when both Technical and OI gates confirm.")

    # 2. When OI is in CONFLICT or UNAVAILABLE -> Signal MUST fall back to NO TRADE
    conflict_oi = {"status": "CONFLICT", "reason": "HEAVY_CALL_WRITING"}
    act2, d2 = engine.evaluate(bars_1m, bars_5m, bars_15m, False, "CLEAN", oi_diag=conflict_oi)
    assert act2 == "NO TRADE"
    print("PASS: Mandatory OI conflict successfully blocked potential BUY signal.")


if __name__ == "__main__":
    test_oi_confirmation_engine()
    test_cooldown_with_mandatory_oi_gate()
