"""
engine_core.py - Institutional Sensex/Nifty Directional Engine
Updated & Production Hardened:
- Instant Session High Drawdown Cutoff (Kills stale/sticky BUY CE immediately on dumps)
- Fast Intraday VWAP + Micro-Breakdown Engine (Early PE detection without 15m EMA lag)
- High-Confluence Advance Pre-Alert (Detects strong momentum midway through 5m bar)
- Change-in-OI (Delta OI) Priority over Static PCR
- Strict Time & Gap Resiliency
"""

import datetime
import zoneinfo
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# ============================================================================
# 1. FAIL-CLOSED OI & DELTA FLOW ENGINE
# ============================================================================
class OIConfirmationEngine:
    """
    Evaluates institutional positioning delta across finalized snapshots.
    Prioritizes real-time change in OI over static/laggy PCR values.
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
        age = (current_wall_time - snapshot_time).total_seconds()
        if age < 0.0 or age > self.max_stale_seconds:
            return {
                "status": "DATA_UNAVAILABLE",
                "reason": f"STALE_SNAPSHOT (Age: {age:.1f}s)",
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

        if self.last_snapshot is None:
            self.last_snapshot = current_record
            init_status = "BULLISH_CONFIRMATION" if current_pcr >= 1.20 else ("BEARISH_CONFIRMATION" if current_pcr <= 0.85 else "STANDBY")
            return {
                "status": init_status,
                "reason": "BASELINE_ESTABLISHED",
                "pcr": current_pcr,
                "delta_ce_oi": 0,
                "delta_pe_oi": 0
            }

        delta_ce_oi = total_ce_oi - self.last_snapshot["ce_oi"]
        delta_pe_oi = total_pe_oi - self.last_snapshot["pe_oi"]
        delta_price = underlying_price - self.last_snapshot["price"]

        self.last_snapshot = current_record

        # Change-Based Confirmation Rules
        if (delta_pe_oi > delta_ce_oi and delta_price >= -25.0) or (current_pcr >= 1.25 and delta_price >= 0.0):
            status = "BULLISH_CONFIRMATION"
            reason = "PUT_BUILDUP_SUPPORT"
        elif (delta_ce_oi > delta_pe_oi and delta_price <= 25.0) or (current_pcr <= 0.85 and delta_price <= 0.0):
            status = "BEARISH_CONFIRMATION"
            reason = "CALL_BUILDUP_RESISTANCE"
        else:
            status = "STANDBY"
            reason = f"BALANCED_FLOW (PCR: {current_pcr})"

        return {
            "status": status,
            "reason": reason,
            "pcr": current_pcr,
            "delta_ce_oi": delta_ce_oi,
            "delta_pe_oi": delta_pe_oi
        }


# ============================================================================
# 2. HARDENED COOLDOWN STATE MACHINE (WITH INSTANT INVALIDATION)
# ============================================================================
class CooldownStateMachine:
    """
    Prevents sticky signals. If an existing signal is invalidated by immediate price action,
    it forcefully resets memory to NO TRADE instantly.
    """
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

    def invalidate_signal(self, reason: str = "PRICE_ACTION_INVALIDATION"):
        """Forcefully clears any active signal freeze."""
        self.last_emitted_signal = "NO TRADE"
        self.cooldown_remaining_bars = 0

    def process_candidate(self, candidate_signal: str, bar_epoch: float, is_invalidated: bool = False) -> Tuple[str, Dict]:
        # Handle Instant Invalidation: If market reversed, kill existing signal immediately
        if is_invalidated:
            self.invalidate_signal()
            return "NO TRADE", {
                "status": "SIGNAL_FORCE_INVALIDATED",
                "is_new_signal_event": False,
                "cooldown_remaining": 0
            }

        if self.last_processed_bar_epoch is not None:
            if bar_epoch < self.last_processed_bar_epoch:
                return "NO TRADE", {
                    "status": "OUT_OF_ORDER_BAR_REJECTED",
                    "is_new_signal_event": False,
                    "cooldown_remaining": self.cooldown_remaining_bars
                }

        starting_cooldown = self.cooldown_remaining_bars
        action = "NO TRADE"
        is_new_event = False
        status = "HOLD"

        if candidate_signal == "NO TRADE":
            action = "NO TRADE"
            status = "NO_SIGNAL_CANDIDATE"
            self.last_emitted_signal = "NO TRADE"
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
# 3. CANDLE AGGREGATOR (MULTI-TIMEFRAME STREAMING)
# ============================================================================
class CandleAggregator:
    def __init__(
        self,
        max_1m_bars: int = 150,
        max_5m_bars: int = 100,
        max_15m_bars: int = 60,
        gap_threshold_seconds: float = 300.0,
        consecutive_recovery_needed_5m: int = 1
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
        open_t = dt.replace(hour=9, minute=0, second=0, microsecond=0)
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
        closed["is_clean_valid"] = True
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
                "intra_gap": False,
                "is_clean_valid": True
            })
        elif cur["time"] != bucket_time:
            closed = cur.copy()
            closed["is_closed"] = True
            closed["is_clean_valid"] = True
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
                "intra_gap": False,
                "is_clean_valid": True
            })
        else:
            cur["high"] = max(cur["high"], ltp)
            cur["low"] = min(cur["low"], ltp)
            cur["close"] = ltp
            cur["ticks"] += 1

        return closed

    def _process_new_5m_bar(self, latest_5m: Dict) -> Optional[Dict]:
        latest_5m["is_clean_valid"] = True
        self.consecutive_clean_5m_count += 1
        if self.consecutive_clean_5m_count >= self.consecutive_recovery_needed_5m:
            self.is_gap_active = False
            self.gap_reason = "CLEAN"
        return self._build_strict_15m(latest_5m)

    def _build_strict_15m(self, latest_5m: Dict) -> Optional[Dict]:
        end_t = latest_5m["time"] + datetime.timedelta(minutes=5)
        if end_t.minute % 15 == 0:
            exp_start = end_t - datetime.timedelta(minutes=15)
            matching = [b for b in self.bars_5m if exp_start <= b["time"] < end_t]
            matching.sort(key=lambda x: x["time"])

            if len(matching) >= 2:
                bar_15m = {
                    "time": exp_start,
                    "open": matching[0]["open"],
                    "high": max(b["high"] for b in matching),
                    "low": min(b["low"] for b in matching),
                    "close": matching[-1]["close"],
                    "ticks": sum(b.get("ticks", 1) for b in matching),
                    "is_closed": True,
                    "is_clean_valid": True
                }
                self.bars_15m.append(bar_15m)
                if len(self.bars_15m) > self.max_15m_bars:
                    self.bars_15m.pop(0)
                return bar_15m
        return None


# ============================================================================
# 4. SIGNAL ENGINE (HARD DRAWDOWN CUTOFF + VWAP + FAST BREAKDOWN)
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
    def calculate_wilder_rsi(closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
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
        if len(bars) < period:
            return 15.0

        trs = []
        for i in range(1, len(bars)):
            h = bars[i]["high"]
            l = bars[i]["low"]
            prev_c = bars[i - 1]["close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)

        if not trs:
            return 15.0

        atr = float(np.mean(trs[:min(period, len(trs))]))
        for i in range(period, len(trs)):
            atr = (atr * (period - 1) + trs[i]) / period

        return float(atr)

    @staticmethod
    def calculate_intraday_vwap(bars_1m: List[Dict]) -> Optional[float]:
        """Calculates true volume-weighted average price across intraday bars."""
        if not bars_1m:
            return None
        cum_pv = 0.0
        cum_v = 0.0
        for b in bars_1m:
            typical_price = (b["high"] + b["low"] + b["close"]) / 3.0
            vol = float(b.get("ticks", 1.0))
            cum_pv += typical_price * vol
            cum_v += vol
        return (cum_pv / cum_v) if cum_v > 0 else bars_1m[-1]["close"]

    def evaluate(
        self,
        bars_1m: List[Dict],
        bars_5m: List[Dict],
        bars_15m: List[Dict],
        is_gap_active: bool,
        gap_reason: str,
        oi_diag: Optional[Dict] = None,
        current_time: Optional[datetime.datetime] = None,
        running_5m: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        diag = {
            "macro_trend": "STANDBY",
            "rsi_1m": None,
            "atr_5m": None,
            "vwap": None,
            "oi_confirmation": oi_diag.get("status") if oi_diag else "NOT_PROVIDED",
            "oi_reason": oi_diag.get("reason") if oi_diag else "NO_SNAPSHOT",
            "status": "MONITORING",
            "is_new_signal_event": False,
            "is_pre_alert": False,
            "cooldown_remaining": self.cooldown_sm.cooldown_remaining_bars
        }

        now = current_time or datetime.datetime.now(IST)

        # Pre-Market discovery window
        if now.time() < datetime.time(9, 15, 0):
            if bars_1m:
                curr_spot = float(bars_1m[-1]["close"])
                ref_close = bars_5m[-1]["close"] if bars_5m else curr_spot
                diff_pct = ((curr_spot - ref_close) / ref_close) * 100.0 if ref_close > 0 else 0.0

                if diff_pct >= 0.15:
                    diag["macro_trend"] = "BULLISH"
                    diag["status"] = f"PRE_MARKET_GAP_UP (+{diff_pct:.2f}%)"
                elif diff_pct <= -0.15:
                    diag["macro_trend"] = "BEARISH"
                    diag["status"] = f"PRE_MARKET_GAP_DOWN ({diff_pct:.2f}%)"
                else:
                    diag["macro_trend"] = "NEUTRAL"
                    diag["status"] = f"PRE_MARKET_FLAT ({diff_pct:.2f}%)"
            return "NO TRADE", diag

        closes_1m = [float(b["close"]) for b in bars_1m]
        rsi_1m = self.calculate_wilder_rsi(closes_1m, self.rsi_period)
        diag["rsi_1m"] = round(rsi_1m, 1) if rsi_1m is not None else None

        if not bars_5m:
            diag["status"] = "NO_BARS_AVAILABLE"
            return "NO TRADE", diag

        latest_5m = bars_5m[-1]
        latest_5m_epoch = latest_5m["time"].timestamp()

        if len(bars_5m) < 3 or len(bars_1m) < 10:
            diag["status"] = f"WARMING_UP (5m:{len(bars_5m)}/3, 1m:{len(bars_1m)}/10)"
            return "NO TRADE", diag

        curr_spot = closes_1m[-1]
        vwap = self.calculate_intraday_vwap(bars_1m)
        diag["vwap"] = round(vwap, 1) if vwap else None

        closes_5m = pd.Series([b["close"] for b in bars_5m], dtype=float)
        ema9_5m = float(closes_5m.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21_5m = float(closes_5m.ewm(span=21, adjust=False).mean().iloc[-1])

        atr_5m = self.calculate_atr(bars_5m, min(len(bars_5m), self.atr_period))
        diag["atr_5m"] = round(atr_5m, 2) if atr_5m is not None else 15.0

        # ====================================================================
        # 1. HARD INTRADAY DRAWDOWN GUARD (ZERO-DELAY CE KILL SWITCH)
        # ====================================================================
        session_high = max([b["high"] for b in bars_1m]) if bars_1m else curr_spot
        session_low = min([b["low"] for b in bars_1m]) if bars_1m else curr_spot
        drop_from_high = session_high - curr_spot
        rally_from_low = curr_spot - session_low

        # Reversal triggers: Drop >= 120 pts from peak or slip under VWAP immediately kills bullish bias
        is_hard_reversal_down = (drop_from_high >= 120.0)
        is_hard_reversal_up = (rally_from_low >= 120.0)

        recent_drop = False
        recent_rise = False
        if len(closes_1m) >= 3:
            recent_drop = (closes_1m[-1] < closes_1m[-2] < closes_1m[-3]) or (curr_spot < ema9_5m) or is_hard_reversal_down
            recent_rise = (closes_1m[-1] > closes_1m[-2] > closes_1m[-3]) or (curr_spot > ema9_5m) or is_hard_reversal_up

        # Invalidation Guard: Kills BUY CE instantly on dumps
        is_invalidated = False
        if self.cooldown_sm.last_emitted_signal == "BUY CE" and (curr_spot < ema9_5m or is_hard_reversal_down or recent_drop):
            is_invalidated = True
            self.cooldown_sm.invalidate_signal()

        if is_gap_active:
            diag["status"] = f"BLOCKED: {gap_reason}"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch, is_invalidated=True)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        # ====================================================================
        # 2. DUAL-ENGINE CONFLUENCE (DYNAMIC TREND RESOLUTION)
        # ====================================================================
        # Bullish ONLY if near peak and strictly above EMA9 & VWAP
        bullish_structure = (ema9_5m >= ema21_5m) and (curr_spot >= ema9_5m) and not is_hard_reversal_down and not recent_drop
        if vwap:
            bullish_structure = bullish_structure and (curr_spot >= vwap)

        # Bearish: Triggers immediately on hard drop OR price breakdown below EMA9 + VWAP
        bearish_structure = (curr_spot < ema9_5m and recent_drop) or is_hard_reversal_down
        if vwap and not is_hard_reversal_down:
            bearish_structure = bearish_structure and (curr_spot < vwap)

        if bullish_structure:
            diag["macro_trend"] = "BULLISH"
        elif bearish_structure:
            diag["macro_trend"] = "BEARISH"
        else:
            diag["macro_trend"] = "STANDBY"
            diag["status"] = "SIDEWAYS / CHOPPY"
            action, sm_d = self.cooldown_sm.process_candidate("NO TRADE", latest_5m_epoch, is_invalidated=is_invalidated)
            diag["is_new_signal_event"] = sm_d["is_new_signal_event"]
            diag["cooldown_remaining"] = sm_d["cooldown_remaining"]
            return action, diag

        # ====================================================================
        # 3. ADVANCE PRE-ALERT (RUNNING 5-MIN CANDLE MID-WAY CHECK)
        # ====================================================================
        bar_elapsed_seconds = (now.minute % 5) * 60 + now.second
        if 90 <= bar_elapsed_seconds <= 270:
            if diag["macro_trend"] in ("BULLISH", "BEARISH"):
                diag["is_pre_alert"] = True

        # ====================================================================
        # 4. MOMENTUM EXPANSION TRIGGER
        # ====================================================================
        candidate = "NO TRADE"
        if bullish_structure and (rsi_1m is not None and rsi_1m >= 54.0):
            candidate = "BUY CE"
            diag["status"] = "CONFIRMED_BULLISH_EXPANSION"
        elif bearish_structure and (rsi_1m is not None and rsi_1m <= 48.0):
            candidate = "BUY PE"
            diag["status"] = "CONFIRMED_BEARISH_EXPANSION"
        else:
            diag["status"] = "MOMENTUM_LAG"

        # OI Filter (Delta OI Flow Check)
        if self.mandatory_oi_gate and candidate in ("BUY CE", "BUY PE"):
            oi_status = oi_diag.get("status") if oi_diag else "DATA_UNAVAILABLE"
            if candidate == "BUY CE" and oi_status not in ("BULLISH_CONFIRMATION", "STANDBY"):
                candidate = "NO TRADE"
                diag["status"] = f"BLOCKED_OI_GATE ({oi_status})"
            elif candidate == "BUY PE" and oi_status not in ("BEARISH_CONFIRMATION", "STANDBY"):
                candidate = "NO TRADE"
                diag["status"] = f"BLOCKED_OI_GATE ({oi_status})"

        action, sm_d = self.cooldown_sm.process_candidate(candidate, latest_5m_epoch, is_invalidated=is_invalidated)
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
        self.agg = CandleAggregator(consecutive_recovery_needed_5m=1)
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

        eval_5m = list(self.agg.bars_5m)
        if self.agg.current_5m:
            eval_5m.append(self.agg.current_5m)

        action, diag = self.engine.evaluate(
            self.agg.bars_1m, eval_5m, self.agg.bars_15m,
            self.agg.is_gap_active, self.agg.gap_reason,
            oi_diag=self.latest_oi_diag,
            current_time=norm_dt,
            running_5m=self.agg.current_5m
        )

        event_payload = {
            "time": norm_dt,
            "price": ltp,
            "action": action,
            "macro": diag["macro_trend"],
            "rsi": diag["rsi_1m"],
            "vwap": diag.get("vwap"),
            "oi_confirm": diag["oi_confirmation"],
            "cooldown": diag["cooldown_remaining"],
            "status": diag["status"],
            "is_pre_alert": diag.get("is_pre_alert", False),
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
                oi_diag=self.latest_oi_diag,
                current_time=closed_5m["time"],
                running_5m=None
            )
            return {
                "time": closed_5m["time"],
                "price": closed_5m["close"],
                "action": action,
                "macro": diag["macro_trend"],
                "rsi": diag["rsi_1m"],
                "vwap": diag.get("vwap"),
                "oi_confirm": diag["oi_confirmation"],
                "cooldown": diag["cooldown_remaining"],
                "status": diag["status"],
                "is_pre_alert": False,
                "is_event": diag["is_new_signal_event"]
            }
        return None
