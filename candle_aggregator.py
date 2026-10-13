"""
candle_aggregator.py - Strict Time-Boundary OHLC Aggregator
"""

import datetime
import zoneinfo
import math
from typing import List, Dict, Optional

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

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
        # Strictly enforces 09:15:00 <= dt < 15:30:00
        open_t = dt.replace(hour=9, minute=15, second=0, microsecond=0)
        close_t = dt.replace(hour=15, minute=30, second=0, microsecond=0)
        return open_t <= dt < close_t

    def reset_intraday_state(self):
        """Purges active in-flight buckets; retains completed history for indicator warm-up."""
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

        # Boundary Guards:
        # 1. Do not finalize if clock has not reached expiry
        # 2. Reject future clock jump (> 10 minutes beyond expiry)
        # 3. Reject finalization if wall time is strictly after session close (+60s tolerance)
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
