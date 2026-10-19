import time
import os
import csv
import datetime
import zoneinfo
import numpy as np
from typing import Optional, Dict, Tuple

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

class ShadowMomentumEngine:
    """
    Experimental Shadow Module:
    Measures 5s-15s micro-tick velocity, acceleration, and rolling VWAP deviation.
    Emits shadow PRE-ALERTS only. Zero trade execution authority.
    Logs telemetry for MFE/MAE empirical verification.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "shadow_momentum_telemetry.csv")
        self._init_csv()

        # Rolling Tick Buffer: (timestamp, price)
        self.tick_buffer = []  # Max 60 ticks (approx 1-2 mins depending on poll)
        self.cum_vol_price = 0.0
        self.cum_vol = 0.0
        
        # State tracking
        self.active_alert: Optional[Dict] = None
        self.alert_history = []
        self.cooldown_until = 0.0

    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_ist", "symbol", "alert_type", "trigger_price",
                    "velocity_15s", "accel", "vwap_dev_pct",
                    "price_1m", "price_3m", "price_5m",
                    "max_mfe_points", "max_mae_points", "is_false_breakout"
                ])

    def push_tick(self, spot_price: float, timestamp: datetime.datetime, volume: float = 1.0) -> Optional[Dict]:
        """
        Consumes live spot tick, computes micro-dynamics, and evaluates shadow pre-alerts.
        """
        t_now = timestamp.timestamp()
        self.tick_buffer.append((t_now, spot_price))

        # Maintain 60-second rolling tick buffer
        cutoff = t_now - 60.0
        self.tick_buffer = [(t, p) for t, p in self.tick_buffer if t >= cutoff]

        # Rolling Session Micro-VWAP approximation
        self.cum_vol_price += spot_price * volume
        self.cum_vol += volume
        current_vwap = (self.cum_vol_price / self.cum_vol) if self.cum_vol > 0 else spot_price

        # Evaluate pending telemetry log tracking
        self._evaluate_telemetry_tracking(spot_price, t_now)

        # Micro-dynamics require at least 5 ticks
        if len(self.tick_buffer) < 5 or time.time() < self.cooldown_until:
            return None

        # 1. 15-Second Price Velocity (points / second)
        t_15s_ago = t_now - 15.0
        past_ticks = [p for t, p in self.tick_buffer if t <= t_15s_ago]
        ref_price_15s = past_ticks[-1] if past_ticks else self.tick_buffer[0][1]
        dt_15s = max(1.0, t_now - self.tick_buffer[0][0])
        velocity = (spot_price - ref_price_15s) / 15.0

        # 2. Acceleration: Velocity difference over short window
        t_30s_ago = t_now - 30.0
        past_ticks_30s = [p for t, p in self.tick_buffer if t <= t_30s_ago]
        ref_price_30s = past_ticks_30s[-1] if past_ticks_30s else ref_price_15s
        prev_velocity = (ref_price_15s - ref_price_30s) / 15.0
        accel = velocity - prev_velocity

        # 3. Micro-VWAP Deviation
        vwap_dev_pct = ((spot_price - current_vwap) / current_vwap) * 100.0

        # Dynamic Thresholds (Index specific sensitivity)
        # NIFTY typical: velocity > 0.8 pt/sec; SENSEX typical: > 2.5 pt/sec
        vel_thresh = 0.8
        accel_thresh = 0.3

        alert_type = None
        if velocity > vel_thresh and accel > accel_thresh and vwap_dev_pct > 0.02:
            alert_type = "SHADOW_BULLISH_VELOCITY"
        elif velocity < -vel_thresh and accel < -accel_thresh and vwap_dev_pct < -0.02:
            alert_type = "SHADOW_BEARISH_VELOCITY"

        if alert_type:
            event = {
                "timestamp_ist": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "trigger_epoch": t_now,
                "type": alert_type,
                "spot": spot_price,
                "velocity": round(velocity, 2),
                "accel": round(accel, 2),
                "vwap_dev_pct": round(vwap_dev_pct, 4),
                "tracking_active": True,
                "max_favorable": 0.0,
                "max_adverse": 0.0,
                "price_1m": None,
                "price_3m": None,
                "price_5m": None
            }
            self.alert_history.append(event)
            self.active_alert = event
            self.cooldown_until = time.time() + 45.0  # 45s cooldown between shadow triggers
            return event

        return None

    def _evaluate_telemetry_tracking(self, current_price: float, t_now: float):
        """Monitors active alerts to calculate actual MFE and MAE over 1m, 3m, 5m."""
        for alert in self.alert_history:
            if not alert.get("tracking_active", False):
                continue

            elapsed = t_now - alert["trigger_epoch"]
            diff = current_price - alert["spot"]
            if alert["type"] == "SHADOW_BEARISH_VELOCITY":
                diff = -diff  # Invert for puts

            # Track MFE / MAE
            if diff > alert["max_favorable"]:
                alert["max_favorable"] = round(diff, 2)
            if diff < alert["max_adverse"]:
                alert["max_adverse"] = round(diff, 2)

            if elapsed >= 60.0 and alert["price_1m"] is None:
                alert["price_1m"] = current_price
            if elapsed >= 180.0 and alert["price_3m"] is None:
                alert["price_3m"] = current_price
            if elapsed >= 300.0 and alert["price_5m"] is None:
                alert["price_5m"] = current_price
                alert["tracking_active"] = False

                # Save finalized record to CSV
                is_false_breakout = alert["max_favorable"] < 10.0 and alert["max_adverse"] < -15.0
                try:
                    with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            alert["timestamp_ist"], "INDEX", alert["type"], alert["spot"],
                            alert["velocity"], alert["accel"], alert["vwap_dev_pct"],
                            alert["price_1m"], alert["price_3m"], alert["price_5m"],
                            alert["max_favorable"], alert["max_adverse"], is_false_breakout
                        ])
                except Exception:
                    pass
