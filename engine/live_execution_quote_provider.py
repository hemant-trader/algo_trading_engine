import time
import datetime
import zoneinfo
import requests
import math
import urllib.parse
from typing import Optional, Dict, Tuple
from core.types import OptionType, NormalizedOptionTick

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

class LiveExecutionQuoteProvider:
    def __init__(self, max_stale_seconds: float = 3.5, max_clock_skew_seconds: float = 0.5):
        self.max_stale_seconds = max_stale_seconds
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self._prev_quotes: Dict[str, Dict] = {}

    def _parse_broker_timestamp(self, raw_val: Optional[object]) -> Optional[float]:
        """Strict parser supporting millisecond int/float, epoch strings, and ISO-8601."""
        if raw_val is None:
            return None
        try:
            if isinstance(raw_val, (int, float)):
                ts = float(raw_val)
                return ts / 1000.0 if ts > 1e11 else ts

            text = str(raw_val).strip()
            if text.isdigit():
                ts = float(text)
                return ts / 1000.0 if ts > 1e11 else ts

            dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError, OverflowError):
            return None

    def _validate_quote_timestamp(self, raw_val: Optional[object]) -> Tuple[bool, Optional[float], str]:
        """Isolated deterministic timestamp verification."""
        parsed = self._parse_broker_timestamp(raw_val)
        if parsed is None:
            return False, None, "FAIL_CLOSED_TIMESTAMP_MISSING"

        now_ts = time.time()
        tick_age = now_ts - parsed

        if tick_age < -self.max_clock_skew_seconds:
            return False, parsed, f"FAIL_CLOSED_NEGATIVE_AGE_SKEW ({tick_age:.2f}s)"

        if tick_age > self.max_stale_seconds:
            return False, parsed, f"FAIL_CLOSED_STALE_QUOTE ({tick_age:.2f}s > {self.max_stale_seconds}s)"

        return True, parsed, "TIMESTAMP_VALID"

    def fetch_execution_quote(
        self,
        instrument_key: str,
        symbol: str,
        underlying: str,
        underlying_spot: float,
        expiry: str,
        strike: float,
        option_type: OptionType,
        token: str,
        tte: float,
        seq_no: int,
        iv_decimal: float
    ) -> Tuple[Optional[NormalizedOptionTick], str]:
        """
        Fetches an isolated, real-time market-data quote for hard execution validation.
        """
        encoded_key = urllib.parse.quote(instrument_key)
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}

        try:
            resp = requests.get(url, headers=headers, timeout=2.5)
            if resp.status_code != 200:
                return None, f"FAIL_CLOSED_HTTP_{resp.status_code}"

            res_json = resp.json()
            if res_json.get("status") != "success" or "data" not in res_json:
                return None, "FAIL_CLOSED_MALFORMED_RESPONSE"

            data_dict = res_json.get("data", {})
            if not data_dict:
                return None, "FAIL_CLOSED_EMPTY_QUOTE_DATA"

            # Multi-pattern resilient key lookup
            key_colon = instrument_key.replace("|", ":")
            key_pipe = instrument_key.replace(":", "|")
            
            quote_data = (
                data_dict.get(instrument_key) or 
                data_dict.get(key_colon) or 
                data_dict.get(key_pipe)
            )

            # Fallback agar API response dictionary me single element ho
            if not quote_data and len(data_dict) == 1:
                quote_data = next(iter(data_dict.values()))

            if not quote_data:
                return None, "FAIL_CLOSED_INSTRUMENT_KEY_NOT_FOUND"

            # Deterministic Timestamp Gate
            ts_valid, broker_timestamp, ts_msg = self._validate_quote_timestamp(quote_data.get("timestamp"))
            if not ts_valid:
                return None, ts_msg

            depth = quote_data.get("depth", {})
            buy_depth = depth.get("buy", [{}])
            sell_depth = depth.get("sell", [{}])

            live_ltp = float(quote_data.get("last_price", 0.0))
            live_bid = float(buy_depth[0].get("price", 0.0)) if buy_depth else 0.0
            live_ask = float(sell_depth[0].get("price", 0.0)) if sell_depth else 0.0
            live_bid_qty = int(buy_depth[0].get("quantity", 0)) if buy_depth else 0
            live_ask_qty = int(sell_depth[0].get("quantity", 0)) if sell_depth else 0
            live_vol = int(quote_data.get("volume", 0))
            live_oi = int(quote_data.get("oi", 0))

            # Numeric Sanity & Orderbook Inversion Guard
            numeric_values = {
                "ltp": live_ltp,
                "bid": live_bid,
                "ask": live_ask,
            }
            for name, value in numeric_values.items():
                if not math.isfinite(value):
                    return None, f"FAIL_CLOSED_NON_FINITE_{name.upper()}"

            if live_ltp <= 0.0:
                return None, "FAIL_CLOSED_INVALID_LTP"

            if live_bid < 0.0 or live_ask < 0.0:
                return None, "FAIL_CLOSED_INVALID_BID_ASK"

            if live_ask < live_bid:
                return None, "FAIL_CLOSED_CROSSED_BOOK"

            # Unknown != Zero
            prev = self._prev_quotes.get(instrument_key)
            if prev is not None:
                vol_change = live_vol - prev["volume"]
                oi_change = live_oi - prev["oi"]
                price_change = round(live_ltp - prev["ltp"], 2)
                price_change_pct = round((price_change / prev["ltp"]) * 100.0, 2) if prev["ltp"] > 0 else 0.0
                oi_change_pct = round((oi_change / prev["oi"]) * 100.0, 2) if prev["oi"] > 0 else 0.0
                prev_oi_val = prev["oi"]
            else:
                vol_change = None
                oi_change = None
                price_change = None
                price_change_pct = None
                oi_change_pct = None
                prev_oi_val = None

            self._prev_quotes[instrument_key] = {
                "ltp": live_ltp,
                "volume": live_vol,
                "oi": live_oi,
                "timestamp": broker_timestamp
            }

            constructed_tick = NormalizedOptionTick(
                symbol=symbol,
                instrument_key=instrument_key,
                underlying=underlying,
                underlying_spot=underlying_spot,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                ltp=live_ltp,
                bid=live_bid,
                ask=live_ask,
                spread=round(max(0.0, live_ask - live_bid), 2),
                bid_qty=live_bid_qty,
                ask_qty=live_ask_qty,
                volume=live_vol,
                oi=live_oi,
                previous_oi=prev_oi_val,
                volume_change=vol_change,
                oi_change=oi_change,
                price_change=price_change,
                price_change_pct=price_change_pct,
                oi_change_pct=oi_change_pct,
                iv=iv_decimal,
                timestamp=broker_timestamp,
                sequence_no=seq_no,
                tte=tte,
                data_source="UPSTOX_LIVE_REST_QUOTE"
            )

            return constructed_tick, "QUOTE_FRESH_OK"

        except Exception as e:
            return None, f"FAIL_CLOSED_EXCEPTION: {str(e)}"
