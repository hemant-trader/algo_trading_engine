import time
import datetime
import zoneinfo
import urllib.parse
import requests
from typing import Optional, List, Dict, Any, Tuple

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

class FailClosedExpiryManager:
    def __init__(
        self,
        cache_ttl_seconds: float = 3600.0,
        same_day_expiry_cutoff: datetime.time = datetime.time(15, 30)
    ):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.same_day_expiry_cutoff = same_day_expiry_cutoff
        
        # Schema: {instrument_key: {"instrument_key": str, "expiry": str, "contracts": list, "fetched_at": float}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Tracks last acknowledged active expiry per instrument to detect rotations
        self._last_active_expiry: Dict[str, str] = {}

    def flush_cache(self, instrument_key: Optional[str] = None) -> None:
        """Explicit cache flush."""
        if instrument_key:
            self._cache.pop(instrument_key, None)
            self._last_active_expiry.pop(instrument_key, None)
        else:
            self._cache.clear()
            self._last_active_expiry.clear()

    def get_active_expiry_and_contracts(
        self,
        instrument_key: str,
        token: str
    ) -> Tuple[Optional[str], List[Dict[str, Any]], bool, str]:
        """
        Fail-Closed Contract & Expiry Resolver.
        Returns: (expiry_date_str, contracts_list, expiry_rotated: bool, status_message)
        """
        now_ts = time.time()
        now_ist = datetime.datetime.now(IST)
        today_iso = now_ist.strftime("%Y-%m-%d")

        # -------------------------------------------------------------
        # 1. Cache Hit Verification
        # -------------------------------------------------------------
        cached_entry = self._cache.get(instrument_key)
        if cached_entry and cached_entry.get("instrument_key") == instrument_key:
            is_fresh = (now_ts - cached_entry["fetched_at"]) < self.cache_ttl_seconds
            cached_exp = cached_entry["expiry"]
            
            is_valid_expiry = cached_exp >= today_iso
            if cached_exp == today_iso and now_ist.time() >= self.same_day_expiry_cutoff:
                is_valid_expiry = False

            if is_fresh and is_valid_expiry:
                last_known = self._last_active_expiry.get(instrument_key)
                rotated = (last_known is not None and last_known != cached_exp)
                self._last_active_expiry[instrument_key] = cached_exp
                return cached_exp, cached_entry["contracts"], rotated, "CACHE_HIT_VALID"

        # -------------------------------------------------------------
        # 2. Strict Network Fetch (No stale emergency fallback)
        # -------------------------------------------------------------
        try:
            encoded_key = urllib.parse.quote(instrument_key)
            url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_key}"
            headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
            
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code != 200:
                self._cache.pop(instrument_key, None)
                return None, [], False, f"FAIL_CLOSED_BROKER_HTTP_{resp.status_code}"

            res_json = resp.json()
            if res_json.get("status") != "success" or "data" not in res_json:
                self._cache.pop(instrument_key, None)
                return None, [], False, "FAIL_CLOSED_MALFORMED_PAYLOAD"

            contracts = res_json.get("data", [])
            if not contracts:
                self._cache.pop(instrument_key, None)
                return None, [], False, "FAIL_CLOSED_EMPTY_CONTRACTS_LIST"

            # -------------------------------------------------------------
            # 3. Dynamic Expiry Resolution & Cutoff Filtering
            # -------------------------------------------------------------
            valid_expiries = sorted({
                c["expiry"] for c in contracts 
                if c.get("expiry") and c["expiry"] >= today_iso
            })

            # Apply configurable same-day cutoff
            if valid_expiries and valid_expiries[0] == today_iso:
                if now_ist.time() >= self.same_day_expiry_cutoff:
                    valid_expiries.pop(0)

            if not valid_expiries:
                self._cache.pop(instrument_key, None)
                return None, [], False, "FAIL_CLOSED_NO_FUTURE_EXPIRIES_FOUND"

            resolved_expiry = valid_expiries[0]

            # Filter contracts strictly for this index and resolved expiry
            target_contracts = [
                c for c in contracts if c.get("expiry") == resolved_expiry
            ]

            if not target_contracts:
                self._cache.pop(instrument_key, None)
                self._last_active_expiry.pop(instrument_key, None)
                return None, [], False, "FAIL_CLOSED_NO_CONTRACTS_FOR_RESOLVED_EXPIRY"

            # Rotation Detection: Compare with last acknowledged expiry
            last_known = self._last_active_expiry.get(instrument_key)
            expiry_rotated = (last_known is not None and last_known != resolved_expiry)
            self._last_active_expiry[instrument_key] = resolved_expiry

            # Store verified entry in cache
            self._cache[instrument_key] = {
                "instrument_key": instrument_key,
                "expiry": resolved_expiry,
                "contracts": target_contracts,
                "fetched_at": now_ts
            }

            return resolved_expiry, target_contracts, expiry_rotated, "FETCH_SUCCESS"

        except Exception as e:
            self._cache.pop(instrument_key, None)
            return None, [], False, f"FAIL_CLOSED_EXCEPTION: {str(e)}"
