import math
from typing import Dict, Any, List, Tuple
from core.types import OptionType

class ContractIdentityAuthority:
    @staticmethod
    def verify_against_broker_universe(
        instrument_key: str,
        expected_underlying_key: str,
        expected_strike: float,
        expected_type: OptionType,
        expected_expiry: str,
        active_contracts: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Independent cross-verification against broker-provided active contract universe.
        Enforces 5-point immutable identity matching with zero exception leakage.
        """
        match = next((c for c in active_contracts if c.get("instrument_key") == instrument_key), None)
        if not match:
            return False, "FAIL_CLOSED_INSTRUMENT_KEY_NOT_IN_UNIVERSE"

        # 1. Fail-Closed Strike Parsing
        try:
            broker_strike = float(match.get("strike_price"))
        except (TypeError, ValueError):
            return False, "FAIL_CLOSED_INVALID_BROKER_STRIKE"

        if not math.isfinite(broker_strike):
            return False, "FAIL_CLOSED_NON_FINITE_BROKER_STRIKE"

        # 2. String Presence & Sanitization
        broker_underlying = str(match.get("underlying_key") or "").strip()
        broker_type_str = str(match.get("instrument_type") or "").strip().upper()
        broker_expiry = str(match.get("expiry") or "").strip()

        if not broker_underlying:
            return False, "FAIL_CLOSED_MISSING_BROKER_UNDERLYING"
        if not broker_type_str:
            return False, "FAIL_CLOSED_MISSING_BROKER_INSTRUMENT_TYPE"
        if not broker_expiry:
            return False, "FAIL_CLOSED_MISSING_BROKER_EXPIRY"

        # 3. 5-Point Immutable Cross-Verification
        if broker_underlying != expected_underlying_key:
            return False, f"IDENTITY_MISMATCH_UNDERLYING (Broker: {broker_underlying} != Exp: {expected_underlying_key})"

        if broker_strike != expected_strike:
            return False, f"IDENTITY_MISMATCH_STRIKE (Broker: {broker_strike} != Exp: {expected_strike})"

        expected_type_str = "CE" if expected_type == OptionType.CE else "PE"
        if broker_type_str != expected_type_str:
            return False, f"IDENTITY_MISMATCH_TYPE (Broker: {broker_type_str} != Exp: {expected_type_str})"

        if broker_expiry != expected_expiry:
            return False, f"IDENTITY_MISMATCH_EXPIRY (Broker: {broker_expiry} != Exp: {expected_expiry})"

        return True, "CONTRACT_IDENTITY_VERIFIED"
