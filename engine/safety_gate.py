from typing import Tuple, Optional
from core.types import MarketDirection, OptionType, NormalizedOptionTick
from core.black_scholes import OptionGreeks

class ZeroTrustFinalSafetyGate:
    @staticmethod
    def verify_execution(
        market_direction: MarketDirection,
        tick: NormalizedOptionTick,
        greeks: Optional[OptionGreeks],
        candidate: Optional[object],
        candidate_ready: bool,
        ttl_state: object,
        quality_grade: str = "GRADE_A",
        max_spread_pct: float = 0.02
    ) -> Tuple[bool, str]:
        """
        Zero-Trust Hard Validation Gate.
        Returns (passed: bool, message: str).
        """
        # 1. Direction Check
        if market_direction == MarketDirection.NEUTRAL:
            return False, "BLOCKED: Market regime is NEUTRAL"

        # 2. Candidate Check
        if candidate is None:
            return False, "BLOCKED: Candidate is None (Scanning)"
        if not candidate_ready:
            return False, f"BLOCKED: Candidate confirmation pending ({getattr(candidate, 'confirmation_count', 0)}/3)"

        # 3. Inversion Guard
        if market_direction == MarketDirection.BULLISH and tick.option_type != OptionType.CE:
            return False, "BLOCKED_INVERSION: Bullish regime cannot execute Put (PE)"
        if market_direction == MarketDirection.BEARISH and tick.option_type != OptionType.PE:
            return False, "BLOCKED_INVERSION: Bearish regime cannot execute Call (CE)"

        # 4. TTL Check
        if ttl_state is None or not getattr(ttl_state, "is_valid", lambda: False)():
            return False, "BLOCKED: Tick Pulse TTL expired or dead"

        # 5. Greeks Validation
        if greeks is None:
            return False, "BLOCKED: Greeks are missing or non-finite"

        abs_delta = abs(greeks.delta)
        if abs_delta < 0.30:
            return False, f"BLOCKED_DELTA: Delta {abs_delta:.3f} is too deep OTM (min 0.30)"
        if abs_delta > 0.70:
            return False, f"BLOCKED_DELTA: Delta {abs_delta:.3f} is too deep ITM (max 0.70)"

        # 6. Spread Check
        if tick.ltp > 0.0:
            calculated_spread = tick.ask - tick.bid
            if calculated_spread < 0:
                return False, "BLOCKED_SPREAD: Inverted spread (ask < bid)"
            spread_pct = calculated_spread / tick.ltp
            if spread_pct > max_spread_pct:
                return False, f"BLOCKED_SPREAD: Spread {spread_pct*100:.2f}% exceeds max {max_spread_pct*100:.1f}%"

        return True, "PASSED_ALL_SAFETY_GATES"
