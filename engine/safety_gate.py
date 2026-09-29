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
        max_spread_pct: float = 0.03,
        min_delta: float = 0.45,
        max_delta: float = 0.65
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
            conf_count = getattr(candidate, "confirmation_count", 0)
            return False, f"BLOCKED_CONFIRMATION_PENDING: Stability verification incomplete ({conf_count})"

        # 3. Inversion Guard
        if market_direction == MarketDirection.BULLISH and tick.option_type != OptionType.CE:
            return False, "BLOCKED_INVERSION: Bullish regime cannot execute Put (PE)"
        if market_direction == MarketDirection.BEARISH and tick.option_type != OptionType.PE:
            return False, "BLOCKED_INVERSION: Bearish regime cannot execute Call (CE)"

        # 4. TTL Check (Dual compatible with is_valid and is_alive)
        is_live = getattr(ttl_state, "is_valid", getattr(ttl_state, "is_alive", lambda: False))()
        if ttl_state is None or not is_live:
            return False, "BLOCKED: Tick Pulse TTL expired or dead"

        # 5. Greeks Validation & Strict Delta Bounds
        if greeks is None:
            return False, "BLOCKED: Greeks are missing or non-finite"

        abs_delta = abs(greeks.delta)
        if abs_delta < min_delta:
            return False, f"BLOCKED_DELTA_BOUNDS: Delta {abs_delta:.3f} below minimum ({min_delta:.2f})"
        if abs_delta > max_delta:
            return False, f"BLOCKED_DELTA_BOUNDS: Delta {abs_delta:.3f} exceeds maximum ({max_delta:.2f})"

        # 6. Strict Liquidity & Spread Check
        if tick.ltp <= 0.0 or tick.ask <= 0.0 or tick.bid <= 0.0:
            return False, "BLOCKED_LIQUIDITY: Zero or negative LTP/Bid/Ask in option contract"

        calculated_spread = tick.ask - tick.bid
        if calculated_spread < 0:
            return False, "BLOCKED_SPREAD: Inverted spread (ask < bid)"

        spread_pct = calculated_spread / tick.ltp
        if spread_pct > max_spread_pct:
            return False, f"BLOCKED_SPREAD: Spread {spread_pct*100:.2f}% exceeds max {max_spread_pct*100:.1f}%"

        return True, "PASSED_ALL_SAFETY_GATES"
