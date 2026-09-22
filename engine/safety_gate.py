from core.types import MarketDirection, OptionType, NormalizedOptionTick, OptionGreeks, CandidateState
from engine.pulse_ttl import PulseTTLStateMachine

class ZeroTrustFinalSafetyGate:
    @staticmethod
    def verify_execution(
        market_direction: MarketDirection,
        tick: NormalizedOptionTick,
        greeks: OptionGreeks,
        candidate: CandidateState,
        candidate_ready: bool,
        ttl_state: PulseTTLStateMachine,
        quality_grade: str,
        max_spread_pct: float = 0.02
    ) -> tuple[bool, str]:
        if market_direction == MarketDirection.BULLISH and tick.option_type != OptionType.CE:
            return False, "INVARIANT_VIOLATION_BULLISH_MUST_BE_CE"
        if market_direction == MarketDirection.BEARISH and tick.option_type != OptionType.PE:
            return False, "INVARIANT_VIOLATION_BEARISH_MUST_BE_PE"
        if market_direction == MarketDirection.NEUTRAL:
            return False, "INVARIANT_NEUTRAL_DIRECTION"

        if not ttl_state.is_alive():
            return False, "PULSE_TTL_EXPIRED"

        if not candidate_ready:
            return False, "CANDIDATE_NOT_CONFIRMED"

        if tick.ltp <= 0 or tick.bid <= 0 or tick.ask <= 0:
            return False, "INVALID_PRICING_TICK"
        if (tick.ask - tick.bid) / tick.ltp > max_spread_pct:
            return False, "SPREAD_TOO_WIDE"
        if tick.tte <= 0:
            return False, "EXPIRED_INSTRUMENT_TTE_ZERO"

        if greeks.iv <= 0 or abs(greeks.delta) < 0.20 or abs(greeks.delta) > 0.85:
            return False, "GREEKS_OUT_OF_BOUNDS"
        if quality_grade not in ["GRADE_A", "GRADE_B"]:
            return False, "QUALITY_GRADE_REJECTED"

        return True, "PASSED_ALL_SAFETY_GATES"
