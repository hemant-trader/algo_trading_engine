import time
from typing import Optional, Tuple
from core.types import OptionType, MarketDirection, CandidateState

class CandidateHysteresisTracker:
    def __init__(self, hysteresis_threshold: float = 5.0, default_confirmations: int = 3):
        self.hysteresis_threshold = hysteresis_threshold
        self.default_confirmations = default_confirmations
        self.active_candidate: Optional[CandidateState] = None
        self.pending_candidate: Optional[CandidateState] = None
        self.last_flip_time: float = 0.0
        self.flip_cooldown_seconds: float = 12.0

    def flush(self) -> None:
        self.active_candidate = None
        self.pending_candidate = None

    def process_candidate(
        self,
        symbol: str,
        strike: float,
        option_type: OptionType,
        score: float,
        direction: MarketDirection = MarketDirection.NEUTRAL,
        required_confirmations: int = 3
    ) -> Tuple[Optional[CandidateState], bool]:
        current_time = time.time()

        if direction == MarketDirection.NEUTRAL:
            self.flush()
            return None, False

        # Direction flip check
        if self.active_candidate is not None:
            if self.active_candidate.direction != direction or self.active_candidate.option_type != option_type:
                self.flush()
                self.last_flip_time = current_time

        # Cooldown guard: Cooldown ke dauran accumulation bilkul freeze rahegi
        if (current_time - self.last_flip_time) < self.flip_cooldown_seconds:
            self.pending_candidate = None
            return None, False

        # Candidate Identity: Symbol + Strike + OptionType + Direction
        same_contract_active = (
            self.active_candidate is not None and
            self.active_candidate.symbol == symbol and
            self.active_candidate.strike == strike and
            self.active_candidate.option_type == option_type and
            self.active_candidate.direction == direction
        )

        same_contract_pending = (
            self.pending_candidate is not None and
            self.pending_candidate.symbol == symbol and
            self.pending_candidate.strike == strike and
            self.pending_candidate.option_type == option_type and
            self.pending_candidate.direction == direction
        )

        # Case A: No active candidate
        if not same_contract_active:
            if not same_contract_pending:
                self.pending_candidate = CandidateState(
                    symbol=symbol,
                    strike=strike,
                    option_type=option_type,
                    composite_score=score,
                    confirmation_count=1,
                    first_seen_time=current_time,
                    last_seen_time=current_time,
                    direction=direction
                )
            else:
                self.pending_candidate.confirmation_count += 1
                self.pending_candidate.composite_score = score
                self.pending_candidate.last_seen_time = current_time

            if self.pending_candidate.confirmation_count >= required_confirmations:
                self.active_candidate = self.pending_candidate
                self.pending_candidate = None
                return self.active_candidate, True

            return self.pending_candidate, False

        # Case B: Active candidate re-confirmation
        self.active_candidate.confirmation_count = min(
            self.active_candidate.confirmation_count + 1, 10
        )
        self.active_candidate.composite_score = score
        self.active_candidate.last_seen_time = current_time
        self.pending_candidate = None
        is_ready = self.active_candidate.confirmation_count >= required_confirmations
        return self.active_candidate, is_ready
