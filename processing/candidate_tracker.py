from typing import Optional
from core.types import CandidateState, OptionType, MarketDirection

class CandidateHysteresisTracker:
    def __init__(
        self, 
        hysteresis_threshold: float = 5.0, 
        min_confirmations: int = 3, 
        max_confirmations: int = 10,
        min_score_threshold: float = 80.0
    ):
        self.threshold: float = hysteresis_threshold
        self.min_confirmations: int = min_confirmations
        self.max_confirmations: int = max_confirmations
        self.min_score_threshold: float = min_score_threshold
        self.active_candidate: Optional[CandidateState] = None
        self.last_direction: Optional[MarketDirection] = None

    def process_candidate(
        self, 
        new_symbol: str, 
        new_strike: float, 
        new_option_type: OptionType, 
        new_score: float,
        market_direction: Optional[MarketDirection] = None
    ) -> tuple[Optional[CandidateState], bool]:
        
        # 1. Hard Invariant: Direction Flip par Instant Full Reset
        if market_direction is not None and self.last_direction is not None:
            if market_direction != self.last_direction:
                self.flush()
                self.last_direction = market_direction
                return None, False
        if market_direction is not None:
            self.last_direction = market_direction

        # 2. Score Threshold Hard Gate (< 80.0 eligible nahi hoga)
        if new_score < self.min_score_threshold:
            if self.active_candidate:
                self.active_candidate.confirmation_count = max(0, self.active_candidate.confirmation_count - 1)
            return self.active_candidate, False

        # 3. Initialize First Candidate
        if self.active_candidate is None:
            self.active_candidate = CandidateState(
                symbol=new_symbol,
                strike=new_strike,
                option_type=new_option_type,
                score=new_score,
                confirmation_count=1
            )
            return self.active_candidate, False

        # 4. Hard Option Type Inversion (CE <-> PE switch par reset)
        if self.active_candidate.option_type != new_option_type:
            self.active_candidate = CandidateState(
                symbol=new_symbol,
                strike=new_strike,
                option_type=new_option_type,
                score=new_score,
                confirmation_count=1
            )
            return self.active_candidate, False

        # 5. Same Strike Re-confirmation (Consecutive Tick Confirmation)
        if self.active_candidate.symbol == new_symbol:
            self.active_candidate.score = new_score
            if self.active_candidate.confirmation_count < self.max_confirmations:
                self.active_candidate.confirmation_count += 1
            is_ready = self.active_candidate.confirmation_count >= self.min_confirmations
            return self.active_candidate, is_ready

        # 6. Competing Strike with +5.0 Hysteresis Barrier
        score_diff = new_score - self.active_candidate.score
        if score_diff >= self.threshold:
            self.active_candidate = CandidateState(
                symbol=new_symbol,
                strike=new_strike,
                option_type=new_option_type,
                score=new_score,
                confirmation_count=1
            )
            return self.active_candidate, False

        # 7. Suppress flickering competitor ticks
        if self.active_candidate.confirmation_count > 1:
            self.active_candidate.confirmation_count -= 1
        is_ready = self.active_candidate.confirmation_count >= self.min_confirmations
        return self.active_candidate, is_ready

    def flush(self) -> None:
        """Flushes active state to NO TRADE instantly."""
        self.active_candidate = None
