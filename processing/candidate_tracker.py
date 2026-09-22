from typing import Optional
from core.types import CandidateState, OptionType

class CandidateHysteresisTracker:
    def __init__(self, hysteresis_threshold: float = 5.0, min_confirmations: int = 3, max_confirmations: int = 10):
        self.threshold: float = hysteresis_threshold
        self.min_confirmations: int = min_confirmations
        self.max_confirmations: int = max_confirmations
        self.active_candidate: Optional[CandidateState] = None

    def process_candidate(
        self, new_symbol: str, new_strike: float, new_option_type: OptionType, new_score: float
    ) -> tuple[Optional[CandidateState], bool]:
        
        # 1. Initialize First Candidate
        if self.active_candidate is None:
            self.active_candidate = CandidateState(
                symbol=new_symbol,
                strike=new_strike,
                option_type=new_option_type,
                score=new_score,
                confirmation_count=1
            )
            return self.active_candidate, False

        # 2. Hard Invariant: Trend / Option Type Inversion (CE <-> PE)
        # Type badalne par hysteresis check bypass karke fresh count shuru hoga
        if self.active_candidate.option_type != new_option_type:
            self.active_candidate = CandidateState(
                symbol=new_symbol,
                strike=new_strike,
                option_type=new_option_type,
                score=new_score,
                confirmation_count=1
            )
            return self.active_candidate, False

        # 3. Same Strike Re-confirmation
        if self.active_candidate.symbol == new_symbol:
            self.active_candidate.score = new_score
            if self.active_candidate.confirmation_count < self.max_confirmations:
                self.active_candidate.confirmation_count += 1
            is_ready = self.active_candidate.confirmation_count >= self.min_confirmations
            return self.active_candidate, is_ready

        # 4. Competing Strike with Hysteresis Barrier
        score_diff = new_score - self.active_candidate.score
        if score_diff >= self.threshold:
            # New strike decisively beat incumbent candidate
            self.active_candidate = CandidateState(
                symbol=new_symbol,
                strike=new_strike,
                option_type=new_option_type,
                score=new_score,
                confirmation_count=1
            )
            return self.active_candidate, False

        # 5. Decay / Suppress unconfirmed competitor ticks
        # Strike change fail hone par incumbent candidate ko blind 'ready' emit nahi karna
        if self.active_candidate.confirmation_count > 1:
            self.active_candidate.confirmation_count -= 1

        is_ready = self.active_candidate.confirmation_count >= self.min_confirmations
        return self.active_candidate, is_ready

    def flush(self) -> None:
        """Flushes active state to NO TRADE."""
        self.active_candidate = None