import time
from dataclasses import dataclass
from typing import Optional, Tuple
from core.types import OptionType, MarketDirection, CandidateState

class CandidateHysteresisTracker:
    def __init__(self, hysteresis_threshold: float = 5.0, min_confirmations: int = 3):
        self.hysteresis_threshold = hysteresis_threshold
        self.min_confirmations = min_confirmations
        self.active_candidate: Optional[CandidateState] = None
        self.pending_candidate: Optional[CandidateState] = None
        self.last_flip_time: float = 0.0
        self.flip_cooldown_seconds: float = 15.0  # Prevent CE <-> PE whipsaw oscillation

    def flush(self) -> None:
        """Clears candidate states on Neutral or explicit reset."""
        self.active_candidate = None
        self.pending_candidate = None

    def process_candidate(
        self,
        symbol: str,
        strike: float,
        option_type: OptionType,
        score: float,
        direction: MarketDirection = MarketDirection.NEUTRAL,
        required_confirmations: Optional[int] = None
    ) -> Tuple[Optional[CandidateState], bool]:
        current_time = time.time()
        effective_min_conf = required_confirmations or self.min_confirmations

        if direction == MarketDirection.NEUTRAL:
            self.flush()
            return None, False

        # Direction flip guard (CE <-> PE switch)
        if self.active_candidate is not None:
            if self.active_candidate.direction != direction or self.active_candidate.option_type != option_type:
                self.flush()
                self.last_flip_time = current_time
                return None, False

        # Anti-oscillation flip cooldown: Reject immediate fast re-entry after a flip
        if (current_time - self.last_flip_time) < self.flip_cooldown_seconds:
            return None, False

        # Case A: No active candidate yet
        if self.active_candidate is None:
            if self.pending_candidate is None or self.pending_candidate.symbol != symbol:
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

            if self.pending_candidate.confirmation_count >= effective_min_conf:
                self.active_candidate = self.pending_candidate
                self.pending_candidate = None
                return self.active_candidate, True

            return self.pending_candidate, False

        # Case B: Active candidate re-confirmation
        if self.active_candidate.symbol == symbol:
            self.active_candidate.confirmation_count = min(
                self.active_candidate.confirmation_count + 1, effective_min_conf
            )
            self.active_candidate.composite_score = score
            self.active_candidate.last_seen_time = current_time
            self.pending_candidate = None
            is_ready = self.active_candidate.confirmation_count >= effective_min_conf
            return self.active_candidate, is_ready

        # Case C: New challenger candidate with +5 hysteresis
        if score >= (self.active_candidate.composite_score + self.hysteresis_threshold):
            if self.pending_candidate is None or self.pending_candidate.symbol != symbol:
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

            if self.pending_candidate.confirmation_count >= effective_min_conf:
                self.active_candidate = self.pending_candidate
                self.pending_candidate = None
                return self.active_candidate, True

            return self.active_candidate, (self.active_candidate.confirmation_count >= effective_min_conf)

        self.pending_candidate = None
        return self.active_candidate, (self.active_candidate.confirmation_count >= effective_min_conf)
