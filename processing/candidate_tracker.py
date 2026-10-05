import time
from typing import Optional, Tuple
from core.types import OptionType, MarketDirection, CandidateState

class CandidateHysteresisTracker:
    def __init__(self, hysteresis_delta: float = 5.0, flip_cooldown_seconds: float = 12.0):
        self.hysteresis_delta = hysteresis_delta
        self.flip_cooldown_seconds = flip_cooldown_seconds
        
        self.active_candidate: Optional[CandidateState] = None
        self.pending_candidate: Optional[CandidateState] = None
        self.last_flip_time: float = 0.0
        self._last_confirmed_bar_ts: Optional[object] = None

    def flush(self) -> None:
        """Purges active and pending candidates completely."""
        self.active_candidate = None
        self.pending_candidate = None
        self._last_confirmed_bar_ts = None

    def process_candidate(
        self,
        symbol: str,
        strike: float,
        option_type: OptionType,
        score: float,
        direction: MarketDirection = MarketDirection.NEUTRAL,
        bar_timestamp: Optional[object] = None,
        is_new_closed_bar: bool = True,
        required_confirmations: int = 2
    ) -> Tuple[Optional[CandidateState], bool]:
        """
        Processes exactly ONE ranked winner per closed 5M bar.
        Enforces unique closed-bar idempotency, direction reversal purges, 
        and strict +5.0 hysteresis challenger hurdle.
        """
        current_time = time.time()

        # Regime Neutral Guard
        if direction == MarketDirection.NEUTRAL:
            self.flush()
            return None, False

        # Direction Inversion Guard
        if self.active_candidate is not None:
            if self.active_candidate.direction != direction or self.active_candidate.option_type != option_type:
                self.flush()
                self.last_flip_time = current_time

        if self.pending_candidate is not None:
            if self.pending_candidate.direction != direction or self.pending_candidate.option_type != option_type:
                self.pending_candidate = None

        # Cooldown Guard
        if (current_time - self.last_flip_time) < self.flip_cooldown_seconds:
            self.pending_candidate = None
            return None, False

        # Idempotency Guard: Ensure one closed-bar advances streak exactly once
        is_duplicate_bar = (
            bar_timestamp is not None and 
            self._last_confirmed_bar_ts is not None and 
            bar_timestamp == self._last_confirmed_bar_ts
        )

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

        # -------------------------------------------------------------
        # CASE A: No Active Candidate Exists
        # -------------------------------------------------------------
        if self.active_candidate is None:
            if not same_contract_pending:
                # Register fresh pending candidate
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
                if is_new_closed_bar and not is_duplicate_bar:
                    self._last_confirmed_bar_ts = bar_timestamp
            else:
                # Progress confirmation streak on new closed bar only
                if is_new_closed_bar and not is_duplicate_bar:
                    self.pending_candidate.confirmation_count += 1
                    self._last_confirmed_bar_ts = bar_timestamp
                
                self.pending_candidate.composite_score = score
                self.pending_candidate.last_seen_time = current_time

            if self.pending_candidate.confirmation_count >= required_confirmations:
                self.active_candidate = self.pending_candidate
                self.pending_candidate = None
                return self.active_candidate, True

            return self.pending_candidate, False

        # -------------------------------------------------------------
        # CASE B: Incumbent Active Candidate Exists
        # -------------------------------------------------------------
        if same_contract_active:
            if is_new_closed_bar and not is_duplicate_bar:
                self.active_candidate.confirmation_count = min(
                    self.active_candidate.confirmation_count + 1, 10
                )
                self._last_confirmed_bar_ts = bar_timestamp

            self.active_candidate.composite_score = score
            self.active_candidate.last_seen_time = current_time
            self.pending_candidate = None
            return self.active_candidate, True

        # -------------------------------------------------------------
        # CASE C: Challenger Encountered (Hysteresis Guard)
        # -------------------------------------------------------------
        # Challenger must exceed incumbent score by at least hysteresis_delta (default +5.0)
        hurdle_score = self.active_candidate.composite_score + self.hysteresis_delta
        if score < hurdle_score:
            # Challenger fails hurdle: Incumbent survives, challenger discarded
            return self.active_candidate, True

        # Challenger clears hurdle: Enter pending state to prove multi-bar persistence
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
            if is_new_closed_bar and not is_duplicate_bar:
                self._last_confirmed_bar_ts = bar_timestamp
        else:
            if is_new_closed_bar and not is_duplicate_bar:
                self.pending_candidate.confirmation_count += 1
                self._last_confirmed_bar_ts = bar_timestamp
            
            self.pending_candidate.composite_score = score
            self.pending_candidate.last_seen_time = current_time

        if self.pending_candidate.confirmation_count >= required_confirmations:
            self.active_candidate = self.pending_candidate
            self.pending_candidate = None
            self.last_flip_time = current_time
            return self.active_candidate, True

        # While challenger is pending confirmation, incumbent remains active
        return self.active_candidate, True
