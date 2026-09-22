from core.types import NormalizedOptionTick

class TickGuardEngine:
    def __init__(self, max_allowed_clock_skew: float = 1.0, max_allowed_latency: float = 3.0):
        """
        :param max_allowed_clock_skew: Maximum seconds tick timestamp can be ahead of local clock.
        :param max_allowed_latency: Maximum age (in seconds) of an incoming tick before considered stale.
        """
        self.last_timestamps: dict[str, float] = {}
        self.last_sequence_nos: dict[str, int] = {}
        self.max_skew = max_allowed_clock_skew
        self.max_latency = max_allowed_latency

    def validate_tick(self, tick: NormalizedOptionTick, current_system_time: float) -> tuple[bool, str]:
        key = f"{tick.underlying}_{tick.expiry}_{tick.symbol}"

        # 1. Clock Skew (Future Tick Guard)
        if tick.timestamp > current_system_time + self.max_skew:
            return False, "FUTURE_TICK_REJECTED"

        # 2. Stale Latency Guard (Historical / Delayed Tick)
        if (current_system_time - tick.timestamp) > self.max_latency:
            return False, "STALE_TICK_LATENCY_EXCEEDED"

        # 3. Numerical & Pricing Sanity Guard
        if tick.ltp <= 0.05 or tick.bid < 0 or tick.ask < 0:
            return False, "INVALID_NUMERICAL_PRICING"

        # 4. Out-of-Order & Sequence Verification
        last_ts = self.last_timestamps.get(key, 0.0)
        if tick.timestamp < last_ts:
            return False, "OUT_OF_ORDER_TICK_REJECTED"
            
        if tick.timestamp == last_ts:
            last_seq = self.last_sequence_nos.get(key, -1)
            if tick.sequence_no <= last_seq:
                return False, "DUPLICATE_OR_OLD_SEQUENCE_TICK_REJECTED"

        # State Update
        self.last_timestamps[key] = tick.timestamp
        self.last_sequence_nos[key] = tick.sequence_no
        return True, "VALID"

    def purge_key(self, key: str) -> None:
        """Removes expired or inactive contract keys from memory."""
        self.last_timestamps.pop(key, None)
        self.last_sequence_nos.pop(key, None)
