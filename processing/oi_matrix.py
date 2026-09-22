from core.types import MarketDirection, OptionType, NormalizedOptionTick

class OptionChainOIEngine:
    @staticmethod
    def calculate_oi_score(
        tick: NormalizedOptionTick, 
        market_direction: MarketDirection,
        min_base_oi: int = 50000
    ) -> float:
        """
        Evaluates Price + OI Action Matrix with dynamic weighting and safety filters.
        Returns continuous score between -10.0 and +10.0 for Candidate Hysteresis.
        """
        if market_direction == MarketDirection.NEUTRAL:
            return 0.0

        # Invariant Directional Guard
        is_correct_option = (
            (market_direction == MarketDirection.BULLISH and tick.option_type == OptionType.CE) or
            (market_direction == MarketDirection.BEARISH and tick.option_type == OptionType.PE)
        )
        if not is_correct_option:
            return -10.0

        # Illiquid Strike Filter (Low absolute OI traps avoid karne ke liye)
        if getattr(tick, 'oi', min_base_oi) < min_base_oi:
            return -5.0

        p_up = tick.price_change > 0
        p_down = tick.price_change < 0
        oi_up = tick.oi_change > 0
        oi_down = tick.oi_change < 0

        # Base Quadrant Multipliers
        if p_up and oi_up:
            quadrant_weight = 1.0     # Long Buildup (Best for buying)
        elif p_up and oi_down:
            quadrant_weight = 0.5     # Short Covering (Fast momentum, short-lived)
        elif p_down and oi_down:
            quadrant_weight = -0.5    # Long Unwinding (Weakness)
        elif p_down and oi_up:
            quadrant_weight = -1.0    # Short Buildup / Writing pressure (Avoid buying)
        else:
            quadrant_weight = 0.0

        # Continuous Score Calculation based on change magnitude
        # Normalized score up to 10 points for Candidate Tracker threshold (+5.0)
        pct_price = getattr(tick, 'price_change_pct', 0.0)
        pct_oi = getattr(tick, 'oi_change_pct', 0.0)

        raw_score = quadrant_weight * (5.0 + min(abs(pct_price) * 0.5 + abs(pct_oi) * 0.5, 5.0))
        return round(raw_score, 2)