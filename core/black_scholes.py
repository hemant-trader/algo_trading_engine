import math
from scipy.stats import norm
from core.types import OptionType, OptionGreeks

class BlackScholesEngine:
    @staticmethod
    def calculate_greeks(
        spot: float,
        strike: float,
        tte: float,
        r: float,
        iv: float,
        option_type: OptionType
    ) -> OptionGreeks:
        # Zero/Negative Guard
        if tte <= 1e-5 or iv <= 1e-4 or spot <= 0 or strike <= 0:
            return OptionGreeks(iv=max(iv, 0.0), delta=0.0, gamma=0.0, theta=0.0, vega=0.0)

        sqrt_tte = math.sqrt(tte)
        d1 = (math.log(spot / strike) + (r + 0.5 * (iv ** 2)) * tte) / (iv * sqrt_tte)
        d2 = d1 - iv * sqrt_tte

        pdf_d1 = norm.pdf(d1)
        discount = math.exp(-r * tte)

        if option_type == OptionType.CE:
            delta = norm.cdf(d1)
            theta = (- (spot * pdf_d1 * iv) / (2 * sqrt_tte) - r * strike * discount * norm.cdf(d2)) / 365.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (- (spot * pdf_d1 * iv) / (2 * sqrt_tte) + r * strike * discount * norm.cdf(-d2)) / 365.0

        gamma = pdf_d1 / (spot * iv * sqrt_tte)
        # Vega normalized per 1% move in IV
        vega = spot * pdf_d1 * sqrt_tte / 100.0

        return OptionGreeks(iv=iv, delta=delta, gamma=gamma, theta=theta, vega=vega)

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        spot: float,
        strike: float,
        tte: float,
        r: float,
        option_type: OptionType,
        max_iterations: int = 15,
        precision: float = 1e-3
    ) -> float:
        # Fast Invalidation Checks
        if market_price <= 0.05 or tte <= 1e-5 or spot <= 0 or strike <= 0:
            return 0.0

        discount = math.exp(-r * tte)
        
        # Intrinsic value bounds check
        if option_type == OptionType.CE:
            intrinsic = max(0.0, spot - strike * discount)
        else:
            intrinsic = max(0.0, strike * discount - spot)

        if market_price < intrinsic:
            return 0.001

        # Initial seed guess
        iv = 0.20
        sqrt_tte = math.sqrt(tte)

        for _ in range(max_iterations):
            d1 = (math.log(spot / strike) + (r + 0.5 * (iv ** 2)) * tte) / (iv * sqrt_tte)
            d2 = d1 - iv * sqrt_tte

            if option_type == OptionType.CE:
                theo_price = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
            else:
                theo_price = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)

            diff = theo_price - market_price
            if abs(diff) < precision:
                return round(iv, 4)

            # Vega calculation for gradient update
            vega = spot * norm.pdf(d1) * sqrt_tte
            if vega < 1e-5:
                break

            step = diff / vega
            iv -= step

            # Keep IV constrained within valid numerical range
            if iv <= 0.01:
                iv = 0.01
            elif iv > 4.0:  # Max 400% IV
                iv = 4.0
                break

        return round(max(0.001, iv), 4)
