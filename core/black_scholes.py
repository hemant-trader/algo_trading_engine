import math
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm
from core.types import OptionType

@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class BlackScholesEngine:
    @staticmethod
    def option_price(
        spot: float,
        strike: float,
        tte: float,
        risk_free_rate: float,
        iv: float,
        option_type: OptionType
    ) -> float:
        """Standard European Option Pricing."""
        if spot <= 0 or strike <= 0 or tte <= 0 or iv <= 0:
            return float("nan")

        sqrt_t = math.sqrt(tte)
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * tte) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t

        if option_type == OptionType.CE:
            return spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * tte) * norm.cdf(d2)
        else:
            return strike * math.exp(-risk_free_rate * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    @staticmethod
    def calculate_greeks(
        spot: float,
        strike: float,
        tte: float,
        risk_free_rate: float,
        iv: float,
        option_type: OptionType
    ) -> Optional[OptionGreeks]:
        """
        Calculates Black-Scholes Greeks.
        Returns None if inputs violate economic bounds (No synthetic fallback).
        """
        if spot <= 0 or strike <= 0 or tte <= 0 or iv <= 0 or not math.isfinite(iv):
            return None

        sqrt_t = math.sqrt(tte)
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * tte) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_minus_d2 = norm.cdf(-d2)

        gamma = pdf_d1 / (spot * iv * sqrt_t)
        vega = (spot * pdf_d1 * sqrt_t) / 100.0  # Per 1% vol change

        if option_type == OptionType.CE:
            delta = cdf_d1
            theta = (-(spot * pdf_d1 * iv) / (2.0 * sqrt_t) 
                     - risk_free_rate * strike * math.exp(-risk_free_rate * tte) * norm.cdf(d2)) / 365.0
            rho = (strike * tte * math.exp(-risk_free_rate * tte) * norm.cdf(d2)) / 100.0
        else:
            delta = cdf_d1 - 1.0  # Negative for Put
            theta = (-(spot * pdf_d1 * iv) / (2.0 * sqrt_t) 
                     + risk_free_rate * strike * math.exp(-risk_free_rate * tte) * cdf_minus_d2) / 365.0
            rho = (-strike * tte * math.exp(-risk_free_rate * tte) * cdf_minus_d2) / 100.0

        return OptionGreeks(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho)
        )

    @staticmethod
    def implied_volatility(
        price: float,
        spot: float,
        strike: float,
        tte: float,
        risk_free_rate: float,
        option_type: OptionType,
        max_iterations: int = 40,
        precision: float = 1e-4
    ) -> Optional[float]:
        """
        Newton-Raphson + Bisection fallback.
        Returns None if price is arbitrage-invalid or fails convergence.
        """
        if (not math.isfinite(price) or not math.isfinite(spot) or 
            not math.isfinite(strike) or not math.isfinite(tte)):
            return None

        if price <= 0 or spot <= 0 or strike <= 0 or tte <= 0:
            return None

        discount = math.exp(-risk_free_rate * tte)
        intrinsic = max(0.0, spot - strike) if option_type == OptionType.CE else max(0.0, strike - spot)
        upper_bound = spot if option_type == OptionType.CE else strike * discount

        if price <= intrinsic + precision or price > upper_bound + precision:
            return None

        low, high = 1e-4, 4.0
        sigma = 0.25

        for _ in range(max_iterations):
            model_price = BlackScholesEngine.option_price(spot, strike, tte, risk_free_rate, sigma, option_type)
            if math.isnan(model_price):
                break
            diff = model_price - price
            if abs(diff) <= precision:
                return float(sigma)

            d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * tte) / (sigma * math.sqrt(tte))
            vega_raw = spot * norm.pdf(d1) * math.sqrt(tte)
            if vega_raw < 1e-8:
                break

            new_sigma = sigma - diff / vega_raw
            if not (low < new_sigma < high):
                break
            sigma = new_sigma

        for _ in range(60):
            mid = (low + high) / 2.0
            model_price = BlackScholesEngine.option_price(spot, strike, tte, risk_free_rate, mid, option_type)
            if math.isnan(model_price):
                return None
            diff = model_price - price
            if abs(diff) <= precision:
                return float(mid)

            if diff > 0:
                high = mid
            else:
                low = mid

        return None
