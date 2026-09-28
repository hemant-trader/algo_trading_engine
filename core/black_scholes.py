import math
from typing import Optional
from scipy.stats import norm
from core.types import OptionType, OptionGreeks

class BlackScholesEngine:
    @staticmethod
    def calculate_greeks(
        spot: float,
        strike: float,
        tte: float,
        rate: float,
        iv: float,
        option_type: OptionType
    ) -> Optional[OptionGreeks]:
        if spot <= 0 or strike <= 0 or tte <= 0.00001 or iv <= 0.001:
            return None
        try:
            sqrt_tte = math.sqrt(tte)
            d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * tte) / (iv * sqrt_tte)
            d2 = d1 - iv * sqrt_tte

            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_d2 = norm.cdf(d2)
            cdf_neg_d2 = norm.cdf(-d2)

            gamma = pdf_d1 / (spot * iv * sqrt_tte)
            vega = (spot * sqrt_tte * pdf_d1) / 100.0

            if option_type == OptionType.CE:
                delta = cdf_d1
                theta = (-(spot * pdf_d1 * iv) / (2 * sqrt_tte) - rate * strike * math.exp(-rate * tte) * cdf_d2) / 365.0
                rho = (strike * tte * math.exp(-rate * tte) * cdf_d2) / 100.0
            else:
                delta = -cdf_neg_d1
                theta = (-(spot * pdf_d1 * iv) / (2 * sqrt_tte) + rate * strike * math.exp(-rate * tte) * cdf_neg_d2) / 365.0
                rho = (-strike * tte * math.exp(-rate * tte) * cdf_neg_d2) / 100.0

            return OptionGreeks(
                delta=float(delta),
                gamma=float(gamma),
                theta=float(theta),
                vega=float(vega),
                rho=float(rho)
            )
        except Exception:
            return None
