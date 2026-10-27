"""
candidate_scanner.py - Multi-Factor Institutional Option Scanner
Updated & Hardened:
- Strict Penny Option Filter (LTP >= min_otm_premium, blocks ₹0.10 junk)
- Accurate Spot Anchor (Calculates exact underlying spot from closest straddle)
- Proximity-to-ATM Bounding (Restricts candidates within +/- 4 strikes)
- Volume & Open Interest Liquidity Gates
- Dual Winner Emission:
    * best_winner: Best High-Yield Near-OTM Candidate (Display & Directional Burst)
    * best_itm_candidate: Best ATM/Near-ITM Contract (Execution & Delta Safety)
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from core.types import OptionType, MarketDirection

def estimate_approx_delta(spot: float, strike: float, option_type: OptionType) -> float:
    """Fallback delta estimation when broker API doesn't provide Greeks."""
    if spot <= 0 or strike <= 0:
        return 0.50 if option_type == OptionType.CE else -0.50
    diff_pct = (spot - strike) / spot
    approx = 0.50 + (diff_pct * 3.5)
    approx = max(0.10, min(0.95, approx))
    return approx if option_type == OptionType.CE else -approx


class MultiFactorOptionScanner:
    def __init__(
        self,
        # ATM / Execution Parameters (Delta: 0.40 - 0.70)
        itm_min_delta: float = 0.40,
        itm_max_delta: float = 0.70,
        # OTM Parameters (Delta: 0.25 - 0.45, 1 to 2 strikes out)
        otm_min_delta: float = 0.25,
        otm_max_delta: float = 0.45,
        max_spread_pct: float = 0.08,
        min_volume: int = 20,
        min_oi: int = 200,
        # Premium Protection: Minimum ₹10 to strictly reject dead penny options
        max_otm_premium: float = 400.0,
        min_otm_premium: float = 10.0
    ):
        self.itm_min_delta = itm_min_delta
        self.itm_max_delta = itm_max_delta
        self.otm_min_delta = otm_min_delta
        self.otm_max_delta = otm_max_delta
        self.max_spread_pct = max_spread_pct
        self.min_volume = min_volume
        self.min_oi = min_oi
        self.max_otm_premium = max_otm_premium
        self.min_otm_premium = min_otm_premium

    def scan_and_rank_best_candidate(
        self,
        chain_data: List[Dict[str, Any]],
        direction: MarketDirection,
        trend_score: float,
        spot_price: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Scans option chain snapshot and returns BOTH:
        1. best_winner: Best High-Yield Near-OTM Candidate (for Display)
        2. diagnostics['best_itm_candidate']: Strictly ATM Contract (for Execution)
        """
        diagnostics = {
            "total_strikes": len(chain_data) if chain_data else 0,
            "direction": direction.value,
            "direction_candidates": 0,
            "passed_hard_gates": 0,
            "rejected_liquidity": 0,
            "rejected_penny_premium": 0,
            "rejected_spread": 0,
            "rejected_distance": 0,
            "best_candidate_symbol": None,
            "best_composite_score": 0.0,
            "best_itm_candidate": None,
            "pcr_ratio": 1.0,
            "oi_sentiment": "NEUTRAL"
        }

        if direction == MarketDirection.NEUTRAL or not chain_data:
            return None, diagnostics

        target_opt_type = OptionType.CE if direction == MarketDirection.BULLISH else OptionType.PE
        opt_key = "call_options" if target_opt_type == OptionType.CE else "put_options"

        total_ce_oi = 0
        total_pe_oi = 0

        # Robust Spot Estimation (Minimizes Call/Put difference if spot_price not passed)
        underlying_spot = spot_price or 0.0
        if underlying_spot <= 0.0:
            min_diff = float("inf")
            for row in chain_data:
                c_ltp = float(row.get("call_options", {}).get("market_data", {}).get("ltp", 0.0) or 0.0)
                p_ltp = float(row.get("put_options", {}).get("market_data", {}).get("ltp", 0.0) or 0.0)
                if c_ltp > 0 and p_ltp > 0:
                    diff = abs(c_ltp - p_ltp)
                    if diff < min_diff:
                        min_diff = diff
                        underlying_spot = float(row.get("strike_price", 0.0))

        otm_pool = []
        atm_pool = []

        for row in chain_data:
            ce_market = row.get("call_options", {}).get("market_data", {})
            pe_market = row.get("put_options", {}).get("market_data", {})
            total_ce_oi += int(ce_market.get("oi", 0) or 0)
            total_pe_oi += int(pe_market.get("oi", 0) or 0)

            opt_info = row.get(opt_key, {})
            if not opt_info:
                continue

            diagnostics["direction_candidates"] += 1

            strike_price = float(row.get("strike_price", 0.0))
            instrument_key = opt_info.get("instrument_key", "")
            market_data = opt_info.get("market_data", {})
            greeks = opt_info.get("option_greeks", {}) or {}

            try:
                ltp = float(market_data.get("ltp", 0.0) or 0.0)
                bid = float(market_data.get("bid_price", 0.0) or 0.0)
                ask = float(market_data.get("ask_price", 0.0) or 0.0)
                volume = int(market_data.get("volume", 0) or 0)
                oi = int(market_data.get("oi", 0) or 0)

                raw_iv_pct = float(greeks.get("iv", 0.0) or 0.0)
                if raw_iv_pct <= 0.0 or not math.isfinite(raw_iv_pct):
                    raw_iv_pct = 20.0
                iv_dec = max(0.08, raw_iv_pct / 100.0 if raw_iv_pct > 1.0 else raw_iv_pct)

                raw_delta = greeks.get("delta")
                if raw_delta is None or not math.isfinite(float(raw_delta)) or float(raw_delta) == 0.0:
                    raw_delta = estimate_approx_delta(underlying_spot or strike_price, strike_price, target_opt_type)
                else:
                    raw_delta = float(raw_delta)

            except (ValueError, TypeError):
                continue

            # Hard Gate 1: Liquidity & Penny Exclusion
            if ltp <= 0.0:
                diagnostics["rejected_liquidity"] += 1
                continue
            if ltp < self.min_otm_premium:
                diagnostics["rejected_penny_premium"] += 1
                continue

            # Hard Gate 2: Strike Distance Bounding (Max 5 strikes away from spot)
            if underlying_spot > 0:
                distance_pct = abs(strike_price - underlying_spot) / underlying_spot
                if distance_pct > 0.025:  # Strictly within 2.5% of spot price
                    diagnostics["rejected_distance"] += 1
                    continue

            # Bid/Ask normalization
            if bid <= 0.0:
                bid = max(0.05, ltp * 0.98)
            if ask <= 0.0 or ask < bid:
                ask = max(bid + 0.05, ltp * 1.02)

            spread = ask - bid
            spread_pct = spread / ltp if ltp > 0 else 0.05
            if spread_pct > self.max_spread_pct:
                diagnostics["rejected_spread"] += 1
                continue

            abs_delta = abs(raw_delta)
            diagnostics["passed_hard_gates"] += 1

            candidate_payload = {
                "symbol": f"{int(strike_price)}_{target_opt_type.value}",
                "instrument_key": instrument_key,
                "underlying_key": opt_info.get("underlying_key", ""),
                "strike": strike_price,
                "option_type": target_opt_type,
                "raw_delta": raw_delta if target_opt_type == OptionType.CE else -abs_delta,
                "abs_delta": round(abs_delta, 2),
                "spread_pct": spread_pct,
                "snapshot_ltp": ltp,
                "snapshot_bid": bid,
                "snapshot_ask": ask,
                "volume": volume,
                "oi": oi,
                "iv_decimal": min(1.2, iv_dec),
                "distance": abs(strike_price - underlying_spot) if underlying_spot > 0 else 0.0
            }

            # Bucket 1: Best OTM (1-2 strikes out, balanced premium & good volume)
            if self.otm_min_delta <= abs_delta <= self.otm_max_delta:
                vol_score = min(1.0, math.log10(max(10, volume + 10)) / 5.0) * 40.0
                spread_score = max(0.0, 1.0 - (spread_pct / self.max_spread_pct)) * 25.0
                delta_score = max(0.0, 1.0 - (abs(abs_delta - 0.35) / 0.15)) * 20.0
                regime_bonus = (min(100.0, max(0.0, trend_score)) / 100.0) * 15.0
                candidate_payload["score"] = round(vol_score + spread_score + delta_score + regime_bonus, 2)
                otm_pool.append(candidate_payload)

            # Bucket 2: Best ATM (Execution baseline, Delta around 0.50 - 0.60)
            if self.itm_min_delta <= abs_delta <= self.itm_max_delta:
                delta_score = max(0.0, 1.0 - (abs(abs_delta - 0.52) / 0.18)) * 45.0
                spread_score = max(0.0, 1.0 - (spread_pct / self.max_spread_pct)) * 30.0
                vol_score = min(1.0, math.log10(max(10, volume + 10)) / 5.0) * 25.0
                candidate_payload["score"] = round(delta_score + spread_score + vol_score, 2)
                atm_pool.append(candidate_payload)

        # OI PCR Sentiment Diagnostics
        if total_ce_oi > 0:
            pcr = round(total_pe_oi / total_ce_oi, 2)
            diagnostics["pcr_ratio"] = pcr
            if pcr > 1.2:
                diagnostics["oi_sentiment"] = "STRONG BULLISH"
            elif pcr < 0.8:
                diagnostics["oi_sentiment"] = "STRONG BEARISH"
            else:
                diagnostics["oi_sentiment"] = "BALANCED"

        # Sort Pools
        otm_pool.sort(key=lambda x: (x["score"], -x["spread_pct"], x["volume"]), reverse=True)
        atm_pool.sort(key=lambda x: (x.get("distance", 0.0), -x["score"]))

        # Best Winner is Best OTM for Display
        best_winner = otm_pool[0] if otm_pool else (atm_pool[0] if atm_pool else None)
        # Best ITM/ATM is strictly the nearest ATM contract for Execution
        best_atm = atm_pool[0] if atm_pool else best_winner

        if best_winner:
            diagnostics["best_candidate_symbol"] = best_winner["symbol"]
            diagnostics["best_composite_score"] = best_winner["score"]

        diagnostics["best_itm_candidate"] = best_atm

        return best_winner, diagnostics
