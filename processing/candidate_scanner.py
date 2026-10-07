import math
from typing import List, Dict, Any, Optional, Tuple
from core.types import OptionType, MarketDirection

class MultiFactorOptionScanner:
    def __init__(
        self,
        # ITM Parameters (High Delta, Quick Follow-through)
        itm_min_delta: float = 0.55,
        itm_max_delta: float = 0.75,
        # OTM Parameters (Low Cost, High Reward Burst)
        otm_min_delta: float = 0.25,
        otm_max_delta: float = 0.45,
        max_spread_pct: float = 0.04,
        min_volume: int = 500,
        min_oi: int = 5000,
        # OTM Investment Budget Limit (Premium Range in ₹)
        max_otm_premium: float = 90.0,
        min_otm_premium: float = 15.0
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
        trend_score: float
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Scans option chain snapshot and returns BOTH:
        1. best_winner: Best High-Yield OTM Candidate (Low Investment, High Profit)
        2. diagnostics['best_itm_candidate']: Best ITM Contract for Tracked Execution
        Includes Institutional OI build-up and Volume flow analysis.
        """
        diagnostics = {
            "total_strikes": len(chain_data),
            "direction": direction.value,
            "direction_candidates": 0,
            "passed_hard_gates": 0,
            "rejected_liquidity": 0,
            "rejected_spread": 0,
            "rejected_iv": 0,
            "rejected_non_finite": 0,
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

        otm_pool = []
        itm_pool = []

        for row in chain_data:
            # Aggregate chain-wide OI for Put-Call Ratio
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
            greeks = opt_info.get("option_greeks", {})

            try:
                ltp = float(market_data.get("ltp", 0.0))
                bid = float(market_data.get("bid_price", 0.0))
                ask = float(market_data.get("ask_price", 0.0))
                volume = int(market_data.get("volume", 0))
                oi = int(market_data.get("oi", 0))
                raw_iv_pct = float(greeks.get("iv", 0.0))
                raw_delta = float(greeks.get("delta", 0.0)) if greeks.get("delta") is not None else None
            except (ValueError, TypeError):
                diagnostics["rejected_non_finite"] += 1
                continue

            # Hard Gate: Finite checks
            if not all(math.isfinite(x) for x in (strike_price, ltp, bid, ask, raw_iv_pct)) or raw_delta is None or not math.isfinite(raw_delta):
                diagnostics["rejected_non_finite"] += 1
                continue

            # Hard Gate: Positive Liquidity & Orderbook sanity
            if ltp <= 0.0 or bid <= 0.0 or ask <= 0.0 or ask < bid:
                diagnostics["rejected_liquidity"] += 1
                continue

            if volume < self.min_volume or oi < self.min_oi:
                diagnostics["rejected_liquidity"] += 1
                continue

            # Hard Gate: Spread Gate
            spread = ask - bid
            spread_pct = spread / ltp
            if spread_pct > self.max_spread_pct:
                diagnostics["rejected_spread"] += 1
                continue

            # Hard Gate: IV Bounds
            if raw_iv_pct <= 0.0 or raw_iv_pct > 300.0:
                diagnostics["rejected_iv"] += 1
                continue
            iv_decimal = raw_iv_pct / 100.0

            # Delta Sign Verification
            if target_opt_type == OptionType.CE and raw_delta <= 0.0:
                continue
            if target_opt_type == OptionType.PE and raw_delta >= 0.0:
                continue

            abs_delta = abs(raw_delta)
            diagnostics["passed_hard_gates"] += 1

            candidate_payload = {
                "symbol": f"{int(strike_price)}_{target_opt_type.value}",
                "instrument_key": instrument_key,
                "underlying_key": opt_info.get("underlying_key", ""),
                "strike": strike_price,
                "option_type": target_opt_type,
                "raw_delta": raw_delta,
                "abs_delta": abs_delta,
                "spread_pct": spread_pct,
                "snapshot_ltp": ltp,
                "snapshot_bid": bid,
                "snapshot_ask": ask,
                "volume": volume,
                "oi": oi,
                "iv_decimal": iv_decimal
            }

            # Bucket 1: Best OTM (Low Investment, High Rocket Potential)
            if self.otm_min_delta <= abs_delta <= self.otm_max_delta:
                if self.min_otm_premium <= ltp <= self.max_otm_premium:
                    # Scoring prioritizes High Volume, Low Price (Cost Efficiency) & Tight Spread
                    cost_efficiency = max(0.0, 1.0 - (ltp / self.max_otm_premium)) * 30.0
                    vol_score = min(1.0, math.log10(max(10, volume)) / 6.0) * 35.0
                    spread_score = max(0.0, 1.0 - (spread_pct / self.max_spread_pct)) * 20.0
                    regime_bonus = (min(100.0, max(0.0, trend_score)) / 100.0) * 15.0

                    candidate_payload["score"] = round(cost_efficiency + vol_score + spread_score + regime_bonus, 2)
                    otm_pool.append(candidate_payload)

            # Bucket 2: Best ITM (High Delta, Quick Points Capture)
            if self.itm_min_delta <= abs_delta <= self.itm_max_delta:
                delta_score = max(0.0, 1.0 - (abs(abs_delta - 0.65) / 0.15)) * 40.0
                spread_score = max(0.0, 1.0 - (spread_pct / self.max_spread_pct)) * 30.0
                vol_score = min(1.0, math.log10(max(10, volume)) / 6.0) * 30.0

                candidate_payload["score"] = round(delta_score + spread_score + vol_score, 2)
                itm_pool.append(candidate_payload)

        # OI Sentiment & PCR Calculation
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
        itm_pool.sort(key=lambda x: (x["score"], -x["spread_pct"], x["volume"]), reverse=True)

        best_winner = otm_pool[0] if otm_pool else (itm_pool[0] if itm_pool else None)
        best_itm = itm_pool[0] if itm_pool else best_winner

        if best_winner:
            diagnostics["best_candidate_symbol"] = best_winner["symbol"]
            diagnostics["best_composite_score"] = best_winner["score"]
        
        diagnostics["best_itm_candidate"] = best_itm

        return best_winner, diagnostics
