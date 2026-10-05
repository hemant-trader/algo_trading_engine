import math
from typing import List, Dict, Any, Optional, Tuple
from core.types import OptionType, MarketDirection

class MultiFactorOptionScanner:
    def __init__(
        self,
        min_delta: float = 0.45,
        max_delta: float = 0.65,
        target_delta: float = 0.55,
        max_spread_pct: float = 0.03,
        min_volume: int = 500,
        min_oi: int = 5000
    ):
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.target_delta = target_delta
        self.max_spread_pct = max_spread_pct
        self.min_volume = min_volume
        self.min_oi = min_oi

    def scan_and_rank_best_candidate(
        self,
        chain_data: List[Dict[str, Any]],
        direction: MarketDirection,
        trend_score: float
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Scans option chain snapshot, enforces finite math bounds,
        and returns EXACTLY ONE best candidate snapshot with audit diagnostics.
        """
        diagnostics = {
            "total_strikes": len(chain_data),
            "direction": direction.value,
            "direction_candidates": 0,
            "passed_hard_gates": 0,
            "rejected_non_finite": 0,
            "rejected_iv": 0,
            "rejected_delta": 0,
            "rejected_spread": 0,
            "rejected_liquidity": 0,
            "best_candidate_symbol": None,
            "best_composite_score": 0.0
        }

        if direction == MarketDirection.NEUTRAL or not chain_data:
            return None, diagnostics

        target_opt_type = OptionType.CE if direction == MarketDirection.BULLISH else OptionType.PE
        opt_key = "call_options" if target_opt_type == OptionType.CE else "put_options"

        scored_candidates = []

        for row in chain_data:
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

            # 1. HARD GATE: Strict Finite Value Checks
            numeric_fields = (strike_price, ltp, bid, ask, raw_iv_pct)
            if not all(math.isfinite(x) for x in numeric_fields) or raw_delta is None or not math.isfinite(raw_delta):
                diagnostics["rejected_non_finite"] += 1
                continue

            # 2. HARD GATE: Positive Liquidity & Orderbook Inversion
            if ltp <= 0.0 or bid <= 0.0 or ask <= 0.0 or ask < bid:
                diagnostics["rejected_liquidity"] += 1
                continue

            if volume < self.min_volume or oi < self.min_oi:
                diagnostics["rejected_liquidity"] += 1
                continue

            # 3. HARD GATE: Spread Gate
            spread = ask - bid
            spread_pct = spread / ltp
            if spread_pct > self.max_spread_pct:
                diagnostics["rejected_spread"] += 1
                continue

            # 4. HARD GATE: Strict Finite IV Bounds & Explicit Decimal Ingestion
            if raw_iv_pct <= 0.0 or raw_iv_pct > 300.0:
                diagnostics["rejected_iv"] += 1
                continue
            iv_decimal = raw_iv_pct / 100.0

            # 5. HARD GATE: Delta Sign & Magnitude Bounds
            if target_opt_type == OptionType.CE and raw_delta <= 0.0:
                diagnostics["rejected_delta"] += 1
                continue
            if target_opt_type == OptionType.PE and raw_delta >= 0.0:
                diagnostics["rejected_delta"] += 1
                continue

            abs_delta = abs(raw_delta)
            if not (self.min_delta <= abs_delta <= self.max_delta):
                diagnostics["rejected_delta"] += 1
                continue

            diagnostics["passed_hard_gates"] += 1

            # 6. DETERMINISTIC MULTI-FACTOR SCORING
            delta_deviation = abs(abs_delta - self.target_delta)
            delta_score = max(0.0, 1.0 - (delta_deviation / 0.15)) * 30.0

            spread_score = max(0.0, 1.0 - (spread_pct / self.max_spread_pct)) * 25.0

            volume_factor = min(1.0, math.log10(max(10, volume)) / 6.0)
            volume_score = volume_factor * 25.0

            regime_factor = (min(100.0, max(0.0, trend_score)) / 100.0) * 20.0

            composite_score = round(delta_score + spread_score + volume_score + regime_factor, 2)

            scored_candidates.append({
                "symbol": f"{int(strike_price)}_{target_opt_type.value}",
                "instrument_key": instrument_key,
                "underlying_key": opt_info.get("underlying_key", ""),
                "strike": strike_price,
                "option_type": target_opt_type,
                "score": composite_score,
                "raw_delta": raw_delta,
                "abs_delta": abs_delta,
                "spread_pct": spread_pct,
                "snapshot_ltp": ltp,
                "snapshot_bid": bid,
                "snapshot_ask": ask,
                "volume": volume,
                "oi": oi,
                "iv_decimal": iv_decimal
            })

        if not scored_candidates:
            return None, diagnostics

        # Deterministic Ranking: Score DESC -> Spread ASC -> Volume DESC
        scored_candidates.sort(
            key=lambda x: (x["score"], -x["spread_pct"], x["volume"]),
            reverse=True
        )

        best_winner = scored_candidates[0]
        diagnostics["best_candidate_symbol"] = best_winner["symbol"]
        diagnostics["best_composite_score"] = best_winner["score"]

        return best_winner, diagnostics
