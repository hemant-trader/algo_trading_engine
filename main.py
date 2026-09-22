import time
from core.types import MarketDirection, OptionType, NormalizedOptionTick
from core.black_scholes import BlackScholesEngine
from processing.tick_guard import TickGuardEngine
from processing.oi_matrix import OptionChainOIEngine
from processing.candidate_tracker import CandidateHysteresisTracker
from engine.pulse_ttl import PulseTTLStateMachine
from engine.safety_gate import ZeroTrustFinalSafetyGate

def execute_pipeline():
    tick_guard = TickGuardEngine()
    candidate_tracker = CandidateHysteresisTracker(hysteresis_threshold=5.0, min_confirmations=3)
    ttl_manager = PulseTTLStateMachine(max_ttl=5)

    market_direction = MarketDirection.BULLISH
    current_time = time.time()

    # Dynamic percentage calculations for Normalized Tick
    ltp = 150.0
    price_change = 12.5
    price_change_pct = (price_change / (ltp - price_change)) * 100.0 if (ltp - price_change) > 0 else 0.0

    oi = 120000
    oi_change = 15000
    oi_change_pct = (oi_change / (oi - oi_change)) * 100.0 if (oi - oi_change) > 0 else 0.0

    tick = NormalizedOptionTick(
        symbol="NIFTY26AUG24500CE",
        underlying="NIFTY",
        expiry="2026-08-27",
        strike=24500.0,
        option_type=OptionType.CE,
        ltp=ltp,
        bid=149.8,
        ask=150.2,
        spread=0.4,
        volume=50000,
        oi=oi,
        volume_change=5000,
        oi_change=oi_change,
        price_change=price_change,
        price_change_pct=price_change_pct,
        oi_change_pct=oi_change_pct,
        timestamp=current_time,
        sequence_no=1001,
        tte=0.0164
    )

    # 1. Tick Guard Ingestion
    is_valid_tick, msg = tick_guard.validate_tick(tick, current_time)
    if not is_valid_tick:
        print(f"Tick Rejected: {msg}")
        return

    # 2. Pulse TTL Activation
    ttl_manager.pulse(is_valid_tick)

    # 3. Quantitative Valuation (Black-Scholes & Greeks)
    spot_price = 24480.0
    risk_free_rate = 0.07
    iv = BlackScholesEngine.implied_volatility(
        tick.ltp, spot_price, tick.strike, tick.tte, risk_free_rate, tick.option_type
    )
    greeks = BlackScholesEngine.calculate_greeks(
        spot_price, tick.strike, tick.tte, risk_free_rate, iv, tick.option_type
    )

    # 4. OI Matrix & Relative Scoring
    oi_score = OptionChainOIEngine.calculate_oi_score(tick, market_direction)
    composite_score = 80.0 + oi_score

    # 5. Hysteresis Confirmation
    candidate, ready = None, False
    for _ in range(3):
        candidate, ready = candidate_tracker.process_candidate(
            tick.symbol, tick.strike, tick.option_type, composite_score
        )

    # 6. Zero-Trust Final Gate Verification (Matching safety_gate.py signature)
    passed, gate_msg = ZeroTrustFinalSafetyGate.verify_execution(
        market_direction=market_direction,
        tick=tick,
        greeks=greeks,
        candidate=candidate,
        candidate_ready=ready,
        ttl_state=ttl_manager,
        quality_grade="GRADE_A",
        max_spread_pct=0.02
    )

    action = f"BUY {candidate.option_type.value}" if passed else "NO_TRADE"

    print("\n=== PIPELINE EXECUTION RESULT ===")
    print(f"Market Direction   : {market_direction.value}")
    print(f"Selected Candidate : {candidate.symbol}")
    print(f"Greeks Calculated  : IV={iv*100:.2f}%, Delta={greeks.delta:.3f}, Theta={greeks.theta:.2f}")
    print(f"OI Score           : {oi_score}")
    print(f"Safety Gate Verdict: {gate_msg}")
    print(f"FINAL ACTION       : {action}\n")

if __name__ == "__main__":
    execute_pipeline()
