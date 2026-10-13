"""
engine_orchestrator.py - Single-Owner Session Orchestrator & Automated Acceptance Test Suite
"""

import datetime
import zoneinfo
from typing import Dict, Optional
from candle_aggregator import CandleAggregator
from signal_engine import SignalEngine, CooldownStateMachine

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

class EngineOrchestrator:
    def __init__(self):
        self.agg = CandleAggregator(consecutive_recovery_needed_5m=3)
        self.engine = SignalEngine(rsi_period=14, atr_period=14, cooldown_bars_5m=2)
        self.last_session_date: Optional[datetime.date] = None

    def process_tick(self, ltp: float, tick_dt: datetime.datetime) -> Dict:
        norm_dt = self.agg.normalize_dt(tick_dt)
        cur_date = norm_dt.date()

        if self.last_session_date is not None and cur_date != self.last_session_date:
            self.agg.reset_intraday_state()
            self.engine.reset_intraday_state()

        self.last_session_date = cur_date

        res = self.agg.add_tick(ltp, norm_dt)
        closed_5m = res.get("closed_5m")

        event_payload = None
        if closed_5m:
            action, diag = self.engine.evaluate(
                self.agg.bars_1m, self.agg.bars_5m, self.agg.bars_15m,
                self.agg.is_gap_active, self.agg.gap_reason
            )
            event_payload = {
                "time": closed_5m["time"],
                "price": ltp,
                "action": action,
                "macro": diag["macro_trend"],
                "rsi": diag["rsi_1m"],
                "cooldown": diag["cooldown_remaining"],
                "status": diag["status"],
                "is_event": diag["is_new_signal_event"]
            }

        return {"agg_res": res, "event": event_payload}

    def finalize_boundary(self, wall_time: datetime.datetime) -> Optional[Dict]:
        boundary_res = self.agg.check_and_finalize_boundary(wall_time)
        closed_5m = boundary_res.get("closed_5m")
        if closed_5m:
            action, diag = self.engine.evaluate(
                self.agg.bars_1m, self.agg.bars_5m, self.agg.bars_15m,
                self.agg.is_gap_active, self.agg.gap_reason
            )
            return {
                "time": closed_5m["time"],
                "price": closed_5m["close"],
                "action": action,
                "macro": diag["macro_trend"],
                "rsi": diag["rsi_1m"],
                "cooldown": diag["cooldown_remaining"],
                "status": diag["status"],
                "is_event": diag["is_new_signal_event"]
            }
        return None


# ============================================================================
# AUTOMATED ACCEPTANCE SUITES
# ============================================================================
def test_pure_cooldown_state_machine():
    print("=" * 70)
    print("TEST 1: COOLDOWN STATE MACHINE INVARIANTS")
    print("=" * 70)

    sm = CooldownStateMachine(cooldown_bars=2)
    t0 = 1000.0

    # 1. Bar 0: BUY CE emitted -> cooldown = 2
    act0, d0 = sm.process_candidate("BUY CE", t0)
    assert act0 == "BUY CE" and d0["is_new_signal_event"] is True
    assert sm.cooldown_remaining_bars == 2
    print("PASS: Bar 0 BUY CE emitted (cooldown=2).")

    # 2. Bar 1: NO TRADE candidate MUST decrement cooldown to 1
    act1, d1 = sm.process_candidate("NO TRADE", t0 + 300.0)
    assert act1 == "NO TRADE" and not d1["is_new_signal_event"]
    assert sm.cooldown_remaining_bars == 1
    print("PASS: Bar 1 NO TRADE decremented cooldown to 1.")

    # 3. Bar 2: Same-direction candidate BUY CE MUST decrement cooldown to 0
    act2, d2 = sm.process_candidate("BUY CE", t0 + 600.0)
    assert act2 == "BUY CE" and not d2["is_new_signal_event"]
    assert sm.cooldown_remaining_bars == 0
    print("PASS: Bar 2 same-direction candidate decremented cooldown to 0.")

    # 4. Bar 3: Reversal to BUY PE cleanly emitted -> cooldown re-arms to 2
    act3, d3 = sm.process_candidate("BUY PE", t0 + 900.0)
    assert act3 == "BUY PE" and d3["is_new_signal_event"] is True
    assert sm.cooldown_remaining_bars == 2
    print("PASS: Bar 3 BUY PE reversal emitted cleanly (cooldown re-armed to 2).")

    # 5. Idempotency on repeated epoch
    act3_rep, d3_rep = sm.process_candidate("BUY PE", t0 + 900.0)
    assert act3_rep == "BUY PE" and not d3_rep["is_new_signal_event"]
    assert sm.cooldown_remaining_bars == 2
    print("PASS: Repeated bar epoch returned committed snapshot without counter corruption.")

    # 6. Out-of-order epoch rejection
    act_old, d_old = sm.process_candidate("BUY CE", t0 + 500.0)
    assert act_old == "NO TRADE" and d_old["status"] == "OUT_OF_ORDER_BAR_REJECTED"
    print("PASS: Strictly past/out-of-order bar rejected.")


def test_full_pipeline_regression():
    print("\n" + "=" * 70)
    print("TEST 2: FULL PIPELINE REGRESSION & BOUNDARIES")
    print("=" * 70)

    orch = EngineOrchestrator()
    events = []

    def feed(price, dt):
        out = orch.process_tick(price, dt)
        if out["event"]:
            events.append(out["event"])

    # Day 1: Build baseline warm-up (360 Mins)
    d1 = datetime.datetime(2026, 10, 13, 9, 15, 0, tzinfo=IST)
    p = 71500.0
    for m in range(360):
        t = d1 + datetime.timedelta(minutes=m)
        p += 1.0
        feed(p, t.replace(second=5))
        feed(p, t.replace(second=35))

    fin1 = orch.finalize_boundary(d1 + datetime.timedelta(minutes=360))
    if fin1:
        events.append(fin1)

    assert len(orch.agg.bars_15m) >= 22, "Day 1 15m warm-up failed"
    print(f"PASS: Day 1 Warm-up achieved ({len(orch.agg.bars_15m)} bars 15m).")

    # Day 2: Morning Bullish Push
    d2 = datetime.datetime(2026, 10, 14, 9, 15, 0, tzinfo=IST)
    p = 72000.0
    for m in range(60):
        t = d2 + datetime.timedelta(minutes=m)
        p += 18.0
        feed(p, t.replace(second=5))
        feed(p, t.replace(second=35))

    day2_events = [e for e in events if e["time"].date() == d2.date()]
    day2_buy_ce = [e for e in day2_events if e["action"] == "BUY CE" and e["is_event"]]
    assert len(day2_buy_ce) >= 1, "GATE 1 FAIL: BUY CE not emitted on Day 2"
    initial_ce = day2_buy_ce[0]
    ce_time = initial_ce["time"]
    print(f"PASS: Day 2 BUY CE emitted at {ce_time.strftime('%H:%M')}.")

    # Day 2: Crash Wave
    crash_t = d2 + datetime.timedelta(minutes=60)
    for m in range(120):
        t = crash_t + datetime.timedelta(minutes=m)
        p -= 55.0
        feed(p, t.replace(second=5))
        feed(p, t.replace(second=35))

    fin2 = orch.finalize_boundary(crash_t + datetime.timedelta(minutes=120))
    if fin2:
        events.append(fin2)

    day2_events = [e for e in events if e["time"].date() == d2.date()]
    post_ce = [e for e in day2_events if e["time"] > ce_time]
    day2_buy_pe = [e for e in post_ce if e["action"] == "BUY PE" and e["is_event"]]

    assert len(day2_buy_pe) >= 1, "GATE 2 FAIL: BUY PE reversal not emitted on Day 2"
    print(f"PASS: Day 2 BUY PE reversal emitted at {day2_buy_pe[0]['time'].strftime('%H:%M')}.")

    # Boundary Session Cap Assertion: Finalize at 15:35 must return None
    late_wall = d2.replace(hour=15, minute=35, second=0, microsecond=0)
    assert orch.agg.check_and_finalize_boundary(late_wall)["closed_5m"] is None
    print("PASS: Out-of-session boundary finalization safely rejected.")

    print("\n" + "=" * 70)
    print("ALL INTEGRATION GATES ASSERTED CLEANLY.")
    print("=" * 70)


if __name__ == "__main__":
    test_pure_cooldown_state_machine()
    test_full_pipeline_regression()
