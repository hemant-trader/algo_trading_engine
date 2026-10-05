import os
import json
import csv
import uuid
import datetime
import zoneinfo
from typing import Dict, Any, Optional

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

class HeadlessObservationLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.run_id = str(uuid.uuid4())[:8]
        self._last_logged_bar_ts: Optional[object] = None
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_headers = [
            "run_id",
            "logger_timestamp",
            "bar_timestamp",
            "sequence_no",
            "underlying",
            "spot_ltp",
            "market_dir",
            "trend_score",
            "rsi_14",
            "price_spread",
            "expiry",
            "expiry_rotated",
            "winner_symbol",
            "instrument_key",
            "strike",
            "option_type",
            "abs_delta",
            "spread_pct",
            "total_strikes",
            "direction_candidates",
            "passed_hard_gates",
            "rejected_non_finite",
            "rejected_iv",
            "rejected_delta",
            "rejected_spread",
            "rejected_liquidity",
            "active_candidate",
            "pending_candidate",
            "confirmation_count",
            "is_ready",
            "contract_identity_status",
            "live_quote_age_ms",
            "gate_status",
            "gate_reason",
            "final_action",
            "hypothetical_fill_ask_price",
            "time_to_fill_ms"
        ]

    def _get_file_paths(self) -> tuple[str, str]:
        date_str = datetime.datetime.now(IST).strftime("%Y%m%d")
        csv_path = os.path.join(self.log_dir, f"paper_monitor_{date_str}.csv")
        jsonl_path = os.path.join(self.log_dir, f"paper_monitor_{date_str}.jsonl")
        return csv_path, jsonl_path

    def _ensure_csv_header(self, csv_path: str):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

    def record_bar_event(
        self,
        bar_timestamp: object,
        sequence_no: int,
        underlying: str,
        spot_ltp: float,
        market_dir: str,
        trend_score: float,
        rsi_14: float,
        price_spread: float,
        expiry: Optional[str],
        expiry_rotated: bool,
        winner_candidate: Optional[Dict[str, Any]],
        scan_diagnostics: Dict[str, Any],
        active_candidate: Optional[Any],
        pending_candidate: Optional[Any],
        is_ready: bool,
        contract_identity_status: str,
        live_quote_age_ms: Optional[float],
        gate_status: str,
        gate_reason: str,
        final_action: str,
        hypothetical_fill_ask_price: Optional[float] = None,
        time_to_fill_ms: Optional[float] = None
    ) -> bool:
        """
        Dual-sink best-effort commit with post-write state commit.
        Invariant: At most one telemetry record per closed bar; zero on logger failure.
        """
        # Idempotency guard: Same bar timestamp cannot advance telemetry commit
        if bar_timestamp is not None and bar_timestamp == self._last_logged_bar_ts:
            return False

        try:
            now_ist = datetime.datetime.now(IST)
            csv_path, jsonl_path = self._get_file_paths()
            self._ensure_csv_header(csv_path)

            winner_sym = winner_candidate.get("symbol") if winner_candidate else None
            inst_key_val = winner_candidate.get("instrument_key") if winner_candidate else None
            strike_val = winner_candidate.get("strike") if winner_candidate else None
            opt_type_val = (
                winner_candidate.get("option_type").value 
                if winner_candidate and hasattr(winner_candidate.get("option_type"), "value") 
                else None
            )
            abs_delta_val = winner_candidate.get("abs_delta") if winner_candidate else None
            spread_pct_val = winner_candidate.get("spread_pct") if winner_candidate else None

            # Authoritative State from Tracker Instance
            active_sym = active_candidate.symbol if active_candidate else None
            pending_sym = pending_candidate.symbol if pending_candidate else None
            conf_count = (
                active_candidate.confirmation_count 
                if active_candidate 
                else (pending_candidate.confirmation_count if pending_candidate else 0)
            )

            record = {
                "run_id": self.run_id,
                "logger_timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "bar_timestamp": str(bar_timestamp),
                "sequence_no": sequence_no,
                "underlying": underlying,
                "spot_ltp": round(spot_ltp, 2) if spot_ltp is not None else None,
                "market_dir": market_dir,
                "trend_score": round(trend_score, 2),
                "rsi_14": round(rsi_14, 2),
                "price_spread": round(price_spread, 2),
                "expiry": expiry,
                "expiry_rotated": expiry_rotated,
                "winner_symbol": winner_sym,
                "instrument_key": inst_key_val,
                "strike": strike_val,
                "option_type": opt_type_val,
                "abs_delta": round(abs_delta_val, 4) if abs_delta_val is not None else None,
                "spread_pct": round(spread_pct_val, 4) if spread_pct_val is not None else None,
                
                # 1:1 Matched Diagnostics Funnel
                "total_strikes": scan_diagnostics.get("total_strikes", 0),
                "direction_candidates": scan_diagnostics.get("direction_candidates", 0),
                "passed_hard_gates": scan_diagnostics.get("passed_hard_gates", 0),
                "rejected_non_finite": scan_diagnostics.get("rejected_non_finite", 0),
                "rejected_iv": scan_diagnostics.get("rejected_iv", 0),
                "rejected_delta": scan_diagnostics.get("rejected_delta", 0),
                "rejected_spread": scan_diagnostics.get("rejected_spread", 0),
                "rejected_liquidity": scan_diagnostics.get("rejected_liquidity", 0),
                
                "active_candidate": active_sym,
                "pending_candidate": pending_sym,
                "confirmation_count": conf_count,
                "is_ready": is_ready,
                "contract_identity_status": contract_identity_status,
                "live_quote_age_ms": round(live_quote_age_ms, 1) if live_quote_age_ms is not None else None,
                "gate_status": gate_status,
                "gate_reason": gate_reason,
                "final_action": final_action,
                "hypothetical_fill_ask_price": round(hypothetical_fill_ask_price, 2) if hypothetical_fill_ask_price is not None else None,
                "time_to_fill_ms": round(time_to_fill_ms, 2) if time_to_fill_ms is not None else None
            }

            # Sink 1: Primary JSONL Journal
            with open(jsonl_path, mode="a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(record) + "\n")

            # Sink 2: Human-Readable CSV Export
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=self.csv_headers)
                writer.writerow(record)

            # Post-write state commit
            self._last_logged_bar_ts = bar_timestamp
            return True

        except Exception:
            # Failsafe: Logger error never leaks to pipeline
            return False
