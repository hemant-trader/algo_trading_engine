import os
import json
import time
import secrets
import datetime
import zoneinfo
import requests
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List, Dict

from config import UPSTOX_CONFIG
from core.types import MarketDirection, OptionType, NormalizedOptionTick
from core.black_scholes import BlackScholesEngine
from processing.tick_guard import TickGuardEngine
from processing.candidate_tracker import CandidateHysteresisTracker
from processing.candidate_scanner import MultiFactorOptionScanner
from engine.pulse_ttl import PulseTTLStateMachine
from engine.safety_gate import ZeroTrustFinalSafetyGate
from engine.expiry_manager import FailClosedExpiryManager
from engine.contract_verifier import ContractIdentityAuthority
from engine.live_execution_quote_provider import LiveExecutionQuoteProvider
from engine.telemetry_logger import HeadlessObservationLogger

st.set_page_config(page_title="Hemant Algo Trading Engine", layout="wide")

TOKEN_FILE = "access_token.json"
IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# Network Latency & Polling Constants
POLL_INTERVAL = 5.0
JITTER_BUFFER = 3.0
INITIAL_TTL_ESTIMATE = max(8.0, POLL_INTERVAL + JITTER_BUFFER + 2.0)
MAX_LIVE_ALIGNMENT_DELAY_SECONDS = 90.0  # Frozen Freshness Policy

INDEX_CONFIG = {
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "step": 50, "strike_mult": 50, "sideways_range": 25.0},
    "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100, "strike_mult": 100, "sideways_range": 60.0},
    "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100, "strike_mult": 100, "sideways_range": 80.0},
}

# Session State Initializations
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = "NIFTY 50"
if "price_history" not in st.session_state:
    st.session_state["price_history"] = []
if "tick_guard" not in st.session_state:
    st.session_state["tick_guard"] = TickGuardEngine()
if "candidate_tracker" not in st.session_state:
    st.session_state["candidate_tracker"] = CandidateHysteresisTracker(hysteresis_delta=5.0, flip_cooldown_seconds=12.0)
if "candidate_scanner" not in st.session_state:
    st.session_state["candidate_scanner"] = MultiFactorOptionScanner()
if "expiry_manager" not in st.session_state:
    st.session_state["expiry_manager"] = FailClosedExpiryManager()
if "quote_provider" not in st.session_state:
    st.session_state["quote_provider"] = LiveExecutionQuoteProvider(max_stale_seconds=3.5, max_clock_skew_seconds=0.5)
if "telemetry_logger" not in st.session_state:
    st.session_state["telemetry_logger"] = HeadlessObservationLogger(log_dir="logs")
if "ttl_manager" not in st.session_state:
    st.session_state["ttl_manager"] = PulseTTLStateMachine(max_ttl=int(INITIAL_TTL_ESTIMATE))
if "seq_counter" not in st.session_state:
    st.session_state["seq_counter"] = 1000
if "last_processed_closed_ts" not in st.session_state:
    st.session_state["last_processed_closed_ts"] = None

def save_token_to_file(token_data):
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f)

def load_token_from_file():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def get_access_token(code):
    url = "https://api.upstox.com/v2/login/authorization/token"
    headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "code": code,
        "client_id": UPSTOX_CONFIG["API_KEY"],
        "client_secret": UPSTOX_CONFIG["API_SECRET"],
        "redirect_uri": UPSTOX_CONFIG["REDIRECT_URI"],
        "grant_type": "authorization_code"
    }
    try:
        response = requests.post(url, headers=headers, data=data, timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_market_quote(instrument_key, token):
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={encoded_key}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

def get_option_chain_data(instrument_key, expiry_date, token):
    url = f"https://api.upstox.com/v2/option/chain?instrument_key={urllib.parse.quote(instrument_key)}&expiry_date={expiry_date}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("status") == "success" and "data" in res_json:
                return res_json["data"]
    except Exception:
        return None
    return None

def calculate_precise_tte(expiry_str):
    try:
        now_ist = datetime.datetime.now(IST)
        exp_date = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        exp_cutoff_ist = datetime.datetime(
            exp_date.year, exp_date.month, exp_date.day, 15, 30, 0, tzinfo=IST
        )
        total_seconds = max(60.0, (exp_cutoff_ist - now_ist).total_seconds())
        return total_seconds / (365.0 * 86400.0)
    except Exception:
        return 0.005

def parse_candle_iso_to_epoch(ts_val: str) -> float:
    if not ts_val:
        return 0.0
    try:
        clean_ts = str(ts_val).replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean_ts)
        return dt.timestamp()
    except Exception:
        return 0.0

def get_unprocessed_closed_5m_bars(instrument_key: str, token: str, last_processed_ts: Optional[str]) -> List[Dict]:
    """
    Fetches 5M candles from Upstox.
    Strictly validates 5-minute grid alignment.
    Strictly excludes the currently forming bar using deterministic wall-clock bucket threshold.
    Returns ordered (oldest to newest) verified closed bars strictly newer than last_processed_ts.
    """
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/5minute"
    headers = {"accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=3.0)
        if resp.status_code != 200:
            return []

        res_json = resp.json()
        raw_candles = res_json.get("data", {}).get("candles", [])
        if not raw_candles:
            return []

        # Current forming bucket cutoff
        now_ist = datetime.datetime.now(IST)
        current_bucket_minute = (now_ist.minute // 5) * 5
        current_bucket_start = now_ist.replace(
            minute=current_bucket_minute, second=0, microsecond=0
        )

        last_epoch = parse_candle_iso_to_epoch(last_processed_ts) if last_processed_ts else 0.0

        closed_candidates = []
        for c in raw_candles:
            if len(c) < 6:
                continue
            c_ts = str(c[0])
            c_epoch = parse_candle_iso_to_epoch(c_ts)
            if c_epoch <= 0:
                continue

            c_dt = datetime.datetime.fromtimestamp(c_epoch, tz=IST)

            # Strict 5-Minute Grid Boundary Alignment
            bucket_minute = (c_dt.minute // 5) * 5
            expected_bucket = c_dt.replace(minute=bucket_minute, second=0, microsecond=0)
            if c_dt != expected_bucket:
                continue

            # Exclude forming bucket
            if c_dt >= current_bucket_start:
                continue

            # Watermark check
            if c_epoch > last_epoch:
                closed_candidates.append({
                    "timestamp": c_ts,
                    "epoch": c_epoch,
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": int(c[5])
                })

        # Ascending chronological sort (oldest
