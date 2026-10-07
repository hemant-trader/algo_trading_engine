import os
import base64
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

POLL_INTERVAL = 5.0
JITTER_BUFFER = 3.0
INITIAL_TTL_ESTIMATE = max(8.0, POLL_INTERVAL + JITTER_BUFFER + 2.0)
MAX_LIVE_ALIGNMENT_DELAY_SECONDS = 90.0

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

# Fluctuation-free Pre-Alert Buffer State
if "pre_alert_direction" not in st.session_state:
    st.session_state["pre_alert_direction"] = "NEUTRAL"
if "pre_alert_tick_count" not in st.session_state:
    st.session_state["pre_alert_tick_count"] = 0
if "cached_itm_candidate" not in st.session_state:
    st.session_state["cached_itm_candidate"] = None

def get_avatar_base64():
    """Finds avatar.png in current or assets folder and returns base64 string"""
    paths = ["avatar.png", "assets/avatar.png"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as img_f:
                    return base64.b64encode(img_f.read()).decode()
            except Exception:
                pass
    return None

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

            bucket_minute = (c_dt.minute // 5) * 5
            expected_bucket = c_dt.replace(minute=bucket_minute, second=0, microsecond=0)
            if c_dt != expected_bucket:
                continue

            if c_dt >= current_bucket_start:
                continue

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

        closed_candidates.sort(key=lambda x: x["epoch"])
        return closed_candidates

    except Exception:
        return []

saved_token_data = load_token_from_file()
if saved_token_data and "access_token" in saved_token_data:
    st.session_state["access_token"] = saved_token_data["access_token"]

query_params = st.query_params
auth_code = query_params.get("code", None)
incoming_state = query_params.get("state", None)

if auth_code and "access_token" not in st.session_state:
    expected_state = st.session_state.get("oauth_state")
    if expected_state and incoming_state != expected_state:
        st.error("OAuth State Validation Failed. Possible CSRF interception.")
        st.stop()
    else:
        res = get_access_token(auth_code)
        if "access_token" in res:
            st.session_state["access_token"] = res["access_token"]
            save_token_to_file(res)

current_token = st.session_state.get("access_token", None)
curr_inst_key = INDEX_CONFIG[st.session_state["selected_index"]]["key"]

active_expiry = None
active_contracts = []
expiry_rotated = False
expiry_status = "TOKEN_MISSING"

if current_token:
    active_expiry, active_contracts, expiry_rotated, expiry_status = st.session_state["expiry_manager"].get_active_expiry_and_contracts(
        curr_inst_key, current_token
    )

# ================= SIDEBAR CONTROLS =================
now_ist = datetime.datetime.now(IST)
current_date_str = now_ist.strftime("%d %b %Y")
current_time_str = now_ist.strftime("%I:%M:%S %p")
today_iso = now_ist.strftime("%Y-%m-%d")

if active_expiry == today_iso:
    expiry_tag = '<span style="color:#ff4d4f; font-weight:700;">🔥 TODAY EXPIRY (0 DTE)</span>'
elif active_expiry:
    try:
        exp_d = datetime.datetime.strptime(active_expiry, "%Y-%m-%d").date()
        exp_day_month = exp_d.strftime("%d/%m")
        days_left = (exp_d - now_ist.date()).days
        days_tag = f"({days_left}d)" if days_left > 0 else ""
        expiry_tag = f'Expiry: <b style="color:#64ffda;">{exp_day_month}</b> {days_tag}'
    except Exception:
        expiry_tag = f'Expiry: <b style="color:#64ffda;">{active_expiry}</b>'
else:
    expiry_tag = f'<span style="color:#ff4d4f;">⚠️ {expiry_status}</span>'

sidebar_top_badge = (
    f'<div style="background-color:#111622; border:1px solid #1f293d; border-radius:8px; padding:8px 10px; margin-bottom:12px; font-size:11px;">'
    f'<div style="display:flex; justify-content:space-between; color:#8b949e; font-family:monospace; margin-bottom:4px;">'
    f'<span>📅 {current_date_str}</span><span>🕒 {current_time_str}</span>'
    f'</div>'
    f'<div style="border-top:1px solid #1f293d; padding-top:4px; text-align:center; color:#ccd6f6;">'
    f'{expiry_tag}'
    f'</div>'
    f'</div>'
)
st.sidebar.markdown(sidebar_top_badge, unsafe_allow_html=True)

st.sidebar.title("⚙️ System & Trade Control")
st.sidebar.markdown("---")

trade_mode = st.sidebar.radio(
    "Execution State",
    [
        "🔴 OFF: Watch & Signal Mode (Paper Mode)",
        "🟡 ARMED: Execution Simulation / Paper Monitor"
    ],
    index=0
)

st.sidebar.markdown("---")
index_keys = list(INDEX_CONFIG.keys())
saved_idx_pos = index_keys.index(st.session_state["selected_index"]) if st.session_state["selected_index"] in index_keys else 0

selected_index = st.sidebar.selectbox(
    "Select Index", 
    index_keys, 
    index=saved_idx_pos,
    key="selected_index_dropdown"
)

if selected_index != st.session_state["selected_index"]:
    st.session_state["selected_index"] = selected_index
    st.session_state["price_history"] = []
    st.session_state["candidate_tracker"].flush()
    st.session_state["expiry_manager"].flush_cache()
    st.session_state["last_processed_closed_ts"] = None
    st.session_state["pre_alert_direction"] = "NEUTRAL"
    st.session_state["pre_alert_tick_count"] = 0
    st.session_state["cached_itm_candidate"] = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Settings")
max_loss = st.sidebar.number_input("Max Daily Loss (₹)", value=2000)
target_pnl = st.sidebar.number_input("Daily Target PnL (₹)", value=4000)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Broker Authentication")
if "access_token" not in st.session_state:
    if "oauth_state" not in st.session_state:
        st.session_state["oauth_state"] = secrets.token_urlsafe(32)
    base_url = "https://api.upstox.com/v2/login/authorization/dialog"
    params = {
        "response_type": "code",
        "client_id": UPSTOX_CONFIG["API_KEY"],
        "redirect_uri": UPSTOX_CONFIG["REDIRECT_URI"],
        "state": st.session_state["oauth_state"]
    }
    login_url = f"{base_url}?{urllib.parse.urlencode(params)}"
    st.sidebar.link_button("Login with Upstox", login_url)
    
    manual_code = st.sidebar.text_input("Paste Authorization Code Here:")
    if st.sidebar.button("Connect"):
        if manual_code:
            res = get_access_token(manual_code)
            if "access_token" in res:
                st.session_state["access_token"] = res["access_token"]
                save_token_to_file(res)
                st.rerun()
else:
    st.sidebar.success("✅ Upstox Session Active")
    if st.sidebar.button("Logout / Reset Session"):
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
        del st.session_state["access_token"]
        st.session_state["price_history"] = []
        st.session_state["candidate_tracker"].flush()
        st.session_state["expiry_manager"].flush_cache()
        st.session_state["last_processed_closed_ts"] = None
        st.session_state["pre_alert_direction"] = "NEUTRAL"
        st.session_state["pre_alert_tick_count"] = 0
        st.session_state["cached_itm_candidate"] = None
        st.rerun()

# ================= TECHNICAL ENGINE =================
def calculate_rsi(price_history, period=14):
    if len(price_history) < period + 1:
        return 50.0
    s = pd.Series(price_history)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0 or np.isnan(last_loss):
        return 100.0 if avg_gain.iloc[-1] > 0 else 50.0
    rs = avg_gain.iloc[-1] / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def calculate_realized_volatility(price_history):
    if len(price_history) < 5:
        return 12.85
    s = pd.Series(price_history)
    log_returns = np.log(s / s.shift(1)).dropna()
    std = log_returns.std()
    if np.isnan(std) or std == 0:
        return 12.85
    annualized_vol = float(std * np.sqrt(252 * 375 * 12) * 100.0)
    return round(max(8.0, min(annualized_vol, 35.0)), 2)

def evaluate_regime_and_direction(price_history, selected_index):
    threshold_range = INDEX_CONFIG[selected_index]["sideways_range"]
    if len(price_history) < 15:
        return MarketDirection.NEUTRAL, 0, 0.0, 50.0, 0.0, "INSUFFICIENT_DATA", threshold_range

    s = pd.Series(price_history)
    price_spread = float(s.max() - s.min())
    ema5 = float(s.ewm(span=5, adjust=False).mean().iloc[-1])
    ema15 = float(s.ewm(span=15, adjust=False).mean().iloc[-1])
    current_price = float(s.iloc[-1])
    price_lookback = float(s.iloc[-6]) if len(s) >= 6 else float(s.iloc[0])
    
    momentum_pct = ((current_price - price_lookback) / price_lookback) * 100.0
    rsi = calculate_rsi(price_history, period=min(14, len(price_history)-1))

    if price_spread <= threshold_range and abs(momentum_pct) < 0.04:
        return MarketDirection.NEUTRAL, 0, 0.0, rsi, price_spread, "SIDEWAYS_RANGEBOUND", threshold_range

    bull_score = 0
    bear_score = 0

    if ema5 > ema15:
        bull_score += 2
    elif ema5 < ema15:
        bear_score += 2

    if momentum_pct > 0.03:
        bull_score += 2
    elif momentum_pct < -0.03:
        bear_score += 2

    if current_price > ema5:
        bull_score += 1
    elif current_price < ema5:
        bear_score += 1

    if rsi >= 55.0:
        bull_score += 2
    elif rsi <= 45.0:
        bear_score += 2

    TOTAL_MAX_POINTS = 7.0

    if bull_score >= 5 and bull_score >= (bear_score + 2):
        trend_strength = (bull_score / TOTAL_MAX_POINTS) * 100.0
        return MarketDirection.BULLISH, bull_score, trend_strength, rsi, price_spread, "TRENDING_BULLISH", threshold_range
    elif bear_score >= 5 and bear_score >= (bull_score + 2):
        trend_strength = (bear_score / TOTAL_MAX_POINTS) * 100.0
        return MarketDirection.BEARISH, bear_score, trend_strength, rsi, price_spread, "TRENDING_BEARISH", threshold_range

    return MarketDirection.NEUTRAL, 0, 0.0, rsi, price_spread, "CHOPPY_NO_TREND", threshold_range

# ================= CUSTOM CSS FOR PULSING PRE-ALERT =================
st.markdown("""
<style>
@keyframes greenPulse {
    0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
    100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
}
@keyframes redPulse {
    0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); }
    100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
}
.pre-alert-bullish {
    background: linear-gradient(90deg, rgba(46,204,113,0.18), rgba(22,31,48,0.8));
    border: 1px solid #2ecc71;
    border-radius: 6px;
    padding: 7px 12px;
    margin-top: 8px;
    color: #2ecc71;
    font-size: 12px;
    font-weight: 700;
    animation: greenPulse 2s infinite;
}
.pre-alert-bearish {
    background: linear-gradient(90deg, rgba(231,76,60,0.18), rgba(22,31,48,0.8));
    border: 1px solid #e74c3c;
    border-radius: 6px;
    padding: 7px 12px;
    margin-top: 8px;
    color: #ff6b6b;
    font-size: 12px;
    font-weight: 700;
    animation: redPulse 2s infinite;
}
.pre-alert-neutral {
    background-color: #111622;
    border: 1px solid #1f293d;
    border-radius: 6px;
    padding: 6px 12px;
    margin-top: 8px;
    color: #8892b0;
    font-size: 11px;
}
</style>
""", unsafe_allow_html=True)

# ================= MAIN RUNTIME WITH RIGHT-ALIGNED SEAMLESS AVATAR =================
avatar_b64 = get_avatar_base64()

if avatar_b64:
    header_html = (
        f'<div style="display:flex; justify-content:space-between; align-items:center; width:100%; margin-bottom:16px; padding:6px 0;">'
        f'  <div style="display:flex; align-items:center; gap:10px;">'
        f'    <span style="font-size:36px; line-height:1;">⚡</span>'
        f'    <h1 style="margin:0; font-size:36px; font-weight:900; line-height:1.1; letter-spacing:-0.5px;">'
        f'      <span style="color:#111111;">Hemant </span>'
        f'      <span style="color:#e74c3c;">Algo </span>'
        f'      <span style="color:#111111;">Trading Engine</span>'
        f'    </h1>'
        f'  </div>'
        f'  <div style="flex-shrink:0; margin-right:8
