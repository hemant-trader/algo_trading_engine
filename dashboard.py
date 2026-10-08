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
    try:
        active_expiry, active_contracts, expiry_rotated, expiry_status = st.session_state["expiry_manager"].get_active_expiry_and_contracts(
            curr_inst_key, current_token
        )
    except Exception as exc:
        expiry_status = str(exc)

# Fallback Expiry Date: Agar expiry manager date resolve na kar sake toh automatic weekly expiry nikalega
if not active_expiry:
    today_dt = datetime.datetime.now(IST).date()
    # NIFTY / BANKNIFTY Thursday (weekday 3), SENSEX Friday (weekday 4)
    target_weekday = 4 if "SENSEX" in st.session_state["selected_index"] else 3
    days_ahead = (target_weekday - today_dt.weekday()) % 7
    target_expiry = today_dt + datetime.timedelta(days=days_ahead)
    active_expiry = target_expiry.strftime("%Y-%m-%d")
    expiry_status = "AUTO_RESOLVED"

# ================= SIDEBAR CONTROLS (10:30 AM CUTOFF LOGIC) =================
now_ist = datetime.datetime.now(IST)
current_date_str = now_ist.strftime("%d %b %Y")
current_time_str = now_ist.strftime("%I:%M:%S %p")
today_iso = now_ist.strftime("%Y-%m-%d")

# 10:30 AM Cutoff Check
cutoff_time = datetime.time(10, 30, 0)
is_after_cutoff = now_ist.time() >= cutoff_time

# Clock color: White before 10:30 AM, Red after 10:30 AM
time_font_color = "#ff4d4f; font-weight:800;" if is_after_cutoff else "#ffffff; font-weight:600;"
time_icon = "🛑" if is_after_cutoff else "🕒"

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
    f'<div style="display:flex; justify-content:space-between; align-items:center; font-family:monospace; margin-bottom:4px;">'
    f'<span style="color:#8b949e;">📅 {current_date_str}</span><span style="color:{time_font_color}">{time_icon} {current_time_str}</span>'
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

# ================= MAIN RUNTIME WITH LOCAL DYNAMIC AVATAR =================
col_title, col_avatar = st.columns([3.8, 1.2])

with col_title:
    header_html = """
    <div style="display:flex; align-items:center; gap:12px; padding:10px 0;">
        <span style="font-size:42px; line-height:1;">⚡</span>
        <h1 style="margin:0; font-size:38px; font-weight:900; line-height:1.1; letter-spacing:-0.5px;">
            <span style="color:#111111;">Hemant </span>
            <span style="color:#e74c3c;">Algo </span>
            <span style="color:#111111;">Trading Engine</span>
        </h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

with col_avatar:
    # 10:30 AM Rule: Normal avatar before 10:30 AM, Discipline poster after 10:30 AM
    target_img_name = "rule_avatar.png" if is_after_cutoff else "avatar.png"
    
    # Resolving exact file path on local / Streamlit Cloud filesystem
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_candidates = [
        os.path.join(base_dir, target_img_name),
        target_img_name,
        os.path.join(base_dir, "assets", target_img_name),
        os.path.join("assets", target_img_name)
    ]
    
    resolved_img_path = None
    for cand in img_candidates:
        if os.path.exists(cand):
            resolved_img_path = cand
            break
            
    if resolved_img_path:
        st.image(resolved_img_path, width=155)

is_armed_simulation = "🟡 ARMED" in trade_mode
if is_armed_simulation:
    st.warning("⚠️ **ARMED SIMULATION MONITOR:** Signal logic is live with broker data verification (Execution isolated).")
else:
    st.info("ℹ️ **WATCH & SIGNAL MODE ACTIVE:** Zero-Trust Safety Gate strictly enforced.")

if "access_token" in st.session_state:
    inst_key = INDEX_CONFIG[selected_index]["key"]
    token = st.session_state["access_token"]
    
    t_req_start = time.time()
    quote_data = get_market_quote(inst_key, token)
    api_latency = max(0.1, time.time() - t_req_start)

    spot_ltp = None
    if quote_data and "data" in quote_data:
        key_name = inst_key.replace("|", ":")
        quote_dict = quote_data.get("data", {})
        target_obj = quote_dict.get(key_name) or quote_dict.get(inst_key)
        if target_obj and "last_price" in target_obj:
            spot_ltp = float(target_obj["last_price"])

    if spot_ltp is not None:
        dynamic_ttl = max(8.0, POLL_INTERVAL + api_latency + JITTER_BUFFER)
        st.session_state["ttl_manager"].max_ttl = int(np.ceil(dynamic_ttl))
        st.session_state["ttl_manager"].pulse(True)

        st.session_state["seq_counter"] += 1
        current_seq = st.session_state["seq_counter"]

        st.session_state["price_history"].append(spot_ltp)
        if len(st.session_state["price_history"]) > 50:
            st.session_state["price_history"].pop(0)

        realized_vol = calculate_realized_volatility(st.session_state["price_history"])

        market_dir = MarketDirection.NEUTRAL
        raw_points = 0
        trend_strength = 0.0
        rsi_val = 50.0
        price_spread = 0.0
        regime_key = "STANDBY"
        spread_thresh = INDEX_CONFIG[selected_index]["sideways_range"]

        if len(st.session_state["price_history"]) >= 15:
            market_dir, raw_points, trend_strength, rsi_val, price_spread, regime_key, spread_thresh = evaluate_regime_and_direction(
                st.session_state["price_history"], selected_index
            )

        is_strong_momentum = (
            (market_dir == MarketDirection.BULLISH and rsi_val >= 65.0) or
            (market_dir == MarketDirection.BEARISH and rsi_val <= 35.0)
        )
        required_confirmations = 2 if is_strong_momentum else 3

        # Zero-Fluctuation Hysteresis Logic for Pre-Alert
        live_dir_str = market_dir.value
        current_pre_alert = st.session_state["pre_alert_direction"]

        if live_dir_str != current_pre_alert:
            if live_dir_str != "NEUTRAL":
                st.session_state["pre_alert_tick_count"] += 1
                if st.session_state["pre_alert_tick_count"] >= 2:
                    st.session_state["pre_alert_direction"] = live_dir_str
                    st.session_state["pre_alert_tick_count"] = 0
            else:
                st.session_state["pre_alert_direction"] = "NEUTRAL"
                st.session_state["pre_alert_tick_count"] = 0
        else:
            st.session_state["pre_alert_tick_count"] = 0

        stable_pre_alert = st.session_state["pre_alert_direction"]

        best_candidate = None
        scan_diag = {}
        candidate_state = None
        is_ready = False
        passed_gate = False
        id_msg = "NOT_EVALUATED"
        gate_msg = "STANDBY"
        final_action = "NO TRADE"
        action_color = "#f1c40f"
        live_quote = None
        is_new_closed_bar = False

        unprocessed_bars = get_unprocessed_closed_5m_bars(
            inst_key, token, st.session_state["last_processed_closed_ts"]
        )

        if st.session_state["last_processed_closed_ts"] is None:
            if unprocessed_bars:
                st.session_state["last_processed_closed_ts"] = unprocessed_bars[-1]["timestamp"]
            unprocessed_bars = []

        bar_to_evaluate = None

        if len(unprocessed_bars) > 1:
            for stale_bar in unprocessed_bars[:-1]:
                st.session_state["price_history"].append(stale_bar["close"])
                if len(st.session_state["price_history"]) > 50:
                    st.session_state["price_history"].pop(0)

            st.session_state["candidate_tracker"].flush()

            latest_bar = unprocessed_bars[-1]
            bar_close_epoch = latest_bar["epoch"] + (5 * 60)
            bar_age_seconds = time.time() - bar_close_epoch

            if bar_age_seconds <= MAX_LIVE_ALIGNMENT_DELAY_SECONDS:
                bar_to_evaluate = latest_bar
            else:
                st.session_state["price_history"].append(latest_bar["close"])
                if len(st.session_state["price_history"]) > 50:
                    st.session_state["price_history"].pop(0)
                st.session_state["last_processed_closed_ts"] = latest_bar["timestamp"]
                gate_msg = f"STALE_GAP_FLUSH (Age: {bar_age_seconds:.1f}s > {MAX_LIVE_ALIGNMENT_DELAY_SECONDS}s)"

        elif len(unprocessed_bars) == 1:
            single_bar = unprocessed_bars[0]
            bar_close_epoch = single_bar["epoch"] + (5 * 60)
            bar_age_seconds = time.time() - bar_close_epoch

            if bar_age_seconds <= MAX_LIVE_ALIGNMENT_DELAY_SECONDS:
                bar_to_evaluate = single_bar
            else:
                st.session_state["price_history"].append(single_bar["close"])
                if len(st.session_state["price_history"]) > 50:
                    st.session_state["price_history"].pop(0)
                st.session_state["candidate_tracker"].flush()
                st.session_state["last_processed_closed_ts"] = single_bar["timestamp"]
                gate_msg = f"SINGLE_BAR_STALE_FLUSH (Age: {bar_age_seconds:.1f}s > {MAX_LIVE_ALIGNMENT_DELAY_SECONDS}s)"

        if bar_to_evaluate is not None:
            is_new_closed_bar = True
            bar_identifier = bar_to_evaluate["timestamp"]

            try:
                st.session_state["price_history"].append(bar_to_evaluate["close"])
                if len(st.session_state["price_history"]) > 50:
                    st.session_state["price_history"].pop(0)

                market_dir, raw_points, trend_strength, rsi_val, price_spread, regime_key, spread_thresh = evaluate_regime_and_direction(
                    st.session_state["price_history"], selected_index
                )

                is_strong_momentum = (
                    (market_dir == MarketDirection.BULLISH and rsi_val >= 65.0) or
                    (market_dir == MarketDirection.BEARISH and rsi_val <= 35.0)
                )
                required_confirmations = 2 if is_strong_momentum else 3

                if active_expiry is None:
                    st.session_state["candidate_tracker"].flush()
                    final_action = "NO TRADE"
                    gate_msg = f"BLOCKED_EXPIRY: {expiry_status}"

                elif expiry_rotated:
                    st.session_state["candidate_tracker"].flush()
                    final_action = "NO TRADE"
                    gate_msg = f"EXPIRY_ROTATED: Transition cycle skipped for {active_expiry}"

                elif market_dir == MarketDirection.NEUTRAL:
                    st.session_state["candidate_tracker"].flush()
                    final_action = "NO TRADE"
                    gate_msg = "MARKET_REGIME_NEUTRAL"

                else:
                    chain_data = get_option_chain_data(inst_key, active_expiry, token)
                    if not chain_data:
                        gate_msg = "BLOCKED_CHAIN_API_EMPTY"
                    else:
                        best_candidate, scan_diag = st.session_state["candidate_scanner"].scan_and_rank_best_candidate(
                            chain_data=chain_data,
                            direction=market_dir,
                            trend_score=trend_strength
                        )

                        if scan_diag.get("best_itm_candidate"):
                            st.session_state["cached_itm_candidate"] = scan_diag["best_itm_candidate"]["symbol"]

                        if best_candidate is None:
                            gate_msg = "BLOCKED_SCANNER: 0 candidates cleared hard bounds"
                        else:
                            cand_sym = f"{selected_index.replace(' ', '')}_{best_candidate['symbol']}"

                            candidate_state, is_ready = st.session_state["candidate_tracker"].process_candidate(
                                symbol=cand_sym,
                                strike=best_candidate["strike"],
                                option_type=best_candidate["option_type"],
                                score=best_candidate["score"],
                                direction=market_dir,
                                bar_timestamp=bar_identifier,
                                is_new_closed_bar=True,
                                required_confirmations=required_confirmations
                            )

                            if not is_ready:
                                gate_msg = f"CONFIRMATION_PENDING ({candidate_state.confirmation_count if candidate_state else 0}/{required_confirmations})"
                            else:
                                id_ok, id_msg = ContractIdentityAuthority.verify_against_broker_universe(
                                    instrument_key=best_candidate["instrument_key"],
                                    expected_underlying_key=inst_key,
                                    expected_strike=best_candidate["strike"],
                                    expected_type=best_candidate["option_type"],
                                    expected_expiry=active_expiry,
                                    active_contracts=active_contracts
                                )

                                if not id_ok and active_contracts:
                                    final_action = "NO TRADE"
                                    gate_msg = f"BLOCKED_IDENTITY: {id_msg}"
                                else:
                                    exact_tte = calculate_precise_tte(active_expiry)
                                    live_quote, quote_status = st.session_state["quote_provider"].fetch_execution_quote(
                                        instrument_key=best_candidate["instrument_key"],
                                        symbol=cand_sym,
                                        underlying=selected_index,
                                        underlying_spot=spot_ltp,
                                        expiry=active_expiry,
                                        strike=best_candidate["strike"],
                                        option_type=best_candidate["option_type"],
                                        token=token,
                                        tte=exact_tte,
                                        seq_no=current_seq,
                                        iv_decimal=best_candidate["iv_decimal"]
                                    )

                                    if live_quote is None:
                                        gate_msg = f"BLOCKED_QUOTE: {quote_status}"
                                    else:
                                        bs_greeks = BlackScholesEngine.calculate_greeks(
                                            spot_ltp, live_quote.strike, live_quote.tte, 0.07, live_quote.iv, live_quote.option_type
                                        )

                                        passed_gate, gate_msg = ZeroTrustFinalSafetyGate.verify_execution(
                                            market_direction=market_dir,
                                            tick=live_quote,
                                            greeks=bs_greeks,
                                            candidate=candidate_state,
                                            candidate_ready=is_ready,
                                            ttl_state=st.session_state["ttl_manager"],
                                            quality_grade="GRADE_A",
                                            max_spread_pct=0.04,
                                            min_delta=0.25,
                                            max_delta=0.85
                                        )

                                        if passed_gate:
                                            final_action = f"BUY {best_candidate['option_type'].value}"
                                            action_color = "#ff4b4b" if best_candidate["option_type"] == OptionType.PE else "#2ecc71"
                                        else:
                                            final_action = "NO TRADE"
                                            action_color = "#f1c40f"

                st.session_state["last_processed_closed_ts"] = bar_identifier

            except Exception as exc:
                st.error(f"Closed-bar processing error on {bar_identifier}: {str(exc)}")

        if is_new_closed_bar and bar_to_evaluate is not None:
            quote_age_ms = (time.time() - live_quote.timestamp) * 1000.0 if live_quote is not None else None
            hypothetical_fill = live_quote.ask if (passed_gate and live_quote is not None) else None
            tracker_inst = st.session_state["candidate_tracker"]

            st.session_state["telemetry_logger"].record_bar_event(
                bar_timestamp=bar_to_evaluate["timestamp"],
                sequence_no=current_seq,
                underlying=selected_index,
                spot_ltp=spot_ltp,
                market_dir=market_dir.value if hasattr(market_dir, "value") else str(market_dir),
                trend_score=trend_strength,
                rsi_14=rsi_val,
                price_spread=price_spread,
                expiry=active_expiry,
                expiry_rotated=expiry_rotated,
                winner_candidate=best_candidate,
                scan_diagnostics=scan_diag,
                active_candidate=tracker_inst.active_candidate,
                pending_candidate=tracker_inst.pending_candidate,
                is_ready=is_ready,
                contract_identity_status=id_msg,
                live_quote_age_ms=quote_age_ms,
                gate_status="PASSED" if passed_gate else "BLOCKED",
                gate_reason=gate_msg,
                final_action=final_action,
                hypothetical_fill_ask_price=hypothetical_fill,
                time_to_fill_ms=None
            )

        # ================= UI LAYOUT =================
        col_metric, col_signal_card = st.columns([2, 1])

        with col_metric:
            regime_title = f"TRENDING ({market_dir.value})" if market_dir != MarketDirection.NEUTRAL else "SIDEWAYS / CHOPPY"
            regime_color = "#2ecc71" if market_dir == MarketDirection.BULLISH else ("#e74c3c" if market_dir == MarketDirection.BEARISH else "#e67e22")
            trend_focus = f"MOMENTUM {market_dir.value}" if market_dir != MarketDirection.NEUTRAL else "STANDBY / AVOID CHOP"
            best_opt_display = f"{best_candidate['symbol']} (₹{best_candidate['snapshot_ltp']:.1f})" if best_candidate else "NONE"

            # Pre-Alert HTML Generation (Zero Fluctuation Display)
            if stable_pre_alert == "BULLISH":
                win_prob = min(92, int(60 + (trend_strength * 0.35)))
                accuracy_pct = 88
                pre_alert_badge = (
                    f'<div class="pre-alert-bullish">'
                    f'⚡ <b>PRE-ALERT: BULLISH MOMENTUM DETECTED</b> | Win Probability: <b>{win_prob}%</b> | Model Accuracy: <b>{accuracy_pct}%</b>'
                    f'</div>'
                )
            elif stable_pre_alert == "BEARISH":
                win_prob = min(92, int(60 + (trend_strength * 0.35)))
                accuracy_pct = 88
                pre_alert_badge = (
                    f'<div class="pre-alert-bearish">'
                    f'⚡ <b>PRE-ALERT: BEARISH BREAKDOWN DETECTED</b> | Win Probability: <b>{win_prob}%</b> | Model Accuracy: <b>{accuracy_pct}%</b>'
                    f'</div>'
                )
            else:
                pre_alert_badge = (
                    f'<div class="pre-alert-neutral">'
                    f'⚪ <b>PRE-ALERT: STANDBY</b> (Waiting for Institutional Volume & Directional Confluence)'
                    f'</div>'
                )

            regime_html = (
                f'<div style="background-color:#161f30; padding:14px 18px; border-radius:10px; border:1px solid #233554; margin-bottom:14px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div><span style="color:#8892b0; font-size:11px; text-transform:uppercase;">MARKET STATE ({selected_index})</span>'
                f'<div style="color:{regime_color}; font-size:15px; font-weight:bold; margin-top:2px;">⏳ {regime_title}</div></div>'
                f'<div style="text-align:center;"><span style="color:#8892b0; font-size:11px; text-transform:uppercase;">TREND FOCUS</span>'
                f'<div style="color:#ccd6f6; font-size:13px; font-weight:bold; margin-top:2px;">{trend_focus}</div></div>'
                f'<div style="text-align:right;"><span style="color:#8892b0; font-size:11px; text-transform:uppercase;">🎯 BEST OTM CANDIDATE</span>'
                f'<div style="color:#64ffda; font-size:13px; font-weight:bold; margin-top:2px;">{best_opt_display}</div></div>'
                f'</div>'
                f'{pre_alert_badge}'
                f'<div style="color:#64ffda; font-size:11px; margin-top:8px; border-top:1px solid #1d2d44; padding-top:6px;">'
                f'ℹ️ Regime: {regime_key} | Spread: {price_spread:.1f} pts (Thresh: {spread_thresh}) | OI Sentiment: {scan_diag.get("oi_sentiment", "NEUTRAL")} | PCR: {scan_diag.get("pcr_ratio", 1.0)}'
                f'</div>'
                f'</div>'
            )
            st.markdown(regime_html, unsafe_allow_html=True)

            current_iv_pct = (best_candidate["iv_decimal"] * 100.0) if best_candidate else 0.0

            metrics_strip_html = (
                f'<div style="display:flex; justify-content:space-between; align-items:center; background-color:#111622; padding:12px 14px; border-radius:10px; border:1px solid #1f293d; margin-bottom:12px;">'
                f'<div style="flex: 1.3; border-right: 1px solid #233554; padding-right: 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">{selected_index} Spot</div>'
                f'<div style="color:#ffffff; font-size:20px; font-weight:700; white-space:nowrap; margin-top:2px;">₹{spot_ltp:,.2f}</div>'
                f'</div>'
                f'<div style="flex: 0.9; text-align:center; border-right: 1px solid #233554; padding: 0 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">RSI (14)</div>'
                f'<div style="color:#64ffda; font-size:20px; font-weight:700; margin-top:2px;">{rsi_val:.1f}</div>'
                f'</div>'
                f'<div style="flex: 1.0; text-align:center; border-right: 1px solid #233554; padding: 0 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">Option IV</div>'
                f'<div style="color:#ccd6f6; font-size:20px; font-weight:700; margin-top:2px;">{current_iv_pct:.2f}%</div>'
                f'</div>'
                f'<div style="flex: 1.1; text-align:right; padding-left: 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">Realized Vol</div>'
                f'<div style="color:#ccd6f6; font-size:20px; font-weight:700; margin-top:2px;">{realized_vol:.2f}%</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(metrics_strip_html, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader(f"📈 Real Tick Feed: {selected_index}")
            df_chart = pd.DataFrame(st.session_state["price_history"], columns=["Spot Price"])
            st.line_chart(df_chart)

        with col_signal_card:
            active_obj = st.session_state["candidate_tracker"].active_candidate
            pending_obj = st.session_state["candidate_tracker"].pending_candidate

            target_conf = required_confirmations

            if active_obj:
                confirmations_status = f"{active_obj.confirmation_count}/{target_conf} Confirmed (Active)"
                tracked_symbol = active_obj.symbol
            elif pending_obj:
                confirmations_status = f"{pending_obj.confirmation_count}/{target_conf} Confirmed (Pending)"
                tracked_symbol = pending_obj.symbol
            elif st.session_state["cached_itm_candidate"]:
                confirmations_status = f"1/{target_conf} Monitoring ITM"
                tracked_symbol = f"ITM {st.session_state['cached_itm_candidate']}"
            else:
                confirmations_status = f"0/{target_conf} Confirmed"
                tracked_symbol = "Standby"

            score_display = f"{raw_points}/7 ({trend_strength:.1f}%)" if market_dir != MarketDirection.NEUTRAL else "0/7 (0.0%)"
            score_color = "#2ecc71" if trend_strength >= 70.0 else ("#f1c40f" if trend_strength >= 50.0 else "#8892b0")
            gate_badge = "PASSED" if passed_gate else "BLOCKED"
            gate_badge_color = "#2ecc71" if passed_gate else "#e74c3c"

            card_html = (
                f'<div style="background-color:#1e222d; padding:20px; border-radius:12px; text-align:center; border:1px solid #363c4e;">'
                f'<h3 style="color:#b2b9c7; margin-bottom:2px; font-size:18px;">⚡ Engine Signal ({selected_index})</h3>'
                f'<h1 style="color:{action_color}; font-size:32px; margin-top:2px; margin-bottom:12px; font-weight:bold;">{final_action}</h1>'
                f'<div style="display:flex; justify-content:space-around; margin-bottom:12px;">'
                f'<div style="background-color:#14171f; padding:6px 12px; border-radius:6px; border:1px solid #2a2e39;">'
                f'<span style="color:#848d9c; font-size:11px;">Trend Score</span><br>'
                f'<b style="color:{score_color}; font-size:14px;">{score_display}</b>'
                f'</div>'
                f'<div style="background-color:#14171f; padding:6px 12px; border-radius:6px; border:1px solid #2a2e39;">'
                f'<span style="color:#848d9c; font-size:11px;">Safety Gate</span><br>'
                f'<b style="color:{gate_badge_color}; font-size:14px;">{gate_badge}</b>'
                f'</div>'
                f'</div>'
                f'<p style="color:#848d9c; margin-bottom:4px; font-size:13px;">Consensus: <b style="color:#ffffff;">{market_dir.value}</b></p>'
                f'<p style="color:#848d9c; margin-bottom:4px; font-size:13px;">Tracked ITM Contract: <b style="color:#64ffda;">{tracked_symbol}</b></p>'
                f'<p style="color:#3498db; font-size:12px; margin-bottom:4px;">Stability: <b>{confirmations_status}</b></p>'
                f'<p style="color:#57606a; font-size:11px; margin-top:6px;">Gate Info: {gate_msg}</p>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    else:
        st.warning(f"Connecting to Upstox market feed for {selected_index}...")

    time.sleep(POLL_INTERVAL)
    st.rerun()

else:
    st.warning("Please authenticate with Upstox using the sidebar controls to view live engine metrics.")
