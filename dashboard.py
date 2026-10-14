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
from core.types import MarketDirection, OptionType
from core.black_scholes import BlackScholesEngine
from processing.candidate_tracker import CandidateHysteresisTracker
from processing.candidate_scanner import MultiFactorOptionScanner
from engine.pulse_ttl import PulseTTLStateMachine
from engine.safety_gate import ZeroTrustFinalSafetyGate
from engine.expiry_manager import FailClosedExpiryManager
from engine.contract_verifier import ContractIdentityAuthority
from engine.live_execution_quote_provider import LiveExecutionQuoteProvider
from engine.telemetry_logger import HeadlessObservationLogger

# Wire Hardened Institutional Core Engine
from engine_orchestrator import EngineOrchestrator

st.set_page_config(page_title="Hemant Algo Trading Engine", layout="wide")

TOKEN_FILE = "access_token.json"
IST = zoneinfo.ZoneInfo("Asia/Kolkata")

POLL_INTERVAL = 3.0
JITTER_BUFFER = 2.0
INITIAL_TTL_ESTIMATE = max(8.0, POLL_INTERVAL + JITTER_BUFFER + 2.0)

INDEX_CONFIG = {
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "step": 50, "strike_mult": 50, "sideways_range": 25.0},
    "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100, "strike_mult": 100, "sideways_range": 60.0},
    "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100, "strike_mult": 100, "sideways_range": 80.0},
}

# ================= SESSION STATE INITIALIZATIONS =================
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = "NIFTY 50"
if "price_history" not in st.session_state:
    st.session_state["price_history"] = []
if "engine_orch" not in st.session_state:
    st.session_state["engine_orch"] = EngineOrchestrator()
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

# Predictive Pre-Alert & Metrics Cache
if "cached_itm_candidate" not in st.session_state:
    st.session_state["cached_itm_candidate"] = None
if "cached_pcr" not in st.session_state:
    st.session_state["cached_pcr"] = 1.0

# Daily Signal Counters
if "daily_call_count" not in st.session_state:
    st.session_state["daily_call_count"] = 0
if "daily_put_count" not in st.session_state:
    st.session_state["daily_put_count"] = 0
if "counter_date" not in st.session_state:
    st.session_state["counter_date"] = datetime.datetime.now(IST).strftime("%Y-%m-%d")
if "last_counted_action" not in st.session_state:
    st.session_state["last_counted_action"] = None


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
        exp_cutoff_ist = datetime.datetime(exp_date.year, exp_date.month, exp_date.day, 15, 30, 0, tzinfo=IST)
        total_seconds = max(60.0, (exp_cutoff_ist - now_ist).total_seconds())
        return total_seconds / (365.0 * 86400.0)
    except Exception:
        return 0.005


saved_token_data = load_token_from_file()
if saved_token_data and "access_token" in saved_token_data:
    st.session_state["access_token"] = saved_token_data["access_token"]

query_params = st.query_params
auth_code = query_params.get("code", None)
incoming_state = query_params.get("state", None)

if auth_code and "access_token" not in st.session_state:
    expected_state = st.session_state.get("oauth_state")
    if expected_state and incoming_state != expected_state:
        st.error("OAuth State Validation Failed.")
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

if not active_expiry:
    today_dt = datetime.datetime.now(IST).date()
    target_weekday = 4 if "SENSEX" in st.session_state["selected_index"] else 3
    days_ahead = (target_weekday - today_dt.weekday()) % 7
    target_expiry = today_dt + datetime.timedelta(days=days_ahead)
    active_expiry = target_expiry.strftime("%Y-%m-%d")
    expiry_status = "AUTO_RESOLVED"

# ================= SIDEBAR & DAILY RESET =================
now_ist = datetime.datetime.now(IST)
current_date_str = now_ist.strftime("%d %b %Y")
current_time_str = now_ist.strftime("%I:%M:%S %p")
today_iso = now_ist.strftime("%Y-%m-%d")

if st.session_state["counter_date"] != today_iso:
    st.session_state["daily_call_count"] = 0
    st.session_state["daily_put_count"] = 0
    st.session_state["counter_date"] = today_iso
    st.session_state["last_counted_action"] = None
    st.session_state["engine_orch"] = EngineOrchestrator()

cutoff_time = datetime.time(10, 30, 0)
is_after_cutoff = now_ist.time() >= cutoff_time

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

selected_index = st.sidebar.selectbox("Select Index", index_keys, index=saved_idx_pos, key="selected_index_dropdown")

if selected_index != st.session_state["selected_index"]:
    st.session_state["selected_index"] = selected_index
    st.session_state["price_history"] = []
    st.session_state["candidate_tracker"].flush()
    st.session_state["expiry_manager"].flush_cache()
    st.session_state["engine_orch"] = EngineOrchestrator()
    st.session_state["cached_itm_candidate"] = None
    st.rerun()

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
        st.session_state["engine_orch"] = EngineOrchestrator()
        st.rerun()

# ================= TOP AVATAR BAR =================
col_title, col_avatar = st.columns([3.8, 1.2])
with col_title:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; padding:10px 0;">
        <span style="font-size:42px; line-height:1;">⚡</span>
        <h1 style="margin:0; font-size:38px; font-weight:900; line-height:1.1; letter-spacing:-0.5px;">
            <span style="color:#ffffff;">Hemant </span><span style="color:#e74c3c;">Algo </span><span style="color:#ffffff;">Trading Engine</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_avatar:
    target_img_name = "rule_avatar.png" if is_after_cutoff else "avatar.png"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, target_img_name)
    if os.path.exists(img_path):
        st.image(img_path, width=155)

# ================= MODE STRIP & DAILY COUNTERS =================
call_cnt = st.session_state["daily_call_count"]
put_cnt = st.session_state["daily_put_count"]

st.markdown(f"""
<div style="background-color:#0e2a47; border:1px solid #1a4971; border-radius:8px; padding:10px 16px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
    <div style="color:#64b5f6; font-size:13px; font-weight:600; display:flex; align-items:center; gap:8px;">
        <span>ℹ️</span> <span>WATCH & SIGNAL MODE ACTIVE: Zero-Trust Safety Gate strictly enforced.</span>
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
        <span style="background-color:#163828; border:1px solid #2ecc71; color:#2ecc71; border-radius:6px; padding:3px 10px; font-size:12px; font-weight:700;">🟢 CALL: {call_cnt}</span>
        <span style="background-color:#381c1c; border:1px solid #e74c3c; color:#ff6b6b; border-radius:6px; padding:3px 10px; font-size:12px; font-weight:700;">🔴 PUT: {put_cnt}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= CORE ENGINE INTEGRATION LOOP =================
if "access_token" in st.session_state:
    inst_key = INDEX_CONFIG[selected_index]["key"]
    token = st.session_state["access_token"]
    
    t_start = time.time()
    quote_data = get_market_quote(inst_key, token)
    api_latency = max(0.1, time.time() - t_start)

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

        # ----------------------------------------------------
        # FEED LIVE TICK TO ENGINE ORCHESTRATOR
        # ----------------------------------------------------
        orch = st.session_state["engine_orch"]
        orch_output = orch.process_tick(spot_ltp, now_ist)
        
        # Boundary finalization for completed candles
        boundary_event = orch.finalize_boundary(now_ist)
        active_event = boundary_event or orch_output.get("event")

        raw_engine_action = active_event["action"] if active_event else "NO TRADE"
        engine_status = active_event["status"] if active_event else "WAITING_BAR_CLOSE"
        rsi_14 = active_event["rsi"] if active_event and active_event["rsi"] else 50.0
        macro_trend = active_event["macro"] if active_event else "STANDBY"
        cooldown_left = active_event["cooldown"] if active_event else 0

        # Map Engine Consensus to MarketDirection
        if raw_engine_action == "BUY CE" or macro_trend == "BULLISH":
            market_dir = MarketDirection.BULLISH
        elif raw_engine_action == "BUY PE" or macro_trend == "BEARISH":
            market_dir = MarketDirection.BEARISH
        else:
            market_dir = MarketDirection.NEUTRAL

        best_candidate = None
        scan_diag = {}
        passed_gate = False
        final_action = "NO TRADE"
        gate_msg = engine_status

        # If Engine permits trade, engage scanner and safety gate
        if raw_engine_action in ("BUY CE", "BUY PE") and active_expiry:
            chain_data = get_option_chain_data(inst_key, active_expiry, token)
            if chain_data:
                best_candidate, scan_diag = st.session_state["candidate_scanner"].scan_and_rank_best_candidate(
                    chain_data=chain_data,
                    direction=market_dir,
                    trend_score=85.0
                )
                if best_candidate:
                    cand_sym = f"{selected_index.replace(' ', '')}_{best_candidate['symbol']}"
                    cand_state, is_ready = st.session_state["candidate_tracker"].process_candidate(
                        symbol=cand_sym,
                        strike=best_candidate["strike"],
                        option_type=best_candidate["option_type"],
                        score=best_candidate["score"],
                        direction=market_dir,
                        bar_timestamp=now_ist.strftime("%H:%M:%S"),
                        is_new_closed_bar=True,
                        required_confirmations=2
                    )

                    exact_tte = calculate_precise_tte(active_expiry)
                    live_quote, q_msg = st.session_state["quote_provider"].fetch_execution_quote(
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

                    if live_quote:
                        bs_greeks = BlackScholesEngine.calculate_greeks(
                            spot_ltp, live_quote.strike, live_quote.tte, 0.07, live_quote.iv, live_quote.option_type
                        )
                        passed_gate, gate_msg = ZeroTrustFinalSafetyGate.verify_execution(
                            market_direction=market_dir,
                            tick=live_quote,
                            greeks=bs_greeks,
                            candidate=cand_state,
                            candidate_ready=is_ready,
                            ttl_state=st.session_state["ttl_manager"]
                        )
                        if passed_gate:
                            final_action = raw_engine_action
                            # Count unique daily signals
                            unique_key = f"{now_ist.strftime('%Y%m%d_%H%M')}_{final_action}"
                            if st.session_state["last_counted_action"] != unique_key:
                                if "CE" in final_action:
                                    st.session_state["daily_call_count"] += 1
                                else:
                                    st.session_state["daily_put_count"] += 1
                                st.session_state["last_counted_action"] = unique_key

        # ================= UI LAYOUT =================
        col_metric, col_signal_card = st.columns([2, 1])

        with col_metric:
            regime_title = f"TRENDING ({market_dir.value})" if market_dir != MarketDirection.NEUTRAL else "SIDEWAYS / CHOPPY"
            regime_color = "#2ecc71" if market_dir == MarketDirection.BULLISH else ("#e74c3c" if market_dir == MarketDirection.BEARISH else "#e67e22")
            best_opt_display = f"{best_candidate['symbol']} (₹{best_candidate['snapshot_ltp']:.1f})" if best_candidate else "NONE"

            st.markdown(f"""
            <div style="background-color:#161f30; padding:14px 18px; border-radius:10px; border:1px solid #233554; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><span style="color:#8892b0; font-size:11px;">MARKET STATE ({selected_index})</span>
                    <div style="color:{regime_color}; font-size:15px; font-weight:bold; margin-top:2px;">⏳ {regime_title}</div></div>
                    <div style="text-align:center;"><span style="color:#8892b0; font-size:11px;">MACRO CONFLUENCE</span>
                    <div style="color:#ccd6f6; font-size:13px; font-weight:bold; margin-top:2px;">{macro_trend}</div></div>
                    <div style="text-align:right;"><span style="color:#8892b0; font-size:11px;">🎯 BEST CANDIDATE</span>
                    <div style="color:#64ffda; font-size:13px; font-weight:bold; margin-top:2px;">{best_opt_display}</div></div>
                </div>
                <div style="color:#64ffda; font-size:11px; margin-top:8px; border-top:1px solid #1d2d44; padding-top:6px;">
                    ℹ️ Cooldown Remaining: <b>{cooldown_left} bars</b> | Aggregated Bars: 1m={len(orch.agg.bars_1m)}, 5m={len(orch.agg.bars_5m)}, 15m={len(orch.agg.bars_15m)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            metrics_strip_html = (
                f'<div style="display:flex; justify-content:space-between; align-items:center; background-color:#111622; padding:12px 14px; border-radius:10px; border:1px solid #1f293d; margin-bottom:12px;">'
                f'<div style="flex: 1.3; border-right: 1px solid #233554; padding-right: 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">{selected_index} Spot</div>'
                f'<div style="color:#ffffff; font-size:20px; font-weight:700; white-space:nowrap; margin-top:2px;">₹{spot_ltp:,.2f}</div>'
                f'</div>'
                f'<div style="flex: 0.9; text-align:center; border-right: 1px solid #233554; padding: 0 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">RSI (14)</div>'
                f'<div style="color:#64ffda; font-size:20px; font-weight:700; margin-top:2px;">{rsi_14:.1f}</div>'
                f'</div>'
                f'<div style="flex: 1.0; text-align:center; border-right: 1px solid #233554; padding: 0 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">Cooldown</div>'
                f'<div style="color:#ccd6f6; font-size:20px; font-weight:700; margin-top:2px;">{cooldown_left} BARS</div>'
                f'</div>'
                f'<div style="flex: 1.1; text-align:right; padding-left: 8px;">'
                f'<div style="color:#8892b0; font-size:11px; font-weight:600; text-transform:uppercase;">Engine State</div>'
                f'<div style="color:#ccd6f6; font-size:14px; font-weight:700; margin-top:2px;">{engine_status[:16]}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(metrics_strip_html, unsafe_allow_html=True)
            st.line_chart(pd.DataFrame(st.session_state["price_history"], columns=["Spot Price"]))

        with col_signal_card:
            action_color = "#2ecc71" if "CE" in final_action else ("#e74c3c" if "PE" in final_action else "#f1c40f")
            gate_badge = "PASSED" if passed_gate else "BLOCKED"
            gate_badge_color = "#2ecc71" if passed_gate else "#e74c3c"

            st.markdown(f"""
            <div style="background-color:#1e222d; padding:20px; border-radius:12px; text-align:center; border:1px solid #363c4e;">
                <h3 style="color:#b2b9c7; margin-bottom:2px; font-size:18px;">⚡ Engine Signal ({selected_index})</h3>
                <h1 style="color:{action_color}; font-size:32px; margin-top:2px; margin-bottom:12px; font-weight:bold;">{final_action}</h1>
                <div style="display:flex; justify-content:space-around; margin-bottom:12px;">
                    <div style="background-color:#14171f; padding:6px 12px; border-radius:6px; border:1px solid #2a2e39;">
                        <span style="color:#848d9c; font-size:11px;">Consensus</span><br>
                        <b style="color:#ffffff; font-size:14px;">{macro_trend}</b>
                    </div>
                    <div style="background-color:#14171f; padding:6px 12px; border-radius:6px; border:1px solid #2a2e39;">
                        <span style="color:#848d9c; font-size:11px;">Safety Gate</span><br>
                        <b style="color:{gate_badge_color}; font-size:14px;">{gate_badge}</b>
                    </div>
                </div>
                <p style="color:#57606a; font-size:11px; margin-top:6px;">Gate Reason: {gate_msg}</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning(f"Connecting to live feed for {selected_index}...")

    time.sleep(POLL_INTERVAL)
    st.rerun()

else:
    st.warning("Please authenticate with Upstox via sidebar to start live feed.")
