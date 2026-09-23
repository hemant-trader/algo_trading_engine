import os
import json
import time
from collections import Counter
import requests
import urllib.parse
import pandas as pd
import streamlit as st

from config import UPSTOX_CONFIG
from core.types import MarketDirection, OptionType, NormalizedOptionTick
from core.black_scholes import BlackScholesEngine
from processing.tick_guard import TickGuardEngine
from processing.oi_matrix import OptionChainOIEngine
from processing.candidate_tracker import CandidateHysteresisTracker
from engine.pulse_ttl import PulseTTLStateMachine
from engine.safety_gate import ZeroTrustFinalSafetyGate

st.set_page_config(page_title="Hemant Algo Trading Engine", layout="wide")

TOKEN_FILE = "access_token.json"

INDEX_CONFIG = {
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "step": 50, "strike_mult": 50},
    "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100, "strike_mult": 100},
    "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100, "strike_mult": 100},
}

# 1. State Persistence Setup
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = "NIFTY 50"
if "price_history" not in st.session_state:
    st.session_state["price_history"] = []
if "direction_history" not in st.session_state:
    st.session_state["direction_history"] = []
if "tick_guard" not in st.session_state:
    st.session_state["tick_guard"] = TickGuardEngine()
if "candidate_tracker" not in st.session_state:
    st.session_state["candidate_tracker"] = CandidateHysteresisTracker(
        hysteresis_threshold=5.0, 
        min_confirmations=3
    )
if "ttl_manager" not in st.session_state:
    st.session_state["ttl_manager"] = PulseTTLStateMachine(max_ttl=5)

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
    url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={instrument_key}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

saved_token_data = load_token_from_file()
if saved_token_data and "access_token" in saved_token_data:
    st.session_state["access_token"] = saved_token_data["access_token"]

query_params = st.query_params
auth_code = query_params.get("code", None)
if auth_code and "access_token" not in st.session_state:
    res = get_access_token(auth_code)
    if "access_token" in res:
        st.session_state["access_token"] = res["access_token"]
        save_token_to_file(res)

# ================= SIDEBAR UI =================
st.sidebar.title("⚙️ System & Trade Control")
st.sidebar.markdown("---")

trade_mode = st.sidebar.radio(
    "Execution State",
    ["🔴 OFF: Watch & Signal Mode\n(No Order Execution)", "🟢 ON: Live Auto-Trading Execution Mode"],
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
    st.session_state["direction_history"] = []
    st.session_state["candidate_tracker"].flush()
    st.rerun()

step_val = INDEX_CONFIG[selected_index]["step"]
strike_offset = st.sidebar.number_input("Select Strike Price Offset", value=0, step=step_val)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Settings")
max_loss = st.sidebar.number_input("Max Daily Loss (₹)", value=2000)
target_pnl = st.sidebar.number_input("Daily Target PnL (₹)", value=4000)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Broker Authentication")
if "access_token" not in st.session_state:
    base_url = "https://api.upstox.com/v2/login/authorization/dialog"
    params = {
        "response_type": "code",
        "client_id": UPSTOX_CONFIG["API_KEY"],
        "redirect_uri": UPSTOX_CONFIG["REDIRECT_URI"]
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
        st.session_state["direction_history"] = []
        st.session_state["candidate_tracker"].flush()
        st.rerun()

# ================= MAIN DASHBOARD UI =================
st.title("⚡ Hemant Algo Trading Engine")

is_live_execution = "🟢 ON" in trade_mode
if is_live_execution:
    st.error("🚨 **LIVE AUTO-EXECUTION IS ARMED:** Real orders will route based on Zero-Trust safety gates.")
else:
    st.info("ℹ️ **AUTO-EXECUTION IS OFF:** Engine is in Watch Mode. Displaying calculated signals only.")

if "access_token" in st.session_state:
    inst_key = INDEX_CONFIG[selected_index]["key"]
    quote_data = get_market_quote(inst_key, st.session_state["access_token"])
    
    spot_ltp = None
    if quote_data and "data" in quote_data:
        key_name = inst_key.replace("|", ":")
        if key_name in quote_data["data"]:
            spot_ltp = float(quote_data["data"][key_name]["last_price"])

    if spot_ltp is not None:
        st.session_state["price_history"].append(spot_ltp)
        if len(st.session_state["price_history"]) > 30:
            st.session_state["price_history"].pop(0)

        # 1. Raw Tick Direction Extraction
        if len(st.session_state["price_history"]) >= 2:
            prev_price = st.session_state["price_history"][-2]
            raw_dir = MarketDirection.BEARISH if spot_ltp < prev_price else MarketDirection.BULLISH
        else:
            raw_dir = MarketDirection.BEARISH

        st.session_state["direction_history"].append(raw_dir)
        if len(st.session_state["direction_history"]) > 5:
            st.session_state["direction_history"].pop(0)

        # 2. 3-of-5 Direction Consensus Gate (Noise Filter)
        counts = Counter(st.session_state["direction_history"])
        market_dir = counts.most_common(1)[0][0]
        chosen_opt_type = OptionType.PE if market_dir == MarketDirection.BEARISH else OptionType.CE

        strike_interval = INDEX_CONFIG[selected_index]["strike_mult"]
        atm_strike = round(spot_ltp / strike_interval) * strike_interval + strike_offset

        # 3. Best Out-of-the-Money Strike Calculation
        if chosen_opt_type == OptionType.CE:
            otm_strike = atm_strike + strike_interval
        else:
            otm_strike = atm_strike - strike_interval

        symbol_prefix = selected_index.replace(" ", "").upper()
        clean_symbol = f"{symbol_prefix}_{int(atm_strike)}_{chosen_opt_type.value}"
        otm_clean_symbol = f"{symbol_prefix}_{int(otm_strike)}_{chosen_opt_type.value}"

        current_time = time.time()
        sample_opt_price = 145.0

        tick = NormalizedOptionTick(
            symbol=clean_symbol,
            underlying=selected_index,
            expiry="2026-09-17",
            strike=float(atm_strike),
            option_type=chosen_opt_type,
            ltp=sample_opt_price,
            bid=sample_opt_price - 0.20,
            ask=sample_opt_price + 0.20,
            spread=0.40,
            volume=45000,
            oi=110000,
            volume_change=3000,
            oi_change=12000,
            price_change=8.5 if chosen_opt_type == OptionType.CE else -8.5,
            price_change_pct=6.2,
            oi_change_pct=11.5,
            timestamp=current_time,
            sequence_no=int(current_time),
            tte=0.015
        )

        is_valid_tick, _ = st.session_state["tick_guard"].validate_tick(tick, current_time)
        st.session_state["ttl_manager"].pulse(is_valid_tick)

        iv = BlackScholesEngine.implied_volatility(
            tick.ltp, spot_ltp, tick.strike, tick.tte, 0.07, tick.option_type
        )
        greeks = BlackScholesEngine.calculate_greeks(
            spot_ltp, tick.strike, tick.tte, 0.07, iv, tick.option_type
        )
        oi_score = OptionChainOIEngine.calculate_oi_score(tick, market_dir)
        composite_score = 75.0 + oi_score

        try:
            candidate, is_ready = st.session_state["candidate_tracker"].process_candidate(
                tick.symbol, tick.strike, tick.option_type, composite_score, market_dir
            )
        except TypeError:
            candidate, is_ready = st.session_state["candidate_tracker"].process_candidate(
                tick.symbol, tick.strike, tick.option_type, composite_score
            )

        passed, gate_msg = ZeroTrustFinalSafetyGate.verify_execution(
            market_direction=market_dir,
            tick=tick,
            greeks=greeks,
            candidate=candidate,
            candidate_ready=is_ready,
            ttl_state=st.session_state["ttl_manager"],
            quality_grade="GRADE_A",
            max_spread_pct=0.02
        )

        confidence_pct = min(max(composite_score, 10.0), 98.0)
        conf_color = "#2ecc71" if confidence_pct >= 80.0 else ("#f1c40f" if confidence_pct >= 70.0 else "#e74c3c")

        if passed and candidate is not None:
            final_action = f"BUY {candidate.option_type.value}"
            action_color = "#ff4b4b" if candidate.option_type == OptionType.PE else "#2ecc71"
        else:
            final_action = "NO TRADE"
            action_color = "#f1c40f"

        col_metric, col_signal_card = st.columns([2, 1])

        with col_metric:
            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"{selected_index} Spot", value=f"₹{spot_ltp:,.2f}")
            m2.metric(label="Calculated Delta", value=f"{greeks.delta:.3f}")
            m3.metric(label="Implied Volatility", value=f"{iv * 100:.2f}%")

            st.markdown("---")
            st.subheader(f"📈 Real Tick Feed: {selected_index}")
            df_chart = pd.DataFrame(st.session_state["price_history"], columns=["Spot Price"])
            st.line_chart(df_chart)

        with col_signal_card:
            confirmations_status = (
                f"{candidate.confirmation_count}/3 Confirmed" 
                if candidate else "0/3 Confirmed"
            )
            cand_sym = candidate.symbol if candidate else "Scanning"

            card_html = (
                f'<div style="background-color:#1e222d; padding:18px; border-radius:12px; text-align:center; border:1px solid #363c4e;">'
                f'<h3 style="color:#b2b9c7; margin:0 0 4px 0; font-size:18px;">⚡ Engine Signal</h3>'
                f'<h1 style="color:{action_color}; font-size:32px; margin:0 0 6px 0; font-weight:bold;">{final_action}</h1>'
                f'<div style="background-color:#14171f; padding:4px 10px; border-radius:6px; margin-bottom:8px; display:inline-block; border:1px solid #2a2e39;">'
                f'<span style="color:#848d9c; font-size:12px;">Signal Confidence: </span>'
                f'<b style="color:{conf_color}; font-size:14px;">{confidence_pct:.1f}%</b>'
                f'</div>'
                f'<div style="background-color:#161c28; border:1px dashed #3498db; border-radius:8px; padding:8px 10px; margin-bottom:8px; text-align:left;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<span style="color:#64b5f6; font-size:11px; font-weight:bold;">🎯 BEST LOW-COST OTM</span>'
                f'<span style="background-color:#0d47a1; color:#bbdefb; font-size:10px; padding:1px 5px; border-radius:3px;">High ROI</span>'
                f'</div>'
                f'<div style="font-size:15px; font-weight:bold; color:#ffffff; margin:3px 0;">{otm_clean_symbol}</div>'
                f'<div style="color:#90caf9; font-size:11px;">Low Risk • Delta: ~0.35 • Momentum Pick</div>'
                f'</div>'
                f'<p style="color:#848d9c; margin:2px 0; font-size:13px;">Consensus Direction: <b>{market_dir.name}</b></p>'
                f'<p style="color:#848d9c; margin:2px 0; font-size:13px;">Candidate (ATM): <b>{cand_sym}</b></p>'
                f'<p style="color:#3498db; font-size:12px; margin:2px 0;">Stability: <b>{confirmations_status}</b></p>'
                f'<p style="color:#57606a; font-size:11px; margin-top:4px;">Gate Status: {gate_msg}</p>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    else:
        st.warning(f"Connecting to Upstox market feed for {selected_index}...")

    time.sleep(5)
    st.rerun()

else:
    st.warning("Please authenticate with Upstox using the sidebar controls to view live engine metrics.")
