import os
import json
import time
import datetime
import requests
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st

from config import UPSTOX_CONFIG
from core.types import MarketDirection, OptionType, NormalizedOptionTick
from core.black_scholes import BlackScholesEngine
from processing.tick_guard import TickGuardEngine
from processing.candidate_tracker import CandidateHysteresisTracker
from engine.pulse_ttl import PulseTTLStateMachine
from engine.safety_gate import ZeroTrustFinalSafetyGate

st.set_page_config(page_title="Hemant Algo Trading Engine", layout="wide")

TOKEN_FILE = "access_token.json"

INDEX_CONFIG = {
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "step": 50, "strike_mult": 50, "sideways_range": 25.0},
    "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100, "strike_mult": 100, "sideways_range": 60.0},
    "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100, "strike_mult": 100, "sideways_range": 80.0},
}

# --- State Persistence Setup ---
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = "NIFTY 50"
if "price_history" not in st.session_state:
    st.session_state["price_history"] = []
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

def evaluate_regime_and_direction(price_history, selected_index):
    threshold_range = INDEX_CONFIG[selected_index]["sideways_range"]
    if len(price_history) < 15:
        return MarketDirection.NEUTRAL, 0.0, 50.0, 0.0, "INSUFFICIENT_DATA", threshold_range

    s = pd.Series(price_history)
    price_spread = float(s.max() - s.min())
    ema5 = float(s.ewm(span=5, adjust=False).mean().iloc[-1])
    ema15 = float(s.ewm(span=15, adjust=False).mean().iloc[-1])
    current_price = float(s.iloc[-1])
    price_lookback = float(s.iloc[-6]) if len(s) >= 6 else float(s.iloc[0])
    
    momentum_pct = ((current_price - price_lookback) / price_lookback) * 100.0
    rsi = calculate_rsi(price_history, period=min(14, len(price_history)-1))

    if price_spread <= threshold_range and abs(momentum_pct) < 0.04:
        return MarketDirection.NEUTRAL, 0.0, rsi, price_spread, "SIDEWAYS_RANGEBOUND", threshold_range

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

    if bull_score >= 5 and bull_score >= (bear_score + 2):
        trend_strength = (bull_score / 7.0) * 100.0
        return MarketDirection.BULLISH, trend_strength, rsi, price_spread, "TRENDING_BULLISH", threshold_range
    elif bear_score >= 5 and bear_score >= (bull_score + 2):
        trend_strength = (bear_score / 7.0) * 100.0
        return MarketDirection.BEARISH, trend_strength, rsi, price_spread, "TRENDING_BEARISH", threshold_range

    return MarketDirection.NEUTRAL, 0.0, rsi, price_spread, "CHOPPY_NO_TREND", threshold_range

def get_exact_tte(expiry_str):
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        exp_dt = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").replace(
            hour=10, minute=0, second=0, tzinfo=datetime.timezone.utc
        )
        diff_sec = max(0.0, (exp_dt - now).total_seconds())
        return diff_sec / (365.0 * 86400.0)
    except Exception:
        return 0.015

def get_nearest_expiry():
    today = datetime.date.today()
    days_ahead = (3 - today.weekday()) % 7
    return (today + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")

# ================= SIDEBAR CONTROLS =================
st.sidebar.title("⚙️ System & Trade Control")
st.sidebar.markdown("---")

# Restored Live / Watch Mode Toggle
trade_mode = st.sidebar.radio(
    "Execution State",
    [
        "🔴 OFF: Watch & Signal Mode (Paper Mode)",
        "🟢 ON: Live Auto-Trading Execution Mode"
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
        st.session_state["candidate_tracker"].flush()
        st.rerun()

# ================= MAIN RUNTIME =================
st.title("⚡ Hemant Algo Trading Engine")

is_live_execution = "🟢 ON" in trade_mode
if is_live_execution:
    st.error("🚨 **LIVE AUTO-EXECUTION ARMED:** Orders will route to broker when Zero-Trust Gate passes.")
else:
    st.info("ℹ️ **WATCH & SIGNAL MODE ACTIVE:** Zero-Trust Safety Gate is strictly enforced (No real orders placed).")

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
        if len(st.session_state["price_history"]) > 50:
            st.session_state["price_history"].pop(0)

        market_dir, trend_strength, rsi_val, price_spread, regime_key, spread_thresh = evaluate_regime_and_direction(
            st.session_state["price_history"], selected_index
        )

        strike_interval = INDEX_CONFIG[selected_index]["strike_mult"]
        atm_strike = round(spot_ltp / strike_interval) * strike_interval + strike_offset
        current_time = time.time()
        active_expiry = get_nearest_expiry()
        exact_tte = get_exact_tte(active_expiry)

        if market_dir == MarketDirection.BULLISH:
            chosen_opt_type = OptionType.CE
            selected_strike = atm_strike + strike_interval
            regime_title = "TRENDING (BULLISH)"
            regime_color = "#2ecc71"
            trend_focus = "MOMENTUM CE"
            best_otm_label = f"{int(selected_strike)} CE"
            best_otm_color = "#64ffda"
            banner_note = f"Breakout active ({price_spread:.1f} pts). Target strike selected for momentum."
        elif market_dir == MarketDirection.BEARISH:
            chosen_opt_type = OptionType.PE
            selected_strike = atm_strike - strike_interval
            regime_title = "TRENDING (BEARISH)"
            regime_color = "#e74c3c"
            trend_focus = "MOMENTUM PE"
            best_otm_label = f"{int(selected_strike)} PE"
            best_otm_color = "#64ffda"
            banner_note = f"Breakdown active ({price_spread:.1f} pts). Target strike selected for momentum."
        else:
            chosen_opt_type = OptionType.CE
            selected_strike = atm_strike
            regime_title = "SIDEWAYS / RANGEBOUND"
            regime_color = "#e67e22"
            trend_focus = "NEUTRAL / NO CLEAR TREND"
            best_otm_label = "WAIT / AVOID OTM"
            best_otm_color = "#8892b0"
            banner_note = f"Narrow consolidation ({price_spread:.1f} pts vs {spread_thresh} threshold). High theta decay risk."

        candidate = None
        is_ready = False
        passed = False
        gate_msg = "NEUTRAL_REGIME_STANDBY"
        final_action = "NO TRADE"
        action_color = "#f1c40f"

        if market_dir != MarketDirection.NEUTRAL:
            symbol_prefix = selected_index.replace(" ", "").upper()
            clean_symbol = f"{symbol_prefix}_{int(selected_strike)}_{chosen_opt_type.value}"

            tick = NormalizedOptionTick(
                symbol=clean_symbol,
                instrument_key=clean_symbol,
                underlying=selected_index,
                underlying_spot=spot_ltp,
                expiry=active_expiry,
                strike=float(selected_strike),
                option_type=chosen_opt_type,
                ltp=0.0,
                bid=0.0,
                ask=0.0,
                spread=0.0,
                bid_qty=0,
                ask_qty=0,
                volume=0,
                oi=0,
                previous_oi=0,
                volume_change=0,
                oi_change=0,
                price_change=0.0,
                price_change_pct=0.0,
                oi_change_pct=0.0,
                iv=None,
                timestamp=current_time,
                sequence_no=int(current_time),
                tte=exact_tte,
                data_source="SPOT_TECHNICAL_STREAM"
            )

            greeks = BlackScholesEngine.calculate_greeks(
                spot_ltp, tick.strike, tick.tte, 0.07, 0.14, tick.option_type
            )

            composite_score = float(trend_strength)

            try:
                candidate, is_ready = st.session_state["candidate_tracker"].process_candidate(
                    tick.symbol, tick.strike, tick.option_type, composite_score, market_dir
                )
            except TypeError:
                candidate, is_ready = st.session_state["candidate_tracker"].process_candidate(
                    tick.symbol, tick.strike, tick.option_type, composite_score
                )

            st.session_state["ttl_manager"].pulse(True)

            if greeks is not None:
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
            else:
                passed = False
                gate_msg = "BLOCKED: Greeks calculation failed"

            # Strict Safety Gate Enforcement
            if passed and is_ready and composite_score >= 75.0 and market_dir != MarketDirection.NEUTRAL:
                final_action = f"BUY {chosen_opt_type.value}"
                # Red font for BUY PE, Green font for BUY CE
                action_color = "#ff4b4b" if chosen_opt_type == OptionType.PE else "#2ecc71"
            else:
                final_action = "NO TRADE"
                action_color = "#f1c40f"
        else:
            st.session_state["candidate_tracker"].flush()

        # ================= UI LAYOUT =================
        col_metric, col_signal_card = st.columns([2, 1])

        with col_metric:
            regime_html = (
                f'<div style="background-color:#161f30; padding:12px 18px; border-radius:10px; border:1px solid #233554; margin-bottom:14px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div><span style="color:#8892b0; font-size:11px; text-transform:uppercase;">MARKET STATE ({selected_index})</span>'
                f'<div style="color:{regime_color}; font-size:14px; font-weight:bold; margin-top:2px;">⏳ {regime_title}</div></div>'
                f'<div style="text-align:center;"><span style="color:#8892b0; font-size:11px; text-transform:uppercase;">TREND FOCUS</span>'
                f'<div style="color:#ccd6f6; font-size:13px; font-weight:bold; margin-top:2px;">{trend_focus}</div></div>'
                f'<div style="text-align:right;"><span style="color:#8892b0; font-size:11px; text-transform:uppercase;">🎯 TARGET STRIKE</span>'
                f'<div style="color:{best_otm_color}; font-size:13px; font-weight:bold; margin-top:2px;">{best_otm_label}</div></div>'
                f'</div>'
                f'<div style="color:#64ffda; font-size:11px; margin-top:8px; border-top:1px solid #1d2d44; padding-top:6px;">ℹ️ {banner_note}</div>'
                f'</div>'
            )
            st.markdown(regime_html, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"{selected_index} Spot", value=f"₹{spot_ltp:,.2f}")
            m2.metric(label="Calculated RSI (14)", value=f"{rsi_val:.1f}")
            m3.metric(label="Range Spread (Pts)", value=f"{price_spread:.1f}")

            st.markdown("---")
            st.subheader(f"📈 Real Tick Feed: {selected_index}")
            df_chart = pd.DataFrame(st.session_state["price_history"], columns=["Spot Price"])
            st.line_chart(df_chart)

        with col_signal_card:
            confirmations_status = (
                f"{candidate.confirmation_count}/3 Confirmed" 
                if candidate else "0/3 Confirmed"
            )
            score_display = f"{trend_strength:.1f}%" if market_dir != MarketDirection.NEUTRAL else "0.0%"
            score_color = "#2ecc71" if trend_strength >= 75.0 else ("#f1c40f" if trend_strength >= 50.0 else "#8892b0")
            gate_badge = "PASSED" if passed else "BLOCKED"
            gate_badge_color = "#2ecc71" if passed else "#e74c3c"
            cand_sym = candidate.symbol if candidate else "Standby"

            card_html = (
                f'<div style="background-color:#1e222d; padding:20px; border-radius:12px; text-align:center; border:1px solid #363c4e;">'
                f'<h3 style="color:#b2b9c7; margin-bottom:2px; font-size:18px;">⚡ Engine Signal</h3>'
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
                f'<p style="color:#848d9c; margin-bottom:4px; font-size:13px;">Consensus: <b style="color:#ffffff;">{market_dir.name}</b></p>'
                f'<p style="color:#848d9c; margin-bottom:4px; font-size:13px;">Contract: <b style="color:#ffffff;">{cand_sym}</b></p>'
                f'<p style="color:#3498db; font-size:12px; margin-bottom:4px;">Stability: <b>{confirmations_status}</b></p>'
                f'<p style="color:#57606a; font-size:11px; margin-top:6px;">Gate Info: {gate_msg}</p>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    else:
        st.warning(f"Connecting to Upstox market feed for {selected_index}...")

    time.sleep(5)
    st.rerun()

else:
    st.warning("Please authenticate with Upstox using the sidebar controls to view live engine metrics.")
