# config.py
import os
import streamlit as st

def get_upstox_config():
    # Streamlit Cloud encrypted secrets se data uthayega
    if hasattr(st, "secrets") and "UPSTOX" in st.secrets:
        return {
            "API_KEY": st.secrets["UPSTOX"]["API_KEY"],
            "API_SECRET": st.secrets["UPSTOX"]["API_SECRET"],
            "REDIRECT_URI": st.secrets["UPSTOX"]["REDIRECT_URI"],
        }
    
    # Fallback
    return {
        "API_KEY": os.getenv("UPSTOX_API_KEY", ""),
        "API_SECRET": os.getenv("UPSTOX_API_SECRET", ""),
        "REDIRECT_URI": os.getenv("UPSTOX_REDIRECT_URI", "https://algotradingengine-hemant-algo-engine.streamlit.app/"),
    }

UPSTOX_CONFIG = get_upstox_config()
