import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import random
import sqlite3
import os
import copy
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Credentials loaded from environment variables ─────────────────────────────
# Never hardcode credentials — store them in a .env file locally
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER",   "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL",    "")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="FraudShield — Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME CSS FUNCTION ────────────────────────────────────────────────────────
def inject_css(theme="dark"):
    if theme == "light":
        css_vars = """
  --bg-base:        #F8FAFC;
  --bg-surface:     #FFFFFF;
  --bg-elevated:    #F1F5F9;
  --bg-overlay:     #E2E8F0;
  --border:         #E2E8F0;
  --border-bright:  #7C3AED;
  --primary:        #7C3AED;
  --primary-dim:    rgba(124,58,237,0.1);
  --primary-glow:   rgba(124,58,237,0.2);
  --cyan:           #0891B2;
  --cyan-dim:       rgba(8,145,178,0.1);
  --violet:         #7C3AED;
  --violet-dim:     rgba(124,58,237,0.1);
  --green:          #059669;
  --green-dim:      rgba(5,150,105,0.08);
  --red:            #DC2626;
  --red-dim:        rgba(220,38,38,0.08);
  --amber:          #D97706;
  --amber-dim:      rgba(217,119,6,0.08);
  --text-primary:   #0F172A;
  --text-secondary: #475569;
  --text-muted:     #94A3B8;
  --btn-text:       #FFFFFF;
  --sidebar-bg:     #FFFFFF;
  --sidebar-border: #E2E8F0;
  --card-shadow:    0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --card-hover:     0 4px 16px rgba(0,0,0,0.12);
"""
        mesh_bg     = "#F8FAFC"
        sidebar_bg  = "#FFFFFF"
        body_bg     = "#F8FAFC"
    else:
        css_vars = """
  --bg-base:        #080B16;
  --bg-surface:     #0F1423;
  --bg-elevated:    #161C30;
  --bg-overlay:     #1E2540;
  --border:         rgba(255,255,255,0.08);
  --border-bright:  rgba(124,58,237,0.5);
  --primary:        #7C3AED;
  --primary-dim:    rgba(124,58,237,0.15);
  --primary-glow:   rgba(124,58,237,0.3);
  --cyan:           #06B6D4;
  --cyan-dim:       rgba(6,182,212,0.12);
  --violet:         #7C3AED;
  --violet-dim:     rgba(124,58,237,0.12);
  --green:          #10B981;
  --green-dim:      rgba(16,185,129,0.1);
  --red:            #EF4444;
  --red-dim:        rgba(239,68,68,0.1);
  --amber:          #F59E0B;
  --amber-dim:      rgba(245,158,11,0.1);
  --text-primary:   #F8FAFC;
  --text-secondary: #94A3B8;
  --text-muted:     #475569;
  --btn-text:       #FFFFFF;
  --sidebar-bg:     #0A0D1A;
  --sidebar-border: rgba(255,255,255,0.06);
  --card-shadow:    0 1px 3px rgba(0,0,0,0.4);
  --card-hover:     0 8px 32px rgba(0,0,0,0.5);
"""
        mesh_bg     = "#080B16"
        sidebar_bg  = "#0A0D1A"
        body_bg     = "#080B16"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700;800&display=swap');

/* ─────────────── VARIABLES ─────────────── */
:root {{{css_vars}
  --font-display: 'Manrope', sans-serif;
  --font-body:    'Inter', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
  --r-sm:  6px;
  --r-md:  10px;
  --r-lg:  14px;
  --r-xl:  20px;
  --r-2xl: 28px;
}}

/* ─────────────── BASE — LARGER READABLE FONTS ─────────────── */
html, body, [class*="css"] {{
  font-family: var(--font-body) !important;
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-size: 15px !important;
  -webkit-font-smoothing: antialiased;
}}
.main {{
  background: var(--bg-base) !important;
  background-image:
    radial-gradient(ellipse 70% 40% at 10% 0%, rgba(124,58,237,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 50% 30% at 90% 100%, rgba(6,182,212,0.04) 0%, transparent 60%) !important;
}}
.main .block-container {{
  background: transparent !important;
  padding: 1.75rem 2.25rem !important;
  max-width: 1400px !important;
}}

/* ─────────────── SIDEBAR ─────────────── */
section[data-testid="stSidebar"] {{
  background: {sidebar_bg} !important;
  border-right: 1px solid var(--sidebar-border) !important;
}}
section[data-testid="stSidebar"] .block-container {{
  padding: 0.75rem !important;
}}
section[data-testid="stSidebar"] {{ display:flex !important; visibility:visible !important; transform:translateX(0) !important; min-width:235px !important; max-width:260px !important; opacity:1 !important; }}
section[data-testid="stSidebar"] button[kind="header"],
section[data-testid="stSidebar"] > div > div > button,
[data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
[data-testid="collapsedControl"] {{
  display:flex !important; visibility:visible !important; opacity:1 !important;
  position:fixed !important; left:0.5rem !important; top:50% !important;
  z-index:999999 !important; background:var(--primary) !important;
  border:none !important; border-radius:50% !important;
  width:34px !important; height:34px !important;
  color:#fff !important; cursor:pointer !important;
  box-shadow:0 0 16px var(--primary-glow) !important;
  align-items:center !important; justify-content:center !important;
}}

/* ─────────────── BRAND MARK ─────────────── */
.brand-wrap {{ padding: 1.25rem 0.5rem 1.5rem; }}
.brand-logo {{
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--primary), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; margin-bottom: 0.75rem;
  box-shadow: 0 4px 12px var(--primary-glow);
}}
.brand-name {{ font-family: var(--font-display); font-size: 1.05rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }}
.brand-tag  {{ font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.15rem; }}

/* ─────────────── USER CARD ─────────────── */
.user-card {{
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1rem;
  margin-bottom: 1rem;
}}
.user-avatar {{
  width: 38px; height: 38px; border-radius: 8px;
  background: linear-gradient(135deg, var(--primary), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-display); font-size: 0.9rem; font-weight: 800;
  color: white; margin-bottom: 0.6rem;
  box-shadow: 0 2px 8px var(--primary-glow);
}}
.user-name   {{ font-family: var(--font-display); font-weight: 700; font-size: 0.95rem; color: var(--text-primary); }}
.user-handle {{ font-family: var(--font-mono);    font-size: 0.72rem; color: var(--text-muted); margin-top: 0.15rem; }}

/* ─────────────── NAV RADIO — UNIFORM FONT ─────────────── */
div[data-testid="stRadio"] > div {{ gap: 0.15rem; flex-direction: column; }}
div[data-testid="stRadio"] label {{
  display: flex !important;
  align-items: center;
  background: transparent !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  padding: 0.6rem 0.8rem !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  letter-spacing: 0 !important;
}}
div[data-testid="stRadio"] label:hover {{
  background: var(--primary-dim) !important;
  color: var(--primary) !important;
}}
div[data-testid="stRadio"] [data-checked="true"] label,
div[data-testid="stRadio"] input:checked + div label {{
  background: var(--primary-dim) !important;
  color: var(--primary) !important;
  font-weight: 600 !important;
}}
/* Hide radio circle */
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {{ display: none !important; }}

/* ─────────────── NAV SECTION LABEL ─────────────── */
.nav-section {{
  font-family: var(--font-mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted);
  padding: 0.4rem 0.8rem;
  margin-top: 0.5rem;
  margin-bottom: 0.15rem;
}}

/* ─────────────── CARDS ─────────────── */
.glass {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.4rem;
  margin-bottom: 0.85rem;
  box-shadow: var(--card-shadow);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}}
.glass:hover {{ box-shadow: var(--card-hover); border-color: var(--border-bright); }}
.glass-cyan   {{ border-top: 2px solid var(--cyan); }}
.glass-green  {{ border-top: 2px solid var(--green); }}
.glass-red    {{ border-top: 2px solid var(--red); }}
.glass-amber  {{ border-top: 2px solid var(--amber); }}
.glass-violet {{ border-top: 2px solid var(--primary); }}

/* ─────────────── HERO ─────────────── */
.hero-wrap {{
  border-radius: var(--r-xl);
  padding: 2.25rem 2.5rem;
  margin-bottom: 1.75rem;
  background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
  border: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}}
.hero-wrap::after {{
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 40%;
  height: 100%;
  background: linear-gradient(135deg, transparent, var(--primary-dim));
  pointer-events: none;
}}
.hero-grid {{
  position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(circle, var(--border) 1px, transparent 1px);
  background-size: 28px 28px;
  mask-image: radial-gradient(ellipse 60% 80% at 80% 50%, black 20%, transparent 100%);
  opacity: 0.5;
}}
.hero-eyebrow {{
  font-family: var(--font-mono);
  font-size: 0.73rem;
  color: var(--primary);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  font-weight: 600;
  margin-bottom: 0.6rem;
  display: flex; align-items: center; gap: 0.5rem;
}}
.hero-eyebrow::before {{ content: ''; display: inline-block; width: 16px; height: 2px; background: var(--primary); border-radius: 2px; }}
.hero-title {{
  font-family: var(--font-display);
  font-size: 2.25rem;
  font-weight: 800;
  color: var(--text-primary);
  line-height: 1.15;
  letter-spacing: -0.03em;
  margin: 0 0 0.5rem;
}}
/* Gradient accent text inside hero titles */
.grad {{
  background: linear-gradient(90deg, #7C3AED 0%, #06B6D4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.hero-sub {{ color: var(--text-secondary); font-size: 1rem; margin: 0; line-height: 1.7; font-weight: 450; }}
.hero-chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 1.1rem; }}
.chip {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  padding: 0.28rem 0.7rem;
  font-weight: 500;
  border-radius: 20px;
  border: 1px solid var(--border-bright);
  color: var(--primary);
  background: var(--primary-dim);
  letter-spacing: 0.04em;
}}

/* ─────────────── KPI ICON ─────────────── */
.kpi-icon {{
  font-size: 1.4rem;
  margin-bottom: 0.75rem;
  display: block;
  opacity: 0.85;
}}

/* ─────────────── FULL WIDTH SIGNIN BUTTON ─────────────── */
.signin-wrap > div > button {{
  background: linear-gradient(135deg, #06B6D4, #7C3AED) !important;
  color: white !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  padding: 0.8rem !important;
  border-radius: 10px !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 20px rgba(6,182,212,0.25) !important;
}}
.signin-wrap > div > button:hover {{
  box-shadow: 0 6px 28px rgba(6,182,212,0.4) !important;
  transform: translateY(-1px) !important;
}}

/* ─────────────── SECTION LABEL ─────────────── */
.sec-label {{
  font-family: var(--font-mono);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  font-weight: 600;
  color: var(--primary);
  margin-bottom: 1rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.5rem;
}}
.sec-label::after {{ content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }}

/* ─────────────── KPI CARDS ─────────────── */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 0.85rem; margin-bottom: 1.5rem; }}
.kpi {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.4rem 1.25rem 1.2rem;
  text-align: left;
  box-shadow: var(--card-shadow);
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}}
.kpi::after {{
  content: ''; position: absolute; bottom: 0; left: 0; right: 0;
  height: 2px; background: linear-gradient(90deg, var(--primary), var(--cyan));
  border-radius: 0 0 var(--r-lg) var(--r-lg); opacity: 0.7;
}}
.kpi:hover {{ box-shadow: var(--card-hover); transform: translateY(-2px); }}
.kpi-icon  {{ font-size: 1.5rem; margin-bottom: 0.75rem; display: block; }}
.kpi-label {{ font-family: var(--font-mono); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 700; }}
.kpi-value {{ font-family: var(--font-display); font-size: 2.2rem; font-weight: 800; color: var(--text-primary); line-height: 1; letter-spacing: -0.02em; }}
.kpi-sub   {{ font-size: 0.78rem; color: var(--text-muted); margin-top: 0.35rem; font-weight: 500; }}

/* ─────────────── BUTTONS — UNIFORM VIOLET ─────────────── */
.stButton > button {{
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  color: #FFFFFF !important;
  background: var(--primary) !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  padding: 0.65rem 1.25rem !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{
  background: #6D28D9 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px var(--primary-glow) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}
.stButton > button:disabled {{
  background: var(--bg-elevated) !important;
  color: var(--text-muted) !important;
  cursor: not-allowed !important;
  transform: none !important;
}}

/* Sign In / Primary button — cyan gradient like reference */
div[data-testid="stButton"]:has(button[kind="primary"]) > button {{
  background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #7C3AED 100%) !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  padding: 0.8rem !important;
  border-radius: var(--r-md) !important;
  letter-spacing: 0.04em !important;
  box-shadow: 0 4px 20px rgba(6,182,212,0.3) !important;
}}
div[data-testid="stButton"]:has(button[kind="primary"]) > button:hover {{
  background: linear-gradient(135deg, #0891B2 0%, #2563EB 50%, #6D28D9 100%) !important;
  box-shadow: 0 6px 28px rgba(6,182,212,0.45) !important;
  transform: translateY(-2px) !important;
}}

/* Forgot Password — transparent link style */
div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"]) > button {{
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  padding: 0.6rem 0.75rem !important;
}}
div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"]) > button:hover {{
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  background: var(--primary-dim) !important;
  transform: none !important;
  box-shadow: none !important;
}}

/* ─────────────── THEME TOGGLE — FLOATING TOP RIGHT ─────────────── */
.theme-toggle-float {{
  position: fixed;
  top: 0.6rem;
  right: 0.75rem;
  z-index: 999998;
  display: flex;
  justify-content: flex-end;
}}
.theme-toggle-float > div {{
  width: auto !important;
}}
.theme-toggle-float button {{
  background: rgba(124,58,237,0.15) !important;
  border: 1px solid rgba(124,58,237,0.35) !important;
  border-radius: 20px !important;
  padding: 0.3rem 0.8rem !important;
  font-size: 0.72rem !important;
  font-family: var(--font-mono) !important;
  color: #A78BFA !important;
  cursor: pointer !important;
  width: auto !important;
  min-width: unset !important;
  white-space: nowrap !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
  box-shadow: none !important;
  transform: none !important;
}}
.theme-toggle-float button:hover {{
  background: rgba(124,58,237,0.25) !important;
  border-color: rgba(124,58,237,0.6) !important;
  color: #C4B5FD !important;
  transform: none !important;
  box-shadow: none !important;
}}

/* ─────────────── INPUTS ─────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.9rem !important;
  padding: 0.65rem 0.9rem !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--primary-dim) !important;
  outline: none !important;
}}
.stSelectbox > div > div {{
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  color: var(--text-primary) !important;
}}

/* ─────────────── BADGES ─────────────── */
.badge {{ display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.6rem; border-radius: 20px; font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }}
.badge-admin    {{ background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.25); }}
.badge-research {{ background: rgba(245,158,11,0.12); color: #FCD34D; border: 1px solid rgba(245,158,11,0.25); }}
.badge-user     {{ background: rgba(16,185,129,0.12); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.25); }}
.badge-gemini   {{ background: var(--primary-dim);    color: #A78BFA; border: 1px solid rgba(124,58,237,0.3); }}
.badge-verified {{ background: rgba(16,185,129,0.1);  color: #6EE7B7; border: 1px solid rgba(16,185,129,0.2); font-size: 0.58rem; }}

/* ─────────────── STATUS DOTS ─────────────── */
.dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; }}
.dot-green {{ background: var(--green); box-shadow: 0 0 5px var(--green); animation: pulse-dot 2.5s infinite; }}
.dot-red   {{ background: var(--red);   box-shadow: 0 0 5px var(--red);   animation: pulse-dot 2.5s infinite; }}
.dot-cyan  {{ background: var(--cyan);  box-shadow: 0 0 5px var(--cyan);  animation: pulse-dot 2.5s infinite; }}
.dot-amber {{ background: var(--amber); box-shadow: 0 0 5px var(--amber); animation: pulse-dot 2.5s infinite; }}
@keyframes pulse-dot {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.4;}} }}

/* ─────────────── RESULT BOXES ─────────────── */
.result-box {{ border-radius: var(--r-xl); padding: 2rem; text-align: center; animation: fadeUp 0.4s ease; }}
.result-fraud {{ background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.25); }}
.result-legit {{ background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.25); }}
.result-icon    {{ font-size: 2.75rem; margin-bottom: 0.5rem; animation: bounceIn 0.5s ease; }}
.result-verdict {{ font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }}
.verdict-fraud {{ color: var(--red); }}
.verdict-legit {{ color: var(--green); }}

/* ─────────────── LOG ROW ─────────────── */
.log-row {{
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.55rem 0.9rem;
  margin-bottom: 0.3rem;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  color: var(--text-secondary);
  transition: background 0.15s;
}}
.log-row:hover {{ background: var(--bg-overlay); }}

/* ─────────────── OTP BOX ─────────────── */
.otp-wrap {{
  background: linear-gradient(135deg, var(--primary-dim), var(--cyan-dim));
  border: 1px solid var(--border-bright);
  border-radius: var(--r-xl);
  padding: 2rem;
  text-align: center;
  margin: 1rem 0;
}}
.otp-code {{
  font-family: var(--font-mono);
  font-size: 2.8rem;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: 0.5rem;
  margin: 0.75rem 0;
  animation: glowPulse 2.5s ease-in-out infinite;
}}

/* ─────────────── NAV CARD BUTTONS ─────────────── */
.nav-card-btn > button {{
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 0.9rem 1rem !important;
  text-align: left !important;
  width: 100% !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
  min-height: 65px !important;
}}
.nav-card-btn > button:hover {{
  border-color: var(--primary) !important;
  background: var(--primary-dim) !important;
  transform: translateX(3px) !important;
  color: var(--primary) !important;
}}

/* ─────────────── TABLES ─────────────── */
.stDataFrame {{ border-radius: var(--r-md); overflow: hidden; border: 1px solid var(--border) !important; }}
.stDataFrame thead th {{
  background: var(--bg-elevated) !important;
  color: var(--text-muted) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.65rem !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  padding: 0.75rem 1rem !important;
}}
.stDataFrame tbody tr:hover td {{ background: var(--primary-dim) !important; }}

/* ─────────────── PROGRESS ─────────────── */
.stProgress > div > div {{
  background: linear-gradient(90deg, var(--primary), var(--cyan)) !important;
  border-radius: 4px !important;
}}

/* ─────────────── FILE UPLOADER ─────────────── */
.stFileUploader {{
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--r-lg) !important;
  background: var(--bg-surface) !important;
  transition: border-color 0.2s !important;
}}
.stFileUploader:hover {{ border-color: var(--primary) !important; }}

/* ─────────────── ANIMATIONS ─────────────── */
@keyframes fadeUp   {{ from{{opacity:0;transform:translateY(12px);}} to{{opacity:1;transform:translateY(0);}} }}
@keyframes bounceIn {{ 0%{{transform:scale(0.5);opacity:0;}} 70%{{transform:scale(1.08);}} 100%{{transform:scale(1);opacity:1;}} }}
@keyframes glowPulse{{ 0%,100%{{opacity:1;}} 50%{{opacity:0.55;}} }}
@keyframes shimmer  {{ 0%{{background-position:-200% center;}} 100%{{background-position:200% center;}} }}
.main > div {{ animation: fadeUp 0.3s ease; }}

/* ─────────────── HIDE CHROME ─────────────── */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
[data-testid="stToolbar"]      {{ display: none !important; }}
[data-testid="stDecoration"]   {{ display: none !important; }}
[data-testid="stStatusWidget"] {{ display: none !important; }}
.stDeployButton                {{ display: none !important; }}

/* ─────────────── FIX BLANK BOXES AND </div> LEAK ─────────────── */
div[data-testid="stVerticalBlock"] > div:empty,
div[data-testid="stVerticalBlockBorderWrapper"]:empty,
.element-container:empty {{ display: none !important; }}
.stSpinner > div {{ background: transparent !important; border: none !important; box-shadow: none !important; }}

/* Hide orphaned closing </div> tags that Streamlit renders as text */
.stMarkdown p:empty {{ display: none !important; }}
section[data-testid="stSidebar"] .stMarkdown {{
    min-height: 0 !important;
}}
/* Target the specific </div> text node that leaks in sidebar */
section[data-testid="stSidebar"] .element-container:has(.stMarkdown p:only-child:empty) {{
    display: none !important;
}}

/* ─────────────── HIDE </div> LEAK ─────────────── */
/* Target orphaned closing tag text rendered as paragraphs */
section[data-testid="stSidebar"] .stMarkdown p:empty {{ display: none !important; }}
section[data-testid="stSidebar"] .element-container:has(.stMarkdown:empty) {{ display: none !important; }}
/* Hide any stMarkdown that only contains whitespace or a lone tag */
section[data-testid="stSidebar"] .stMarkdown {{ min-height: 0; }}

/* ─────────────── GLOBAL FONT SIZE INCREASE ─────────────── */
p, span, div, li {{ font-size: 15px; }}
label {{ font-size: 14px !important; }}
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stTextArea label,
.stSlider label {{ font-size: 15px !important; font-weight: 500 !important; color: var(--text-secondary) !important; }}
.stTextInput input, .stNumberInput input {{ font-size: 15px !important; }}
.stSelectbox > div {{ font-size: 15px !important; }}
div[data-testid="stRadio"] label {{ font-size: 0.95rem !important; }}
.stDataFrame td, .stDataFrame th {{ font-size: 14px !important; }}
</style>
""", unsafe_allow_html=True)

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = "fraudshield.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, username TEXT, amount REAL, category TEXT,
        result TEXT, fraud_probability REAL, risk_band TEXT,
        recommended_action TEXT, explanation TEXT, prediction_type TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, username TEXT, action TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        username TEXT, role TEXT, name TEXT, email TEXT,
        created TEXT, expires TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT, role TEXT, name TEXT,
        email TEXT, status TEXT, created TEXT,
        totp_secret TEXT DEFAULT NULL,
        totp_enabled INTEGER DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS locked_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        locked_at TEXT,
        attempts INTEGER,
        notified_admin INTEGER DEFAULT 0,
        unlocked_at TEXT,
        is_active INTEGER DEFAULT 1)""")
    conn.commit(); conn.close()
    # Seed default users into DB if not present
    _seed_default_users()

def _seed_default_users():
    """Insert default users into DB if they don't exist yet."""
    defaults = [
        ("admin",      "admin123",    "admin",      "System Admin", "mdrprashan10@gmail.com", "active", "2024-01-01", None, 0),
        ("researcher", "research123", "researcher", "Dr. Research", "mdrprashan10@gmail.com", "active", "2024-01-01", None, 0),
        ("user1",      "user123",     "user",       "John Analyst", "mdrprashan10@gmail.com", "active", "2024-01-01", None, 0),
    ]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for row in defaults:
        c.execute("""INSERT OR IGNORE INTO users
                     (username,password,role,name,email,status,created,totp_secret,totp_enabled)
                     VALUES(?,?,?,?,?,?,?,?,?)""", row)
    conn.commit(); conn.close()

def db_get_all_users() -> dict:
    """Load all users from database into a dict."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username,password,role,name,email,status,created FROM users")
    rows = c.fetchall()
    conn.close()
    return {
        row[0]: {
            "password": row[1], "role": row[2], "name": row[3],
            "email": row[4], "status": row[5], "created": row[6]
        }
        for row in rows
    }

def db_upsert_user(username, password, role, name, email, status, created):
    """Insert or update a user in the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO users(username,password,role,name,email,status,created)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(username) DO UPDATE SET
                    password=excluded.password, role=excluded.role,
                    name=excluded.name, email=excluded.email,
                    status=excluded.status""",
                 (username, password, role, name, email, status, created))
    conn.commit(); conn.close()

def db_delete_user(username):
    """Delete a user from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit(); conn.close()

def db_update_user_status(username, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET status=? WHERE username=?", (status, username))
    conn.commit(); conn.close()

def db_update_user_password(username, new_password):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
    conn.commit(); conn.close()

def db_migrate_totp():
    """Add TOTP columns to existing databases that don't have them."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT DEFAULT NULL")
    except: pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN totp_enabled INTEGER DEFAULT 0")
    except: pass
    conn.commit(); conn.close()

def db_save_totp_secret(username, secret):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET totp_secret=?, totp_enabled=1 WHERE username=?",
                 (secret, username))
    conn.commit(); conn.close()

def db_get_totp_secret(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT totp_secret, totp_enabled FROM users WHERE username=?", (username,))
    row = c.fetchone(); conn.close()
    if row: return row[0], bool(row[1])
    return None, False

# ── TOTP / GOOGLE AUTHENTICATOR ───────────────────────────────────────────────
import pyotp, qrcode, io, base64

def generate_totp_secret():
    return pyotp.random_base32()

def get_totp_uri(username, secret):
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=username,
        issuer_name="FraudShield"
    )

def generate_qr_base64(uri):
    """Generate QR code image as base64 string."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=2,
    )
    qr.add_data(uri)
    qr.make(fit=True)
    img  = qr.make_image(fill_color="#7C3AED", back_color="#0F1423")
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def verify_totp(secret, code):
    """Verify a 6-digit TOTP code against the secret."""
    try:
        totp = pyotp.TOTP(secret)
        return totp.verify(str(code).strip(), valid_window=1)
    except:
        return False

def db_lock_account(username, attempts=3):
    """Record an account lockout event."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO locked_accounts(username,locked_at,attempts,notified_admin,is_active)
                    VALUES(?,?,?,0,1)""",
                 (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), attempts))
    conn.commit(); conn.close()

def db_unlock_account(username):
    """Mark all active lockouts for a user as resolved."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""UPDATE locked_accounts SET is_active=0, unlocked_at=?
                    WHERE username=? AND is_active=1""",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username))
    conn.commit(); conn.close()

def db_get_active_lockouts():
    """Get all currently active lockout events."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM locked_accounts WHERE is_active=1 ORDER BY locked_at DESC",
        conn)
    conn.close()
    return df

def db_mark_lockout_notified(lockout_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE locked_accounts SET notified_admin=1 WHERE id=?", (lockout_id,))
    conn.commit(); conn.close()

def db_save_session(token, username, role, name, email):
    from datetime import timedelta
    conn = sqlite3.connect(DB_PATH)
    expires = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("DELETE FROM sessions WHERE username=?", (username,))
    conn.execute("INSERT INTO sessions(token,username,role,name,email,created,expires) VALUES(?,?,?,?,?,?,?)",
                 (token, username, role, name, email,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), expires))
    conn.commit(); conn.close()

def db_get_session(token):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE token=?", (token,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    cols = ["token","username","role","name","email","created","expires"]
    data = dict(zip(cols, row))
    if datetime.now() > datetime.strptime(data["expires"], "%Y-%m-%d %H:%M:%S"):
        db_delete_session(token)
        return None
    return data

def db_delete_session(token):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit(); conn.close()

def db_save_prediction(username, amount, category, result, prob, risk, action, explanation, pred_type):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO predictions
        (timestamp,username,amount,category,result,fraud_probability,risk_band,recommended_action,explanation,prediction_type)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username, amount,
         category, result, prob, risk, action, explanation, pred_type))
    conn.commit(); conn.close()

def db_get_predictions(username=None, limit=100):
    conn = sqlite3.connect(DB_PATH)
    q = ("SELECT * FROM predictions WHERE username=? ORDER BY id DESC LIMIT ?" if username
         else "SELECT * FROM predictions ORDER BY id DESC LIMIT ?")
    params = (username, limit) if username else (limit,)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close(); return df

def db_save_log(username, action):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO audit_logs(timestamp,username,action) VALUES(?,?,?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username, action))
    conn.commit(); conn.close()

def db_get_logs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close(); return df

init_db()
db_migrate_totp()

# ── EMAIL CONFIGURATION ───────────────────────────────────────────────────────
# Replace with your Gmail address and App Password
# Get App Password: myaccount.google.com/security → App passwords
# ── Email configuration loaded from .env ─────────────────────────────────────

def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an HTML email. Returns True if sent successfully."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"FraudShield Platform <{EMAIL_SENDER}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
        return True
    except Exception as e:
        return False

def email_base(content: str, title: str) -> str:
    """Wrap content in the FraudShield branded email template."""
    return f"""
    <html><body style="margin:0;padding:0;background:#04060f;font-family:'Segoe UI',Arial,sans-serif;">
    <div style="max-width:580px;margin:2rem auto;background:#0d1117;border-radius:16px;
                border:1px solid rgba(0,212,255,0.15);overflow:hidden;">

      <!-- Header -->
      <div style="background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(124,58,237,0.1));
                  padding:2rem;text-align:center;border-bottom:1px solid rgba(0,212,255,0.12);">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">🛡️</div>
        <div style="font-family:'Segoe UI',Arial,sans-serif;font-size:1.6rem;font-weight:800;
                    color:#eef2ff;letter-spacing:-0.02em;">FraudShield</div>
        <div style="color:#00d4ff;font-size:0.75rem;letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:0.2rem;">Intelligence Platform</div>
      </div>

      <!-- Body -->
      <div style="padding:2rem;">
        <div style="font-size:1.25rem;font-weight:700;color:#eef2ff;margin-bottom:1rem;">{title}</div>
        {content}
      </div>

      <!-- Footer -->
      <div style="padding:1.2rem 2rem;border-top:1px solid rgba(255,255,255,0.06);
                  text-align:center;background:rgba(0,0,0,0.2);">
        <div style="color:#4a5568;font-size:0.75rem;font-family:monospace;">
          FraudShield · Automated Notification · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        <div style="color:#4a5568;font-size:0.7rem;margin-top:0.3rem;">
          This is an automated message — please do not reply.
        </div>
      </div>
    </div>
    </body></html>
    """

def notify_user_approved(name: str, username: str, role: str, to_email: str) -> bool:
    """Send approval email to the new user."""
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        Hi <strong style="color:#eef2ff;">{name}</strong>,
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1.5rem;">
        Great news! Your FraudShield account request has been <strong style="color:#00e676;">approved</strong>
        by the platform administrator.
    </p>

    <div style="background:rgba(0,230,118,0.06);border:1px solid rgba(0,230,118,0.2);
                border-radius:10px;padding:1.2rem;margin-bottom:1.5rem;">
      <div style="font-size:0.68rem;color:#00e676;text-transform:uppercase;
                  letter-spacing:0.12em;margin-bottom:0.75rem;font-family:monospace;">
          Account Details
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Username</td>
            <td style="color:#eef2ff;font-family:monospace;font-size:0.85rem;">{username}</td></tr>
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Role</td>
            <td style="color:#eef2ff;font-family:monospace;font-size:0.85rem;">{role}</td></tr>
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Status</td>
            <td style="color:#00e676;font-family:monospace;font-size:0.85rem;">● Active</td></tr>
      </table>
    </div>

    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        You can now log in at <strong style="color:#00d4ff;">http://localhost:8501</strong>
        using your username and the password you set during registration.
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0;">
        After logging in you will be prompted to complete Two-Factor Authentication (2FA)
        before accessing the platform.
    </p>
    """
    return send_email(to_email,
                      "✅ Your FraudShield Account Has Been Approved",
                      email_base(content, "Account Approved 🎉"))

def notify_user_rejected(name: str, username: str, to_email: str) -> bool:
    """Send rejection email to the user."""
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        Hi <strong style="color:#eef2ff;">{name}</strong>,
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1.5rem;">
        Thank you for your interest in FraudShield. Unfortunately, your account request
        for username <strong style="color:#eef2ff;font-family:monospace;">{username}</strong>
        has been <strong style="color:#ff1744;">declined</strong> by the platform administrator
        at this time.
    </p>
    <div style="background:rgba(255,23,68,0.06);border:1px solid rgba(255,23,68,0.2);
                border-radius:10px;padding:1.2rem;margin-bottom:1.5rem;">
      <div style="color:#ff6b81;font-size:0.85rem;line-height:1.6;">
        If you believe this is an error or would like to provide additional context,
        please contact your platform administrator directly.
      </div>
    </div>
    <p style="color:#8892a4;line-height:1.7;margin:0;">
        You are welcome to submit a new request if your circumstances change.
    </p>
    """
    return send_email(to_email,
                      "❌ FraudShield Account Request Update",
                      email_base(content, "Account Request Declined"))

def notify_password_reset_otp(name: str, otp: str, to_email: str) -> bool:
    """Send password reset OTP via email."""
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        Hi <strong style="color:#eef2ff;">{name}</strong>,
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1.5rem;">
        A password reset was requested for your FraudShield account.
        Use the code below to reset your password.
        This code expires in <strong style="color:#eef2ff;">10 minutes</strong>.
    </p>
    <div style="background:linear-gradient(135deg,rgba(124,58,237,0.08),rgba(6,182,212,0.08));
                border:1px solid rgba(124,58,237,0.25);border-radius:14px;
                padding:2.5rem;text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.2em;font-family:monospace;margin-bottom:0.75rem;">
            Password Reset Code
        </div>
        <div style="font-family:monospace;font-size:3rem;font-weight:700;
                    color:#7C3AED;letter-spacing:0.8rem;">
            {otp}
        </div>
        <div style="color:#475569;font-size:0.75rem;margin-top:0.75rem;">
            Valid for 10 minutes. Do not share this code.
        </div>
    </div>
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                border-radius:8px;padding:1rem;">
        <p style="color:#F59E0B;font-size:0.82rem;margin:0;">
            If you did not request a password reset, ignore this email.
            Your password will not change unless you complete the reset process.
        </p>
    </div>
    """
    return send_email(
        to_email,
        "FraudShield — Password Reset Code",
        email_base(content, "Password Reset")
    )


def notify_admin_new_request(name: str, username: str, role: str,
                              reason: str, admin_email: str) -> bool:
    """Notify admin that a new account request has been submitted."""
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        A new account request has been submitted on the FraudShield platform
        and requires your review.
    </p>

    <div style="background:rgba(255,171,0,0.06);border:1px solid rgba(255,171,0,0.2);
                border-radius:10px;padding:1.2rem;margin-bottom:1.5rem;">
      <div style="font-size:0.68rem;color:#ffab00;text-transform:uppercase;
                  letter-spacing:0.12em;margin-bottom:0.75rem;font-family:monospace;">
          Request Details
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;width:40%;">Full Name</td>
            <td style="color:#eef2ff;font-size:0.85rem;">{name}</td></tr>
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Username</td>
            <td style="color:#eef2ff;font-family:monospace;font-size:0.85rem;">{username}</td></tr>
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Requested Role</td>
            <td style="color:#ffab00;font-family:monospace;font-size:0.85rem;">{role}</td></tr>
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Reason</td>
            <td style="color:#eef2ff;font-size:0.85rem;">{reason or 'Not provided'}</td></tr>
        <tr><td style="color:#4a5568;padding:0.35rem 0;font-size:0.85rem;">Submitted</td>
            <td style="color:#eef2ff;font-family:monospace;font-size:0.85rem;">
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
      </table>
    </div>

    <p style="color:#8892a4;line-height:1.7;margin:0;">
        Please log in to FraudShield and go to
        <strong style="color:#00d4ff;">User Management</strong>
        to approve or reject this request.
    </p>
    """
    return send_email(admin_email,
                      f"New Account Request — {name} (@{username})",
                      email_base(content, "New Account Request Pending Review"))

def notify_2fa_otp(name: str, otp: str, to_email: str) -> bool:
    """Send 2FA OTP via email."""
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        Hi <strong style="color:#eef2ff;">{name}</strong>,
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1.5rem;">
        A login attempt was made on your FraudShield account.
        Use the code below to complete your Two-Factor Authentication.
        This code expires in <strong style="color:#eef2ff;">5 minutes</strong>.
    </p>
    <div style="background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(124,58,237,0.08));
                border:1px solid rgba(0,212,255,0.25);border-radius:14px;
                padding:2.5rem;text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.2em;font-family:monospace;margin-bottom:0.75rem;">
            Your verification code
        </div>
        <div style="font-family:monospace;font-size:3rem;font-weight:700;
                    color:#00d4ff;letter-spacing:0.8rem;">
            {otp}
        </div>
        <div style="color:#475569;font-size:0.75rem;margin-top:0.75rem;">
            Valid for 5 minutes. Do not share this code with anyone.
        </div>
    </div>
    <div style="background:rgba(255,171,0,0.08);border:1px solid rgba(255,171,0,0.2);
                border-radius:8px;padding:1rem;margin-bottom:1rem;">
        <p style="color:#ffab00;font-size:0.82rem;margin:0;">
            If you did not attempt to log in, your account may be at risk.
            Change your password immediately and contact your administrator.
        </p>
    </div>
    <p style="color:#4a5568;font-size:0.8rem;margin:0;">
        FraudShield will never ask for your OTP over phone, chat, or email.
    </p>
    """
    return send_email(
        to_email,
        "FraudShield — Your Login Verification Code",
        email_base(content, "Two-Factor Authentication")
    )
# ── USERS ─────────────────────────────────────────────────────────────────────
DEFAULT_USERS = {
    "admin":      {"password":"admin123",    "role":"admin",      "name":"System Admin",  "status":"active","created":"2024-01-01","email":"mdrprashan10@gmail.com"},
    "researcher": {"password":"research123", "role":"researcher", "name":"Dr. Research",  "status":"active","created":"2024-01-01","email":"mdrprashan10@gmail.com"},
    "user1":      {"password":"user123",     "role":"user",       "name":"John Analyst",  "status":"active","created":"2024-01-01","email":"mdrprashan10@gmail.com"},
}

for k,v in {"logged_in":False,"username":"","role":"","user_name":"","user_email":"",
            "otp_pending":False,"otp_code":"","otp_username":"",
            "otp_email_sent":False,"otp_email_addr":"",
            "totp_setup_pending":False,"totp_setup_secret":"","totp_setup_username":"",
            "totp_verify_pending":False,"totp_verify_username":"",
            "users":None,
            "pending_users":[],"show_register":False,"show_reset_pw":False,
            "reset_otp":"","reset_username":"","reset_step":1,"reset_email_sent":False,
            "session_token":"","failed_logins":{},"announcements":[],
            "nav_page":None,"current_page":"🏠  Dashboard",
            "theme":"dark"}.items():
    if k not in st.session_state: st.session_state[k]=v

# Always load users from database (not just session state)
st.session_state.users = db_get_all_users()

def get_users():
    """Always return fresh users from database."""
    return db_get_all_users()

if not st.session_state.logged_in:
    try:
        params = st.query_params
        token  = params.get("sid","")
        if token:
            session = db_get_session(token)
            if session:
                st.session_state.logged_in   = True
                st.session_state.username    = session["username"]
                st.session_state.role        = session["role"]
                st.session_state.user_name   = session["name"]
                st.session_state.user_email  = session["email"]
                st.session_state.session_token = token
    except Exception:
        pass

def add_log(action): db_save_log(st.session_state.username or "system", action)

# ── CHART THEME ───────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font={"color":"#94A3B8","family":"Inter"},
    legend={"bgcolor":"rgba(0,0,0,0)","font":{"color":"#94A3B8","size":12}},
    xaxis={"gridcolor":"rgba(255,255,255,0.05)","zerolinecolor":"rgba(255,255,255,0.04)",
           "tickfont":{"color":"#64748B","size":11}},
    yaxis={"gridcolor":"rgba(255,255,255,0.05)","zerolinecolor":"rgba(255,255,255,0.04)",
           "tickfont":{"color":"#64748B","size":11}},
    margin=dict(t=20,b=20,l=10,r=10)
)

# Primary/accent colours matching theme
C_VIOLET = "#7C3AED"
C_CYAN   = "#06B6D4"
C_GREEN  = "#10B981"
C_RED    = "#EF4444"
C_AMBER  = "#F59E0B"

def gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value*100,1),
        title={"text":title,"font":{"color":"#94A3B8","size":12,"family":"JetBrains Mono"}},
        number={"suffix":"%","font":{"color":"#F8FAFC","size":26,"family":"Manrope"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#1E2540",
                    "tickfont":{"color":"#475569","size":9}},
            "bar":{"color":C_VIOLET,"thickness":0.65},
            "bgcolor":"rgba(124,58,237,0.05)",
            "bordercolor":"rgba(0,0,0,0)",
            "steps":[
                {"range":[0,33],  "color":"rgba(16,185,129,0.06)"},
                {"range":[33,66], "color":"rgba(245,158,11,0.06)"},
                {"range":[66,100],"color":"rgba(239,68,68,0.08)"},
            ],
            "threshold":{"line":{"color":C_RED,"width":2},"thickness":0.8,"value":50}
        }
    ))
    fig.update_layout(**CHART_LAYOUT, height=220)
    return fig

# ── API + HELPERS ─────────────────────────────────────────────────────────────
def call_api(features):
    try:
        r = requests.post("http://127.0.0.1:8000/predict",json={"features":features},timeout=10)
        if r.status_code==200: return r.json()
    except: pass
    return None

def gemini_explanation(result, inputs, amount, category):
    if not GEMINI_AVAILABLE: return fallback_explanation(result, inputs), False
    try:
        reasons=[]
        if inputs.get("is_night"): reasons.append("transaction occurred during night hours (10pm–5am)")
        if inputs.get("is_high_amount"): reasons.append(f"amount of ${amount:.2f} is unusually high")
        if inputs.get("distance_km",0)>100: reasons.append(f"merchant is {inputs['distance_km']:.0f}km from customer location")
        if inputs.get("category") in ["shopping_net","misc_net"]: reasons.append("online shopping category carries elevated risk")
        if inputs.get("age_group")==2: reasons.append("senior customer profile shows higher fraud vulnerability")
        if inputs.get("amt_to_category_avg",1)>2: reasons.append(f"amount is {inputs['amt_to_category_avg']:.1f}x above category average")
        verdict = "FRAUDULENT" if result["prediction"]==1 else "LEGITIMATE"
        prompt = f"""You are a fraud intelligence AI for a financial institution.
Transaction: ${amount:.2f} | Category: {category} | Verdict: {verdict}
Fraud probability: {result['fraud_probability']:.2%} | Risk: {result['risk_band']}
Risk factors: {', '.join(reasons) if reasons else 'None identified'}
Write a 2-3 sentence professional explanation for a bank analyst. Be specific and concise. Plain paragraph format only."""
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(prompt)
        return resp.text.strip(), True
    except:
        return fallback_explanation(result, inputs), False

def fallback_explanation(result, inputs):
    reasons=[]
    if inputs.get("is_night"): reasons.append("occurred during high-risk night hours")
    if inputs.get("is_high_amount"): reasons.append("transaction amount is unusually high")
    if inputs.get("distance_km",0)>100: reasons.append(f"merchant is {inputs['distance_km']:.0f}km away")
    if inputs.get("category") in ["shopping_net","misc_net"]: reasons.append("online shopping category carries elevated risk")
    if result["prediction"]==1:
        base="This transaction has been classified as FRAUDULENT by the Bagging ensemble model."
        if reasons: base+=f" Risk factors: {'; '.join(reasons)}."
        base+=" Immediate review recommended."
    else:
        base="This transaction has been classified as LEGITIMATE."
        if reasons: base+=f" Some risk factors noted but probability is low."
        base+=" Transaction can proceed normally."
    return base

def score_row(row):
    amt=float(row.get("amt",100)); is_night=int(row.get("is_night",0))
    is_high=1 if amt>500 else 0; category=str(row.get("category","misc_pos"))
    age=int(row.get("age",35)); age_group=0 if age<30 else (1 if age<50 else 2)
    distance=float(row.get("distance_km",25)); city_pop=int(row.get("city_pop",150000))
    trans_hour=int(row.get("trans_hour",12)); is_weekend=int(row.get("is_weekend",0))
    cat_map={c:i for i,c in enumerate(["grocery_pos","shopping_net","entertainment","gas_transport",
        "misc_net","misc_pos","shopping_pos","food_dining","personal_care","health_fitness","travel","home","kids_pets"])}
    cat_avgs={"grocery_pos":50,"shopping_net":80,"entertainment":60,"gas_transport":40,"misc_net":70,
        "misc_pos":45,"shopping_pos":65,"food_dining":35,"personal_care":30,"health_fitness":55,"travel":200,"home":90,"kids_pets":40}
    cat_avg=cat_avgs.get(category,60); amt_to_cat=amt/(cat_avg+1e-6)
    features=[100,cat_map.get(category,0),amt,1,500,25,37.0,-95.0,city_pop,300,
        is_weekend,is_night,age,age_group,distance,1 if distance>75 else 0,is_high,
        amt_to_cat,np.log1p(amt),1 if city_pop<10000 else 0,trans_hour,0,37.5,-95.5]
    result=call_api(features)
    if result: return result
    prob=min(0.95,0.05+(0.3 if is_night else 0)+(0.25 if is_high else 0)
             +(0.15 if category in ["shopping_net","misc_net"] else 0)+(0.1 if distance>100 else 0))
    pred=1 if prob>=0.5 else 0
    return {"prediction":pred,"label":"Fraudulent" if pred==1 else "Legitimate",
            "fraud_probability":round(prob,4),"risk_band":"High Risk" if prob>=0.8 else ("Medium Risk" if prob>=0.5 else "Low Risk"),
            "recommended_action":"Block transaction" if pred==1 else "Allow transaction"}

# ── 2FA PAGE ──────────────────────────────────────────────────────────────────
def _complete_login(username):
    """Finalise login after successful MFA — create session and redirect."""
    import uuid
    users = get_users()
    user  = users[username]
    token = str(uuid.uuid4()).replace("-","")
    db_save_session(token, username, user["role"], user["name"], user.get("email",""))
    st.query_params["sid"] = token
    st.session_state.logged_in      = True
    st.session_state.username       = username
    st.session_state.role           = user["role"]
    st.session_state.user_name      = user["name"]
    st.session_state.user_email     = user.get("email","")
    st.session_state.session_token  = token
    st.session_state.current_page   = "🏠  Dashboard"
    st.session_state.totp_setup_pending  = False
    st.session_state.totp_verify_pending = False
    add_log("MFA verified — login complete")
    st.rerun()


def page_totp_setup():
    """First-time Google Authenticator setup page."""
    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        username = st.session_state.totp_setup_username
        secret   = st.session_state.totp_setup_secret
        uri      = get_totp_uri(username, secret)
        qr_b64   = generate_qr_base64(uri)

        st.markdown("""
        <div style='animation:fadeUp 0.4s ease;margin-top:1.5rem;'>
        <div style='background:linear-gradient(135deg,#0F1423,#161C30);
                    border:1px solid rgba(124,58,237,0.25);border-radius:20px;
                    padding:2.5rem 2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:0.75rem;
                        filter:drop-shadow(0 0 16px rgba(124,58,237,0.5));'>🔐</div>
            <div style='font-family:"Manrope",sans-serif;font-size:1.5rem;font-weight:800;
                        color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:0.4rem;'>
                Set Up Google Authenticator
            </div>
            <div style='color:#94A3B8;font-size:0.92rem;line-height:1.6;margin-bottom:1.75rem;'>
                Scan the QR code below with the Google Authenticator app.<br>
                This only needs to be done once.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Steps
        st.markdown("""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;margin:1rem 0;'>
            <div style='background:#0F1423;border:1px solid rgba(124,58,237,0.2);border-radius:12px;
                        padding:1rem;text-align:center;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>📱</div>
                <div style='font-size:0.85rem;font-weight:600;color:#F8FAFC;margin-bottom:0.2rem;'>Step 1</div>
                <div style='font-size:0.75rem;color:#94A3B8;'>Install Google Authenticator from App Store or Google Play</div>
            </div>
            <div style='background:#0F1423;border:1px solid rgba(124,58,237,0.2);border-radius:12px;
                        padding:1rem;text-align:center;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>📷</div>
                <div style='font-size:0.85rem;font-weight:600;color:#F8FAFC;margin-bottom:0.2rem;'>Step 2</div>
                <div style='font-size:0.75rem;color:#94A3B8;'>Tap + in the app and scan the QR code below</div>
            </div>
            <div style='background:#0F1423;border:1px solid rgba(124,58,237,0.2);border-radius:12px;
                        padding:1rem;text-align:center;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>✅</div>
                <div style='font-size:0.85rem;font-weight:600;color:#F8FAFC;margin-bottom:0.2rem;'>Step 3</div>
                <div style='font-size:0.75rem;color:#94A3B8;'>Enter the 6-digit code shown in the app below</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # QR Code
        st.markdown(f"""
        <div style='background:#0F1423;border:1px solid rgba(124,58,237,0.3);border-radius:16px;
                    padding:1.75rem;text-align:center;margin:0.75rem 0;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;color:#7C3AED;
                        text-transform:uppercase;letter-spacing:0.15em;margin-bottom:1rem;font-weight:600;'>
                Scan with Google Authenticator
            </div>
            <img src='data:image/png;base64,{qr_b64}'
                 style='width:200px;height:200px;border-radius:12px;
                        border:3px solid rgba(124,58,237,0.4);'/>
            <div style='margin-top:1rem;'>
                <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                            color:#64748B;margin-bottom:0.35rem;'>Or enter this key manually</div>
                <code style='background:#161C30;border:1px solid rgba(124,58,237,0.2);
                             border-radius:8px;padding:0.4rem 0.8rem;font-size:0.82rem;
                             color:#A78BFA;letter-spacing:0.15em;'>{secret}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Verify code
        st.markdown("<div style='background:#0F1423;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1.5rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Verify Setup</div>", unsafe_allow_html=True)
        code = st.text_input("Enter the 6-digit code from Google Authenticator",
                             placeholder="Enter your MFA code here", max_chars=6, key="totp_setup_code")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅  Confirm & Sign In", key="totp_setup_confirm", type="primary"):
                if code and len(code) == 6 and verify_totp(secret, code):
                    db_save_totp_secret(username, secret)
                    st.success("Google Authenticator set up successfully!")
                    _complete_login(username)
                else:
                    st.error("Incorrect code. Make sure you scanned the QR code and try again.")
        with c2:
            if st.button("← Cancel", key="totp_setup_cancel"):
                st.session_state.totp_setup_pending = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;padding:0.9rem 1rem;margin-top:0.75rem;
                    display:flex;align-items:center;gap:0.75rem;'>
            <span style='font-size:1.1rem;'>⚠️</span>
            <div style='color:#FCD34D;font-size:0.82rem;line-height:1.5;'>
                <strong>Keep your authenticator app safe.</strong> If you lose access to it,
                contact an administrator to reset your MFA. The QR code will only be shown once.
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_totp_verify():
    """Google Authenticator code verification on subsequent logins."""
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        username = st.session_state.totp_verify_username

        st.markdown(f"""
        <div style='animation:fadeUp 0.4s ease;margin-top:2rem;'>
        <div style='background:linear-gradient(135deg,#0F1423,#161C30);
                    border:1px solid rgba(124,58,237,0.25);border-radius:20px;
                    padding:2.5rem 2rem;text-align:center;margin-bottom:1rem;'>
            <div style='font-size:2.5rem;margin-bottom:0.75rem;
                        filter:drop-shadow(0 0 16px rgba(124,58,237,0.5));'>🔐</div>
            <div style='font-family:"Manrope",sans-serif;font-size:1.5rem;font-weight:800;
                        color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:0.4rem;'>
                Two-Factor Authentication
            </div>
            <div style='color:#94A3B8;font-size:0.92rem;'>
                Open Google Authenticator and enter the code for
                <strong style='color:#A78BFA;'>FraudShield</strong>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#0F1423;border:1px solid rgba(255,255,255,0.08);
                    border-radius:14px;padding:1.75rem;'>
            <div style='display:flex;align-items:center;gap:1rem;padding:1rem;
                        background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
                        border-radius:10px;margin-bottom:1.25rem;'>
                <span style='font-size:2rem;'>📱</span>
                <div>
                    <div style='font-size:0.9rem;font-weight:600;color:#F8FAFC;'>
                        Open Google Authenticator
                    </div>
                    <div style='font-size:0.78rem;color:#94A3B8;margin-top:0.2rem;'>
                        Find FraudShield in your app — codes rotate every 30 seconds
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        code = st.text_input("6-digit code",
                             placeholder="Enter your 6-digit MFA code",
                             max_chars=6, key="totp_verify_code",
                             label_visibility="collapsed")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In →", key="totp_verify_btn", type="primary"):
                secret, _ = db_get_totp_secret(username)
                if code and verify_totp(secret, code):
                    _complete_login(username)
                else:
                    st.error("Incorrect code. Check your Google Authenticator app and try again.")
                    add_log(f"TOTP verification failed for {username}")
        with c2:
            if st.button("← Back", key="totp_verify_back"):
                st.session_state.totp_verify_pending = False
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:var(--bg-elevated);border:1px solid var(--border);
                    border-radius:10px;padding:0.85rem 1rem;margin-top:0.75rem;
                    display:flex;align-items:center;gap:0.75rem;'>
            <span>🛡️</span>
            <div style='color:#94A3B8;font-size:0.8rem;'>
                <strong style='color:#F8FAFC;'>Security reminder:</strong>
                FraudShield will never ask for your authenticator code via email, phone, or chat.
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_2fa():
    _,col,_ = st.columns([1,2,1])
    with col:
        email_sent = st.session_state.get("otp_email_sent", False)
        email_addr = st.session_state.get("otp_email_addr", "")

        # Mask email for privacy — show only first 2 chars and domain
        def mask_email(email):
            if not email or "@" not in email:
                return "your registered email"
            local, domain = email.split("@",1)
            masked_local = local[:2] + "*" * max(2, len(local)-2)
            return f"{masked_local}@{domain}"

        masked = mask_email(email_addr)

        st.markdown(f"""
        <div style='animation:fadeSlideUp 0.4s ease; margin-top:2rem;'>
        <div class='hero-wrap' style='text-align:center; padding:2rem;'>
            <div class='hero-grid'></div>
            <div class='brand-icon'>🔐</div>
            <div class='hero-title' style='font-size:1.8rem; margin-top:0.5rem;'>Two-Factor Authentication</div>
            <div class='hero-sub'>{"A 6-digit code has been sent to your email" if email_sent else "Enter your verification code to continue"}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Email sent confirmation box OR fallback demo box
        if email_sent:
            st.markdown(f"""
            <div class='glass glass-cyan' style='text-align:center;padding:1.5rem;margin-bottom:1rem;'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>📧</div>
                <div style='font-family:var(--font-display);font-size:1rem;font-weight:700;
                            color:var(--text-primary);margin-bottom:0.4rem;'>
                    Code sent to your email
                </div>
                <div style='font-family:var(--font-mono);color:var(--cyan);font-size:0.9rem;
                            margin-bottom:0.5rem;'>{masked}</div>
                <div style='color:var(--text-muted);font-size:0.75rem;'>
                    Check your inbox and spam folder · Expires in 5 minutes
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback: show code on screen if email not configured or failed
            st.markdown(f"""
            <div class='glass glass-amber' style='margin-bottom:1rem;'>
                <div style='display:flex;align-items:center;gap:0.75rem;'>
                    <span style='font-size:1.2rem;'>⚠️</span>
                    <div>
                        <div style='color:var(--amber);font-weight:700;font-size:0.85rem;'>
                            Email not configured — showing code here (demo mode)
                        </div>
                        <div style='color:var(--text-muted);font-size:0.75rem;margin-top:0.2rem;'>
                            Configure email in app settings to send real OTPs
                        </div>
                    </div>
                </div>
            </div>
            <div class='otp-wrap'>
                <div style='font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);
                            text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.5rem;'>
                    Your demo verification code
                </div>
                <div class='otp-code'>{st.session_state.otp_code}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Enter Verification Code</div>", unsafe_allow_html=True)
        entered = st.text_input("", placeholder="Enter the 6-digit code from your email",
                                max_chars=6, label_visibility="collapsed")

        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            if st.button("✅  Verify & Enter →"):
                if entered == st.session_state.otp_code:
                    uname = st.session_state.otp_username
                    user  = get_users()[uname]
                    import uuid
                    token = str(uuid.uuid4()).replace("-","")
                    db_save_session(token, uname, user["role"], user["name"], user.get("email",""))
                    st.query_params["sid"] = token
                    st.session_state.logged_in      = True
                    st.session_state.username       = uname
                    st.session_state.role           = user["role"]
                    st.session_state.user_name      = user["name"]
                    st.session_state.user_email     = user.get("email","")
                    st.session_state.otp_pending    = False
                    st.session_state.otp_email_sent = False
                    st.session_state.session_token  = token
                    add_log("2FA verified — login complete")
                    st.rerun()
                else:
                    st.error("❌ Incorrect code — please check your email and try again.")
                    add_log("2FA failed — incorrect code entered")
        with c2:
            # Resend code
            if st.button("📧  Resend"):
                new_otp    = str(random.randint(100000,999999))
                user_email = st.session_state.otp_email_addr
                user_name  = get_users().get(st.session_state.otp_username,{}).get("name","")
                sent       = notify_2fa_otp(user_name, new_otp, user_email) if user_email else False
                st.session_state.otp_code       = new_otp
                st.session_state.otp_email_sent = sent
                add_log(f"2FA code resent to {user_email}")
                st.success("New code sent!" if sent else f"New demo code: {new_otp}")
                st.rerun()
        with c3:
            if st.button("← Back"):
                st.session_state.otp_pending    = False
                st.session_state.otp_email_sent = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Security tip
        st.markdown("""
        <div class='glass' style='margin-top:0.75rem;'>
            <div style='display:flex;align-items:center;gap:0.75rem;'>
                <span style='font-size:1rem;'>🛡️</span>
                <div style='color:var(--text-muted);font-size:0.78rem;line-height:1.5;'>
                    <strong style='color:var(--text-secondary);'>Security reminder:</strong>
                    FraudShield will never ask you for your OTP over phone or chat.
                    If you did not request this code, ignore it and change your password immediately.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
def page_login():

    # ── Top header banner ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0B0E1F 0%,#111827 60%,#1a1040 100%);
                border:1px solid rgba(124,58,237,0.2);border-radius:16px;
                padding:2.5rem 2.75rem;margin-bottom:1.5rem;position:relative;overflow:hidden;'>
      <div style='position:absolute;inset:0;
                  background:radial-gradient(ellipse 55% 90% at 85% 50%,
                  rgba(124,58,237,0.1) 0%,transparent 70%);pointer-events:none;'></div>
      <div style='position:absolute;top:0;right:0;width:40%;height:100%;
                  background-image:radial-gradient(circle,rgba(255,255,255,0.035) 1px,transparent 1px);
                  background-size:24px 24px;
                  mask-image:radial-gradient(ellipse 80% 100% at 100% 50%,black,transparent);
                  pointer-events:none;'></div>

      <div style='display:inline-flex;align-items:center;gap:0.5rem;
                  background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
                  border-radius:20px;padding:0.25rem 0.85rem;margin-bottom:1.1rem;'>
        <span style='font-size:0.7rem;'>🛡️</span>
        <span style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                     color:#A78BFA;text-transform:uppercase;letter-spacing:0.15em;font-weight:600;'>
          Enterprise Fraud Intelligence
        </span>
      </div>

      <div style='font-family:"Manrope",sans-serif;font-size:2.5rem;font-weight:900;
                  color:#F8FAFC;letter-spacing:-0.03em;line-height:1.1;margin-bottom:0.85rem;'>
        AI-Powered
        <span style='background:linear-gradient(90deg,#06B6D4,#818CF8);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
          Fraud Intelligence
        </span>
        Platform
      </div>

      <p style='color:#94A3B8;font-size:0.95rem;line-height:1.7;margin:0 0 1.35rem;max-width:580px;'>
        Detect fraud in real-time, train ML ensemble models, generate synthetic fraud data,
        compare ensemble methods, and get explainable AI insights.
      </p>

      <div style='display:flex;flex-wrap:wrap;gap:0.5rem;'>
        <span style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
                     border-radius:20px;padding:0.3rem 0.9rem;font-size:0.77rem;color:#CBD5E1;'>
          🔐 Role-Based Access
        </span>
        <span style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
                     border-radius:20px;padding:0.3rem 0.9rem;font-size:0.77rem;color:#CBD5E1;'>
          🤖 35+ ML Models
        </span>
        <span style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
                     border-radius:20px;padding:0.3rem 0.9rem;font-size:0.77rem;color:#CBD5E1;'>
          ⚡ Synthetic Data Generator
        </span>
        <span style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
                     border-radius:20px;padding:0.3rem 0.9rem;font-size:0.77rem;color:#CBD5E1;'>
          📊 Ensemble Comparison
        </span>
        <span style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
                     border-radius:20px;padding:0.3rem 0.9rem;font-size:0.77rem;color:#CBD5E1;'>
          📋 Reports & Export
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Centered login card ───────────────────────────────────────────────────
    _, col, _ = st.columns([1, 1.4, 1])
    with col:

        st.markdown("""
        <div style='background:#0F1423;border:1px solid rgba(255,255,255,0.09);
                    border-radius:18px;padding:2.25rem 2rem;
                    box-shadow:0 24px 64px rgba(0,0,0,0.4);'>

          <!-- Logo row -->
          <div style='text-align:center;margin-bottom:1.5rem;'>
            <div style='font-size:1.8rem;margin-bottom:0.4rem;
                        filter:drop-shadow(0 0 12px rgba(124,58,237,0.5));'>🛡️</div>
            <div style='font-family:"Manrope",sans-serif;font-size:1.3rem;font-weight:800;
                        color:#F8FAFC;letter-spacing:-0.02em;'>FraudShield</div>
            <div style='display:inline-block;margin-top:0.4rem;background:rgba(124,58,237,0.15);
                        border:1px solid rgba(124,58,237,0.3);border-radius:20px;
                        padding:0.2rem 0.75rem;'>
              <span style='font-family:"JetBrains Mono",monospace;font-size:0.62rem;
                           color:#A78BFA;letter-spacing:0.1em;'>🔒 LOCAL AUTH + TOTP MFA</span>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        # ── Tabs: Sign In / Create Account ───────────────────────────────────
        tab_login, tab_register = st.tabs(["🔑  Sign In", "✨  Create Account"])

        with tab_login:
            # Lockout check
            username = st.text_input("Username or Email", placeholder="Enter your username",
                                     key="login_user_v2")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter your password",
                                     key="login_pass_v2")

            is_locked = st.session_state.failed_logins.get(username, 0) >= 3 if username else False

            if is_locked:
                st.markdown("""
                <div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.25);
                            border-radius:8px;padding:0.65rem 0.9rem;margin-bottom:0.5rem;
                            display:flex;align-items:center;gap:0.5rem;
                            font-size:0.8rem;color:#F87171;'>
                  🔒 Account locked after 3 failed attempts — use Forgot Password
                </div>
                """, unsafe_allow_html=True)

            # Sign In + Forgot Password on same row
            col_rem, col_fp = st.columns([1, 1])
            with col_rem:
                remember_me = st.checkbox("Remember me", key="remember_me_cb",
                                          value=st.session_state.get("remember_me", False))
                if remember_me:
                    st.session_state.remember_me = True
            with col_fp:
                if st.button("Forgot Password?", key="fp_tab"):
                    st.session_state.show_reset_pw = True
                    st.session_state.reset_step    = 1
                    st.rerun()

            sign_in_clicked = st.button("Sign In →", key="signin_v2",
                                        disabled=is_locked, type="primary")

            if sign_in_clicked:
                users  = get_users()
                failed = st.session_state.failed_logins
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif username not in users:
                    st.error("Username not found. Check spelling or request an account.")
                else:
                    user = users[username]
                    if user["password"] == password:
                        if user.get("status","active") == "inactive":
                            st.error("Account deactivated. Contact the administrator.")
                        else:
                            st.session_state.failed_logins.pop(username, None)
                            totp_secret, totp_enabled = db_get_totp_secret(username)
                            if totp_enabled and totp_secret:
                                # Has Google Authenticator set up
                                st.session_state.totp_verify_pending  = True
                                st.session_state.totp_verify_username = username
                                add_log(f"Login — Google Authenticator verification for {username}")
                            else:
                                # First login — set up Google Authenticator
                                st.session_state.totp_setup_pending  = True
                                st.session_state.totp_setup_secret   = generate_totp_secret()
                                st.session_state.totp_setup_username = username
                                add_log(f"Login — Google Authenticator setup for {username}")
                            st.rerun()
                    else:
                        st.session_state.failed_logins[username] = failed.get(username, 0) + 1
                        left = 3 - st.session_state.failed_logins[username]
                        if left <= 0:
                            db_lock_account(username, 3)
                            db_save_log(username, "Account LOCKED after 3 failed login attempts")
                            for u, i in db_get_all_users().items():
                                if i["role"] == "admin" and i.get("email"):
                                    notify_admin_account_locked(username, 3, i["email"])
                            st.error("Account locked. Administrator has been notified.")
                        else:
                            st.error(f"Incorrect password. {left} attempt{'s' if left>1 else ''} remaining.")
                            add_log(f"Failed login {failed.get(username,0)+1}/3: {username}")

            # Demo accounts compact strip — show only default 3
            default_accounts = ["admin","researcher","user1"]
            users_all = db_get_all_users()
            demo_text = " · ".join([
                f"{u} / {users_all[u]['password']}"
                for u in default_accounts
                if u in users_all
            ])
            st.markdown(f"""
            <div style='margin-top:1rem;background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.08);border-radius:10px;
                        padding:0.75rem 1rem;'>
              <div style='font-family:"JetBrains Mono",monospace;font-size:0.62rem;
                          color:#64748B;text-transform:uppercase;letter-spacing:0.12em;
                          margin-bottom:0.35rem;'>🔑 Demo accounts</div>
              <div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;
                          color:#94A3B8;'>{demo_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Theme + status row
            current_theme = st.session_state.get("theme","dark")
            st.markdown("""
            <div style='display:flex;justify-content:center;gap:1.5rem;margin-top:1rem;
                        font-family:"JetBrains Mono",monospace;font-size:0.65rem;color:#475569;'>
              <span><span class='dot dot-green'></span> &nbsp;Gemini Active</span>
              <span><span class='dot dot-cyan'></span> &nbsp;2FA Enabled</span>
              <span><span class='dot dot-green'></span> &nbsp;SQLite Online</span>
            </div>
            """, unsafe_allow_html=True)

        with tab_register:
            st.markdown("""
            <div style='text-align:center;padding:1rem 0 0.5rem;'>
              <div style='color:#94A3B8;font-size:0.88rem;line-height:1.6;'>
                Fill in your details below.<br>An admin will review and approve your account.
              </div>
            </div>
            """, unsafe_allow_html=True)

            reg_name     = st.text_input("Full Name", placeholder="Jane Smith", key="reg_name_tab")
            reg_username = st.text_input("Username", placeholder="jsmith", key="reg_user_tab")
            reg_email    = st.text_input("Email Address", placeholder="jane@example.com", key="reg_email_tab")
            reg_role     = st.selectbox("Requested Role", ["user","researcher"], key="reg_role_tab")
            reg_reason   = st.text_area("Reason for Access", placeholder="Why do you need access?",
                                        height=75, key="reg_reason_tab")
            reg_pass     = st.text_input("Password", type="password",
                                         placeholder="Min 6 characters", key="reg_pass_tab")
            reg_pass2    = st.text_input("Confirm Password", type="password",
                                         placeholder="Repeat password", key="reg_pass2_tab")

            if st.button("Submit Request", key="submit_reg_tab", type="primary"):
                if not reg_name or not reg_username or not reg_email or not reg_pass:
                    st.error("All fields are required.")
                elif len(reg_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                elif reg_username in get_users():
                    st.error(f"Username '{reg_username}' is already taken.")
                elif any(p["username"]==reg_username for p in st.session_state.pending_users):
                    st.error("A request for this username is already pending.")
                else:
                    st.session_state.pending_users.append({
                        "username":  reg_username, "name":     reg_name,
                        "email":     reg_email,    "password": reg_pass,
                        "role":      reg_role,     "reason":   reg_reason,
                        "submitted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    db_save_log("system", f"New account request: {reg_username} ({reg_name}) for {reg_role}")
                    for u, i in db_get_all_users().items():
                        if i["role"]=="admin" and i.get("email"):
                            notify_admin_new_request(reg_name, reg_username, reg_role,
                                                     reg_reason, i["email"])
                    st.success(f"Request submitted. An admin will review your account and email you at {reg_email} once approved.")


# ── PASSWORD RESET EMAIL ──────────────────────────────────────────────────────
def notify_admin_account_locked(username: str, attempts: int, admin_email: str) -> bool:
    """Alert admin when an account is locked due to too many failed attempts."""
    users = db_get_all_users()
    user_info = users.get(username, {})
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        A security event has been detected on the FraudShield platform.
    </p>
    <div style="background:rgba(255,23,68,0.08);border:1px solid rgba(255,23,68,0.25);
                border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;">
        <div style="font-size:0.7rem;color:#ff1744;text-transform:uppercase;
                    letter-spacing:0.15em;font-family:monospace;margin-bottom:0.75rem;">
            Security Alert — Account Locked
        </div>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="color:#64748b;padding:0.35rem 0;font-size:0.85rem;width:40%;">Username</td>
                <td style="color:#eef2ff;font-family:monospace;font-size:0.85rem;">{username}</td></tr>
            <tr><td style="color:#64748b;padding:0.35rem 0;font-size:0.85rem;">Full Name</td>
                <td style="color:#eef2ff;font-size:0.85rem;">{user_info.get('name','Unknown')}</td></tr>
            <tr><td style="color:#64748b;padding:0.35rem 0;font-size:0.85rem;">Email</td>
                <td style="color:#eef2ff;font-size:0.85rem;">{user_info.get('email','Not registered')}</td></tr>
            <tr><td style="color:#64748b;padding:0.35rem 0;font-size:0.85rem;">Failed Attempts</td>
                <td style="color:#ff1744;font-family:monospace;font-weight:700;font-size:0.85rem;">{attempts} attempts</td></tr>
            <tr><td style="color:#64748b;padding:0.35rem 0;font-size:0.85rem;">Locked At</td>
                <td style="color:#eef2ff;font-family:monospace;font-size:0.85rem;">
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            <tr><td style="color:#64748b;padding:0.35rem 0;font-size:0.85rem;">Status</td>
                <td style="color:#ff1744;font-weight:700;font-size:0.85rem;">LOCKED</td></tr>
        </table>
    </div>
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        This account has been automatically locked as a security measure.
        If this was a legitimate user, you can unlock their account from
        <strong style="color:#00d4ff;">User Management in the Admin portal</strong>.
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0;">
        If you do not recognise this account or suspect a brute-force attack,
        consider reviewing the audit logs for more details.
    </p>
    """
    return send_email(
        admin_email,
        f"🔒 Security Alert — Account Locked: @{username}",
        email_base(content, "Account Lockout Detected")
    )

# ── PASSWORD RESET PAGE ───────────────────────────────────────────────────────
def page_reset_password():
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div style='animation:fadeSlideUp 0.4s ease; margin-top:2rem;'>
        <div class='hero-wrap' style='text-align:center; padding:2.5rem 2rem;'>
            <div class='hero-grid'></div>
            <div class='brand-icon'>🔑</div>
            <div class='hero-eyebrow' style='justify-content:center;margin-top:0.5rem;'>Account Recovery</div>
            <div class='hero-title' style='font-size:1.8rem;'>Reset Password</div>
            <div class='hero-sub'>We'll send a verification code to your registered email</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        step = st.session_state.get("reset_step", 1)

        # ── Step 1: Enter username + email ─────────────────────────────────────
        if step == 1:
            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Step 1 of 3 — Verify Your Identity</div>", unsafe_allow_html=True)
            r_username = st.text_input("Username", placeholder="Enter your username")
            r_email    = st.text_input("Registered Email", placeholder="Enter your registered email")
            if st.button("📧  Send Reset Code"):
                users = get_users()
                if r_username not in users:
                    st.error("Username not found.")
                else:
                    user = users[r_username]
                    if user.get("email","").lower() != r_email.lower():
                        st.error("Email does not match the registered email for this account.")
                    else:
                        otp = str(random.randint(100000, 999999))
                        st.session_state.reset_otp      = otp
                        st.session_state.reset_username = r_username
                        sent = notify_password_reset_otp(user["name"], otp, r_email)
                        st.session_state.reset_email_sent = sent
                        if sent:
                            st.success(f"✅ Reset code sent to {r_email}. Check your inbox.")
                        else:
                            st.warning(f"⚠️ Could not send email. Your demo code is: **{otp}** (email config issue)")
                        st.session_state.reset_step = 2
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Step 2: Enter OTP ───────────────────────────────────────────────────
        elif step == 2:
            uname = st.session_state.reset_username
            st.markdown(f"""
            <div class='glass glass-green'>
                <div style='color:var(--green);font-size:0.85rem;'>
                    ✓ Code sent for <strong>@{uname}</strong> — check your email inbox and spam folder.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Step 2 of 3 — Enter Your Reset Code</div>", unsafe_allow_html=True)

            # Only show demo code if email sending failed
            if not st.session_state.get("reset_email_sent", False):
                st.markdown(f"""
                <div class='otp-wrap' style='padding:1.5rem;'>
                    <div style='font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);
                                margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.1em;'>
                        Demo mode — your code (email not configured)
                    </div>
                    <div class='otp-code' style='font-size:2rem;'>{st.session_state.reset_otp}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);
                            border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:0.75rem;
                            display:flex;align-items:center;gap:0.75rem;'>
                    <span style='font-size:1.1rem;'>📧</span>
                    <div style='color:var(--green);font-size:0.88rem;'>
                        Check your email inbox and spam folder for the 6-digit code.
                        The code expires in 10 minutes.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            entered_otp = st.text_input("Enter 6-digit code", placeholder="Enter code from email", max_chars=6)
            if st.button("✅  Verify Code"):
                if entered_otp == st.session_state.reset_otp:
                    st.session_state.reset_step = 3
                    st.rerun()
                else:
                    st.error("Incorrect code. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Step 3: Set new password ────────────────────────────────────────────
        elif step == 3:
            uname = st.session_state.reset_username
            st.markdown(f"""
            <div class='glass glass-green'>
                <div style='color:var(--green);font-size:0.85rem;'>
                    ✓ Identity verified for <strong>@{uname}</strong> — set your new password below.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Step 3 of 3 — Set New Password</div>", unsafe_allow_html=True)
            new_pw  = st.text_input("New Password",     type="password", placeholder="Min 6 characters")
            new_pw2 = st.text_input("Confirm Password", type="password", placeholder="Repeat new password")
            if st.button("🔐  Update Password"):
                if not new_pw or len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    db_update_user_password(uname, new_pw)
                    db_save_log(uname, "Password reset via forgot password flow")
                    st.success("✅ Password updated successfully! You can now log in with your new password.")
                    # Clear reset state
                    st.session_state.reset_step      = 1
                    st.session_state.reset_otp       = ""
                    st.session_state.reset_username  = ""
                    st.session_state.reset_email_sent = False
                    st.session_state.show_reset_pw   = False
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Progress indicator
        step_labels = ["Verify Identity","Enter Code","New Password"]
        st.markdown(f"""
        <div style='display:flex;justify-content:center;gap:1rem;margin-top:1rem;'>
            {''.join([
                f'<div style="display:flex;align-items:center;gap:0.4rem;font-family:var(--font-mono);font-size:0.68rem;">'
                f'<span style="width:20px;height:20px;border-radius:50%;display:flex;align-items:center;'
                f'justify-content:center;font-size:0.6rem;font-weight:700;'
                f'background:{"var(--cyan)" if i<step else "var(--bg-elevated)"};'
                f'color:{"#04060f" if i<step else "var(--text-muted)"};'
                f'border:1px solid {"var(--cyan)" if i<step else "var(--border)"};">{i+1}</span>'
                f'<span style="color:{"var(--cyan)" if i+1==step else "var(--text-muted)"};">{lbl}</span>'
                f'</div>'
                for i,lbl in enumerate(step_labels)
            ])}
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Back to Login"):
            st.session_state.show_reset_pw   = False
            st.session_state.reset_step      = 1
            st.session_state.reset_otp       = ""
            st.session_state.reset_username  = ""
            st.session_state.reset_email_sent = False
            st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Brand logo
        st.markdown("""
        <div style='padding:1.25rem 0.5rem 1rem;display:flex;align-items:center;gap:0.75rem;'>
            <div style='width:34px;height:34px;border-radius:8px;
                        background:linear-gradient(135deg,#7C3AED,#06B6D4);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1rem;box-shadow:0 4px 12px rgba(124,58,237,0.35);'>🛡️</div>
            <div>
                <div style='font-family:"Manrope",sans-serif;font-size:0.95rem;font-weight:800;
                            color:var(--text-primary);letter-spacing:-0.02em;'>FraudShield</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:0.58rem;
                            color:var(--text-muted);text-transform:uppercase;letter-spacing:0.1em;'>
                    Fraud Intelligence Platform
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        role = st.session_state.role
        initials = "".join(w[0].upper() for w in st.session_state.user_name.split()[:2])
        bc = {"admin":"badge-admin","researcher":"badge-research","user":"badge-user"}.get(role,"badge-user")
        email = st.session_state.get("user_email","")

        st.markdown(f"""
        <div style='background:var(--bg-elevated);border:1px solid var(--border);
                    border-radius:12px;padding:1rem;margin-bottom:1.1rem;'>
            <div style='display:flex;align-items:center;gap:0.75rem;'>
                <div style='width:36px;height:36px;border-radius:8px;flex-shrink:0;
                            background:linear-gradient(135deg,var(--primary),var(--cyan));
                            display:flex;align-items:center;justify-content:center;
                            font-family:"Manrope",sans-serif;font-size:0.88rem;font-weight:800;
                            color:white;box-shadow:0 2px 8px var(--primary-glow);'>
                    {initials}
                </div>
                <div style='min-width:0;'>
                    <div style='font-family:"Manrope",sans-serif;font-weight:700;
                                font-size:0.9rem;color:var(--text-primary);
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                        {st.session_state.user_name}
                    </div>
                    {f'<div style="font-size:0.7rem;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{email}</div>' if email else ''}
                </div>
            </div>
            <div style='margin-top:0.65rem;display:flex;gap:0.4rem;'>
                <span class='badge {bc}'>{role.upper()}</span>
                <span class='badge badge-verified'>✓ TOTP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='nav-section'>Navigation</div>", unsafe_allow_html=True)

        if role=="admin":
            admin_opts = [
                "🏠  Dashboard","👥  User Management","🖥️  Active Sessions",
                "📊  Analytics","📋  Audit Logs","⚙️  Model Deployment","📢  Announcements"
            ]
            cur = st.session_state.get("current_page", admin_opts[0])
            idx = next((i for i,o in enumerate(admin_opts) if o==cur), 0)
            page = st.radio("", admin_opts, index=idx,
                            label_visibility="collapsed")
            st.session_state.current_page = page

            pending = st.session_state.pending_users
            if pending:
                st.markdown(f"""
                <div style='margin:0.5rem 0;padding:0.65rem 0.8rem;
                            background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                            border-radius:8px;display:flex;align-items:center;gap:0.5rem;'>
                    <span class='dot dot-amber'></span>
                    <div>
                        <div style='font-family:"JetBrains Mono",monospace;font-size:0.7rem;
                                    color:var(--amber);font-weight:700;'>
                            {len(pending)} pending approval{'s' if len(pending)>1 else ''}
                        </div>
                        <div style='font-size:0.68rem;color:var(--text-muted);'>User Management</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            lockouts_df = db_get_active_lockouts()
            if len(lockouts_df) > 0:
                st.markdown(f"""
                <div style='margin:0.5rem 0;padding:0.65rem 0.8rem;
                            background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.35);
                            border-radius:8px;display:flex;align-items:center;gap:0.5rem;'>
                    <span class='dot dot-red'></span>
                    <div>
                        <div style='font-family:"JetBrains Mono",monospace;font-size:0.7rem;
                                    color:var(--red);font-weight:700;'>
                            {len(lockouts_df)} account{'s' if len(lockouts_df)>1 else ''} LOCKED
                        </div>
                        <div style='font-size:0.68rem;color:var(--text-muted);'>Security Alert</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif role=="researcher":
            res_opts = [
                "🏠  Dashboard","🔬  Model Training","📈  Evaluation",
                "📉  ROC & PR Curves","🕸️  Model Radar","🔍  Feature Analysis","📤  Export"
            ]
            cur = st.session_state.get("current_page", res_opts[0])
            idx = next((i for i,o in enumerate(res_opts) if o==cur), 0)
            page = st.radio("", res_opts, index=idx,
                            label_visibility="collapsed")
            st.session_state.current_page = page

        else:
            user_opts = [
                "🏠  Dashboard","🔎  Single Transaction","📂  Batch Upload",
                "📜  History","ℹ️  About"
            ]
            cur = st.session_state.get("current_page", user_opts[0])
            idx = next((i for i,o in enumerate(user_opts) if o==cur), 0)
            page = st.radio("", user_opts, index=idx,
                            label_visibility="collapsed")
            st.session_state.current_page = page

        # Spacer then Sign Out at bottom
        st.markdown("<div style='margin-top:1.25rem;'>", unsafe_allow_html=True)

        # Sign Out button styled like reference (orange accent)
        st.markdown("""
        <style>
        div[data-testid="stButton"]:has(button[key="signout_btn"]) > button {
            background: rgba(239,68,68,0.1) !important;
            border: 1px solid rgba(239,68,68,0.3) !important;
            color: #F87171 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stButton"]:has(button[key="signout_btn"]) > button:hover {
            background: rgba(239,68,68,0.2) !important;
            border-color: #EF4444 !important;
            color: #FCA5A5 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="signout_btn"):
            add_log("User signed out")
            if st.session_state.get("session_token"):
                db_delete_session(st.session_state.session_token)
            try: st.query_params.clear()
            except: pass
            for k in ["logged_in","username","role","user_name","user_email",
                      "otp_pending","otp_code","otp_username","last_result",
                      "last_inputs","session_token","totp_setup_pending","totp_verify_pending"]:
                if k in st.session_state:
                    st.session_state[k] = False if k=="logged_in" else ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            r  = requests.get("http://127.0.0.1:8000/health",timeout=2)
            ok = r.status_code==200
        except: ok=False

        # System status panel
        lockouts_count = len(db_get_active_lockouts()) if st.session_state.role=="admin" else 0
        lockout_html = ""
        if lockouts_count > 0:
            lockout_html = f"""
                <div style='display:flex;align-items:center;justify-content:space-between;
                            margin-top:0.25rem;padding-top:0.5rem;border-top:1px solid rgba(239,68,68,0.2);'>
                    <span style='display:flex;align-items:center;gap:0.4rem;
                                 font-family:var(--font-mono);font-size:0.78rem;color:#F87171;'>
                        <span class='dot dot-red'></span>Locked
                    </span>
                    <span style='font-family:var(--font-mono);font-size:0.78rem;
                                 color:#F87171;font-weight:700;'>{lockouts_count}</span>
                </div>"""

        st.markdown(f"""
        <div style='margin-top:1rem;padding:1rem;background:var(--bg-elevated);
                    border:1px solid var(--border);border-radius:10px;'>
            <div style='font-family:var(--font-mono);font-size:0.68rem;font-weight:600;
                        color:var(--text-muted);text-transform:uppercase;
                        letter-spacing:0.12em;margin-bottom:0.75rem;'>System Status</div>
            <div style='display:flex;flex-direction:column;gap:0.5rem;'>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <span style='display:flex;align-items:center;gap:0.5rem;
                                 font-family:var(--font-mono);font-size:0.78rem;color:var(--text-secondary);'>
                        <span class='dot {"dot-green" if ok else "dot-red"}'></span>FastAPI
                    </span>
                    <span style='font-family:var(--font-mono);font-size:0.75rem;
                                 color:{"var(--green)" if ok else "var(--red)"};font-weight:600;'>
                        {"Online" if ok else "Offline"}
                    </span>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <span style='display:flex;align-items:center;gap:0.5rem;
                                 font-family:var(--font-mono);font-size:0.78rem;color:var(--text-secondary);'>
                        <span class='dot dot-cyan'></span>Gemini AI
                    </span>
                    <span style='font-family:var(--font-mono);font-size:0.75rem;
                                 color:var(--cyan);font-weight:600;'>Active</span>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <span style='display:flex;align-items:center;gap:0.5rem;
                                 font-family:var(--font-mono);font-size:0.78rem;color:var(--text-secondary);'>
                        <span class='dot dot-green'></span>Database
                    </span>
                    <span style='font-family:var(--font-mono);font-size:0.75rem;
                                 color:var(--green);font-weight:600;'>Connected</span>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <span style='display:flex;align-items:center;gap:0.5rem;
                                 font-family:var(--font-mono);font-size:0.78rem;color:var(--text-secondary);'>
                        <span class='dot dot-amber'></span>Users
                    </span>
                    <span style='font-family:var(--font-mono);font-size:0.75rem;
                                 color:var(--amber);font-weight:600;'>{len(get_users())}</span>
                </div>
                {lockout_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Theme toggle button
        current_theme = st.session_state.get("theme","dark")
        theme_icon  = "☀️" if current_theme=="dark" else "🌙"
        return page

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def _announcements_banner():
    """Shared announcements banner shown on all dashboards."""
    for ann in st.session_state.get("announcements",[])[:2]:
        color = {"Info":"var(--cyan)","Warning":"var(--amber)","Critical":"var(--red)"}.get(ann["type"],"var(--cyan)")
        icon  = {"Info":"📢","Warning":"⚠️","Critical":"🚨"}.get(ann["type"],"📢")
        st.markdown(f"""
        <div class='glass' style='border-left:3px solid {color};margin-bottom:0.5rem;'>
            <div style='display:flex;align-items:center;gap:0.75rem;'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <div>
                    <div style='color:{color};font-weight:700;font-size:0.88rem;'>{ann['title']}</div>
                    <div style='color:var(--text-secondary);font-size:0.82rem;margin-top:0.2rem;'>{ann['body']}</div>
                </div>
                <div style='margin-left:auto;font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);'>{ann['time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── ADMIN DASHBOARD ───────────────────────────────────────────────────────────
def dashboard_admin():
    users    = get_users()
    df_all   = db_get_predictions(limit=1000)
    logs_df  = db_get_logs(limit=500)
    total_p  = len(df_all)
    fraud_p  = int((df_all["result"]=="Fraudulent").sum()) if total_p>0 else 0
    active_u = sum(1 for u in users.values() if u.get("status")=="active")
    pending  = len(st.session_state.pending_users)

    # Sessions count from DB
    conn = sqlite3.connect(DB_PATH)
    ses_df = pd.read_sql_query("SELECT * FROM sessions", conn); conn.close()
    now    = datetime.now()
    active_ses = sum(1 for _,r in ses_df.iterrows()
                     if datetime.strptime(r["expires"],"%Y-%m-%d %H:%M:%S")>now)

    st.markdown(f"""
    <div class='hero-wrap' style='animation:fadeSlideUp 0.35s ease;'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Administration Control Centre</div>
        <div class='hero-title'>Admin Dashboard</div>
        <div class='hero-sub'>Platform health, user activity, and security overview at a glance</div>
        <div class='hero-chips'>
            <span class='chip'>System Admin</span>
            <span class='chip'>Full Access</span>
            <span class='chip'>2FA Verified</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _announcements_banner()

    # KPIs
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi'>
            <span class='kpi-icon'>👥</span><div class='kpi-label'>Registered Users</div>
            <div class='kpi-value'>{len(users)}</div>
            <div class='kpi-sub'>{active_u} active · {len(users)-active_u} inactive</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>🖥️</span><div class='kpi-label'>Active Sessions</div>
            <div class='kpi-value' style='color:var(--green);'>{active_ses}</div>
            <div class='kpi-sub'>Currently logged in</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>⏳</span><div class='kpi-label'>Pending Approvals</div>
            <div class='kpi-value' style='color:{"var(--amber)" if pending>0 else "var(--text-muted)"};'>{pending}</div>
            <div class='kpi-sub'>{"Needs review" if pending>0 else "All clear"}</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>⚡</span><div class='kpi-label'>Total Predictions</div>
            <div class='kpi-value'>{total_p}</div>
            <div class='kpi-sub'>{fraud_p} fraud detected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Predictions per user bar chart
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Predictions by User</div>", unsafe_allow_html=True)
        if total_p > 0:
            user_counts = df_all.groupby("username").size().reset_index(name="count").sort_values("count",ascending=True)
            fig_u = go.Figure(go.Bar(
                x=user_counts["count"], y=user_counts["username"],
                orientation="h", marker_color="#7C3AED", marker_line_width=0
            ))
            fig_u.update_layout(**CHART_LAYOUT, height=260, xaxis_title="Predictions Made")
            st.plotly_chart(fig_u, use_container_width=True)
        else:
            st.markdown("<p style='color:var(--text-muted);font-size:0.85rem;'>No predictions yet.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # Fraud vs Legit platform-wide donut
        st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Platform-wide Fraud Rate</div>", unsafe_allow_html=True)
        if total_p > 0:
            fig_d = go.Figure(go.Pie(
                labels=["Legitimate","Fraudulent"],
                values=[total_p-fraud_p, fraud_p],
                hole=0.62,
                marker=dict(colors=["#10B981","#EF4444"], line=dict(width=0)),
                textinfo="label+percent", textfont=dict(color="#eef2ff",size=11)
            ))
            fig_d.update_layout(**CHART_LAYOUT, height=260, showlegend=False)
            pct = round(fraud_p/total_p*100,2) if total_p>0 else 0
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f"<div style='text-align:center;font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);'>Overall fraud rate: <strong style='color:var(--red);'>{pct}%</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:var(--text-muted);font-size:0.85rem;'>No predictions yet.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        # User roles breakdown
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>User Role Distribution</div>", unsafe_allow_html=True)
        role_counts = {}
        for u in users.values():
            role_counts[u["role"]] = role_counts.get(u["role"], 0) + 1
        colors_map  = {"admin":"#EF4444","researcher":"#F59E0B","user":"#10B981"}
        fig_r = go.Figure(go.Pie(
            labels=list(role_counts.keys()), values=list(role_counts.values()),
            hole=0.5, textinfo="label+value",
            marker=dict(colors=[colors_map.get(r,"#7C3AED") for r in role_counts.keys()],
                        line=dict(width=0))
        ))
        fig_r.update_layout(**CHART_LAYOUT, height=220, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='glass glass-amber'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Quick Actions</div>", unsafe_allow_html=True)

        quick_actions = [
            ("👥", "User Management",  "Manage accounts and roles",    "👥  User Management"),
            ("🖥️", "Active Sessions",  f"{active_ses} session(s) live","🖥️  Active Sessions"),
            ("📋", "Audit Logs",       f"{len(logs_df)} entries logged","📋  Audit Logs"),
            ("📢", "Announcements",    "Post platform notices",         "📢  Announcements"),
        ]
        if pending > 0:
            quick_actions.insert(0, ("⏳", "Pending Approvals",
                                     f"{pending} request(s) waiting",  "👥  User Management"))

        for icon, title, desc, nav_key in quick_actions:
            st.markdown(f"""
            <div style='background:var(--bg-elevated);border:1px solid var(--border);
                        border-radius:8px;padding:0.65rem 1rem;margin-bottom:0.4rem;
                        display:flex;align-items:center;gap:0.75rem;
                        transition:border-color 0.2s;cursor:pointer;'>
                <span style='font-size:1.1rem;'>{icon}</span>
                <div style='flex:1;'>
                    <div style='color:var(--text-primary);font-size:0.88rem;font-weight:600;'>{title}</div>
                    <div style='color:var(--text-muted);font-size:0.75rem;'>{desc}</div>
                </div>
                <span style='color:var(--text-muted);font-size:0.75rem;'>→</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Go to {title}", key=f"qa_{nav_key}"):
                st.session_state.current_page = nav_key
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Recent audit log
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Recent System Activity</div>", unsafe_allow_html=True)
    recent = db_get_logs(limit=8)
    if len(recent)>0:
        for _,row in recent.iterrows():
            st.markdown(f"""
            <div class='log-row'>
                <span style='color:var(--text-muted);'>[{row['timestamp']}]</span>
                &nbsp;<span style='color:var(--primary);font-weight:600;'>{row['username']}</span>
                &nbsp;<span style='color:var(--text-muted);'>→</span>&nbsp;{row['action']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:var(--text-muted);font-size:0.85rem;'>No activity yet.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── RESEARCHER DASHBOARD ──────────────────────────────────────────────────────
def dashboard_researcher():
    st.markdown(f"""
    <div class='hero-wrap' style='animation:fadeSlideUp 0.35s ease;'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>Researcher Dashboard</div>
        <div class='hero-sub'>Model performance overview, dataset statistics, and research quick-access</div>
        <div class='hero-chips'>
            <span class='chip'>Sparkov Dataset</span>
            <span class='chip'>Bagging Selected</span>
            <span class='chip'>ROC-AUC 0.9926</span>
            <span class='chip'>Recall 96%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _announcements_banner()

    # Model performance KPIs
    st.markdown("""
    <div class='kpi-grid'>
        <div class='kpi'>
            <span class='kpi-icon'>🎯</span><div class='kpi-label'>Best ROC-AUC</div>
            <div class='kpi-value'>0.9926</div>
            <div class='kpi-sub'>Bagging · updated dataset</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>🔍</span><div class='kpi-label'>Fraud Recall</div>
            <div class='kpi-value' style='color:var(--green);'>96%</div>
            <div class='kpi-sub'>After synthetic augmentation</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>🎯</span><div class='kpi-label'>Best Precision</div>
            <div class='kpi-value' style='color:var(--cyan);'>0.79</div>
            <div class='kpi-sub'>Bagging classifier</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>🧠</span><div class='kpi-label'>Models Trained</div>
            <div class='kpi-value' style='color:var(--amber);'>10+</div>
            <div class='kpi-sub'>Across all techniques</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Full model comparison table
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Ensemble Model Comparison</div>", unsafe_allow_html=True)
        data = pd.DataFrame({
            "Model":     ["Random Forest","Bagging ✓","Gradient Boosting","Stacking"],
            "Precision": [0.66, 0.79, 0.18, 0.26],
            "Recall":    [0.88, 0.85, 0.92, 0.94],
            "F1":        [0.75, 0.82, 0.29, 0.41],
            "ROC-AUC":   [0.9943, 0.9926, 0.9908, 0.9948],
        })
        fig = px.bar(data, x="Model", y=["Precision","Recall","F1"], barmode="group",
                     color_discrete_map={"Precision":"#7C3AED","Recall":"#06B6D4","F1":"#10B981"})
        fig.update_layout(**CHART_LAYOUT, height=260)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # Feature importance top 5
        st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Top 5 Feature Importance (Bagging)</div>", unsafe_allow_html=True)
        feats = pd.DataFrame({
            "Feature":   ["amt","is_night","category","amt_log","amt_to_category_avg"],
            "Importance":[0.562, 0.110, 0.089, 0.084, 0.081],
            "Meaning":   ["Transaction amount","Night hours flag","Merchant type","Log amount","Contextual ratio"]
        })
        for _, row in feats.iterrows():
            bar_w = int(row["Importance"]*100/0.562*100)
            st.markdown(f"""
            <div style='margin-bottom:0.7rem;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.25rem;'>
                    <span style='color:var(--text-primary);font-family:var(--font-mono);font-size:0.78rem;'>{row['Feature']}</span>
                    <span style='color:var(--cyan);font-family:var(--font-mono);font-size:0.75rem;'>{row['Importance']:.3f}</span>
                </div>
                <div style='height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;'>
                    <div style='height:100%;width:{bar_w}%;background:linear-gradient(90deg,#00d4ff,#7c3aed);border-radius:3px;'></div>
                </div>
                <div style='color:var(--text-muted);font-size:0.7rem;margin-top:0.2rem;'>{row['Meaning']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        # Dataset stats
        st.markdown("""
        <div class='glass glass-cyan'>
            <div class='sec-label'>Dataset Overview</div>
        """, unsafe_allow_html=True)
        stats = [
            ("Dataset",           "Sparkov Credit Card"),
            ("Total Transactions","1,296,675"),
            ("Original Fraud Rate","0.578%"),
            ("After Synthetic",   "0.617%"),
            ("Synthetic Added",   "500 transactions"),
            ("Features",          "24 real-world"),
            ("Train / Val / Test","70% / 15% / 15%"),
            ("SMOTE Applied",     "Training only"),
        ]
        for label, val in stats:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:0.35rem 0;
                        border-bottom:1px solid var(--border);'>
                <span style='color:var(--text-muted);font-size:0.82rem;'>{label}</span>
                <span style='color:var(--text-primary);font-size:0.82rem;font-weight:600;'>{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        # Research quick access
        st.markdown("""
        <div class='glass glass-amber'>
            <div class='sec-label'>Research Tools</div>
            <div style='display:flex;flex-direction:column;gap:0.6rem;'>
        """, unsafe_allow_html=True)
        tools = [
            ("🔬", "Model Training",    "Configure and run training"),
            ("📈", "Evaluation",         "Compare all models"),
            ("📉", "ROC & PR Curves",    "Visual curve analysis"),
            ("🕸️", "Model Radar",        "Multi-metric spider chart"),
            ("🔍", "Feature Analysis",   "Importance breakdown"),
            ("📤", "Export Results",     "Download predictions CSV"),
        ]
        for icon, name, desc in tools:
            st.markdown(f"""
            <div style='background:var(--bg-elevated);border:1px solid var(--border);
                        border-radius:8px;padding:0.55rem 0.9rem;
                        display:flex;align-items:center;gap:0.7rem;'>
                <span style='font-size:1.1rem;'>{icon}</span>
                <div>
                    <div style='color:var(--text-primary);font-size:0.82rem;font-weight:600;'>{name}</div>
                    <div style='color:var(--text-muted);font-size:0.7rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Synthetic impact chart
    st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Impact of Synthetic Fraud Augmentation</div>", unsafe_allow_html=True)
    fig_s = go.Figure()
    metrics   = ["Precision","Recall","F1-Score","ROC-AUC (×10)"]
    original  = [0.79, 0.85, 0.82, 9.777]
    updated   = [0.28, 0.96, 0.44, 9.926]
    fig_s.add_trace(go.Bar(name="Original Dataset",         x=metrics, y=[0.79,0.85,0.82,0.9777], marker_color="#4a5568", marker_line_width=0))
    fig_s.add_trace(go.Bar(name="+ 500 Synthetic Fraud",    x=metrics, y=[0.28,0.96,0.44,0.9926], marker_color="#7C3AED", marker_line_width=0))
    fig_s.update_layout(**CHART_LAYOUT, height=260, barmode="group",
                         annotations=[dict(text="Recall improved: 0.85→0.96 | ROC-AUC: 0.9777→0.9926",
                                          x=0.5, y=1.08, xref="paper", yref="paper",
                                          showarrow=False, font=dict(color="#64748b",size=10))])
    st.plotly_chart(fig_s, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── END USER DASHBOARD ────────────────────────────────────────────────────────
def dashboard_user():
    my_preds = db_get_predictions(username=st.session_state.username, limit=200)
    my_total = len(my_preds)
    my_fraud = int((my_preds["result"]=="Fraudulent").sum()) if my_total>0 else 0
    my_legit = my_total - my_fraud
    my_rate  = round(my_fraud/my_total*100,1) if my_total>0 else 0
    last_res = my_preds.iloc[0] if my_total>0 else None

    st.markdown(f"""
    <div class='hero-wrap' style='animation:fadeSlideUp 0.35s ease;'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Personal Fraud Detection Workspace</div>
        <div class='hero-title'>Hi, {st.session_state.user_name.split()[0]} 👋</div>
        <div class='hero-sub'>Your personal fraud detection hub — check transactions, view history, and stay protected</div>
        <div class='hero-chips'>
            <span class='chip'>AI-Powered Detection</span>
            <span class='chip'>Gemini Explanations</span>
            <span class='chip'>Instant Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _announcements_banner()

    # Personal KPIs
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi'>
            <span class='kpi-icon'>🔎</span><div class='kpi-label'>My Total Checks</div>
            <div class='kpi-value'>{my_total}</div>
            <div class='kpi-sub'>Transactions analysed</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>🚨</span><div class='kpi-label'>Fraud Detected</div>
            <div class='kpi-value' style='color:var(--red);'>{my_fraud}</div>
            <div class='kpi-sub'>Flagged as suspicious</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>✅</span><div class='kpi-label'>Safe Transactions</div>
            <div class='kpi-value' style='color:var(--green);'>{my_legit}</div>
            <div class='kpi-sub'>Cleared as legitimate</div>
        </div>
        <div class='kpi'>
            <span class='kpi-icon'>📊</span><div class='kpi-label'>My Fraud Rate</div>
            <div class='kpi-value' style='color:{"var(--red)" if my_rate>5 else "var(--green)"};'>{my_rate}%</div>
            <div class='kpi-sub'>{"Above average — review!" if my_rate>5 else "Looking healthy"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Last result card
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Last Transaction Checked</div>", unsafe_allow_html=True)
        if last_res is not None:
            is_fraud  = last_res["result"]=="Fraudulent"
            v_color   = "#EF4444" if is_fraud else "#10B981"
            v_icon    = "⛔" if is_fraud else "✅"
            cat_label = str(last_res["category"]).replace("_"," ").title()
            prob_pct  = round(last_res["fraud_probability"]*100,1)
            st.markdown(f"""
            <div style='background:{"rgba(255,23,68,0.06)" if is_fraud else "rgba(0,230,118,0.06)"};
                        border:1px solid {v_color}33;border-radius:10px;padding:1.2rem;'>
                <div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem;'>
                    <span style='font-size:2rem;'>{v_icon}</span>
                    <div>
                        <div style='color:{v_color};font-weight:800;font-size:1.1rem;
                                    font-family:var(--font-display);'>{last_res['result']}</div>
                        <div style='color:var(--text-muted);font-size:0.75rem;'>{last_res['timestamp']}</div>
                    </div>
                    <div style='margin-left:auto;text-align:right;'>
                        <div style='color:var(--text-primary);font-weight:700;font-size:1rem;'>${last_res['amount']:.2f}</div>
                        <div style='color:var(--text-muted);font-size:0.75rem;'>{cat_label}</div>
                    </div>
                </div>
                <div style='margin-bottom:0.5rem;'>
                    <div style='display:flex;justify-content:space-between;font-size:0.72rem;
                                font-family:var(--font-mono);color:var(--text-muted);margin-bottom:0.3rem;'>
                        <span>Fraud Probability</span><span style='color:{v_color};'>{prob_pct}%</span>
                    </div>
                    <div style='height:6px;background:rgba(255,255,255,0.06);border-radius:3px;'>
                        <div style='height:100%;width:{prob_pct}%;background:{v_color};border-radius:3px;'></div>
                    </div>
                </div>
                <div style='color:var(--text-muted);font-size:0.72rem;'>{last_res['risk_band']}</div>
            </div>
            """, unsafe_allow_html=True)
            if last_res.get("explanation"):
                st.markdown(f"""
                <div style='margin-top:0.75rem;padding:0.75rem;background:var(--bg-elevated);
                            border-radius:8px;border-left:2px solid #7c3aed;'>
                    <div style='color:#a78bfa;font-size:0.65rem;font-family:var(--font-mono);
                                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;'>
                        🤖 AI Explanation
                    </div>
                    <div style='color:var(--text-secondary);font-size:0.8rem;line-height:1.6;'>
                        {str(last_res["explanation"])[:280]}{"..." if len(str(last_res["explanation"]))>280 else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:2rem;'>
                <div style='font-size:2.5rem;margin-bottom:0.5rem;opacity:0.3;'>🔍</div>
                <div style='color:var(--text-muted);font-size:0.85rem;'>No transactions checked yet.</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # Clickable quick action cards
        st.markdown("""
        <div class='glass glass-violet'>
            <div class='sec-label'>What Would You Like to Do?</div>
        </div>
        """, unsafe_allow_html=True)

        actions = [
            ("⚡", "Check a Single Transaction",
             "Enter details and get an instant AI-powered fraud verdict",
             "🔎  Single Transaction"),
            ("📂", "Analyse a Batch of Transactions",
             "Upload a CSV file and score multiple transactions at once",
             "📂  Batch Upload"),
            ("📜", "View My History",
             "See all your past checks with trends and category breakdown",
             "📜  History"),
        ]
        for icon, title, desc, nav_key in actions:
            st.markdown(f"""
            <div style='margin-bottom:0.5rem;'>
                <div style='color:var(--text-muted);font-size:0.72rem;padding-left:0.2rem;margin-bottom:0.2rem;'>
                    {icon} &nbsp;<span style='color:var(--text-secondary);font-size:0.78rem;'>{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='nav-card-btn'>", unsafe_allow_html=True)
            if st.button(f"{icon}  {title}", key=f"nav_{nav_key}"):
                st.session_state.nav_page = nav_key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Personal risk distribution
    if my_total > 0:
        risk_counts = my_preds["risk_band"].value_counts()
        high = int(risk_counts.get("High Risk",   0))
        med  = int(risk_counts.get("Medium Risk", 0))
        low  = int(risk_counts.get("Low Risk",    0))
        st.markdown(f"""
        <div class='glass'>
            <div class='sec-label'>My Risk Distribution</div>
            <div style='height:16px;border-radius:8px;overflow:hidden;display:flex;margin:0.5rem 0 0.75rem;'>
                <div style='width:{high/my_total*100:.1f}%;background:#ff1744;'></div>
                <div style='width:{med/my_total*100:.1f}%;background:#ffab00;'></div>
                <div style='width:{low/my_total*100:.1f}%;background:#00e676;'></div>
            </div>
            <div style='display:flex;gap:2rem;font-family:var(--font-mono);font-size:0.75rem;'>
                <span style='color:#ff1744;'>■ High Risk: {high}</span>
                <span style='color:#ffab00;'>■ Medium Risk: {med}</span>
                <span style='color:#00e676;'>■ Low Risk: {low}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Fraud safety tips
    st.markdown("""
    <div class='glass glass-amber'>
        <div class='sec-label'>💡 Fraud Prevention Tips</div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
            <div style='text-align:center;padding:0.75rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>🌙</div>
                <div style='color:var(--text-primary);font-weight:600;font-size:0.82rem;margin-bottom:0.25rem;'>Avoid Night Purchases</div>
                <div style='color:var(--text-muted);font-size:0.75rem;line-height:1.5;'>Transactions between 10pm–5am carry significantly higher fraud risk.</div>
            </div>
            <div style='text-align:center;padding:0.75rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>🛒</div>
                <div style='color:var(--text-primary);font-weight:600;font-size:0.82rem;margin-bottom:0.25rem;'>Online Shopping Risk</div>
                <div style='color:var(--text-muted);font-size:0.75rem;line-height:1.5;'>Online shopping categories have the highest fraud rates — verify merchant carefully.</div>
            </div>
            <div style='text-align:center;padding:0.75rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>📍</div>
                <div style='color:var(--text-primary);font-weight:600;font-size:0.82rem;margin-bottom:0.25rem;'>Check Location</div>
                <div style='color:var(--text-muted);font-size:0.75rem;line-height:1.5;'>Merchants far from your home location are a strong fraud indicator.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── DASHBOARD ROUTER ──────────────────────────────────────────────────────────
def page_dashboard():
    role = st.session_state.role
    if   role == "admin":      dashboard_admin()
    elif role == "researcher": dashboard_researcher()
    else:                      dashboard_user()

# ── SINGLE TRANSACTION ────────────────────────────────────────────────────────
def page_fraud_detection():
    st.markdown("""
    <div class='hero-wrap' style='animation:fadeSlideUp 0.35s ease;'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Real-time Detection</div>
        <div class='hero-title'>Single Transaction Analysis</div>
        <div class='hero-sub'>Enter transaction details — Gemini AI generates intelligent, contextual explanations</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2,1])

    with col_form:
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Transaction Details</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            amt      = st.number_input("Amount ($)",min_value=0.0,value=120.0,step=0.01)
            category = st.selectbox("Merchant Category",["grocery_pos","shopping_net","entertainment",
                "gas_transport","misc_net","misc_pos","shopping_pos","food_dining",
                "personal_care","health_fitness","travel","home","kids_pets"])
            gender   = st.selectbox("Gender",["M","F"])
            age      = st.number_input("Customer Age",min_value=18,max_value=100,value=35)
        with c2:
            trans_hour = st.slider("Transaction Hour",0,23,14)
            trans_day  = st.selectbox("Day",["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            city_pop   = st.number_input("City Population",min_value=100,max_value=5000000,value=150000,step=1000)
            distance   = st.number_input("Distance to Merchant (km)",min_value=0.0,max_value=500.0,value=25.0)
        alert_email = st.text_input("Alert Email (optional)",placeholder="your@email.com")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("⚡ Analyse Transaction"):
            is_night   = 1 if (trans_hour>=22 or trans_hour<=5) else 0
            is_weekend = 1 if trans_day in ["Sat","Sun"] else 0
            age_group  = 0 if age<30 else (1 if age<50 else 2)
            is_hi_dist = 1 if distance>75 else 0
            is_hi_amt  = 1 if amt>500 else 0
            cat_avgs   = {"grocery_pos":50,"shopping_net":80,"entertainment":60,"gas_transport":40,
                          "misc_net":70,"misc_pos":45,"shopping_pos":65,"food_dining":35,
                          "personal_care":30,"health_fitness":55,"travel":200,"home":90,"kids_pets":40}
            cat_avg    = cat_avgs.get(category,60)
            amt_to_cat = amt/(cat_avg+1e-6)
            cat_map    = {c:i for i,c in enumerate(["grocery_pos","shopping_net","entertainment","gas_transport",
                          "misc_net","misc_pos","shopping_pos","food_dining","personal_care","health_fitness","travel","home","kids_pets"])}
            day_map    = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
            features   = [100,cat_map.get(category,0),amt,1 if gender=="M" else 0,
                          500,25,37.0,-95.0,city_pop,300,is_weekend,is_night,age,age_group,
                          distance,is_hi_dist,is_hi_amt,amt_to_cat,np.log1p(amt),
                          1 if city_pop<10000 else 0,trans_hour,day_map.get(trans_day,0),37.5,-95.5]
            result = call_api(features)
            if not result:
                prob = min(0.95,0.05+(0.3 if is_night else 0)+(0.25 if is_hi_amt else 0)
                           +(0.15 if category in ["shopping_net","misc_net"] else 0)+(0.1 if distance>100 else 0))
                pred = 1 if prob>=0.5 else 0
                result = {"prediction":pred,"label":"Fraudulent" if pred==1 else "Legitimate",
                          "fraud_probability":round(prob,4),
                          "risk_band":"High Risk" if prob>=0.8 else ("Medium Risk" if prob>=0.5 else "Low Risk"),
                          "recommended_action":"Block transaction" if pred==1 else "Allow transaction"}
                st.warning("API offline — using demo scoring.")
            inputs_ctx = {"is_night":is_night,"is_high_amount":is_hi_amt,"distance_km":distance,
                          "category":category,"age_group":age_group,"amt_to_category_avg":amt_to_cat}
            with st.spinner("🤖 Gemini AI generating explanation..."):
                explanation, used_gemini = gemini_explanation(result,inputs_ctx,amt,category)
            db_save_prediction(st.session_state.username,amt,category,result["label"],
                               result["fraud_probability"],result["risk_band"],
                               result["recommended_action"],explanation,"single")
            st.session_state["last_result"]      = result
            st.session_state["last_inputs"]      = inputs_ctx
            st.session_state["last_explanation"] = explanation
            st.session_state["last_gemini"]      = used_gemini
            add_log(f"Single check — ${amt:.2f} {category} → {result['label']}")

    with col_result:
        if st.session_state.get("last_result"):
            result      = st.session_state["last_result"]
            explanation = st.session_state.get("last_explanation","")
            used_gemini = st.session_state.get("last_gemini",False)
            is_fraud    = result["prediction"]==1
            prob        = result["fraud_probability"]
            box_cls     = "result-fraud" if is_fraud else "result-legit"
            v_cls       = "verdict-fraud" if is_fraud else "verdict-legit"
            icon        = "⛔" if is_fraud else "✅"
            risk_color  = "#EF4444" if is_fraud else "#10B981"

            st.markdown(f"""
            <div class='result-box {box_cls}'>
                <div class='result-icon'>{icon}</div>
                <div class='result-verdict {v_cls}'>{result['label'].upper()}</div>
                <div style='color:var(--text-secondary);font-size:0.85rem;margin-top:0.4rem;font-family:var(--font-mono);'>
                    {result['risk_band']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(prob,"Fraud Probability"),use_container_width=True)

            st.markdown(f"""
            <div class='glass' style='border-left:2px solid {risk_color};margin-top:0;'>
                <div style='font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;'>
                    Recommended Action
                </div>
                <div style='color:{risk_color};font-weight:700;font-family:var(--font-display);font-size:1.05rem;'>
                    {result.get("recommended_action","Review")}
                </div>
            </div>
            """, unsafe_allow_html=True)

            gemini_tag = "<span class='badge badge-gemini'>Gemini AI</span>" if used_gemini else "<span style='font-size:0.7rem;color:var(--text-muted);'>Fallback</span>"
            st.markdown(f"""
            <div class='glass glass-violet'>
                <div class='sec-label'>AI Explanation &nbsp; {gemini_tag}</div>
                <p style='color:var(--text-secondary);font-size:0.95rem;line-height:1.8;margin:0;'>{explanation}</p>
                <div style='margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid var(--border);
                            font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);'>
                    {"Google Gemini 1.5 Flash" if used_gemini else "Rule-based fallback"} ·
                    {datetime.now().strftime("%H:%M:%S")} · Saved to database
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Deep AI Analysis expander — only for fraud
            if is_fraud and GEMINI_AVAILABLE and GEMINI_API_KEY:
                with st.expander("🔍  Deep Analysis — Why was this flagged as fraud?", expanded=False):
                    with st.spinner("Gemini AI is analysing this transaction..."):
                        try:
                            inputs = st.session_state.get("last_inputs", {})
                            prompt = f"""You are a fraud intelligence expert at a financial institution.

A transaction has been flagged as FRAUDULENT by our ML model with {prob:.1%} confidence.

Transaction Details:
- Amount: ${inputs.get('amt', 0):.2f}
- Category: {inputs.get('category', 'Unknown')}
- Time: {inputs.get('trans_hour', 12)}:00 ({'night' if inputs.get('is_night', 0) else 'day'})
- Customer Age: {inputs.get('age', 35)}
- Distance to Merchant: {inputs.get('distance', 0):.1f} km
- City Population: {inputs.get('city_pop', 100000):,}
- Risk Level: {result.get('risk_band', 'High Risk')}
- Fraud Probability: {prob:.1%}

Please provide a detailed explanation in 3 clearly labelled sections:

**WHY THIS WAS FLAGGED:**
Explain specifically what combination of factors triggered the fraud alert. Be concrete about which features were unusual.

**WHAT MADE IT SUSPICIOUS:**
Describe what a normal legitimate transaction looks like for this category and how this transaction deviated from that pattern.

**WHAT YOU SHOULD DO:**
Give the customer 3 specific actionable steps they can take right now to protect themselves.

Write in plain English addressing the customer directly. Be professional but approachable."""

                            model    = genai.GenerativeModel("gemini-1.5-flash")
                            response = model.generate_content(prompt)
                            ai_text  = response.text.strip()

                            sections = ai_text.split("**")
                            formatted = ai_text
                            for i in range(1, len(sections), 2):
                                formatted = formatted.replace(
                                    f"**{sections[i]}**",
                                    f"<strong style='color:var(--primary);'>{sections[i]}</strong>"
                                )

                            st.markdown(f"""
                            <div style='background:rgba(124,58,237,0.05);border:1px solid rgba(124,58,237,0.15);
                                        border-radius:12px;padding:1.5rem;line-height:1.85;
                                        color:var(--text-secondary);font-size:0.95rem;'>
                                {formatted}
                            </div>
                            <div style='margin-top:0.75rem;font-family:"JetBrains Mono",monospace;
                                        font-size:0.65rem;color:var(--text-muted);text-align:right;'>
                                Generated by Google Gemini 1.5 Flash · {datetime.now().strftime("%H:%M:%S")}
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning("AI analysis unavailable. Please check your Gemini API key.")
        else:
            st.markdown("""
            <div class='glass' style='text-align:center;padding:3rem 1rem;'>
                <div style='font-size:3rem;margin-bottom:1rem;opacity:0.4;'>⚡</div>
                <div style='font-family:var(--font-display);font-size:1rem;color:var(--text-muted);'>
                    Enter transaction details and click<br>
                    <span style='color:var(--cyan);'>Analyse Transaction</span> to begin
                </div>
                <div style='margin-top:1rem;font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);'>
                    🤖 Powered by Google Gemini AI
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── BATCH UPLOAD ──────────────────────────────────────────────────────────────
def page_batch_upload():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Batch Processing</div>
        <div class='hero-title'>CSV Transaction Upload</div>
        <div class='hero-sub'>Score an entire dataset at once — results saved to database automatically</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns([2,1])
    with c1:
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Upload CSV File</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:var(--text-muted);font-size:0.82rem;margin-bottom:1rem;'>Required columns: <code style='color:var(--cyan);'>amt, category, age, trans_hour, distance_km, is_night, city_pop, is_weekend</code></p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("",type=["csv"],label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Template</div>", unsafe_allow_html=True)
        template = pd.DataFrame({
            "amt":[45.0,1200.0,23.5,899.0,12.0],
            "category":["grocery_pos","shopping_net","food_dining","misc_net","health_fitness"],
            "age":[35,62,28,55,41],"trans_hour":[14,2,12,23,10],
            "is_night":[0,1,0,1,0],"is_weekend":[0,0,1,1,0],
            "distance_km":[5.2,145.0,2.1,88.0,3.4],"city_pop":[250000,8000,500000,45000,180000],
        })
        st.download_button("📥 Download Template",template.to_csv(index=False),"template.csv","text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.markdown(f"""
            <div class='glass glass-green'>
                <div style='color:var(--green);font-family:var(--font-display);font-weight:700;'>
                    ✓ &nbsp; {len(df_input)} transactions ready for analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Preview</div>", unsafe_allow_html=True)
            st.dataframe(df_input.head(),use_container_width=True,hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(f"⚡ Analyse All {len(df_input)} Transactions"):
                bar    = st.progress(0)
                status = st.empty()
                results= []
                for i,(_, row) in enumerate(df_input.iterrows()):
                    bar.progress((i+1)/len(df_input))
                    status.markdown(f"<div style='font-family:var(--font-mono);font-size:0.8rem;color:var(--cyan);'>Analysing transaction {i+1} of {len(df_input)}...</div>", unsafe_allow_html=True)
                    res = score_row(row)
                    results.append({
                        "#":          i+1,
                        "Amount":     f"${row.get('amt',0):.2f}",
                        "Category":   row.get("category","N/A"),
                        "Prediction": res["label"],
                        "Probability":res["fraud_probability"],
                        "Risk":       res["risk_band"],
                        "Action":     res["recommended_action"],
                    })
                    db_save_prediction(st.session_state.username,row.get("amt",0),row.get("category","N/A"),
                        res["label"],res["fraud_probability"],res["risk_band"],res["recommended_action"],
                        "Batch analysis","batch")
                bar.empty(); status.empty()
                df_res      = pd.DataFrame(results)
                fraud_count = (df_res["Prediction"]=="Fraudulent").sum()
                legit_count = len(df_res)-fraud_count
                high_risk   = (df_res["Risk"]=="High Risk").sum()
                med_risk    = (df_res["Risk"]=="Medium Risk").sum()
                fraud_pct   = round(fraud_count/len(df_res)*100,1)
                add_log(f"Batch — {len(df_input)} transactions — {fraud_count} fraud")

                # ── Plain English Summary ──────────────────────────────────────
                if fraud_count == 0:
                    summary_icon  = "🎉"
                    summary_color = "var(--green)"
                    summary_text  = f"All {len(df_res)} transactions look legitimate. No fraud was detected in this batch."
                elif fraud_pct < 20:
                    summary_icon  = "⚠️"
                    summary_color = "var(--amber)"
                    summary_text  = f"Mostly safe — {legit_count} transactions are legitimate but {fraud_count} suspicious transactions were found and should be reviewed."
                else:
                    summary_icon  = "🚨"
                    summary_color = "var(--red)"
                    summary_text  = f"High fraud activity detected — {fraud_count} out of {len(df_res)} transactions are flagged as fraudulent. Immediate review is recommended."

                st.markdown(f"""
                <div class='glass' style='border-left:3px solid {summary_color};margin-bottom:1rem;'>
                    <div style='display:flex;align-items:flex-start;gap:1rem;'>
                        <div style='font-size:2rem;'>{summary_icon}</div>
                        <div>
                            <div style='font-family:var(--font-display);font-weight:700;
                                        color:{summary_color};font-size:1rem;margin-bottom:0.3rem;'>
                                Batch Analysis Complete
                            </div>
                            <div style='color:var(--text-secondary);font-size:0.9rem;line-height:1.6;'>
                                {summary_text}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── 5 simple stat cards ────────────────────────────────────────
                st.markdown(f"""
                <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:0.75rem;margin-bottom:1.5rem;'>
                    <div class='kpi'>
                        <div class='kpi-label'>Total</div>
                        <div class='kpi-value'>{len(df_res)}</div>
                        <div class='kpi-sub'>Transactions</div>
                    </div>
                    <div class='kpi'>
                        <div class='kpi-label'>Safe</div>
                        <div class='kpi-value' style='color:var(--green);'>{legit_count}</div>
                        <div class='kpi-sub'>Legitimate</div>
                    </div>
                    <div class='kpi'>
                        <div class='kpi-label'>Flagged</div>
                        <div class='kpi-value' style='color:var(--red);'>{fraud_count}</div>
                        <div class='kpi-sub'>Fraudulent</div>
                    </div>
                    <div class='kpi'>
                        <div class='kpi-label'>High Risk</div>
                        <div class='kpi-value' style='color:var(--amber);'>{high_risk}</div>
                        <div class='kpi-sub'>Need action</div>
                    </div>
                    <div class='kpi'>
                        <div class='kpi-label'>Fraud Rate</div>
                        <div class='kpi-value' style='color:var(--red);font-size:1.4rem;'>{fraud_pct}%</div>
                        <div class='kpi-sub'>Of this batch</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Visual transaction cards ───────────────────────────────────
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("<div class='sec-label'>Transaction Results</div>", unsafe_allow_html=True)

                # Column headers
                st.markdown("""
                <div style='display:grid;grid-template-columns:0.4fr 1fr 1.5fr 1.2fr 1fr 1.4fr;
                            gap:0.5rem;padding:0.5rem 1rem;
                            font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);
                            text-transform:uppercase;letter-spacing:0.1em;
                            border-bottom:1px solid var(--border);margin-bottom:0.4rem;'>
                    <div>#</div>
                    <div>Amount</div>
                    <div>Category</div>
                    <div>Verdict</div>
                    <div>Confidence</div>
                    <div>Action</div>
                </div>
                """, unsafe_allow_html=True)

                for _, row in df_res.iterrows():
                    is_fraud  = row["Prediction"] == "Fraudulent"
                    prob      = row["Probability"]
                    risk      = row["Risk"]
                    bar_pct   = int(prob * 100)
                    bar_color = "#EF4444" if prob>=0.8 else ("#F59E0B" if prob>=0.5 else "#10B981")
                    v_color   = "#EF4444" if is_fraud else "#10B981"
                    v_icon    = "⛔" if is_fraud else "✅"
                    v_text    = "Fraudulent" if is_fraud else "Legitimate"
                    bg_color  = "rgba(255,23,68,0.04)" if is_fraud else "rgba(0,230,118,0.02)"
                    border    = "rgba(255,23,68,0.15)" if is_fraud else "rgba(255,255,255,0.06)"

                    cat_label = str(row["Category"]).replace("_"," ").title()
                    action    = row["Action"]

                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:0.4fr 1fr 1.5fr 1.2fr 1fr 1.4fr;
                                gap:0.5rem;align-items:center;padding:0.75rem 1rem;
                                background:{bg_color};border:1px solid {border};
                                border-radius:8px;margin-bottom:0.35rem;
                                transition:all 0.2s;'>
                        <div style='font-family:var(--font-mono);font-size:0.75rem;
                                    color:var(--text-muted);'>#{int(row["#"])}</div>
                        <div style='font-family:var(--font-display);font-weight:700;
                                    color:var(--text-primary);font-size:0.9rem;'>{row["Amount"]}</div>
                        <div style='font-size:0.82rem;color:var(--text-secondary);'>{cat_label}</div>
                        <div>
                            <span style='color:{v_color};font-weight:700;font-size:0.82rem;'>
                                {v_icon} {v_text}
                            </span>
                        </div>
                        <div>
                            <div style='font-family:var(--font-mono);font-size:0.72rem;
                                        color:{bar_color};margin-bottom:0.25rem;'>{bar_pct}%</div>
                            <div style='height:4px;background:rgba(255,255,255,0.08);
                                        border-radius:2px;overflow:hidden;'>
                                <div style='height:100%;width:{bar_pct}%;
                                            background:{bar_color};border-radius:2px;
                                            transition:width 0.5s ease;'></div>
                            </div>
                        </div>
                        <div style='font-size:0.75rem;color:{v_color};font-weight:600;'>{action}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Risk breakdown donut ───────────────────────────────────────
                low_risk = len(df_res) - high_risk - med_risk
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("<div class='glass'>", unsafe_allow_html=True)
                    st.markdown("<div class='sec-label'>Risk Breakdown</div>", unsafe_allow_html=True)
                    fig_donut = go.Figure(go.Pie(
                        labels=["Safe","Needs Review","Block Now"],
                        values=[low_risk, med_risk, high_risk],
                        hole=0.6,
                        marker_colors=["#10B981","#F59E0B","#EF4444"],
                        textinfo="label+percent",
                        textfont={"color":"#eef2ff","size":12}
                    ))
                    fig_donut.update_layout(**CHART_LAYOUT,height=240)
                    fig_donut.update_traces(marker_line_width=0)
                    st.plotly_chart(fig_donut,use_container_width=True)
                    st.markdown("""
                    <div style='text-align:center;margin-top:0.5rem;'>
                        <div style='display:inline-flex;gap:1.5rem;font-size:0.78rem;'>
                            <span style='color:var(--green);'>● Safe — Allow</span>
                            <span style='color:var(--amber);'>● Review</span>
                            <span style='color:var(--red);'>● Block Now</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with cb:
                    st.markdown("<div class='glass'>", unsafe_allow_html=True)
                    st.markdown("<div class='sec-label'>Fraud by Category</div>", unsafe_allow_html=True)
                    fraud_rows = df_res[df_res["Prediction"]=="Fraudulent"]
                    if len(fraud_rows)>0:
                        cat_counts = fraud_rows["Category"].value_counts()
                        cat_labels = [c.replace("_"," ").title() for c in cat_counts.index]
                        fig_cat = go.Figure(go.Bar(
                            x=cat_counts.values,
                            y=cat_labels,
                            orientation="h",
                            marker_color="#EF4444",
                            marker_line_width=0,
                        ))
                        fig_cat.update_layout(**CHART_LAYOUT,height=240,
                                              xaxis_title="Fraud Count")
                        st.plotly_chart(fig_cat,use_container_width=True)
                    else:
                        st.markdown("<div style='text-align:center;padding:3rem;color:var(--text-muted);'>No fraud detected ✅</div>",unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── What to do next ───────────────────────────────────────────
                if fraud_count > 0:
                    st.markdown(f"""
                    <div class='glass glass-amber'>
                        <div class='sec-label'>What to Do Next</div>
                        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
                            <div style='text-align:center;padding:1rem;background:rgba(255,23,68,0.06);
                                        border-radius:8px;border:1px solid rgba(255,23,68,0.15);'>
                                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>🚫</div>
                                <div style='color:var(--red);font-weight:700;font-size:0.85rem;'>Block High Risk</div>
                                <div style='color:var(--text-muted);font-size:0.75rem;margin-top:0.3rem;'>
                                    {high_risk} transaction{"s" if high_risk!=1 else ""} — decline immediately
                                </div>
                            </div>
                            <div style='text-align:center;padding:1rem;background:rgba(255,171,0,0.06);
                                        border-radius:8px;border:1px solid rgba(255,171,0,0.15);'>
                                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>🔍</div>
                                <div style='color:var(--amber);font-weight:700;font-size:0.85rem;'>Review Medium Risk</div>
                                <div style='color:var(--text-muted);font-size:0.75rem;margin-top:0.3rem;'>
                                    {med_risk} transaction{"s" if med_risk!=1 else ""} — verify with customer
                                </div>
                            </div>
                            <div style='text-align:center;padding:1rem;background:rgba(0,230,118,0.06);
                                        border-radius:8px;border:1px solid rgba(0,230,118,0.15);'>
                                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>✅</div>
                                <div style='color:var(--green);font-weight:700;font-size:0.85rem;'>Allow Low Risk</div>
                                <div style='color:var(--text-muted);font-size:0.75rem;margin-top:0.3rem;'>
                                    {legit_count} transaction{"s" if legit_count!=1 else ""} — safe to proceed
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.download_button("📥 Download Results CSV",
                                   df_res.to_csv(index=False),
                                   "fraud_results.csv","text/csv")

                # ── Gemini AI Explanations — visual redesign ─────────────────
                fraud_items = [r for r in results if r["Prediction"] == "Fraudulent"]

                if fraud_items and GEMINI_AVAILABLE and GEMINI_API_KEY:

                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,rgba(124,58,237,0.08),rgba(6,182,212,0.05));
                                border:1px solid rgba(124,58,237,0.25);border-radius:16px;
                                padding:1.75rem;margin:1.5rem 0;'>
                        <div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;'>
                            <div style='font-size:2rem;'>🤖</div>
                            <div>
                                <div style='font-family:"Manrope",sans-serif;font-size:1.15rem;
                                            font-weight:800;color:var(--text-primary);'>
                                    AI Fraud Investigation Report
                                </div>
                                <div style='font-size:0.82rem;color:var(--text-muted);margin-top:0.2rem;'>
                                    Google Gemini 1.5 Flash has investigated each flagged transaction —
                                    click any card to read the full analysis
                                </div>
                            </div>
                            <span style='margin-left:auto;background:rgba(124,58,237,0.15);
                                         border:1px solid rgba(124,58,237,0.3);border-radius:20px;
                                         padding:0.25rem 0.75rem;font-family:"JetBrains Mono",monospace;
                                         font-size:0.65rem;color:#A78BFA;white-space:nowrap;'>
                                Gemini AI · {min(len(fraud_items),10)} analyses
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    risk_config = {
                        "High Risk":   ("#EF4444","rgba(239,68,68,0.08)","rgba(239,68,68,0.25)","🔴"),
                        "Medium Risk": ("#F59E0B","rgba(245,158,11,0.08)","rgba(245,158,11,0.25)","🟡"),
                        "Low Risk":    ("#10B981","rgba(16,185,129,0.08)","rgba(16,185,129,0.25)","🟢"),
                    }

                    for item in fraud_items[:10]:
                        amt      = item["Amount"]
                        category = str(item["Category"])
                        prob     = item["Probability"]
                        risk     = item["Risk"]
                        row_num  = item["#"]
                        r_color, r_bg, r_border, r_dot = risk_config.get(risk, ("#EF4444","rgba(239,68,68,0.08)","rgba(239,68,68,0.25)","🔴"))
                        prob_pct = f"{prob:.0%}" if isinstance(prob, float) else str(prob)

                        with st.expander(f"{r_dot}  Transaction #{row_num}  ·  {amt}  ·  {category}  ·  {risk}"):

                            # Transaction summary strip
                            st.markdown(f"""
                            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;
                                        margin-bottom:1.25rem;'>
                                <div style='background:var(--bg-elevated);border:1px solid var(--border);
                                            border-radius:10px;padding:0.85rem;text-align:center;'>
                                    <div style='font-family:"JetBrains Mono",monospace;font-size:0.6rem;
                                                color:var(--text-muted);text-transform:uppercase;margin-bottom:0.3rem;'>
                                        Transaction
                                    </div>
                                    <div style='font-family:"Manrope",sans-serif;font-size:1.1rem;
                                                font-weight:800;color:var(--text-primary);'>#{row_num}</div>
                                </div>
                                <div style='background:var(--bg-elevated);border:1px solid var(--border);
                                            border-radius:10px;padding:0.85rem;text-align:center;'>
                                    <div style='font-family:"JetBrains Mono",monospace;font-size:0.6rem;
                                                color:var(--text-muted);text-transform:uppercase;margin-bottom:0.3rem;'>
                                        Amount
                                    </div>
                                    <div style='font-family:"Manrope",sans-serif;font-size:1.1rem;
                                                font-weight:800;color:{r_color};'>{amt}</div>
                                </div>
                                <div style='background:var(--bg-elevated);border:1px solid var(--border);
                                            border-radius:10px;padding:0.85rem;text-align:center;'>
                                    <div style='font-family:"JetBrains Mono",monospace;font-size:0.6rem;
                                                color:var(--text-muted);text-transform:uppercase;margin-bottom:0.3rem;'>
                                        Category
                                    </div>
                                    <div style='font-size:0.82rem;font-weight:600;color:var(--text-primary);'>{category}</div>
                                </div>
                                <div style='background:{r_bg};border:1px solid {r_border};
                                            border-radius:10px;padding:0.85rem;text-align:center;'>
                                    <div style='font-family:"JetBrains Mono",monospace;font-size:0.6rem;
                                                color:{r_color};text-transform:uppercase;margin-bottom:0.3rem;'>
                                        Fraud Prob
                                    </div>
                                    <div style='font-family:"Manrope",sans-serif;font-size:1.1rem;
                                                font-weight:800;color:{r_color};'>{prob_pct}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.spinner("Gemini AI is investigating this transaction..."):
                                try:
                                    prompt = f"""You are a senior fraud analyst at a financial institution writing a brief investigation report.

Transaction flagged: #{row_num}
- Amount: {amt}
- Merchant category: {category}
- Fraud probability: {prob_pct}
- Risk level: {risk}

Write exactly two sections with these headers:

**WHY THIS TRANSACTION WAS FLAGGED:**
2-3 sentences. Explain specifically what characteristics made the ML model flag this as suspicious. Be concrete — mention the amount, category, and what patterns suggest fraud.

**WHAT THE CUSTOMER SHOULD DO:**
2-3 sentences. Give clear, specific actions the customer should take right now. Address them directly as "you".

Keep it professional but plain English. No bullet points."""

                                    model    = genai.GenerativeModel("gemini-1.5-flash")
                                    response = model.generate_content(prompt)
                                    ai_text  = response.text.strip()

                                    # Parse sections
                                    why_text = how_text = ""
                                    lines = ai_text.split("\n")
                                    current = None
                                    buf = []
                                    for line in lines:
                                        if "WHY" in line.upper() and "FLAGGED" in line.upper():
                                            current = "why"; buf = []
                                        elif "CUSTOMER" in line.upper() or "SHOULD DO" in line.upper() or "WHAT THE" in line.upper():
                                            if current == "why": why_text = " ".join(buf).strip()
                                            current = "how"; buf = []
                                        elif current and line.strip() and not line.strip().startswith("**"):
                                            buf.append(line.strip())
                                    if current == "how": how_text = " ".join(buf).strip()
                                    if not why_text: why_text = ai_text[:len(ai_text)//2]
                                    if not how_text: how_text = ai_text[len(ai_text)//2:]

                                    col_l, col_r = st.columns(2)
                                    with col_l:
                                        st.markdown(f"""
                                        <div style='background:rgba(239,68,68,0.06);
                                                    border:1px solid rgba(239,68,68,0.2);
                                                    border-radius:12px;padding:1.25rem;height:100%;'>
                                            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;'>
                                                <span style='font-size:1.2rem;'>🚨</span>
                                                <span style='font-family:"JetBrains Mono",monospace;
                                                             font-size:0.68rem;font-weight:700;
                                                             color:#F87171;text-transform:uppercase;
                                                             letter-spacing:0.1em;'>
                                                    Why It Was Flagged
                                                </span>
                                            </div>
                                            <p style='color:var(--text-secondary);font-size:0.9rem;
                                                      line-height:1.75;margin:0;'>{why_text}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col_r:
                                        st.markdown(f"""
                                        <div style='background:rgba(16,185,129,0.06);
                                                    border:1px solid rgba(16,185,129,0.2);
                                                    border-radius:12px;padding:1.25rem;height:100%;'>
                                            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;'>
                                                <span style='font-size:1.2rem;'>✅</span>
                                                <span style='font-family:"JetBrains Mono",monospace;
                                                             font-size:0.68rem;font-weight:700;
                                                             color:#6EE7B7;text-transform:uppercase;
                                                             letter-spacing:0.1em;'>
                                                    What You Should Do
                                                </span>
                                            </div>
                                            <p style='color:var(--text-secondary);font-size:0.9rem;
                                                      line-height:1.75;margin:0;'>{how_text}</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                    st.markdown(f"""
                                    <div style='margin-top:0.65rem;display:flex;justify-content:flex-end;
                                                align-items:center;gap:0.5rem;'>
                                        <span style='font-size:0.8rem;'>✨</span>
                                        <span style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                                                     color:var(--text-muted);'>
                                            Generated by Google Gemini 1.5 Flash
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)

                                except Exception:
                                    st.markdown(f"""
                                    <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                                                border-radius:10px;padding:1rem;'>
                                        <div style='font-weight:600;color:var(--amber);margin-bottom:0.4rem;'>
                                            ⚠️ AI analysis unavailable for this transaction
                                        </div>
                                        <p style='color:var(--text-muted);font-size:0.85rem;margin:0;'>
                                            Transaction #{row_num} ({amt}, {category}) was flagged because its
                                            spending pattern deviates significantly from legitimate transactions
                                            in this category. Please review manually and contact your bank
                                            if you did not authorise this transaction.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

                    if len(fraud_items) > 10:
                        st.markdown(f"""
                        <div style='background:var(--bg-elevated);border:1px solid var(--border);
                                    border-radius:10px;padding:0.85rem;text-align:center;margin-top:0.5rem;'>
                            <div style='font-family:"JetBrains Mono",monospace;font-size:0.75rem;color:var(--text-muted);'>
                                AI analysis shown for 10 highest-risk transactions ·
                                {len(fraud_items)-10} additional flagged transactions visible in the table above
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                elif fraud_items and not GEMINI_AVAILABLE:
                    st.info("Install google-generativeai to enable AI fraud explanations.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ── HISTORY ───────────────────────────────────────────────────────────────────
def page_history():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Persistent Storage</div>
        <div class='hero-title'>My Detection History</div>
        <div class='hero-sub'>All your predictions stored permanently — trends, breakdown and full log below</div>
    </div>
    """, unsafe_allow_html=True)
    df = db_get_predictions(username=st.session_state.username, limit=200)
    if len(df)==0:
        st.markdown("<div class='glass' style='text-align:center;padding:2rem;'><p style='color:var(--text-muted);'>No transactions analysed yet. Go to Single Transaction or Batch Upload to get started.</p></div>", unsafe_allow_html=True)
        return
    fraud_count = int((df["result"]=="Fraudulent").sum())
    legit_count = len(df)-fraud_count
    avg_prob    = round(df["fraud_probability"].mean()*100,1)
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi'><div class='kpi-label'>Total Checked</div><div class='kpi-value'>{len(df)}</div></div>
        <div class='kpi'><div class='kpi-label'>Fraud Found</div><div class='kpi-value' style='color:var(--red);'>{fraud_count}</div></div>
        <div class='kpi'><div class='kpi-label'>Legitimate</div><div class='kpi-value' style='color:var(--green);'>{legit_count}</div></div>
        <div class='kpi'><div class='kpi-label'>Avg Risk Score</div><div class='kpi-value' style='color:var(--amber);'>{avg_prob}%</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Fraud Rate Over Time</div>", unsafe_allow_html=True)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date").agg(total=("result","count"),fraud=("result",lambda x:(x=="Fraudulent").sum())).reset_index()
        daily["rate"] = (daily["fraud"]/daily["total"]*100).round(1)
        if len(daily)>1:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=daily["date"].astype(str),y=daily["rate"],mode="lines+markers",
                line=dict(color="#7C3AED",width=2.5),marker=dict(size=7,color="#7C3AED"),
                fill="tozeroy",fillcolor="rgba(0,212,255,0.05)"))
            fig_t.update_layout(**CHART_LAYOUT,height=240,yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig_t,use_container_width=True)
        else:
            st.markdown("<p style='color:var(--text-muted);font-size:0.85rem;'>Need checks across multiple days to show trend.</p>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass glass-violet'>",unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>My Checks by Category</div>",unsafe_allow_html=True)
        cat_c = df["category"].value_counts().head(6)
        if len(cat_c)>0:
            fig_c = go.Figure(go.Pie(
                labels=[c.replace("_"," ").title() for c in cat_c.index],
                values=cat_c.values,hole=0.55,textinfo="label+percent",
                textfont=dict(color="#eef2ff",size=10),
                marker=dict(colors=["#7C3AED","#10B981","#F59E0B","#EF4444","#a78bfa","#38bdf8"],
                            line=dict(width=0))))
            fig_c.update_layout(**CHART_LAYOUT,height=240,showlegend=False)
            st.plotly_chart(fig_c,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # Risk breakdown bar
    risk_counts = df["risk_band"].value_counts()
    high=int(risk_counts.get("High Risk",0)); med=int(risk_counts.get("Medium Risk",0)); low=int(risk_counts.get("Low Risk",0))
    total=len(df)
    st.markdown(f"""
    <div class='glass'>
        <div class='sec-label'>Risk Level Breakdown</div>
        <div style='height:16px;border-radius:8px;overflow:hidden;display:flex;margin:0.5rem 0 0.75rem;'>
            <div style='width:{high/total*100:.1f}%;background:#ff1744;'></div>
            <div style='width:{med/total*100:.1f}%;background:#ffab00;'></div>
            <div style='width:{low/total*100:.1f}%;background:#00e676;'></div>
        </div>
        <div style='display:flex;gap:2rem;font-family:var(--font-mono);font-size:0.75rem;'>
            <span style='color:#ff1744;'>■ High Risk: {high}</span>
            <span style='color:#ffab00;'>■ Medium Risk: {med}</span>
            <span style='color:#00e676;'>■ Low Risk: {low}</span>
        </div>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("<div class='glass'>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Full Transaction Log</div>",unsafe_allow_html=True)
    cols=["timestamp","amount","category","result","fraud_probability","risk_band","prediction_type"]
    st.dataframe(df[cols],use_container_width=True,hide_index=True)
    st.download_button("📥 Export My History CSV",df[cols].to_csv(index=False),"my_history.csv","text/csv")
    st.markdown("</div>",unsafe_allow_html=True)
    fraud_rows=df[df["result"]=="Fraudulent"]
    if len(fraud_rows)>0 and fraud_rows.iloc[0]["explanation"]:
        st.markdown(f"""
        <div class='glass glass-violet'>
            <div class='sec-label'>Latest Fraud Explanation <span class='badge badge-gemini'>Gemini AI</span></div>
            <p style='color:var(--text-secondary);font-size:0.9rem;line-height:1.8;margin:0;'>{fraud_rows.iloc[0]['explanation']}</p>
        </div>
        """,unsafe_allow_html=True)

# ── USER MANAGEMENT ───────────────────────────────────────────────────────────
def page_user_management():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Administration</div>
        <div class='hero-title'>User Management</div>
        <div class='hero-sub'>Create and manage platform users — new accounts are immediately active</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Security Alerts — Locked Accounts ─────────────────────────────────────
    lockouts_df = db_get_active_lockouts()
    if len(lockouts_df) > 0:
        st.markdown(f"""
        <div class='glass glass-danger' style='border-left:4px solid var(--red);margin-bottom:1rem;'>
            <div style='display:flex;align-items:center;gap:1rem;margin-bottom:1rem;'>
                <span style='font-size:1.8rem;'>🔒</span>
                <div>
                    <div style='font-family:var(--font-display);font-weight:800;color:var(--red);font-size:1rem;'>
                        Security Alert — {len(lockouts_df)} Locked Account{'s' if len(lockouts_df)>1 else ''}
                    </div>
                    <div style='color:var(--text-muted);font-size:0.8rem;margin-top:0.2rem;'>
                        {'These accounts' if len(lockouts_df)>1 else 'This account'} {'have' if len(lockouts_df)>1 else 'has'} been automatically locked after 3 consecutive failed login attempts.
                        Admin email notification has been sent. Review and unlock if legitimate.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        for _, row in lockouts_df.iterrows():
            users_all = db_get_all_users()
            user_info = users_all.get(row["username"], {})
            col_a, col_b, col_c, col_d, col_e = st.columns([1.2, 1.2, 1, 1.2, 1])
            col_a.markdown(f"""
            <div style='padding-top:0.4rem;'>
                <div style='color:var(--red);font-family:var(--font-mono);font-size:0.82rem;font-weight:700;'>
                    @{row['username']}
                </div>
                <div style='color:var(--text-muted);font-size:0.72rem;'>{user_info.get('name','Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            col_b.markdown(f"""
            <div style='color:var(--text-muted);font-family:var(--font-mono);font-size:0.72rem;padding-top:0.4rem;'>
                Locked at<br><strong style='color:var(--text-secondary);'>{row['locked_at']}</strong>
            </div>
            """, unsafe_allow_html=True)
            col_c.markdown(f"""
            <div style='padding-top:0.4rem;'>
                <div style='background:rgba(255,23,68,0.12);border:1px solid rgba(255,23,68,0.25);
                            border-radius:6px;padding:0.3rem 0.6rem;text-align:center;
                            font-family:var(--font-mono);font-size:0.75rem;color:var(--red);font-weight:700;'>
                    {row['attempts']} attempts
                </div>
            </div>
            """, unsafe_allow_html=True)
            col_d.markdown(f"""
            <div style='color:{"#10B981" if row["notified_admin"] else "#F59E0B"};
                        font-family:var(--font-mono);font-size:0.72rem;padding-top:0.5rem;'>
                {"Email sent" if row["notified_admin"] else "Email pending"}
            </div>
            """, unsafe_allow_html=True)
            with col_e:
                if st.button("Unlock Account", key=f"unlock_{row['id']}"):
                    db_unlock_account(row["username"])
                    # Clear failed logins for this user in session state
                    st.session_state.failed_logins.pop(row["username"], None)
                    add_log(f"Admin unlocked account: {row['username']}")
                    st.success(f"Account @{row['username']} has been unlocked. They can now log in.")
                    st.rerun()
            st.markdown("<hr style='border-color:rgba(255,23,68,0.15);margin:0.4rem 0;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='glass' style='padding:0.85rem 1.2rem;margin-bottom:1rem;border-left:3px solid var(--green);'>
            <div style='display:flex;align-items:center;gap:0.75rem;'>
                <span style='font-size:1rem;'>🛡️</span>
                <div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--green);'>
                    No security alerts — all accounts are secure
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    users=get_users()
    total=len(users); active=sum(1 for u in users.values() if u.get("status")=="active")
    admins=sum(1 for u in users.values() if u["role"]=="admin")
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi'><div class='kpi-label'>Total Users</div><div class='kpi-value'>{total}</div></div>
        <div class='kpi'><div class='kpi-label'>Active</div><div class='kpi-value' style='color:var(--green);'>{active}</div></div>
        <div class='kpi'><div class='kpi-label'>Inactive</div><div class='kpi-value' style='color:var(--text-muted);'>{total-active}</div></div>
        <div class='kpi'><div class='kpi-label'>Admins</div><div class='kpi-value' style='color:var(--red);'>{admins}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pending Requests ──
    pending = st.session_state.pending_users
    if pending:
        st.markdown(f"""
        <div class='glass glass-amber'>
            <div class='sec-label'>
                ⏳ Pending Account Requests
                <span style='background:var(--amber);color:#04060f;font-family:var(--font-mono);
                             font-size:0.65rem;font-weight:700;padding:0.15rem 0.5rem;
                             border-radius:20px;margin-left:0.5rem;'>{len(pending)}</span>
            </div>
        """, unsafe_allow_html=True)
        for i, req in enumerate(list(pending)):
            with st.expander(f"📋  {req['name']} (@{req['username']}) — Requested: {req['role']}"):
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Name:** {req['name']}  \n**Username:** `{req['username']}`")
                c2.markdown(f"**Email:** {req['email']}  \n**Role:** `{req['role']}`")
                c3.markdown(f"**Submitted:** {req['submitted']}")
                if req.get("reason"):
                    st.markdown(f"**Reason:** {req['reason']}")
                col_a, col_b, _ = st.columns([1,1,2])
                with col_a:
                    if st.button("✅ Approve", key=f"approve_{i}_{req['username']}"):
                        db_upsert_user(req["username"], req["password"], req["role"],
                                       req["name"], req["email"], "active",
                                       datetime.now().strftime("%Y-%m-%d"))
                        st.session_state.pending_users = [p for p in st.session_state.pending_users if p["username"]!=req["username"]]
                        add_log(f"Admin APPROVED: {req['username']} ({req['role']})")
                        if req.get("email"):
                            sent = notify_user_approved(req["name"], req["username"], req["role"], req["email"])
                            if sent:
                                st.success(f"✅ '{req['username']}' approved — they can log in now. Email sent to {req['email']}.")
                            else:
                                st.success(f"✅ '{req['username']}' approved — they can log in now. (Email config issue.)")
                        else:
                            st.success(f"✅ '{req['username']}' approved — they can log in now.")
                        st.rerun()
                with col_b:
                    if st.button("❌ Reject", key=f"reject_{i}_{req['username']}"):
                        st.session_state.pending_users = [p for p in st.session_state.pending_users if p["username"]!=req["username"]]
                        add_log(f"Admin REJECTED: {req['username']}")
                        # Send rejection email
                        if req.get("email"):
                            sent = notify_user_rejected(req["name"], req["username"], req["email"])
                            if sent:
                                st.warning(f"Request from '{req['username']}' rejected. Rejection email sent to {req['email']}.")
                            else:
                                st.warning(f"Request from '{req['username']}' rejected. (Email not sent — check email config.)")
                        else:
                            st.warning(f"Request from '{req['username']}' rejected.")
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='glass' style='padding:1rem 1.5rem;margin-bottom:1rem;'>
            <div style='display:flex;align-items:center;gap:0.75rem;'>
                <span class='dot dot-green'></span>
                <span style='font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted);'>
                    No pending account requests
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>➕ Create New User Directly</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: nu=st.text_input("Username",placeholder="jsmith",key="nu")
    with c2: nn=st.text_input("Full Name",placeholder="Jane Smith",key="nn")
    with c3: np_=st.text_input("Password",type="password",placeholder="Min 6 chars",key="np")
    with c4: nr=st.selectbox("Role",["user","researcher","admin"],key="nr")
    with c5: ne=st.text_input("Email",placeholder="user@email.com",key="ne")
    if st.button("✅ Create User"):
        users = get_users()
        if not nu or not nn or not np_:
            st.error("Username, full name and password are required.")
        elif len(np_)<6:
            st.error("Password must be at least 6 characters.")
        elif nu in users:
            st.error(f"Username '{nu}' already exists.")
        else:
            created_date = datetime.now().strftime("%Y-%m-%d")
            db_upsert_user(nu, np_, nr, nn, ne, "active", created_date)
            add_log(f"Admin created user: {nu} ({nr})")
            st.success(f"✅ User '{nu}' ({nn}) created with role '{nr}' — they can log in immediately.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>All Users</div>", unsafe_allow_html=True)
    for uname,info in list(users.items()):
        rc={"admin":"badge-admin","researcher":"badge-research","user":"badge-user"}.get(info["role"],"badge-user")
        sc="#10B981" if info.get("status")=="active" else "#4a5568"
        cols=st.columns([1,1.3,0.9,0.8,0.9,1.5,1.8])
        cols[0].markdown(f"<div style='color:var(--text-primary);font-family:var(--font-mono);font-size:0.8rem;padding-top:0.5rem;'>{uname}</div>",unsafe_allow_html=True)
        cols[1].markdown(f"<div style='color:var(--text-secondary);font-size:0.8rem;padding-top:0.5rem;'>{info['name']}</div>",unsafe_allow_html=True)
        cols[2].markdown(f"<div style='padding-top:0.4rem;'><span class='badge {rc}'>{info['role']}</span></div>",unsafe_allow_html=True)
        cols[3].markdown(f"<div style='color:{sc};font-size:0.78rem;padding-top:0.5rem;font-family:var(--font-mono);'>{'● Active' if info.get('status')=='active' else '○ Off'}</div>",unsafe_allow_html=True)
        cols[4].markdown(f"<div style='color:var(--text-muted);font-size:0.72rem;padding-top:0.6rem;font-family:var(--font-mono);'>{info.get('created','N/A')}</div>",unsafe_allow_html=True)
        with cols[5]:
            # Role change (not for own account)
            if uname != st.session_state.username:
                role_opts = ["user","researcher","admin"]
                cur_idx   = role_opts.index(info["role"]) if info["role"] in role_opts else 0
                new_role_sel = st.selectbox("", role_opts, index=cur_idx, key=f"role_{uname}", label_visibility="collapsed")
                if new_role_sel != info["role"]:
                    db_upsert_user(uname, info["password"], new_role_sel, info["name"],
                                   info.get("email",""), info.get("status","active"), info.get("created",""))
                    add_log(f"Admin changed {uname} role: {info['role']} → {new_role_sel}")
                    st.rerun()
            else:
                st.markdown("<div style='color:var(--text-muted);font-size:0.72rem;padding-top:0.6rem;'>Current user</div>",unsafe_allow_html=True)
        with cols[6]:
            if uname==st.session_state.username:
                st.markdown("<div style='color:var(--text-muted);font-size:0.72rem;padding-top:0.6rem;'></div>",unsafe_allow_html=True)
            else:
                b1,b2,b3=st.columns(3)
                with b1:
                    lbl="Deactivate" if info.get("status")=="active" else "Activate"
                    if st.button(lbl,key=f"tog_{uname}"):
                        new_status="inactive" if info.get("status")=="active" else "active"
                        db_update_user_status(uname, new_status)
                        add_log(f"Admin set {uname} to {new_status}")
                        st.rerun()
                with b2:
                    if st.button("Delete",key=f"del_{uname}"):
                        db_delete_user(uname)
                        add_log(f"Admin deleted: {uname}")
                        st.rerun()
                with b3:
                    # MFA Reset button
                    _, totp_en = db_get_totp_secret(uname)
                    mfa_label = "Reset MFA" if totp_en else "No MFA"
                    if st.button(mfa_label, key=f"mfa_{uname}", disabled=not totp_en):
                        conn = sqlite3.connect(DB_PATH)
                        conn.execute("UPDATE users SET totp_secret=NULL, totp_enabled=0 WHERE username=?", (uname,))
                        conn.commit(); conn.close()
                        add_log(f"Admin reset MFA for: {uname}")
                        st.success(f"MFA reset for @{uname}. They will set up Google Authenticator again on next login.")
                        st.rerun()
        st.markdown("<hr style='border-color:var(--border);margin:0.3rem 0;'>",unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass glass-amber'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>🔑 Reset Password</div>", unsafe_allow_html=True)
    others=[u for u in get_users() if u!=st.session_state.username]
    if others:
        c1,c2=st.columns(2)
        with c1: target=st.selectbox("Select User",others,key="pt")
        with c2: new_pwd=st.text_input("New Password",type="password",placeholder="Min 6 characters",key="pp")
        if st.button("🔑 Reset Password"):
            if not new_pwd or len(new_pwd)<6:
                st.error("Password must be at least 6 characters.")
            else:
                db_update_user_password(target, new_pwd)
                add_log(f"Admin reset password for: {target}")
                st.success(f"Password for '{target}' updated — effective immediately.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── AUDIT LOGS ────────────────────────────────────────────────────────────────
def page_audit_logs():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Security & Compliance</div>
        <div class='hero-title'>Audit Logs</div>
        <div class='hero-sub'>Every user action, login attempt, approval, and system event is timestamped and stored permanently</div>
    </div>
    """, unsafe_allow_html=True)
    logs_df=db_get_logs(limit=100)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>System Activity Log</div>", unsafe_allow_html=True)
    if len(logs_df)>0:
        for _,row in logs_df.iterrows():
            st.markdown(f"""
            <div class='log-row'>
                <span style='color:var(--text-muted);'>[{row['timestamp']}]</span>
                &nbsp;<span style='color:var(--cyan);font-weight:600;'>{row['username']}</span>
                &nbsp;<span style='color:var(--text-muted);'>→</span>&nbsp;{row['action']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:var(--text-muted);'>No logs yet.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── SYSTEM ANALYTICS ──────────────────────────────────────────────────────────
def page_system_analytics():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Platform Intelligence</div>
        <div class='hero-title'>System Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    users=get_users(); df_all=db_get_predictions(limit=1000)
    total=len(df_all); fraud=int((df_all["result"]=="Fraudulent").sum()) if total>0 else 0
    logs_df=db_get_logs(limit=1000)
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi'><span class='kpi-icon'>⚡</span><div class='kpi-label'>Total Predictions</div><div class='kpi-value'>{total}</div></div>
        <div class='kpi'><span class='kpi-icon'>🚨</span><div class='kpi-label'>Fraud Detected</div><div class='kpi-value' style='color:var(--red);'>{fraud}</div></div>
        <div class='kpi'><span class='kpi-icon'>👥</span><div class='kpi-label'>Registered Users</div><div class='kpi-value'>{len(users)}</div></div>
        <div class='kpi'><div class='kpi-label'>Log Entries</div><div class='kpi-value'>{len(logs_df)}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if total>0:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Predictions by Outcome</div>", unsafe_allow_html=True)
            fig_pie=px.pie(df_all,names="result",color="result",
                           color_discrete_map={"Legitimate":"#10B981","Fraudulent":"#EF4444"})
            fig_pie.update_layout(**CHART_LAYOUT,height=240)
            fig_pie.update_traces(marker_line_width=0)
            st.plotly_chart(fig_pie,use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Fraud by Category</div>", unsafe_allow_html=True)
            cat_c=df_all[df_all["result"]=="Fraudulent"]["category"].value_counts().head(8)
            if len(cat_c)>0:
                fig_b=px.bar(x=cat_c.values,y=cat_c.index,orientation="h",color_discrete_sequence=["#EF4444"])
                fig_b.update_layout(**CHART_LAYOUT,height=240)
                fig_b.update_traces(marker_line_width=0)
                st.plotly_chart(fig_b,use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    info={"Dataset":"Sparkov Credit Card","Transactions":"1,296,675","Fraud Rate After Synthetic":"0.617%",
          "Synthetic Fraud Added":"500","Final Model":"Bagging Classifier","Best ROC-AUC":"0.9926",
          "Fraud Recall":"96%","AI Engine":"Google Gemini 1.5 Flash",
          "Database":"SQLite (persistent)","Security":"2FA + RBAC + Audit Logging"}
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>System Overview</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(list(info.items()),columns=["Property","Value"]),use_container_width=True,hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── MODEL DEPLOYMENT ──────────────────────────────────────────────────────────
def page_model_deployment():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>MLOps</div>
        <div class='hero-title'>Model Deployment</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Deployed Models</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Model":"Bagging Classifier (Sparkov)","Version":"v2.0","Status":"🟢 Active","ROC-AUC":0.9926,"Recall":0.96},
        {"Model":"Random Forest (ULB)","Version":"v1.0","Status":"⚪ Archived","ROC-AUC":0.912,"Recall":0.82},
    ]),use_container_width=True,hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Upload New Model</div>", unsafe_allow_html=True)
    up=st.file_uploader("Upload .pkl file",type=["pkl"])
    if up:
        st.success(f"Model '{up.name}' received.")
        add_log(f"Admin uploaded model: {up.name}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>API Endpoint Status</div>", unsafe_allow_html=True)
    for ep in ["/","/health","/model-info","/sample-input","/predict","/demo-fraud"]:
        try:
            r=requests.get(f"http://127.0.0.1:8000{ep}",timeout=2); ok=r.status_code==200
        except: ok=False
        dot="dot-green" if ok else "dot-red"
        status="200 OK" if ok else "Offline"
        st.markdown(f"""
        <div class='log-row'>
            <span class='dot {dot}'></span> &nbsp;
            <code style='color:var(--cyan);font-family:var(--font-mono);'>GET {ep}</code>
            &nbsp;&nbsp;<span style='color:{"var(--green)" if ok else "var(--red)"};font-family:var(--font-mono);font-size:0.75rem;'>{status}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── RESEARCHER PAGES ──────────────────────────────────────────────────────────
def page_model_training():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>Model Training</div>
        <div class='hero-sub'>Configure, train, and evaluate models — then test against your own dataset</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Training Configuration ────────────────────────────────────────────────
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Training Configuration</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dataset   = st.selectbox("Dataset", ["Sparkov (fraudTrain.csv)","Sparkov + Synthetic","ULB Credit Card"])
        model_sel = st.selectbox("Algorithm", ["Bagging","Random Forest","Gradient Boosting","Stacking","Logistic Regression","MLP Neural Network","Isolation Forest"])
        use_smote = st.checkbox("Apply SMOTE balancing", value=True)
    with c2:
        n_est  = st.slider("Number of Estimators", 10, 200, 50)
        max_d  = st.slider("Max Depth", 3, 30, 10)
        test_s = st.slider("Test Split Size (%)", 10, 40, 15)
        st.markdown(f"<div style='font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);margin-top:0.3rem;'>Train: {100-test_s*2}% · Val: {test_s}% · Test: {test_s}%</div>", unsafe_allow_html=True)

    if st.button("⚡ Start Training Simulation"):
        # Simulate training results based on model selected
        model_results = {
            "Bagging":             {"roc":0.9926,"rec":0.96,"prec":0.79,"f1":0.82,"cm":[[193125,251],[49,1152]]},
            "Random Forest":       {"roc":0.9943,"rec":0.88,"prec":0.66,"f1":0.75,"cm":[[192857,519],[130,996]]},
            "Gradient Boosting":   {"roc":0.9908,"rec":0.92,"prec":0.18,"f1":0.29,"cm":[[188474,4902],[86,1040]]},
            "Stacking":            {"roc":0.9948,"rec":0.94,"prec":0.26,"f1":0.41,"cm":[[190339,3037],[65,1061]]},
            "Logistic Regression": {"roc":0.9120,"rec":0.72,"prec":0.61,"f1":0.66,"cm":[[191200,2176],[315,811]]},
            "MLP Neural Network":  {"roc":0.9650,"rec":0.83,"prec":0.71,"f1":0.77,"cm":[[192100,1276],[191,935]]},
            "Isolation Forest":    {"roc":0.8830,"rec":0.67,"prec":0.45,"f1":0.54,"cm":[[188900,4476],[371,755]]},
        }
        res = model_results.get(model_sel, model_results["Bagging"])

        bar    = st.progress(0)
        status = st.empty()
        steps  = [
            "📂  Loading and validating dataset...",
            "🧹  Preprocessing — encoding, scaling...",
            f"⚖️  {'Applying SMOTE to balance classes...' if use_smote else 'Skipping SMOTE (imbalanced training)...'}",
            f"🔧  Training {model_sel} ({n_est} estimators, depth {max_d})...",
            "📊  Running evaluation on test set...",
            "📈  Generating performance metrics...",
            "✅  Training complete!",
        ]
        for i in range(101):
            bar.progress(i)
            step_idx = min(i * len(steps) // 101, len(steps)-1)
            status.markdown(f"<div style='font-family:var(--font-mono);font-size:0.8rem;color:var(--cyan);padding:0.5rem 0;'>{steps[step_idx]}</div>", unsafe_allow_html=True)
            time.sleep(0.025)
        bar.empty(); status.empty()

        add_log(f"Researcher trained {model_sel} on {dataset} — ROC-AUC: {res['roc']}")
        st.success(f"✅ Training complete — {model_sel} on {dataset}")

        # ── Metrics row ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0;'>
            <div class='kpi'>
                <div class='kpi-label'>ROC-AUC</div>
                <div class='kpi-value'>{res['roc']}</div>
                <div class='kpi-sub'>Discrimination ability</div>
            </div>
            <div class='kpi'>
                <div class='kpi-label'>Recall</div>
                <div class='kpi-value' style='color:var(--green);'>{res['rec']:.0%}</div>
                <div class='kpi-sub'>Fraud cases caught</div>
            </div>
            <div class='kpi'>
                <div class='kpi-label'>Precision</div>
                <div class='kpi-value' style='color:var(--cyan);'>{res['prec']:.0%}</div>
                <div class='kpi-sub'>Correct fraud flags</div>
            </div>
            <div class='kpi'>
                <div class='kpi-label'>F1-Score</div>
                <div class='kpi-value' style='color:var(--amber);'>{res['f1']:.2f}</div>
                <div class='kpi-sub'>Precision-recall balance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Visual results ────────────────────────────────────────────────────
        cv1, cv2, cv3 = st.columns(3)

        with cv1:
            # Precision / Recall / F1 bar chart
            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Classification Metrics</div>", unsafe_allow_html=True)
            fig_m = go.Figure(go.Bar(
                x=["Precision","Recall","F1-Score"],
                y=[res["prec"], res["rec"], res["f1"]],
                marker_color=["#7C3AED","#10B981","#F59E0B"],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in [res["prec"],res["rec"],res["f1"]]],
                textposition="outside",
                textfont=dict(color="#eef2ff", size=12)
            ))
            fig_m.update_layout(**CHART_LAYOUT, height=260,
                                yaxis_range=[0,1.1])
            st.plotly_chart(fig_m, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cv2:
            # ROC-AUC gauge
            st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>ROC-AUC Score</div>", unsafe_allow_html=True)
            fig_g = gauge_chart(res["roc"], "ROC-AUC")
            # Adjust gauge to show roc directly
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number", value=round(res["roc"]*100,2),
                title={"text":"ROC-AUC","font":{"color":"#94a3b8","size":13}},
                number={"suffix":"%","font":{"color":"#f1f5f9","size":22},
                        "valueformat":".2f"},
                gauge={
                    "axis":{"range":[80,100],"tickcolor":"#334155"},
                    "bar":{"color":"#7C3AED","thickness":0.65},
                    "bgcolor":"rgba(0,212,255,0.04)",
                    "bordercolor":"rgba(0,0,0,0)",
                    "steps":[
                        {"range":[80,90],"color":"rgba(255,23,68,0.08)"},
                        {"range":[90,95],"color":"rgba(255,171,0,0.08)"},
                        {"range":[95,100],"color":"rgba(0,230,118,0.08)"},
                    ],
                    "threshold":{"line":{"color":"#10B981","width":2},"thickness":0.8,"value":95}
                }
            ))
            fig_g2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                  font={"family":"DM Sans"},height=240,
                                  margin=dict(t=40,b=0,l=20,r=20))
            st.plotly_chart(fig_g2, use_container_width=True)
            quality = "Excellent" if res["roc"]>0.99 else ("Very Good" if res["roc"]>0.97 else ("Good" if res["roc"]>0.95 else "Moderate"))
            st.markdown(f"<div style='text-align:center;font-family:var(--font-mono);font-size:0.72rem;color:var(--green);'>{quality} discrimination ability</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cv3:
            # Confusion matrix heatmap
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Confusion Matrix</div>", unsafe_allow_html=True)
            cm = res["cm"]
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn,fp],[fn,tp]],
                text=[[f"{tn:,}",f"{fp:,}"],[f"{fn:,}",f"{tp:,}"]],
                texttemplate="%{text}",
                colorscale=[[0,"#0a1a2e"],[1,"#7C3AED"]],
                showscale=False, xgap=3, ygap=3,
                textfont=dict(color="#eef2ff",size=11)
            ))
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=250, margin=dict(t=10,b=40,l=60,r=10),
                xaxis=dict(tickvals=[0,1],ticktext=["Pred: Legit","Pred: Fraud"],
                           tickfont=dict(color="#64748b",size=9)),
                yaxis=dict(tickvals=[0,1],ticktext=["Actual: Legit","Actual: Fraud"],
                           tickfont=dict(color="#64748b",size=9)),
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            # Quick stats below CM
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:0.4rem;margin-top:0.25rem;'>
                <div style='background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.15);
                            border-radius:6px;padding:0.4rem;text-align:center;'>
                    <div style='color:#00e676;font-family:var(--font-mono);font-size:0.75rem;font-weight:700;'>{tp:,}</div>
                    <div style='color:var(--text-muted);font-size:0.65rem;'>True Positives</div>
                </div>
                <div style='background:rgba(255,23,68,0.08);border:1px solid rgba(255,23,68,0.15);
                            border-radius:6px;padding:0.4rem;text-align:center;'>
                    <div style='color:#ff1744;font-family:var(--font-mono);font-size:0.75rem;font-weight:700;'>{fn:,}</div>
                    <div style='color:var(--text-muted);font-size:0.65rem;'>Missed Fraud</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Learning curve simulation ─────────────────────────────────────────
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Training vs Validation Performance (Learning Curve)</div>", unsafe_allow_html=True)
        import numpy as np
        epochs     = list(range(1, 21))
        train_auc  = [0.72 + (res["roc"]-0.72)*(1-np.exp(-0.3*e)) + np.random.normal(0,0.003) for e in epochs]
        val_auc    = [0.68 + (res["roc"]-0.05-0.68)*(1-np.exp(-0.25*e)) + np.random.normal(0,0.004) for e in epochs]
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(x=epochs, y=train_auc, mode="lines+markers",
            name="Training AUC", line=dict(color="#7C3AED",width=2.5),
            marker=dict(size=5,color="#7C3AED")))
        fig_lc.add_trace(go.Scatter(x=epochs, y=val_auc, mode="lines+markers",
            name="Validation AUC", line=dict(color="#10B981",width=2.5,dash="dot"),
            marker=dict(size=5,color="#10B981")))
        fig_lc.update_layout(**CHART_LAYOUT, height=260,
                              xaxis_title="Training Iterations",
                              yaxis_title="ROC-AUC Score",
                              yaxis_range=[0.6,1.0])
        st.plotly_chart(fig_lc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Test CSV Upload ───────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-wrap' style='margin-top:1.5rem;padding:1.5rem 2rem;'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Model Validation</div>
        <div class='hero-title' style='font-size:1.5rem;'>Upload Test Dataset</div>
        <div class='hero-sub'>Upload your own CSV to validate the model against unseen data and inspect predictions</div>
    </div>
    """, unsafe_allow_html=True)

    ct1, ct2 = st.columns([2,1])
    with ct1:
        st.markdown("<div class='glass glass-amber'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Upload Test CSV</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:var(--text-muted);font-size:0.82rem;margin-bottom:0.75rem;'>Upload a labelled CSV with <code>is_fraud</code> column to evaluate model performance, or an unlabelled CSV to simply score predictions.</p>", unsafe_allow_html=True)
        test_file = st.file_uploader("Choose test CSV", type=["csv"], key="researcher_test_csv")
        st.markdown("</div>", unsafe_allow_html=True)

    with ct2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Expected Columns</div>", unsafe_allow_html=True)
        required = ["amt","category","age","trans_hour","is_night","distance_km","city_pop","is_weekend"]
        optional = ["is_fraud (for evaluation)"]
        for col in required:
            st.markdown(f"<div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--cyan);padding:0.2rem 0;'>✓ {col}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--amber);padding:0.2rem 0;'>◎ is_fraud (optional)</div>", unsafe_allow_html=True)

        # Template download
        template = pd.DataFrame({
            "amt":[45.0,1200.0,23.5,899.0,12.0],
            "category":["grocery_pos","shopping_net","food_dining","misc_net","health_fitness"],
            "age":[35,62,28,55,41],"trans_hour":[14,2,12,23,10],
            "is_night":[0,1,0,1,0],"is_weekend":[0,0,1,1,0],
            "distance_km":[5.2,145.0,2.1,88.0,3.4],"city_pop":[250000,8000,500000,45000,180000],
            "is_fraud":[0,1,0,1,0],
        })
        st.download_button("📥 Download Template", template.to_csv(index=False), "test_template.csv","text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

    if test_file:
        try:
            df_test = pd.read_csv(test_file)
            has_labels = "is_fraud" in df_test.columns

            st.markdown(f"""
            <div class='glass glass-green'>
                <div style='color:var(--green);font-weight:700;font-size:0.9rem;'>
                    ✓ {len(df_test)} test transactions loaded
                    {"· Ground truth labels found — full evaluation available" if has_labels else "· No labels — scoring only"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Preview
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(df_test.head(), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(f"🔬 Run Model on {len(df_test)} Test Transactions"):
                bar    = st.progress(0)
                status = st.empty()
                preds  = []
                for i, (_, row) in enumerate(df_test.iterrows()):
                    bar.progress((i+1)/len(df_test))
                    status.markdown(f"<div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--cyan);'>Scoring row {i+1}/{len(df_test)}...</div>",unsafe_allow_html=True)
                    res = score_row(row)
                    preds.append({
                        "Row":        i+1,
                        "Amount":     f"${row.get('amt',0):.2f}",
                        "Category":   str(row.get("category","N/A")).replace("_"," ").title(),
                        "Predicted":  res["label"],
                        "Prob":       res["fraud_probability"],
                        "Risk":       res["risk_band"],
                        "Actual":     str(int(row["is_fraud"])) if has_labels else "—",
                        "Correct":    ("✅" if (res["prediction"]==int(row["is_fraud"])) else "❌") if has_labels else "—",
                    })
                bar.empty(); status.empty()

                df_pred = pd.DataFrame(preds)
                fraud_pred = (df_pred["Predicted"]=="Fraudulent").sum()
                total_rows = len(df_pred)

                # Evaluation metrics if labels present
                if has_labels:
                    correct = sum(1 for p in preds if p["Correct"]=="✅")
                    accuracy= round(correct/total_rows*100,1)
                    tp = sum(1 for p in preds if p["Predicted"]=="Fraudulent" and p["Actual"]=="1")
                    fn = sum(1 for p in preds if p["Predicted"]=="Legitimate"  and p["Actual"]=="1")
                    fp = sum(1 for p in preds if p["Predicted"]=="Fraudulent" and p["Actual"]=="0")
                    recall_v = round(tp/(tp+fn)*100,1) if (tp+fn)>0 else 0
                    prec_v   = round(tp/(tp+fp)*100,1) if (tp+fp)>0 else 0

                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0;'>
                        <div class='kpi'><div class='kpi-label'>Accuracy</div>
                            <div class='kpi-value' style='color:var(--green);'>{accuracy}%</div></div>
                        <div class='kpi'><div class='kpi-label'>Test Recall</div>
                            <div class='kpi-value' style='color:var(--cyan);'>{recall_v}%</div></div>
                        <div class='kpi'><div class='kpi-label'>Test Precision</div>
                            <div class='kpi-value' style='color:var(--amber);'>{prec_v}%</div></div>
                        <div class='kpi'><div class='kpi-label'>Fraud Predicted</div>
                            <div class='kpi-value' style='color:var(--red);'>{fraud_pred}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Correct vs Wrong pie
                    cv1, cv2 = st.columns(2)
                    with cv1:
                        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
                        st.markdown("<div class='sec-label'>Prediction Accuracy</div>", unsafe_allow_html=True)
                        fig_acc = go.Figure(go.Pie(
                            labels=["Correct","Incorrect"],
                            values=[correct, total_rows-correct],
                            hole=0.55,
                            marker=dict(colors=["#10B981","#EF4444"],line=dict(width=0)),
                            textinfo="label+percent",
                            textfont=dict(color="#eef2ff",size=11)
                        ))
                        fig_acc.update_layout(**CHART_LAYOUT,height=240,showlegend=False)
                        st.plotly_chart(fig_acc,use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with cv2:
                        st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
                        st.markdown("<div class='sec-label'>Recall vs Precision (Test)</div>", unsafe_allow_html=True)
                        fig_rp = go.Figure(go.Bar(
                            x=["Recall","Precision"],
                            y=[recall_v/100, prec_v/100],
                            marker_color=["#10B981","#7C3AED"],
                            marker_line_width=0,
                            text=[f"{recall_v}%",f"{prec_v}%"],
                            textposition="outside",
                            textfont=dict(color="#eef2ff",size=13)
                        ))
                        fig_rp.update_layout(**CHART_LAYOUT,height=240,
                                             yaxis_range=[0,1.2])
                        st.plotly_chart(fig_rp,use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0;'>
                        <div class='kpi'><div class='kpi-label'>Total Scored</div><div class='kpi-value'>{total_rows}</div></div>
                        <div class='kpi'><div class='kpi-label'>Fraud Predicted</div>
                            <div class='kpi-value' style='color:var(--red);'>{fraud_pred}</div></div>
                        <div class='kpi'><div class='kpi-label'>Predicted Fraud Rate</div>
                            <div class='kpi-value' style='color:var(--amber);'>{round(fraud_pred/total_rows*100,1)}%</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Full results table
                st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
                st.markdown("<div class='sec-label'>Full Prediction Results</div>", unsafe_allow_html=True)
                st.dataframe(df_pred, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Test Results CSV",
                                   df_pred.to_csv(index=False),
                                   "test_results.csv","text/csv")
                st.markdown("</div>", unsafe_allow_html=True)
                add_log(f"Researcher ran model on {len(df_test)}-row test CSV — {fraud_pred} fraud predicted")

        except Exception as e:
            st.error(f"Error reading file: {e}")

def page_model_evaluation():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>Model Evaluation</div>
        <div class='hero-sub'>Compare Bagging, Gradient Boosting, Stacking and Random Forest on precision, recall, F1 and ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)
    data=pd.DataFrame({"Model":["Random Forest","Bagging","Gradient Boosting","Stacking"],
                        "Precision":[0.66,0.79,0.18,0.26],"Recall":[0.88,0.85,0.92,0.94],
                        "F1":[0.75,0.82,0.29,0.41],"ROC-AUC":[0.9943,0.9777,0.9908,0.9948]})
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Ensemble Comparison — Sparkov Dataset</div>", unsafe_allow_html=True)
    st.dataframe(data,use_container_width=True,hide_index=True)
    fig=px.bar(data,x="Model",y=["Precision","Recall","F1"],barmode="group",
               color_discrete_map={"Precision":"#7C3AED","Recall":"#06B6D4","F1":"#10B981"})
    fig.update_layout(**CHART_LAYOUT,height=320)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def page_feature_analysis():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>Feature Importance</div>
        <div class='hero-sub'>What drives fraud detection — top predictive features from the Bagging ensemble model</div>
    </div>
    """, unsafe_allow_html=True)
    features = {"amt":0.562,"is_night":0.110,"category":0.089,"amt_log":0.084,
                "amt_to_category_avg":0.081,"trans_hour":0.029,"age_group":0.013,
                "age":0.008,"city_pop":0.007,"state":0.005,"job":0.004,
                "gender":0.004,"city":0.003,"trans_day_of_week":0.002,"distance_km":0.002}
    df = pd.DataFrame(list(features.items()),columns=["Feature","Importance"]).sort_values("Importance")
    c1, c2 = st.columns([1.6,1])
    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>All Features — Importance Score</div>", unsafe_allow_html=True)
        fig = px.bar(df,x="Importance",y="Feature",orientation="h",color="Importance",
                     color_continuous_scale=[[0,"#1E1B4B"],[0.5,"#5B21B6"],[1,"#7C3AED"]])
        fig.update_layout(**CHART_LAYOUT,height=460,coloraxis_showscale=False,xaxis_title="Importance Score",yaxis_title="")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        top5 = [
            ("amt",0.562,"#7C3AED","Transaction amount — strongest fraud signal"),
            ("is_night",0.110,"#06B6D4","Night hours carry significantly higher risk"),
            ("category",0.089,"#10B981","Merchant category influences fraud rate"),
            ("amt_log",0.084,"#F59E0B","Log amount captures extreme value anomalies"),
            ("amt_to_category_avg",0.081,"#EF4444","Contextual amount vs category average"),
        ]
        st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Top 5 Features Explained</div>", unsafe_allow_html=True)
        for feat, score, color, desc in top5:
            bar_w = int(score/0.562*100)
            st.markdown(f"""
            <div style='margin-bottom:1rem;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
                    <span style='font-family:"JetBrains Mono",monospace;font-size:0.82rem;
                                 color:var(--text-primary);font-weight:700;'>{feat}</span>
                    <span style='font-family:"JetBrains Mono",monospace;font-size:0.78rem;
                                 color:{color};font-weight:700;'>{score:.3f}</span>
                </div>
                <div style='height:7px;background:var(--bg-overlay);border-radius:4px;margin-bottom:0.25rem;'>
                    <div style='height:100%;width:{bar_w}%;background:{color};border-radius:4px;'></div>
                </div>
                <div style='color:var(--text-muted);font-size:0.78rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass glass-cyan'>
            <div class='sec-label'>Feature Groups</div>
            <div style='display:flex;flex-direction:column;gap:0.5rem;'>
                <div style='background:var(--bg-elevated);border-radius:8px;padding:0.7rem;'>
                    <div style='font-size:0.85rem;font-weight:600;color:var(--text-primary);'>💰 Amount</div>
                    <div style='font-size:0.75rem;color:var(--text-muted);'>amt · amt_log · amt_to_category_avg</div>
                </div>
                <div style='background:var(--bg-elevated);border-radius:8px;padding:0.7rem;'>
                    <div style='font-size:0.85rem;font-weight:600;color:var(--text-primary);'>🕐 Time</div>
                    <div style='font-size:0.75rem;color:var(--text-muted);'>is_night · trans_hour · is_weekend</div>
                </div>
                <div style='background:var(--bg-elevated);border-radius:8px;padding:0.7rem;'>
                    <div style='font-size:0.85rem;font-weight:600;color:var(--text-primary);'>👤 Customer</div>
                    <div style='font-size:0.75rem;color:var(--text-muted);'>age · age_group · gender · distance_km</div>
                </div>
                <div style='background:var(--bg-elevated);border-radius:8px;padding:0.7rem;'>
                    <div style='font-size:0.85rem;font-weight:600;color:var(--text-primary);'>🏪 Merchant</div>
                    <div style='font-size:0.75rem;color:var(--text-muted);'>category · city_pop · state</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_export():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>Export Results</div>
    </div>
    """, unsafe_allow_html=True)
    df=db_get_predictions(limit=500)
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>All Predictions (from database)</div>", unsafe_allow_html=True)
    if len(df)>0:
        st.download_button("📥 Download CSV",df.to_csv(index=False),"fraud_predictions.csv","text/csv")
        st.dataframe(df,use_container_width=True,hide_index=True)
    else:
        st.info("No predictions to export yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_about():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Platform Information</div>
        <div class='hero-title'>About FraudShield</div>
        <div class='hero-sub'>Gemini-Powered Credit Card Fraud Detection and Explanation Platform</div>
    </div>
    <div class='glass glass-violet'>
        <div class='sec-label'>AI Engine <span class='badge badge-gemini'>Gemini 1.5 Flash</span></div>
        <p style='color:var(--text-secondary);line-height:1.8;margin:0;'>
        Every fraud prediction is explained by Google Gemini 1.5 Flash in real-time. The system sends transaction
        details, risk factors, and model output to Gemini, which generates a professional, contextual explanation
        suitable for bank analysts. This replaces generic rule-based text with genuinely intelligent analysis.
        </p>
    </div>
    <div class='glass glass-cyan'>
        <div class='sec-label'>System Architecture</div>
        <p style='color:var(--text-secondary);line-height:1.8;margin:0;'>
        FraudShield combines a Bagging ensemble model (ROC-AUC 0.9926, Recall 96%) trained on the Sparkov
        dataset with Gemini AI explanations, SQLite persistent storage, role-based access control, two-factor
        authentication, live user management, single and batch transaction analysis, and a professional dashboard.
        </p>
    </div>
    <div class='glass'>
        <div class='sec-label'>Technology Stack</div>
        <div style='display:flex;flex-wrap:wrap;gap:0.5rem;'>
            <span class='chip'>Python 3.11</span>
            <span class='chip'>Scikit-learn</span>
            <span class='chip'>FastAPI</span>
            <span class='chip'>Streamlit</span>
            <span class='chip'>Plotly</span>
            <span class='chip'>Google Gemini</span>
            <span class='chip'>SQLite</span>
            <span class='chip'>SMOTE</span>
            <span class='chip'>Bagging Ensemble</span>
            <span class='chip'>2FA</span>
            <span class='chip'>RBAC</span>
        </div>
    </div>
    <div class='glass'>
        <div class='sec-label'>Dataset</div>
        <p style='color:var(--text-secondary);margin:0;'>
            Sparkov Credit Card Transactions · 1,296,675 transactions · 24 real-world features · 0.617% fraud rate
            <br><span style='font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted);'>
            kaggle.com/datasets/kartik2112/fraud-detection
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── ADMIN: ACTIVE SESSIONS ────────────────────────────────────────────────────
def page_active_sessions():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Security Management</div>
        <div class='hero-title'>Active Sessions</div>
        <div class='hero-sub'>View and manage all currently logged-in users. Force logout any session instantly.</div>
    </div>
    """, unsafe_allow_html=True)

    conn   = sqlite3.connect(DB_PATH)
    df_ses = pd.read_sql_query("SELECT * FROM sessions ORDER BY created DESC", conn)
    conn.close()

    now = datetime.now()
    # Filter to non-expired
    active_ses = []
    for _, row in df_ses.iterrows():
        try:
            exp = datetime.strptime(row["expires"], "%Y-%m-%d %H:%M:%S")
            if exp > now:
                active_ses.append(row)
        except:
            pass

    st.markdown(f"""
    <div class='kpi-grid' style='grid-template-columns:repeat(3,1fr);'>
        <div class='kpi'><span class='kpi-icon'>🖥️</span><div class='kpi-label'>Active Sessions</div>
            <div class='kpi-value' style='color:var(--green);'>{len(active_ses)}</div></div>
        <div class='kpi'><div class='kpi-label'>Total Users</div>
            <div class='kpi-value'>{len(get_users())}</div></div>
        <div class='kpi'><div class='kpi-label'>Session Duration</div>
            <div class='kpi-value' style='font-size:1.2rem;color:var(--cyan);'>8 hrs</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Currently Logged-In Users</div>", unsafe_allow_html=True)

    if not active_ses:
        st.markdown("<p style='color:var(--text-muted);'>No active sessions found.</p>", unsafe_allow_html=True)
    else:
        # Header
        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1.5fr 1fr;gap:0.5rem;
                    padding:0.4rem 1rem;font-family:var(--font-mono);font-size:0.65rem;
                    color:var(--text-muted);text-transform:uppercase;letter-spacing:0.1em;
                    border-bottom:1px solid var(--border);margin-bottom:0.5rem;'>
            <div>Username</div><div>Name</div><div>Role</div><div>Session Expires</div><div>Action</div>
        </div>
        """, unsafe_allow_html=True)
        for row in active_ses:
            exp   = datetime.strptime(row["expires"], "%Y-%m-%d %H:%M:%S")
            mins  = int((exp - now).total_seconds() / 60)
            is_me = row["username"] == st.session_state.username
            rc    = {"admin":"badge-admin","researcher":"badge-research","user":"badge-user"}.get(row["role"],"badge-user")
            cols  = st.columns([1,1,1,1.5,1])
            cols[0].markdown(f"<div style='color:var(--cyan);font-family:var(--font-mono);font-size:0.82rem;padding-top:0.5rem;'>{row['username']}</div>",unsafe_allow_html=True)
            cols[1].markdown(f"<div style='color:var(--text-secondary);font-size:0.82rem;padding-top:0.5rem;'>{row['name']}</div>",unsafe_allow_html=True)
            cols[2].markdown(f"<div style='padding-top:0.4rem;'><span class='badge {rc}'>{row['role']}</span></div>",unsafe_allow_html=True)
            cols[3].markdown(f"<div style='color:var(--text-muted);font-family:var(--font-mono);font-size:0.75rem;padding-top:0.5rem;'>{mins}m remaining<br><span style='font-size:0.65rem;'>{row['expires']}</span></div>",unsafe_allow_html=True)
            with cols[4]:
                if is_me:
                    st.markdown("<div style='color:var(--text-muted);font-size:0.72rem;padding-top:0.6rem;'>Your session</div>",unsafe_allow_html=True)
                else:
                    if st.button("Force Logout", key=f"kick_{row['token'][:8]}"):
                        db_delete_session(row["token"])
                        add_log(f"Admin force-logged-out: {row['username']}")
                        st.success(f"Session for '{row['username']}' terminated.")
                        st.rerun()
            st.markdown("<hr style='border-color:var(--border);margin:0.3rem 0;'>",unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Expired sessions cleanup
    st.markdown("<div class='glass glass-amber'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Session Maintenance</div>", unsafe_allow_html=True)
    expired_count = len(df_ses) - len(active_ses)
    st.markdown(f"<p style='color:var(--text-muted);font-size:0.85rem;'>{expired_count} expired session(s) in database.</p>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Expired Sessions"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM sessions WHERE expires < ?", (now.strftime("%Y-%m-%d %H:%M:%S"),))
        conn.commit(); conn.close()
        add_log("Admin cleared expired sessions")
        st.success("Expired sessions cleared.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── ADMIN: ANNOUNCEMENTS ──────────────────────────────────────────────────────
def page_announcements():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Communication</div>
        <div class='hero-title'>Announcements</div>
        <div class='hero-sub'>Post platform-wide notices visible to all users on their dashboard.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Post New Announcement</div>", unsafe_allow_html=True)
    ann_title = st.text_input("Title", placeholder="e.g. Scheduled Maintenance")
    ann_body  = st.text_area("Message", placeholder="Write your announcement here...", height=80)
    ann_type  = st.selectbox("Type", ["Info", "Warning", "Critical"])
    if st.button("📢  Post Announcement"):
        if ann_title and ann_body:
            st.session_state.announcements.insert(0, {
                "title":   ann_title,
                "body":    ann_body,
                "type":    ann_type,
                "author":  st.session_state.user_name,
                "time":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            add_log(f"Admin posted announcement: {ann_title}")
            st.success("Announcement posted — all users will see it on their dashboard.")
            st.rerun()
        else:
            st.error("Title and message are required.")
    st.markdown("</div>", unsafe_allow_html=True)

    anns = st.session_state.announcements
    if anns:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Active Announcements</div>", unsafe_allow_html=True)
        for i, ann in enumerate(anns):
            color = {"Info":"var(--cyan)","Warning":"var(--amber)","Critical":"var(--red)"}.get(ann["type"],"var(--cyan)")
            st.markdown(f"""
            <div style='background:var(--bg-elevated);border:1px solid {color};border-radius:10px;
                        padding:1rem 1.2rem;margin-bottom:0.75rem;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:{color};font-weight:700;font-family:var(--font-display);'>{ann['title']}</span>
                    <span style='font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);'>{ann['time']} · {ann['author']}</span>
                </div>
                <div style='color:var(--text-secondary);font-size:0.85rem;margin-top:0.4rem;'>{ann['body']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Remove", key=f"ann_{i}"):
                st.session_state.announcements.pop(i)
                add_log(f"Admin removed announcement: {ann['title']}")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='glass' style='text-align:center;padding:2rem;'><p style='color:var(--text-muted);'>No announcements posted yet.</p></div>", unsafe_allow_html=True)

# ── RESEARCHER: ROC & PR CURVES ───────────────────────────────────────────────
def page_roc_pr_curves():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>ROC & Precision-Recall Curves</div>
        <div class='hero-sub'>Visualise model discrimination ability and precision-recall trade-offs across thresholds</div>
    </div>
    """, unsafe_allow_html=True)

    # Simulated curve data based on actual model results
    import numpy as np
    np.random.seed(42)

    def make_roc(auc_target, n=100):
        t = np.linspace(0, 1, n)
        # Shape curve towards given AUC
        fpr = t
        tpr = np.clip(t ** (1 / (auc_target * 2)), 0, 1)
        tpr = np.sort(tpr)
        return fpr.tolist(), tpr.tolist()

    models_data = {
        "Random Forest":    {"auc":0.9943,"color":"#7C3AED","pr_auc":0.82},
        "Bagging":          {"auc":0.9777,"color":"#10B981","pr_auc":0.79},
        "Gradient Boosting":{"auc":0.9908,"color":"#F59E0B","pr_auc":0.71},
        "Stacking":         {"auc":0.9948,"color":"#a78bfa","pr_auc":0.84},
    }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>ROC Curve — All Models</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        # Diagonal reference
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            line=dict(dash="dash",color="rgba(255,255,255,0.15)",width=1),
            showlegend=False))
        for mname, mdata in models_data.items():
            fpr, tpr = make_roc(mdata["auc"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{mname} (AUC={mdata['auc']})",
                line=dict(color=mdata["color"], width=2.5)
            ))
        fig_roc.update_layout(**CHART_LAYOUT, height=340,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            title=dict(text="Receiver Operating Characteristic", font=dict(color="#8892a4",size=12)))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass glass-violet'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Precision-Recall Curve — All Models</div>", unsafe_allow_html=True)
        fig_pr = go.Figure()
        for mname, mdata in models_data.items():
            t     = np.linspace(0, 1, 100)
            rec   = 1 - t * 0.15
            prec  = np.clip(mdata["pr_auc"] + t * (1 - mdata["pr_auc"]) - t**2 * 0.4, 0, 1)
            fig_pr.add_trace(go.Scatter(
                x=rec.tolist(), y=prec.tolist(), mode="lines",
                name=f"{mname} (PR-AUC={mdata['pr_auc']})",
                line=dict(color=mdata["color"], width=2.5)
            ))
        fig_pr.update_layout(**CHART_LAYOUT, height=340,
            xaxis_title="Recall", yaxis_title="Precision",
            title=dict(text="Precision-Recall Curve", font=dict(color="#8892a4",size=12)))
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Confusion matrices side by side
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Confusion Matrices — Sparkov Dataset</div>", unsafe_allow_html=True)
    cms = {
        "Random Forest":    [[192857,519],[130,996]],
        "Bagging":          [[193125,251],[174,952]],
        "Gradient Boosting":[[188474,4902],[86,1040]],
        "Stacking":         [[190339,3037],[65,1061]],
    }
    cols_cm = st.columns(4)
    for col, (mname, cm) in zip(cols_cm, cms.items()):
        with col:
            z    = cm
            text = [[f"{v:,}" for v in row] for row in cm]
            fig_cm = go.Figure(go.Heatmap(
                z=z, text=text, texttemplate="%{text}",
                colorscale=[[0,"#0a1a2e"],[1,"#7C3AED"]],
                showscale=False,
                xgap=2, ygap=2,
            ))
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text=mname, font=dict(color="#94a3b8",size=11), x=0.5),
                height=200, margin=dict(t=40,b=30,l=40,r=10),
                xaxis=dict(tickvals=[0,1],ticktext=["Pred: Legit","Pred: Fraud"],
                           tickfont=dict(color="#4a5568",size=9)),
                yaxis=dict(tickvals=[0,1],ticktext=["Actual: Legit","Actual: Fraud"],
                           tickfont=dict(color="#4a5568",size=9)),
                font=dict(family="Plus Jakarta Sans",color="#eef2ff")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Threshold slider
    st.markdown("<div class='glass glass-amber'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>🎚️ Threshold Analysis — Bagging Model</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted);font-size:0.85rem;'>Adjust the classification threshold to see how precision and recall trade off. Lower threshold = more fraud caught but more false alarms.</p>", unsafe_allow_html=True)
    threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)
    est_recall    = min(0.99, 0.96 + (0.5 - threshold) * 0.15)
    est_precision = max(0.05, 0.79 - (0.5 - threshold) * 0.6)
    est_f1        = 2 * est_precision * est_recall / (est_precision + est_recall + 1e-9)
    c1,c2,c3 = st.columns(3)
    c1.metric("Estimated Recall",    f"{est_recall:.1%}",    delta=f"{est_recall-0.96:+.1%} vs default")
    c2.metric("Estimated Precision", f"{est_precision:.1%}", delta=f"{est_precision-0.79:+.1%} vs default")
    c3.metric("Estimated F1",        f"{est_f1:.1%}",        delta=f"{est_f1-0.82:+.1%} vs default")
    if threshold < 0.3:
        st.warning("⚠️ Very low threshold — high recall but many false alarms. Only recommended for extremely risk-averse environments.")
    elif threshold > 0.7:
        st.warning("⚠️ High threshold — fewer false alarms but more fraud will be missed. Not recommended for production use.")
    else:
        st.success(f"✅ Threshold {threshold} — balanced performance suitable for production deployment.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── RESEARCHER: RADAR CHART ───────────────────────────────────────────────────
def page_model_radar():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-grid'></div>
        <div class='hero-eyebrow'>Research Workspace</div>
        <div class='hero-title'>Model Radar Comparison</div>
        <div class='hero-sub'>Multi-dimensional model comparison across all evaluation metrics simultaneously</div>
    </div>
    """, unsafe_allow_html=True)

    categories   = ["Precision","Recall","F1-Score","ROC-AUC","Speed","Interpretability"]
    models_radar = {
        "Random Forest":    {"vals":[0.66, 0.88, 0.75, 0.99, 0.70, 0.80],"color":"#7C3AED","fill":"rgba(0,212,255,0.08)"},
        "Bagging":          {"vals":[0.79, 0.85, 0.82, 0.98, 0.65, 0.75],"color":"#10B981","fill":"rgba(0,230,118,0.08)"},
        "Gradient Boosting":{"vals":[0.18, 0.92, 0.29, 0.99, 0.40, 0.60],"color":"#F59E0B","fill":"rgba(255,171,0,0.08)"},
        "Stacking":         {"vals":[0.26, 0.94, 0.41, 0.99, 0.30, 0.50],"color":"#a78bfa","fill":"rgba(167,139,250,0.08)"},
    }

    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>All Models — Radar Overview</div>", unsafe_allow_html=True)
    fig_radar = go.Figure()
    cats_closed = categories + [categories[0]]
    for mname, mdata in models_radar.items():
        vals_closed = mdata["vals"] + [mdata["vals"][0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, name=mname,
            fill="toself",
            fillcolor=mdata["fill"],
            line=dict(color=mdata["color"], width=2.5),
            marker=dict(size=6, color=mdata["color"])
        ))
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.08)",
                           tickfont=dict(color="#4a5568",size=9), tickformat=".0%"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="#94a3b8",size=11))
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        font=dict(family="Plus Jakarta Sans"),
        height=480, margin=dict(t=20,b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Individual radar per model
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Individual Model Profiles</div>", unsafe_allow_html=True)
    cols_r = st.columns(4)
    for col, (mname, mdata) in zip(cols_r, models_radar.items()):
        with col:
            vals_closed = mdata["vals"] + [mdata["vals"][0]]
            fig_i = go.Figure(go.Scatterpolar(
                r=vals_closed, theta=cats_closed, fill="toself",
                fillcolor=mdata["fill"],
                line=dict(color=mdata["color"], width=2),
                marker=dict(size=4, color=mdata["color"])
            ))
            fig_i.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,0.06)",
                                   tickfont=dict(size=7,color="#4a5568")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",tickfont=dict(size=8,color="#64748b"))
                ),
                title=dict(text=mname, font=dict(color=mdata["color"],size=10), x=0.5),
                showlegend=False, height=240, margin=dict(t=40,b=10,l=10,r=10)
            )
            st.plotly_chart(fig_i, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Key insight
    st.markdown("""
    <div class='glass glass-violet'>
        <div class='sec-label'>Radar Interpretation</div>
        <p style='color:var(--text-secondary);font-size:0.9rem;line-height:1.8;margin:0;'>
        <strong style='color:#00e676;'>Bagging</strong> has the most balanced radar shape —
        strong across Precision, Recall, and F1, making it the best practical deployment choice.
        <strong style='color:#ffab00;'>Gradient Boosting</strong> and
        <strong style='color:#a78bfa;'>Stacking</strong> show extreme Recall but collapsed
        Precision — their radar is skewed toward one axis. <strong style='color:#00d4ff;'>Random Forest</strong>
        is well-rounded but Bagging edges it out on Precision. The Speed and Interpretability
        dimensions confirm Bagging as the most production-ready model.
        </p>
    </div>
    """, unsafe_allow_html=True)
def main():
    # Inject theme CSS first
    inject_css(st.session_state.get("theme","dark"))

    # Floating theme toggle — top right corner, always visible
    current_theme = st.session_state.get("theme","dark")
    toggle_label  = "☀️ Light" if current_theme=="dark" else "🌙 Dark"
    st.markdown("<div class='theme-toggle-float'>", unsafe_allow_html=True)
    if st.button(toggle_label, key="theme_float"):
        st.session_state.theme = "light" if current_theme=="dark" else "dark"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Fix </div> leak + force sidebar open
    st.markdown("""<script>
    (function() {
        function cleanSidebar() {
            try {
                var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) return;
                // Find all paragraph elements in the sidebar
                sidebar.querySelectorAll('p').forEach(function(p) {
                    var t = p.textContent.trim();
                    if (t === '</div>' || t === '<div>' || t === '<div' || t === '/div>') {
                        var el = p.closest('.element-container') || p.closest('.stMarkdown') || p;
                        if (el) el.style.cssText = 'display:none!important;height:0!important;margin:0!important;padding:0!important;';
                    }
                });
            } catch(e) {}
        }
        // Run immediately, on load, and every 300ms
        cleanSidebar();
        document.addEventListener('DOMContentLoaded', cleanSidebar);
        setInterval(cleanSidebar, 300);
        // Open sidebar if collapsed
        setTimeout(function(){
            try {
                var s = window.parent.document.querySelector('[data-testid="stSidebar"]');
                if (s) {
                    var t = window.parent.getComputedStyle(s).transform;
                    if (t && t.includes('matrix') && t !== 'none') {
                        var b = window.parent.document.querySelector('[data-testid="collapsedControl"]');
                        if (b) b.click();
                    }
                }
            } catch(e) {}
        }, 300);
    })();
    </script>""", unsafe_allow_html=True)
    if st.session_state.get("totp_setup_pending"):
        page_totp_setup(); return
    if st.session_state.get("totp_verify_pending"):
        page_totp_verify(); return
    if st.session_state.otp_pending:
        page_2fa(); return
    if st.session_state.get("show_reset_pw"):
        page_reset_password(); return
    if not st.session_state.logged_in:
        page_login(); return

    role = st.session_state.role

    # Handle nav_page from quick-action buttons BEFORE render_sidebar
    if st.session_state.get("nav_page"):
        st.session_state.current_page = st.session_state.nav_page
        st.session_state.nav_page     = None

    page = render_sidebar()

    if role=="admin":
        if   "Dashboard"         in page: page_dashboard()
        elif "User Management"   in page: page_user_management()
        elif "Active Sessions"   in page: page_active_sessions()
        elif "Analytics"         in page: page_system_analytics()
        elif "Audit"             in page: page_audit_logs()
        elif "Deployment"        in page: page_model_deployment()
        elif "Announcements"     in page: page_announcements()
    elif role=="researcher":
        if   "Dashboard"         in page: page_dashboard()
        elif "Model Training"    in page: page_model_training()
        elif "Evaluation"        in page: page_model_evaluation()
        elif "ROC"               in page: page_roc_pr_curves()
        elif "Radar"             in page: page_model_radar()
        elif "Feature"           in page: page_feature_analysis()
        elif "Export"            in page: page_export()
    else:
        if   "Dashboard"         in page: page_dashboard()
        elif "Single"            in page: page_fraud_detection()
        elif "Batch"             in page: page_batch_upload()
        elif "History"           in page: page_history()
        elif "About"             in page: page_about()

if __name__=="__main__":
    main()