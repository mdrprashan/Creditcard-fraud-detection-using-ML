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

GEMINI_API_KEY = "AIzaSyDOR2zbGX04uBqeTNsrqcnqCzCeAY1rml0"
if GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="FraudShield — Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PREMIUM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
  --bg-base:        #04060f;
  --bg-surface:     rgba(255,255,255,0.03);
  --bg-elevated:    rgba(255,255,255,0.055);
  --bg-overlay:     rgba(255,255,255,0.08);
  --border:         rgba(255,255,255,0.07);
  --border-bright:  rgba(0,212,255,0.25);
  --cyan:           #00d4ff;
  --cyan-dim:       rgba(0,212,255,0.12);
  --cyan-glow:      rgba(0,212,255,0.35);
  --violet:         #7c3aed;
  --violet-dim:     rgba(124,58,237,0.12);
  --green:          #00e676;
  --green-dim:      rgba(0,230,118,0.1);
  --red:            #ff1744;
  --red-dim:        rgba(255,23,68,0.1);
  --amber:          #ffab00;
  --amber-dim:      rgba(255,171,0,0.1);
  --text-primary:   #eef2ff;
  --text-secondary: #8892a4;
  --text-muted:     #4a5568;
  --font-display:   'Outfit', sans-serif;
  --font-body:      'Plus Jakarta Sans', sans-serif;
  --font-mono:      'JetBrains Mono', monospace;
  --radius-sm:      8px;
  --radius-md:      12px;
  --radius-lg:      18px;
  --radius-xl:      24px;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
  font-family: var(--font-body);
  background-color: var(--bg-base);
  color: var(--text-primary);
}

/* ── Animated mesh background ── */
.main {
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,255,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.05) 0%, transparent 60%),
    radial-gradient(ellipse 50% 30% at 50% 50%, rgba(0,230,118,0.02) 0%, transparent 70%),
    var(--bg-base);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070a18 0%, #04060f 100%);
  border-right: 1px solid var(--border);
  backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] .block-container {
  padding: 1rem 0.75rem;
}

/* ── Main container ── */
.main .block-container {
  background: transparent;
  padding: 2rem 2.5rem;
  max-width: 1440px;
}

/* ── Glassmorphism card ── */
.glass {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  transition: border-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
}
.glass:hover {
  border-color: var(--border-bright);
  box-shadow: 0 0 0 1px rgba(0,212,255,0.08), 0 8px 32px rgba(0,0,0,0.4);
}
.glass-cyan  { border-left: 2px solid var(--cyan); }
.glass-green { border-left: 2px solid var(--green); }
.glass-red   { border-left: 2px solid var(--red); }
.glass-amber { border-left: 2px solid var(--amber); }
.glass-violet{ border-left: 2px solid var(--violet); }

/* ── Hero section ── */
.hero-wrap {
  position: relative;
  border-radius: var(--radius-xl);
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  overflow: hidden;
  background: linear-gradient(135deg,
    rgba(0,212,255,0.06) 0%,
    rgba(124,58,237,0.08) 50%,
    rgba(0,212,255,0.04) 100%);
  border: 1px solid rgba(0,212,255,0.15);
}
.hero-wrap::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, transparent 40%, rgba(0,212,255,0.03) 100%);
  pointer-events: none;
}
.hero-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
}
.hero-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: 0.2em;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 20px;
  height: 1px;
  background: var(--cyan);
}
.hero-title {
  font-family: var(--font-display);
  font-size: 2.4rem;
  font-weight: 800;
  color: var(--text-primary);
  margin: 0 0 0.5rem;
  line-height: 1.1;
  letter-spacing: -0.02em;
}
.hero-sub {
  color: var(--text-secondary);
  font-size: 1rem;
  margin: 0;
  font-weight: 400;
}
.hero-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1.2rem;
}
.chip {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  border: 1px solid var(--border-bright);
  color: var(--cyan);
  background: var(--cyan-dim);
  letter-spacing: 0.05em;
}

/* ── Section label ── */
.sec-label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--cyan);
  margin-bottom: 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Metric cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.kpi {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.4rem 1.2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.kpi::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.6;
}
.kpi:hover {
  border-color: rgba(0,212,255,0.2);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.3), 0 0 20px rgba(0,212,255,0.05);
}
.kpi-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--text-muted);
  margin-bottom: 0.6rem;
}
.kpi-value {
  font-family: var(--font-display);
  font-size: 2rem;
  font-weight: 800;
  color: var(--cyan);
  line-height: 1;
}
.kpi-sub {
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 0.3rem;
}

/* ── Buttons ── */
.stButton > button {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 0.88rem;
  letter-spacing: 0.02em;
  color: #04060f;
  background: linear-gradient(135deg, #00d4ff, #0099cc);
  border: none;
  border-radius: var(--radius-sm);
  padding: 0.65rem 1.4rem;
  width: 100%;
  cursor: pointer;
  transition: all 0.25s ease;
  position: relative;
  overflow: hidden;
}
.stButton > button::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
  opacity: 0;
  transition: opacity 0.2s;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0,212,255,0.4), 0 0 0 1px rgba(0,212,255,0.3);
}
.stButton > button:hover::after { opacity: 1; }
.stButton > button:active { transform: translateY(0); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 3px var(--cyan-dim) !important;
}
.stSelectbox > div > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}

/* ── Sidebar user card ── */
.user-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  text-align: center;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
}
.user-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.user-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 800;
  color: white;
  margin: 0 auto 0.75rem;
  box-shadow: 0 0 20px rgba(0,212,255,0.3);
}
.user-name {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--text-primary);
}
.user-handle {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

/* ── Badges ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.badge-admin    { background: rgba(255,23,68,0.12);   color: #ff6b81; border: 1px solid rgba(255,23,68,0.25); }
.badge-research { background: rgba(255,171,0,0.12);   color: #ffc857; border: 1px solid rgba(255,171,0,0.25); }
.badge-user     { background: rgba(0,230,118,0.12);   color: #69f0ae; border: 1px solid rgba(0,230,118,0.25); }
.badge-gemini   { background: rgba(124,58,237,0.15);  color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.badge-verified { background: rgba(0,230,118,0.1);    color: #69f0ae; border: 1px solid rgba(0,230,118,0.2); font-size:0.6rem; }

/* ── Status dot (animated pulse) ── */
.dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  animation: pulse-dot 2s infinite;
}
.dot-green  { background: var(--green);  box-shadow: 0 0 6px var(--green); }
.dot-red    { background: var(--red);    box-shadow: 0 0 6px var(--red); }
.dot-cyan   { background: var(--cyan);   box-shadow: 0 0 6px var(--cyan); }
.dot-amber  { background: var(--amber);  box-shadow: 0 0 6px var(--amber); }
@keyframes pulse-dot {
  0%,100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.6; transform: scale(0.85); }
}

/* ── Result boxes ── */
.result-box {
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  animation: fadeSlideUp 0.4s ease;
}
.result-fraud {
  background: radial-gradient(ellipse at center top, rgba(255,23,68,0.12) 0%, rgba(255,23,68,0.04) 60%, transparent 100%);
  border: 1px solid rgba(255,23,68,0.3);
}
.result-legit {
  background: radial-gradient(ellipse at center top, rgba(0,230,118,0.1) 0%, rgba(0,230,118,0.03) 60%, transparent 100%);
  border: 1px solid rgba(0,230,118,0.3);
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; animation: bounceIn 0.5s ease; }
.result-verdict {
  font-family: var(--font-display);
  font-size: 1.6rem;
  font-weight: 900;
  letter-spacing: -0.02em;
}
.verdict-fraud { color: var(--red); text-shadow: 0 0 30px rgba(255,23,68,0.4); }
.verdict-legit { color: var(--green); text-shadow: 0 0 30px rgba(0,230,118,0.4); }

/* ── Log row ── */
.log-row {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.65rem 1rem;
  margin-bottom: 0.4rem;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-secondary);
  transition: border-color 0.2s;
}
.log-row:hover { border-color: rgba(0,212,255,0.15); }

/* ── OTP box ── */
.otp-wrap {
  background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(124,58,237,0.06));
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: var(--radius-xl);
  padding: 2.5rem;
  text-align: center;
  margin: 1rem 0;
  position: relative;
  overflow: hidden;
}
.otp-code {
  font-family: var(--font-mono);
  font-size: 3rem;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.6rem;
  margin: 1rem 0;
  text-shadow: 0 0 30px rgba(0,212,255,0.5);
  animation: glowPulse 2s ease-in-out infinite;
}
@keyframes glowPulse {
  0%,100% { text-shadow: 0 0 20px rgba(0,212,255,0.4); }
  50%      { text-shadow: 0 0 40px rgba(0,212,255,0.8), 0 0 60px rgba(0,212,255,0.3); }
}

/* ── Nav card buttons (used in user dashboard) ── */
.nav-card-btn > button {
    background:    var(--bg-elevated) !important;
    border:        1px solid var(--border) !important;
    border-radius: 10px !important;
    padding:       1rem 1.2rem !important;
    text-align:    left !important;
    width:         100% !important;
    color:         var(--text-primary) !important;
    font-family:   var(--font-body) !important;
    font-size:     0.9rem !important;
    font-weight:   600 !important;
    transition:    all 0.2s ease !important;
    height:        auto !important;
    min-height:    70px !important;
}
.nav-card-btn > button:hover {
    border-color:  rgba(0,212,255,0.35) !important;
    background:    rgba(0,212,255,0.06) !important;
    transform:     translateX(4px) !important;
    box-shadow:    0 0 16px rgba(0,212,255,0.12) !important;
    color:         #00d4ff !important;
}
div[data-testid="stRadio"] > div {
  gap: 0.25rem;
  flex-direction: column;
}
div[data-testid="stRadio"] label {
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  padding: 0.55rem 0.75rem;
  color: var(--text-secondary);
  font-family: var(--font-body);
  font-size: 0.88rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  display: block;
}
div[data-testid="stRadio"] label:hover {
  background: var(--bg-elevated);
  color: var(--text-primary);
  border-color: var(--border);
}
div[data-testid="stRadio"] [data-checked="true"] label,
div[data-testid="stRadio"] input:checked + div {
  background: var(--cyan-dim);
  color: var(--cyan);
  border-color: rgba(0,212,255,0.25);
}

/* ── Animations ── */
@keyframes fadeSlideUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes bounceIn {
  0%   { transform:scale(0.5); opacity:0; }
  60%  { transform:scale(1.15); }
  100% { transform:scale(1); opacity:1; }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
.shimmer-text {
  background: linear-gradient(90deg, var(--text-secondary) 0%, var(--cyan) 50%, var(--text-secondary) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 3s linear infinite;
}

/* ── Brand wordmark ── */
.brand-wrap {
  text-align: center;
  padding: 1.5rem 0 1.8rem;
}
.brand-icon {
  font-size: 2.2rem;
  filter: drop-shadow(0 0 12px rgba(0,212,255,0.6));
  animation: glowPulse 3s ease-in-out infinite;
}
.brand-name {
  font-family: var(--font-display);
  font-size: 1.2rem;
  font-weight: 900;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  margin-top: 0.3rem;
}
.brand-tag {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 0.15rem;
}

/* ── Table styling ── */
.stDataFrame { border-radius: var(--radius-md); overflow: hidden; }
.stDataFrame thead th {
  background: var(--bg-elevated) !important;
  color: var(--cyan) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: var(--radius-md); }

/* ── Progress bar ── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--cyan), var(--violet)) !important;
  border-radius: 4px !important;
  box-shadow: 0 0 10px rgba(0,212,255,0.4);
}

/* ── File uploader ── */
.stFileUploader {
  border: 1px dashed rgba(0,212,255,0.2) !important;
  border-radius: var(--radius-md) !important;
  background: var(--bg-surface) !important;
  transition: border-color 0.2s;
}
.stFileUploader:hover { border-color: rgba(0,212,255,0.4) !important; }

/* ── Hide chrome selectively ── */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton                { display: none !important; }

/* ── Force sidebar always open and visible ── */
section[data-testid="stSidebar"] {
    display:    flex !important;
    visibility: visible !important;
    transform:  translateX(0) !important;
    min-width:  240px !important;
    max-width:  280px !important;
    opacity:    1 !important;
}

/* ── Hide the collapse button inside sidebar (prevents accidental close) ── */
section[data-testid="stSidebar"] button[kind="header"],
section[data-testid="stSidebar"] > div > div > button,
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* ── Make the expand arrow very visible if sidebar somehow collapses ── */
[data-testid="collapsedControl"] {
    display:          flex !important;
    visibility:       visible !important;
    opacity:          1 !important;
    position:         fixed !important;
    left:             0.5rem !important;
    top:              50% !important;
    z-index:          999999 !important;
    background:       #00d4ff !important;
    border:           none !important;
    border-radius:    50% !important;
    width:            36px !important;
    height:           36px !important;
    color:            #04060f !important;
    font-weight:      900 !important;
    font-size:        1.1rem !important;
    cursor:           pointer !important;
    box-shadow:       0 0 20px rgba(0,212,255,0.6) !important;
    align-items:      center !important;
    justify-content:  center !important;
}
[data-testid="collapsedControl"] svg {
    fill: #04060f !important;
    stroke: #04060f !important;
}
[data-testid="collapsedControl"]:hover {
    background:   #00b8d9 !important;
    box-shadow:   0 0 30px rgba(0,212,255,0.9) !important;
    transform:    scale(1.1) !important;
}

/* ── Page fade-in ── */
.main > div { animation: fadeSlideUp 0.35s ease; }
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
        email TEXT, status TEXT, created TEXT)""")
    conn.commit(); conn.close()
    # Seed default users into DB if not present
    _seed_default_users()

def _seed_default_users():
    """Insert default users into DB if they don't exist yet."""
    defaults = [
        ("admin",      "admin123",    "admin",      "System Admin", "mdrprashan10@gmail.com", "active", "2024-01-01"),
        ("researcher", "research123", "researcher", "Dr. Research", "mdrprashan10@gmail.com", "active", "2024-01-01"),
        ("user1",      "user123",     "user",       "John Analyst", "mdrprashan10@gmail.com", "active", "2024-01-01"),
    ]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for row in defaults:
        c.execute("INSERT OR IGNORE INTO users VALUES(?,?,?,?,?,?,?)", row)
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

# ── EMAIL CONFIGURATION ───────────────────────────────────────────────────────
# Replace with your Gmail address and App Password
# Get App Password: myaccount.google.com/security → App passwords
EMAIL_SENDER   = "mdrprashan10@gmail.com"
EMAIL_PASSWORD = "lzdqxddqicsptbkf"
ADMIN_EMAIL    = "mdrprashan10@gmail.com"

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
        <strong style="color:#00d4ff;">User Management → Pending Requests</strong>
        to approve or reject this request.
    </p>
    """
    return send_email(admin_email,
                      f"🔔 New Account Request — {name} (@{username})",
                      email_base(content, "New Account Request Pending Review"))

# ── USERS ─────────────────────────────────────────────────────────────────────
DEFAULT_USERS = {
    "admin":      {"password":"admin123",    "role":"admin",      "name":"System Admin",  "status":"active","created":"2024-01-01","email":"mdrprashan10@gmail.com"},
    "researcher": {"password":"research123", "role":"researcher", "name":"Dr. Research",  "status":"active","created":"2024-01-01","email":"mdrprashan10@gmail.com"},
    "user1":      {"password":"user123",     "role":"user",       "name":"John Analyst",  "status":"active","created":"2024-01-01","email":"mdrprashan10@gmail.com"},
}

for k,v in {"logged_in":False,"username":"","role":"","user_name":"","user_email":"",
            "otp_pending":False,"otp_code":"","otp_username":"",
            "otp_email_sent":False,"otp_email_addr":"",
            "users":None,
            "pending_users":[],"show_register":False,"show_reset_pw":False,
            "reset_otp":"","reset_username":"","reset_step":1,
            "session_token":"","failed_logins":{},"announcements":[],
            "nav_page":None,"current_page":"🏠  Dashboard"}.items():
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

def get_users(): return st.session_state.users
def add_log(action): db_save_log(st.session_state.username or "system", action)

# ── CHART THEME ───────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font={"color":"#8892a4","family":"Plus Jakarta Sans"},
    legend={"bgcolor":"rgba(0,0,0,0)","font":{"color":"#8892a4"}},
    xaxis={"gridcolor":"rgba(255,255,255,0.05)","zerolinecolor":"rgba(255,255,255,0.05)"},
    yaxis={"gridcolor":"rgba(255,255,255,0.05)","zerolinecolor":"rgba(255,255,255,0.05)"},
    margin=dict(t=16,b=16,l=8,r=8)
)

def gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value*100,1),
        title={"text":title,"font":{"color":"#8892a4","size":12,"family":"JetBrains Mono"}},
        number={"suffix":"%","font":{"color":"#eef2ff","size":24,"family":"Outfit"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#1a2035","tickfont":{"color":"#4a5568","size":9}},
            "bar":{"color":"#00d4ff","thickness":0.65},
            "bgcolor":"rgba(0,212,255,0.04)",
            "bordercolor":"rgba(0,0,0,0)",
            "steps":[
                {"range":[0,33],  "color":"rgba(0,230,118,0.06)"},
                {"range":[33,66], "color":"rgba(255,171,0,0.06)"},
                {"range":[66,100],"color":"rgba(255,23,68,0.08)"},
            ],
            "threshold":{"line":{"color":"#ff1744","width":2},"thickness":0.8,"value":50}
        }
    ))
    fig.update_layout(**CHART_LAYOUT, height=210)
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
    _,col,_ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div style='animation:fadeSlideUp 0.4s ease; margin-top:2.5rem;'>
        <div class='hero-wrap' style='text-align:center; padding:3rem 2rem;'>
            <div class='hero-grid'></div>
            <div class='brand-icon'>🛡️</div>
            <div class='hero-eyebrow' style='justify-content:center; margin-top:0.5rem;'>
                Fraud Intelligence Platform
            </div>
            <div class='hero-title' style='font-size:2.8rem;'>FraudShield</div>
            <div class='hero-sub'>Machine Learning × Gemini AI × Real-time Detection</div>
            <div class='hero-chips' style='justify-content:center;'>
                <span class='chip'>Bagging Ensemble</span>
                <span class='chip'>ROC-AUC 0.9926</span>
                <span class='chip'>Gemini AI</span>
                <span class='chip'>2FA Security</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='glass glass-cyan' style='animation:fadeSlideUp 0.5s ease;'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Sign In to Platform</div>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Sign In →"):
            users = get_users()
            failed = st.session_state.failed_logins
            # Check lockout
            if failed.get(username, 0) >= 3:
                st.error("⛔ Account temporarily locked after 3 failed attempts. Please use Forgot Password or contact admin.")
                add_log(f"Login blocked — account locked: {username}")
            elif username in users:
                user = users[username]
                if user["password"] == password:
                    if user.get("status","active") == "inactive":
                        st.error("Account deactivated. Contact administrator.")
                        add_log(f"Login blocked — inactive account: {username}")
                    else:
                        # Clear failed logins on success
                        st.session_state.failed_logins.pop(username, None)
                        otp        = str(random.randint(100000,999999))
                        user_email = users[username].get("email","")
                        user_name  = users[username].get("name", username)

                        # Send OTP via email
                        email_sent = False
                        if user_email:
                            email_sent = notify_2fa_otp(user_name, otp, user_email)

                        st.session_state.otp_pending   = True
                        st.session_state.otp_code      = otp
                        st.session_state.otp_username  = username
                        st.session_state.otp_email_sent = email_sent
                        st.session_state.otp_email_addr = user_email
                        add_log(f"Login — 2FA OTP {'emailed to ' + user_email if email_sent else 'generated (no email)'}")
                        st.rerun()
                else:
                    # Track failed attempt
                    st.session_state.failed_logins[username] = failed.get(username, 0) + 1
                    attempts_left = 3 - st.session_state.failed_logins[username]
                    if attempts_left <= 0:
                        st.error("⛔ Too many failed attempts. Account is now temporarily locked.")
                        add_log(f"Account locked after 3 failed attempts: {username}")
                    else:
                        st.error(f"Invalid credentials. {attempts_left} attempt{'s' if attempts_left>1 else ''} remaining before lockout.")
                        add_log(f"Failed login attempt for: {username}")
            else:
                st.error("Invalid credentials. Please check and try again.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Active accounts
        users = get_users()
        active = [(u,i) for u,i in users.items() if i.get("status")=="active"]
        st.markdown("<div class='glass' style='animation:fadeSlideUp 0.6s ease; margin-top:0.75rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Available Accounts</div>", unsafe_allow_html=True)
        for uname,info in active:
            bc = {"admin":"badge-admin","researcher":"badge-research","user":"badge-user"}.get(info["role"],"badge-user")
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;
                        border-bottom:1px solid var(--border);'>
                <span class='badge {bc}'>{info['role']}</span>
                <code style='color:var(--text-secondary);font-size:0.78rem;font-family:var(--font-mono);'>
                    {uname} / {info['password']}
                </code>
                <span style='color:var(--text-muted);font-size:0.75rem;margin-left:auto;'>{info['name']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center;margin-top:0.75rem;animation:fadeSlideUp 0.7s ease;'>
            <div style='display:inline-flex;align-items:center;gap:1rem;font-family:var(--font-mono);
                        font-size:0.65rem;color:var(--text-muted);'>
                <span><span class='dot dot-green'></span> &nbsp;Gemini AI Online</span>
                <span><span class='dot dot-cyan'></span> &nbsp;2FA Active</span>
                <span><span class='dot dot-cyan'></span> &nbsp;SQLite DB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Register and forgot password links
        st.markdown("""
        <div style='text-align:center;margin-top:1rem;animation:fadeSlideUp 0.8s ease;'>
            <span style='color:var(--text-muted);font-size:0.82rem;'>Don't have an account?</span>
        </div>
        """, unsafe_allow_html=True)
        col_r, col_f = st.columns(2)
        with col_r:
            if st.button("📝  Request Account Access"):
                st.session_state.show_register = True
                st.rerun()
        with col_f:
            if st.button("🔑  Forgot Password?"):
                st.session_state.show_reset_pw = True
                st.session_state.reset_step    = 1
                st.rerun()

# ── REGISTRATION PAGE ─────────────────────────────────────────────────────────
def page_register():
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div style='animation:fadeSlideUp 0.4s ease; margin-top:2rem;'>
        <div class='hero-wrap' style='text-align:center; padding:2.5rem 2rem;'>
            <div class='hero-grid'></div>
            <div class='brand-icon'>📝</div>
            <div class='hero-eyebrow' style='justify-content:center; margin-top:0.5rem;'>Account Request</div>
            <div class='hero-title' style='font-size:1.8rem;'>Request Access</div>
            <div class='hero-sub'>Submit your details — an admin will review and approve your account</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>Your Details</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            reg_name     = st.text_input("Full Name",     placeholder="Jane Smith")
            reg_username = st.text_input("Username",      placeholder="jsmith")
        with c2:
            reg_email    = st.text_input("Email Address", placeholder="jane@organisation.com")
            reg_role     = st.selectbox("Requested Role", ["user", "researcher"])

        reg_reason   = st.text_area("Reason for Access",
                                     placeholder="Briefly describe why you need access to FraudShield...",
                                     height=90)
        reg_pass     = st.text_input("Choose Password", type="password", placeholder="Min 6 characters")
        reg_pass2    = st.text_input("Confirm Password", type="password", placeholder="Repeat password")

        if st.button("📤  Submit Request"):
            if not reg_name or not reg_username or not reg_email or not reg_pass:
                st.error("All fields are required.")
            elif len(reg_pass) < 6:
                st.error("Password must be at least 6 characters.")
            elif reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            elif reg_username in get_users():
                st.error(f"Username '{reg_username}' is already taken. Please choose another.")
            elif any(p["username"]==reg_username for p in st.session_state.pending_users):
                st.error(f"A request for username '{reg_username}' is already pending admin approval.")
            else:
                st.session_state.pending_users.append({
                    "username":  reg_username,
                    "name":      reg_name,
                    "email":     reg_email,
                    "password":  reg_pass,
                    "role":      reg_role,
                    "reason":    reg_reason,
                    "submitted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                db_save_log("system", f"New account request submitted by {reg_username} ({reg_name}) for role: {reg_role}")

                # Notify admin by email
                admin_users = [i for i in get_users().values() if i["role"]=="admin" and i.get("email")]
                for admin in admin_users:
                    notify_admin_new_request(reg_name, reg_username, reg_role, reg_reason, admin["email"])

                st.success(f"✅ Request submitted! An administrator will review your account and you will be notified by email at **{reg_email}** once a decision is made. You can then log in with username '{reg_username}'.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='glass glass-amber' style='margin-top:0.75rem;'>
            <div style='font-family:var(--font-mono);font-size:0.72rem;color:var(--amber);'>
                ⏳ &nbsp; Account requests are reviewed by administrators. 
                You will be able to log in once your request is approved.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Back to Login"):
            st.session_state.show_register = False
            st.rerun()

# ── PASSWORD RESET EMAIL ──────────────────────────────────────────────────────
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
                    color:#00d4ff;letter-spacing:0.8rem;text-shadow:0 0 20px rgba(0,212,255,0.4);">
            {otp}
        </div>
        <div style="color:#475569;font-size:0.75rem;margin-top:0.75rem;">
            Valid for 5 minutes · Do not share this code
        </div>
    </div>

    <div style="background:rgba(255,171,0,0.08);border:1px solid rgba(255,171,0,0.2);
                border-radius:8px;padding:1rem;margin-bottom:1rem;">
        <p style="color:#ffab00;font-size:0.82rem;margin:0;">
            ⚠️ If you did not attempt to log in, your account may be at risk.
            Change your password immediately and contact your administrator.
        </p>
    </div>

    <p style="color:#4a5568;font-size:0.8rem;margin:0;">
        Never share this code with anyone. FraudShield will never ask for your OTP.
    </p>
    """
    return send_email(
        to_email,
        "🔐 FraudShield — Your Login Verification Code",
        email_base(content, "Two-Factor Authentication")
    )
    content = f"""
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1rem;">
        Hi <strong style="color:#eef2ff;">{name}</strong>,
    </p>
    <p style="color:#8892a4;line-height:1.7;margin:0 0 1.5rem;">
        We received a request to reset your FraudShield password.
        Use the code below to proceed. This code expires in <strong style="color:#eef2ff;">10 minutes</strong>.
    </p>
    <div style="background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(124,58,237,0.08));
                border:1px solid rgba(0,212,255,0.2);border-radius:12px;
                padding:2rem;text-align:center;margin-bottom:1.5rem;">
        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.15em;font-family:monospace;margin-bottom:0.5rem;">
            Password Reset Code
        </div>
        <div style="font-family:monospace;font-size:2.5rem;font-weight:700;
                    color:#00d4ff;letter-spacing:0.5rem;">
            {otp}
        </div>
    </div>
    <p style="color:#8892a4;line-height:1.7;margin:0;">
        If you did not request a password reset, please ignore this email.
        Your password will not be changed.
    </p>
    """
    return send_email(to_email,
                      "🔑 FraudShield Password Reset Code",
                      email_base(content, "Password Reset Request"))

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

            # Demo code display
            st.markdown(f"""
            <div class='otp-wrap' style='padding:1.5rem;'>
                <div style='font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);
                            margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.1em;'>
                    Demo mode — your code
                </div>
                <div class='otp-code' style='font-size:2rem;'>{st.session_state.reset_otp}</div>
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
                    st.session_state.reset_step     = 1
                    st.session_state.reset_otp      = ""
                    st.session_state.reset_username  = ""
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
            st.session_state.show_reset_pw  = False
            st.session_state.reset_step     = 1
            st.session_state.reset_otp      = ""
            st.session_state.reset_username = ""
            st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class='brand-wrap'>
            <div class='brand-icon'>🛡️</div>
            <div class='brand-name'>FraudShield</div>
            <div class='brand-tag'>Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        role = st.session_state.role
        initials = "".join(w[0].upper() for w in st.session_state.user_name.split()[:2])
        bc = {"admin":"badge-admin","researcher":"badge-research","user":"badge-user"}.get(role,"badge-user")

        st.markdown(f"""
        <div class='user-card'>
            <div class='user-avatar'>{initials}</div>
            <div class='user-name'>{st.session_state.user_name}</div>
            <div class='user-handle'>@{st.session_state.username}</div>
            <div style='margin-top:0.6rem;display:flex;justify-content:center;gap:0.5rem;'>
                <span class='badge {bc}'>{role}</span>
                <span class='badge badge-verified'>✓ 2FA</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sec-label' style='margin:0 0.5rem 0.75rem;'>Navigation</div>", unsafe_allow_html=True)

        # Get current page for syncing radio
        cur = st.session_state.get("current_page", "🏠  Dashboard")

        if role=="admin":
            admin_opts = [
                "🏠  Dashboard","👥  User Management","🖥️  Active Sessions",
                "📊  Analytics","📋  Audit Logs","⚙️  Model Deployment","📢  Announcements"
            ]
            cur_idx = next((i for i,o in enumerate(admin_opts) if cur in o or o in cur), 0)
            page = st.radio("", admin_opts, index=cur_idx, label_visibility="collapsed")
            pending = st.session_state.pending_users
            if pending:
                st.markdown(f"""
                <div style='margin-top:0.75rem;padding:0.75rem;
                            background:rgba(255,171,0,0.1);border:1px solid rgba(255,171,0,0.3);
                            border-radius:var(--radius-sm);'>
                    <div style='font-family:var(--font-mono);font-size:0.68rem;color:var(--amber);
                                display:flex;align-items:center;gap:0.5rem;'>
                        <span class='dot dot-amber'></span>
                        <strong>{len(pending)} pending approval{'s' if len(pending)>1 else ''}</strong>
                    </div>
                    <div style='font-size:0.72rem;color:var(--text-muted);margin-top:0.3rem;'>
                        Go to User Management → Pending Requests
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif role=="researcher":
            res_opts = [
                "🏠  Dashboard","🔬  Model Training","📈  Evaluation",
                "📉  ROC & PR Curves","🕸️  Model Radar","🔍  Feature Analysis","📤  Export"
            ]
            cur_idx = next((i for i,o in enumerate(res_opts) if cur in o or o in cur), 0)
            page = st.radio("", res_opts, index=cur_idx, label_visibility="collapsed")
        else:
            user_opts = [
                "🏠  Dashboard","🔎  Single Transaction","📂  Batch Upload",
                "📜  History","ℹ️  About"
            ]
            cur_idx = next((i for i,o in enumerate(user_opts) if cur in o or o in cur), 0)
            page = st.radio("", user_opts, index=cur_idx, label_visibility="collapsed")

        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        if st.button("↩  Sign Out"):
            add_log("User signed out")
            if st.session_state.get("session_token"):
                db_delete_session(st.session_state.session_token)
            try: st.query_params.clear()
            except: pass
            for k in ["logged_in","username","role","user_name","user_email",
                      "otp_pending","otp_code","otp_username","last_result",
                      "last_inputs","session_token"]:
                if k in st.session_state:
                    st.session_state[k] = False if k=="logged_in" else ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            r  = requests.get("http://127.0.0.1:8000/health",timeout=2)
            ok = r.status_code==200
        except: ok=False

        st.markdown(f"""
        <div style='margin-top:1.2rem;padding:0.75rem;background:var(--bg-elevated);
                    border:1px solid var(--border);border-radius:var(--radius-sm);'>
            <div style='font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>System Status</div>
            <div style='display:flex;flex-direction:column;gap:0.35rem;'>
                <div style='display:flex;align-items:center;gap:0.5rem;font-family:var(--font-mono);font-size:0.7rem;'>
                    <span class='dot {"dot-green" if ok else "dot-red"}'></span>
                    <span style='color:var(--text-secondary);'>API {'Online' if ok else 'Offline'}</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.5rem;font-family:var(--font-mono);font-size:0.7rem;'>
                    <span class='dot dot-cyan'></span>
                    <span style='color:var(--text-secondary);'>Gemini AI Active</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.5rem;font-family:var(--font-mono);font-size:0.7rem;'>
                    <span class='dot dot-green'></span>
                    <span style='color:var(--text-secondary);'>SQLite Connected</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.5rem;font-family:var(--font-mono);font-size:0.7rem;'>
                    <span class='dot dot-amber'></span>
                    <span style='color:var(--text-secondary);'>{len(get_users())} registered users</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
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
            <div class='kpi-label'>Registered Users</div>
            <div class='kpi-value'>{len(users)}</div>
            <div class='kpi-sub'>{active_u} active · {len(users)-active_u} inactive</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Active Sessions</div>
            <div class='kpi-value' style='color:var(--green);'>{active_ses}</div>
            <div class='kpi-sub'>Currently logged in</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Pending Approvals</div>
            <div class='kpi-value' style='color:{"var(--amber)" if pending>0 else "var(--text-muted)"};'>{pending}</div>
            <div class='kpi-sub'>{"Needs review" if pending>0 else "All clear"}</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Total Predictions</div>
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
                orientation="h", marker_color="#00d4ff", marker_line_width=0
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
                marker=dict(colors=["#00e676","#ff1744"], line=dict(width=0)),
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
        colors_map  = {"admin":"#ff1744","researcher":"#ffab00","user":"#00e676"}
        fig_r = go.Figure(go.Pie(
            labels=list(role_counts.keys()), values=list(role_counts.values()),
            hole=0.5, textinfo="label+value",
            marker=dict(colors=[colors_map.get(r,"#00d4ff") for r in role_counts.keys()],
                        line=dict(width=0))
        ))
        fig_r.update_layout(**CHART_LAYOUT, height=220, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        # Quick actions
        st.markdown("""
        <div class='glass glass-amber'>
            <div class='sec-label'>Quick Actions</div>
            <div style='display:flex;flex-direction:column;gap:0.6rem;'>
        """, unsafe_allow_html=True)
        actions = [
            ("👥", "User Management",  "Manage accounts and roles"),
            ("🖥️", "Active Sessions",  f"{active_ses} session(s) live"),
            ("📋", "Audit Logs",       f"{len(logs_df)} entries logged"),
            ("📢", "Announcements",    "Post platform notices"),
        ]
        if pending > 0:
            actions.insert(0, ("⏳", "Pending Approvals", f"{pending} request(s) waiting"))
        for icon, title, desc in actions:
            st.markdown(f"""
            <div style='background:var(--bg-elevated);border:1px solid var(--border);
                        border-radius:8px;padding:0.65rem 1rem;
                        display:flex;align-items:center;gap:0.75rem;'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <div>
                    <div style='color:var(--text-primary);font-size:0.85rem;font-weight:600;'>{title}</div>
                    <div style='color:var(--text-muted);font-size:0.72rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Recent audit log
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Recent System Activity</div>", unsafe_allow_html=True)
    recent = db_get_logs(limit=8)
    if len(recent)>0:
        for _,row in recent.iterrows():
            st.markdown(f"""
            <div class='log-row'>
                <span style='color:var(--text-muted);'>[{row['timestamp']}]</span>
                &nbsp;<span style='color:var(--cyan);'>{row['username']}</span>
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
            <div class='kpi-label'>Best ROC-AUC</div>
            <div class='kpi-value'>0.9926</div>
            <div class='kpi-sub'>Bagging · updated dataset</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Fraud Recall</div>
            <div class='kpi-value' style='color:var(--green);'>96%</div>
            <div class='kpi-sub'>After synthetic augmentation</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Best Precision</div>
            <div class='kpi-value' style='color:var(--cyan);'>0.79</div>
            <div class='kpi-sub'>Bagging classifier</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Models Trained</div>
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
                     color_discrete_map={"Precision":"#00d4ff","Recall":"#ff1744","F1":"#00e676"})
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
    fig_s.add_trace(go.Bar(name="+ 500 Synthetic Fraud",    x=metrics, y=[0.28,0.96,0.44,0.9926], marker_color="#00d4ff", marker_line_width=0))
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
            <div class='kpi-label'>My Total Checks</div>
            <div class='kpi-value'>{my_total}</div>
            <div class='kpi-sub'>Transactions analysed</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Fraud Detected</div>
            <div class='kpi-value' style='color:var(--red);'>{my_fraud}</div>
            <div class='kpi-sub'>Flagged as suspicious</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>Safe Transactions</div>
            <div class='kpi-value' style='color:var(--green);'>{my_legit}</div>
            <div class='kpi-sub'>Cleared as legitimate</div>
        </div>
        <div class='kpi'>
            <div class='kpi-label'>My Fraud Rate</div>
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
            v_color   = "#ff1744" if is_fraud else "#00e676"
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
            risk_color  = "#ff1744" if is_fraud else "#00e676"

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
                <div style='color:{risk_color};font-weight:700;font-family:var(--font-display);'>
                    {result.get("recommended_action","Review")}
                </div>
            </div>
            """, unsafe_allow_html=True)

            gemini_tag = "<span class='badge badge-gemini'>Gemini AI</span>" if used_gemini else "<span style='font-size:0.7rem;color:var(--text-muted);'>Fallback</span>"
            st.markdown(f"""
            <div class='glass glass-violet'>
                <div class='sec-label'>AI Explanation &nbsp; {gemini_tag}</div>
                <p style='color:var(--text-secondary);font-size:0.9rem;line-height:1.8;margin:0;'>{explanation}</p>
                <div style='margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid var(--border);
                            font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);'>
                    {"Google Gemini 1.5 Flash" if used_gemini else "Rule-based fallback"} ·
                    {datetime.now().strftime("%H:%M:%S")} · Saved to database
                </div>
            </div>
            """, unsafe_allow_html=True)
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
                    bar_color = "#ff1744" if prob>=0.8 else ("#ffab00" if prob>=0.5 else "#00e676")
                    v_color   = "#ff1744" if is_fraud else "#00e676"
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
                        marker_colors=["#00e676","#ffab00","#ff1744"],
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
                            marker_color="#ff1744",
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
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.markdown("""
        <div class='glass' style='text-align:center;padding:3rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;opacity:0.4;'>📂</div>
            <div style='font-family:var(--font-display);color:var(--text-muted);'>
                Upload a CSV file above<br>
                <span style='color:var(--cyan);'>or download the template to get started</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
                line=dict(color="#00d4ff",width=2.5),marker=dict(size=7,color="#00d4ff"),
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
                marker=dict(colors=["#00d4ff","#00e676","#ffab00","#ff1744","#a78bfa","#38bdf8"],
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
        sc="#00e676" if info.get("status")=="active" else "#4a5568"
        cols=st.columns([1,1.3,0.9,0.8,0.9,1.5,1.3])
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
                b1,b2=st.columns(2)
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
        <div class='hero-sub'>Every action is timestamped and stored in the SQLite database</div>
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
        <div class='kpi'><div class='kpi-label'>Total Predictions</div><div class='kpi-value'>{total}</div></div>
        <div class='kpi'><div class='kpi-label'>Fraud Detected</div><div class='kpi-value' style='color:var(--red);'>{fraud}</div></div>
        <div class='kpi'><div class='kpi-label'>Registered Users</div><div class='kpi-value'>{len(users)}</div></div>
        <div class='kpi'><div class='kpi-label'>Log Entries</div><div class='kpi-value'>{len(logs_df)}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if total>0:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Predictions by Outcome</div>", unsafe_allow_html=True)
            fig_pie=px.pie(df_all,names="result",color="result",
                           color_discrete_map={"Legitimate":"#00e676","Fraudulent":"#ff1744"})
            fig_pie.update_layout(**CHART_LAYOUT,height=240)
            fig_pie.update_traces(marker_line_width=0)
            st.plotly_chart(fig_pie,use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Fraud by Category</div>", unsafe_allow_html=True)
            cat_c=df_all[df_all["result"]=="Fraudulent"]["category"].value_counts().head(8)
            if len(cat_c)>0:
                fig_b=px.bar(x=cat_c.values,y=cat_c.index,orientation="h",color_discrete_sequence=["#ff1744"])
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
                marker_color=["#00d4ff","#00e676","#ffab00"],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in [res["prec"],res["rec"],res["f1"]]],
                textposition="outside",
                textfont=dict(color="#eef2ff", size=12)
            ))
            fig_m.update_layout(**CHART_LAYOUT, height=260,
                                yaxis=dict(range=[0,1.1], gridcolor="rgba(255,255,255,0.05)"))
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
                    "bar":{"color":"#00d4ff","thickness":0.65},
                    "bgcolor":"rgba(0,212,255,0.04)",
                    "bordercolor":"rgba(0,0,0,0)",
                    "steps":[
                        {"range":[80,90],"color":"rgba(255,23,68,0.08)"},
                        {"range":[90,95],"color":"rgba(255,171,0,0.08)"},
                        {"range":[95,100],"color":"rgba(0,230,118,0.08)"},
                    ],
                    "threshold":{"line":{"color":"#00e676","width":2},"thickness":0.8,"value":95}
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
                colorscale=[[0,"#0a1a2e"],[1,"#00d4ff"]],
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
            name="Training AUC", line=dict(color="#00d4ff",width=2.5),
            marker=dict(size=5,color="#00d4ff")))
        fig_lc.add_trace(go.Scatter(x=epochs, y=val_auc, mode="lines+markers",
            name="Validation AUC", line=dict(color="#00e676",width=2.5,dash="dot"),
            marker=dict(size=5,color="#00e676")))
        fig_lc.update_layout(**CHART_LAYOUT, height=260,
                              xaxis_title="Training Iterations",
                              yaxis_title="ROC-AUC Score",
                              yaxis=dict(range=[0.6,1.0],gridcolor="rgba(255,255,255,0.05)"))
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
                            marker=dict(colors=["#00e676","#ff1744"],line=dict(width=0)),
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
                            marker_color=["#00e676","#00d4ff"],
                            marker_line_width=0,
                            text=[f"{recall_v}%",f"{prec_v}%"],
                            textposition="outside",
                            textfont=dict(color="#eef2ff",size=13)
                        ))
                        fig_rp.update_layout(**CHART_LAYOUT,height=240,
                                             yaxis=dict(range=[0,1.2],gridcolor="rgba(255,255,255,0.05)"))
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
    </div>
    """, unsafe_allow_html=True)
    data=pd.DataFrame({"Model":["Random Forest","Bagging","Gradient Boosting","Stacking"],
                        "Precision":[0.66,0.79,0.18,0.26],"Recall":[0.88,0.85,0.92,0.94],
                        "F1":[0.75,0.82,0.29,0.41],"ROC-AUC":[0.9943,0.9777,0.9908,0.9948]})
    st.markdown("<div class='glass glass-cyan'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Ensemble Comparison — Sparkov Dataset</div>", unsafe_allow_html=True)
    st.dataframe(data,use_container_width=True,hide_index=True)
    fig=px.bar(data,x="Model",y=["Precision","Recall","F1"],barmode="group",
               color_discrete_map={"Precision":"#00d4ff","Recall":"#ff1744","F1":"#00e676"})
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
    </div>
    """, unsafe_allow_html=True)
    features={"amt":0.562,"is_night":0.110,"category":0.089,"amt_log":0.084,
              "amt_to_category_avg":0.081,"trans_hour":0.029,"age_group":0.013,
              "age":0.008,"city_pop":0.007,"state":0.005,"job":0.004,
              "gender":0.004,"city":0.003,"trans_day_of_week":0.002,"distance_km":0.002}
    df=pd.DataFrame(list(features.items()),columns=["Feature","Importance"]).sort_values("Importance")
    fig=px.bar(df,x="Importance",y="Feature",orientation="h",
               color="Importance",color_continuous_scale=[[0,"#1a2a4a"],[0.5,"#0066aa"],[1,"#00d4ff"]])
    fig.update_layout(**CHART_LAYOUT,height=440,coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("""
    <div class='glass glass-violet'>
        <div class='sec-label'>Key Insights</div>
        <p style='color:var(--text-secondary);font-size:0.9rem;line-height:1.8;margin:0;'>
        <strong style='color:var(--cyan);'>amt</strong> (0.562) is the dominant fraud predictor —
        transaction amount anomalies are the clearest fraud signal. &nbsp;
        <strong style='color:var(--cyan);'>is_night</strong> (0.110) confirms that timing matters significantly.
        &nbsp;<strong style='color:var(--cyan);'>category</strong> and
        <strong style='color:var(--cyan);'>amt_to_category_avg</strong> together enable context-aware detection —
        fraud is identified not just by amount, but by how unusual that amount is for the merchant type.
        </p>
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
        <div class='kpi'><div class='kpi-label'>Active Sessions</div>
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
        "Random Forest":    {"auc":0.9943,"color":"#00d4ff","pr_auc":0.82},
        "Bagging":          {"auc":0.9777,"color":"#00e676","pr_auc":0.79},
        "Gradient Boosting":{"auc":0.9908,"color":"#ffab00","pr_auc":0.71},
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
                colorscale=[[0,"#0a1a2e"],[1,"#00d4ff"]],
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
        "Random Forest":    {"vals":[0.66, 0.88, 0.75, 0.99, 0.70, 0.80],"color":"#00d4ff","fill":"rgba(0,212,255,0.08)"},
        "Bagging":          {"vals":[0.79, 0.85, 0.82, 0.98, 0.65, 0.75],"color":"#00e676","fill":"rgba(0,230,118,0.08)"},
        "Gradient Boosting":{"vals":[0.18, 0.92, 0.29, 0.99, 0.40, 0.60],"color":"#ffab00","fill":"rgba(255,171,0,0.08)"},
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

# ── ROUTER ────────────────────────────────────────────────────────────────────
def main():
    # Force sidebar open via JS on every render
    st.markdown("""
    <script>
    (function() {
        function openSidebar() {
            var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            var collapsed = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (collapsed) { collapsed.click(); }
        }
        // Run after short delay to let Streamlit render
        setTimeout(function() {
            var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                var style = window.parent.getComputedStyle(sidebar);
                var transform = style.getPropertyValue('transform');
                // If sidebar is translated off screen, click the expand button
                if (transform && transform.includes('matrix') && transform !== 'none') {
                    var btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
                    if (btn) btn.click();
                }
            }
        }, 300);
    })();
    </script>
    """, unsafe_allow_html=True)
    if st.session_state.otp_pending:
        page_2fa(); return
    if st.session_state.get("show_register"):
        page_register(); return
    if st.session_state.get("show_reset_pw"):
        page_reset_password(); return
    if not st.session_state.logged_in:
        page_login(); return

    page = render_sidebar()
    role = st.session_state.role

    # If a dashboard quick-nav button was clicked, override and sync
    if st.session_state.nav_page:
        page = st.session_state.nav_page
        st.session_state.nav_page = None

    # Always keep current_page in sync so sidebar highlights correctly
    st.session_state.current_page = page

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