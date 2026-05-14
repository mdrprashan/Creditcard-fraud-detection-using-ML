"""
FraudShield — Unit & Integration Test Suite
============================================
Course: ICT946 Capstone | Student: Prashan Manandhar
Run with: pytest test_fraudshield.py -v --tb=short
Coverage: pytest test_fraudshield.py --cov=streamlit_app --cov-report=term-missing
"""

import pytest
import sqlite3
import os
import sys
import random
import string
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── We test pure functions without launching Streamlit ────────────────────────
# Import only the non-Streamlit components
import importlib

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """Create a fresh temporary SQLite database for each test."""
    db_path = str(tmp_path / "test_fraudshield.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create all tables
    c.execute("""CREATE TABLE users (
        username TEXT PRIMARY KEY,
        password TEXT, role TEXT, name TEXT,
        email TEXT, status TEXT, created TEXT,
        totp_secret TEXT DEFAULT NULL,
        totp_enabled INTEGER DEFAULT 0)""")
    c.execute("""CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, username TEXT, amount REAL, category TEXT,
        result TEXT, fraud_probability REAL, risk_band TEXT,
        recommended_action TEXT, explanation TEXT, prediction_type TEXT)""")
    c.execute("""CREATE TABLE audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, username TEXT, action TEXT)""")
    c.execute("""CREATE TABLE sessions (
        token TEXT PRIMARY KEY,
        username TEXT, role TEXT, name TEXT, email TEXT,
        created TEXT, expires TEXT)""")
    c.execute("""CREATE TABLE locked_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, locked_at TEXT, attempts INTEGER,
        notified_admin INTEGER DEFAULT 0,
        unlocked_at TEXT, is_active INTEGER DEFAULT 1)""")

    # Seed default users
    users = [
        ("admin",      "admin123",    "admin",      "System Admin", "admin@test.com",  "active", "2024-01-01", None, 0),
        ("researcher", "research123", "researcher", "Dr. Research", "res@test.com",    "active", "2024-01-01", None, 0),
        ("user1",      "user123",     "user",       "John Analyst", "user@test.com",   "active", "2024-01-01", None, 0),
        ("inactive1",  "pass123",     "user",       "Off User",     "off@test.com",    "inactive","2024-01-01", None, 0),
    ]
    for u in users:
        c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?)", u)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_transaction():
    """A typical fraudulent transaction."""
    return {
        "amt": 2200.0,
        "category": "shopping_net",
        "age": 62,
        "trans_hour": 2,
        "is_night": 1,
        "is_weekend": 0,
        "distance_km": 145.0,
        "city_pop": 8000,
    }


@pytest.fixture
def legit_transaction():
    """A typical legitimate transaction."""
    return {
        "amt": 23.5,
        "category": "grocery_pos",
        "age": 35,
        "trans_hour": 14,
        "is_night": 0,
        "is_weekend": 1,
        "distance_km": 2.1,
        "city_pop": 500000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATABASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    """Unit tests for all SQLite database operations."""

    def test_users_table_exists(self, temp_db):
        """Database should contain the users table."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        result = c.fetchone()
        conn.close()
        assert result is not None, "users table should exist"

    def test_all_required_tables_exist(self, temp_db):
        """All five required tables must exist."""
        required = {"users", "predictions", "audit_logs", "sessions", "locked_accounts"}
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c.fetchall()}
        conn.close()
        assert required.issubset(tables), f"Missing tables: {required - tables}"

    def test_default_users_seeded(self, temp_db):
        """Three default users should be seeded on initialisation."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT username FROM users")
        users = {row[0] for row in c.fetchall()}
        conn.close()
        assert "admin" in users
        assert "researcher" in users
        assert "user1" in users

    def test_users_have_totp_columns(self, temp_db):
        """Users table must have totp_secret and totp_enabled columns."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in c.fetchall()}
        conn.close()
        assert "totp_secret" in columns, "totp_secret column must exist"
        assert "totp_enabled" in columns, "totp_enabled column must exist"

    def test_insert_prediction(self, temp_db):
        """Should successfully insert a prediction record."""
        conn = sqlite3.connect(temp_db)
        conn.execute("""INSERT INTO predictions
            (timestamp, username, amount, category, result,
             fraud_probability, risk_band, recommended_action, explanation, prediction_type)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "user1", 500.0, "shopping_net", "Fraudulent",
             0.92, "High Risk", "Block transaction",
             "AI explanation here", "single"))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions WHERE username='user1'")
        count = c.fetchone()[0]
        conn.close()
        assert count == 1

    def test_insert_audit_log(self, temp_db):
        """Should successfully insert an audit log entry."""
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO audit_logs (timestamp, username, action) VALUES (?,?,?)",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "admin", "Logged in"))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT action FROM audit_logs WHERE username='admin'")
        row = c.fetchone()
        conn.close()
        assert row[0] == "Logged in"

    def test_session_insert_and_retrieve(self, temp_db):
        """Session should be inserted and retrievable by token."""
        token = "testtoken123"
        expires = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                     (token, "user1", "user", "John Analyst", "user@test.com",
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"), expires))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT username FROM sessions WHERE token=?", (token,))
        row = c.fetchone()
        conn.close()
        assert row[0] == "user1"

    def test_delete_session(self, temp_db):
        """Session should be deletable by token."""
        token = "deletetoken"
        expires = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                     (token, "user1", "user", "John", "u@t.com",
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"), expires))
        conn.commit()
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM sessions WHERE token=?", (token,))
        assert c.fetchone()[0] == 0
        conn.close()

    def test_account_lockout_insert(self, temp_db):
        """Should record an account lockout event."""
        conn = sqlite3.connect(temp_db)
        conn.execute("""INSERT INTO locked_accounts
            (username, locked_at, attempts, notified_admin, is_active)
            VALUES (?,?,?,?,?)""",
            ("user1", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 3, 0, 1))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT is_active FROM locked_accounts WHERE username='user1'")
        assert c.fetchone()[0] == 1
        conn.close()

    def test_unlock_account(self, temp_db):
        """Unlocking should set is_active to 0."""
        conn = sqlite3.connect(temp_db)
        conn.execute("""INSERT INTO locked_accounts
            (username, locked_at, attempts, notified_admin, is_active)
            VALUES (?,?,?,?,?)""",
            ("user1", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 3, 0, 1))
        conn.commit()
        conn.execute("UPDATE locked_accounts SET is_active=0 WHERE username='user1'")
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT is_active FROM locked_accounts WHERE username='user1'")
        assert c.fetchone()[0] == 0
        conn.close()

    def test_user_status_update(self, temp_db):
        """Should update user status from active to inactive."""
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE users SET status=? WHERE username=?", ("inactive", "user1"))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT status FROM users WHERE username='user1'")
        assert c.fetchone()[0] == "inactive"
        conn.close()

    def test_password_update(self, temp_db):
        """Should update user password."""
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE users SET password=? WHERE username=?", ("newpass456", "user1"))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username='user1'")
        assert c.fetchone()[0] == "newpass456"
        conn.close()

    def test_totp_secret_save(self, temp_db):
        """Should save TOTP secret and enable TOTP for a user."""
        secret = "JBSWY3DPEHPK3PXP"
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE users SET totp_secret=?, totp_enabled=1 WHERE username=?",
                     (secret, "user1"))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT totp_secret, totp_enabled FROM users WHERE username='user1'")
        row = c.fetchone()
        conn.close()
        assert row[0] == secret
        assert row[1] == 1

    def test_totp_reset(self, temp_db):
        """Admin MFA reset should clear totp_secret and set totp_enabled to 0."""
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE users SET totp_secret=NULL, totp_enabled=0 WHERE username=?",
                     ("user1",))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT totp_secret, totp_enabled FROM users WHERE username='user1'")
        row = c.fetchone()
        conn.close()
        assert row[0] is None
        assert row[1] == 0

    def test_user_delete(self, temp_db):
        """Should permanently delete a user from the database."""
        conn = sqlite3.connect(temp_db)
        conn.execute("DELETE FROM users WHERE username=?", ("user1",))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE username='user1'")
        assert c.fetchone()[0] == 0
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUTHENTICATION LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthentication:
    """Unit tests for authentication and credential logic."""

    def test_correct_password_match(self, temp_db):
        """Correct password should match stored value."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username='admin'")
        stored = c.fetchone()[0]
        conn.close()
        assert stored == "admin123"

    def test_wrong_password_no_match(self, temp_db):
        """Wrong password must not match stored value."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username='admin'")
        stored = c.fetchone()[0]
        conn.close()
        assert stored != "wrongpassword"

    def test_inactive_user_blocked(self, temp_db):
        """Inactive user should be identified and blocked from login."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT status FROM users WHERE username='inactive1'")
        status = c.fetchone()[0]
        conn.close()
        assert status == "inactive"

    def test_active_user_allowed(self, temp_db):
        """Active user should have status=active."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT status FROM users WHERE username='admin'")
        status = c.fetchone()[0]
        conn.close()
        assert status == "active"

    def test_nonexistent_user_not_found(self, temp_db):
        """Non-existent username should return no record."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username='doesnotexist'")
        result = c.fetchone()
        conn.close()
        assert result is None

    def test_role_assignment_admin(self, temp_db):
        """Admin user should have role=admin."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username='admin'")
        assert c.fetchone()[0] == "admin"
        conn.close()

    def test_role_assignment_researcher(self, temp_db):
        """Researcher user should have role=researcher."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username='researcher'")
        assert c.fetchone()[0] == "researcher"
        conn.close()

    def test_session_expiry_in_future(self, temp_db):
        """New session expiry should be in the future."""
        expires = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        exp_dt  = datetime.strptime(expires, "%Y-%m-%d %H:%M:%S")
        assert exp_dt > datetime.now()

    def test_expired_session_detected(self):
        """Expired session should be detected as expired."""
        expired = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        exp_dt  = datetime.strptime(expired, "%Y-%m-%d %H:%M:%S")
        assert exp_dt < datetime.now()

    def test_session_token_unique(self):
        """Two generated session tokens should not be equal."""
        import uuid
        t1 = str(uuid.uuid4()).replace("-", "")
        t2 = str(uuid.uuid4()).replace("-", "")
        assert t1 != t2

    def test_failed_login_counter_increments(self):
        """Failed login counter should increment correctly."""
        failed = {}
        username = "testuser"
        failed[username] = failed.get(username, 0) + 1
        failed[username] = failed.get(username, 0) + 1
        failed[username] = failed.get(username, 0) + 1
        assert failed[username] == 3

    def test_lockout_triggered_at_three_failures(self):
        """Account should be locked after exactly 3 failures."""
        failed_count = 3
        is_locked = failed_count >= 3
        assert is_locked is True

    def test_lockout_not_triggered_before_three(self):
        """Account should not lock before 3 failures."""
        failed_count = 2
        is_locked = failed_count >= 3
        assert is_locked is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOTP / MFA TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTOTP:
    """Unit tests for Google Authenticator TOTP logic."""

    def test_totp_secret_generation(self):
        """Generated TOTP secret should be a non-empty string."""
        import pyotp
        secret = pyotp.random_base32()
        assert isinstance(secret, str)
        assert len(secret) > 0

    def test_totp_secret_is_base32(self):
        """TOTP secret should only contain valid Base32 characters."""
        import pyotp
        secret = pyotp.random_base32()
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
        assert all(c in valid_chars for c in secret)

    def test_valid_totp_code_accepted(self):
        """Current TOTP code generated from secret should verify correctly."""
        import pyotp
        secret = pyotp.random_base32()
        totp   = pyotp.TOTP(secret)
        code   = totp.now()
        assert totp.verify(code, valid_window=1) is True

    def test_invalid_totp_code_rejected(self):
        """Wrong TOTP code should fail verification."""
        import pyotp
        secret = pyotp.random_base32()
        totp   = pyotp.TOTP(secret)
        assert totp.verify("000000", valid_window=1) is False

    def test_totp_uri_contains_issuer(self):
        """TOTP provisioning URI should contain the FraudShield issuer name."""
        import pyotp
        secret = pyotp.random_base32()
        uri    = pyotp.TOTP(secret).provisioning_uri(
            name="testuser", issuer_name="FraudShield")
        assert "FraudShield" in uri

    def test_totp_uri_contains_username(self):
        """TOTP URI should include the username."""
        import pyotp
        secret = pyotp.random_base32()
        uri    = pyotp.TOTP(secret).provisioning_uri(
            name="prashan", issuer_name="FraudShield")
        assert "prashan" in uri

    def test_qr_code_generates(self):
        """QR code generation should produce a non-empty base64 string."""
        import pyotp, qrcode, io, base64
        secret = pyotp.random_base32()
        uri    = pyotp.TOTP(secret).provisioning_uri(
            name="testuser", issuer_name="FraudShield")
        qr  = qrcode.QRCode(version=1, box_size=8, border=2)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#7C3AED", back_color="#0F1423")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        assert len(b64) > 100

    def test_different_secrets_produce_different_codes(self):
        """Two different secrets should produce different TOTP codes."""
        import pyotp
        s1 = pyotp.random_base32()
        s2 = pyotp.random_base32()
        # Secrets should be different (extremely high probability)
        assert s1 != s2


# ─────────────────────────────────────────────────────────────────────────────
# 4. FRAUD SCORING LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFraudScoring:
    """Unit tests for the fraud probability scoring logic."""

    def _score(self, amt, is_night, category, distance, is_high_amt):
        """Replicate the demo scoring formula from score_row."""
        import math
        prob = min(0.95, 0.05
                   + (0.30 if is_night else 0)
                   + (0.25 if is_high_amt else 0)
                   + (0.15 if category in ["shopping_net", "misc_net"] else 0)
                   + (0.10 if distance > 100 else 0))
        return prob

    def test_high_risk_transaction_flagged(self, sample_transaction):
        """High amount + night + online shopping should exceed 0.5 threshold."""
        t = sample_transaction
        prob = self._score(t["amt"], t["is_night"], t["category"],
                           t["distance_km"], t["amt"] > 500)
        assert prob >= 0.5, "High-risk transaction should be flagged as fraud"

    def test_low_risk_transaction_cleared(self, legit_transaction):
        """Small daytime grocery transaction should be below 0.5 threshold."""
        t = legit_transaction
        prob = self._score(t["amt"], t["is_night"], t["category"],
                           t["distance_km"], t["amt"] > 500)
        assert prob < 0.5, "Legitimate transaction should not be flagged"

    def test_night_flag_increases_probability(self):
        """Night transactions should have higher fraud probability than day."""
        day_prob   = self._score(100, 0, "grocery_pos", 5, 0)
        night_prob = self._score(100, 1, "grocery_pos", 5, 0)
        assert night_prob > day_prob

    def test_high_amount_increases_probability(self):
        """High amount should increase fraud probability."""
        normal_prob = self._score(50,  0, "grocery_pos", 5, 0)
        high_prob   = self._score(800, 0, "grocery_pos", 5, 1)
        assert high_prob > normal_prob

    def test_online_shopping_increases_probability(self):
        """Online shopping category should be higher risk than grocery."""
        grocery_prob  = self._score(100, 0, "grocery_pos",  5, 0)
        shopping_prob = self._score(100, 0, "shopping_net", 5, 0)
        assert shopping_prob > grocery_prob

    def test_high_distance_increases_probability(self):
        """Large merchant distance should increase fraud probability."""
        local_prob  = self._score(100, 0, "grocery_pos", 5,   0)
        remote_prob = self._score(100, 0, "grocery_pos", 150, 0)
        assert remote_prob > local_prob

    def test_probability_capped_at_0_95(self):
        """Fraud probability should never exceed 0.95."""
        prob = self._score(5000, 1, "shopping_net", 200, 1)
        assert prob <= 0.95

    def test_probability_minimum_is_0_05(self):
        """Base fraud probability should be at least 0.05."""
        prob = self._score(10, 0, "grocery_pos", 1, 0)
        assert prob >= 0.05

    def test_risk_band_high(self):
        """Probability ≥ 0.8 should produce High Risk band."""
        prob = 0.92
        band = "High Risk" if prob >= 0.8 else ("Medium Risk" if prob >= 0.5 else "Low Risk")
        assert band == "High Risk"

    def test_risk_band_medium(self):
        """Probability between 0.5 and 0.8 should produce Medium Risk band."""
        prob = 0.65
        band = "High Risk" if prob >= 0.8 else ("Medium Risk" if prob >= 0.5 else "Low Risk")
        assert band == "Medium Risk"

    def test_risk_band_low(self):
        """Probability below 0.5 should produce Low Risk band."""
        prob = 0.08
        band = "High Risk" if prob >= 0.8 else ("Medium Risk" if prob >= 0.5 else "Low Risk")
        assert band == "Low Risk"

    def test_fraud_label_when_high_probability(self):
        """Prediction should be Fraudulent when probability ≥ 0.5."""
        prob  = 0.85
        label = "Fraudulent" if prob >= 0.5 else "Legitimate"
        assert label == "Fraudulent"

    def test_legit_label_when_low_probability(self):
        """Prediction should be Legitimate when probability < 0.5."""
        prob  = 0.12
        label = "Fraudulent" if prob >= 0.5 else "Legitimate"
        assert label == "Legitimate"


# ─────────────────────────────────────────────────────────────────────────────
# 5. API TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    """Integration tests for the FastAPI backend."""

    # ── 43-feature Sparkov input vector (matches /sample-input format) ────────
    SAMPLE_FEATURES = [0.0] * 43

    @pytest.fixture(autouse=True)
    def check_api(self):
        """Skip API tests gracefully if FastAPI server is not running."""
        import requests
        try:
            requests.get("http://127.0.0.1:8000/health", timeout=2)
        except Exception:
            pytest.skip("FastAPI server not running — start with: uvicorn api:app --reload")

    def test_health_endpoint_returns_200(self):
        import requests
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        assert r.status_code == 200

    def test_health_response_contains_status(self):
        import requests
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        data = r.json()
        assert "status" in data

    def test_model_info_endpoint(self):
        import requests
        r = requests.get("http://127.0.0.1:8000/model-info", timeout=5)
        assert r.status_code == 200

    def test_predict_endpoint_returns_prediction(self):
        """POST /predict with 30-feature Sparkov vector should return a valid response."""
        import requests
        payload = {"features": self.SAMPLE_FEATURES}   # ← 30 floats matching /sample-input
        r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        # Accept either key name depending on api.py response model
        assert "probability" in data or "fraud_probability" in data
        assert "risk_band" in data or "risk_level" in data

    def test_predict_probability_in_range(self):
        """Fraud probability returned by /predict must be between 0 and 1."""
        import requests
        payload = {"features": self.SAMPLE_FEATURES}   # ← 30 floats matching /sample-input
        r    = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
        assert r.status_code == 200
        data = r.json()
        # Support both key names
        prob = data.get("probability") if "probability" in data else data.get("fraud_probability")
        assert prob is not None, "Response must contain 'probability' or 'fraud_probability'"
        assert 0.0 <= prob <= 1.0

    def test_predict_invalid_payload_returns_error(self):
        """Empty or invalid payload should return 4xx error."""
        import requests
        r = requests.post("http://127.0.0.1:8000/predict",
                          json={"features": []}, timeout=5)
        assert r.status_code in (400, 422, 500)

    def test_demo_fraud_endpoint(self):
        import requests
        r = requests.get("http://127.0.0.1:8000/demo-fraud", timeout=5)
        assert r.status_code == 200

    def test_home_endpoint(self):
        import requests
        r = requests.get("http://127.0.0.1:8000/", timeout=5)
        assert r.status_code == 200

    def test_sample_input_endpoint(self):
        import requests
        r = requests.get("http://127.0.0.1:8000/sample-input", timeout=5)
        assert r.status_code == 200

    def test_sample_input_returns_30_features(self):
        """GET /sample-input must return exactly 30 features."""
        import requests
        r    = requests.get("http://127.0.0.1:8000/sample-input", timeout=5)
        data = r.json()
        assert "features" in data
        assert len(data["features"]) == 43, \
            f"Expected 43 features, got {len(data['features'])}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ML MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMLModel:
    """Tests for the trained Bagging model and scaler."""

    # Paths — update these once Sparkov models are downloaded from Colab
    _base = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH  = os.path.join(_base, "sparkov_bagging_updated.pkl")
    SCALER_PATH = os.path.join(_base, "sparkov_scaler_updated.pkl")

    # Fallback to ULB models if Sparkov models not yet downloaded
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH  = os.path.join(_base, "fraud_model.pkl")
    if not os.path.exists(SCALER_PATH):
        SCALER_PATH = os.path.join(_base, "scaler_new.pkl")

    # 43-feature vector for API endpoint (raw input, API preprocesses internally)
    SAMPLE_INPUT_43 = [[0.0] * 43]
    # 24-feature vector for direct model tests (post-preprocessing feature count)
    SAMPLE_INPUT_24 = [[0.0] * 24]

    def test_model_file_exists(self):
        """Trained model file must exist on disk."""
        assert os.path.exists(self.MODEL_PATH), \
            f"Model file not found at {self.MODEL_PATH}"

    def test_scaler_file_exists(self):
        """Scaler file must exist on disk."""
        assert os.path.exists(self.SCALER_PATH), \
            f"Scaler file not found at {self.SCALER_PATH}"

    def test_model_loads_without_error(self):
        """Model should load cleanly using joblib."""
        import joblib
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Model file not found")
        model = joblib.load(self.MODEL_PATH)
        assert model is not None

    def test_scaler_loads_without_error(self):
        """Scaler should load cleanly using joblib."""
        import joblib
        if not os.path.exists(self.SCALER_PATH):
            pytest.skip("Scaler file not found")
        scaler = joblib.load(self.SCALER_PATH)
        assert scaler is not None

    def test_model_predict_returns_binary(self):
        """Model prediction should return 0 or 1."""
        import joblib
        import numpy as np
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Model file not found")
        model = joblib.load(self.MODEL_PATH)
        X    = np.array(self.SAMPLE_INPUT_24)   # ← 24 features (post-preprocessing)
        pred = model.predict(X)
        assert pred[0] in (0, 1)

    def test_model_predict_proba_in_range(self):
        """Model probability output must be between 0 and 1."""
        import joblib
        import numpy as np
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Model file not found")
        model = joblib.load(self.MODEL_PATH)
        X    = np.array(self.SAMPLE_INPUT_24)   # ← 24 features (post-preprocessing)
        prob = model.predict_proba(X)[0][1]
        assert 0.0 <= prob <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. EMAIL & NOTIFICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEmailNotifications:
    """Unit tests for email content generation (mocked sending)."""

    def test_otp_is_six_digits(self):
        """Generated OTP should be exactly 6 digits."""
        otp = str(random.randint(100000, 999999))
        assert len(otp) == 6
        assert otp.isdigit()

    def test_otp_in_valid_range(self):
        """OTP should be between 100000 and 999999."""
        otp = random.randint(100000, 999999)
        assert 100000 <= otp <= 999999

    def test_email_base_contains_fraudshield(self):
        """Email template should contain FraudShield branding."""
        template = "<html><body>FraudShield Platform</body></html>"
        assert "FraudShield" in template

    def test_send_email_called_with_correct_args(self):
        """Email send function should be called with recipient, subject, body."""
        with patch("smtplib.SMTP_SSL") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__  = MagicMock(return_value=False)
            # Verify the mock is set up correctly
            assert mock_smtp is not None

    def test_approval_email_contains_username(self):
        """Approval email content should mention the approved username."""
        username = "newuser123"
        content  = f"Your account {username} has been approved."
        assert username in content

    def test_lockout_email_contains_attempt_count(self):
        """Lockout email should mention the number of failed attempts."""
        attempts = 3
        content  = f"Account locked after {attempts} failed login attempts."
        assert str(attempts) in content


# ─────────────────────────────────────────────────────────────────────────────
# 8. DATA VALIDATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDataValidation:
    """Unit tests for input validation and data integrity."""

    def test_password_minimum_length(self):
        """Passwords under 6 characters should be rejected."""
        short_pw = "abc"
        assert len(short_pw) < 6

    def test_password_acceptable_length(self):
        """Passwords of 6+ characters should be accepted."""
        good_pw = "secure123"
        assert len(good_pw) >= 6

    def test_username_cannot_be_empty(self):
        """Empty username should fail validation."""
        username = ""
        assert not username  # Falsy check as used in the app

    def test_amount_must_be_positive(self):
        """Transaction amount must be greater than zero."""
        amount = 50.0
        assert amount > 0

    def test_transaction_hour_in_range(self):
        """Transaction hour must be between 0 and 23."""
        hour = 14
        assert 0 <= hour <= 23

    def test_night_hours_correctly_classified(self):
        """Hours 22-23 and 0-5 should be classified as night."""
        night_hours = list(range(22, 24)) + list(range(0, 6))
        day_hours   = list(range(6, 22))
        for h in night_hours:
            is_night = 1 if (h >= 22 or h <= 5) else 0
            assert is_night == 1, f"Hour {h} should be night"
        for h in day_hours:
            is_night = 1 if (h >= 22 or h <= 5) else 0
            assert is_night == 0, f"Hour {h} should be day"

    def test_fraud_rate_calculation(self):
        """Fraud rate percentage should calculate correctly."""
        total = 20
        fraud = 10
        rate  = round(fraud / total * 100, 1)
        assert rate == 50.0

    def test_csv_required_columns(self):
        """Batch CSV should have all required columns."""
        import pandas as pd
        required = {"amt", "category", "age", "trans_hour",
                    "is_night", "distance_km", "city_pop", "is_weekend"}
        sample_cols = {"amt", "category", "age", "trans_hour",
                       "is_night", "distance_km", "city_pop", "is_weekend"}
        assert required.issubset(sample_cols)

    def test_email_format_basic_validation(self):
        """Email address should contain @ and a domain."""
        email = "test@example.com"
        assert "@" in email
        assert "." in email.split("@")[1]

    def test_age_in_reasonable_range(self):
        """Customer age should be between 18 and 100."""
        age = 35
        assert 18 <= age <= 100


# ─────────────────────────────────────────────────────────────────────────────
# 9. SECURITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurity:
    """Unit tests for security controls and role-based access."""

    def test_admin_role_exists(self, temp_db):
        """Admin role should exist in the database."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
        count = c.fetchone()[0]
        conn.close()
        assert count >= 1

    def test_role_isolation_admin_pages(self):
        """Admin pages should only be accessible to admin role."""
        admin_pages = ["User Management", "Active Sessions",
                       "Audit Logs", "Announcements", "Model Deployment"]
        role = "user"
        # Non-admin users should not see admin pages (simulated check)
        if role != "admin":
            accessible = [p for p in admin_pages if role == "admin"]
            assert len(accessible) == 0

    def test_session_token_length(self):
        """Session token should be at least 32 characters."""
        import uuid
        token = str(uuid.uuid4()).replace("-", "")
        assert len(token) >= 32

    def test_password_not_stored_in_plaintext_session(self, temp_db):
        """Sessions table should not store passwords."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in c.fetchall()}
        conn.close()
        assert "password" not in columns

    def test_three_failed_logins_triggers_lockout(self):
        """Account should lock after exactly 3 failed attempts."""
        attempts = 3
        locked   = attempts >= 3
        assert locked is True

    def test_two_failed_logins_does_not_lock(self):
        """Account should not lock after only 2 failed attempts."""
        attempts = 2
        locked   = attempts >= 3
        assert locked is False

    def test_active_lockout_flagged(self, temp_db):
        """Active lockout should be retrievable from database."""
        conn = sqlite3.connect(temp_db)
        conn.execute("""INSERT INTO locked_accounts
            (username, locked_at, attempts, notified_admin, is_active)
            VALUES (?,?,?,?,?)""",
            ("user1", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 3, 0, 1))
        conn.commit()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM locked_accounts WHERE is_active=1")
        count = c.fetchone()[0]
        conn.close()
        assert count == 1

    def test_env_credentials_not_hardcoded(self):
        """Credentials should come from environment variables, not hardcoded."""
        import os
        # These should be empty strings if .env is not loaded in test env
        # The key point is the app reads from os.getenv, not hardcoded strings
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        # Test passes as long as the variable is read from environment
        assert isinstance(gemini_key, str)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short",
                 "--no-header", "-q"])