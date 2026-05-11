# 🛡️ FraudShield — LLM-Assisted Credit Card Fraud Detection Platform

> **Course:** ICT946 Capstone Project | **Institution:** Crown Institute of Higher Education (CIHE), Australia
> **Student:** Prashan Manandhar | **Dataset:** Sparkov Credit Card Transactions (1.3M records)

---

## 📌 Project Overview

FraudShield is a production-grade AI-powered credit card fraud detection platform combining advanced machine learning ensemble models with Google Gemini 1.5 Flash for real-time fraud classification and human-understandable explanations.

**Key achievements:**
- 🎯 **ROC-AUC 0.9926** on 1.3M real transactions
- 🔍 **96% fraud recall** after synthetic data augmentation
- 🤖 **Real-time Gemini AI explanations** for every prediction
- 🔐 **Google Authenticator TOTP MFA** for all users
- 👥 **Three role-based dashboards** (Admin / Researcher / End User)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       FraudShield Platform                          │
├──────────────────┬─────────────────────────┬───────────────────────┤
│   Layer 1        │      Layer 2            │      Layer 3          │
│   Intelligence   │      Application        │      Integration      │
├──────────────────┼─────────────────────────┼───────────────────────┤
│ Sparkov Dataset  │ FastAPI Backend         │ Google Gemini AI      │
│ Feature Eng.     │ Streamlit Frontend      │ SQLite Database       │
│ SMOTE Balancing  │ Role-Based Access       │ Gmail Email Alerts    │
│ Bagging Model    │ TOTP MFA (Google Auth)  │ Session Persistence   │
│ Ensemble Comp.   │ User Management         │ Audit Logging         │
└──────────────────┴─────────────────────────┴───────────────────────┘
```

---

## 📈 Weekly Development Progress

### Week 1 — Environment Setup & Project Initialisation
- Defined project scope and objectives
- Configured Python environment using Anaconda (`conda env: fraud-ml`)
- Loaded the ULB Credit Card Fraud Detection dataset from Kaggle
- Initialised GitHub repository

### Week 2 — Exploratory Data Analysis
- Identified severe class imbalance — only **0.17% fraud rate** in ULB dataset
- Visualised transaction patterns using histograms, boxplots, and correlation heatmaps

### Week 3 — Data Preprocessing
- Applied **StandardScaler**, stratified **70/15/15 split**, and **SMOTE** on training data only
- Trained initial Logistic Regression baseline

### Week 4 — Baseline Model Development

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.61 | 0.72 | 0.66 | 0.912 |
| Random Forest | 0.87 | 0.82 | 0.85 | 0.905 |
| SVM | 0.52 | 0.68 | 0.59 | 0.880 |
| Isolation Forest | 0.45 | 0.67 | 0.54 | 0.861 |

### Week 5 — Feature Engineering
- Engineered 13 real-world proxy features — dataset expanded from 30 to **43 features**
- Enhanced Random Forest ROC-AUC improved to **0.912**

### Week 6 — Advanced Models & Deep Learning

| Model | Type | ROC-AUC |
|---|---|---|
| Enhanced Random Forest | Supervised (K-Fold) | 0.912 |
| Isolation Forest | One-class anomaly | 0.883 |
| Local Outlier Factor | Density-based | 0.871 |
| MLP Neural Network | Deep learning | 0.965 |
| Autoencoder | Unsupervised | ~0.960 |
| LSTM | Sequential DL | ~0.520 |

### Week 7 — FastAPI Backend Design

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | API health check |
| `/model-info` | GET | Deployed model information |
| `/predict` | POST | Real-time fraud prediction |
| `/demo-fraud` | GET | Demonstration endpoint |

### Week 8 — Sparkov Dataset, Synthetic Fraud & Ensemble Comparison

#### Dataset Upgrade

| Property | ULB Dataset | Sparkov Dataset |
|---|---|---|
| Transactions | 284,807 | 1,296,675 |
| Fraud Rate | 0.17% | 0.58% |
| Features | Anonymised V1–V28 | Real-world (merchant, category, age, location) |
| Explainability | ❌ | ✅ |

#### Ensemble Comparison

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.66 | 0.88 | 0.75 | 0.9943 |
| **Bagging ✓ Selected** | **0.79** | **0.85** | **0.82** | **0.9777** |
| Gradient Boosting | 0.18 | 0.92 | 0.29 | 0.9908 |
| Stacking | 0.26 | 0.94 | 0.41 | 0.9948 |

**Bagging selected** — best practical balance between precision and recall.

#### Impact of 500 Synthetic Fraud Transactions

| Metric | Original | After Synthetic |
|---|---|---|
| Recall | 0.85 | **0.96** |
| Missed Fraud | 174 cases | **49 cases** |
| ROC-AUC | 0.9777 | **0.9926** |

#### Top Feature Importance (Bagging)

| Rank | Feature | Importance | Meaning |
|---|---|---|---|
| 1 | `amt` | 0.562 | Transaction amount — strongest signal |
| 2 | `is_night` | 0.110 | Night transactions are significantly riskier |
| 3 | `category` | 0.089 | Merchant type influences fraud |
| 4 | `amt_log` | 0.084 | Log-transformed amount |
| 5 | `amt_to_category_avg` | 0.081 | Contextual spending comparison |

### Week 9 — Platform Stabilisation & Security Hardening

- ✅ Replaced email OTP with **Google Authenticator TOTP** (pyotp + QR code setup)
- ✅ Fixed all `StreamlitAPIException` / `StreamlitDuplicateElementKey` navigation errors
- ✅ Account lockout after 3 failed attempts with admin email alert
- ✅ Admin **MFA Reset** feature in User Management
- ✅ **Remember Me** checkbox on login page
- ✅ Redesigned batch upload **AI Fraud Investigation Report** (two-panel cards)
- ✅ Deep AI analysis expander on single transaction fraud results
- ✅ Fixed password reset demo code showing when email already sent
- ✅ Removed 3 dead code blocks (110+ lines) and duplicate page content
- ✅ Fixed `yaxis=` conflict with `**CHART_LAYOUT` in Plotly charts
- ✅ All navigation working correctly across all three roles

---

## 🌐 Platform Features

### 🔴 Admin Role

| Feature | Description |
|---|---|
| Dashboard | User activity, fraud rate, session count, pending approvals |
| User Management | Full CRUD — create, activate, deactivate, delete users |
| Account Approvals | Review and approve or reject self-registration requests |
| MFA Reset | Reset Google Authenticator for any user |
| Active Sessions | View all logged-in users, force-logout any session |
| Audit Logs | Every action timestamped and stored in SQLite |
| System Analytics | Prediction charts, fraud by category, platform overview |
| Announcements | Post platform-wide notices visible on all dashboards |
| Password Reset | Reset any user's password with immediate effect |

### 🟡 Researcher Role

| Feature | Description |
|---|---|
| Dashboard | Model KPIs, ensemble comparison chart, feature importance, dataset stats |
| Model Training | Configure and simulate training with visual results per model |
| Training Visuals | Metrics bar chart, ROC-AUC gauge, confusion matrix, learning curve |
| Test CSV Upload | Upload labelled or unlabelled CSV — get full accuracy and recall metrics |
| Model Evaluation | Comparison table and grouped bar chart for all four ensemble models |
| ROC & PR Curves | ROC curves, Precision-Recall curves, threshold analysis slider |
| Model Radar | Spider chart comparing all models across 6 dimensions simultaneously |
| Feature Analysis | Horizontal importance chart with plain English feature descriptions |
| Export Results | Download all predictions from database as CSV |

### 🟢 End User Role

| Feature | Description |
|---|---|
| Dashboard | Personal stats, last transaction result, quick-action cards, fraud tips |
| Single Transaction | Real-world fields — amount, category, age, hour, distance, city population |
| Gemini AI Explanation | Natural language explanation for every prediction |
| Deep Analysis | Expandable 3-section fraud investigation (why flagged, patterns, actions) |
| Batch CSV Upload | Upload CSV, score all rows, visual cards, risk donut, category chart |
| AI Batch Report | Per-transaction two-panel AI cards (Why Flagged + What To Do) |
| My History | Fraud rate trend, category breakdown, risk distribution, full log |

---

## 🔒 Security Features

| Feature | Implementation |
|---|---|
| Two-Factor Authentication | **Google Authenticator TOTP** (pyotp) — QR code on first login |
| Account Lockout | Locked after 3 failed attempts — admin notified by email |
| Admin MFA Reset | Admin resets TOTP — user re-scans QR on next login |
| Session Persistence | SQLite token with 8-hour expiry — survives page refresh |
| Role-Based Access Control | Each role sees completely different navigation and pages |
| Audit Logging | Every action timestamped and persisted in SQLite |
| Self-Registration Approval | New users request access — admin approves or rejects |
| Email Notifications | Approval, rejection, password reset OTP, lockout alert |
| Password Reset | 3-step OTP-verified flow — demo code hidden when email sent |
| Force Logout | Admin can terminate any active session instantly |

---

## 🤖 Google Gemini AI Integration

Every prediction triggers a real call to **Google Gemini 1.5 Flash**:

- **Single transaction:** 2–3 sentence professional explanation
- **Deep analysis (fraud only):** 3-section investigation — why flagged, what's suspicious, what to do
- **Batch analysis:** Per-transaction two-panel cards — 🚨 Why Flagged + ✅ What To Do

---

## 🗄️ Database Schema (SQLite — `fraudshield.db`)

| Table | Contents |
|---|---|
| `users` | username, password, role, email, status, totp_secret, totp_enabled |
| `predictions` | Every fraud check with full Gemini explanation stored |
| `audit_logs` | Full timestamped activity log |
| `sessions` | Active login tokens with 8-hour expiry |
| `locked_accounts` | Lockout events with admin notification status |

---

## 📧 Email Notifications

| Event | Recipient |
|---|---|
| New registration request | Admin — immediate alert |
| Account approved | New user — login instructions |
| Account rejected | Applicant — professional decline |
| Password reset OTP | User — 6-digit verification code |
| Account locked | Admin — security alert with details |

---

## 📊 EDA Key Findings (Sparkov Dataset)

| Finding | Detail |
|---|---|
| Night transactions | 16× higher fraud rate than daytime |
| Highest risk category | `shopping_net` (online shopping) |
| Highest risk age group | Senior customers (>50) |
| Highest risk day | Friday |
| Highest risk state | Delaware (DE) |

---

## 🔧 Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| ML Framework | Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Charts | Plotly |
| AI Explanation | Google Gemini 1.5 Flash |
| MFA | pyotp + qrcode + Google Authenticator |
| Database | SQLite |
| Email | Gmail SMTP |
| Environment | Anaconda |
| Version Control | GitHub |

---

## 📁 Project Structure

```
Creditcard-fraud-detection-using-ML/
│
├── streamlit_app.py                      # FraudShield platform (main UI)
├── api.py                                # FastAPI backend
├── requirements.txt                      # Python dependencies
├── .env                                  # Credentials (gitignored)
├── README.md                             # This file
│
├── models/
│   ├── sparkov_bagging_updated.pkl       # Final Bagging model
│   ├── sparkov_scaler_updated.pkl        # Scaler for Sparkov features
│   ├── fraud_model.pkl                   # ULB Enhanced RF (archived)
│   └── scaler_new.pkl                    # ULB scaler (archived)
│
├── notebooks/
│   ├── week8_sparkov.ipynb               # Week 8 Sparkov training
│   ├── creditcardfraud.ipynb             # Weeks 1–7 ULB notebook
│   └── model.ipynb                       # Advanced models notebook
│
├── data/
│   └── fraudTrain_updated_synthetic.csv  # Sparkov + 500 synthetic fraud
│
└── fraudshield.db                        # SQLite database (gitignored)
```

---

## 🚀 Running the Application

### Prerequisites
```bash
conda activate fraud-ml
pip install -r requirements.txt
```

### Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key
EMAIL_SENDER=your_gmail@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
ADMIN_EMAIL=admin@email.com
```

### Start the API (Terminal 1)
```bash
cd C:\Users\Prashan\Creditcard-fraud-detection-using-ML
python -m uvicorn api:app --reload
```
API running at: `http://127.0.0.1:8000`

### Start the Platform (Terminal 2)
```bash
python -m streamlit run streamlit_app.py
```
Platform running at: `http://localhost:8501`

> **Note:** Delete `fraudshield.db` and restart if you see a schema error — the database will be recreated automatically with the correct schema.

---

## 👤 Demo Accounts

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin — full platform control |
| `researcher` | `research123` | Researcher — model training and evaluation |
| `user1` | `user123` | End User — transaction analysis |

> All accounts require **Google Authenticator TOTP** after login. Scan the QR code on first login and save it in your authenticator app.

---

## ✅ Supervisor Requirements Status

| Requirement | Status |
|---|---|
| Real-world dataset with interpretable features | ✅ Sparkov dataset |
| Class imbalance handling | ✅ SMOTE applied |
| 500 synthetic fraud transactions | ✅ Completed |
| Bagging, Boosting, Stacking, RF comparison | ✅ All four compared |
| K-fold cross-validation | ✅ Implemented |
| Isolation Forest (one-class) | ✅ Implemented |
| Local Outlier Factor | ✅ Implemented |
| MLP Neural Network | ✅ Implemented |
| Autoencoder | ✅ Implemented |
| LSTM | ✅ Implemented |
| FastAPI backend | ✅ Running |
| GUI with role-based access | ✅ Three roles |
| LLM integration | ✅ Google Gemini 1.5 Flash |
| 2FA security | ✅ Google Authenticator TOTP |
| Audit logging | ✅ SQLite persistent |
| User management | ✅ Full CRUD with DB |
| Email notifications | ✅ Gmail SMTP |
| Session persistence on refresh | ✅ Token-based (8hr) |
| Password reset | ✅ Email OTP flow |
| Cloud deployment | 🔄 Planned |

---

## 📌 Dataset Citation

**Sparkov Credit Card Transactions Fraud Detection Dataset**
- **Author:** Kartik Shenoy (generated using Sparkov simulation tool)
- **Source:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Period:** January 2019 – December 2020
- **Records:** 1,852,394 total (1,296,675 training used)
- **License:** Public domain