# 🛡️ FraudShield — LLM-Assisted Credit Card Fraud Detection Platform

> **Course:** ICT946 | **Institution:** Crown Institute of Higher Education (CIHE) Australia
> **Student:** Prashan Manandhar | **Dataset:** Sparkov Credit Card Transactions

---

## 📌 Project Overview

FraudShield is an end-to-end AI-powered credit card fraud detection platform that combines advanced machine learning ensemble models with Google Gemini AI to deliver real-time fraud classification and human-understandable explanations.

The system evolves from exploratory data analysis and model experimentation into a fully deployed multi-role web platform — simulating how a real financial institution would adopt machine learning-based fraud detection in production.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FraudShield Platform                         │
├────────────────┬────────────────────────┬───────────────────────┤
│   Layer 1      │      Layer 2           │      Layer 3          │
│  Intelligence  │    Application         │    Integration        │
├────────────────┼────────────────────────┼───────────────────────┤
│ Sparkov Dataset│ FastAPI Backend        │ Google Gemini AI      │
│ Feature Eng.   │ Streamlit Frontend     │ SQLite Database       │
│ SMOTE Balancing│ Role-Based Access      │ Gmail Email Alerts    │
│ Bagging Model  │ 2FA Authentication     │ Session Persistence   │
│ Ensemble Comp. │ User Management        │ Audit Logging         │
└────────────────┴────────────────────────┴───────────────────────┘
```

---

## 📈 Weekly Development Progress

### Week 1 — Environment Setup & Project Initialisation
- Defined project scope and objectives
- Configured Python environment using Anaconda (conda env: `fraud-ml`)
- Loaded the ULB Credit Card Fraud Detection dataset from Kaggle
- Organised project folder structure for machine learning development
- Initialised GitHub repository: `Creditcard-fraud-detection-using-ML`

---

### Week 2 — Exploratory Data Analysis
- Performed summary statistics and distribution analysis
- Identified severe class imbalance — only **0.17% fraud rate** in ULB dataset
- Visualised transaction patterns using histograms, boxplots, and correlation heatmaps
- Confirmed zero missing values and identified duplicate records

---

### Week 3 — Data Preprocessing
- Removed duplicate records and treated outliers using IQR capping
- Applied **StandardScaler** for feature normalisation
- Implemented **stratified train/validation/test split (70/15/15)**
- Applied **SMOTE** exclusively on training data to address class imbalance
- Trained an initial Logistic Regression baseline model

---

### Week 4 — Baseline Model Development
Implemented and compared four baseline models:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.61 | 0.72 | 0.66 | 0.912 |
| Random Forest | 0.87 | 0.82 | 0.85 | 0.905 |
| SVM | 0.52 | 0.68 | 0.59 | 0.880 |
| Isolation Forest | 0.45 | 0.67 | 0.54 | 0.861 |

- Identified **Random Forest** as the strongest baseline
- Performed hyperparameter tuning using **RandomizedSearchCV**
- Generated feature importance visualisations

---

### Week 5 — Feature Engineering & Model Enhancement
Engineered 13 real-world proxy features from the ULB dataset:

| Feature | Purpose |
|---|---|
| `transaction_hour` | Captures time-based fraud patterns |
| `is_night_transaction` | Detects unusual night activity |
| `is_high_amount` | Flags high-value transactions |
| `amount_to_median_ratio` | Deviation from normal spending |
| `amount_log` | Reduces skewness in amount distribution |
| `amount_spike` | Detects sudden spending changes |
| `high_risk_combo` | Combines multiple fraud indicators |
| `amount_percentile` | Measures relative transaction ranking |

- Dataset expanded from 30 to **43 features**
- Added **Gradient Boosting** for comparison
- Enhanced Random Forest: ROC-AUC improved to **0.912**

---

### Week 6 — Advanced Models & Deep Learning
Extended the system with anomaly detection and deep learning approaches:

| Model | Type | ROC-AUC | Notes |
|---|---|---|---|
| Enhanced Random Forest | Supervised (K-Fold) | 0.912 | Best balance |
| Isolation Forest | One-class anomaly | 0.883 | Normal-only training |
| Local Outlier Factor | Density-based | 0.871 | Novelty detection |
| MLP Neural Network | Deep learning | 0.965 | 64→32 hidden layers |
| Autoencoder | Unsupervised | ~0.960 | 95th percentile threshold |
| LSTM | Sequential DL | ~0.520 | Underperformed (tabular data) |

**Finding:** Autoencoder achieved highest ROC-AUC but more false positives. Enhanced Random Forest selected for deployment.

---

### Week 7 — System Design & API Deployment
Designed the LLM-Assisted Fraud Detection Platform with three user roles:

**FastAPI Backend endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Platform home |
| `/health` | GET | API health check |
| `/model-info` | GET | Deployed model information |
| `/sample-input` | GET | Sample transaction format |
| `/predict` | POST | Real-time fraud prediction |
| `/demo-fraud` | GET | Demonstration endpoint |

The API returns: `prediction label`, `fraud probability`, `risk band`, `recommended action`, and `explanation`.

---

### Week 8 — Sparkov Dataset, Synthetic Fraud & Ensemble Comparison

#### Dataset Upgrade
Switched from ULB (anonymised V1–V28) to **Sparkov Credit Card Transactions** dataset with real-world interpretable features.

| Property | ULB Dataset | Sparkov Dataset |
|---|---|---|
| Transactions | 284,807 | 1,296,675 |
| Fraud Rate | 0.17% | 0.58% |
| Features | Anonymised V1–V28 | Real-world (merchant, category, age, location) |
| Explainability | ❌ | ✅ |

#### Feature Engineering (12 new features)

| Feature | Description |
|---|---|
| `trans_hour` | Hour of transaction |
| `is_night` | Night transaction flag (10pm–5am) |
| `is_weekend` | Weekend flag |
| `age` | Customer age from date of birth |
| `age_group` | Young / Middle-aged / Senior |
| `distance_km` | Haversine distance — customer to merchant |
| `is_high_distance` | Distance above 75th percentile |
| `is_high_amount` | Amount above 95th percentile |
| `amt_to_category_avg` | Amount vs category average ratio |
| `amt_log` | Log-transformed amount |
| `is_low_pop_city` | Low population city flag |
| `trans_day_of_week` | Day of week |

#### Synthetic Fraud Generation
- Generated **500 synthetic fraud transactions** by sampling existing fraud with controlled noise
- Fraud rate increased from **0.578% → 0.617%**
- Dataset saved as `fraudTrain_updated_synthetic.csv`

#### Dataset Links
- **Original:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Updated (with synthetic fraud):** `fraudTrain_updated_synthetic.csv` (Google Drive)

#### Ensemble Method Comparison

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.66 | 0.88 | 0.75 | 0.9943 |
| **Bagging ✓** | **0.79** | **0.85** | **0.82** | **0.9777** |
| Gradient Boosting | 0.18 | 0.92 | 0.29 | 0.9908 |
| Stacking | 0.26 | 0.94 | 0.41 | 0.9948 |

**Bagging selected** as final model — best practical balance between precision and recall.

#### Impact of Synthetic Fraud Augmentation

| Metric | Original | After Synthetic |
|---|---|---|
| Recall | 0.85 | **0.96** |
| Missed Fraud | 174 cases | **49 cases** |
| ROC-AUC | 0.9777 | **0.9926** |

#### Feature Importance (Bagging)

| Rank | Feature | Importance | Meaning |
|---|---|---|---|
| 1 | `amt` | 0.562 | Transaction amount — strongest signal |
| 2 | `is_night` | 0.110 | Night transactions are significantly riskier |
| 3 | `category` | 0.089 | Merchant type influences fraud |
| 4 | `amt_log` | 0.084 | Log-transformed amount |
| 5 | `amt_to_category_avg` | 0.081 | Contextual spending comparison |

---

## 🌐 FraudShield Platform

### Platform Overview

A full-stack web application with a premium dark fintech interface, connecting the FastAPI backend to a Streamlit frontend with three role-specific experiences.

### User Roles

#### 🔴 Admin
| Feature | Description |
|---|---|
| Role-specific Dashboard | User activity, platform fraud rate, session count, pending approvals |
| User Management | Full CRUD — create, activate, deactivate, delete users with live effect |
| Account Approvals | Review and approve or reject self-registration requests |
| Role Assignment | Change any user's role from the user management table |
| Active Sessions | View all logged-in users, force-logout any session instantly |
| Audit Logs | Every action timestamped and stored in SQLite |
| System Analytics | Prediction charts, fraud by category, platform overview |
| Model Deployment | Deployed model table, API endpoint status monitor |
| Announcements | Post platform-wide notices visible to all users on their dashboards |
| Password Reset | Reset any user's password with immediate effect |

#### 🟡 Researcher
| Feature | Description |
|---|---|
| Role-specific Dashboard | Model KPIs, ensemble comparison chart, feature importance, dataset stats |
| Model Training | Configure and simulate training with real per-model visual results |
| Training Visuals | Metrics bar chart, ROC-AUC gauge, confusion matrix heatmap, learning curve |
| Test CSV Upload | Upload labelled or unlabelled CSV — get accuracy, recall, precision, full results |
| Model Evaluation | Comparison table and grouped bar chart for all four ensemble models |
| ROC & PR Curves | ROC curves, Precision-Recall curves, threshold analysis slider |
| Model Radar | Spider chart comparing all models across 6 dimensions simultaneously |
| Feature Analysis | Horizontal importance chart with plain English insights |
| Export Results | Download all predictions from database as CSV |

#### 🟢 End User
| Feature | Description |
|---|---|
| Role-specific Dashboard | Personal stats, last transaction result, clickable quick-action cards, fraud prevention tips |
| Single Transaction | Real-world fields — amount, category, age, hour, distance, city population |
| Fraud Result | Verdict, fraud probability gauge, risk band, recommended action |
| Gemini AI Explanation | Real natural language explanation for every prediction |
| Batch CSV Upload | Upload CSV, score all rows, visual cards, risk donut, category chart, download |
| My History | Personal fraud rate trend, category breakdown, risk distribution bar, full log |

---

## 🔒 Security Features

| Feature | Implementation |
|---|---|
| Two-Factor Authentication | 6-digit OTP sent to registered email via Gmail SMTP |
| Account Lockout | Locked after 3 consecutive failed login attempts |
| Session Persistence | Token stored in SQLite with 8-hour expiry — survives page refresh |
| Role-Based Access Control | Each role sees completely different navigation and pages |
| Audit Logging | Every action timestamped and persisted in SQLite |
| Self-Registration Approval | New users request access — admin reviews and approves or rejects |
| Email Notifications | Approval, rejection, password reset OTP, new registration alert |
| Password Reset | 3-step OTP-verified flow via email |
| Force Logout | Admin can terminate any active session instantly |

---

## 🤖 Google Gemini AI Integration

Every single transaction prediction triggers a real call to **Google Gemini 1.5 Flash**.

The system sends:
- Transaction amount and category
- Fraud probability score
- Risk band
- Detected risk factors (night hours, high amount, distance, age group, category ratio)

Gemini returns a professional natural language explanation such as:

> *"This transaction has been classified as fraudulent. The amount of $1,500 is significantly above typical spending for online shopping, it occurred at 2am which is a high-risk period, and the merchant is 180km from the customer's registered home address. Immediate review is recommended."*

---

## 🗄️ Database (SQLite)

All data persists in `fraudshield.db` across sessions and restarts:

| Table | Contents |
|---|---|
| `users` | All registered users — survives app restarts |
| `predictions` | Every fraud check with full Gemini explanation stored |
| `audit_logs` | Full timestamped activity log |
| `sessions` | Active login tokens with 8-hour expiry |

---

## 📧 Email Notification System

Automated branded HTML emails sent via Gmail SMTP for four events:

| Event | Recipient |
|---|---|
| New registration request | Admin — notified immediately |
| Account approved | New user — receives login instructions |
| Account rejected | Applicant — receives professional decline notice |
| Login 2FA OTP | User — 6-digit code for verification |
| Password reset OTP | User — 6-digit code to verify identity |

---

## 📊 EDA Key Findings (Sparkov Dataset)

| Finding | Detail |
|---|---|
| Night transactions | 16× higher fraud rate than daytime |
| Highest risk category | `shopping_net` (online shopping) |
| Highest risk gender | Male customers |
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
| Database | SQLite |
| Email | Gmail SMTP |
| Environment | Anaconda |
| Version Control | GitHub |

---

## 📁 Project Structure

```
Creditcard-fraud-detection-using-ML/
│
├── streamlit_app.py          # FraudShield platform (main UI)
├── api.py                    # FastAPI backend
├── requirements.txt          # Python dependencies
│
├── models/
│   ├── sparkov_bagging_updated.pkl   # Final Bagging model
│   ├── sparkov_scaler_updated.pkl    # Scaler for Sparkov features
│   ├── fraud_model.pkl               # ULB Enhanced RF (archived)
│   └── scaler_new.pkl                # ULB scaler (archived)
│
├── notebooks/
│   ├── week8_sparkov.ipynb           # Week 8 Sparkov training notebook
│   ├── creditcardfraud.ipynb         # Weeks 1-7 ULB notebook
│   └── model.ipynb                   # Advanced models notebook
│
├── data/
│   └── fraudTrain_updated_synthetic.csv  # Sparkov + 500 synthetic fraud
│
├── fraudshield.db            # SQLite persistent database
└── README.md
```

---

## 🚀 Running the Application

### Prerequisites
```bash
conda activate fraud-ml
pip install -r requirements.txt
```

### Start the API (Terminal 1)
```bash
cd C:\Users\Prashan\Creditcard-fraud-detection-using-ML
python -m uvicorn api:app --reload
```
API running at: http://127.0.0.1:8000

### Start the Platform (Terminal 2)
```bash
python -m streamlit run streamlit_app.py
```
Platform running at: http://localhost:8501

---

## 👤 Demo Accounts

| Username | Password | Role | Access |
|---|---|---|---|
| `admin` | `admin123` | Admin | Full platform control |
| `researcher` | `research123` | Researcher | Model training and evaluation |
| `user1` | `user123` | End User | Transaction analysis |

All accounts require **Two-Factor Authentication** (2FA) after login.

---

## ✅ Supervisor Requirements Status

| Requirement | Status |
|---|---|
| Real-world dataset with interpretable features | ✅ Sparkov dataset |
| Class imbalance handling | ✅ SMOTE applied |
| 500 synthetic fraud transactions | ✅ Completed |
| Bagging, Boosting, Stacking comparison | ✅ All three compared |
| K-fold cross-validation | ✅ Implemented |
| Isolation Forest (one-class) | ✅ Implemented |
| Local Outlier Factor | ✅ Implemented |
| MLP Neural Network | ✅ Implemented |
| Autoencoder | ✅ Implemented |
| LSTM | ✅ Implemented |
| FastAPI backend | ✅ Running |
| GUI with role-based access | ✅ Three roles |
| LLM integration | ✅ Google Gemini 1.5 Flash |
| 2FA security | ✅ Email OTP |
| Audit logging | ✅ SQLite persistent |
| User management | ✅ Full CRUD with DB |
| Email notifications | ✅ Gmail SMTP |
| Session persistence on refresh | ✅ Token-based |
| Password reset | ✅ Email OTP flow |
| Cloud deployment | 🔄 Planned |

---

## 📌 Dataset Citation

Sparkov Credit Card Transactions Fraud Detection Dataset
- **Author:** Kartik Shenoy (generated using Sparkov tool)
- **Source:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Period:** January 2019 – December 2020
- **Records:** 1,852,394 total (1,296,675 training used)
- **License:** Public domain