# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Due to the highly imbalanced nature of fraud datasets, advanced preprocessing and model evaluation strategies were applied to improve detection performance.

The project was developed as part of a capstone assignment, demonstrating practical implementation of data preprocessing, model development, and evaluation.

---

## 📊 Dataset
- Source: ULB Credit Card Fraud Dataset
- Total Transactions: 284,807
- Features: 30 (including anonymized features V1–V28, Time, Amount)
- Target Variable:
  - `0` → Normal transaction
  - `1` → Fraudulent transaction

---

## ⚙️ Project Workflow

### 🔹 Week 1: Setup & Data Loading
- Environment setup using Anaconda
- Required libraries installed
- Dataset loaded into Pandas DataFrame
- Initial data inspection performed

### 🔹 Week 2: Exploratory Data Analysis (EDA)
- Summary statistics generated
- Class imbalance identified
- Data visualization (histograms, boxplots, correlation heatmap)
- Data quality checks (missing values, duplicates, outliers)

### 🔹 Week 3: Data Preprocessing
- Missing values handled (none found)
- Duplicate records checked and removed
- Outliers treated using IQR and capping
- Feature scaling using StandardScaler
- Class imbalance handled using SMOTE
- Feature engineering (Amount_capped)
- Dataset split into training, validation, and test sets

### 🔹 Week 4: Model Development & Evaluation
- Implemented multiple models:
  - Logistic Regression
  - Random Forest ✅ (Best Model)
  - Support Vector Machine (SVM)
  - Isolation Forest (Anomaly Detection)
- Evaluated models using:
  - Confusion Matrix
  - Precision, Recall, F1-score
- Performed feature importance analysis

---

## 🧠 Model Performance Summary

| Model                | Performance |
|---------------------|------------|
| Random Forest       | ⭐ Best performance (high precision & recall) |
| Logistic Regression | Moderate performance |
| SVM                 | Poor performance (convergence issues) |
| Isolation Forest    | Moderate (useful for anomaly detection) |

---

## 🏆 Final Model Selection

Random Forest was selected as the final model due to its strong ability to:
- Detect fraudulent transactions accurately
- Maintain a balance between precision and recall
- Handle complex patterns in the dataset

---

## 🔍 Key Insights
- The dataset is highly imbalanced (~0.17% fraud cases)
- Feature importance analysis revealed key variables influencing fraud detection
- SMOTE significantly improved model performance
- Ensemble models outperform simpler models for fraud detection

---

## 🚨 Real-World Application

This model can be integrated into real-time financial systems to:
- Monitor transactions continuously
- Flag suspicious transactions instantly
- Trigger alerts for fraud investigation
- Reduce financial losses due to fraud

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## 📁 Project Structure
├── data/
├── notebooks/
├── models/
├── reports/
├── src/
├── eda.py
├── main.py
├── model.ipynb
├── requirements.txt
└── README.md


---

## 📌 Author
**Prashan Manandhar**

---

## 📎 Notes
This project is for academic and learning purposes, demonstrating machine learning techniques for fraud detection.