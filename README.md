# 💳 Credit Card Fraud Detection API (Machine Learning Project)

## 📌 Project Overview

This project develops a machine learning-based fraud detection system and extends it into a realistic application using a FastAPI service. The system detects fraudulent credit card transactions using supervised learning techniques and is designed to simulate a real-world financial fraud detection pipeline.

The project evolves from data preprocessing and model development to a deployable API capable of scoring transactions in real time. It also demonstrates a full machine learning lifecycle, progressing from data analysis to a production-ready fraud detection system.

---

## 📈 Weekly Progress Overview

This project was developed progressively across multiple weeks, evolving from basic data analysis to a real-world deployable fraud detection system.

### Week 1 – Problem Understanding & Setup
- Defined project scope and objectives
- Set up development environment using Python, Jupyter, and GitHub
- Loaded and explored the dataset structure
- Organized project folder structure for machine learning development

### Week 2 – Exploratory Data Analysis (EDA)
- Performed summary statistics and data understanding
- Identified severe class imbalance in the fraud dataset
- Visualized transaction patterns using histograms, boxplots, and correlation heatmaps
- Checked data quality issues such as missing values and duplicates

### Week 3 – Data Preprocessing
- Handled duplicate and inconsistent records
- Detected and treated outliers using statistical methods
- Applied feature scaling using StandardScaler
- Split the dataset into training, validation, and test sets
- Applied SMOTE to address class imbalance
- Trained an initial Logistic Regression model

### Week 4 – Model Development
- Implemented multiple baseline models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Isolation Forest
- Evaluated models using:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - ROC-AUC
- Identified Random Forest as the strongest baseline model
- Performed feature importance analysis

### Week 5 – Model Optimization & Real-World Enhancement
- Applied hyperparameter tuning using RandomizedSearchCV
- Engineered real-world inspired fraud features such as transaction timing and abnormal spending indicators
- Implemented Gradient Boosting for additional comparison
- Compared model performance and selected the tuned Random Forest model
- Improved real-world applicability of the model using behavioural features

### Final Stage – Real-World Application
- Designed a FastAPI-based fraud detection API
- Enabled real-time transaction scoring workflow
- Structured the project as a practical deployment-ready system

---

## 🎯 Objectives

- Detect fraudulent credit card transactions using machine learning
- Handle imbalanced datasets effectively
- Improve fraud detection using real-world feature engineering
- Compare multiple machine learning models and select the best performer
- Deploy the trained model as an API for real-time inference
- Simulate a realistic fraud detection system for banking or fintech use cases

---

## 🧠 Machine Learning Approach

### Models Implemented
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Isolation Forest
- Gradient Boosting

### Final Model Selection
**Tuned Random Forest** was selected as the final model due to:
- strong balance between precision and recall
- high F1-score for fraud detection
- robust performance on imbalanced data
- better real-world applicability compared to other models
- feature importance interpretability

---

## ⚙️ Feature Engineering (Real-World Inspired)

To simulate realistic fraud detection scenarios, additional features were created:

- `transaction_hour` → captures time-based patterns
- `is_night_transaction` → detects unusual night activity
- `is_high_amount` → flags high-value transactions
- `amount_to_median_ratio` → deviation from normal spending
- `amount_deviation_from_mean` → abnormal behaviour detection
- `amount_log` → reduces skewness in transaction amount
- `risk_amount_flag` → identifies top 1% high-risk transactions
- `amount_percentile` → measures relative transaction ranking
- `amount_spike` → detects sudden spending changes
- `high_risk_combo` → combines multiple fraud indicators
- `amount_squared` → captures extreme value influence
- `log_time` → improves time-based modelling
- `time_amount_interaction` → behavioural interaction feature

These features improve the model’s ability to capture real-world fraud patterns and make the system more practical for deployment.

---

## 📊 Model Performance

### Final Tuned Random Forest
- Precision (Fraud): ~0.87–0.90
- Recall (Fraud): ~0.81–0.82
- F1-score (Fraud): ~0.85
- ROC-AUC: ~0.91+

### Gradient Boosting
- Higher fraud recall (~0.85)
- Significantly lower fraud precision (~0.11)
- Less suitable due to high false positives

### SVM
- Weak performance on the large imbalanced dataset
- Convergence and scalability limitations

### Logistic Regression
- Useful baseline model
- Lower ability to capture complex fraud patterns compared to Random Forest

---

## 🔧 Data Preprocessing

The dataset was preprocessed to ensure data quality and suitability for model training.

Preprocessing included:
- checking for missing values
- handling duplicate records
- treating outliers using statistical methods
- applying StandardScaler for numerical normalization
- handling class imbalance using SMOTE
- splitting data into training, validation, and test sets

---

## 🔍 Hyperparameter Tuning

RandomizedSearchCV was used to optimize the Random Forest model. The tuning process explored parameters such as:
- number of estimators
- maximum depth
- minimum samples split
- minimum samples leaf
- class weights
- feature selection strategy

This improved model generalization, stability, and fraud detection performance.

---

## 🌍 Real-World System Design

This project extends beyond notebook experimentation into a practical fraud detection architecture.

### Workflow
1. Historical transaction data is used to train the model
2. The trained model is saved as a reusable artifact
3. A FastAPI service loads the model
4. New transactions are sent to the API
5. The API returns:
   - fraud probability
   - predicted label
   - risk band
6. The result can be used by banking systems, fintech platforms, or payment gateways

This structure is suitable for academic demonstration and can be extended into a production deployment later.

---

## 📁 Project Structure

```text
app/
  main.py            # FastAPI application
  model_service.py   # Loads model artifact and performs inference
  schemas.py         # Request/response models
data/
  raw/creditcard.csv # Source dataset
models/              # Saved trained model artifacts
src/
  train_model.py     # Training script that saves the model
main.py              # ASGI entrypoint
README.md
requirements.txt

## Week 7-8 Progress (Advanced Models & Deep Learning)

This stage extends the fraud detection system using anomaly detection and deep learning approaches.

### Models Implemented
- Enhanced Random Forest (with K-Fold Validation)
- Isolation Forest (one-class classification)
- Local Outlier Factor (LOF)
- Multi-Layer Perceptron (MLP)
- Autoencoder (unsupervised anomaly detection)
- LSTM (sequential deep learning model)

### Key Findings
- Autoencoder achieved the highest ROC-AUC (~0.96) but produced more false positives.
- Enhanced Random Forest provided the best balance of precision and recall.
- LSTM underperformed due to lack of sequential structure in the dataset.

### Conclusion
The Enhanced Random Forest model was selected as the final model for deployment due to its robustness and balanced performance.

### Future Work
- Integrate LLM for explainable fraud detection
- Develop API and dashboard interface
- Improve sequential modelling using real customer-level data