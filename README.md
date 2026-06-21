# 🏦 SmartLoan AI — Credit Risk Prediction & Loan Approval System

An AI-powered loan approval system that predicts whether a customer is likely to default on a loan using Machine Learning.

The project compares **Logistic Regression**, **Random Forest**, and **XGBoost**, selects the best model automatically, and provides an interactive Streamlit dashboard with risk analysis and explainability.

---

##  Live Demo

Add your Streamlit link here:

`[https://your-app.streamlit.app](https://smartloan-ai-2jftvhogitmxjjbyauduqz.streamlit.app/)`

---

## 📊 Dataset

Dataset used:

**Kaggle Credit Risk Dataset**

https://www.kaggle.com/datasets/laotse/credit-risk-dataset

### Dataset Information

* Total Records: ~32,000
* Target Variable: `loan_status`

  * `0` → Loan Repaid
  * `1` → Loan Defaulted

### Features Used

* Person Age
* Annual Income
* Employment Length
* Loan Amount
* Interest Rate
* Credit History Length
* Loan Grade (encoded)
* Loan-to-Income Ratio
* Income per Employment Year
* Interest × Loan Amount Feature

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Streamlit
* Pickle

---

## 🧠 Machine Learning Pipeline

### 1. Data Cleaning

* Removed unrealistic ages
* Removed unrealistic employment years
* Removed extreme income values
* Filled missing values using median

### 2. Feature Engineering

Created new features:

* Loan-to-Income Ratio

```python
loan_to_income = loan_amount / income
```

* Income per Employment Year

```python
income_per_year = income / (employment_length + 1)
```

* Interest × Loan Amount

```python
rate_x_loan = interest_rate * loan_amount / 1000
```

### 3. Data Preprocessing

* Train-Test Split (80/20)
* Stratified Sampling
* StandardScaler normalization

### 4. Models Trained

1. Logistic Regression
2. Random Forest
3. XGBoost

Models are compared using:

* AUC-ROC
* Accuracy
* Precision
* Recall
* F1 Score

The model with the highest **AUC-ROC** is automatically selected as the final model.

---

## 📈 Model Performance

| Model               | AUC-ROC | Accuracy | F1 Score |
| ------------------- | ------: | -------: | -------: |
| Logistic Regression |  0.8519 |   0.8420 |   0.5513 |
| Random Forest       |  0.8970 |   0.8767 |   0.6877 |
| XGBoost          |  0.9138 |   0.8835 |   0.7094 |

###  Best Model

**XGBoost**

* AUC-ROC: **0.9138**
* Accuracy: **88.35%**
* F1 Score: **0.7094**

---

## 🔍 Explainable AI with SHAP

This project uses **SHAP (SHapley Additive Explanations)** to explain:

* Which features increase default risk
* Which features reduce default risk
* Why the model approved or rejected a loan

Positive SHAP values:

🔴 Increase default risk

Negative SHAP values:

🟢 Reduce default risk

---

## 🎯 Features

✅ Loan approval prediction

✅ Risk level classification

✅ Approval probability score

✅ SHAP explainability

✅ Improvement suggestions for rejected applicants

✅ Comparison of 3 ML models

✅ Interactive Streamlit dashboard

---

## 📷 Screenshots

### Home Page

Add screenshot here: 


```text
<img width="887" height="852" alt="image" src="https://github.com/user-attachments/assets/cf6c2aab-b927-44e6-a1da-a282aa60648e" />

```

### Prediction Result

```text
<img width="955" height="367" alt="image" src="https://github.com/user-attachments/assets/88a9cfee-226c-40ac-9493-5cc5ced2b123" />

```

### SHAP Explanation

```text
<img width="917" height="775" alt="image" src="https://github.com/user-attachments/assets/303742e3-7d6b-45d6-a87d-a3986067c1d0" />

```

### Model Comparison

```text
<img width="780" height="686" alt="image" src="https://github.com/user-attachments/assets/fdeb8ed9-944b-45b2-adfb-2b039a5aaeda" />

```

---

## 📂 Project Structure

```text
SmartLoan-AI/

│── app.py
│── train_model.py
│── credit_risk_dataset.csv
│── model_results.csv

│── saved_models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl

│── model_comparison.png
│── roc_comparison.png

│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/SmartLoan-AI.git

cd SmartLoan-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

** David**

If you like this project, give it a ⭐ on GitHub.
