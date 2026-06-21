# train_model.py
# Trains 3 models, compares them, saves the best one

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              precision_score, recall_score,
                              f1_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("    SMARTLOAN AI — MODEL TRAINING")
print("=" * 55)

# ── Load Data ──
df = pd.read_csv('credit_risk_dataset.csv')
print(f"\nDataset loaded: {df.shape[0]:,} rows")

# ── Clean ──
df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 60]
df = df[df['person_income'] < 4000000]
df['loan_int_rate'].fillna(
    df['loan_int_rate'].median(), inplace=True
)
df['person_emp_length'].fillna(
    df['person_emp_length'].median(), inplace=True
)
print(f"After cleaning: {df.shape[0]:,} rows")

# ── Encode ──
grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
df['grade_num'] = df['loan_grade'].map(grade_map)

# ── Feature Engineering ──
df['loan_to_income']  = df['loan_amnt'] / df['person_income']
df['income_per_year'] = df['person_income'] / (df['person_emp_length'] + 1)
df['rate_x_loan']     = df['loan_int_rate'] * df['loan_amnt'] / 1000

# ── Features ──
feature_cols = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_to_income',
    'cb_person_cred_hist_length',
    'grade_num',
    'income_per_year',
    'rate_x_loan'
]

X = df[feature_cols]
y = df['loan_status']

print(f"Features: {len(feature_cols)}")
print(f"Default rate: {y.mean():.2%}")

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ── Scale ──
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train_sc, columns=feature_cols)
X_test_df  = pd.DataFrame(X_test_sc,  columns=feature_cols)

# ══════════════════════════════════════════
# TRAIN ALL 3 MODELS
# ══════════════════════════════════════════
print("\n" + "=" * 55)
print("    TRAINING 3 MODELS")
print("=" * 55)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

results   = {}
all_preds = {}

print(f"\n{'Model':<22} {'AUC-ROC':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 70)

for name, model in models.items():
    print(f"Training {name}...", end=' ', flush=True)

    # Train
    model.fit(X_train_sc, y_train)

    # Predict
    y_pred      = model.predict(X_test_sc)
    y_pred_prob = model.predict_proba(X_test_sc)[:,1]

    # Metrics
    auc  = roc_auc_score(y_test, y_pred_prob)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {
        'model':     model,
        'auc':       auc,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'y_pred':    y_pred,
        'y_prob':    y_pred_prob
    }
    all_preds[name] = y_pred_prob

    print(f"Done! AUC={auc:.4f} Acc={acc:.4f} "
          f"Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

print("-" * 70)

# ── Find Best Model ──
best_name  = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']

print(f"\nBEST MODEL: {best_name}")
print(f"   AUC-ROC:   {results[best_name]['auc']:.4f}")
print(f"   Accuracy:  {results[best_name]['accuracy']:.4f}")
print(f"   F1 Score:  {results[best_name]['f1']:.4f}")

# ══════════════════════════════════════════
# COMPARISON CHARTS
# ══════════════════════════════════════════
print("\nGenerating comparison charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

model_names  = list(results.keys())
short_names  = ['LR', 'RF', 'XGB']
colors       = ['#3498DB', '#2ECC71', '#E74C3C']

metrics_list = ['auc', 'accuracy', 'f1']
titles_list  = ['AUC-ROC Score\n(Higher = Better)',
                'Accuracy\n(Higher = Better)',
                'F1 Score\n(Higher = Better)']

for idx, (metric, title) in enumerate(zip(metrics_list, titles_list)):
    values = [results[n][metric] for n in model_names]
    bars   = axes[idx].bar(
        short_names, values,
        color=colors, alpha=0.85,
        edgecolor='black', linewidth=0.5,
        width=0.5
    )
    axes[idx].set_title(title, fontweight='bold', fontsize=12)
    axes[idx].set_ylim(0, 1.2)
    axes[idx].set_ylabel('Score')
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)

    for bar, val, name in zip(bars, values, model_names):
        star = " ★" if name == best_name else ""
        axes[idx].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.3f}{star}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

plt.suptitle('SmartLoan AI — Model Comparison\nLogistic Regression vs Random Forest vs XGBoost',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Comparison chart saved!")

# ── ROC Curves ──
from sklearn.metrics import roc_curve

plt.figure(figsize=(9, 7))
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, color=color, lw=2.5,
             label=f"{name} (AUC={res['auc']:.3f})")

plt.plot([0,1],[0,1],'k--', lw=1, alpha=0.5, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison\nAll 3 Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(" ROC curves saved!")

# ══════════════════════════════════════════
# SAVE EVERYTHING
# ══════════════════════════════════════════
import os
os.makedirs('saved_models', exist_ok=True)

# Save best model
pickle.dump(best_model,   open('saved_models/best_model.pkl', 'wb'))
pickle.dump(scaler,       open('saved_models/scaler.pkl', 'wb'))
pickle.dump(feature_cols, open('saved_models/features.pkl', 'wb'))

# Save all 3 models
pickle.dump(results['Logistic Regression']['model'],
            open('saved_models/logistic_model.pkl', 'wb'))
pickle.dump(results['Random Forest']['model'],
            open('saved_models/rf_model.pkl', 'wb'))
pickle.dump(results['XGBoost']['model'],
            open('saved_models/xgb_model.pkl', 'wb'))

# Save results summary
summary = pd.DataFrame({
    'Model':     model_names,
    'AUC-ROC':   [results[n]['auc']       for n in model_names],
    'Accuracy':  [results[n]['accuracy']  for n in model_names],
    'Precision': [results[n]['precision'] for n in model_names],
    'Recall':    [results[n]['recall']    for n in model_names],
    'F1 Score':  [results[n]['f1']        for n in model_names],
    'Best':      [n == best_name          for n in model_names]
}).round(4)

summary.to_csv('model_results.csv', index=False)

print("\n" + "=" * 55)
print("    FILES SAVED")
print("=" * 55)
print("saved_models/best_model.pkl     ← Best model for predictions")
print("saved_models/scaler.pkl         ← StandardScaler")
print("saved_models/features.pkl       ← Feature column names")
print("saved_models/logistic_model.pkl ← Logistic Regression")
print("saved_models/rf_model.pkl       ← Random Forest")
print("saved_models/xgb_model.pkl      ← XGBoost")
print("model_results.csv               ← Comparison table")
print("model_comparison.png            ← Bar chart")
print("roc_comparison.png              ← ROC curves")

print(f"\nWINNER: {best_name} with AUC-ROC = {results[best_name]['auc']:.4f}")
print("\n Training complete! Now run: streamlit run app.py")

'''
 The file  SUMMARY

1. Loaded the Credit Risk Dataset using Pandas and explored the dataset size.

2. Performed data cleaning by:
   - Removing unrealistic values (Age > 100, Employment Length > 60 years,
     Income > 4,000,000)
   - Filling missing values in loan interest rate and employment length
     using median values.

3. Performed feature engineering by creating:
   - loan_to_income  = loan_amnt / person_income
   - income_per_year = person_income / (person_emp_length + 1)
   - rate_x_loan     = loan_int_rate * loan_amnt / 1000

4. Selected 10 important features and split the data into:
   - 80% Training data
   - 20% Testing data
   Using stratify=y to preserve the class distribution.

5. Applied StandardScaler:
   - fit_transform() on training data
   - transform() on testing data
   to standardize features and avoid data leakage.

6. Trained three machine learning models:
   - Logistic Regression
   - Random Forest
   - XGBoost

7. Evaluated each model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - AUC-ROC

8. Selected the best model based on the highest AUC-ROC score using:

       best_name = max(results,
                       key=lambda x: results[x]['auc'])

9. Created visualizations:
   - Model comparison chart (AUC, Accuracy, F1)
   - ROC Curve comparison for all models

10. Saved the trained artifacts using Pickle:
    - best_model.pkl
    - scaler.pkl
    - features.pkl

11. Deployed the best model using Streamlit to build an
    interactive Loan Approval Prediction application.

PROJECT FLOW:

Load Data
    ↓
Clean Data
    ↓
Feature Engineering
    ↓
Train-Test Split
    ↓
Standard Scaling
    ↓
Train LR + RF + XGBoost
    ↓
Evaluate Models
    ↓
Select Best Model (AUC-ROC)
    ↓
Visualize Results
    ↓
Save Model using Pickle
    ↓
Deploy with Streamlit

Interview Summary:

"I built an end-to-end Loan Default Prediction system called SmartLoan AI.
I cleaned the Credit Risk dataset, engineered features, trained Logistic
Regression, Random Forest, and XGBoost models, evaluated them using multiple
metrics, selected the best model based on AUC-ROC, saved it using Pickle,
and deployed it using Streamlit for real-time loan approval prediction."
'''