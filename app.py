# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──
st.set_page_config(
    page_title="SmartLoan AI",
    page_icon= Image.open("images.jpg"),
    layout="centered"
)

# ── Load all 3 models ──
@st.cache_resource
def load_models():
    best_model = pickle.load(open('saved_models/best_model.pkl', 'rb'))
    scaler     = pickle.load(open('saved_models/scaler.pkl', 'rb'))
    features   = pickle.load(open('saved_models/features.pkl', 'rb'))
    lr_model   = pickle.load(open('saved_models/logistic_model.pkl', 'rb'))
    rf_model   = pickle.load(open('saved_models/rf_model.pkl', 'rb'))
    xgb_model  = pickle.load(open('saved_models/xgb_model.pkl', 'rb'))
    return best_model, scaler, features, lr_model, rf_model, xgb_model

best_model, scaler, feature_cols, lr_model, rf_model, xgb_model = load_models()

all_models = {
    'Logistic Regression': lr_model,
    'Random Forest':       rf_model,
    'XGBoost':             xgb_model
}

# ── Try to load comparison results ──
try:
    model_results = pd.read_csv('model_results.csv')
    best_model_name = model_results[model_results['Best']==True]['Model'].values[0]
except:
    best_model_name = "XGBoost"

# ── CSS ──
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero {
    background: linear-gradient(135deg, #1F4E79, #2E75B6);
    padding: 30px; border-radius: 16px;
    text-align: center; margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(31,78,121,0.4);
}
.hero h1 { color: white; font-size: 2.3rem; font-weight: 700; margin: 0; }
.hero p  { color: rgba(255,255,255,0.8); font-size: 0.95rem; margin: 8px 0 0 0; }
.model-card {
    border: 1.5px solid #333;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 5px 0;
    transition: all 0.2s;
}
.model-winner {
    border: 2px solid #F39C12;
    background: rgba(243,156,18,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 5px 0;
}
.result-card {
    background: linear-gradient(135deg, #1F4E79, #2E75B6);
    border-radius: 16px; padding: 28px;
    text-align: center; margin-bottom: 20px; color: white;
}
.probability-big { font-size: 4.5rem; font-weight: 800; line-height: 1; }
.badge-approve { background:#2ECC71; color:white; padding:10px 28px;
    border-radius:50px; font-size:1.1rem; font-weight:700; display:inline-block; margin:10px 0; }
.badge-reject  { background:#E74C3C; color:white; padding:10px 28px;
    border-radius:50px; font-size:1.1rem; font-weight:700; display:inline-block; margin:10px 0; }
.badge-review  { background:#F39C12; color:white; padding:10px 28px;
    border-radius:50px; font-size:1.1rem; font-weight:700; display:inline-block; margin:10px 0; }
.factor-pos { background:rgba(46,204,113,0.1); border-left:4px solid #2ECC71;
    border-radius:8px; padding:10px 14px; margin:6px 0; font-size:0.9rem; }
.factor-neg { background:rgba(231,76,60,0.1); border-left:4px solid #E74C3C;
    border-radius:8px; padding:10px 14px; margin:6px 0; font-size:0.9rem; }
.step-active   { background:#1F4E79; color:white; padding:5px 14px;
    border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; margin:3px; }
.step-done     { background:#2ECC71; color:white; padding:5px 14px;
    border-radius:20px; font-size:0.82rem; display:inline-block; margin:3px; }
.step-inactive { background:#333; color:#888; padding:5px 14px;
    border-radius:20px; font-size:0.82rem; display:inline-block; margin:3px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if 'page'      not in st.session_state: st.session_state.page      = 1
if 'result'    not in st.session_state: st.session_state.result    = None
if 'user_data' not in st.session_state: st.session_state.user_data = None

# ── Hero Header ──
st.markdown("""
<div class="hero">
    <h1> SmartLoan AI</h1>
    <p>AI-Powered Loan Eligibility | Logistic Regression vs Random Forest vs XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ── Progress Steps ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.session_state.page == 1:
        st.markdown('<span class="step-active">① Details</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="step-done">✓ Details</span>', unsafe_allow_html=True)
with c2:
    if st.session_state.page == 2:
        st.markdown('<span class="step-active">② Compare</span>', unsafe_allow_html=True)
    elif st.session_state.page > 2:
        st.markdown('<span class="step-done">✓ Compare</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="step-inactive">② Compare</span>', unsafe_allow_html=True)
with c3:
    if st.session_state.page == 3:
        st.markdown('<span class="step-active">③ Decision</span>', unsafe_allow_html=True)
    elif st.session_state.page > 3:
        st.markdown('<span class="step-done">✓ Decision</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="step-inactive">③ Decision</span>', unsafe_allow_html=True)
with c4:
    if st.session_state.page == 4:
        st.markdown('<span class="step-active">④ Why?</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="step-inactive">④ Why?</span>', unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════
# PAGE 1 — INPUT FORM
# ════════════════════════════════════
if st.session_state.page == 1:

    st.subheader("📋 Enter Your Details")

    col1, col2 = st.columns(2)
    with col1:
        name       = st.text_input("Full Name", placeholder=" David Kumar")
        age        = st.number_input("Age", min_value=21, max_value=60, value=28)
        income     = st.number_input("Annual Income (Rs)", min_value=100000,
                                      max_value=5000000, value=540000, step=10000)
        emp_length = st.number_input("Employment Length (years)",
                                      min_value=0, max_value=40, value=2)
        cred_hist  = st.number_input("Credit History (years)",
                                      min_value=0, max_value=30, value=3)

    with col2:
        loan_amount = st.number_input("Loan Amount (Rs)", min_value=10000,
                                       max_value=2000000, value=200000, step=10000)
        int_rate    = st.slider("Interest Rate (%)", 8.0, 25.0, 12.0, 0.5)
        loan_grade  = st.selectbox("Credit Score Range", ["Excellent (750+)", "Good (700-749)","Fair (650-699)", "Poor (<650)"],
                                    )
        prev_default = st.selectbox("Previous Defaults?", ["No", "Yes"])
        loan_purpose = st.selectbox("Loan Purpose",
                                     ["Personal","Education","Medical",
                                      "Home Improvement","Venture","Debt Consolidation"])

    st.markdown("")
    if st.button(" Analyze with All 3 Models", use_container_width=True, type="primary"):
        if not name:
            st.error("Please enter your name!")
        else:
            grade_map      = {"Excellent (750+)": '1',
                              "Good (700-749)": '2',
                              "Fair (650-699)": '3',
                              "Poor (<650)": '4'
        }
            loan_to_income = loan_amount / income
            income_per_year = income / (emp_length + 1)
            rate_x_loan    = int_rate * loan_amount / 1000

            input_array = np.array([[
                age, income, emp_length,
                loan_amount, int_rate,
                loan_to_income, cred_hist,
                grade_map[loan_grade],
                income_per_year, rate_x_loan
            ]])
            input_scaled = scaler.transform(input_array)
            input_df     = pd.DataFrame(input_scaled, columns=feature_cols)

            # Get predictions from ALL 3 models
            model_preds = {}
            for mname, mmodel in all_models.items():
                prob_default  = mmodel.predict_proba(input_scaled)[0][1]
                prob_approval = 1 - prob_default
                model_preds[mname] = {
                    'prob_approval': prob_approval,
                    'prob_default':  prob_default
                }

            # Best model prediction
            best_prob_default  = best_model.predict_proba(input_scaled)[0][1]
            best_prob_approval = 1 - best_prob_default

            if best_prob_default < 0.25:   risk = "LOW"
            elif best_prob_default < 0.50: risk = "MEDIUM"
            else:                           risk = "HIGH"

            if best_prob_approval >= 0.70:  decision = "APPROVE"
            elif best_prob_approval >= 0.50: decision = "REVIEW"
            else:                            decision = "REJECT"

            # SHAP for best model
            try:
                explainer   = shap.TreeExplainer(best_model)
                shap_vals   = explainer.shap_values(input_df)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]
            except:
                sv = np.zeros(len(feature_cols))

            # Factor labels
            feature_labels = {
                'person_age':                 'Age',
                'person_income':              'Annual Income',
                'person_emp_length':          'Employment Length',
                'loan_amnt':                  'Loan Amount',
                'loan_int_rate':              'Interest Rate',
                'loan_to_income':             'Loan to Income Ratio',
                'cb_person_cred_hist_length': 'Credit History',
                'grade_num':                  'Loan Grade',
                'income_per_year':            'Income per Year of Experience',
                'rate_x_loan':                'Interest Burden'
            }

            factors = sorted([
                {
                    'feature': feature_labels.get(f, f),
                    'impact':  v
                }
                for f, v in zip(feature_cols, sv)
            ], key=lambda x: abs(x['impact']), reverse=True)

            # Save to session
            st.session_state.result = {
                'name':             name,
                'model_preds':      model_preds,
                'best_prob_approval': best_prob_approval,
                'best_prob_default':  best_prob_default,
                'risk':             risk,
                'decision':         decision,
                'factors':          factors,
                'input_df':         input_df
            }
            st.session_state.user_data = {
                'name':        name,
                'age':         age,
                'income':      income,
                'loan_amount': loan_amount,
                'emp_length':  emp_length,
                'loan_grade':  loan_grade,
                'int_rate':    int_rate
            }
            st.session_state.page = 2
            st.rerun()


# ════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ════════════════════════════════════
elif st.session_state.page == 2:

    result = st.session_state.result
    name   = result['name']

    st.subheader(f" Model Comparison for {name}")
    st.write("Here is what each AI model predicts for your application:")

    # Show 3 model cards
    col1, col2, col3 = st.columns(3)
    cols   = [col1, col2, col3]
    mnames = ['Logistic Regression', 'Random Forest', 'XGBoost']
    icons  = ['📊', '🌲', '🚀']
    colors_m = ['#3498DB', '#2ECC71', '#E74C3C']

    for col, mname, icon, color in zip(cols, mnames, icons, colors_m):
        pred   = result['model_preds'][mname]
        pct    = pred['prob_approval'] * 100
        is_best = (mname == best_model_name)
        card_class = "model-winner" if is_best else "model-card"

        winner_badge = " BEST MODEL" if is_best else ""

        if pct >= 70:   dec = " APPROVE"
        elif pct >= 50: dec = " REVIEW"
        else:           dec = " REJECT"

        col.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="font-size:0.85rem; font-weight:600;
                        color:#aaa; margin:4px 0;">{mname}</div>
            <div style="font-size:2.5rem; font-weight:800;
                        color:{color};">{pct:.0f}%</div>
            <div style="font-size:0.8rem; color:#888;">Approval Probability</div>
            <div style="font-size:0.9rem; font-weight:600;
                        margin-top:8px;">{dec}</div>
            <div style="font-size:0.75rem; color:#F39C12;
                        margin-top:4px; font-weight:600;">{winner_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Bar chart comparing 3 models
    st.subheader(" Approval Probability — Visual Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    probs  = [result['model_preds'][n]['prob_approval'] * 100 for n in mnames]
    colors_bar = ['#3498DB', '#2ECC71', '#E74C3C']

    bars = ax.barh(
        ['LR', 'RF', 'XGB'],
        probs,
        color=colors_bar,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.3,
        height=0.5
    )

    # Add threshold line
    ax.axvline(x=70, color='#2ECC71', linewidth=1.5,
               linestyle='--', alpha=0.8, label='Approval Threshold (70%)')
    ax.axvline(x=50, color='#F39C12', linewidth=1.5,
               linestyle=':', alpha=0.8, label='Review Threshold (50%)')

    for bar, val, mname in zip(bars, probs, mnames):
        star = " ★" if mname == best_model_name else ""
        ax.text(
            min(val + 1.5, 98),
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%{star}',
            va='center', color='white', fontweight='bold', fontsize=11
        )

    ax.set_xlabel('Approval Probability (%)', color='white', fontsize=11)
    ax.set_title(f'Model Comparison — {name}\'s Application',
                 color='white', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.tick_params(colors='white', labelsize=11)
    ax.spines['bottom'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.legend(loc='lower right', fontsize=9,
              facecolor='#1E1E1E', labelcolor='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Model accuracy from training
    st.subheader("Model Training Performance (on test data)")
    try:
        mr = pd.read_csv('model_results.csv')
        display_cols = ['Model','AUC-ROC','Accuracy','Precision','Recall','F1 Score']
        st.dataframe(
            mr[display_cols].style.highlight_max(
                subset=['AUC-ROC','Accuracy','F1 Score'],
                color='rgba(46,204,113,0.3)'
            ).format({
                'AUC-ROC':   '{:.4f}',
                'Accuracy':  '{:.4f}',
                'Precision': '{:.4f}',
                'Recall':    '{:.4f}',
                'F1 Score':  '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f" Best model: **{best_model_name}** — used for final decision")
    except:
        st.info("Run train_model.py to see model comparison table")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Start Over", use_container_width=True):
            st.session_state.page   = 1
            st.session_state.result = None
            st.rerun()
    with col2:
        if st.button(f"See Final Decision →", use_container_width=True, type="primary"):
            st.session_state.page = 3
            st.rerun()


# ════════════════════════════════════
# PAGE 3 — FINAL DECISION
# ════════════════════════════════════
elif st.session_state.page == 3:

    result   = st.session_state.result
    name     = result['name']
    prob_pct = result['best_prob_approval'] * 100
    decision = result['decision']
    risk     = result['risk']

    st.subheader(f" Final AI Decision for {name}")
    st.caption(f"Based on {best_model_name} (best performing model)")

    if decision == "APPROVE":
        badge = '<span class="badge-approve"> APPROVED</span>'
    elif decision == "REVIEW":
        badge = '<span class="badge-review">MANUAL REVIEW</span>'
    else:
        badge = '<span class="badge-reject"> REJECTED</span>'

    if risk == "LOW":     risk_icon = "🟢"
    elif risk == "MEDIUM": risk_icon = "🟡"
    else:                  risk_icon = "🔴"

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:0.9rem; opacity:0.8; margin-bottom:5px;">
            Approval Probability
        </div>
        <div class="probability-big">{prob_pct:.0f}%</div>
        <div style="margin:15px 0;">{badge}</div>
        <div style="font-size:1rem;">
            Risk Level: {risk_icon} <strong>{risk}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 model comparison mini
    st.markdown("**All 3 models agreed:**")
    col1, col2, col3 = st.columns(3)
    mnames = ['Logistic Regression', 'Random Forest', 'XGBoost']
    icons  = ['📊', '🌲', '🚀']

    for col, mname, icon in zip([col1,col2,col3], mnames, icons):
        pred = result['model_preds'][mname]
        pct  = pred['prob_approval'] * 100
        col.metric(
            f"{icon} {mname.split()[0]}",
            f"{pct:.0f}%",
            delta="Best " if mname == best_model_name else None
        )

    st.divider()

    if decision == "APPROVE":
        st.success(f"""
        **Congratulations {name}!**
        Your loan application looks strong.
        {best_model_name} predicts {prob_pct:.0f}% approval probability
        with **{risk} risk** of default.
        """)
    elif decision == "REVIEW":
        st.warning(f"""
        **{name}, your application needs review.**
        Our AI shows moderate risk factors.
        A loan officer will review your application.
        """)
    else:
        st.error(f"""
        **{name}, we cannot approve this application.**
        Our AI detected high default risk.
        Please see the explanation to know what to improve.
        """)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with col2:
        if st.button(" Why This Decision?", use_container_width=True, type="primary"):
            st.session_state.page = 4
            st.rerun()
    with col3:
        if st.button(" New Application", use_container_width=True):
            st.session_state.page   = 1
            st.session_state.result = None
            st.rerun()


# ════════════════════════════════════
# PAGE 4 — SHAP EXPLANATION
# ════════════════════════════════════
elif st.session_state.page == 4:

    result = st.session_state.result
    name   = result['name']

    st.subheader(f" Why Did AI Decide This for {name}?")
    st.write(f"SHAP explanation from **{best_model_name}** — the best model")

    if result['decision'] == "APPROVE":
        st.success(f" APPROVED | Probability: {result['best_prob_approval']:.1%}")
    elif result['decision'] == "REVIEW":
        st.warning(f" REVIEW | Probability: {result['best_prob_approval']:.1%}")
    else:
        st.error(f" REJECTED | Probability: {result['best_prob_approval']:.1%}")

    st.divider()

    # Positive vs Negative factors
    factors  = result['factors']
    positive = [f for f in factors if f['impact'] < 0][:4]
    negative = [f for f in factors if f['impact'] > 0][:4]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("** Working FOR You:**")
        if positive:
            for f in positive:
                strength = "Strong" if abs(f['impact'])>0.3 else "Moderate" if abs(f['impact'])>0.1 else "Minor"
                st.markdown(f"""
                <div class="factor-pos">
                     <strong>{f['feature']}</strong><br>
                    <small style="color:#2ECC71">{strength} positive factor</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong positive factors")

    with col2:
        st.markdown("** Working AGAINST You:**")
        if negative:
            for f in negative:
                strength = "Strong" if abs(f['impact'])>0.3 else "Moderate" if abs(f['impact'])>0.1 else "Minor"
                st.markdown(f"""
                <div class="factor-neg">
                     <strong>{f['feature']}</strong><br>
                    <small style="color:#E74C3C">{strength} negative factor</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong negative factors")

    st.divider()

    # SHAP Chart
    st.subheader(" Full SHAP Impact Chart")
    st.caption("🟢 Green = reduces default risk (GOOD) | 🔴 Red = increases default risk (BAD)")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    top_factors = sorted(factors, key=lambda x: x['impact'])
    feat_names  = [f['feature'] for f in top_factors]
    impacts     = [f['impact']  for f in top_factors]
    bar_colors  = ['#2ECC71' if v < 0 else '#E74C3C' for v in impacts]

    bars = ax.barh(feat_names, impacts, color=bar_colors,
                   alpha=0.85, edgecolor='white', linewidth=0.2)
    ax.axvline(x=0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('SHAP Value', color='white', fontsize=11)
    ax.set_title(f'Why Did {best_model_name} Make This Decision?\n'
                 f'Negative = Good for approval | Positive = Bad for approval',
                 color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Improvement tips
    st.subheader(" How to Improve Your Score:")
    user = st.session_state.user_data

    tips = []
    if result['best_prob_default'] > 0.3:
        if user['loan_amount'] / user['income'] > 0.3:
            tips.append(" Reduce loan amount — loan-to-income ratio is too high")
        if user['int_rate'] > 12:
            tips.append(" Improve credit score — better score = lower rate = better approval")
        if user['emp_length'] < 2:
            tips.append(" Build employment history — stable job history helps")
        if user['loan_grade'] in ["Fair (650-699)","Poor (<650)"]:
            tips.append(" Pay existing loans on time to improve your loan grade")
        tips.append(" Reapply after 6 months with improved financial profile")

    if tips:
        for tip in tips:
            st.markdown(f"""
            <div class="factor-pos" style="margin:5px 0;">{tip}</div>
            """, unsafe_allow_html=True)
    else:
        st.success(" Your profile is strong! No major improvements needed.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Decision", use_container_width=True):
            st.session_state.page = 3
            st.rerun()
    with col2:
        if st.button("New Application", use_container_width=True, type="primary"):
            st.session_state.page   = 1
            st.session_state.result = None
            st.rerun()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:11px; padding:8px;'>
        SmartLoan AI | Built by Stephen David |
        Logistic Regression + Random Forest + XGBoost + SHAP |
        AI & Data Science Portfolio Project
        <br>⚠️ Educational AI model only. Not a substitute for actual banking decisions.
    </div>
    """, unsafe_allow_html=True)