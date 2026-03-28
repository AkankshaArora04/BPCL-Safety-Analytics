import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BPCL Safety Analytics",
    page_icon="🦺",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 2.5rem; }
.card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    border-left: 4px solid #ff6b35;
    margin-bottom: 1rem;
}
.card-val { font-size: 2rem; font-weight: 700; color: #ff6b35; }
.card-lbl { font-size: 0.78rem; color: #888; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ff6b35;
    border-bottom: 1px solid #333;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
}
.winner-box {
    background: linear-gradient(135deg, #1a2a1a, #0f1f0f);
    border: 1px solid #2ecc71;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.insight-box {
    background: #1a1a2e;
    border-left: 3px solid #c44dff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #ccc;
}
</style>
""", unsafe_allow_html=True)

# ── Load & Prep Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel('data/VA ALERTS.XLSX')
    df['created'] = pd.to_datetime(df['created'])
    df['hour'] = df['created'].dt.hour
    df['dayofweek'] = df['created'].dt.dayofweek
    df['month'] = df['created'].dt.month
    df['year'] = df['created'].dt.year
    df['month_str'] = df['created'].dt.to_period('M').astype(str)

    # Clean violation names — merge similar ones
    df['violation'] = df['interlockname'].str.strip()
    df['violation'] = df['violation'].replace({
        'Non-wearing of Helmet': 'Non wearing of Helmet',
        'Non Wearing of Helmet': 'Non wearing of Helmet',
        'Non-wearing of Safety Belt at Height.': 'Non Wearing of Safety Belt at Height',
        'Non wearing of Safety Belt at Height.': 'Non Wearing of Safety Belt at Height',
    })
    # Keep only top 8 violations for clean classification
    top8 = df['violation'].value_counts().head(8).index.tolist()
    df = df[df['violation'].isin(top8)]
    return df

df = load_data()

# ── Train All Models ───────────────────────────────────────────────────────
@st.cache_resource
def train_all_models(df):
    le_unit = LabelEncoder()
    le_viol = LabelEncoder()

    df['unit_enc'] = le_unit.fit_transform(df['unitName'])
    df['viol_enc'] = le_viol.fit_transform(df['violation'])

    X = df[['hour', 'dayofweek', 'month', 'sapId', 'unit_enc']]
    y = df['viol_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':            XGBClassifier(n_estimators=100, random_state=42,
                                            eval_metric='mlogloss', verbosity=0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM':         LinearSVC(max_iter=2000, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = round(accuracy_score(y_test, preds) * 100, 2)
        results[name] = {'model': m, 'acc': acc, 'preds': preds}

    best_name = max(results, key=lambda k: results[k]['acc'])
    return results, best_name, le_unit, le_viol, X_test, y_test, \
           df[['hour','dayofweek','month','sapId','unit_enc','viol_enc','violation','unitName']]

results, best_name, le_unit, le_viol, X_test, y_test, df_model = train_all_models(df)

BASELINE = 60.0

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#12001f,#001020);
            border:1px solid #ff6b35;border-radius:14px;
            padding:1.8rem 2.5rem;margin-bottom:1.5rem'>
    <div style='font-size:1.8rem;font-weight:700;color:#ff6b35'>
        🦺 BPCL Safety Violation Intelligence System
    </div>
    <div style='color:#888;font-size:0.9rem;margin-top:6px'>
        Multi-model ML classifier trained on 42,000+ real CCTV safety alerts · 2021–2024 · 56 units across India
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview", "📈  Trend Analysis", "🤖  Model Comparison", "🔮  Predict"
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{len(df):,}", "Total Alerts"),
        (f"{df['unitName'].nunique()}", "BPCL Units"),
        (f"{df['violation'].nunique()}", "Violation Types"),
        (f"{results[best_name]['acc']}%", "Best Model Accuracy"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""<div class='card'>
                <div class='card-val'>{val}</div>
                <div class='card-lbl'>{lbl}</div></div>""",
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title'>Violation Distribution</div>",
                    unsafe_allow_html=True)
        vc = df['violation'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        colors = ['#ff6b35' if i == 0 else '#c44dff' if i == 1
                  else '#4C72B0' for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1])
        ax.tick_params(colors='#aaa', labelsize=8)
        ax.set_xlabel('Alert Count', color='#aaa')
        ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("<div class='section-title'>Violations by Unit (Top 12)</div>",
                    unsafe_allow_html=True)
        unit_v = df['unitName'].value_counts().head(12)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor('#0f0f1a')
        ax2.set_facecolor('#0f0f1a')
        ax2.barh(unit_v.index[::-1], unit_v.values[::-1], color='#4C72B0')
        ax2.tick_params(colors='#aaa', labelsize=8)
        ax2.set_xlabel('Alert Count', color='#aaa')
        ax2.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("<div class='section-title'>Hourly Alert Pattern</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='insight-box'>📌 This chart shows <b>which hours of the day</b> have the most safety violations — useful for deciding when to increase supervision.</div>",
                unsafe_allow_html=True)
    hourly = df.groupby('hour').size()
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    fig3.patch.set_facecolor('#0f0f1a')
    ax3.set_facecolor('#0f0f1a')
    ax3.bar(hourly.index, hourly.values,
            color=['#ff6b35' if v == hourly.max() else '#334' for v in hourly.values])
    ax3.tick_params(colors='#aaa')
    ax3.set_xlabel('Hour of Day (0 = midnight)', color='#aaa')
    ax3.set_ylabel('Alerts', color='#aaa')
    ax3.spines[:].set_color('#333')
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — TREND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>Monthly Violation Trend (2021–2024)</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='insight-box'>📌 This timeline shows how safety violations changed over 3 years across all BPCL units.</div>",
                unsafe_allow_html=True)

    monthly = df.groupby('month_str').size().reset_index(name='count')
    fig4, ax4 = plt.subplots(figsize=(14, 4))
    fig4.patch.set_facecolor('#0f0f1a')
    ax4.set_facecolor('#0f0f1a')
    ax4.plot(range(len(monthly)), monthly['count'],
             color='#ff6b35', linewidth=2, marker='o', markersize=3)
    ax4.fill_between(range(len(monthly)), monthly['count'],
                     alpha=0.12, color='#ff6b35')
    step = max(1, len(monthly)//12)
    ax4.set_xticks(range(0, len(monthly), step))
    ax4.set_xticklabels(monthly['month_str'][::step],
                        rotation=45, color='#aaa', fontsize=7)
    ax4.tick_params(axis='y', colors='#aaa')
    ax4.spines[:].set_color('#333')
    ax4.grid(True, alpha=0.08)
    plt.tight_layout()
    st.pyplot(fig4); plt.close()

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("<div class='section-title'>Violations by Day of Week</div>",
                    unsafe_allow_html=True)
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        dow = df.groupby('dayofweek').size()
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        fig5.patch.set_facecolor('#0f0f1a')
        ax5.set_facecolor('#0f0f1a')
        ax5.bar(days, dow.values, color='#c44dff', alpha=0.85)
        ax5.tick_params(colors='#aaa', labelsize=9)
        ax5.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig5); plt.close()

    with col_t2:
        st.markdown("<div class='section-title'>Yearly Comparison</div>",
                    unsafe_allow_html=True)
        yearly = df.groupby('year').size()
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        fig6.patch.set_facecolor('#0f0f1a')
        ax6.set_facecolor('#0f0f1a')
        ax6.bar(yearly.index.astype(str), yearly.values,
                color='#4C72B0', alpha=0.85)
        ax6.tick_params(colors='#aaa', labelsize=9)
        ax6.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig6); plt.close()

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class='insight-box'>
    📌 <b>Why multiple models?</b> BPCL's existing system had ~60% accuracy.
    We trained 4 different ML algorithms on the same dataset to find which one
    best classifies safety violations — and by how much it beats the baseline.
    </div>
    """, unsafe_allow_html=True)

    # Accuracy comparison bar chart
    st.markdown("<div class='section-title'>Model Accuracy Comparison</div>",
                unsafe_allow_html=True)

    model_names = list(results.keys())
    accs = [results[m]['acc'] for m in model_names]

    fig7, ax7 = plt.subplots(figsize=(10, 4))
    fig7.patch.set_facecolor('#0f0f1a')
    ax7.set_facecolor('#0f0f1a')
    bar_colors = ['#2ecc71' if n == best_name else '#4C72B0' for n in model_names]
    bars = ax7.bar(model_names, accs, color=bar_colors, width=0.5)
    ax7.axhline(BASELINE, color='#ff6b35', linestyle='--',
                linewidth=1.5, label=f'BPCL Baseline ({BASELINE}%)')
    for bar, acc in zip(bars, accs):
        ax7.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{acc}%', ha='center', color='white', fontsize=10, fontweight='600')
    ax7.set_ylabel('Accuracy (%)', color='#aaa')
    ax7.set_ylim(0, 105)
    ax7.tick_params(colors='#aaa')
    ax7.spines[:].set_color('#333')
    ax7.legend(facecolor='#1e1e2e', labelcolor='#aaa')
    plt.tight_layout()
    st.pyplot(fig7); plt.close()

    # Winner box
    best_acc = results[best_name]['acc']
    improvement = round(best_acc - BASELINE, 2)
    st.markdown(f"""
    <div class='winner-box'>
        <div style='color:#2ecc71;font-size:1.3rem;font-weight:700'>
            🏆 Best Model: {best_name}
        </div>
        <div style='color:#aaa;margin-top:8px;font-size:0.95rem'>
            Accuracy: <b style='color:white'>{best_acc}%</b> &nbsp;|&nbsp;
            BPCL Baseline: <b style='color:#ff6b35'>{BASELINE}%</b> &nbsp;|&nbsp;
            Improvement: <b style='color:#2ecc71'>+{improvement}%</b>
        </div>
        <div style='color:#777;margin-top:8px;font-size:0.85rem'>
            This model correctly classifies safety violations {improvement}% more accurately
            than BPCL's existing detection system.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model results table
    st.markdown("<div class='section-title'>Detailed Results</div>",
                unsafe_allow_html=True)
    table_data = {
        'Model': model_names,
        'Accuracy (%)': accs,
        'vs Baseline': [f"+{round(a-BASELINE,2)}%" if a > BASELINE
                        else f"{round(a-BASELINE,2)}%" for a in accs],
        'Status': ['✅ Best' if n == best_name else
                   '✅ Beats Baseline' if results[n]['acc'] > BASELINE
                   else '❌ Below Baseline' for n in model_names]
    }
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # Feature importance (Random Forest only)
    st.markdown("<div class='section-title'>Feature Importance (Random Forest)</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='insight-box'>📌 This shows which input features the model relies on most to classify violations.</div>",
                unsafe_allow_html=True)
    rf = results['Random Forest']['model']
    fi = pd.Series(rf.feature_importances_,
                   index=['Hour','Day of Week','Month','SAP ID','Unit']
                   ).sort_values()
    fig8, ax8 = plt.subplots(figsize=(8, 3))
    fig8.patch.set_facecolor('#0f0f1a')
    ax8.set_facecolor('#0f0f1a')
    fi.plot(kind='barh', color='#ff6b35', ax=ax8)
    ax8.tick_params(colors='#aaa')
    ax8.set_xlabel('Importance Score', color='#aaa')
    ax8.spines[:].set_color('#333')
    plt.tight_layout()
    st.pyplot(fig8); plt.close()

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — PREDICT
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class='insight-box'>
    📌 <b>How this works:</b> Enter the time, location, and unit details of a new alert.
    The best-performing model will predict what type of safety violation it is —
    before a human even reviews it.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Enter Alert Details</div>",
                unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        hour_i = st.slider("Hour of Alert", 0, 23, 10)
        dow_i = st.selectbox("Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,
                   "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
    with col_p2:
        month_i = st.slider("Month", 1, 12, 6)
        sap_i = st.number_input("SAP ID", min_value=3100, max_value=3500, value=3246)
    with col_p3:
        unit_i = st.selectbox("Unit Name", sorted(df['unitName'].unique()))

    if st.button("⚡ Predict Violation Type", use_container_width=True):
        unit_enc = le_unit.transform([unit_i])[0]
        inp = pd.DataFrame([[hour_i, dow_map[dow_i], month_i, sap_i, unit_enc]],
                           columns=['hour','dayofweek','month','sapId','unit_enc'])

        best_model = results[best_name]['model']
        pred_enc = best_model.predict(inp)[0]
        pred_label = le_viol.inverse_transform([pred_enc])[0]

        st.markdown(f"""
        <div style='background:#1a2a1a;border:1px solid #2ecc71;
                    border-radius:12px;padding:1.5rem 2rem;margin-top:1rem'>
            <div style='color:#2ecc71;font-size:1.2rem;font-weight:700'>
                🔮 Predicted Violation
            </div>
            <div style='color:white;font-size:1.5rem;font-weight:700;margin-top:8px'>
                {pred_label}
            </div>
            <div style='color:#888;font-size:0.85rem;margin-top:6px'>
                Predicted by: {best_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#222;margin-top:3rem'/>
<div style='text-align:center;color:#444;font-size:0.8rem;padding-bottom:1rem'>
    BPCL Safety Violation Intelligence System ·
    Built with Python · Pandas · Scikit-learn · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)