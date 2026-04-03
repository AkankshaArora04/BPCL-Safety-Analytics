import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
from PIL import Image
from src.alert_logic import generate_alerts
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Industrial Safety AI", page_icon="🦺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
    color: #1a1a2e;
}
.main { background-color: #f8f9fa; }
.block-container { padding: 2rem 3rem; }

.card {
    background: white;
    border-radius: 8px;
    padding: 1.4rem;
    text-align: center;
    border: 1px solid #e0e0e0;
    border-top: 3px solid #2c3e7a;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.card-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2c3e7a;
}
.card-lbl {
    font-size: 0.72rem;
    color: #888;
    margin-top: 4px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #2c3e7a;
    border-bottom: 2px solid #e8ecf4;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
}
.insight-box {
    background: #f0f3fb;
    border-left: 3px solid #2c3e7a;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #444;
}
.winner-box {
    background: #f0fbf4;
    border: 1px solid #b7dfc4;
    border-left: 4px solid #27ae60;
    border-radius: 8px;
    padding: 1.2rem 1.6rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATASETS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_bpcl():
    df = pd.read_excel('data/bpcl/VA ALERTS.XLSX')
    df['created'] = pd.to_datetime(df['created'])
    df['hour'] = df['created'].dt.hour
    df['dayofweek'] = df['created'].dt.dayofweek
    df['month'] = df['created'].dt.month
    df['month_str'] = df['created'].dt.to_period('M').astype(str)
    df['violation'] = df['interlockname'].str.strip()
    df['violation'] = df['violation'].replace({
        'Non-wearing of Helmet': 'Non wearing of Helmet',
        'Non Wearing of Helmet': 'Non wearing of Helmet',
    })
    top8 = df['violation'].value_counts().head(8).index.tolist()
    df = df[df['violation'].isin(top8)]
    return df

@st.cache_data
def load_ihm():
    import os, glob
    files = glob.glob('data/industrial/*.csv')
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding='latin1'))
        except:
            pass
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df

bpcl = load_bpcl()
ihm = load_ihm()

# ── TRAIN MODELS ──────────────────────────────────────────────────────────────
MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
}

@st.cache_resource
def train_bpcl(df):
    le_unit = LabelEncoder()
    le_viol = LabelEncoder()
    df = df.copy()
    df['unit_enc'] = le_unit.fit_transform(df['unitName'])
    df['viol_enc'] = le_viol.fit_transform(df['violation'])
    X = df[['hour', 'dayofweek', 'month', 'sapId', 'unit_enc']]
    y = df['viol_enc']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    results = {}
    for name, m in MODELS.items():
        m.fit(X_train, y_train)
        acc = round(accuracy_score(y_test, m.predict(X_test)) * 100, 2)
        results[name] = {'model': m, 'acc': acc}
    best = max(results, key=lambda k: results[k]['acc'])
    return results, best, le_unit, le_viol

@st.cache_resource
def train_ihm(df):
    if df.empty or 'Accident Level' not in df.columns:
        return None, None
    df = df.copy().dropna(subset=['Accident Level'])
    le = LabelEncoder()
    results = {}
    # Features: genre, employee type, industry sector
    feature_cols = []
    for col in ['Genre', 'Employee or Third Party', 'Industry Sector', 'Local']:
        if col in df.columns:
            df[col+'_enc'] = LabelEncoder().fit_transform(df[col].astype(str))
            feature_cols.append(col+'_enc')
    if not feature_cols:
        return None, None
    df['target'] = le.fit_transform(df['Accident Level'].astype(str))
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for name, m in MODELS.items():
        try:
            m.fit(X_train, y_train)
            acc = round(accuracy_score(y_test, m.predict(X_test)) * 100, 2)
            results[name] = {'acc': acc}
        except:
            results[name] = {'acc': 0}
    best = max(results, key=lambda k: results[k]['acc'])
    return results, best

bpcl_results, bpcl_best, le_unit, le_viol = train_bpcl(bpcl)
ihm_results, ihm_best = train_ihm(ihm)

BASELINE = 60.0

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:white;border:1px solid #e0e0e0;
            border-left:4px solid #2c3e7a;border-radius:8px;
            padding:1.6rem 2rem;margin-bottom:1.5rem;
            box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
    <div style='font-size:1.6rem;font-weight:700;color:#2c3e7a'>
        Industrial Safety AI — Multi-Dataset Benchmark
    </div>
    <div style='color:#888;font-size:0.88rem;margin-top:4px'>
        Comparing 5 ML models across 2 real-world industrial safety datasets
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
    "Overview", "BPCL Analysis",
    "Industrial Analysis", "Model Benchmark",
    "Predict", "Image Detection", "Video Detection"
])

# ── TAB 1 — OVERVIEW ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    This project benchmarks <b>5 machine learning models</b> across <b>2 real-world industrial safety datasets</b>
    to find which algorithm best classifies safety violations — and by how much it improves over baseline systems.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{len(bpcl):,}", "BPCL Alert Records"),
        (f"{len(ihm):,}", "Industrial Accident Records"),
        ("5", "ML Models Compared"),
        (f"{bpcl_results[bpcl_best]['acc']}%", "Best Accuracy (BPCL)"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""<div class='card'>
                <div class='card-val'>{val}</div>
                <div class='card-lbl'>{lbl}</div></div>""",
                unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Datasets Used</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:#1e1e2e;border-radius:12px;padding:1.2rem;border-left:4px solid #ff6b35'>
        <b style='color:#ff6b35'>📍 Dataset 1 — BPCL Safety Alerts</b><br><br>
        <span style='color:#ccc;font-size:0.9rem'>
        Real CCTV-based safety violation alerts from Bharat Petroleum units across India.
        Covers helmet violations, LPG leakage, fire hazards and more across 56 units from 2021–2024.
        </span>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#1e1e2e;border-radius:12px;padding:1.2rem;border-left:4px solid #c44dff'>
        <b style='color:#c44dff'>🏭 Dataset 2 — IHM Industrial Accidents</b><br><br>
        <span style='color:#ccc;font-size:0.9rem'>
        Real accident records from 12 manufacturing plants across 3 countries by IHM Stefanini, Brazil.
        Covers accident severity levels, industry sectors, and employee types.
        </span>
        </div>""", unsafe_allow_html=True)

# ── TAB 2 — BPCL ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>BPCL Dataset — Violation Distribution</div>",
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        vc = bpcl['violation'].value_counts()
        fig, ax = plt.subplots(figsize=(7,4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        colors = ['#ff6b35' if i==0 else '#c44dff' if i==1 else '#4C72B0'
                  for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1])
        ax.tick_params(colors='#444', labelsize=8)
        ax.set_xlabel('Alert Count', color='#aaa')
        ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("<div class='section-title'>Hourly Pattern</div>",
                    unsafe_allow_html=True)
        hourly = bpcl.groupby('hour').size()
        fig2, ax2 = plt.subplots(figsize=(7,4))
        fig2.patch.set_facecolor('#0f0f1a')
        ax2.set_facecolor('#0f0f1a')
        ax2.bar(hourly.index, hourly.values,
                color=['#ff6b35' if v==hourly.max() else '#334'
                       for v in hourly.values])
        ax2.tick_params(colors='#aaa')
        ax2.set_xlabel('Hour of Day', color='#aaa')
        ax2.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("<div class='section-title'>Monthly Trend</div>", unsafe_allow_html=True)
    monthly = bpcl.groupby('month_str').size().reset_index(name='count')
    fig3, ax3 = plt.subplots(figsize=(14,3))
    fig3.patch.set_facecolor('#0f0f1a')
    ax3.set_facecolor('#0f0f1a')
    ax3.plot(range(len(monthly)), monthly['count'],
             color='#ff6b35', linewidth=2, marker='o', markersize=3)
    ax3.fill_between(range(len(monthly)), monthly['count'],
                     alpha=0.12, color='#ff6b35')
    step = max(1, len(monthly)//12)
    ax3.set_xticks(range(0, len(monthly), step))
    ax3.set_xticklabels(monthly['month_str'][::step],
                        rotation=45, color='#aaa', fontsize=7)
    ax3.tick_params(axis='y', colors='#aaa')
    ax3.spines[:].set_color('#333')
    ax3.grid(True, alpha=0.08)
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

# ── TAB 3 — IHM ───────────────────────────────────────────────────────────────
with tab3:
    if ihm.empty:
        st.warning("IHM dataset not loaded. Check data/industrial/ folder.")
    else:
        st.markdown("<div class='section-title'>IHM Industrial Dataset — Overview</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='card'>
                <div class='card-val'>{len(ihm):,}</div>
                <div class='card-lbl'>Total Accidents</div></div>""",
                unsafe_allow_html=True)
        with c2:
            if 'Accident Level' in ihm.columns:
                st.markdown(f"""<div class='card'>
                    <div class='card-val'>{ihm['Accident Level'].nunique()}</div>
                    <div class='card-lbl'>Severity Levels</div></div>""",
                    unsafe_allow_html=True)
        with c3:
            if 'Industry Sector' in ihm.columns:
                st.markdown(f"""<div class='card'>
                    <div class='card-val'>{ihm['Industry Sector'].nunique()}</div>
                    <div class='card-lbl'>Industry Sectors</div></div>""",
                    unsafe_allow_html=True)

        if 'Accident Level' in ihm.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div class='section-title'>Accident Severity Distribution</div>",
                            unsafe_allow_html=True)
                acc_lev = ihm['Accident Level'].value_counts()
                fig4, ax4 = plt.subplots(figsize=(7,4))
                fig4.patch.set_facecolor('#0f0f1a')
                ax4.set_facecolor('#0f0f1a')
                ax4.bar(acc_lev.index, acc_lev.values, color='#c44dff', alpha=0.85)
                ax4.tick_params(colors='#aaa')
                ax4.set_xlabel('Accident Level', color='#aaa')
                ax4.spines[:].set_color('#333')
                plt.tight_layout()
                st.pyplot(fig4); plt.close()

            with col_b:
                if 'Industry Sector' in ihm.columns:
                    st.markdown("<div class='section-title'>Accidents by Industry</div>",
                                unsafe_allow_html=True)
                    ind = ihm['Industry Sector'].value_counts().head(8)
                    fig5, ax5 = plt.subplots(figsize=(7,4))
                    fig5.patch.set_facecolor('#0f0f1a')
                    ax5.set_facecolor('#0f0f1a')
                    ax5.barh(ind.index[::-1], ind.values[::-1], color='#4C72B0')
                    ax5.tick_params(colors='#aaa', labelsize=8)
                    ax5.spines[:].set_color('#333')
                    plt.tight_layout()
                    st.pyplot(fig5); plt.close()

# ── TAB 4 — BENCHMARK ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class='insight-box'>
    📌 <b>Why benchmark?</b> Different ML algorithms perform differently on different datasets.
    By testing 5 models on 2 real safety datasets, we can find which is most reliable for
    industrial safety violation classification.
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)

    def plot_benchmark(results, best, title, baseline, ax, fig):
        names = list(results.keys())
        accs = [results[n]['acc'] for n in names]
        colors = ['#2ecc71' if n==best else '#4C72B0' for n in names]
        bars = ax.bar(names, accs, color=colors, width=0.5)
        ax.axhline(baseline, color='#ff6b35', linestyle='--',
                   linewidth=1.5, label=f'Baseline ({baseline}%)')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.3,
                    f'{acc}%', ha='center', color='white',
                    fontsize=9, fontweight='600')
        ax.set_title(title, color='#ff6b35', fontsize=11)
        ax.set_ylabel('Accuracy (%)', color='#aaa')
        ax.set_ylim(0, 110)
        ax.tick_params(colors='#aaa', labelsize=8)
        ax.spines[:].set_color('#333')
        plt.xticks(rotation=15)
        ax.legend(facecolor='#1e1e2e', labelcolor='#aaa')

    with col_b1:
        st.markdown("<div class='section-title'>BPCL Dataset Results</div>",
                    unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(7,4))
        fig6.patch.set_facecolor('#0f0f1a')
        ax6.set_facecolor('#0f0f1a')
        plot_benchmark(bpcl_results, bpcl_best, 'BPCL Safety Alerts', BASELINE, ax6, fig6)
        plt.tight_layout()
        st.pyplot(fig6); plt.close()

        best_acc = bpcl_results[bpcl_best]['acc']
        st.markdown(f"""
        <div class='winner-box'>
        <div style='color:#2ecc71;font-weight:700'>🏆 Best: {bpcl_best}</div>
        <div style='color:#aaa;margin-top:6px;font-size:0.9rem'>
        Accuracy: <b style='color:white'>{best_acc}%</b> &nbsp;|&nbsp;
        Improvement: <b style='color:#2ecc71'>+{round(best_acc-BASELINE,2)}%</b> over baseline
        </div></div>""", unsafe_allow_html=True)

    with col_b2:
        if ihm_results:
            st.markdown("<div class='section-title'>IHM Industrial Dataset Results</div>",
                        unsafe_allow_html=True)
            fig7, ax7 = plt.subplots(figsize=(7,4))
            fig7.patch.set_facecolor('#0f0f1a')
            ax7.set_facecolor('#0f0f1a')
            plot_benchmark(ihm_results, ihm_best,
                           'IHM Industrial Accidents', 50.0, ax7, fig7)
            plt.tight_layout()
            st.pyplot(fig7); plt.close()

            ihm_best_acc = ihm_results[ihm_best]['acc']
            st.markdown(f"""
            <div class='winner-box'>
            <div style='color:#2ecc71;font-weight:700'>🏆 Best: {ihm_best}</div>
            <div style='color:#aaa;margin-top:6px;font-size:0.9rem'>
            Accuracy: <b style='color:white'>{ihm_best_acc}%</b>
            </div></div>""", unsafe_allow_html=True)

    # Comparison table
    st.markdown("<div class='section-title'>Full Comparison Table</div>",
                unsafe_allow_html=True)
    table = []
    for name in MODELS.keys():
        row = {'Model': name,'BPCL Accuracy': f"{bpcl_results[name]['acc']}%",'IHM Accuracy': f"{ihm_results[name]['acc']}%" if ihm_results else 'N/A',
            'BPCL vs Baseline': f"+{round(bpcl_results[name]['acc']-BASELINE,2)}%"
        }
        table.append(row)
    st.dataframe(pd.DataFrame(table), use_container_width=True)

# ── TAB 5 — PREDICT ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-title'>🔮 Predict Violation Type — BPCL</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    Enter alert details below. The best-performing model will predict
    what type of safety violation this alert corresponds to.
    </div>""", unsafe_allow_html=True)

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
        unit_i = st.selectbox("Unit", sorted(bpcl['unitName'].unique()))

    if st.button("⚡ Predict", use_container_width=True):
        unit_enc = le_unit.transform([unit_i])[0]
        inp = pd.DataFrame([[hour_i, dow_map[dow_i], month_i, sap_i, unit_enc]],
                           columns=['hour','dayofweek','month','sapId','unit_enc'])
        pred = bpcl_results[bpcl_best]['model'].predict(inp)[0]
        label = le_viol.inverse_transform([pred])[0]
        st.markdown(f"""
        <div style='background:#1a2a1a;border:1px solid #2ecc71;
                    border-radius:12px;padding:1.5rem 2rem;margin-top:1rem'>
            <div style='color:#2ecc71;font-size:1.1rem;font-weight:700'>Predicted Violation</div>
            <div style='color:white;font-size:1.4rem;font-weight:700;margin-top:8px'>{label}</div>
            <div style='color:#888;font-size:0.85rem;margin-top:6px'>Model: {bpcl_best}</div>
        </div>""", unsafe_allow_html=True)

with tab6:
    st.markdown("## Image Detection ")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        from src.detection import detect_objects
        import numpy as np

        results = detect_objects(image)

        # plot result
        res_plotted = results[0].plot()

        st.image(res_plotted, caption="Detected Image", use_column_width=True)

        alerts = generate_alerts(results)

        for alert in alerts:
            if "Violation" in alert:
                st.error(alert)
            else:
                st.success(alert)

with tab7:
    st.markdown("Video Detection")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if video_file:
        import tempfile
        import cv2

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
        h, w, _ = frame.shape

        # define rectangle zone (center area)
        zone_start = (int(w*0.3), int(h*0.3))
        zone_end = (int(w*0.7), int(h*0.7))

        cv2.rectangle(frame, zone_start, zone_end, (0, 0, 255), 2)

        # detection
        results = detect_objects(frame)
        alerts = generate_alerts(results)

        boxes = results[0].boxes

        if boxes is not None:
            for box, cls in zip(boxes.xyxy, boxes.cls):
                if int(cls) == 0:  # person

                    x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

            # check if inside zone
                if (zone_start[0] < cx < zone_end[0]) and (zone_start[1] < cy < zone_end[1]):
                    cv2.putText(frame, "INTRUSION ALERT", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # plot
        res_plotted = frame

        # show frame
        stframe.image(res_plotted, channels="BGR")
        cap.release()



# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#222;margin-top:3rem'/>
<div style='text-align:center;color:#444;font-size:0.8rem;padding-bottom:1rem'>
    Industrial Safety AI · Python · Pandas · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)

