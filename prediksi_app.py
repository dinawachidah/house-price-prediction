# ===============================
# STREAMLIT APP — PREDIKSI HARGA RUMAH
# XGBoost + Bayesian Optimization
# Enhanced UI Version + Model Evaluation
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS - MODERN & INTERACTIVE
# ===============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header Styles */
    h1 {
        color: #ffffff;
        font-weight: 700;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-in-out;
    }
    
    h2, h3 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in-out;
    }
    
    /* Card Styles */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease-in-out;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        color: white;
        animation: bounceIn 0.8s ease-in-out;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-label {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input Card */
    .input-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-17eq0hr {
        color: white;
    }
    
    /* Radio Button Styles */
    .stRadio > label {
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Number Input Styles */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Label Styles */
    label {
        font-weight: 600 !important;
        color: #333 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Stats Box */
    .stats-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Evaluation Metric Box */
    .eval-metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .eval-metric-box:hover {
        transform: translateY(-5px);
    }
    
    .eval-metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .eval-metric-label {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    
    .eval-metric-desc {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & DATA
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_bo2.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("DATA-RUMAH.csv")
    df.drop(["NO", "NAMA RUMAH"], axis=1, inplace=True, errors="ignore")
    df["RASIO_LB_LT"] = df["LB"] / (df["LT"] + 1)
    return df

model = load_model()
df = load_data()
feature_cols = ["LT", "LB", "KT", "KM", "GRS", "RASIO_LB_LT"]

# ===============================
# SIDEBAR NAVIGATION
# ===============================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 0;'>🏠 Navigation</h2>", unsafe_allow_html=True)
    
    menu = st.radio(
        "",
        ["🏡 Prediksi Harga", "📈 Evaluasi Model", "📊 Analisis Data", "⭐ Feature Importance", "🧠 SHAP Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
        <h4 style='color: white; margin-top: 0;'>📚 Info Model</h4>
        <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0;'>
            <strong>Algoritma:</strong> XGBoost Regression
        </p>
        <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0;'>
            <strong>Optimasi:</strong> Bayesian Optimization (Optuna-TPE)
        </p>
        <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0;'>
            <strong>Dataset:</strong> Tebet, Jakarta Selatan
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# 🏡 PREDIKSI HARGA
# ===============================
if menu == "🏡 Prediksi Harga":
    st.markdown("<h1>🏡 Prediksi Harga Rumah</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Masukkan spesifikasi rumah untuk mendapatkan estimasi harga</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        LT = st.number_input("🌍 Luas Tanah (m²)", min_value=10, max_value=1000, value=150, step=10)
        KT = st.number_input("🛏️ Kamar Tidur", min_value=1, max_value=10, value=3, step=1)

    with col2:
        LB = st.number_input("🏗️ Luas Bangunan (m²)", min_value=10, max_value=1000, value=120, step=10)
        KM = st.number_input("🚿 Kamar Mandi", min_value=1, max_value=10, value=2, step=1)

    with col3:
        GRS = st.number_input("🚗 Garasi (Mobil)", min_value=0, max_value=5, value=1, step=1)
    
    st.markdown("</div>", unsafe_allow_html=True)

    rasio = LB / (LT + 1)
    input_df = pd.DataFrame([{
        "LT": LT,
        "LB": LB,
        "KT": KT,
        "KM": KM,
        "GRS": GRS,
        "RASIO_LB_LT": rasio
    }])

    # Predict Button
    if st.button("🔮 Prediksi Harga"):
        with st.spinner("🔄 Sedang memproses prediksi..."):
            harga_log = model.predict(input_df)[0]
            harga = np.expm1(harga_log)
            
            # Display Result
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💰 Estimasi Harga Rumah</div>
                <div class="metric-value">Rp {harga:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional Info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='stats-box'>
                    <div class='stats-number'>{rasio:.2f}</div>
                    <div class='stats-label'>Rasio LB/LT</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                harga_per_m2 = harga / LB if LB > 0 else 0
                st.markdown(f"""
                <div class='stats-box'>
                    <div class='stats-number'>Rp {harga_per_m2:,.0f}</div>
                    <div class='stats-label'>Harga per m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_rooms = KT + KM
                st.markdown(f"""
                <div class='stats-box'>
                    <div class='stats-number'>{total_rooms}</div>
                    <div class='stats-label'>Total Ruangan</div>
                </div>
                """, unsafe_allow_html=True)

# ===============================
# 📈 EVALUASI MODEL (MENU BARU)
# ===============================
elif menu == "📈 Evaluasi Model":
    st.markdown("<h1>📈 Evaluasi Performa Model</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Hasil pengujian model XGBoost dengan Bayesian Optimization</div>", unsafe_allow_html=True)

    # ===============================
    # NILAI HASIL EVALUASI (DARI NOTEBOOK)
    # ===============================
    RMSE_TEST = 2973281782     # Rp
    MAE_TEST  = 1702461330     # Rp
    R2_TEST   = 0.8106

    RMSE_TRAIN = 2477063434
    R2_TRAIN   = 0.8897

    gap = R2_TRAIN - R2_TEST

    # ===============================
    # INFO BOX
    # ===============================
    st.markdown("""
    <div class='info-box'>
        <h3>📊 Tentang Evaluasi Model</h3>
        <p>
        Evaluasi dilakukan menggunakan <strong>data uji (test set)</strong> yang tidak digunakan
        dalam proses pelatihan model. Metrik ini menunjukkan kemampuan generalisasi model
        terhadap data baru.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # METRIK UTAMA
    # ===============================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div class='eval-metric-label'>📏 RMSE (Test)</div>
            <div class='eval-metric-value'>Rp {RMSE_TEST/1e9:.2f} M</div>
            <div class='eval-metric-desc'>Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div class='eval-metric-label'>📐 MAE (Test)</div>
            <div class='eval-metric-value'>Rp {MAE_TEST/1e9:.2f} M</div>
            <div class='eval-metric-desc'>Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='eval-metric-box' style='background:linear-gradient(135deg,#f093fb,#f5576c);'>
            <div class='eval-metric-label'>🎯 R² Score (Test)</div>
            <div class='eval-metric-value'>{R2_TEST:.4f}</div>
            <div class='eval-metric-desc'>{R2_TEST*100:.2f}% variasi harga dijelaskan</div>
        </div>
        """, unsafe_allow_html=True)

    # ===============================
    # INTERPRETASI R²
    # ===============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📖 Interpretasi R²</h3>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#667eea,#764ba2);
                padding:2rem;border-radius:15px;color:white;text-align:center;'>
        <h2>R² = {R2_TEST:.4f}</h2>
        <p>Model mampu menjelaskan <strong>{R2_TEST*100:.2f}%</strong> variasi harga rumah</p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"""
        <div style='background:#e8f5e9;padding:1.5rem;border-radius:10px;'>
            <h4>✅ Yang Dijelaskan Model</h4>
            <p><strong>{R2_TEST*100:.2f}%</strong> variasi harga berdasarkan:</p>
            <ul>
                <li>Luas Tanah (LT)</li>
                <li>Luas Bangunan (LB)</li>
                <li>Kamar Tidur & Mandi</li>
                <li>Garasi</li>
                <li>Rasio LB/LT</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div style='background:#fff3e0;padding:1.5rem;border-radius:10px;'>
            <h4>⚪ Faktor di Luar Model</h4>
            <p><strong>{(1-R2_TEST)*100:.2f}%</strong> dipengaruhi oleh:</p>
            <ul>
                <li>Lokasi detail</li>
                <li>Kondisi bangunan</li>
                <li>Tahun bangun</li>
                <li>Kondisi pasar</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # KATEGORI PERFORMA
    # ===============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🏆 Kategori Performa Model</h3>", unsafe_allow_html=True)

    categories = [
        ("❌ R² < 0.50", "Buruk"),
        ("⚠️ R² 0.50 – 0.70", "Cukup Baik"),
        ("✅ R² 0.70 – 0.90", "Baik – Sangat Baik"),
        ("⭐ R² > 0.90", "Excellent")
    ]

    for text, label in categories:
        highlight = "background:#4caf50;color:white;" if "0.70 – 0.90" in text else "background:#f5f5f5;"
        st.markdown(f"""
        <div style='{highlight} padding:1rem;border-radius:10px;margin-bottom:0.5rem;'>
            <strong>{text}</strong> — {label}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # TRAIN VS TEST
    # ===============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Perbandingan Train vs Test</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Train", f"{R2_TRAIN:.4f}")
    c2.metric("R² Test", f"{R2_TEST:.4f}")
    c3.metric("Gap", f"{gap:.4f}")

    st.info(
        "Gap R² yang rendah menunjukkan model memiliki generalisasi yang baik "
        "dan tidak mengalami overfitting signifikan."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # KESIMPULAN
    # ===============================
    st.markdown(f"""
    <div class='card'>
        <h3>✅ Kesimpulan Evaluasi</h3>
    
        <div style='background:linear-gradient(135deg,#4caf50,#43a047);
                    padding:2rem;border-radius:15px;color:white;'>
    
            <div style='margin:1.5rem 0;'>
                <ul style='font-size:1.05rem;line-height:1.8;'>
                    <li>✅ <strong>R² Score = {R2_TEST:.4f}</strong> (Kategori <strong>Baik – Sangat Baik</strong>)</li>
                    <li>✅ <strong>Akurasi Tinggi</strong> → Model menjelaskan <strong>{R2_TEST*100:.2f}%</strong> variasi harga</li>
                    <li>✅ <strong>Generalisasi Baik</strong> → Gap train–test sebesar <strong>{gap:.4f}</strong> (rendah)</li>
                    <li>✅ <strong>Reliable</strong> → Rata-rata kesalahan (MAE) <strong>Rp {MAE_TEST/1e9:.2f} Miliar</strong></li>
                </ul>
            </div>
    
            <div style='margin-top:1.5rem;padding-top:1rem;
                        border-top:2px solid rgba(255,255,255,0.3);
                        font-weight:600;font-size:1.1rem;'>
                Model ini layak digunakan sebagai sistem pendukung keputusan
                untuk estimasi harga properti di kawasan Tebet.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# 📊 ANALISIS DATA
# ===============================
elif menu == "📊 Analisis Data":
    st.markdown("<h1>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analisis Visual Data Harga Rumah</div>", unsafe_allow_html=True)
    
    # Statistics Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stats-box'>
            <div class='stats-number'>{len(df)}</div>
            <div class='stats-label'>Total Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = df["HARGA"].mean()
        st.markdown(f"""
        <div class='stats-box'>
            <div class='stats-number'>Rp {avg_price/1e6:.1f}M</div>
            <div class='stats-label'>Rata-rata Harga</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        min_price = df["HARGA"].min()
        st.markdown(f"""
        <div class='stats-box'>
            <div class='stats-number'>Rp {min_price/1e6:.1f}M</div>
            <div class='stats-label'>Harga Terendah</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_price = df["HARGA"].max()
        st.markdown(f"""
        <div class='stats-box'>
            <div class='stats-number'>Rp {max_price/1e6:.1f}M</div>
            <div class='stats-label'>Harga Tertinggi</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📈 Distribusi Harga Rumah</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["HARGA"], bins=30, kde=True, ax=ax, color="#667eea")
        ax.set_xlabel("Harga (Rp)", fontsize=12)
        ax.set_ylabel("Frekuensi", fontsize=12)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📦 Deteksi Outlier Harga</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df["HARGA"], ax=ax, color="#764ba2")
        ax.set_xlabel("Harga (Rp)", fontsize=12)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🔥 Heatmap Korelasi Fitur</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df[["LT", "LB", "KT", "KM", "GRS", "HARGA"]].corr(),
        annot=True, cmap="RdYlBu_r", fmt=".2f", ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Korelasi antar Fitur", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ⭐ FEATURE IMPORTANCE
# ===============================
elif menu == "⭐ Feature Importance":
    st.markdown("<h1>⭐ Feature Importance Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analisis Tingkat Kepentingan Setiap Fitur dalam Model</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Tingkat Kepentingan Fitur (XGBoost)</h3>", unsafe_allow_html=True)
    
    xgb_model = model.named_steps["xgb"]
    importance = xgb_model.feature_importances_

    imp_df = pd.DataFrame({
        "Fitur": feature_cols,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(data=imp_df, x="Importance", y="Fitur", ax=ax, 
                       palette="viridis")
    ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fitur", fontsize=12, fontweight='bold')
    ax.set_title("Feature Importance", fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=10, 
                fontweight='bold', color='#333')
    
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature Importance Table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Tabel Feature Importance</h3>", unsafe_allow_html=True)
    
    imp_df['Persentase'] = (imp_df['Importance'] / imp_df['Importance'].sum() * 100).round(2)
    imp_df.index = range(1, len(imp_df) + 1)
    
    st.dataframe(
        imp_df.style.background_gradient(cmap='Blues', subset=['Importance', 'Persentase']),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# 🧠 SHAP ANALYSIS
# ===============================
elif menu == "🧠 SHAP Analysis":
    st.markdown("<h1>🧠 SHAP — Model Interpretability</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analisis Kontribusi Fitur terhadap Prediksi Menggunakan SHAP Values</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0;'>💡 Tentang SHAP</h3>
        <p style='margin-bottom: 0;'>SHAP (SHapley Additive exPlanations) menjelaskan kontribusi setiap fitur terhadap prediksi model. Semakin tinggi nilai SHAP, semakin besar pengaruh fitur tersebut.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Menghitung SHAP values..."):
        explainer = shap.TreeExplainer(model.named_steps["xgb"])
        X_sample = df[feature_cols].sample(100, random_state=42)
        X_trans = model.named_steps["prep"].transform(X_sample)
        shap_values = explainer.shap_values(X_trans)

    # SHAP Feature Importance
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 SHAP Feature Importance (Global)</h3>", unsafe_allow_html=True)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_cols,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
    st.pyplot(plt.gcf())
    st.markdown("</div>", unsafe_allow_html=True)

    # SHAP Summary Plot
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🎯 SHAP Summary Plot</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>Visualisasi distribusi SHAP values untuk setiap fitur. Warna menunjukkan nilai fitur (merah = tinggi, biru = rendah)</p>", unsafe_allow_html=True)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_cols,
        show=False
    )
    plt.title("SHAP Summary Plot", fontsize=14, fontweight='bold', pad=20)
    st.pyplot(plt.gcf())
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        © 2025 Sistem Prediksi Harga Rumah | Developed for Skripsi
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.8;'>
        Powered by XGBoost + Bayesian Optimization
    </p>
</div>
""", unsafe_allow_html=True)



