# ===============================
# STREAMLIT APP — PREDIKSI HARGA RUMAH
# XGBoost + Bayesian Optimization
# VERSI LENGKAP & FINAL
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
    page_title="Prediksi Harga Rumah Tebet",
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
        transform: translateY(-3px);
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
    """Load trained model and preprocessor"""
    try:
        model = joblib.load("xgb_bo2.pkl")
        preprocessor = joblib.load("preprocess_xgb_bo2.pkl")
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan. Pastikan file 'xgb_bo2.pkl' dan 'preprocess_xgb_bo2.pkl' ada di folder yang sama.")
        st.stop()

@st.cache_data
def load_data():
    """Load dataset for EDA"""
    try:
        df = pd.read_csv("DATA-RUMAH.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ File dataset tidak ditemukan. Menu Analisis Data tidak tersedia.")
        return None

# Load model
model, preprocessor = load_model()
df = load_data()
feature_cols = ["LT", "LB", "KT", "KM", "GRS", "RASIO_LB_LT"]

# Metrik evaluasi dari training
R2_TRAIN = 0.8897
R2_TEST = 0.8106
RMSE_TRAIN = 2_192_835_648
RMSE_TEST = 2_972_868_903
MAE_TRAIN = 1_276_933_427
MAE_TEST = 1_708_359_846
MSE_TRAIN = RMSE_TRAIN ** 2
MSE_TEST = RMSE_TEST ** 2

# ===============================
# FUNGSI UNTUK PREDICTION INTERVAL
# ===============================
def calculate_prediction_interval(y_pred, rmse, confidence=0.95):
    """
    Menghitung prediction interval untuk prediksi individual
    
    Parameters:
    -----------
    y_pred : float
        Nilai prediksi
    rmse : float
        RMSE dari model (dari evaluasi test set)
    confidence : float
        Confidence level (default 0.95 untuk 95%)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound, margin)
    """
    # Z-score untuk confidence level
    if confidence == 0.95:
        z_score = 1.96
    elif confidence == 0.90:
        z_score = 1.645
    elif confidence == 0.99:
        z_score = 2.576
    else:
        z_score = 1.96
    
    # Margin of error
    margin = z_score * rmse
    
    # Interval (minimal 50% dari prediksi, tidak boleh 0)
    lower_bound = max(y_pred * 0.5, y_pred - margin)
    upper_bound = y_pred + margin
    
    return lower_bound, upper_bound, margin

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.markdown("<h1 style='color: white; text-align: center; margin-bottom: 2rem;'>🏠 Menu</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "🏠 Prediksi Harga",
        "📈 Evaluasi Model",
        "📊 Analisis Data",
        "⭐ Feature Importance",
        "🧠 SHAP Analysis"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='color: white; padding: 1rem; font-size: 0.85rem;'>
    <p style='margin: 0;'><strong>📊 Model Info:</strong></p>
    <p style='margin: 0.5rem 0;'>• Algorithm: XGBoost</p>
    <p style='margin: 0.5rem 0;'>• Optimization: Bayesian (Optuna-TPE)</p>
    <p style='margin: 0.5rem 0;'>• R² Score: 0.8106</p>
    <p style='margin: 0.5rem 0;'>• Dataset: 1010 rumah Tebet</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# 🏠 PREDIKSI HARGA
# ===============================
if menu == "🏠 Prediksi Harga":
    st.markdown("<h1>🏠 Prediksi Harga Rumah di Tebet</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Masukkan spesifikasi rumah untuk mendapatkan estimasi harga menggunakan XGBoost + Bayesian Optimization</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📝 Input Data Rumah</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        LT = st.number_input("🏞️ Luas Tanah (m²)", min_value=10, max_value=2000, value=150, step=10,
                            help="Luas tanah total dalam meter persegi")
        LB = st.number_input("🏠 Luas Bangunan (m²)", min_value=10, max_value=1500, value=120, step=10,
                            help="Luas bangunan yang dapat dihuni")
        KT = st.number_input("🛏️ Jumlah Kamar Tidur", min_value=1, max_value=15, value=3, step=1)
    
    with col2:
        KM = st.number_input("🚿 Jumlah Kamar Mandi", min_value=1, max_value=10, value=2, step=1)
        GRS = st.number_input("🚗 Kapasitas Garasi (mobil)", min_value=0, max_value=10, value=1, step=1,
                             help="Jumlah mobil yang dapat diparkir")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction Button
    if st.button("🔮 Prediksi Harga Sekarang"):
        try:
            # Feature engineering
            RASIO_LB_LT = LB / LT if LT > 0 else 0
            
            # Prepare input
            input_data = pd.DataFrame({
                "LT": [LT],
                "LB": [LB],
                "KT": [KT],
                "KM": [KM],
                "GRS": [GRS],
                "RASIO_LB_LT": [RASIO_LB_LT]
            })
            
            # Predict
            y_pred_log = model.predict(input_data)
            harga_prediksi = np.expm1(y_pred_log[0])
            
            # Hitung Prediction Interval
            lower, upper, margin = calculate_prediction_interval(
                harga_prediksi, 
                RMSE_TEST,
                confidence=0.95
            )
            
            # ==========================================
            # DISPLAY HASIL PREDIKSI
            # ==========================================
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>🎯 Hasil Prediksi</h3>", unsafe_allow_html=True)
            
            # 1. HASIL PREDIKSI UTAMA
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>💰 Estimasi Harga Rumah</div>
                <div class='metric-value'>Rp {harga_prediksi/1e9:.2f} Miliar</div>
                <p style='margin-top: 1rem; font-size: 0.95rem; opacity: 0.9;'>
                    atau <strong>Rp {harga_prediksi:,.0f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 2. PREDICTION INTERVAL
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 2rem; border-radius: 15px; color: white;
                        box-shadow: 0 8px 20px rgba(79, 172, 254, 0.3);'>
                <h4 style='color: white; margin-top: 0; font-size: 1.2rem;'>
                    🎯 Interval Kepercayaan Prediksi (95%)
                </h4>
                <p style='margin: 1rem 0 0.5rem 0; opacity: 0.95;'>
                    Berdasarkan analisis statistik, harga rumah ini diperkirakan berada dalam rentang:
                </p>
            """, unsafe_allow_html=True)
            
            col_int1, col_int2 = st.columns(2)
            
            with col_int1:
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.2); padding: 1rem; 
                            border-radius: 10px; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; opacity: 0.9; color: white;'>Harga Minimum</p>
                    <h3 style='margin: 0.5rem 0 0 0; color: white;'>
                        Rp {lower/1e9:.2f} Miliar
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col_int2:
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.2); padding: 1rem; 
                            border-radius: 10px; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; opacity: 0.9; color: white;'>Harga Maksimum</p>
                    <h3 style='margin: 0.5rem 0 0 0; color: white;'>
                        Rp {upper/1e9:.2f} Miliar
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='margin: 1.5rem 0 0 0; font-size: 0.95rem; opacity: 0.9; color: white;'>
                    ⚠️ <strong>Margin Error:</strong> ± Rp {margin/1e9:.2f} Miliar
                </p>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.85; color: white;'>
                    Artinya: Ada kemungkinan 95% bahwa harga aktual berada dalam rentang ini
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 3. MODEL RELIABILITY - GUNAKAN NATIVE STREAMLIT
            st.markdown("#### 📊 Indikator Reliabilitas Model")
            st.write("Prediksi ini dibuat oleh model XGBoost yang telah dievaluasi pada 202 data test:")
            
            # Metrics in columns
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric(
                    label="R² Score",
                    value=f"{R2_TEST:.4f}",
                    delta=f"{R2_TEST*100:.2f}%"
                )
            
            with col_m2:
                st.metric(
                    label="RMSE",
                    value=f"Rp {RMSE_TEST/1e9:.2f}M"
                )
            
            with col_m3:
                st.metric(
                    label="MAE",
                    value=f"Rp {MAE_TEST/1e9:.2f}M"
                )
            
            st.success(f"""
            ✅ Model dapat menjelaskan **{R2_TEST*100:.2f}%** variasi harga rumah  
            ✅ Kategori: **Baik - Sangat Baik** (R² = 0.70 - 0.90)
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 4. INFO TAMBAHAN
            st.info(f"""
            **📌 Catatan Penting:**
            
            • **Estimasi Harga (Rp {harga_prediksi/1e9:.2f} M)** adalah prediksi tunggal berdasarkan karakteristik rumah yang diinput
            
            • **Interval Kepercayaan 95%** menunjukkan rentang dimana harga aktual kemungkinan besar berada. 
              Interval ini dihitung berdasarkan RMSE model dari evaluasi pada data test.
            
            • **R² Score (0.8106)** adalah metrik kualitas MODEL secara keseluruhan (dari 202 data test), 
              BUKAN confidence untuk prediksi individual ini. R² menunjukkan bahwa model mampu 
              menjelaskan 81% variasi harga di dataset.
            
            • **Margin Error (± {margin/1e9:.2f} M)** adalah estimasi rata-rata kesalahan prediksi berdasarkan 
              performa historis model.
            """)
            
            # 5. Feature Contribution
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>📊 Kontribusi Fitur terhadap Prediksi</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            values = [LT, LB, KT, KM, GRS, RASIO_LB_LT]
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
            bars = ax.barh(feature_cols, values, color=colors)
            ax.set_xlabel("Nilai Fitur", fontsize=12, fontweight='bold')
            ax.set_title("Nilai Input Features", fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{val:.2f}', ha='left', va='center', 
                       fontsize=10, fontweight='bold', color='#333')
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {str(e)}")

# ===============================
# 📈 EVALUASI MODEL
# ===============================
elif menu == "📈 Evaluasi Model":
    st.markdown("<h1>📈 Evaluasi Performa Model</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Hasil evaluasi model XGBoost + Bayesian Optimization pada data Test</div>", unsafe_allow_html=True)
    
    # Main Metrics
    st.markdown("<h3 style='color: white;'>📊 Metrik Evaluasi Utama</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>R² Score</div>
            <div style='font-size: 2.5rem; font-weight: 700;'>{R2_TEST:.4f}</div>
            <div style='font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.85;'>Koefisien Determinasi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>RMSE</div>
            <div style='font-size: 2rem; font-weight: 700;'>Rp {RMSE_TEST/1e9:.2f}M</div>
            <div style='font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.85;'>Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>MAE</div>
            <div style='font-size: 2rem; font-weight: 700;'>Rp {MAE_TEST/1e9:.2f}M</div>
            <div style='font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.85;'>Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>MSE</div>
            <div style='font-size: 1.5rem; font-weight: 700;'>{MSE_TEST:.2e}</div>
            <div style='font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.85;'>Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Kategori R² - GUNAKAN NATIVE STREAMLIT
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🏆 Kategori Performa Model")
    
    # Tentukan kategori model saat ini
    if R2_TEST >= 0.90:
        current_category = "Excellent"
        current_range = "R² > 0.90"
    elif R2_TEST >= 0.70:
        current_category = "Baik - Sangat Baik"
        current_range = "R² 0.70-0.90"
    elif R2_TEST >= 0.50:
        current_category = "Cukup Baik"
        current_range = "R² 0.50-0.70"
    else:
        current_category = "Buruk"
        current_range = "R² < 0.50"
    
    # Tampilkan kategori
    st.info(f"**Model Anda**: {current_range} → **{current_category}**")
    
    # Tabel kategori
    st.write("**Standar Interpretasi R² Score:**")
    
    categories_df = pd.DataFrame({
        'Range': ['R² > 0.90', 'R² 0.70-0.90', 'R² 0.50-0.70', 'R² < 0.50'],
        'Kategori': ['⭐ Excellent', '✅ Baik - Sangat Baik', '⚠️ Cukup Baik', '❌ Buruk'],
        'Status': [
            '👉 Model Anda' if R2_TEST >= 0.90 else '',
            '👉 Model Anda' if 0.70 <= R2_TEST < 0.90 else '',
            '👉 Model Anda' if 0.50 <= R2_TEST < 0.70 else '',
            '👉 Model Anda' if R2_TEST < 0.50 else ''
        ]
    })
    
    st.dataframe(categories_df, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance Comparison
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Perbandingan Train vs Test</h3>", unsafe_allow_html=True)
    
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    
    with col_comp1:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem; background: #e3f2fd; border-radius: 10px;'>
            <p style='margin: 0; color: #1565c0; font-weight: bold;'>R² Train</p>
            <h2 style='margin: 0.5rem 0; color: #0d47a1;'>{R2_TRAIN:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_comp2:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem; background: #f3e5f5; border-radius: 10px;'>
            <p style='margin: 0; color: #7b1fa2; font-weight: bold;'>R² Test</p>
            <h2 style='margin: 0.5rem 0; color: #4a148c;'>{R2_TEST:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_comp3:
        gap = R2_TRAIN - R2_TEST
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem; background: #{"e8f5e9" if gap < 0.1 else "#fff3e0"}; border-radius: 10px;'>
            <p style='margin: 0; color: #{"2e7d32" if gap < 0.1 else "#e65100"}; font-weight: bold;'>Gap (Overfitting)</p>
            <h2 style='margin: 0.5rem 0; color: #{"1b5e20" if gap < 0.1 else "#bf360c"};'>{gap:.4f}</h2>
            <p style='margin: 0; font-size: 0.9rem; color: #{"2e7d32" if gap < 0.1 else "#e65100"};'>
                {"✅ Rendah (Baik)" if gap < 0.1 else "⚠️ Sedang"}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info(f"""
    **📌 Analisis Overfitting:**
    
    Gap antara R² Train ({R2_TRAIN:.4f}) dan R² Test ({R2_TEST:.4f}) adalah **{gap:.4f} ({gap*100:.2f}%)**. 
    
    {'✅ Model memiliki generalisasi yang sangat baik dengan gap yang rendah.' if gap < 0.1 else '⚠️ Model memiliki sedikit overfitting, namun masih dalam batas wajar untuk model regresi.'}
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Kesimpulan - GUNAKAN NATIVE STREAMLIT
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ✅ Kesimpulan Evaluasi")
    
    st.success(f"""
    **Model XGBoost dengan Bayesian Optimization memiliki performa yang sangat baik:**
    
    ✅ **R² Score = {R2_TEST:.4f}** → Termasuk kategori "**Baik - Sangat Baik**"
    
    ✅ **Akurasi Tinggi** → Model dapat menjelaskan **{R2_TEST*100:.2f}%** variasi harga
    
    ✅ **Generalisasi Baik** → Gap train-test hanya **{gap:.4f}** (rendah)
    
    ✅ **Reliable** → Rata-rata kesalahan (MAE) sebesar **Rp {MAE_TEST/1e9:.2f} Miliar**
    
    ---
    
    **Model ini layak digunakan sebagai sistem pendukung keputusan untuk estimasi harga properti di kawasan Tebet.**
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# 📊 ANALISIS DATA
# ===============================
elif menu == "📊 Analisis Data":
    if df is None:
        st.error("❌ Dataset tidak tersedia. Pastikan file 'DAFTAR-HARGA-RUMAH.xlsx' ada di folder yang sama.")
    else:
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
    if df is None:
        st.error("❌ Dataset tidak tersedia untuk analisis SHAP.")
    else:
        st.markdown("<h1>🧠 SHAP — Model Interpretability</h1>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Analisis Kontribusi Fitur terhadap Prediksi Menggunakan SHAP Values</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box'>
            <h3 style='margin-top: 0;'>💡 Tentang SHAP</h3>
            <p style='margin-bottom: 0;'>SHAP (SHapley Additive exPlanations) menjelaskan kontribusi setiap fitur terhadap prediksi model. Semakin tinggi nilai SHAP, semakin besar pengaruh fitur tersebut.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🔄 Menghitung SHAP values..."):
            # ===== FIX: Preprocessing dataset dulu =====
            df_sample = df.copy()
            
            # Hapus kolom identifier jika ada
            if 'NO' in df_sample.columns:
                df_sample = df_sample.drop(columns=['NO'])
            if 'NAMA RUMAH' in df_sample.columns:
                df_sample = df_sample.drop(columns=['NAMA RUMAH'])
            
            # Feature engineering: buat RASIO_LB_LT
            if 'LT' in df_sample.columns and 'LB' in df_sample.columns:
                df_sample['RASIO_LB_LT'] = df_sample['LB'] / (df_sample['LT'] + 1)
            
            # Validasi kolom yang dibutuhkan
            required_cols = ["LT", "LB", "KT", "KM", "GRS", "RASIO_LB_LT"]
            missing_cols = [col for col in required_cols if col not in df_sample.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom tidak ditemukan: {', '.join(missing_cols)}")
                st.info("Pastikan dataset memiliki kolom: LT, LB, KT, KM, GRS")
                st.stop()
            
            # Sample data (max 100 atau sesuai ukuran dataset)
            n_samples = min(100, len(df_sample))
            X_sample = df_sample[required_cols].sample(n=n_samples, random_state=42)
            
            # ===== FIX SHAP: Cek apakah model adalah Pipeline atau XGBoost langsung =====
            try:
                # Cek apakah model adalah pipeline (punya named_steps)
                if hasattr(model, 'named_steps'):
                    # Model adalah Pipeline, extract XGBoost-nya
                    st.info("🔍 Detected Pipeline model, extracting XGBoost estimator...")
                    
                    # Cari XGBoost di pipeline
                    if 'xgb' in model.named_steps:
                        xgb_model = model.named_steps['xgb']
                    elif 'regressor' in model.named_steps:
                        xgb_model = model.named_steps['regressor']
                    elif 'estimator' in model.named_steps:
                        xgb_model = model.named_steps['estimator']
                    else:
                        # Ambil step terakhir (biasanya estimator)
                        xgb_model = list(model.named_steps.values())[-1]
                    
                    st.success(f"✅ Extracted: {type(xgb_model).__name__}")
                else:
                    # Model sudah XGBoost langsung
                    xgb_model = model
                
                # Hitung SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X_sample)
                
            except Exception as e:
                st.error(f"❌ Error saat membuat SHAP explainer: {str(e)}")
                st.info(f"""
                **Debug Info:**
                - Model type: {type(model)}
                - Model attributes: {dir(model)[:10]}...
                
                Model Anda mungkin bukan XGBoost atau Pipeline yang valid.
                """)
                st.stop()

        # SHAP Feature Importance
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📊 SHAP Feature Importance (Global)</h3>", unsafe_allow_html=True)
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=required_cols,
            plot_type="bar",
            show=False
        )
        plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
        st.pyplot(plt.gcf())
        st.markdown("</div>", unsafe_allow_html=True)

        # SHAP Summary Plot
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🎯 SHAP Summary Plot</h3>", unsafe_allow_html=True)
        st.markdown("<p>Visualisasi distribusi SHAP values untuk setiap fitur. Warna menunjukkan nilai fitur (merah = tinggi, biru = rendah)</p>", unsafe_allow_html=True)
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=required_cols,
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

