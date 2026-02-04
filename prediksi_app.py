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
    return joblib.load("model_xgb_bo.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("rumah_tebet.csv")
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
    st.markdown("<div class='subtitle'>Hasil Pengujian Model XGBoost + Bayesian Optimization</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # GANTI NILAI DI BAWAH INI DENGAN HASIL DARI NOTEBOOK ANDA
    # ========================================================================
    RMSE_TEST = 2973281782   # Rp 2.97 Miliar
    MAE_TEST = 1702461330    # Rp 1.70 Miliar
    R2_TEST = 0.8106         # R² = 0.8106
    
    # Metrik tambahan (opsional, sesuaikan dengan hasil Anda)
    RMSE_TRAIN = 2477063434  # RMSE pada data training
    R2_TRAIN = 0.8897        # R² pada data training
    # ========================================================================
    
    # Info Section
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0;'>📊 Tentang Evaluasi Model</h3>
        <p style='margin-bottom: 0;'>
            Model telah dievaluasi menggunakan data test (202 sampel) yang tidak pernah 
            "dilihat" selama proses pelatihan. Metrik evaluasi di bawah menunjukkan 
            tingkat akurasi dan kemampuan generalisasi model terhadap data baru.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div class='eval-metric-label'>📏 RMSE (Test)</div>
            <div class='eval-metric-value'>Rp {RMSE_TEST/1e9:.2f}M</div>
            <div class='eval-metric-desc'>Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='eval-metric-box'>
            <div class='eval-metric-label'>📐 MAE (Test)</div>
            <div class='eval-metric-value'>Rp {MAE_TEST/1e9:.2f}M</div>
            <div class='eval-metric-desc'>Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='eval-metric-box' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <div class='eval-metric-label'>🎯 R² Score (Test)</div>
            <div class='eval-metric-value'>{R2_TEST:.4f}</div>
            <div class='eval-metric-desc'>{R2_TEST*100:.2f}% variance explained</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Explanation
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📖 Interpretasi R² Score</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
        <h2 style='margin-top: 0; color: white; text-align: center;'>
            R² = {R2_TEST:.4f}
        </h2>
        <p style='text-align: center; font-size: 1.2rem; margin: 0;'>
            Model dapat menjelaskan <strong>{R2_TEST*100:.2f}%</strong> variasi harga rumah
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"""
        <div style='background: #e8f5e9; padding: 1.5rem; border-radius: 10px; 
                    border-left: 5px solid #4caf50; margin-bottom: 1rem;'>
            <h4 style='color: #2e7d32; margin-top: 0;'>✅ Yang Dapat Dijelaskan Model</h4>
            <p style='color: #1b5e20; margin: 0; font-size: 1.1rem;'>
                <strong>{R2_TEST*100:.2f}%</strong> dari variasi harga rumah dapat diprediksi 
                berdasarkan fitur:
            </p>
            <ul style='color: #1b5e20; margin-top: 0.5rem;'>
                <li>Luas Tanah (LT)</li>
                <li>Luas Bangunan (LB)</li>
                <li>Jumlah Kamar Tidur (KT)</li>
                <li>Jumlah Kamar Mandi (KM)</li>
                <li>Kapasitas Garasi (GRS)</li>
                <li>Rasio LB/LT</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
        <div style='background: #fff3e0; padding: 1.5rem; border-radius: 10px; 
                    border-left: 5px solid #ff9800; margin-bottom: 1rem;'>
            <h4 style='color: #e65100; margin-top: 0;'>⚪ Faktor Lain ({(1-R2_TEST)*100:.2f}%)</h4>
            <p style='color: #bf360c; margin: 0; font-size: 1.1rem;'>
                <strong>{(1-R2_TEST)*100:.2f}%</strong> sisanya dipengaruhi oleh 
                faktor di luar model:
            </p>
            <ul style='color: #bf360c; margin-top: 0.5rem;'>
                <li>Lokasi detail (jarak ke fasilitas)</li>
                <li>Kualitas finishing bangunan</li>
                <li>Tahun pembangunan</li>
                <li>View dan orientasi</li>
                <li>Kondisi pasar properti</li>
                <li>Faktor psikologis pembeli</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Kategori R²
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🏆 Kategori Performa Model</h3>", unsafe_allow_html=True)
    
    categories = [
        ("R² < 0.50", "Buruk", "#f44336", "❌"),
        ("R² 0.50-0.70", "Cukup Baik", "#ff9800", "⚠️"),
        ("R² 0.70-0.90", "Baik - Sangat Baik", "#4caf50", "✅"),
        ("R² > 0.90", "Excellent", "#2196f3", "⭐"),
    ]
    
    for range_val, label, color, icon in categories:
        # PERBAIKAN: Logika deteksi yang benar untuk SEMUA kategori
        if "< 0.50" in range_val:
            is_current = R2_TEST < 0.50
        elif "0.50-0.70" in range_val:
            is_current = 0.50 <= R2_TEST < 0.70
        elif "0.70-0.90" in range_val:
            is_current = 0.70 <= R2_TEST < 0.90
        else:  # > 0.90
            is_current = R2_TEST >= 0.90
        
        # Background gradient jika ini kategori model Anda
        bg_style = f"linear-gradient(135deg, {color} 0%, {color}dd 100%)" if is_current else "#f5f5f5"
        text_color = "white" if is_current else "#333"
        border = f"3px solid {color}" if is_current else "1px solid #e0e0e0"
        font_weight = "bold" if is_current else "normal"
        
        st.markdown(f"""
        <div style='background: {bg_style}; 
                    padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;
                    border: {border};'>
            <p style='margin: 0; color: {text_color}; font-weight: {font_weight};'>
                {icon} <strong>{range_val}</strong>: {label}
                {"<span style='float: right; font-size: 1.2rem;'>👉 Model Anda</span>" if is_current else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Kesimpulan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>✅ Kesimpulan Evaluasi</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); 
                padding: 2rem; border-radius: 15px; color: white; 
                box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);'>
        <h4 style='color: white; margin-top: 0; font-size: 1.3rem;'>
            Model XGBoost dengan Bayesian Optimization ini memiliki performa yang sangat baik untuk prediksi harga rumah di Tebet:
        </h4>
        
        <div style='margin: 1.5rem 0;'>
            <p style='color: white; font-size: 1.05rem; margin: 0.8rem 0; line-height: 1.6;'>
                ✅ <strong>R² Score = {R2_TEST:.4f}</strong> → Termasuk kategori "<strong>Baik - Sangat Baik</strong>"
            </p>
            
            <p style='color: white; font-size: 1.05rem; margin: 0.8rem 0; line-height: 1.6;'>
                ✅ <strong>Akurasi Tinggi</strong> → Model dapat menjelaskan <strong>{R2_TEST*100:.2f}%</strong> variasi harga
            </p>
            
            <p style='color: white; font-size: 1.05rem; margin: 0.8rem 0; line-height: 1.6;'>
                ✅ <strong>Generalisasi Baik</strong> → Gap train-test hanya <strong>{gap:.4f}</strong> (rendah)
            </p>
            
            <p style='color: white; font-size: 1.05rem; margin: 0.8rem 0; line-height: 1.6;'>
                ✅ <strong>Reliable</strong> → Rata-rata kesalahan (MAE) sebesar <strong>Rp {MAE_TEST/1e9:.2f} Miliar</strong>
            </p>
        </div>
        
        <p style='color: white; font-size: 1.1rem; margin: 1.5rem 0 0 0; font-weight: 600; 
                  border-top: 2px solid rgba(255,255,255,0.3); padding-top: 1rem;'>
            Model ini layak digunakan sebagai sistem pendukung keputusan untuk estimasi harga properti di kawasan Tebet.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

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
