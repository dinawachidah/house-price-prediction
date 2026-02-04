"""
SISTEM PREDIKSI HARGA RUMAH DENGAN FITUR PENGUJIAN
===================================================

Fitur:
1. Prediksi harga rumah baru (input manual)
2. Pengujian batch (upload CSV dengan harga aktual)
3. Perhitungan R², RMSE, MAE secara real-time untuk batch testing
4. Tampilan akurasi model dari training

Author: Dina Wachidah Septiana
NIM: 4611422027
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Prediksi Harga Rumah Tebet",
    page_icon="🏠",
    layout="wide"
)

# ============================================================================
# LOAD CUSTOM CSS
# ============================================================================
def load_css(file_name):
    """Load CSS file eksternal"""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ File CSS '{file_name}' tidak ditemukan. Styling default akan digunakan.")

# Load CSS
load_css('style.css')

# ============================================================================
# LOAD MODEL DAN METADATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load model XGBoost dan metadata akurasi"""
    try:
        with open('xgb_bo2.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan 'xgb_bo2.pkl' ada di folder yang sama.")
        return None

# Metadata akurasi dari pengujian model (dari Bab 4)
MODEL_METADATA = {
    'r2_train': 0.8897,
    'r2_test': 0.8106,
    'rmse_test': 2.974,  # dalam miliar Rupiah
    'mae_test': 1.701,   # dalam miliar Rupiah
}

model = load_model()

# ============================================================================
# FUNGSI PREDIKSI
# ============================================================================
def predict_price(lt, lb, kt, km, grs):
    """
    Prediksi harga rumah berdasarkan input fitur
    
    Parameters:
    -----------
    lt : float - Luas Tanah (m²)
    lb : float - Luas Bangunan (m²)
    kt : int - Jumlah Kamar Tidur
    km : int - Jumlah Kamar Mandi
    grs : int - Jumlah Garasi (mobil)
    
    Returns:
    --------
    float - Prediksi harga dalam Rupiah
    """
    if model is None:
        return None
    
    # Feature engineering: RASIO_LB_LT
    rasio_lb_lt = lb / (lt + 1)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'LT': [lt],
        'LB': [lb],
        'KT': [kt],
        'KM': [km],
        'GRS': [grs],
        'RASIO_LB_LT': [rasio_lb_lt]
    })
    
    # Prediksi dalam skala log
    log_prediction = model.predict(input_data)[0]
    
    # Inverse transform (exp(x) - 1)
    price = np.expm1(log_prediction)
    
    return price

def batch_predict(df):
    """
    Prediksi batch untuk dataframe dengan kolom LT, LB, KT, KM, GRS
    
    Parameters:
    -----------
    df : pd.DataFrame - DataFrame dengan kolom input
    
    Returns:
    --------
    np.array - Array prediksi harga
    """
    if model is None:
        return None
    
    # Feature engineering
    df['RASIO_LB_LT'] = df['LB'] / (df['LT'] + 1)
    
    # Kolom yang dibutuhkan model
    feature_cols = ['LT', 'LB', 'KT', 'KM', 'GRS', 'RASIO_LB_LT']
    
    # Prediksi
    log_predictions = model.predict(df[feature_cols])
    predictions = np.expm1(log_predictions)
    
    return predictions

# ============================================================================
# FUNGSI PERHITUNGAN METRIK
# ============================================================================
def calculate_metrics(y_true, y_pred):
    """
    Hitung R², RMSE, dan MAE
    
    Parameters:
    -----------
    y_true : array-like - Nilai aktual
    y_pred : array-like - Nilai prediksi
    
    Returns:
    --------
    dict - Dictionary berisi metrik evaluasi
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

# ============================================================================
# HEADER APLIKASI
# ============================================================================
st.title("🏠 Sistem Prediksi Harga Rumah Tebet, Jakarta Selatan")
st.markdown("### Berbasis XGBoost + Bayesian Optimization")

# Badge akurasi model
col_badge1, col_badge2, col_badge3 = st.columns(3)

with col_badge1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 15px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📊 R² Model</h3>
        <h1 style='color: white; margin: 5px 0;'>{MODEL_METADATA['r2_test']*100:.2f}%</h1>
        <p style='color: white; margin: 0; font-size: 12px;'>Akurasi pada data uji (202 data)</p>
    </div>
    """, unsafe_allow_html=True)

with col_badge2:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 15px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📉 RMSE</h3>
        <h1 style='color: white; margin: 5px 0;'>Rp{MODEL_METADATA['rmse_test']:.2f}M</h1>
        <p style='color: white; margin: 0; font-size: 12px;'>Root Mean Squared Error</p>
    </div>
    """, unsafe_allow_html=True)

with col_badge3:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 15px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📈 MAE</h3>
        <h1 style='color: white; margin: 5px 0;'>Rp{MODEL_METADATA['mae_test']:.2f}M</h1>
        <p style='color: white; margin: 0; font-size: 12px;'>Mean Absolute Error</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS: MODE PREDIKSI vs MODE PENGUJIAN
# ============================================================================
tab1, tab2 = st.tabs(["🔮 Prediksi Individual", "🧪 Pengujian Batch (dengan R²)"])

# ============================================================================
# TAB 1: PREDIKSI INDIVIDUAL (Input Manual)
# ============================================================================
with tab1:
    st.header("Prediksi Harga Rumah Individual")
    st.info("💡 **Mode ini untuk prediksi rumah baru** yang belum diketahui harga aktualnya. R² tidak dapat dihitung karena tidak ada ground truth.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lt = st.number_input("🏞️ Luas Tanah (m²)", min_value=25, max_value=2000, value=200, step=10)
        lb = st.number_input("🏠 Luas Bangunan (m²)", min_value=20, max_value=1500, value=150, step=10)
        kt = st.number_input("🛏️ Jumlah Kamar Tidur", min_value=1, max_value=15, value=3, step=1)
    
    with col2:
        km = st.number_input("🚿 Jumlah Kamar Mandi", min_value=1, max_value=10, value=2, step=1)
        grs = st.number_input("🚗 Jumlah Garasi (mobil)", min_value=0, max_value=10, value=1, step=1)
    
    if st.button("🔮 Prediksi Harga", key="predict_single", type="primary"):
        if model is not None:
            predicted_price = predict_price(lt, lb, kt, km, grs)
            rasio = lb / (lt + 1)
            
            st.success("### ✅ Hasil Prediksi")
            
            # Display prediksi dalam card besar
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;'>
                <h2 style='color: white; margin: 0;'>Estimasi Harga Rumah</h2>
                <h1 style='color: white; font-size: 48px; margin: 10px 0;'>
                    Rp {predicted_price:,.0f}
                </h1>
                <p style='color: white; margin: 0;'>
                    Prediksi berdasarkan model dengan akurasi R² = {MODEL_METADATA['r2_test']*100:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Informasi tambahan
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("📐 Rasio LB/LT", f"{rasio:.2f}")
            
            with col_info2:
                price_per_sqm = predicted_price / lb
                st.metric("💰 Harga per m²", f"Rp {price_per_sqm:,.0f}")
            
            with col_info3:
                total_rooms = kt + km
                st.metric("🚪 Total Ruangan", f"{total_rooms} ruang")
            
            # Info box kepercayaan
            st.info(f"""
            **ℹ️ Tentang Tingkat Kepercayaan Prediksi**
            
            Model ini memiliki nilai **R² = {MODEL_METADATA['r2_test']*100:.2f}%**, yang berarti model dapat menjelaskan 
            {MODEL_METADATA['r2_test']*100:.1f}% dari variasi harga rumah berdasarkan fitur yang diinputkan.
            
            **Rata-rata kesalahan prediksi (MAE)**: ±Rp{MODEL_METADATA['mae_test']:.2f} miliar
            
            Prediksi lebih akurat untuk rumah dengan spesifikasi yang umum di kawasan Tebet:
            - Luas Tanah: 100-400 m²
            - Luas Bangunan: 80-300 m²
            - Harga: Rp3-15 miliar
            """)

# ============================================================================
# TAB 2: PENGUJIAN BATCH (Upload CSV)
# ============================================================================
with tab2:
    st.header("Pengujian Sistem dengan Data Batch")
    st.success("✅ **Mode ini untuk pengujian sistem** dengan data yang sudah diketahui harga aktualnya. R², RMSE, dan MAE akan dihitung secara otomatis.")
    
    st.markdown("""
    ### 📋 Panduan Penggunaan:
    
    1. **Siapkan file CSV** dengan format kolom berikut:
       - `LT` - Luas Tanah (m²)
       - `LB` - Luas Bangunan (m²)
       - `KT` - Jumlah Kamar Tidur
       - `KM` - Jumlah Kamar Mandi
       - `GRS` - Jumlah Garasi
       - `HARGA` - Harga Aktual (Rupiah) ← **Wajib ada untuk menghitung R²**
    
    2. Upload file CSV
    3. Sistem akan otomatis menghitung prediksi dan metrik evaluasi
    """)
    
    # Download template
    st.markdown("### 📥 Download Template CSV")
    template_data = pd.DataFrame({
        'LT': [200, 300, 150, 250, 180],
        'LB': [150, 220, 110, 200, 130],
        'KT': [4, 5, 3, 4, 3],
        'KM': [3, 4, 2, 3, 2],
        'GRS': [2, 3, 1, 2, 2],
        'HARGA': [8000000000, 13000000000, 5500000000, 10500000000, 6800000000]
    })
    
    csv_template = template_data.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Template CSV",
        data=csv_template,
        file_name="template_pengujian_rumah.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader("📤 Upload File CSV untuk Pengujian", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df_test = pd.read_csv(uploaded_file)
            
            # Validasi kolom
            required_cols = ['LT', 'LB', 'KT', 'KM', 'GRS', 'HARGA']
            missing_cols = [col for col in required_cols if col not in df_test.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ File berhasil diupload! Total data: {len(df_test)} rumah")
                
                # Tampilkan preview data
                with st.expander("👁️ Preview Data (5 baris pertama)"):
                    st.dataframe(df_test.head())
                
                # Tombol untuk mulai pengujian
                if st.button("🧪 Mulai Pengujian", type="primary"):
                    with st.spinner("⏳ Sedang melakukan prediksi dan perhitungan metrik..."):
                        # Prediksi batch
                        predictions = batch_predict(df_test)
                        
                        # Hitung metrik
                        y_true = df_test['HARGA'].values
                        y_pred = predictions
                        
                        metrics = calculate_metrics(y_true, y_pred)
                        
                        # Hasil metrik dalam cards
                        st.markdown("### 📊 Hasil Pengujian Sistem")
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center;'>
                                <h3 style='color: white; margin: 0;'>R² (Coefficient of Determination)</h3>
                                <h1 style='color: white; font-size: 42px; margin: 10px 0;'>{metrics['r2']*100:.2f}%</h1>
                                <p style='color: white; margin: 0; font-size: 13px;'>
                                    Model menjelaskan {metrics['r2']*100:.1f}% variasi harga
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_m2:
                            rmse_miliar = metrics['rmse'] / 1e9
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center;'>
                                <h3 style='color: white; margin: 0;'>RMSE</h3>
                                <h1 style='color: white; font-size: 42px; margin: 10px 0;'>Rp{rmse_miliar:.2f}M</h1>
                                <p style='color: white; margin: 0; font-size: 13px;'>
                                    Root Mean Squared Error
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_m3:
                            mae_miliar = metrics['mae'] / 1e9
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center;'>
                                <h3 style='color: white; margin: 0;'>MAE</h3>
                                <h1 style='color: white; font-size: 42px; margin: 10px 0;'>Rp{mae_miliar:.2f}M</h1>
                                <p style='color: white; margin: 0; font-size: 13px;'>
                                    Mean Absolute Error
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Interpretasi hasil
                        st.markdown("### 📝 Interpretasi Hasil")
                        
                        if metrics['r2'] >= 0.8:
                            interpretation = "🟢 **Excellent** - Model memiliki akurasi yang sangat baik"
                        elif metrics['r2'] >= 0.6:
                            interpretation = "🟡 **Good** - Model memiliki akurasi yang baik"
                        elif metrics['r2'] >= 0.4:
                            interpretation = "🟠 **Fair** - Model memiliki akurasi yang cukup"
                        else:
                            interpretation = "🔴 **Poor** - Model perlu perbaikan"
                        
                        st.info(f"""
                        **Status Akurasi**: {interpretation}
                        
                        - **R² = {metrics['r2']:.4f}**: Model dapat menjelaskan {metrics['r2']*100:.2f}% variasi harga pada data pengujian ini
                        - **RMSE = Rp{rmse_miliar:.2f} miliar**: Rata-rata kesalahan kuadrat (sensitif terhadap error besar)
                        - **MAE = Rp{mae_miliar:.2f} miliar**: Rata-rata kesalahan absolut (interpretasi lebih mudah)
                        
                        **Persentase Error Rata-rata**: {(mae_miliar / (y_true.mean()/1e9)) * 100:.2f}%
                        """)
                        
                        # Tabel hasil prediksi
                        st.markdown("### 📋 Detail Hasil Prediksi")
                        
                        df_result = df_test.copy()
                        df_result['PREDIKSI'] = predictions
                        df_result['ERROR_ABSOLUT'] = np.abs(y_true - y_pred)
                        df_result['ERROR_PERSEN'] = (df_result['ERROR_ABSOLUT'] / df_result['HARGA']) * 100
                        
                        # Format tampilan
                        df_display = df_result.copy()
                        df_display['HARGA'] = df_display['HARGA'].apply(lambda x: f"Rp {x:,.0f}")
                        df_display['PREDIKSI'] = df_display['PREDIKSI'].apply(lambda x: f"Rp {x:,.0f}")
                        df_display['ERROR_ABSOLUT'] = df_display['ERROR_ABSOLUT'].apply(lambda x: f"Rp {x:,.0f}")
                        df_display['ERROR_PERSEN'] = df_display['ERROR_PERSEN'].apply(lambda x: f"{x:.2f}%")
                        
                        st.dataframe(df_display, use_container_width=True)
                        
                        # Download hasil
                        csv_result = df_result.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Hasil Pengujian (CSV)",
                            data=csv_result,
                            file_name="hasil_pengujian_sistem.csv",
                            mime="text/csv"
                        )
                        
                        # Visualisasi: Scatter Plot Prediksi vs Aktual
                        st.markdown("### 📈 Visualisasi: Prediksi vs Harga Aktual")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Scatter plot
                        ax.scatter(y_true/1e9, y_pred/1e9, alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
                        
                        # Garis diagonal (prediksi sempurna)
                        min_val = min(y_true.min(), y_pred.min()) / 1e9
                        max_val = max(y_true.max(), y_pred.max()) / 1e9
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prediksi Sempurna')
                        
                        # Styling
                        ax.set_xlabel('Harga Aktual (Miliar Rupiah)', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Prediksi Sistem (Miliar Rupiah)', fontsize=12, fontweight='bold')
                        ax.set_title(f'Scatter Plot: Prediksi vs Aktual (R² = {metrics["r2"]:.4f})', 
                                    fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11)
                        ax.grid(alpha=0.3, linestyle='--')
                        
                        # Tambahkan teks R²
                        ax.text(0.05, 0.95, f"R² = {metrics['r2']:.4f}\nRMSE = Rp{rmse_miliar:.2f}M\nMAE = Rp{mae_miliar:.2f}M",
                               transform=ax.transAxes, fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                        st.pyplot(fig)
                        
                        # Visualisasi: Distribution of Errors
                        st.markdown("### 📊 Distribusi Error Prediksi")
                        
                        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # Histogram error
                        errors = (y_pred - y_true) / 1e9
                        ax1.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                        ax1.set_xlabel('Error (Miliar Rupiah)', fontsize=11, fontweight='bold')
                        ax1.set_ylabel('Frekuensi', fontsize=11, fontweight='bold')
                        ax1.set_title('Distribusi Error Prediksi', fontsize=12, fontweight='bold')
                        ax1.legend()
                        ax1.grid(alpha=0.3)
                        
                        # Box plot error percentage
                        error_pct = df_result['ERROR_PERSEN'].values
                        ax2.boxplot(error_pct, vert=True)
                        ax2.set_ylabel('Error (%)', fontsize=11, fontweight='bold')
                        ax2.set_title('Boxplot Persentase Error', fontsize=12, fontweight='bold')
                        ax2.grid(alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                        # Statistik error
                        st.markdown("### 📑 Statistik Error")
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("Error Minimum", f"{error_pct.min():.2f}%")
                        
                        with col_stat2:
                            st.metric("Error Maksimum", f"{error_pct.max():.2f}%")
                        
                        with col_stat3:
                            st.metric("Error Median", f"{np.median(error_pct):.2f}%")
                        
                        with col_stat4:
                            st.metric("Std Deviasi Error", f"{error_pct.std():.2f}%")
                
        except Exception as e:
            st.error(f"❌ Error saat memproses file: {str(e)}")
            st.info("Pastikan format file CSV sesuai dengan template yang disediakan.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Sistem Prediksi Harga Rumah Tebet, Jakarta Selatan</strong></p>
    <p>Menggunakan XGBoost + Bayesian Optimization (Optuna-TPE)</p>
    <p>Model Accuracy (dari pengujian): R² = {MODEL_METADATA['r2_test']*100:.2f}% | 
       RMSE = Rp{MODEL_METADATA['rmse_test']:.2f}M | 
       MAE = Rp{MODEL_METADATA['mae_test']:.2f}M</p>
    <p style='font-size: 12px;'>Dina Wachidah Septiana | NIM: 4611422027</p>
</div>
""", unsafe_allow_html=True)


