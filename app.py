import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings

# ================================
# 0. KONFIGURASI AWAL
# ================================
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

st.set_page_config(
    page_title="Viewer Prediksi Saham TLKM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("Viewer Prediksi Saham TLKM (LSTM vs GRU)")

# ================================
# 1. INPUT SECTION
# ================================
st.markdown("### Input File")

c1, c2, c3 = st.columns(3)
with c1:
    forecast_file = st.file_uploader("Forecast CSV (Wajib)", type=["csv"])
with c2:
    evaluated_file = st.file_uploader("Evaluated CSV (Opsional)", type=["csv"])
with c3:
    historical_file = st.file_uploader("Historical CSV (Opsional)", type=["csv"])

load_btn = st.button("Load & Tampilkan", type="primary", use_container_width=True)

# Fungsi helper untuk membersihkan data
def clean_dataframe(df):
    # 1. Deteksi kolom tanggal
    cols = df.columns
    date_col = next((c for c in cols if 'date' in c.lower() or 'tanggal' in c.lower()), cols[0])
    
    # 2. Convert to datetime dengan error handling 'coerce'
    # 'coerce' akan mengubah text ngawur (seperti 'Ticker') menjadi NaT (Not a Time)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # 3. Buang baris yang tanggalnya NaT (baris header sampah/kosong)
    df = df.dropna(subset=[date_col])
    
    # 4. Sorting
    df = df.sort_values(date_col)
    
    return df, date_col

# ================================
# 2. LOGIKA PROSES DATA
# ================================
if load_btn and forecast_file:
    try:
        # --- A. BACA FORECAST CSV ---
        df_forecast = pd.read_csv(forecast_file)
        df_forecast, date_col = clean_dataframe(df_forecast)

        # Deteksi Kolom LSTM & GRU
        cols = df_forecast.columns
        lstm_col = next((c for c in cols if 'lstm' in c.lower()), None)
        gru_col = next((c for c in cols if 'gru' in c.lower()), None)

        if not lstm_col and len(cols) > 1: lstm_col = cols[1]
        if not gru_col and len(cols) > 2: gru_col = cols[2]

        # --- B. BACA HISTORICAL CSV (Jika ada) ---
        df_hist = None
        hist_date_col = None
        hist_price_col = None
        
        if historical_file:
            df_hist = pd.read_csv(historical_file)
            # Bersihkan historical juga
            df_hist, hist_date_col = clean_dataframe(df_hist)
            
            # Cari kolom harga (Close)
            h_cols = df_hist.columns
            hist_price_col = next((c for c in h_cols if 'close' in c.lower()), h_cols[1])

        # --- C. BACA EVALUATED CSV ---
        df_eval = None
        if evaluated_file:
            df_eval = pd.read_csv(evaluated_file)

        # ================================
        # 3. TAMPILAN OUTPUT
        # ================================
        st.divider()
        col_left, col_right = st.columns([3.5, 6.5])

        with col_left:
            # TABEL HASIL
            st.subheader("Hasil Prediksi (Close)")
            
            disp_df = df_forecast[[date_col, lstm_col, gru_col]].copy()
            disp_df[date_col] = disp_df[date_col].dt.strftime('%Y-%m-%d')
            disp_df.columns = ["Tanggal", "LSTM (IDR)", "GRU (IDR)"]
            
            st.dataframe(
                disp_df.style.format({"LSTM (IDR)": "{:,.2f}", "GRU (IDR)": "{:,.2f}"}),
                use_container_width=True,
                hide_index=True,
                height=150
            )

            # RATA-RATA
            last_3 = df_forecast.tail(3)
            avg_lstm = last_3[lstm_col].mean()
            avg_gru = last_3[gru_col].mean()
            diff = avg_lstm - avg_gru
            ket = "lebih tinggi" if diff > 0 else "lebih rendah"

            st.info(f"""
            **Rata-rata 3 hari:**
            - LSTM: **{avg_lstm:,.2f} IDR**
            - GRU : **{avg_gru:,.2f} IDR**
            
            Rata-rata LSTM {ket} **{abs(diff):.2f} IDR** dari GRU.
            """)

            # METRIKS
            st.subheader("Metriks Evaluasi (opsional)")
            if df_eval is not None:
                st.dataframe(df_eval, hide_index=True, use_container_width=True)
            else:
                st.caption("File Evaluasi tidak diunggah.")

        with col_right:
            # GRAFIK
            st.subheader("Visualisasi (Historical 30 hari + Forecast)")

            fig = go.Figure()

            # Plot Historical
            if df_hist is not None:
                last_30 = df_hist.sort_values(hist_date_col).tail(30)
                fig.add_trace(go.Scatter(
                    x=last_30[hist_date_col],
                    y=last_30[hist_price_col],
                    mode='lines',
                    name='Historical Close (last 30)',
                    line=dict(color='#FF6B6B', width=2.5)
                ))

            # Plot LSTM
            fig.add_trace(go.Scatter(
                x=df_forecast[date_col],
                y=df_forecast[lstm_col],
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color='#DAA520', width=2),
                marker=dict(symbol='circle', size=8)
            ))

            # Plot GRU
            fig.add_trace(go.Scatter(
                x=df_forecast[date_col],
                y=df_forecast[gru_col],
                mode='lines+markers',
                name='GRU Forecast',
                line=dict(color='#2ECC71', width=2),
                marker=dict(symbol='triangle-up', size=10)
            ))

            fig.update_layout(
                title=dict(text="Close + Forecast", x=0.5),
                xaxis_title="Date",
                yaxis_title="Price (IDR)",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=500,
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.write("Tips: Pastikan file CSV tidak memiliki baris header ganda.")

elif load_btn and not forecast_file:
    st.warning("⚠️ Mohon upload setidaknya file 'Forecast CSV'.")