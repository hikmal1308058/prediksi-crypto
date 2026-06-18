import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import yfinance as yf
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

# ===== CONFIG =====
st.set_page_config(
    page_title="CryptoAI — Prediksi Bitcoin",
    page_icon="₿",
    layout="wide"
)

# ===== STYLE BIRU =====
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020b18 0%, #041428 50%, #020d1f 100%);
    font-family: 'Plus Jakarta Sans', sans-serif;
  }
  .stApp { background: transparent; }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020e1f 0%, #031524 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.25);
  }

  div[data-testid="metric-container"] {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 16px;
    padding: 20px;
  }

  div[data-testid="metric-container"]:hover {
    border-color: rgba(59,130,246,0.5);
    background: rgba(59,130,246,0.12);
    transition: all 0.2s;
  }

  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
  }

  h1, h2, h3 { color: #e0f2fe !important; }
  hr { border-color: rgba(59,130,246,0.15) !important; }
  .stDataFrame { background: rgba(59,130,246,0.05) !important; border-radius: 12px; }

  .stTabs [data-baseweb="tab-list"] {
    background: rgba(59,130,246,0.08);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #93c5fd;
    border-radius: 10px;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
    color: white !important;
  }

  .stTextInput input, .stNumberInput input {
    background: rgba(59,130,246,0.08) !important;
    color: #e0f2fe !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
  }
  .stSelectbox > div > div {
    background: rgba(59,130,246,0.08) !important;
    color: #e0f2fe !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
  }
  .stCaption { color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

# ===== AUTO-UPDATE DATA =====
def auto_update_data():
    """
    Cek apakah data harian sudah up-to-date. Jika data terakhir lebih dari
    1 hari yang lalu, tarik data baru dari yFinance dan gabungkan ke CSV.
    Return: (status: str, jumlah_baris_baru: int)
    """
    path_data = 'data/btc_harian.csv'
    df_cek = pd.read_csv(path_data)
    tanggal_terakhir = pd.to_datetime(df_cek['Date'].iloc[-1])
    hari_ini = datetime.now()

    if (hari_ini - tanggal_terakhir).days <= 1:
        return "up_to_date", 0

    try:
        # Ambil data dari H+1 supaya tidak mengulang baris terakhir yang sudah ada,
        # dan tambahkan 1 hari pada end karena yfinance parameter end bersifat eksklusif.
        tanggal_mulai = tanggal_terakhir + pd.Timedelta(days=1)
        tanggal_akhir = hari_ini + pd.Timedelta(days=1)

        btc_baru = yf.download(
            "BTC-USD",
            start=str(tanggal_mulai.date()),
            end=str(tanggal_akhir.date()),
            auto_adjust=True,
            progress=False
        )

    except Exception:
        return "gagal_fetch", 0


    if len(btc_baru) == 0:
        return "tidak_ada_data_baru", 0

    # Flatten MultiIndex kolom (kasus yfinance versi baru)
    if isinstance(btc_baru.columns, pd.MultiIndex):
        btc_baru.columns = btc_baru.columns.get_level_values(0)

    btc_baru = btc_baru[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    btc_baru.index.name = 'Date'
    btc_baru = btc_baru.reset_index()
    btc_baru['Date'] = pd.to_datetime(btc_baru['Date']).dt.strftime('%Y-%m-%d')

    df_gabung = pd.concat([df_cek, btc_baru], ignore_index=True)
    df_gabung = df_gabung.dropna(subset=['Date', 'Close']).drop_duplicates(subset='Date')
    df_gabung = df_gabung.sort_values('Date').reset_index(drop=True)

    baris_sebelum = len(df_cek)
    df_gabung.to_csv(path_data, index=False)
    baris_baru = len(df_gabung) - baris_sebelum

    return "berhasil", baris_baru


# Jalankan SEBELUM load_everything() (yang di-cache), supaya CSV
# sudah versi terbaru saat dimuat ke memori.
status_update, jumlah_baru = auto_update_data()

# ===== LOAD =====
def hitung_evaluasi(model, scaler, df, lookback=60):
    """
    Hitung ulang MAE & R2 dari 20% data terakhir (data test), persis
    seperti skema split di train.py, supaya metrik di dashboard selalu
    sinkron dengan model yang sedang dipakai (tidak perlu update manual).
    """
    harga = df['Close'].values.reshape(-1, 1)
    harga_scaled = scaler.transform(harga)

    X, y = [], []
    for i in range(lookback, len(harga_scaled)):
        X.append(harga_scaled[i - lookback:i, 0])
        y.append(harga_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    if len(X_test) == 0:
        return None, None

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    return mae, r2


@st.cache_resource
def load_everything():
    model = load_model('model_lstm.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv('data/btc_harian.csv')
    mae, r2 = hitung_evaluasi(model, scaler, df)
    return model, scaler, df, mae, r2

# ===== PREDIKSI =====
def prediksi(model, scaler, df, hari=7):
    harga = df['Close'].values[-60:].reshape(-1, 1)
    scaled = scaler.transform(harga)
    hasil = []
    seq = scaled.copy()
    for _ in range(hari):
        X = seq[-60:].reshape(1, 60, 1)
        p = model.predict(X, verbose=0)
        hasil.append(p[0][0])
        seq = np.append(seq, p, axis=0)
    return scaler.inverse_transform(np.array(hasil).reshape(-1, 1)).flatten()

# ===== JURNAL =====
JURNAL_FILE = 'jurnal_trading.csv'

def load_jurnal():
    if os.path.exists(JURNAL_FILE):
        return pd.read_csv(JURNAL_FILE)
    return pd.DataFrame(columns=[
        'Tanggal','Jenis','Harga Beli','Harga Jual',
        'Jumlah BTC','Profit/Loss','Catatan'
    ])

def simpan_jurnal(df_j):
    df_j.to_csv(JURNAL_FILE, index=False)

model, scaler, df, mae_model, r2_model = load_everything()
r2_pct = f"{r2_model*100:.1f}%" if r2_model is not None else "N/A"

harga_terakhir = df['Close'].iloc[-1]
harga_kemarin  = df['Close'].iloc[-2]
perubahan      = ((harga_terakhir - harga_kemarin) / harga_kemarin) * 100
pred           = prediksi(model, scaler, df, 7)
chg7           = ((pred[6] - harga_terakhir) / harga_terakhir) * 100
last_date      = df['Date'].iloc[-1] if 'Date' in df.columns else ''
tahun_akhir    = last_date[:4] if last_date else "2026"

# Sinyal
if chg7 > 5:
    sinyal = "⚡ STRONG BUY"; warna_s = "#10b981"; warna_bg = "rgba(16,185,129,0.12)"
elif chg7 > 1:
    sinyal = "🟢 BUY"; warna_s = "#34d399"; warna_bg = "rgba(52,211,153,0.08)"
elif chg7 < -5:
    sinyal = "🔴 STRONG SELL"; warna_s = "#f43f5e"; warna_bg = "rgba(244,63,94,0.12)"
elif chg7 < -1:
    sinyal = "🟠 SELL"; warna_s = "#fb923c"; warna_bg = "rgba(251,146,60,0.08)"
else:
    sinyal = "⚪ HOLD"; warna_s = "#fbbf24"; warna_bg = "rgba(251,191,36,0.08)"

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px'>
      <div style='width:56px; height:56px;
           background:linear-gradient(135deg,#1d4ed8,#0ea5e9);
           border-radius:16px; display:flex;
           align-items:center; justify-content:center;
           font-size:28px; margin:0 auto 12px;
           box-shadow:0 4px 20px rgba(59,130,246,0.4)'>₿</div>
      <div style='font-size:18px; font-weight:800; color:#e0f2fe'>CryptoAI</div>
      <div style='font-size:12px; color:#475569'>Prediksi Bitcoin LSTM</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**📊 Data Real-time**")
    st.metric("Harga BTC", f"${harga_terakhir:,.0f}", f"{perubahan:+.2f}%")
    st.metric("Prediksi Besok", f"${pred[0]:,.0f}",
              f"{((pred[0]-harga_terakhir)/harga_terakhir*100):+.2f}%")

    # Indikator status auto-update data
    if status_update == "berhasil":
        st.caption(f"🔄 Data diperbarui · +{jumlah_baru} baris baru")
    elif status_update == "gagal_fetch":
        st.caption("⚠️ Gagal ambil data terbaru dari yFinance")
    elif status_update == "tidak_ada_data_baru":
        st.caption("ℹ️ Tidak ada data baru tersedia")
    else:
        st.caption("✅ Data sudah terbaru")

    st.divider()
    st.markdown(f"""
    <div style='background:rgba(59,130,246,0.08);
         border:1px solid rgba(59,130,246,0.2);
         border-radius:12px; padding:14px'>
      <div style='font-size:11px; color:#475569;
           text-transform:uppercase; letter-spacing:0.1em;
           margin-bottom:8px'>AI Signal</div>
      <div style='font-size:20px; font-weight:800;
           color:{warna_s}'>{sinyal}</div>
      <div style='font-size:12px; color:#475569;
           margin-top:4px'>7 hari: {chg7:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='font-size:12px; color:#475569'>
      <b style='color:#93c5fd'>Model:</b> LSTM<br>
      <b style='color:#93c5fd'>Dataset:</b> 2017–{tahun_akhir}<br>
      <b style='color:#93c5fd'>R² Score:</b> {r2_pct}<br>
      <b style='color:#93c5fd'>Data terakhir:</b> {last_date}
    </div>
    """, unsafe_allow_html=True)

# ===== HEADER =====
st.markdown(f"""
<div style='background:linear-gradient(135deg,
     rgba(29,78,216,0.25), rgba(14,165,233,0.2), rgba(59,130,246,0.15));
     border:1px solid rgba(59,130,246,0.35);
     border-radius:24px; padding:32px; margin-bottom:24px;
     position:relative; overflow:hidden;
     box-shadow:0 8px 32px rgba(29,78,216,0.2)'>
  <div style='position:absolute; top:-60px; right:-60px;
       width:250px; height:250px; border-radius:50%;
       background:radial-gradient(circle,
       rgba(14,165,233,0.15), transparent)'></div>
  <div style='position:absolute; bottom:-40px; left:-40px;
       width:180px; height:180px; border-radius:50%;
       background:radial-gradient(circle,
       rgba(29,78,216,0.12), transparent)'></div>
  <div style='font-size:13px; font-weight:700; color:#38bdf8;
       letter-spacing:0.12em; text-transform:uppercase;
       margin-bottom:8px'>⚡ Live AI Trading Dashboard</div>
  <h1 style='font-size:36px; font-weight:800; margin:0 0 8px;
       background:linear-gradient(135deg,#e0f2fe,#7dd3fc);
       -webkit-background-clip:text;
       -webkit-text-fill-color:transparent'>
    ₿ CryptoAI — Prediksi Harga Bitcoin
  </h1>
  <p style='color:#7dd3fc; margin:0; font-size:15px'>
    Model LSTM · Dataset 2017–{tahun_akhir} · Data terakhir: {last_date}
  </p>
</div>
""", unsafe_allow_html=True)

# ===== METRICS =====
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Harga Terakhir",  f"${harga_terakhir:,.0f}", f"{perubahan:+.2f}%")
c2.metric("🔮 Prediksi Besok",  f"${pred[0]:,.0f}",
          f"{((pred[0]-harga_terakhir)/harga_terakhir*100):+.2f}%")
c3.metric("📈 Harga Tertinggi", f"${df['High'].max():,.0f}")
c4.metric("📉 Harga Terendah",  f"${df['Low'].min():,.0f}")

st.divider()

# ===== TABS =====
tab1, tab2, tab3 = st.tabs([
    "📊 Grafik & Prediksi",
    "🔔 Sinyal Trading",
    "📝 Jurnal Trading"
])

# ========== TAB 1 ==========
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Grafik Harga + Prediksi LSTM")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#020b18')
        ax.set_facecolor('#020b18')

        hist = df['Close'].tail(90).values
        ax.plot(hist, color='#38bdf8', linewidth=2.5, label='Historis')
        ax.fill_between(range(len(hist)), hist, hist.min(),
                        alpha=0.12, color='#38bdf8')

        x_pred = range(len(hist)-1, len(hist)-1+len(pred))
        ax.plot(list(x_pred), pred,
                color='#10b981', linewidth=2.5,
                linestyle='--', marker='o', markersize=7,
                label='Prediksi LSTM',
                markerfacecolor='#34d399',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(list(x_pred), pred*0.95, pred*1.05,
                        alpha=0.12, color='#10b981')

        ax.tick_params(colors='#475569')
        for sp in ax.spines.values():
            sp.set_color('#0c1f35')
        ax.set_ylabel('Harga (USD)', color='#475569')
        ax.legend(facecolor='#020b18', labelcolor='#e0f2fe',
                  fontsize=11, framealpha=0.8)
        ax.grid(axis='y', color='#0c1f35', linewidth=0.7)
        ax.grid(axis='x', color='#0c1f35', linewidth=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("🔮 Detail Prediksi")
        colors = ['#38bdf8','#60a5fa','#818cf8',
                  '#a78bfa','#c084fc','#e879f9','#f472b6']
        for i, p in enumerate(pred):
            chg = ((p - harga_terakhir) / harga_terakhir) * 100
            c = colors[i]
            st.markdown(f"""
            <div style='background:rgba(59,130,246,0.06);
                 border:1px solid {c}44;
                 border-left:3px solid {c};
                 border-radius:12px; padding:12px 16px;
                 margin-bottom:8px'>
              <div style='display:flex; justify-content:space-between'>
                <span style='color:#93c5fd; font-size:12px;
                  font-weight:600'>Hari +{i+1}</span>
                <span style='color:{"#10b981" if chg>=0 else "#f43f5e"};
                  font-weight:700; font-size:13px'>
                  {"▲" if chg>=0 else "▼"} {chg:+.2f}%
                </span>
              </div>
              <div style='font-size:22px; font-weight:800;
                color:#e0f2fe; margin-top:4px'>${p:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 2 ==========
with tab2:
    st.subheader("🔔 Sinyal Trading AI")
    st.markdown(f"""
    <div style='background:{warna_bg};
         border:2px solid {warna_s}55;
         border-radius:20px; padding:36px;
         text-align:center; margin-bottom:24px'>
      <div style='font-size:12px; font-weight:700;
           color:#475569; letter-spacing:0.14em;
           text-transform:uppercase; margin-bottom:12px'>
        REKOMENDASI AI — 7 HARI KE DEPAN
      </div>
      <div style='font-size:52px; font-weight:800;
           color:{warna_s}; margin-bottom:8px'>{sinyal}</div>
      <div style='font-size:15px; color:#93c5fd'>
        Perubahan 7 hari: <b style='color:{warna_s}'>{chg7:+.2f}%</b>
        &nbsp;|&nbsp; Confidence: {r2_pct}
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        tren_7  = df['Close'].tail(7).mean()
        tren_30 = df['Close'].tail(30).mean()
        st.markdown("""
        <div style='background:rgba(59,130,246,0.08);
             border:1px solid rgba(59,130,246,0.2);
             border-radius:16px; padding:16px; margin-bottom:8px'>
          <div style='font-size:13px; font-weight:700;
               color:#60a5fa'>📊 Analisis Tren</div>
        </div>
        """, unsafe_allow_html=True)
        if tren_7 > tren_30:
            st.success("MA7 > MA30 → **UPTREND** 📈")
        else:
            st.error("MA7 < MA30 → **DOWNTREND** 📉")
        st.metric("MA 7 Hari",  f"${tren_7:,.0f}")
        st.metric("MA 30 Hari", f"${tren_30:,.0f}")

    with col2:
        vol     = df['Close'].tail(30).std()
        vol_pct = (vol / harga_terakhir) * 100
        st.markdown("""
        <div style='background:rgba(251,191,36,0.08);
             border:1px solid rgba(251,191,36,0.2);
             border-radius:16px; padding:16px; margin-bottom:8px'>
          <div style='font-size:13px; font-weight:700;
               color:#fbbf24'>💹 Volatilitas</div>
        </div>
        """, unsafe_allow_html=True)
        if vol_pct > 5:
            st.warning(f"Volatilitas **TINGGI**: {vol_pct:.1f}%")
        else:
            st.success(f"Volatilitas **RENDAH**: {vol_pct:.1f}%")
        st.metric("Std Deviasi 30H", f"${vol:,.0f}")
        st.metric("Volatilitas %",   f"{vol_pct:.2f}%")

    with col3:
        target_beli = harga_terakhir * 0.97
        target_jual = pred[2]
        stop_loss   = harga_terakhir * 0.95
        st.markdown("""
        <div style='background:rgba(16,185,129,0.08);
             border:1px solid rgba(16,185,129,0.2);
             border-radius:16px; padding:16px; margin-bottom:8px'>
          <div style='font-size:13px; font-weight:700;
               color:#10b981'>🎯 Target Harga</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("🟢 Target Beli",  f"${target_beli:,.0f}")
        st.metric("🎯 Target Jual",  f"${target_jual:,.0f}")
        st.metric("🛑 Stop Loss",    f"${stop_loss:,.0f}",
                  "-5%", delta_color="inverse")

    st.divider()
    st.warning("⚠️ Sinyal hanya untuk keperluan skripsi, bukan saran investasi!")

# ========== TAB 3 ==========
with tab3:
    st.subheader("📝 Jurnal Trading")
    df_jurnal = load_jurnal()

    with st.expander("➕ Tambah Catatan Trading", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tgl    = st.date_input("📅 Tanggal")
            jenis  = st.selectbox("📌 Jenis", ["BUY", "SELL"])
            h_beli = st.number_input("💵 Harga Beli ($)",
                                      value=float(harga_terakhir), step=100.0)
        with col2:
            h_jual  = st.number_input("💰 Harga Jual ($)",
                                       value=float(pred[0]), step=100.0)
            jumlah  = st.number_input("₿ Jumlah BTC",
                                       value=0.01, step=0.001, format="%.4f")
            catatan = st.text_input("📝 Catatan")

        pl     = (h_jual - h_beli) * jumlah
        pl_pct = ((h_jual - h_beli) / h_beli * 100)

        st.markdown(f"""
        <div style='background:{"rgba(16,185,129,0.1)" if pl>=0 else "rgba(244,63,94,0.1)"};
             border:1px solid {"rgba(16,185,129,0.3)" if pl>=0 else "rgba(244,63,94,0.3)"};
             border-radius:12px; padding:16px; text-align:center'>
          <div style='font-size:12px; color:#475569'>Estimasi Profit/Loss</div>
          <div style='font-size:28px; font-weight:800;
               color:{"#10b981" if pl>=0 else "#f43f5e"}'>
            {"+" if pl>=0 else ""}${pl:,.2f}
          </div>
          <div style='font-size:13px;
               color:{"#10b981" if pl>=0 else "#f43f5e"}'>{pl_pct:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Simpan ke Jurnal", type="primary", use_container_width=True):
            baru = pd.DataFrame([{
                'Tanggal': str(tgl), 'Jenis': jenis,
                'Harga Beli': h_beli, 'Harga Jual': h_jual,
                'Jumlah BTC': jumlah, 'Profit/Loss': round(pl, 2),
                'Catatan': catatan
            }])
            df_jurnal = pd.concat([df_jurnal, baru], ignore_index=True)
            simpan_jurnal(df_jurnal)
            st.success("✅ Berhasil disimpan!")
            st.rerun()

    if len(df_jurnal) > 0:
        st.divider()
        st.subheader("📋 Riwayat Trading")
        total_profit = df_jurnal['Profit/Loss'].sum()
        total_trade  = len(df_jurnal)
        win_trade    = len(df_jurnal[df_jurnal['Profit/Loss'] > 0])
        winrate      = (win_trade/total_trade*100) if total_trade > 0 else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("💰 Total P/L",    f"${total_profit:,.2f}")
        c2.metric("📊 Total Trade",  total_trade)
        c3.metric("🏆 Win Rate",     f"{winrate:.1f}%")
        c4.metric("✅ Profit Trade", win_trade)

        st.dataframe(df_jurnal, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            csv = df_jurnal.to_csv(index=False)
            st.download_button("⬇️ Download Jurnal CSV", csv,
                               "jurnal_trading.csv", "text/csv",
                               use_container_width=True)
        with col2:
            if st.button("🗑️ Hapus Semua", use_container_width=True):
                os.remove(JURNAL_FILE)
                st.rerun()
    else:
        st.info("📭 Belum ada catatan. Tambahkan di atas!")

st.divider()
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:center;
     padding:16px 0; border-top:1px solid rgba(59,130,246,0.15)'>
  <div style='font-size:12px; color:#475569'>
    ⚠️ Aplikasi ini untuk keperluan skripsi saja, bukan saran investasi.
  </div>
  <div style='font-size:13px; font-weight:700;
       background:linear-gradient(135deg,#38bdf8,#818cf8);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
    ₿ CryptoAI · Dibuat oleh Hikmal Herdiansyah · {tahun_akhir}
  </div>
</div>
""", unsafe_allow_html=True)