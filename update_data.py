import yfinance as yf
import pandas as pd

print("Mengunduh data terbaru...")

# Download data baru
btc = yf.download("BTC-USD", start="2024-01-01", end="2026-04-26")

# Lihat kolom yang ada dulu
print(f"Kolom: {btc.columns.tolist()}")

# Flatten kolom kalau MultiIndex
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

# Ambil kolom yang dibutuhkan saja
btc = btc[['Open','High','Low','Close','Volume']].copy()
btc.index.name = 'Date'
btc = btc.reset_index()
btc['Date'] = pd.to_datetime(btc['Date']).dt.strftime('%Y-%m-%d')

# Load data lama
df_lama = pd.read_csv('data/btc_harian.csv')
df_lama['Date'] = pd.to_datetime(
    df_lama['Date'], errors='coerce'
).dt.strftime('%Y-%m-%d')

print(f"Data lama: {len(df_lama)} hari")
print(f"Data baru: {len(btc)} hari")

# Gabungkan
df_gabung = pd.concat([df_lama, btc], ignore_index=True)
df_gabung = df_gabung.dropna(subset=['Date','Close'])
df_gabung = df_gabung.drop_duplicates(subset='Date')
df_gabung = df_gabung.sort_values('Date').reset_index(drop=True)

# Simpan
df_gabung.to_csv('data/btc_harian.csv', index=False)

print(f"✅ Total data: {len(df_gabung)} hari")
print(f"Data pertama : {df_gabung['Date'].iloc[0]}")
print(f"Data terakhir: {df_gabung['Date'].iloc[-1]}")