import pandas as pd

# Load data
df = pd.read_csv('data/btc_harga.csv')

# Ubah kolom date jadi format tanggal
df['date'] = pd.to_datetime(df['date'])

# Ubah data per menit → per hari
df_harian = df.groupby(df['date'].dt.date).agg({
    'open'  : 'first',  # harga pembukaan
    'high'  : 'max',    # harga tertinggi
    'low'   : 'min',    # harga terendah
    'close' : 'last',   # harga penutupan
    'Volume USD': 'sum' # total volume
}).reset_index()

# Rename kolom date
df_harian.columns = ['Date','Open','High','Low','Close','Volume']

# Urutkan dari lama ke baru
df_harian = df_harian.sort_values('Date').reset_index(drop=True)

# Cek hasilnya
print(f"Total data harian: {len(df_harian)} hari")
print(df_harian.head())

# Simpan
df_harian.to_csv('data/btc_harian.csv', index=False)
print("✅ Data harian tersimpan di data/btc_harian.csv!")  