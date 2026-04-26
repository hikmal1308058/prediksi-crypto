import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

# ===== 1. LOAD DATA =====
df = pd.read_csv('data/btc_harian.csv')
print(f"✅ Data dimuat: {len(df)} hari")

# ===== 2. AMBIL KOLOM CLOSE =====
harga = df['Close'].values.reshape(-1, 1)

# ===== 3. NORMALISASI =====
scaler = MinMaxScaler()
harga_scaled = scaler.fit_transform(harga)

# Simpan scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ===== 4. BUAT SEQUENCES =====
def buat_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = buat_sequences(harga_scaled, lookback=60)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ===== 5. BAGI DATA =====
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# ===== 6. BUAT MODEL =====
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# ===== 7. TRAINING =====
print("\n🚀 Mulai training...")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ===== 8. EVALUASI =====
y_pred_scaled = model.predict(X_test)
y_pred   = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1,1))

mae = mean_absolute_error(y_actual, y_pred)
r2  = r2_score(y_actual, y_pred)

print(f"\n📊 Hasil Evaluasi:")
print(f"MAE : ${mae:,.0f}")
print(f"R²  : {r2:.4f} ({r2*100:.1f}%)")

# ===== 9. SIMPAN MODEL =====
model.save('model_lstm.keras')
print("✅ Model tersimpan di model_lstm.keras!")