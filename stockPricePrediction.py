# ============================================================
#  STOCK PRICE PREDICTION - Applied ML Lab Experiment
#  Dataset : Yahoo Finance (auto-downloaded via yfinance)
#  Author  : B.Tech CSE - Semester 4
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ── OUTPUT FOLDER ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("      STOCK PRICE PREDICTION EXPERIMENT")
print("=" * 55)

# ── 1. DOWNLOAD DATA ──────────────────────────────────────────
# Change TICKER to any stock: GOOGL, TSLA, MSFT, TCS.NS, etc.
TICKER = "AAPL"

print(f"\n📥 Downloading {TICKER} stock data (2020–2024)...")
raw = yf.download(TICKER, start="2020-01-01", end="2024-12-31", progress=False)

# Flatten multi-level columns if present
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
df.index = pd.to_datetime(df.index)

print(f"✅ {len(df)} trading days loaded.")
print(f"   Date Range : {df.index.min().date()} → {df.index.max().date()}")
print(f"   Price Range: ${float(df['Close'].min()):.2f} — ${float(df['Close'].max()):.2f}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────
df["MA_7"]       = df["Close"].rolling(7).mean()    # 7-day moving average
df["MA_21"]      = df["Close"].rolling(21).mean()   # 21-day moving average
df["Lag_1"]      = df["Close"].shift(1)             # yesterday's price
df["Lag_2"]      = df["Close"].shift(2)             # 2 days ago
df["Lag_3"]      = df["Close"].shift(3)             # 3 days ago
df["Price_Range"]= df["High"] - df["Low"]           # daily range
df["Target"]     = df["Close"].shift(-1)            # NEXT day's price (what we predict)
df.dropna(inplace=True)

print(f"✅ Features created. {len(df)} usable rows after cleanup.")

# ── 3. CLOSING PRICE CHART ────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["Close"], color="#3498db", lw=1.5, label="Close")
plt.plot(df.index, df["MA_7"],  color="#e74c3c", lw=1,   linestyle="--", label="MA 7")
plt.plot(df.index, df["MA_21"], color="#2ecc71", lw=1,   linestyle="--", label="MA 21")
plt.title(f"{TICKER} Closing Price with Moving Averages", fontweight="bold")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stock_price_chart.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Price chart saved.")

# ── 4. PREPARE DATA ───────────────────────────────────────────
features = ["Open", "High", "Low", "Close", "Volume",
            "MA_7", "MA_21", "Lag_1", "Lag_2", "Lag_3", "Price_Range"]

X = df[features].values
y = df["Target"].values

# Scale data to 0–1 range
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Time-ordered split (80% train, 20% test) — NO shuffle for time series!
split    = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
dates_test      = df.index[split:]
print(f"\n✂️  Train: {len(X_train)} days | Test: {len(X_test)} days")

# ── 5. TRAIN & EVALUATE MODELS ────────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 55)
print("             MODEL RESULTS")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred_scaled  = model.predict(X_test)
    pred_actual  = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    true_actual  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(true_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(true_actual, pred_actual))
    r2   = r2_score(true_actual, pred_actual)

    results[name] = {"model": model, "pred": pred_actual, "true": true_actual,
                     "MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n🔹 {name}")
    print(f"   MAE  : ${mae:.2f}   (average dollar error)")
    print(f"   RMSE : ${rmse:.2f}   (root mean squared error)")
    print(f"   R²   : {r2:.4f}    (1.0 = perfect)")

# ── 6. ACTUAL vs PREDICTED PLOTS ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f"{TICKER} — Actual vs Predicted Price (Test Set)", fontsize=13, fontweight="bold")

line_colors = ["#e74c3c", "#2ecc71"]
for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    ax.plot(dates_test, res["true"], color="#3498db", lw=1.5, label="Actual")
    ax.plot(dates_test, res["pred"], color=line_colors[i], lw=1.5,
            linestyle="--", label="Predicted")
    ax.set_title(f"{name}\nMAE=${res['MAE']:.2f}  R²={res['R2']:.4f}", fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stock_predictions.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Prediction plot saved.")

# ── 7. NEXT DAY PREDICTION ────────────────────────────────────
print("\n" + "=" * 55)
print("         NEXT DAY PRICE PREDICTION")
print("=" * 55)

best_name  = min(results, key=lambda k: results[k]["RMSE"])
best_model = results[best_name]["model"]
last_row   = scaler_X.transform(df[features].iloc[[-1]].values)
next_price = scaler_y.inverse_transform(
                best_model.predict(last_row).reshape(-1, 1))[0][0]
last_close = float(df["Close"].iloc[-1])
change     = ((next_price - last_close) / last_close) * 100

print(f"\n  Stock      : {TICKER}")
print(f"  Last Close : ${last_close:.2f}")
print(f"  Predicted  : ${next_price:.2f}  ({best_name})")
print(f"  Direction  : {'📈 UP' if change > 0 else '📉 DOWN'}  ({change:+.2f}%)")

# ── 8. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("             FINAL SUMMARY")
print("=" * 55)
print(f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
print("  " + "-" * 46)
for name, r in results.items():
    print(f"  {name:<22} ${r['MAE']:>5.2f}  ${r['RMSE']:>5.2f}  {r['R2']:>6.4f}")
print(f"\n🏆 Best Model : {best_name}")
print("✅ Experiment complete!")
