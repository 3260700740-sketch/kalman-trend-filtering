import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from kalman_filter import KalmanFilter1D


# ============================================================
# C Stage: Kalman Filter on Real Financial Data
# ============================================================

# -----------------------------
# 1. Download real market data
# -----------------------------
ticker = "SPY"          # 你也可以换成 "AAPL"
start_date = "2018-01-01"
end_date = "2024-01-01"

data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# 使用 Close（最稳定、最不容易出问题）
price_series = data["Close"].dropna()
dates = price_series.index
price = price_series.values


# -----------------------------
# 2. Kalman filter runner
# -----------------------------
def run_kalman(price_array, q, r):
    kf = KalmanFilter1D(
        q=q,
        r=r,
        initial_state=price_array[0]
    )

    filtered = []
    for p in price_array:
        kf.predict()
        kf.update(p)
        filtered.append(kf.x)

    return np.array(filtered)


# -----------------------------
# 3. Representative parameters
# -----------------------------
r_fixed = 1.0
q_values = [0.01, 0.1, 0.3]


filtered_results = {
    q: run_kalman(price, q=q, r=r_fixed)
    for q in q_values
}


# -----------------------------
# 4. Visualization
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(dates, price, label="Observed Price", alpha=0.4)

for q in q_values:
    plt.plot(
        dates,
        filtered_results[q],
        label=f"Kalman Trend (q={q})",
        linewidth=2
    )

plt.title(f"Kalman Filter Trend Estimation on {ticker}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()