import matplotlib
import numpy as np
from kalman_filter import KalmanFilter1D
import matplotlib.pyplot as plt

# -----------------------------
# 1. 构造一个“有趋势 + 有噪声”的价格序列
# -----------------------------
np.random.seed(42)

true_trend = np.linspace(100, 110, 50)      # 潜在真实趋势
noise = np.random.normal(0, 1.0, size=50)  # 市场噪声
observed_price = true_trend + noise


# -----------------------------
# 2. 初始化 Kalman Filter
# -----------------------------
kf = KalmanFilter1D(
    q=0.01,                 # 趋势变化不确定性（较小）
    r=1.0,                  # 观测噪声
    initial_state=observed_price[0]
)


# -----------------------------
# 3. 逐步更新滤波器
# -----------------------------
filtered_trend = []

for price in observed_price:
    kf.predict()
    kf.update(price)
    filtered_trend.append(kf.x)


# -----------------------------
# 4. 打印前几步结果（用于理解）
# -----------------------------
for i in range(5):
    print(
        f"t={i:2d} | "
        f"price={observed_price[i]:.2f} | "
        f"filtered_trend={filtered_trend[i]:.2f}"
    )

    # -----------------------------
    # 5. Visualization
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(observed_price, label="Observed Price", alpha=0.6)
    plt.plot(true_trend, label="True Trend", linestyle="--")
    plt.plot(filtered_trend, label="Kalman Filtered Trend", linewidth=2)

    plt.title("Kalman Filter Trend Estimation")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()