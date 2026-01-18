import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter1D


def run_kf(observed_price, q, r):
    kf = KalmanFilter1D(q=q, r=r, initial_state=observed_price[0])
    filtered = []
    for price in observed_price:
        kf.predict()
        kf.update(price)
        filtered.append(kf.x)
    return np.array(filtered)


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))


# -----------------------------
# 1) Controlled synthetic data (fixed seed)
# -----------------------------
np.random.seed(42)
n = 80
true_trend = np.linspace(100, 112, n)
noise = np.random.normal(0, 1.0, size=n)
observed_price = true_trend + noise

# -----------------------------
# 2) Sensitivity setup
# -----------------------------
r_fixed = 1.0
q_list = [0.001, 0.01, 0.1, 1.0]

# -----------------------------
# 3) Run and collect results
# -----------------------------
results = []
filtered_map = {}

for q in q_list:
    filtered = run_kf(observed_price, q=q, r=r_fixed)
    filtered_map[q] = filtered
    results.append((q, rmse(filtered, true_trend)))

# -----------------------------
# 4) Plot: price / true trend / filtered (multiple q)
# -----------------------------
plt.figure(figsize=(11, 5))
plt.plot(observed_price, label="Observed Price", alpha=0.45)
plt.plot(true_trend, label="True Trend", linestyle="--")

for q in q_list:
    plt.plot(filtered_map[q], label=f"Filtered (q={q}, r={r_fixed})", linewidth=2)

plt.title("Kalman Filter Sensitivity to Process Noise q (r fixed)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Print RMSE table (for write-up)
# -----------------------------
print("RMSE vs True Trend (lower is better in this synthetic setup):")
for q, e in results:
    print(f"q={q:<6}  r={r_fixed:<4}  RMSE={e:.4f}")

# ============================================================
# Extended sensitivity analysis (log-scale q grid)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter1D


# ============================================================
# Utility functions
# ============================================================

def run_kf(observed_price, q, r):
    """
    Run 1D Kalman Filter on a price series.
    """
    kf = KalmanFilter1D(q=q, r=r, initial_state=observed_price[0])
    filtered = []

    for price in observed_price:
        kf.predict()
        kf.update(price)
        filtered.append(kf.x)

    return np.array(filtered)


def rmse(a, b):
    """
    Root Mean Squared Error
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))


# ============================================================
# 1. Controlled synthetic data (fixed seed)
# ============================================================

np.random.seed(42)

n = 80
true_trend = np.linspace(100, 112, n)
noise = np.random.normal(0, 1.0, size=n)
observed_price = true_trend + noise


# ============================================================
# 2. Sensitivity setup
# ============================================================

r_fixed = 1.0

# Log-scale q grid (dense for RMSE, sparse for visualization)
q_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
q_plot = [1e-3, 1e-2, 1e-1, 1.0]


# ============================================================
# 3. Run Kalman Filter for each q
# ============================================================

filtered_map = {}
rmse_results = []

for q in q_list:
    filtered = run_kf(observed_price, q=q, r=r_fixed)
    filtered_map[q] = filtered
    rmse_results.append((q, rmse(filtered, true_trend)))


# ============================================================
# 4. Visualization (representative q only)
# ============================================================

plt.figure(figsize=(11, 5))

plt.plot(observed_price, label="Observed Price", alpha=0.45)
plt.plot(true_trend, label="True Trend", linestyle="--")

for q in q_plot:
    plt.plot(
        filtered_map[q],
        linewidth=2,
        label=f"Filtered (q={q}, r={r_fixed})"
    )

plt.title("Kalman Filter Sensitivity to Process Noise q (r fixed)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 5. Quantitative comparison (RMSE table)
# ============================================================

print("\nRMSE vs True Trend (synthetic data)")
print("-" * 45)
print(f"{'q':>10} | {'r':>5} | {'RMSE':>10}")
print("-" * 45)

for q, e in rmse_results:
    print(f"{q:>10.4g} | {r_fixed:>5.2f} | {e:>10.4f}")

print("-" * 45)