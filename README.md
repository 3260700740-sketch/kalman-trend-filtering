# kalman-trend-filtering
Trend estimation under noisy financial time series using state-space models and Kalman filtering, with backtesting validation.
# Kalman-Based Trend Estimation in Financial Time Series

## Overview

This project studies trend estimation in noisy financial time series using a linear Gaussian state-space model and Kalman filtering.  
Rather than focusing on return prediction or trading optimization, the goal is to examine how latent trend signals can be extracted under noise and how modeling assumptions affect the resulting trend estimates.

The project follows a structured workflow:
1. Implementing a Kalman filter for trend estimation from scratch.
2. Validating model behavior on synthetic data with known ground truth.
3. Performing parameter sensitivity analysis on the process noise assumption.
4. Applying the model to real equity price data.

---

## Mathematical Setup

We model the observed price series as the sum of a latent trend component and observation noise:

State equation:  
x_t = x_{t-1} + η_t, η_t ~ N(0, Q)

Observation equation:  
y_t = x_t + ε_t, ε_t ~ N(0, R)

Here, x_t represents the unobserved trend, while y_t is the observed market price.  
The process noise variance Q controls how rapidly the trend is allowed to evolve, and the observation noise variance R captures short-term market fluctuations.

---

## Methodology

A one-dimensional Kalman filter is implemented to recursively estimate the latent trend under the linear Gaussian state-space assumption.  
The filter consists of a prediction step, which propagates uncertainty in the trend, and an update step, which incorporates new price observations.

All model components and parameters are explicitly defined and implemented without relying on black-box libraries, ensuring full interpretability.

---

## Experiments

### A. Synthetic Data Validation

The Kalman filter is first validated on synthetic price data with a known underlying trend.  
This controlled setup allows direct inspection of estimation behavior and confirms that the filter can recover low-frequency trend structure while suppressing high-frequency noise.

---

### B. Parameter Sensitivity Analysis

A systematic sensitivity analysis is conducted on the process noise parameter Q while keeping the observation noise R fixed.  
Results show that smaller values of Q enforce smoother but more lagged trend estimates, while larger values of Q produce more responsive trends at the cost of increased noise.

Quantitative evaluation using RMSE on synthetic data reveals a clear bias–variance tradeoff, with an intermediate range of Q providing the best balance between smoothness and responsiveness.

---

### C. Application to Real Market Data

The model is then applied to real equity price data (SPY).  
Since the true trend is unobservable in real markets, evaluation focuses on qualitative consistency rather than numerical error metrics.

Observed trend estimates under different Q values exhibit behavior consistent with the synthetic experiments: lower Q values yield smoother and slower-moving trends, while higher Q values react more quickly to market movements.  
This consistency supports the external validity of the sensitivity analysis.

---

## Key Findings

- Kalman filtering provides a principled framework for trend estimation under noisy financial time series.
- Trend estimates are highly sensitive to the process noise assumption, reflecting a clear bias–variance tradeoff.
- Sensitivity patterns observed in synthetic data carry over qualitatively to real market data.
- Model interpretability is preserved throughout, with all parameters having clear statistical meaning.

---

## Limitations and Scope

This project focuses on trend estimation rather than trading performance.  
No transaction costs, portfolio construction, or predictive evaluation is considered.  
The linear Gaussian assumption may not fully capture real market dynamics, and extensions to regime-switching or volatility-aware models are left for future work.

---

## Repository Structure

- src/kalman_filter.py: Core Kalman filter implementation.
- src/test_kalman_filter.py: Basic validation on synthetic data.
- src/kalman_sensitivity.py: Parameter sensitivity analysis.
- src/kalman_real_data.py: Application to real market data.
