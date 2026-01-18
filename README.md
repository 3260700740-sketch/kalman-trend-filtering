# kalman-trend-filtering
Trend estimation under noisy financial time series using state-space models and Kalman filtering, with backtesting validation.
# Trend Estimation under Noise with Kalman Filtering

## Overview
This project studies trend estimation in noisy financial time series using a linear Gaussian state-space model and Kalman filtering.  
Rather than focusing on alpha optimization, the goal is to examine whether signal extraction under noise has economically meaningful implications when validated through backtesting.

## Mathematical Setup
We model the observed price series as the sum of a latent trend component and observation noise.

State equation:
x_t = x_{t-1} + η_t

Observation equation:
y_t = x_t + ε_t

where η_t and ε_t are assumed to be Gaussian white noise processes.

## Methodology
- Formulate a state-space model for financial price dynamics
- Apply Kalman filtering to estimate the latent trend in real time
- Construct a simple rule-based trading strategy based on the estimated trend
- Validate the signal through historical backtesting

## Backtesting
Backtesting is conducted using the Backtrader framework, which is used strictly as an execution and evaluation engine rather than a modeling component.

## Scope and Limitations
This project is intended as an applied financial mathematics prototype rather than a production trading system.  
Model assumptions (linearity, Gaussian noise) are made explicit and are not optimized for profitability.

## Planned Extensions
- Regime-dependent noise modeling
- Volatility-aware position sizing
- Multi-asset extension
