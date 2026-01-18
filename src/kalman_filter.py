import numpy as np


class KalmanFilter1D:
    """
    One-dimensional Kalman Filter for trend estimation
    under a linear Gaussian state-space model.

    State equation:
        x_t = x_{t-1} + eta_t,    eta_t ~ N(0, Q)

    Observation equation:
        y_t = x_t + epsilon_t,    epsilon_t ~ N(0, R)
    """

    def __init__(self, q: float, r: float, initial_state: float):
        """
        Parameters
        ----------
        q : float
            Process noise variance (Q)
        r : float
            Observation noise variance (R)
        initial_state : float
            Initial estimate of the latent state x_0
        """
        self.q = q
        self.r = r

        # State estimate x_t
        self.x = initial_state

        # State covariance P_t
        self.p = 1.0

    def predict(self):
        """
        Prediction step:
            x_{t|t-1} = x_{t-1|t-1}
            P_{t|t-1} = P_{t-1|t-1} + Q
        """
        self.p = self.p + self.q

    def update(self, observation: float):
        """
        Update step:
            K_t = P / (P + R)
            x_t = x + K_t (y_t - x)
            P_t = (1 - K_t) P
        """
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (observation - self.x)
        self.p = (1 - k) * self.p