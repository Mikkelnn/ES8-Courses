import numpy as np
from typing import Callable, Tuple, Optional


class KalmanFilter:
    """
    State-space Kalman filter supporting classic (linear), extended, and unscented variants.

    State-space model:
        x[k+1] = A*x[k] + B*u[k] + w[k],  w ~ N(0, Q)
        z[k] = C*x[k] + D*u[k] + v[k],    v ~ N(0, R)
    """

    def __init__(self, extended: bool = False, scented: bool = False):
        """
        Initialize Kalman filter.

        Args:
            extended: True for extended Kalman filter
            scented: True for unscented Kalman filter
            Both False: classic (linear) Kalman filter
        """
        if extended and scented:
            raise ValueError("Cannot be both extended and scented")

        if extended:
            self.mode = "extended"
        elif scented:
            self.mode = "unscented"
        else:
            self.mode = "classic"

        # State variables
        self.x = None  # State estimate [n x 1]
        self.P = None  # Covariance estimate [n x n]

        # System matrices
        self.A = None  # State transition [n x n]
        self.B = None  # Input matrix [n x m]
        self.C = None  # Measurement matrix [p x n]
        self.D = None  # Feedthrough [p x m]

        # Noise covariances
        self.Q = None  # Process noise [n x n]
        self.R = None  # Measurement noise [p x p]

        # Extended Kalman filter
        self.f = None  # Nonlinear state function: x_next = f(x, u)
        self.h = None  # Nonlinear measurement function: z = h(x, u)
        self.F_jacobian = None  # Jacobian of f w.r.t. state
        self.H_jacobian = None  # Jacobian of h w.r.t. state

        # Unscented Kalman filter
        self.alpha = 1e-3  # Spread of sigma points
        self.beta = 2.0    # Distribution info (2 for Gaussian)
        self.kappa = 0.0   # Secondary scaling
        self.gamma = None  # Sigma point scaling

        self._setup_complete = False

    def setup(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
              A: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None,
              C: Optional[np.ndarray] = None, D: Optional[np.ndarray] = None,
              f: Optional[Callable] = None, h: Optional[Callable] = None,
              F_jacobian: Optional[Callable] = None, H_jacobian: Optional[Callable] = None) -> None:
        """
        Setup Kalman filter with initial state and system matrices.

        Args:
            x0: Initial state estimate [n x 1]
            P0: Initial covariance [n x n]
            Q: Process noise covariance [n x n]
            R: Measurement noise covariance [p x p]
            A: State transition matrix [n x n] (classic/linear only)
            B: Input matrix [n x m] (classic/linear only)
            C: Measurement matrix [p x n] (classic/linear only)
            D: Feedthrough matrix [p x m] (optional)
            f: Nonlinear state function (extended/unscented)
            h: Nonlinear measurement function (extended/unscented)
            F_jacobian: Jacobian of f (extended only)
            H_jacobian: Jacobian of h (extended only)
        """
        self.x = x0.reshape(-1, 1) if x0.ndim == 1 else x0
        self.P = P0
        self.Q = Q
        self.R = R

        n = self.x.shape[0]

        if self.mode == "classic":
            if A is None:
                raise ValueError("classic mode requires state matrix A")
            self.A = A
            self.B = B if B is not None else np.zeros((n, 1))
            self.C = C if C is not None else np.eye(n)
            self.D = D if D is not None else np.zeros((self.C.shape[0], self.B.shape[1]))

        elif self.mode == "extended":
            if f is None or h is None:
                raise ValueError("extended mode requires functions f and h")
            if F_jacobian is None or H_jacobian is None:
                raise ValueError("extended mode requires Jacobians F_jacobian and H_jacobian")
            self.f = f
            self.h = h
            self.F_jacobian = F_jacobian
            self.H_jacobian = H_jacobian

        elif self.mode == "unscented":
            if f is None or h is None:
                raise ValueError("unscented mode requires functions f and h")
            self.f = f
            self.h = h
            # Precompute lambda and weights
            self.gamma = np.sqrt(n + self.kappa)
            self.Wm0 = self.kappa / (n + self.kappa)
            self.Wc0 = self.kappa / (n + self.kappa) + (1 - self.alpha**2 + self.beta)
            self.Wmi = 1 / (2 * (n + self.kappa))
            self.Wci = 1 / (2 * (n + self.kappa))

        self._setup_complete = True

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """Prediction step: x[k+1|k], P[k+1|k]"""
        if not self._setup_complete:
            raise RuntimeError("Call setup() first")

        assert self.x is not None and self.P is not None and self.Q is not None

        if u is None:
            if self.mode == "classic":
                assert self.B is not None
                u = np.zeros((self.B.shape[1], 1))
            else:
                u = np.zeros((0, 1))
        u = u.reshape(-1, 1) if u.ndim == 1 else u

        if self.mode == "classic":
            assert self.A is not None and self.B is not None
            # Linear prediction: x = A*x + B*u
            self.x = self.A @ self.x + self.B @ u
            # Covariance: P = A*P*A^T + Q
            self.P = self.A @ self.P @ self.A.T + self.Q

        elif self.mode == "extended":
            assert self.f is not None and self.F_jacobian is not None
            # Nonlinear prediction: x = f(x, u)
            self.x = self.f(self.x, u).reshape(-1, 1)
            # Linearize: F = df/dx
            F = self.F_jacobian(self.x, u)
            # Covariance: P = F*P*F^T + Q
            self.P = F @ self.P @ F.T + self.Q

        elif self.mode == "unscented":
            assert self.f is not None
            # Generate sigma points
            sigma_points = self._generate_sigma_points(self.x, self.P)
            # Propagate sigma points: x_propagated = f(x_sigma, u)
            sigma_propagated = np.column_stack([
                self.f(sigma_points[:, i:i+1], u) for i in range(sigma_points.shape[1])
            ])
            # Compute mean and covariance from propagated sigma points
            self.x, self.P = self._compute_mean_cov(sigma_propagated, self.Q)

    def update(self, z: np.ndarray) -> None:
        """Update step: x[k|k], P[k|k]"""
        if not self._setup_complete:
            raise RuntimeError("Call setup() first")

        assert self.x is not None and self.P is not None and self.R is not None

        z = z.reshape(-1, 1) if z.ndim == 1 else z

        if self.mode == "classic":
            assert self.A is not None and self.C is not None
            # Innovation: y = z - C*x
            y = z - self.C @ self.x
            # Innovation covariance: S = C*P*C^T + R
            S = self.C @ self.P @ self.C.T + self.R
            # Kalman gain: K = P*C^T*S^-1
            K = self.P @ self.C.T @ np.linalg.inv(S)
            # Update state: x = x + K*y
            self.x = self.x + K @ y
            # Update covariance: P = (I - K*C)*P
            n = self.x.shape[0]
            self.P = (np.eye(n) - K @ self.C) @ self.P

        elif self.mode == "extended":
            assert self.h is not None and self.H_jacobian is not None
            # Innovation: y = z - h(x)
            z_pred = self.h(self.x)
            y = z - z_pred.reshape(-1, 1)
            # Linearize: H = dh/dx
            H = self.H_jacobian(self.x)
            # Innovation covariance: S = H*P*H^T + R
            S = H @ self.P @ H.T + self.R
            # Kalman gain: K = P*H^T*S^-1
            K = self.P @ H.T @ np.linalg.inv(S)
            # Update state: x = x + K*y
            self.x = self.x + K @ y
            # Update covariance: P = (I - K*H)*P
            n = self.x.shape[0]
            self.P = (np.eye(n) - K @ H) @ self.P

        elif self.mode == "unscented":
            assert self.h is not None
            # Generate sigma points from predicted state
            sigma_points = self._generate_sigma_points(self.x, self.P)
            # Measurement sigma points: z_sigma = h(x_sigma)
            z_sigma = np.column_stack([
                self.h(sigma_points[:, i:i+1]) for i in range(sigma_points.shape[1])
            ])
            # Mean and covariance of measurements
            z_mean, Pzz = self._compute_mean_cov(z_sigma, self.R)
            # Cross-covariance: Pxz = E[(x - x_mean)(z - z_mean)^T]
            Pxz = self._compute_cross_cov(sigma_points, z_sigma)
            # Kalman gain: K = Pxz * Pzz^-1
            K = Pxz @ np.linalg.inv(Pzz)
            # Innovation: y = z - z_mean
            y = z - z_mean.reshape(-1, 1)
            # Update state: x = x + K*y
            self.x = self.x + K @ y
            # Update covariance: P = P - K*Pzz*K^T
            self.P = self.P - K @ Pzz @ K.T

    def iterate(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> None:
        """Single iteration: predict then update"""
        self.predict(u)
        self.update(z)

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points for unscented transform"""
        n = x.shape[0]
        sigma = np.zeros((n, 2*n + 1))

        # Mean sigma point
        sigma[:, 0:1] = x

        # Compute square root of P
        sqrt_P = np.linalg.cholesky((n + self.kappa) * P)

        # Spread points around mean
        for i in range(n):
            sigma[:, i+1:i+2] = x + sqrt_P[:, i:i+1]
            sigma[:, n+i+1:n+i+2] = x - sqrt_P[:, i:i+1]

        return sigma

    def _compute_mean_cov(self, sigma: np.ndarray, noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted mean and covariance from sigma points"""
        # Weighted mean
        mean = self.Wm0 * sigma[:, 0:1]
        for i in range(1, sigma.shape[1]):
            mean = mean + self.Wmi * sigma[:, i:i+1]

        # Weighted covariance
        cov = self.Wc0 * ((sigma[:, 0:1] - mean) @ (sigma[:, 0:1] - mean).T)
        for i in range(1, sigma.shape[1]):
            cov = cov + self.Wci * ((sigma[:, i:i+1] - mean) @ (sigma[:, i:i+1] - mean).T)

        cov = cov + noise_cov

        return mean, cov

    def _compute_cross_cov(self, sigma_x: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        """Compute cross-covariance between state and measurement sigma points"""
        # Means
        x_mean = self.Wm0 * sigma_x[:, 0:1]
        for i in range(1, sigma_x.shape[1]):
            x_mean = x_mean + self.Wmi * sigma_x[:, i:i+1]

        z_mean = self.Wm0 * sigma_z[:, 0:1]
        for i in range(1, sigma_z.shape[1]):
            z_mean = z_mean + self.Wmi * sigma_z[:, i:i+1]

        # Cross-covariance
        cov = self.Wc0 * ((sigma_x[:, 0:1] - x_mean) @ (sigma_z[:, 0:1] - z_mean).T)
        for i in range(1, sigma_x.shape[1]):
            cov = cov + self.Wci * ((sigma_x[:, i:i+1] - x_mean) @ (sigma_z[:, i:i+1] - z_mean).T)

        return cov

    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        assert self.x is not None
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current covariance estimate"""
        assert self.P is not None
        return self.P.copy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # System parameters: HIGHLY NONLINEAR system
    dt = 0.05
    T = 8.0
    t = np.arange(0, T, dt)
    N = len(t)

    # True state: 2D nonlinear pendulum with strong nonlinearity
    x_true = np.zeros((2, N))
    x_true[0, 0] = np.pi / 2  # Large initial angle
    x_true[1, 0] = 0.0
    for k in range(N - 1):
        pos = x_true[0, k]
        vel = x_true[1, k]
        # Highly nonlinear: strong sin(pos) coupling
        new_pos = pos + vel * dt
        new_vel = vel - 3.0 * np.sin(pos) * dt - 0.15 * vel * dt
        x_true[0, k + 1] = new_pos
        x_true[1, k + 1] = new_vel

    # Highly nonlinear measurement: arctangent
    z_clean = np.arctan(2.0 * x_true[0, :])
    z = z_clean + np.random.normal(0, 0.25, N)

    x0 = np.array([np.tan(z[0]) / 2.0, 0.0])
    P0 = np.eye(2)
    Q = 0.005 * np.eye(2)
    R = np.array([[0.0625]])

    # ============ CLASSIC KALMAN FILTER (Linear approximation) ============
    A_classic = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[2.0, 0.0]])  # Linear approx of arctan

    kf_classic = KalmanFilter(extended=False, scented=False)
    kf_classic.setup(x0, P0, Q, R, A=A_classic, C=C)

    x_classic = np.zeros((2, N))
    x_classic[:, 0] = x0
    t_classic_start = time.perf_counter()
    for k in range(N - 1):
        kf_classic.predict()
        kf_classic.update(np.array([z[k]]))
        x_classic[:, k + 1] = kf_classic.get_state().flatten()
    t_classic = (time.perf_counter() - t_classic_start) * 1000

    # ============ EXTENDED KALMAN FILTER ============
    def f_ekf(x, u):
        x_val = x.flatten()
        pos, vel = x_val[0], x_val[1]
        new_pos = pos + vel * dt
        new_vel = vel - 3.0 * np.sin(pos) * dt - 0.15 * vel * dt
        return np.array([new_pos, new_vel])

    def F_ekf(x, u):
        x_val = x.flatten()
        pos = x_val[0]
        J = np.array([[1.0, dt], [-3.0 * np.cos(pos) * dt, 1.0 - 0.15 * dt]])
        return J

    def h_ekf(x):
        x_val = x.flatten()
        pos = x_val[0]
        return np.array([[np.arctan(2.0 * pos)]])

    def H_ekf(x):
        x_val = x.flatten()
        pos = x_val[0]
        return np.array([[2.0 / (1.0 + (2.0 * pos) ** 2), 0.0]])

    kf_ekf = KalmanFilter(extended=True, scented=False)
    kf_ekf.setup(x0, P0, Q, R, f=f_ekf, h=h_ekf, F_jacobian=F_ekf, H_jacobian=H_ekf)

    x_ekf = np.zeros((2, N))
    x_ekf[:, 0] = x0
    t_ekf_start = time.perf_counter()
    for k in range(N - 1):
        kf_ekf.predict()
        kf_ekf.update(np.array([z[k]]))
        x_ekf[:, k + 1] = kf_ekf.get_state().flatten()
    t_ekf = (time.perf_counter() - t_ekf_start) * 1000

    # ============ UNSCENTED KALMAN FILTER ============
    def f_ukf(x, u):
        x_val = x.flatten()
        pos, vel = x_val[0], x_val[1]
        new_pos = pos + vel * dt
        new_vel = vel - 3.0 * np.sin(pos) * dt - 0.15 * vel * dt
        return np.array([new_pos, new_vel])

    def h_ukf(x):
        x_val = x.flatten()
        pos = x_val[0]
        return np.array([[np.arctan(2.0 * pos)]])

    kf_ukf = KalmanFilter(extended=False, scented=True)
    kf_ukf.setup(x0, P0, Q, R, f=f_ukf, h=h_ukf)

    x_ukf = np.zeros((2, N))
    x_ukf[:, 0] = x0
    t_ukf_start = time.perf_counter()
    for k in range(N - 1):
        kf_ukf.predict()
        kf_ukf.update(np.array([z[k]]))
        x_ukf[:, k + 1] = kf_ukf.get_state().flatten()
    t_ukf = (time.perf_counter() - t_ukf_start) * 1000

    # ============ PLOTTING ============
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Position
    axes[0].plot(t, x_true[0, :], "k-", linewidth=2.5, label="True state")
    axes[0].plot(t, z / 2.0, "r.", markersize=4, alpha=0.4, label="Noisy arctan measurement")
    axes[0].plot(t, x_classic[0, :], "b-", linewidth=2, label=f"Classic KF ({t_classic:.1f}ms)")
    axes[0].plot(t, x_ekf[0, :], "g--", linewidth=2, label=f"Extended KF ({t_ekf:.1f}ms)")
    axes[0].plot(t, x_ukf[0, :], "m:", linewidth=2.5, label=f"Unscented KF ({t_ukf:.1f}ms)")
    axes[0].set_ylabel("Position (rad)", fontsize=11, fontweight='bold')
    axes[0].set_title("Highly Nonlinear Pendulum with Arctan Measurement", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Velocity
    axes[1].plot(t, x_true[1, :], "k-", linewidth=2.5, label="True")
    axes[1].plot(t, x_classic[1, :], "b-", linewidth=2, label="Classic KF")
    axes[1].plot(t, x_ekf[1, :], "g--", linewidth=2, label="Extended KF")
    axes[1].plot(t, x_ukf[1, :], "m:", linewidth=2.5, label="Unscented KF")
    axes[1].set_ylabel("Velocity (rad/s)", fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Error comparison
    err_classic_pos = np.abs(x_classic[0, :] - x_true[0, :])
    err_ekf_pos = np.abs(x_ekf[0, :] - x_true[0, :])
    err_ukf_pos = np.abs(x_ukf[0, :] - x_true[0, :])
    axes[2].semilogy(t, err_classic_pos, "b-", linewidth=2, label=f"Classic RMSE: {np.sqrt(np.mean(err_classic_pos**2)):.4f}")
    axes[2].semilogy(t, err_ekf_pos, "g--", linewidth=2, label=f"Extended RMSE: {np.sqrt(np.mean(err_ekf_pos**2)):.4f}")
    axes[2].semilogy(t, err_ukf_pos, "m:", linewidth=2.5, label=f"Unscented RMSE: {np.sqrt(np.mean(err_ukf_pos**2)):.4f}")
    axes[2].set_ylabel("|Position Error| (log)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("kalman_comparison.png", dpi=150, bbox_inches='tight')
    print("Plot saved to kalman_comparison.png")