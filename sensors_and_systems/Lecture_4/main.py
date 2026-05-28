import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square as square_wave
from dataclasses import dataclass


@dataclass
class SystemParams:
    a: float
    b: float
    c: float
    phif: float
    phih: float
    fu: float
    Q: float
    R: float
    x0: float
    P0: float
    n: int = 100


def make_base_params(c=1.0, phif=0.0, phih=0.0) -> SystemParams:
    a = 0.95
    k = 1.0
    b = k * (1 - a)
    fu = 0.02
    sigma_w = 0.05 * np.sqrt(1 - a**2)
    sigma_v = 0.1
    sigma_x0 = sigma_w
    return SystemParams(
        a=a,
        b=b,
        c=c,
        phif=phif,
        phih=phih,
        fu=fu,
        Q=sigma_w**2,
        R=sigma_v**2,
        x0=0.0,
        P0=sigma_x0**2,
    )


def simulate_nonlinear_system(p: SystemParams, seed=42):
    rng = np.random.default_rng(seed)
    w = np.sqrt(p.Q) * rng.standard_normal(p.n)
    v = np.sqrt(p.R) * rng.standard_normal(p.n)
    u = square_wave(np.arange(1, p.n + 1) * p.fu * 2 * np.pi)

    x = np.zeros(p.n)
    z = np.zeros(p.n)
    x[0] = np.sqrt(p.P0) * rng.standard_normal() + p.x0
    z[0] = np.sin(p.c * x[0] + p.phih) + v[0]

    for k in range(1, p.n):
        x[k] = p.a * np.sin(x[k - 1] + p.phif) + p.b * u[k - 1] + w[k - 1]
        z[k] = np.sin(p.c * x[k] + p.phih) + v[k]

    return x, z, u


def extended_kalman_filter(z, u, p: SystemParams):
    x_prior = np.zeros(p.n)
    x_posterior = np.zeros(p.n)
    z_predicted = np.zeros(p.n)
    K_log = np.zeros(p.n)
    P_prior_log = np.zeros(p.n)
    P_posterior_log = np.zeros(p.n)

    # init
    x_hat_prior = p.x0
    P_prior = p.P0

    for k in range(p.n):
        z_hat = np.sin(p.c * x_hat_prior + p.phih)

        x_prior[k] = x_hat_prior
        z_predicted[k] = z_hat
        P_prior_log[k] = P_prior

        # update
        H_k = p.c * np.cos(p.c * x_hat_prior + p.phih)
        K = P_prior * H_k / (H_k * P_prior * H_k + p.R)
        x_hat_posterior = x_hat_prior + K * (z[k] - z_hat)
        P_posterior = (1 - K * H_k) * P_prior * (1 - K * H_k) + K * p.R * K

        x_posterior[k] = x_hat_posterior
        K_log[k] = K
        P_posterior_log[k] = P_posterior

        # predict
        Phi_k = p.a * np.cos(x_hat_posterior + p.phif)
        x_hat_prior = p.a * np.sin(x_hat_posterior + p.phif) + p.b * u[k]
        P_prior = Phi_k * P_posterior * Phi_k + p.Q

    return x_prior, x_posterior, z_predicted, K_log, P_prior_log, P_posterior_log


def linear_kalman_filter(z, u, p: SystemParams):
    x_prior = np.zeros(p.n)
    x_posterior = np.zeros(p.n)
    z_predicted = np.zeros(p.n)
    K_log = np.zeros(p.n)
    P_prior_log = np.zeros(p.n)
    P_posterior_log = np.zeros(p.n)

    H = p.c
    Phi = p.a

    # init
    x_hat_prior = p.x0
    P_prior = p.P0

    for k in range(p.n):
        z_hat = H * x_hat_prior

        x_prior[k] = x_hat_prior
        z_predicted[k] = z_hat
        P_prior_log[k] = P_prior

        # update
        K = P_prior * H / (H * P_prior * H + p.R)
        x_hat_posterior = x_hat_prior + K * (z[k] - z_hat)
        P_posterior = (1 - K * H) * P_prior * (1 - K * H) + K * p.R * K

        x_posterior[k] = x_hat_posterior
        K_log[k] = K
        P_posterior_log[k] = P_posterior

        # predict
        x_hat_prior = p.a * x_hat_posterior + p.b * u[k]
        P_prior = Phi * P_posterior * Phi + p.Q

    return x_prior, x_posterior, z_predicted, K_log, P_prior_log, P_posterior_log


def rmse(error):
    return np.sqrt(np.mean(error**2))

def run_comparison(params: SystemParams, title: str, seed=42):
    x, z, u = simulate_nonlinear_system(params, seed=seed)

    ekf_xm, ekf_xp, ekf_zm, ekf_K, ekf_Pm, ekf_Pp = extended_kalman_filter(z, u, params)
    kf_xm, kf_xp, kf_zm, kf_K, kf_Pm, kf_Pp = linear_kalman_filter(z, u, params)

    print_performance_table(title, x, z, ekf_xm, ekf_xp, ekf_zm, kf_xm, kf_xp, kf_zm)

    plot_comparison(
        title,
        x,
        z,
        u,
        ekf_xm,
        ekf_xp,
        ekf_zm,
        ekf_K,
        ekf_Pm,
        ekf_Pp,
        kf_xm,
        kf_xp,
        kf_zm,
        kf_K,
        kf_Pm,
        kf_Pp,
    )

def print_performance_table(title, x, z, ekf_xm, ekf_xp, ekf_zm, kf_xm, kf_xp, kf_zm):
    ekf_residual = z - ekf_zm
    kf_residual = z - kf_zm

    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"{'Metric':<32} {'EKF':>10} {'Linear KF':>10}")
    print(f"{'-' * 55}")
    print(
        f"{'RMSE prior state error':<32} {rmse(x - ekf_xm):>10.4f} {rmse(x - kf_xm):>10.4f}"
    )
    print(
        f"{'RMSE posterior state error':<32} {rmse(x - ekf_xp):>10.4f} {rmse(x - kf_xp):>10.4f}"
    )
    print(
        f"{'RMSE residual':<32} {rmse(ekf_residual):>10.4f} {rmse(kf_residual):>10.4f}"
    )


def plot_comparison(
    title,
    x,
    z,
    u,
    ekf_xm,
    ekf_xp,
    ekf_zm,
    ekf_K,
    ekf_Pm,
    ekf_Pp,
    kf_xm,
    kf_xp,
    kf_zm,
    kf_K,
    kf_Pm,
    kf_Pp,
):
    k_axis = np.arange(len(x))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=13)

    axes[0, 0].plot(k_axis, x, "k-", label="true x", linewidth=1.5)
    axes[0, 0].plot(k_axis, ekf_xm, "b--", label="EKF prior", linewidth=1)
    axes[0, 0].plot(k_axis, kf_xm, "r:", label="KF prior", linewidth=1)
    axes[0, 0].set_title("State x")
    axes[0, 0].legend()

    axes[0, 1].plot(k_axis, x - ekf_xm, "b-", label="EKF")
    axes[0, 1].plot(k_axis, x - kf_xm, "r-", label="KF")
    axes[0, 1].set_title("Prior state error")
    axes[0, 1].legend()

    axes[1, 0].plot(k_axis, z - ekf_zm, "b-", label="EKF")
    axes[1, 0].plot(k_axis, z - kf_zm, "r-", label="KF")
    axes[1, 0].set_title("Residual")
    axes[1, 0].legend()

    axes[1, 1].plot(k_axis, ekf_K, "b-", label="EKF")
    axes[1, 1].plot(k_axis, kf_K, "r-", label="KF")
    axes[1, 1].set_title("Kalman gain K")
    axes[1, 1].legend()

    axes[2, 0].plot(k_axis, ekf_Pm, "b-", label="EKF")
    axes[2, 0].plot(k_axis, kf_Pm, "r-", label="KF")
    axes[2, 0].set_title("Prior error covariance")
    axes[2, 0].legend()

    axes[2, 1].plot(k_axis, ekf_Pp, "b-", label="EKF")
    axes[2, 1].plot(k_axis, kf_Pp, "r-", label="KF")
    axes[2, 1].set_title("Posterior error covariance")
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison(
        make_base_params(c=1, phif=0, phih=0),
        "Task 4a: a=0.95, b=0.05, c=1,  phif=0,     phih=0,     fu=0.02",
    )
    run_comparison(
        make_base_params(c=1, phif=0, phih=np.pi / 16),
        "Task 4b: a=0.95, b=0.05, c=1,  phif=0,     phih=pi/16, fu=0.02",
    )
    run_comparison(
        make_base_params(c=1, phif=np.pi / 16, phih=0),
        "Task 4c: a=0.95, b=0.05, c=1,  phif=pi/16, phih=0,     fu=0.02",
    )
    run_comparison(
        make_base_params(c=10, phif=0, phih=0),
        "Task 4d: a=0.95, b=0.05, c=10, phif=0,     phih=0,     fu=0.02",
    )

#   4a: Both work. EKF 2x better using exact model.
#
#   4b: KF measurement model wrong — biased prediction every step. Update worsens estimate.
#
#   4c: KF dynamics model wrong — error compounds each step. Worst case.
#
#   4d: KF breaks — measurement function oscillates too fast for linear approximation. EKF slightly degrades too.
#
#   EKF re-linearises at current estimate each step, so it tracks the true system behaviour.
#   KF uses fixed linear approximation valid only at origin — breaks when system deviates.
