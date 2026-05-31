# Lecture 9 — All exercises in one file. KF/EKF/UKF ported from Lecture 4 (functional style).
# Linear system:    x[i] = a*x[i-1] + b*u[i-1] + w,        y[i] = c*x[i] + v
# Nonlinear system: x[i] = a*sin(x[i-1]+phi_f) + b*u + w,   y[i] = sin(c*x[i]+phi_h) + v

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import square as square_wave
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from normplot import plot_normplot_panel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# System Parameters
# =============================================================================

@dataclass
class SystemParams:
    # All parameters needed to simulate and filter both linear and nonlinear systems
    n:     int
    a:     float
    b:     float
    c:     float
    phi_f: float
    phi_h: float
    fu:    float
    Q:     float   # process noise variance
    R:     float   # measurement noise variance
    x0:    float
    P0:    float   # initial state variance


def make_linear_params(n=100):
    # Linear system: a=0.95, c=1, sigmaw=0.05*sqrt(1-a^2)=0.01561, sigmav=0.01
    a      = 0.95
    b      = 1.0 * (1.0 - a)
    c      = 1.0
    sigmaw = 0.05 * np.sqrt(1.0 - a**2)
    sigmav = 0.01
    return SystemParams(n=n, a=a, b=b, c=c, phi_f=0.0, phi_h=0.0,
                        fu=0.02, Q=sigmaw**2, R=sigmav**2,
                        x0=0.0, P0=sigmaw**2)


def make_nonlinear_params(n=100):
    # Nonlinear system: a=0.95, c=10 (fast oscillating measurement), sigmav=0.1
    a      = 0.95
    b      = 1.0 * (1.0 - a)
    c      = 10.0
    sigmaw = 0.05 * np.sqrt(1.0 - a**2)
    sigmav = 0.1
    return SystemParams(n=n, a=a, b=b, c=c, phi_f=0.0, phi_h=0.0,
                        fu=0.02, Q=sigmaw**2, R=sigmav**2,
                        x0=0.0, P0=sigmaw**2)


# =============================================================================
# Simulation Functions
# =============================================================================

def simulate_linear_system(p: SystemParams, seed=None):
    # Simulate x[i] = a*x[i-1] + b*u[i-1] + w,  y[i] = c*x[i] + v
    rng               = np.random.default_rng(seed)
    process_noise     = np.sqrt(p.Q) * rng.standard_normal(p.n)
    measurement_noise = np.sqrt(p.R) * rng.standard_normal(p.n)
    input_signal      = square_wave(2.0 * np.pi * p.fu * np.arange(1, p.n + 1))

    true_state    = np.zeros(p.n)
    measurement   = np.zeros(p.n)
    true_state[0] = np.sqrt(p.P0) * rng.standard_normal() + p.x0
    measurement[0] = p.c * true_state[0] + measurement_noise[0]

    for i in range(1, p.n):
        true_state[i]  = p.a * true_state[i-1] + p.b * input_signal[i-1] + process_noise[i-1]
        measurement[i] = p.c * true_state[i]    + measurement_noise[i]

    return true_state, measurement, input_signal


def simulate_nonlinear_system(p: SystemParams, seed=None):
    # Simulate x[i] = a*sin(x[i-1]+phi_f) + b*u + w,  y[i] = sin(c*x[i]+phi_h) + v
    rng               = np.random.default_rng(seed)
    process_noise     = np.sqrt(p.Q) * rng.standard_normal(p.n)
    measurement_noise = np.sqrt(p.R) * rng.standard_normal(p.n)
    input_signal      = square_wave(2.0 * np.pi * p.fu * np.arange(1, p.n + 1))

    true_state    = np.zeros(p.n)
    measurement   = np.zeros(p.n)
    true_state[0] = np.sqrt(p.P0) * rng.standard_normal() + p.x0
    measurement[0] = np.sin(p.c * true_state[0] + p.phi_h) + measurement_noise[0]

    for i in range(1, p.n):
        true_state[i]  = (p.a * np.sin(true_state[i-1] + p.phi_f)
                          + p.b * input_signal[i-1] + process_noise[i-1])
        measurement[i] = np.sin(p.c * true_state[i] + p.phi_h) + measurement_noise[i]

    return true_state, measurement, input_signal


def simulate_linear_system_with_noise_type(p: SystemParams, use_normal_process,
                                           use_normal_measurement, seed=None):
    # Linear system with switchable noise: Uniform U(-s*sqrt(3), s*sqrt(3)) has same variance as N(0,s^2)
    rng    = np.random.default_rng(seed)
    sigmaw = np.sqrt(p.Q)
    sigmav = np.sqrt(p.R)

    if use_normal_process:
        process_noise = sigmaw * rng.standard_normal(p.n)
    else:
        process_noise = sigmaw * (rng.random(p.n) - 0.5) * np.sqrt(12.0)

    if use_normal_measurement:
        measurement_noise = sigmav * rng.standard_normal(p.n)
    else:
        measurement_noise = sigmav * (rng.random(p.n) - 0.5) * np.sqrt(12.0)

    input_signal   = square_wave(2.0 * np.pi * p.fu * np.arange(1, p.n + 1))
    true_state     = np.zeros(p.n)
    measurement    = np.zeros(p.n)
    true_state[0]  = np.sqrt(p.P0) * rng.standard_normal() + p.x0
    measurement[0] = p.c * true_state[0] + measurement_noise[0]

    for i in range(1, p.n):
        true_state[i]  = p.a * true_state[i-1] + p.b * input_signal[i-1] + process_noise[i-1]
        measurement[i] = p.c * true_state[i]    + measurement_noise[i]

    return true_state, measurement, input_signal


# =============================================================================
# Kalman Filter — ported from Lecture_4/main.py
# =============================================================================

def run_kalman_filter(measurement_y, input_u, p: SystemParams):
    # Linear KF: update with K = Pm*H/(H*Pm*H+R), then predict with Phi=a (constant Jacobians)
    n = p.n
    prior_state_estimate      = np.zeros(n)
    posterior_state_estimate  = np.zeros(n)
    predicted_measurement_log = np.zeros(n)
    kalman_gain_log           = np.zeros(n)
    prior_covariance_log      = np.zeros(n)
    posterior_covariance_log  = np.zeros(n)

    H_linear   = p.c   # constant measurement Jacobian for linear system
    Phi_linear = p.a   # constant state transition Jacobian for linear system

    x_hat_prior = p.x0
    P_prior     = p.P0

    for k in range(n):
        z_hat = H_linear * x_hat_prior

        prior_state_estimate[k]      = x_hat_prior
        predicted_measurement_log[k] = z_hat
        prior_covariance_log[k]      = P_prior

        # Measurement update: Joseph form Pp = (1-KH)*Pm*(1-KH) + K*R*K
        K_gain          = P_prior * H_linear / (H_linear * P_prior * H_linear + p.R)
        x_hat_posterior = x_hat_prior + K_gain * (measurement_y[k] - z_hat)
        P_posterior     = ((1 - K_gain * H_linear) * P_prior
                          Exam/Topic_1/removeme.txt * (1 - K_gain * H_linear) + K_gain * p.R * K_gain)

        posterior_state_estimate[k] = x_hat_posterior
        kalman_gain_log[k]          = K_gain
        posterior_covariance_log[k] = P_posterior

        # Time update: Pm_next = Phi*Pp*Phi + Q
        x_hat_prior = p.a * x_hat_posterior + p.b * input_u[k]
        P_prior     = Phi_linear * P_posterior * Phi_linear + p.Q

    return dict(XHM=prior_state_estimate, YHM=predicted_measurement_log,
                XHP=posterior_state_estimate, K=kalman_gain_log,
                Pm=prior_covariance_log, Pp=posterior_covariance_log)


# =============================================================================
# Extended Kalman Filter — ported from Lecture_4/main.py
# =============================================================================

def run_extended_kalman_filter(measurement_y, input_u, p: SystemParams):
    # EKF: linearize at each step via Jacobians H=c*cos(c*x+phi_h), Phi=a*cos(x+phi_f)
    n = p.n
    prior_state_estimate      = np.zeros(n)
    posterior_state_estimate  = np.zeros(n)
    predicted_measurement_log = np.zeros(n)
    kalman_gain_log           = np.zeros(n)
    prior_covariance_log      = np.zeros(n)
    posterior_covariance_log  = np.zeros(n)

    x_hat_prior = p.x0
    P_prior     = p.P0

    for k in range(n):
        z_hat = np.sin(p.c * x_hat_prior + p.phi_h)

        prior_state_estimate[k]      = x_hat_prior
        predicted_measurement_log[k] = z_hat
        prior_covariance_log[k]      = P_prior

        # Jacobian of h(x)=sin(c*x+phi_h) at current prior
        H_k = p.c * np.cos(p.c * x_hat_prior + p.phi_h)

        # Measurement update (Joseph form)
        K_gain          = P_prior * H_k / (H_k * P_prior * H_k + p.R)
        x_hat_posterior = x_hat_prior + K_gain * (measurement_y[k] - z_hat)
        P_posterior     = ((1 - K_gain * H_k) * P_prior
                           * (1 - K_gain * H_k) + K_gain * p.R * K_gain)

        posterior_state_estimate[k] = x_hat_posterior
        kalman_gain_log[k]          = K_gain
        posterior_covariance_log[k] = P_posterior

        # Jacobian of f(x)=a*sin(x+phi_f) at current posterior, then time update
        Phi_k       = p.a * np.cos(x_hat_posterior + p.phi_f)
        x_hat_prior = p.a * np.sin(x_hat_posterior + p.phi_f) + p.b * input_u[k]
        P_prior     = Phi_k * P_posterior * Phi_k + p.Q

    return dict(XHM=prior_state_estimate, YHM=predicted_measurement_log,
                XHP=posterior_state_estimate, K=kalman_gain_log,
                Pm=prior_covariance_log, Pp=posterior_covariance_log)


# =============================================================================
# Unscented Kalman Filter — ported from Lecture_4/lecture4_exercises.py
# =============================================================================

def generate_sigma_points(mean, variance, scale):
    # 3 sigma points for scalar state: x0=mean, x1=mean+sqrt(scale*P), x2=mean-sqrt(scale*P)
    variance = max(float(variance), 1e-14)
    spread   = np.sqrt(scale * variance)
    return np.array([mean, mean + spread, mean - spread])


def run_unscented_kalman_filter(measurement_y, input_u, p: SystemParams):
    # UKF: propagate 3 sigma points through nonlinear functions — no Jacobians needed
    # alpha=1, kappa=2, beta=0 -> lambda=2, scale=3, Wm=[2/3, 1/6, 1/6], Wc=[2/3, 1/6, 1/6]
    N       = p.n
    n_state = 1
    alpha, kappa, beta = 1.0, 2.0, 0.0

    lambda_ = alpha**2 * (n_state + kappa) - n_state   # = 2
    scale   = float(n_state + lambda_)                  # = 3
    n_sigma = 2 * n_state + 1                           # = 3

    sigma_weights_mean    = np.full(n_sigma, 1.0 / (2.0 * scale))
    sigma_weights_cov     = np.full(n_sigma, 1.0 / (2.0 * scale))
    sigma_weights_mean[0] = lambda_ / scale
    sigma_weights_cov[0]  = lambda_ / scale + 1.0 - alpha**2 + beta

    prior_state_estimate      = np.zeros(N)
    predicted_measurement_log = np.zeros(N)
    posterior_state_estimate  = np.zeros(N)
    prior_covariance_log      = np.zeros(N)
    posterior_covariance_log  = np.zeros(N)
    kalman_gain_log           = np.zeros(N)

    x_hat_prior = p.x0
    P_prior     = p.P0

    for i in range(N):
        prior_state_estimate[i] = x_hat_prior
        prior_covariance_log[i] = P_prior

        # Measurement update: propagate sigma points through h(x)=sin(c*x+phi_h)
        X_sigma = generate_sigma_points(x_hat_prior, P_prior, scale)
        Y_sigma = np.sin(p.c * X_sigma + p.phi_h)
        z_hat   = float(np.dot(sigma_weights_mean, Y_sigma))
        predicted_measurement_log[i] = z_hat

        # Innovation variance S = R + sum(Wc*(Y-z_hat)^2), cross-cov Cxy = sum(Wc*(X-xhm)*(Y-z_hat))
        innovation_variance = p.R
        cross_covariance    = 0.0
        for j in range(n_sigma):
            dy                   = Y_sigma[j] - z_hat
            dx                   = X_sigma[j] - x_hat_prior
            innovation_variance += sigma_weights_cov[j] * dy * dy
            cross_covariance    += sigma_weights_cov[j] * dx * dy

        K_gain          = cross_covariance / innovation_variance
        x_hat_posterior = x_hat_prior + K_gain * (measurement_y[i] - z_hat)
        P_posterior     = max(P_prior - K_gain * innovation_variance * K_gain, 1e-14)

        posterior_state_estimate[i] = x_hat_posterior
        posterior_covariance_log[i] = P_posterior
        kalman_gain_log[i]          = K_gain

        # Time update: propagate sigma points through f(x)=a*sin(x+phi_f)+b*u
        X_sigma     = generate_sigma_points(x_hat_posterior, P_posterior, scale)
        X_next      = p.a * np.sin(X_sigma + p.phi_f) + p.b * input_u[i]
        x_hat_prior = float(np.dot(sigma_weights_mean, X_next))

        P_next = p.Q
        for j in range(n_sigma):
            dx      = X_next[j] - x_hat_prior
            P_next += sigma_weights_cov[j] * dx * dx
        P_prior = max(float(P_next), 1e-14)

    return dict(XHM=prior_state_estimate, YHM=predicted_measurement_log,
                XHP=posterior_state_estimate, K=kalman_gain_log,
                Pm=prior_covariance_log, Pp=posterior_covariance_log)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_rmse(error_array):
    # Root Mean Squared Error
    return float(np.sqrt(np.mean(error_array**2)))


def compute_autocorrelation(signal, max_lag=24):
    # Normalized autocorrelation: rho[k] = E[(x-mu)(x_k-mu)] / Var(x), lags 0..max_lag
    n         = len(signal)
    xd        = signal - np.mean(signal)
    full_corr = np.correlate(xd, xd, mode='full')
    return full_corr[n - 1: n + max_lag] / full_corr[n - 1]


def save_text_results(filename, text_lines):
    # Write list of strings as a text file into the results directory
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as fh:
        fh.write('\n'.join(text_lines))
    return filepath


# =============================================================================
# Exercise 1 — Whiteness Test
# =============================================================================

def plot_whiteness_test_panel(ax, innovations, panel_title, max_lag=24):
    # Chi-squared whiteness test: stat = N * sum(rho[1..m]^2), df=m; p>0.05 means white
    n_samples      = len(innovations)
    rho            = compute_autocorrelation(innovations, max_lag)
    ci_bound       = norm_dist.ppf(0.975) / np.sqrt(n_samples)   # 1.96 / sqrt(N)
    chi2_statistic = float(rho[1:] @ rho[1:]) * n_samples
    p_value        = 1.0 - chi2_dist.cdf(chi2_statistic, df=max_lag)

    ax.plot(np.arange(max_lag + 1), rho, '-o', markersize=3, color='steelblue')
    ax.axhline( ci_bound, linestyle='--', color='red', label=f'+-95% CI = +-{ci_bound:.3f}')
    ax.axhline(-ci_bound, linestyle='--', color='red')
    ax.set_xlabel('Lag')
    ax.set_title(f'{panel_title}\np = {p_value:.4f}  ->  '
                 f'{"WHITE" if p_value > 0.05 else "NOT white"}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return p_value, chi2_statistic


def run_exercise_1(seed=42):
    # Whiteness test: correct filters (KF, EKF, UKF) produce white innovations; KF mismatch does not
    print("\n" + "=" * 60)
    print("Exercise 1 - Whiteness Test")
    print("=" * 60)

    p_lin               = make_linear_params(n=100)
    x_lin, y_lin, u_lin = simulate_linear_system(p_lin, seed=seed)
    innovations_kf      = y_lin - run_kalman_filter(y_lin, u_lin, p_lin)['YHM']

    p_nl                = make_nonlinear_params(n=100)
    x_nl, y_nl, u_nl    = simulate_nonlinear_system(p_nl, seed=seed)
    innovations_ekf     = y_nl - run_extended_kalman_filter(y_nl, u_nl, p_nl)['YHM']
    innovations_ukf     = y_nl - run_unscented_kalman_filter(y_nl, u_nl, p_nl)['YHM']

    # Mismatch: apply linear KF (c=1) to nonlinear data (y=sin(10*x)) — wrong model
    p_mismatch          = make_nonlinear_params(n=100)
    p_mismatch.c        = 1.0
    innovations_mismatch = y_nl - run_kalman_filter(y_nl, u_nl, p_mismatch)['YHM']

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle('Exercise 1 - Whiteness Test on Innovations', fontsize=13)
    p1, _ = plot_whiteness_test_panel(axs[0], innovations_kf,        'Linear + KF (correct)')
    p2, _ = plot_whiteness_test_panel(axs[1], innovations_ekf,       'Nonlinear + EKF (correct)')
    p3, _ = plot_whiteness_test_panel(axs[2], innovations_ukf,       'Nonlinear + UKF (correct)')
    p4, _ = plot_whiteness_test_panel(axs[3], innovations_mismatch,  'Nonlinear + KF (MISMATCH)')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex1_whiteness.png'), dpi=150)

    text_lines = [
        'Exercise 1 - Whiteness Test Results',
        '=' * 60,
        f'{"Case":<40} {"p-value":>10}  {"White?":>8}',
        f'{"Linear system + KF":<40} {p1:>10.4f}  {"YES" if p1 > 0.05 else "NO":>8}',
        f'{"Nonlinear system + EKF":<40} {p2:>10.4f}  {"YES" if p2 > 0.05 else "NO":>8}',
        f'{"Nonlinear system + UKF":<40} {p3:>10.4f}  {"YES" if p3 > 0.05 else "NO":>8}',
        f'{"Nonlinear system + KF (mismatch)":<40} {p4:>10.4f}  {"YES" if p4 > 0.05 else "NO":>8}',
        '',
        'p > 0.05 -> white innovations (well-tuned filter).',
        'p < 0.05 -> correlated innovations (model mismatch).',
    ]
    print('\n'.join(text_lines))
    save_text_results('ex1_whiteness.txt', text_lines)
    plt.show()


# =============================================================================
# Exercise 2 — Normality Test
# =============================================================================


def run_exercise_2(seed=42):
    # Normality test: KF on linear system gives Gaussian innovations; EKF/UKF may deviate
    print("\n" + "=" * 60)
    print("Exercise 2 - Normality Test")
    print("=" * 60)

    p_lin               = make_linear_params(n=100)
    x_lin, y_lin, u_lin = simulate_linear_system(p_lin, seed=seed)
    innovations_kf      = y_lin - run_kalman_filter(y_lin, u_lin, p_lin)['YHM']

    p_nl              = make_nonlinear_params(n=100)
    x_nl, y_nl, u_nl  = simulate_nonlinear_system(p_nl, seed=seed)
    innovations_ekf   = y_nl - run_extended_kalman_filter(y_nl, u_nl, p_nl)['YHM']
    innovations_ukf   = y_nl - run_unscented_kalman_filter(y_nl, u_nl, p_nl)['YHM']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Exercise 2 - Normality Test (normplot)', fontsize=13)
    plot_normplot_panel(axs[0], innovations_kf,  'KF (linear system)')
    plot_normplot_panel(axs[1], innovations_ekf, 'EKF (nonlinear, c=10)')
    plot_normplot_panel(axs[2], innovations_ukf, 'UKF (nonlinear, c=10)')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex2_normality.png'), dpi=150)

    text_lines = [
        'Exercise 2 - Normality Test Results',
        '=' * 65,
        'Normality assessed visually via normplot — see ex2_normality.png.',
        'Straight-line alignment = Gaussian innovations.',
        '',
        f'{"Filter":<28} {"mean(innov)":>12} {"std(innov)":>12}',
        f'{"KF  (linear system)":<28} {np.mean(innovations_kf):>12.6f} {np.std(innovations_kf):>12.6f}',
        f'{"EKF (nonlinear system)":<28} {np.mean(innovations_ekf):>12.6f} {np.std(innovations_ekf):>12.6f}',
        f'{"UKF (nonlinear system)":<28} {np.mean(innovations_ukf):>12.6f} {np.std(innovations_ukf):>12.6f}',
        '',
        'KF innovations should be Gaussian (linear system + Gaussian noise).',
        'EKF linearization errors may cause deviations; UKF avoids linearization.',
    ]
    print('\n'.join(text_lines))
    save_text_results('ex2_normality.txt', text_lines)
    plt.show()


# =============================================================================
# Exercise 3 — Uniform vs Normal Noise
# =============================================================================

def run_exercise_3(seed=42):
    # KF is BLUE for any zero-mean noise: whiteness always passes; normality fails for uniform noise
    print("\n" + "=" * 60)
    print("Exercise 3 - Uniform vs Normal Noise")
    print("=" * 60)

    noise_cases = [
        ('Normal w, Normal v',   True,  True),
        ('Uniform w, Normal v',  False, True),
        ('Normal w, Uniform v',  True,  False),
        ('Uniform w, Uniform v', False, False),
    ]

    p   = make_linear_params(n=100)
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('Exercise 3 - Uniform vs Normal Noise (KF)', fontsize=13)

    text_lines = ['Exercise 3 - Uniform vs Normal Noise', '=' * 65,
                  'Normality assessed visually via normplot — see ex3_noise_types.png.',
                  '',
                  f'{"Case":<28} {"p_white":>10}']

    for col, (label, use_normal_process, use_normal_measurement) in enumerate(noise_cases):
        x, y, u     = simulate_linear_system_with_noise_type(
            p, use_normal_process, use_normal_measurement, seed=seed)
        innovations = y - run_kalman_filter(y, u, p)['YHM']

        # Whiteness: chi2 stat = N * sum(rho^2), CI bound = 1.96/sqrt(N)
        rho      = compute_autocorrelation(innovations, max_lag=24)
        ci_bound = norm_dist.ppf(0.975) / np.sqrt(p.n)
        p_white  = 1.0 - chi2_dist.cdf(float(rho[1:] @ rho[1:]) * p.n, df=24)

        axs[0, col].plot(np.arange(25), rho, '-o', markersize=3, color='steelblue')
        axs[0, col].axhline( ci_bound, linestyle='--', color='red')
        axs[0, col].axhline(-ci_bound, linestyle='--', color='red')
        axs[0, col].set_title(f'{label}\np_white={p_white:.3f}', fontsize=9)
        axs[0, col].grid(True, alpha=0.3)

        # Normality: normplot (visual, no test statistic)
        plot_normplot_panel(axs[1, col], innovations, label)

        text_lines.append(f'{label:<28} {p_white:>10.4f}')
        print(f'  {label:<28}  p_white={p_white:.4f}')

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex3_noise_types.png'), dpi=150)

    text_lines += ['', 'Whiteness passes for all cases (KF is BLUE).',
                   'Normality fails when uniform noise is present — visible as curved tails in normplot.']
    save_text_results('ex3_noise_types.txt', text_lines)
    plt.show()


# =============================================================================
# Exercise 4 — Effect of Sample Size N
# =============================================================================

def run_exercise_4(seed=42):
    # CI width = 2*1.96/sqrt(N) shrinks with N; larger N detects even tiny correlations
    print("\n" + "=" * 60)
    print("Exercise 4 - Effect of Sample Size N")
    print("=" * 60)

    N_values = [10, 100, 1000, 10000]

    fig, axs = plt.subplots(len(N_values), 2, figsize=(12, 3 * len(N_values)))
    fig.suptitle('Exercise 4 - Effect of N on Whiteness Test', fontsize=13)

    text_lines = ['Exercise 4 - Effect of Sample Size N', '=' * 75,
                  f'{"N":>8} {"RMSE_innov":>12} {"RMSE_xtp":>12} {"p_white":>10} {"CI_width":>10}']

    for row, n in enumerate(N_values):
        max_lag = min(24, max(3, n // 4))

        p                    = make_linear_params(n=n)
        x, y, u              = simulate_linear_system(p, seed=seed)
        res                  = run_kalman_filter(y, u, p)
        innovations          = y - res['YHM']
        posterior_state_error = x - res['XHP']

        # 95% CI bound = 1.96/sqrt(N); chi2 stat = N*sum(rho[1..m]^2)
        ci_bound        = norm_dist.ppf(0.975) / np.sqrt(n)
        rho             = compute_autocorrelation(innovations, max_lag)
        p_white         = 1.0 - chi2_dist.cdf(float(rho[1:] @ rho[1:]) * n, df=max_lag)
        rmse_innovation = compute_rmse(innovations)
        rmse_posterior  = compute_rmse(posterior_state_error)

        axs[row, 0].plot(np.arange(max_lag + 1), rho, '-o', markersize=3, color='steelblue')
        axs[row, 0].axhline( ci_bound, linestyle='--', color='red', label=f'CI=+-{ci_bound:.3f}')
        axs[row, 0].axhline(-ci_bound, linestyle='--', color='red')
        axs[row, 0].set_title(f'N={n}  p={p_white:.4f}  '
                               f'{"WHITE" if p_white > 0.05 else "NOT white"}')
        axs[row, 0].legend(fontsize=7)
        axs[row, 0].grid(True, alpha=0.3)

        n_bins = min(30, max(5, n // 10))
        axs[row, 1].hist(innovations, bins=n_bins, color='steelblue', edgecolor='white', alpha=0.8)
        axs[row, 1].set_title(f'N={n}  Innovation histogram  RMSE={rmse_innovation:.5f}')
        axs[row, 1].grid(True, alpha=0.3)

        text_lines.append(f'{n:>8} {rmse_innovation:>12.6f} {rmse_posterior:>12.6f} '
                           f'{p_white:>10.4f} {2*ci_bound:>10.4f}')
        print(f'  N={n:6d}  CI=+-{ci_bound:.4f}  p_white={p_white:.4f}  '
              f'RMSE_innov={rmse_innovation:.5f}')

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex4_n_effect.png'), dpi=150)
    save_text_results('ex4_n_effect.txt', text_lines)
    plt.show()


# =============================================================================
# Exercise 5 — EKF Joint State and Parameter Estimation
# =============================================================================

def run_ekf_for_parameter_estimation(measurement_y, input_u, p: SystemParams,
                                      a0_initial_guess, sigma_a0,
                                      parameter_evolution_coefficient, parameter_drift_noise_std):
    # Augmented EKF: state = [x; a], jointly estimates system state and parameter a
    # State transition (bilinear -> needs EKF):
    #   x_next = a_hat * x_hat + b*u   (nonlinear in augmented state [x;a])
    #   a_next = aa * a_hat            (parameter evolution model)
    # Measurement (linear in augmented state):
    #   y = c*x + v  ->  H = [c, 0]   (constant Jacobian)
    # Jacobian of state transition at posterior [xhp[0], xhp[1]]:
    #   Phi = [[a_hat, x_hat], [0, aa]]
    num_time_steps              = len(measurement_y)
    measurement_jacobian_vector = np.array([p.c, 0.0])                    # shape (2,) — constant measurement Jacobian
    augmented_process_noise_cov = np.diag([p.Q, parameter_drift_noise_std**2])  # 2x2 process noise (x, a)

    augmented_prior_state       = np.array([p.x0, a0_initial_guess])      # augmented prior state [x; a]
    prior_error_covariance      = np.diag([p.P0, sigma_a0**2])            # 2x2 initial covariance

    augmented_prior_state_log       = np.zeros((2, num_time_steps))   # prior estimates: row0=x, row1=a
    augmented_posterior_state_log   = np.zeros((2, num_time_steps))   # posterior estimates
    predicted_measurement_log       = np.zeros(num_time_steps)        # predicted measurements
    kalman_gain_vector_log          = np.zeros((2, num_time_steps))   # Kalman gain (2x1 per step)
    prior_covariance_diagonal_log   = np.zeros((2, num_time_steps))   # diagonal of prior covariance
    posterior_covariance_diagonal_log = np.zeros((2, num_time_steps)) # diagonal of posterior covariance

    predicted_measurement = float(measurement_jacobian_vector @ augmented_prior_state)

    for i in range(num_time_steps):
        augmented_prior_state_log[:, i]     = augmented_prior_state
        predicted_measurement_log[i]        = predicted_measurement
        prior_covariance_diagonal_log[:, i] = np.diag(prior_error_covariance)

        # Measurement update: S = H*Pm*H' + R,  K = Pm*H'/S
        innovation_variance  = float(measurement_jacobian_vector @ prior_error_covariance @ measurement_jacobian_vector) + p.R
        kalman_gain_vector   = (prior_error_covariance @ measurement_jacobian_vector) / innovation_variance  # shape (2,)
        augmented_posterior_state = (augmented_prior_state
                                     + kalman_gain_vector * (measurement_y[i] - predicted_measurement))

        # Joseph form: Pp = (I-KH)*Pm*(I-KH)' + R*(K*K')
        kalman_gain_times_jacobian = np.outer(kalman_gain_vector, measurement_jacobian_vector)  # (2,2)
        innovation_weighting_matrix = np.eye(2) - kalman_gain_times_jacobian
        posterior_error_covariance = (innovation_weighting_matrix @ prior_error_covariance @ innovation_weighting_matrix.T
                                      + p.R * np.outer(kalman_gain_vector, kalman_gain_vector))

        augmented_posterior_state_log[:, i]     = augmented_posterior_state
        kalman_gain_vector_log[:, i]            = kalman_gain_vector
        posterior_covariance_diagonal_log[:, i] = np.diag(posterior_error_covariance)

        # Time update: propagate nonlinearly, then compute EKF Jacobian Phi
        state_estimate_x  = augmented_posterior_state[0]
        parameter_estimate_a = augmented_posterior_state[1]
        augmented_prior_state = np.array(
            [parameter_estimate_a * state_estimate_x + p.b * input_u[i],   # x_next = a*x + b*u
             parameter_evolution_coefficient * parameter_estimate_a])        # a_next = aa*a
        state_transition_jacobian = np.array(
            [[parameter_estimate_a, state_estimate_x],   # d(x_next)/d[x,a]
             [0.0,                  parameter_evolution_coefficient]])        # d(a_next)/d[x,a]
        prior_error_covariance = (state_transition_jacobian @ posterior_error_covariance
                                  @ state_transition_jacobian.T + augmented_process_noise_cov)
        predicted_measurement  = float(measurement_jacobian_vector @ augmented_prior_state)

    return dict(XHM=augmented_prior_state_log, XHP=augmented_posterior_state_log,
                YHM=predicted_measurement_log, K=kalman_gain_vector_log,
                Pm=prior_covariance_diagonal_log, Pp=posterior_covariance_diagonal_log)


def run_exercise_5(seed=42):
    # EKF parameter estimation: augment state [x;a] so the filter tracks a alongside x
    print("\n" + "=" * 60)
    print("Exercise 5 - EKF Joint State and Parameter Estimation")
    print("=" * 60)

    linear_system_params              = make_linear_params(n=100)
    true_parameter_a                  = linear_system_params.a
    true_state_sequence, measurement_sequence, input_sequence = simulate_linear_system(
        linear_system_params, seed=seed)

    # Default case from MATLAB: aa=1, sigma_wa=0 (a is modelled as a constant)
    # The filter starts with a wrong guess a0=0.5 and must learn the true a=0.95
    a0_initial_guess             = 0.5
    initial_uncertainty_on_a     = 0.1    # initial uncertainty on a
    parameter_evolution_coeff    = 1.0    # a(i) = 1*a(i-1): random walk with zero noise = constant
    parameter_drift_noise_std    = 0.0    # no drift on parameter

    ekf_param_estimation_result = run_ekf_for_parameter_estimation(
        measurement_sequence, input_sequence, linear_system_params,
        a0_initial_guess, initial_uncertainty_on_a,
        parameter_evolution_coeff, parameter_drift_noise_std)

    state_prior_estimate_sequence     = ekf_param_estimation_result['XHM'][0, :]
    parameter_prior_estimate_sequence = ekf_param_estimation_result['XHM'][1, :]
    state_posterior_estimate_sequence = ekf_param_estimation_result['XHP'][0, :]
    innovation_sequence               = measurement_sequence - ekf_param_estimation_result['YHM']

    print(f'  True a:           {true_parameter_a:.5f}')
    print(f'  Initial guess a0: {a0_initial_guess:.5f}')
    print(f'  Final estimate a: {parameter_prior_estimate_sequence[-1]:.5f}')
    print(f'  Final error:      {abs(parameter_prior_estimate_sequence[-1] - true_parameter_a):.5f}')

    time_step_axis = np.arange(linear_system_params.n)

    # Figure 1: state estimates and innovations (mirrors MATLAB figure 2)
    fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle('Exercise 5 - EKF Parameter Estimation: State Estimates', fontsize=13)

    axs[0, 0].plot(time_step_axis, true_state_sequence,             label='true x')
    axs[0, 0].plot(time_step_axis, state_prior_estimate_sequence,   '--', label='prior estimate XHM[x]')
    axs[0, 0].plot(time_step_axis, state_posterior_estimate_sequence, ':', label='posterior estimate XHP[x]')
    axs[0, 0].set_title('State: true vs prior and posterior estimates')
    axs[0, 0].set_xlabel('Time step k')
    axs[0, 0].set_ylabel('State value x')
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(time_step_axis, true_state_sequence - state_prior_estimate_sequence)
    axs[0, 1].set_title('Prior state estimation error (x - XHM[x])')
    axs[0, 1].set_xlabel('Time step k')
    axs[0, 1].set_ylabel('State estimation error')
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(time_step_axis, true_state_sequence - state_posterior_estimate_sequence)
    axs[1, 0].set_title('Posterior state estimation error (x - XHP[x])')
    axs[1, 0].set_xlabel('Time step k')
    axs[1, 0].set_ylabel('State estimation error')
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(time_step_axis, innovation_sequence)
    axs[1, 1].set_title('Innovations (measurement - predicted measurement)')
    axs[1, 1].set_xlabel('Time step k')
    axs[1, 1].set_ylabel('Innovation y - ŷ')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(os.path.join(RESULTS_DIR, 'ex5_ekf_states.png'), dpi=150)

    # Figure 2: parameter a convergence (mirrors MATLAB figure 4)
    fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle('Exercise 5 - Parameter a Estimate Convergence', fontsize=13)

    axs[0].plot(time_step_axis, parameter_prior_estimate_sequence,
                color='steelblue', label='estimated a (prior)')
    axs[0].axhline(true_parameter_a, color='k', linestyle='--', lw=2,
                   label=f'true a = {true_parameter_a}')
    axs[0].set_title('Parameter a: estimated vs true value')
    axs[0].set_xlabel('Time step k')
    axs[0].set_ylabel('Parameter a value')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time_step_axis, ekf_param_estimation_result['Pm'][1, :],
                label='Pm[1,1] prior variance on a')
    axs[1].plot(time_step_axis, ekf_param_estimation_result['Pp'][1, :],
                '--', label='Pp[1,1] posterior variance on a')
    axs[1].set_title('Variance of a estimate (shrinks as filter learns)')
    axs[1].set_xlabel('Time step k')
    axs[1].set_ylabel('Variance of parameter estimate')
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, 'ex5_ekf_parameter.png'), dpi=150)

    # Figure 3: Kalman gains and covariances (mirrors MATLAB figure 3)
    fig3, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig3.suptitle('Exercise 5 - Kalman Gains and Covariances', fontsize=13)

    axs[0, 0].plot(time_step_axis, ekf_param_estimation_result['K'][0, :])
    axs[0, 0].set_title('Kalman gain K[0] — correction weight on state x')
    axs[0, 0].set_xlabel('Time step k')
    axs[0, 0].set_ylabel('Kalman gain K[0]')
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(time_step_axis, ekf_param_estimation_result['K'][1, :])
    axs[0, 1].set_title('Kalman gain K[1] — correction weight on parameter a')
    axs[0, 1].set_xlabel('Time step k')
    axs[0, 1].set_ylabel('Kalman gain K[1]')
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(time_step_axis, ekf_param_estimation_result['Pm'][0, :], label='x')
    axs[1, 0].plot(time_step_axis, ekf_param_estimation_result['Pm'][1, :], '--', label='a')
    axs[1, 0].set_title('Prior covariance diag(Pm) for x and a')
    axs[1, 0].set_xlabel('Time step k')
    axs[1, 0].set_ylabel('Prior covariance')
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(time_step_axis, ekf_param_estimation_result['Pp'][0, :], label='x')
    axs[1, 1].plot(time_step_axis, ekf_param_estimation_result['Pp'][1, :], '--', label='a')
    axs[1, 1].set_title('Posterior covariance diag(Pp) for x and a')
    axs[1, 1].set_xlabel('Time step k')
    axs[1, 1].set_ylabel('Posterior covariance')
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, 'ex5_ekf_gains.png'), dpi=150)

    text_lines = [
        'Exercise 5 - EKF Joint State and Parameter Estimation',
        '=' * 60,
        f'True a:              {true_parameter_a:.5f}',
        f'Initial guess a0:    {a0_initial_guess:.5f}',
        f'Final estimate a:    {parameter_prior_estimate_sequence[-1]:.5f}',
        f'Final error |a-a*|:  {abs(parameter_prior_estimate_sequence[-1] - true_parameter_a):.5f}',
        f'Final Pm[1,1]:       {ekf_param_estimation_result["Pm"][1, -1]:.5e}  (variance on a)',
        '',
        'Method: augment state vector with parameter a -> [x; a]',
        '  x_next = a_hat * x_hat + b*u   bilinear in [x;a] -> EKF needed',
        '  a_next = aa * a_hat            constant model (aa=1, sigma_wa=0)',
        '  y      = c*x + v              linear -> H=[c,0] (constant Jacobian)',
        '  Phi    = [[a_hat, x_hat],      Jacobian of state transition',
        '            [0,     aa    ]]',
    ]
    print('\n'.join(text_lines[4:]))
    save_text_results('ex5_ekf_results.txt', text_lines)
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    SEED = 42
    run_exercise_1(seed=SEED)
    run_exercise_2(seed=SEED)
    run_exercise_3(seed=SEED)
    run_exercise_4(seed=SEED)
    run_exercise_5(seed=SEED)
    print(f"\nAll results saved to: {RESULTS_DIR}")
