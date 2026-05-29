"""
Exercise 1 - Whiteness test on KF and EKF innovations.
Tests three combinations: LSim+KF (correct), NLSim+EKF (correct), NLSim+KF (mismatch).

Whiteness test: innovations of a well-tuned filter should be uncorrelated (white).
Chi-squared stat = N * sum(rho[1..m]^2), df=m.  p > 0.05 → white.

Run: uv run python Lecture_9/exercise1.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from KF import KalmanFilter, simulate_linear
from EKF import ExtendedKalmanFilter, simulate_nonlinear

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def autocorr(x, m=24):
    """Normalized autocorrelation for lags 0..m."""
    n = len(x)
    xd = x - np.mean(x)
    c = np.correlate(xd, xd, mode='full')
    return c[n - 1:n + m] / c[n - 1]


def plot_whiteness(ax, innov, title, m=24):
    """Plot autocorrelation with confidence bounds and chi-squared p-value."""
    n = len(innov)
    rho = autocorr(innov, m)
    clim = norm_dist.ppf(0.975) / np.sqrt(n)
    chi2_stat = float(rho[1:] @ rho[1:]) * n
    p = 1 - chi2_dist.cdf(chi2_stat, df=m)

    ax.plot(np.arange(m + 1), rho, '-o', markersize=3, color='steelblue')
    ax.axhline( clim, linestyle='--', color='red', label=f'±95% CI = ±{clim:.3f}')
    ax.axhline(-clim, linestyle='--', color='red')
    ax.set_xlabel('Lag')
    ax.set_title(f'{title}\np = {p:.4f}  →  {"WHITE ✓" if p > 0.05 else "NOT white ✗"}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return p, chi2_stat


def main(n=100, seed=42):
    # LSim + KF (correct model)
    u_l, x_l, y_l, pl = simulate_linear(n=n, seed=seed)
    kf = KalmanFilter(a=pl['a'], b=pl['b'], c=pl['c'],
                      Q=pl['Q'], R=pl['R'], x0=pl['x0'], P0=pl['P0'])
    innov_kf = y_l - kf.run(y_l, u_l)['YHM']

    # NLSim + EKF (correct model)
    u_n, x_n, y_n, pn = simulate_nonlinear(n=n, seed=seed)
    ekf = ExtendedKalmanFilter(a=pn['a'], b=pn['b'], c=pn['c'],
                               Q=pn['Q'], R=pn['R'],
                               phi_f=pn['phi_f'], phi_h=pn['phi_h'],
                               x0=pn['x0'], P0=pn['P0'])
    innov_ekf = y_n - ekf.run(y_n, u_n)['YHM']

    # NLSim + KF (mismatch: linear KF applied to nonlinear data)
    kf_mis = KalmanFilter(a=pn['a'], b=pn['b'], c=1.0,
                          Q=pn['Q'], R=pn['R'], x0=pn['x0'], P0=pn['P0'])
    innov_mis = y_n - kf_mis.run(y_n, u_n)['YHM']

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Exercise 1 — Whiteness Test', fontsize=13)
    p1, _ = plot_whiteness(axs[0], innov_kf,  'LSim + KF (correct)')
    p2, _ = plot_whiteness(axs[1], innov_ekf, 'NLSim + EKF (correct)')
    p3, _ = plot_whiteness(axs[2], innov_mis, 'NLSim + KF (mismatch)')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex1_whiteness.png'), dpi=150)

    lines = [
        'Exercise 1 — Whiteness Test Results',
        '=' * 50,
        f'{"Combination":<30} {"p-value":>10}  {"White?":>8}',
        f'{"LSim + KF":<30} {p1:>10.4f}  {"YES" if p1 > 0.05 else "NO":>8}',
        f'{"NLSim + EKF":<30} {p2:>10.4f}  {"YES" if p2 > 0.05 else "NO":>8}',
        f'{"NLSim + KF (mismatch)":<30} {p3:>10.4f}  {"YES" if p3 > 0.05 else "NO":>8}',
        '',
        'H0: innovations are white (uncorrelated).',
        'p > 0.05 → fail to reject H0 → white innovations (good filter).',
        'p < 0.05 → reject H0 → correlated innovations (model mismatch).',
    ]
    print('\n'.join(lines))
    with open(os.path.join(RESULTS_DIR, 'ex1_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
