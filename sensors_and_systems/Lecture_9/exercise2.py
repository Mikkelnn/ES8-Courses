"""
Exercise 2 - Normal test (normplot) on KF and EKF innovations.
MATLAB's normplot is a Q-Q plot of data against a standard normal distribution.
The Shapiro-Wilk test formally tests H0: data is normally distributed.

Run: uv run python Lecture_9/exercise2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot, shapiro
from KF import KalmanFilter, simulate_linear
from EKF import ExtendedKalmanFilter, simulate_nonlinear

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def normplot(ax, x, title):
    """Q-Q plot against normal distribution (MATLAB normplot equivalent). Returns Shapiro-Wilk (W, p)."""
    (osm, osr), (slope, intercept, _) = probplot(x, dist='norm', plot=None)
    ax.plot(osm, osr, 'o', markersize=3, color='steelblue', label='data')
    ax.plot(osm, slope * osm + intercept, 'r-', lw=1.5, label='normal fit')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    W, p = shapiro(x)
    ax.set_title(f'{title}\nShapiro-Wilk: W={W:.4f}  p={p:.4f}  → {"Normal ✓" if p > 0.05 else "NOT Normal ✗"}')
    return W, p


def main(n=100, seed=42):
    # LSim + KF innovations
    u_l, x_l, y_l, pl = simulate_linear(n=n, seed=seed)
    kf = KalmanFilter(a=pl['a'], b=pl['b'], c=pl['c'],
                      Q=pl['Q'], R=pl['R'], x0=pl['x0'], P0=pl['P0'])
    innov_kf = y_l - kf.run(y_l, u_l)['YHM']

    # NLSim + EKF innovations
    u_n, x_n, y_n, pn = simulate_nonlinear(n=n, seed=seed)
    ekf = ExtendedKalmanFilter(a=pn['a'], b=pn['b'], c=pn['c'],
                               Q=pn['Q'], R=pn['R'],
                               phi_f=pn['phi_f'], phi_h=pn['phi_h'],
                               x0=pn['x0'], P0=pn['P0'])
    innov_ekf = y_n - ekf.run(y_n, u_n)['YHM']

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Exercise 2 — Normal Plot (Q-Q) of Innovations', fontsize=13)
    W_kf,  p_kf  = normplot(axs[0], innov_kf,  'KF innovations (LSim)')
    W_ekf, p_ekf = normplot(axs[1], innov_ekf, 'EKF innovations (NLSim)')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex2_normplot.png'), dpi=150)

    lines = [
        'Exercise 2 — Normality Test Results',
        '=' * 55,
        f'{"Filter":<22} {"Shapiro-Wilk W":>16} {"p-value":>10} {"Normal?":>10}',
        f'{"KF (LSim)":<22} {W_kf:>16.4f} {p_kf:>10.4f} {"YES" if p_kf > 0.05 else "NO":>10}',
        f'{"EKF (NLSim)":<22} {W_ekf:>16.4f} {p_ekf:>10.4f} {"YES" if p_ekf > 0.05 else "NO":>10}',
        '',
        'Shapiro-Wilk H0: data is normally distributed.',
        'p > 0.05 → innovations appear Gaussian (expected for KF with Gaussian noise).',
        'EKF innovations may deviate from Gaussian due to linearization errors.',
    ]
    print('\n'.join(lines))
    with open(os.path.join(RESULTS_DIR, 'ex2_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
