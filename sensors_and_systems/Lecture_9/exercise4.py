"""
Exercise 4 - Effect of increasing number of samples N = 10, 100, 1000, 10000.

Key insight: confidence bounds on autocorrelation = ±1.96 / sqrt(N).
  N=10    → bounds ≈ ±0.620  (very wide, hard to detect correlation)
  N=100   → bounds ≈ ±0.196
  N=1000  → bounds ≈ ±0.062
  N=10000 → bounds ≈ ±0.020  (very tight, even tiny residual correlations are detected)

Run: uv run python Lecture_9/exercise4.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from KF import KalmanFilter, simulate_linear

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_VALUES = [10, 100, 1000, 10000]


def autocorr(x, m):
    """Normalized autocorrelation for lags 0..m."""
    n = len(x); xd = x - np.mean(x)
    c = np.correlate(xd, xd, mode='full')
    return c[n - 1:n + m] / c[n - 1]


def main(seed=42):
    fig, axs = plt.subplots(len(N_VALUES), 2, figsize=(12, 3 * len(N_VALUES)))
    fig.suptitle('Exercise 4 — Effect of N on KF Whiteness Test', fontsize=13)

    lines = ['Exercise 4 — Effect of Increasing N', '=' * 65,
             f'{"N":>8} {"RMSE_innov":>12} {"RMSE_xtp":>12} {"p_white":>10} {"CI_width":>10}']

    for row, n in enumerate(N_VALUES):
        m = min(24, max(3, n // 4))   # adapt lag count to sample size

        u, x, y, p = simulate_linear(n=n, seed=seed)
        kf = KalmanFilter(a=p['a'], b=p['b'], c=p['c'],
                          Q=p['Q'], R=p['R'], x0=p['x0'], P0=p['P0'])
        res   = kf.run(y, u)
        innov = y - res['YHM']
        xtp   = x - res['XHP']

        clim     = norm_dist.ppf(0.975) / np.sqrt(n)
        rho      = autocorr(innov, m)
        pw       = 1 - chi2_dist.cdf(float(rho[1:] @ rho[1:]) * n, df=m)
        rmse_i   = np.sqrt(np.mean(innov ** 2))
        rmse_xtp = np.sqrt(np.mean(xtp ** 2))

        # Autocorrelation plot
        axs[row, 0].plot(np.arange(m + 1), rho, '-o', markersize=3, color='steelblue')
        axs[row, 0].axhline( clim, linestyle='--', color='red', label=f'CI=±{clim:.3f}')
        axs[row, 0].axhline(-clim, linestyle='--', color='red')
        axs[row, 0].set_title(f'N={n}  p={pw:.4f}  {"WHITE ✓" if pw > 0.05 else "NOT white ✗"}')
        axs[row, 0].legend(fontsize=7); axs[row, 0].grid(True, alpha=0.3)

        # Innovation histogram
        bins = min(30, max(5, n // 10))
        axs[row, 1].hist(innov, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
        axs[row, 1].set_title(f'N={n}  Innovation histogram  RMSE={rmse_i:.5f}')
        axs[row, 1].grid(True, alpha=0.3)

        lines.append(f'{n:>8} {rmse_i:>12.6f} {rmse_xtp:>12.6f} {pw:>10.4f} {2*clim:>10.4f}')

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex4_n_effect.png'), dpi=150)

    lines += [
        '',
        'Observations:',
        '  1. CI width = 2*1.96/sqrt(N) shrinks with N → whiteness test becomes stricter.',
        '  2. RMSE decreases with N as the filter converges to steady-state gain faster.',
        '  3. For large N (10000), even tiny filter imperfections are detected as non-white.',
        '  4. Innovation histogram becomes more Gaussian-looking as N grows (CLT).',
    ]
    print('\n'.join(lines))
    with open(os.path.join(RESULTS_DIR, 'ex4_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
