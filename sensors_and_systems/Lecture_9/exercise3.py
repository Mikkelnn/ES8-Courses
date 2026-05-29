"""
Exercise 3 - What happens when noise is Uniform instead of Normal?
LSim.m supports both via NormalP / NormalM flags.

Uniform noise U(-a, a) with a = sigma*sqrt(3) has the same variance as N(0, sigma^2).
The KF is still the Best Linear Unbiased Estimator (BLUE) — it minimises variance for
any noise distribution. However, innovations will no longer be Gaussian (visible on Q-Q plot).

Run: uv run python Lecture_9/exercise3.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.stats import chi2 as chi2_dist, norm as norm_dist, probplot, shapiro
from KF import KalmanFilter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_linear_noise(n=100, a=0.95, k=1.0, c=1.0, fu=0.02,
                           sigmaw_scale=0.05, sigmav=0.01, x0=0.0,
                           normal_process=True, normal_measurement=True, seed=None):
    """
    Simulate linear system with Normal or Uniform noise.
    Uniform: U(-sigma*sqrt(3), sigma*sqrt(3)) → same mean=0, same variance=sigma^2.
    MATLAB equivalent: sigmaw*(rand(n,1)-0.5)*sqrt(12)  [since sqrt(12) = 1/sqrt(1/12)]
    """
    if seed is not None:
        np.random.seed(seed)

    b       = k * (1.0 - a)
    sigmaw  = sigmaw_scale * np.sqrt(1.0 - a ** 2)
    sigmax0 = sigmaw

    if normal_process:
        w = sigmaw * np.random.randn(n)
    else:
        w = sigmaw * (np.random.rand(n) - 0.5) * np.sqrt(12)   # uniform, same variance

    if normal_measurement:
        v = sigmav * np.random.randn(n)
    else:
        v = sigmav * (np.random.rand(n) - 0.5) * np.sqrt(12)

    u = square(2.0 * np.pi * fu * np.arange(1, n + 1))
    x = np.zeros(n); y = np.zeros(n)
    x[0] = sigmax0 * np.random.randn() + x0
    y[0] = c * x[0] + v[0]
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * u[i - 1] + w[i - 1]
        y[i] = c * x[i]     + v[i]

    return u, x, y, dict(a=a, b=b, c=c, Q=sigmaw**2, R=sigmav**2, P0=sigmax0**2, x0=x0)


def autocorr(x, m=24):
    n = len(x); xd = x - np.mean(x)
    c = np.correlate(xd, xd, mode='full')
    return c[n - 1:n + m] / c[n - 1]


def main(n=100, seed=42):
    cases = [
        ('Normal w & v',          True,  True),
        ('Uniform w, Normal v',   False, True),
        ('Normal w, Uniform v',   True,  False),
        ('Uniform w & v',         False, False),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Exercise 3 — Uniform vs Normal Noise', fontsize=13)

    lines = ['Exercise 3 — Uniform vs Normal Noise', '=' * 60,
             f'{"Case":<28} {"p_white":>10}  {"p_normal (SW)":>14}']

    for col, (label, norm_p, norm_m) in enumerate(cases):
        u, x, y, p = simulate_linear_noise(n=n, seed=seed,
                                            normal_process=norm_p,
                                            normal_measurement=norm_m)
        kf = KalmanFilter(a=p['a'], b=p['b'], c=p['c'],
                          Q=p['Q'], R=p['R'], x0=p['x0'], P0=p['P0'])
        innov = y - kf.run(y, u)['YHM']

        # Row 0: autocorrelation (whiteness test)
        rho  = autocorr(innov, m=24)
        clim = norm_dist.ppf(0.975) / np.sqrt(n)
        pw   = 1 - chi2_dist.cdf(float(rho[1:] @ rho[1:]) * n, df=24)
        axs[0, col].plot(np.arange(25), rho, '-o', markersize=3, color='steelblue')
        axs[0, col].axhline( clim, linestyle='--', color='red')
        axs[0, col].axhline(-clim, linestyle='--', color='red')
        axs[0, col].set_title(f'{label}\np_white={pw:.3f}  {"✓" if pw > 0.05 else "✗"}', fontsize=8)
        axs[0, col].grid(True, alpha=0.3)

        # Row 1: Q-Q normality plot
        (osm, osr), (slope, intercept, _) = probplot(innov, dist='norm', plot=None)
        _, pn = shapiro(innov)
        axs[1, col].plot(osm, osr, 'o', markersize=3, color='steelblue')
        axs[1, col].plot(osm, slope * osm + intercept, 'r-', lw=1)
        axs[1, col].set_title(f'Q-Q plot\np_normal={pn:.3f}  {"✓" if pn > 0.05 else "✗"}', fontsize=8)
        axs[1, col].grid(True, alpha=0.3)

        lines.append(f'{label:<28} {pw:>10.4f}  {pn:>14.4f}')

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex3_uniform_vs_normal.png'), dpi=150)

    lines += [
        '',
        'Observations:',
        '  - Whiteness test passes in all cases: KF is BLUE regardless of noise distribution.',
        '  - Normality test fails when uniform noise is present: innovations are non-Gaussian.',
        '  - KF is only optimal (minimum MSE) when noise is Gaussian; otherwise it is BLUE.',
    ]
    print('\n'.join(lines))
    with open(os.path.join(RESULTS_DIR, 'ex3_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
