"""
Exercise 5 - Least Squares estimation of parameter a in the linear system.

From x[i] = a*x[i-1] + (1-a)*u[i-1] + w, rearranging:
    x[i] - u[i-1] = a*(x[i-1] - u[i-1]) + w

Define for j = 0..n-2:
    phi[j] = xhp[j]   - u[j]    (regressor)
    Y[j]   = xhp[j+1] - u[j]    (response)

LS estimator:  a_hat = (phi . Y) / (phi . phi)
Std deviation: sigma_a = sqrt(Q / (phi . phi))  → decays as O(1/sqrt(N))
95% CI:        a_hat ± 1.96 * sigma_a

Numerical example (N=100, true a=0.95):
  Suppose sum(phi^2) = 5.0 and Q = 2.44e-4:
  sigma_a = sqrt(2.44e-4 / 5.0) = sqrt(4.88e-5) = 0.00699
  95% CI width = 2 * 1.96 * 0.00699 ≈ ±0.0137  → CI = [0.936, 0.964]

Run: uv run python Lecture_9/exercise5.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from KF import KalmanFilter, simulate_linear

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_VALUES  = [10, 100, 1000, 10000]
TRUE_A    = 0.95


def estimate_a(xhp, u, Q):
    """
    LS estimate of a using KF posterior states and inputs.
    phi[j] = xhp[j] - u[j],  Y[j] = xhp[j+1] - u[j]  for j = 0..n-2.
    """
    phi   = xhp[:-1] - u[:-1]
    Y     = xhp[1:]  - u[:-1]
    a_hat = float(phi @ Y) / float(phi @ phi)
    sigma = np.sqrt(Q / float(phi @ phi))   # std of estimate
    return a_hat, sigma


def main(seed=42):
    a_hats, sigmas = [], []

    lines = ['Exercise 5 — LS Parameter Estimation of a', '=' * 65,
             f'{"N":>8} {"a_hat":>10} {"a_true":>8} {"error":>10}  {"95% CI"}']

    for n in N_VALUES:
        u, x, y, p = simulate_linear(n=n, seed=seed)
        kf  = KalmanFilter(a=p['a'], b=p['b'], c=p['c'],
                           Q=p['Q'], R=p['R'], x0=p['x0'], P0=p['P0'])
        res = kf.run(y, u)
        a_hat, sigma = estimate_a(res['XHP'], u, p['Q'])
        a_hats.append(a_hat); sigmas.append(sigma)
        ci = 1.96 * sigma
        lines.append(f'{n:>8} {a_hat:>10.5f} {TRUE_A:>8.5f} {a_hat - TRUE_A:>10.5f}'
                     f'  [{a_hat - ci:.5f}, {a_hat + ci:.5f}]')

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Exercise 5 — Estimation of a by Least Squares', fontsize=13)

    # Plot 1: a_hat vs N with 95% confidence intervals
    ci_vals = [1.96 * s for s in sigmas]
    axs[0].semilogx(N_VALUES, [TRUE_A] * len(N_VALUES), 'k--', lw=2, label=f'True a = {TRUE_A}')
    axs[0].errorbar(N_VALUES, a_hats, yerr=ci_vals,
                    fmt='o-', color='steelblue', capsize=5, label='a_hat ± 1.96σ')
    axs[0].set_xlabel('N (log scale)'); axs[0].set_ylabel('a')
    axs[0].set_title('LS estimate of a vs N')
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    # Plot 2: error and std vs N (log-log, should show 1/sqrt(N) decay)
    errors = [abs(ah - TRUE_A) for ah in a_hats]
    axs[1].loglog(N_VALUES, errors,  'o-', color='orange',   label='|a_hat - a_true|')
    axs[1].loglog(N_VALUES, sigmas,  's--', color='steelblue', label='sigma_a (std)')
    # Reference line: proportional to 1/sqrt(N)
    ref = [sigmas[0] * (N_VALUES[0] / n) ** 0.5 for n in N_VALUES]
    axs[1].loglog(N_VALUES, ref, 'k:', lw=1, label='1/√N reference')
    axs[1].set_xlabel('N (log scale)'); axs[1].set_ylabel('Error / std')
    axs[1].set_title('Estimation error decays as 1/√N')
    axs[1].legend(); axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ex5_parameter_estimation.png'), dpi=150)

    lines += [
        '',
        'Method: Least Squares on KF posterior estimates XHP.',
        '  Regressor: phi[j] = xhp[j] - u[j]',
        '  Response:  Y[j]   = xhp[j+1] - u[j]',
        '  Estimator: a_hat  = (phi . Y) / (phi . phi)',
        '  Std:       sigma_a = sqrt(Q / (phi . phi))  ∝  1/sqrt(N)',
        '',
        'The estimator is consistent: error → 0 as N → ∞.',
    ]
    print('\n'.join(lines))
    with open(os.path.join(RESULTS_DIR, 'ex5_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
