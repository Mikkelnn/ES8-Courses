"""
Lecture 9 - Kalman Filter on a Linear System.
Equivalent to LSim.m + KF.m in MATLAB.

Run:  uv run python KF.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class KalmanFilter:
    """
    KF for the scalar linear system:
        x[i] = a*x[i-1] + b*u[i-1] + w,  w ~ N(0, Q)
        y[i] = c*x[i] + v,                v ~ N(0, R)
    """

    def __init__(self, a, b, c, Q, R, x0=0.0, P0=1.0):
        self.a, self.b, self.c = a, b, c
        self.Q, self.R = Q, R
        self.x0, self.P0 = x0, P0
        self.reset()

    def reset(self):
        self.xhm = self.x0
        self.Pm  = self.P0
        self.yhm = self.c * self.xhm

    def step(self, y, u):
        """One measurement update then one time update. Returns logged values."""
        a, b, c, Q, R = self.a, self.b, self.c, self.Q, self.R
        xhm, yhm, Pm = self.xhm, self.yhm, self.Pm

        # Measurement update
        K   = Pm * c / (c * Pm * c + R)
        xhp = xhm + K * (y - yhm)
        Pp  = (1 - K * c) * Pm * (1 - K * c) + K * R * K  # Joseph form

        # Time update
        self.xhm = a * xhp + b * u
        self.Pm  = a * Pp * a + Q
        self.yhm = c * self.xhm

        return dict(xhm=xhm, yhm=yhm, xhp=xhp, K=K, Pm=Pm, Pp=Pp)

    def run(self, y, u):
        """Run filter over the full sequence. Returns dict of arrays shaped (n,)."""
        self.reset()
        n = len(y)
        XHM = np.zeros(n); YHM = np.zeros(n); XHP = np.zeros(n)
        K_log = np.zeros(n); Pm_log = np.zeros(n); Pp_log = np.zeros(n)

        for i in range(n):
            r = self.step(y[i], u[i])
            XHM[i]    = r['xhm'];  YHM[i]   = r['yhm'];  XHP[i]   = r['xhp']
            K_log[i]  = r['K'];    Pm_log[i] = r['Pm'];   Pp_log[i] = r['Pp']

        return dict(XHM=XHM, YHM=YHM, XHP=XHP, K=K_log, Pm=Pm_log, Pp=Pp_log)


def simulate_linear(n=100, a=0.95, k=1.0, c=1.0, fu=0.02,
                    sigmaw_scale=0.05, sigmav=0.01, x0=0.0, seed=None):
    """
    Simulate the linear system for n steps.
    sigmaw = sigmaw_scale * sqrt(1 - a^2) so steady-state Var(x) = sigmaw_scale^2.
    """
    if seed is not None:
        np.random.seed(seed)

    b       = k * (1.0 - a)
    sigmaw  = sigmaw_scale * np.sqrt(1.0 - a ** 2)
    sigmax0 = sigmaw

    w = sigmaw * np.random.randn(n)
    v = sigmav * np.random.randn(n)
    u = square(2.0 * np.pi * fu * np.arange(1, n + 1))

    x = np.zeros(n); y = np.zeros(n)
    x[0] = sigmax0 * np.random.randn() + x0
    y[0] = c * x[0] + v[0]
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * u[i - 1] + w[i - 1]
        y[i] = c * x[i]     + v[i]

    params = dict(a=a, b=b, c=c, Q=sigmaw**2, R=sigmav**2, P0=sigmax0**2, x0=x0)
    return u, x, y, params


def main(n=100, seed=42):
    # Simulate
    u, x, y, p = simulate_linear(n=n, seed=seed)

    # Run KF
    kf  = KalmanFilter(a=p['a'], b=p['b'], c=p['c'],
                       Q=p['Q'], R=p['R'], x0=p['x0'], P0=p['P0'])
    res = kf.run(y, u)
    XHM, YHM, XHP = res['XHM'], res['YHM'], res['XHP']
    K_log, Pm_log, Pp_log = res['K'], res['Pm'], res['Pp']

    ytm = y - YHM   # innovations
    xtm = x - XHM   # prior state error
    xtp = x - XHP   # posterior state error

    # Print residual statistics
    print(f"{'':>10} {'ytm':>12} {'xtm':>12} {'xtp':>12}")
    for label, vals in [('Mean', [np.mean(ytm), np.mean(xtm), np.mean(xtp)]),
                        ('Std',  [np.std(ytm),  np.std(xtm),  np.std(xtp)]),
                        ('RMSE', [np.sqrt(np.mean(ytm**2)), np.sqrt(np.mean(xtm**2)), np.sqrt(np.mean(xtp**2))])]:
        print(f"{label:>10} {vals[0]:>12.6f} {vals[1]:>12.6f} {vals[2]:>12.6f}")

    # Save statistics to text file
    lines = [f"{'':>10} {'ytm':>12} {'xtm':>12} {'xtp':>12}"]
    for label, vals in [('Mean', [np.mean(ytm), np.mean(xtm), np.mean(xtp)]),
                        ('Std',  [np.std(ytm),  np.std(xtm),  np.std(xtp)]),
                        ('RMSE', [np.sqrt(np.mean(ytm**2)), np.sqrt(np.mean(xtm**2)), np.sqrt(np.mean(xtp**2))])]:
        lines.append(f"{label:>10} {vals[0]:>12.6f} {vals[1]:>12.6f} {vals[2]:>12.6f}")
    with open(os.path.join(RESULTS_DIR, 'kf_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    # Plot 1: Simulation data (Figure 1 from MATLAB)
    fig1, axs = plt.subplots(3, 1, figsize=(10, 7))
    fig1.suptitle('Linear System Simulation', fontsize=13)
    axs[0].plot(u);  axs[0].set_title('u');  axs[0].grid(True, alpha=0.3)
    axs[1].plot(x);  axs[1].set_title('x');  axs[1].grid(True, alpha=0.3)
    axs[2].plot(y);  axs[2].set_title('y');  axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(RESULTS_DIR, 'kf_simulation.png'), dpi=150)

    # Plot 2: KF estimates and residuals (Figure 2 from MATLAB)
    fig2, axs = plt.subplots(3, 2, figsize=(12, 9))
    fig2.suptitle('Kalman Filter Results', fontsize=13)
    axs[0, 0].plot(x, label='x');  axs[0, 0].plot(XHM, '--', label='XHM');  axs[0, 0].plot(XHP, ':', label='XHP')
    axs[0, 0].set_title('x XHM XHP');  axs[0, 0].legend(fontsize=8);  axs[0, 0].grid(True, alpha=0.3)
    axs[0, 1].plot(xtm);  axs[0, 1].set_title('x-XHM');  axs[0, 1].grid(True, alpha=0.3)
    axs[1, 0].plot(xtp);  axs[1, 0].set_title('x-XHP');  axs[1, 0].grid(True, alpha=0.3)
    axs[1, 1].plot(ytm);  axs[1, 1].set_title('y-YHM');  axs[1, 1].grid(True, alpha=0.3)
    axs[2, 0].axis('off');  axs[2, 1].axis('off')
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, 'kf_estimates.png'), dpi=150)

    # Plot 3: Kalman gain and covariances (Figure 3 from MATLAB)
    fig3, axs = plt.subplots(3, 1, figsize=(10, 7))
    fig3.suptitle('KF Internal Variables', fontsize=13)
    axs[0].plot(K_log);   axs[0].set_title('K');   axs[0].grid(True, alpha=0.3)
    axs[1].plot(Pm_log);  axs[1].set_title('Pm');  axs[1].grid(True, alpha=0.3)
    axs[2].plot(Pp_log);  axs[2].set_title('Pp');  axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, 'kf_gains.png'), dpi=150)

    print(f"\nResults saved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
