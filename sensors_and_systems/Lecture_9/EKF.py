"""
Lecture 9 - Extended Kalman Filter on a Nonlinear System.
Equivalent to NLSim.m + EKF.m in MATLAB.

Run:  uv run python EKF.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class ExtendedKalmanFilter:
    """
    EKF for the scalar nonlinear system:
        x[i] = a*sin(x[i-1] + phi_f) + b*u[i-1] + w,  w ~ N(0, Q)
        y[i] = sin(c*x[i] + phi_h) + v,                v ~ N(0, R)

    Jacobians used for linearization:
        H   = d/dx [sin(c*x + phi_h)] = c * cos(c*x + phi_h)   (measurement)
        Phi = d/dx [a*sin(x + phi_f)] = a * cos(x  + phi_f)    (state transition)
    """

    def __init__(self, a, b, c, Q, R, phi_f=0.0, phi_h=0.0, x0=0.0, P0=1.0):
        self.a, self.b, self.c = a, b, c
        self.Q, self.R = Q, R
        self.phi_f, self.phi_h = phi_f, phi_h
        self.x0, self.P0 = x0, P0
        self.reset()

    def reset(self):
        self.xhm = self.x0
        self.Pm  = self.P0
        self.yhm = np.sin(self.c * self.xhm + self.phi_h)

    def step(self, y, u):
        """One EKF cycle: linearize around current estimate, then standard KF update."""
        a, b, c, Q, R = self.a, self.b, self.c, self.Q, self.R
        phi_f, phi_h = self.phi_f, self.phi_h
        xhm, yhm, Pm = self.xhm, self.yhm, self.Pm

        # Measurement update — linearize h(x)=sin(c*x+phi_h) at xhm
        H   = c * np.cos(c * xhm + phi_h)
        K   = Pm * H / (H * Pm * H + R)
        xhp = xhm + K * (y - yhm)
        Pp  = (1 - K * H) * Pm * (1 - K * H) + K * R * K  # Joseph form

        # Time update — linearize f(x)=a*sin(x+phi_f) at xhp
        self.xhm = a * np.sin(xhp + phi_f) + b * u
        Phi      = a * np.cos(xhp + phi_f)
        self.Pm  = Phi * Pp * Phi + Q
        self.yhm = np.sin(c * self.xhm + phi_h)

        return dict(xhm=xhm, yhm=yhm, xhp=xhp, K=K, Pm=Pm, Pp=Pp)

    def run(self, y, u):
        """Run EKF over the full sequence. Returns dict of arrays shaped (n,)."""
        self.reset()
        n = len(y)
        XHM = np.zeros(n); YHM = np.zeros(n); XHP = np.zeros(n)
        K_log = np.zeros(n); Pm_log = np.zeros(n); Pp_log = np.zeros(n)

        for i in range(n):
            r = self.step(y[i], u[i])
            XHM[i]    = r['xhm'];  YHM[i]   = r['yhm'];  XHP[i]   = r['xhp']
            K_log[i]  = r['K'];    Pm_log[i] = r['Pm'];   Pp_log[i] = r['Pp']

        return dict(XHM=XHM, YHM=YHM, XHP=XHP, K=K_log, Pm=Pm_log, Pp=Pp_log)


def simulate_nonlinear(n=100, a=0.95, k=1.0, c=10.0, fu=0.02,
                       phi_f=0.0, phi_h=0.0,
                       sigmaw_scale=0.05, sigmav=0.1, x0=0.0, seed=None):
    """
    Simulate the nonlinear system for n steps.
    c=10 matches the active branch 'if 1; c=10; end' in the original NLSim.m.
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
    y[0] = np.sin(c * x[0] + phi_h) + v[0]
    for i in range(1, n):
        x[i] = a * np.sin(x[i - 1] + phi_f) + b * u[i - 1] + w[i - 1]
        y[i] = np.sin(c * x[i]     + phi_h) + v[i]

    params = dict(a=a, b=b, c=c, Q=sigmaw**2, R=sigmav**2,
                  P0=sigmax0**2, x0=x0, phi_f=phi_f, phi_h=phi_h)
    return u, x, y, params


def main(n=100, seed=42):
    # Simulate
    u, x, y, p = simulate_nonlinear(n=n, seed=seed)

    # Print summary statistics (matches NLSim.m disp output)
    for label, arr in [('u', u), ('x', x), ('y', y)]:
        print(f"  {label}:  mean={np.mean(arr):.4f}  std={np.std(arr):.4f}")

    # Run EKF
    ekf = ExtendedKalmanFilter(a=p['a'], b=p['b'], c=p['c'],
                               Q=p['Q'], R=p['R'],
                               phi_f=p['phi_f'], phi_h=p['phi_h'],
                               x0=p['x0'], P0=p['P0'])
    res = ekf.run(y, u)
    XHM, YHM, XHP = res['XHM'], res['YHM'], res['XHP']
    K_log, Pm_log, Pp_log = res['K'], res['Pm'], res['Pp']

    ytm = y - YHM
    xtm = x - XHM
    xtp = x - XHP

    # Print residual statistics
    print(f"\n{'':>10} {'ytm':>12} {'xtm':>12} {'xtp':>12}")
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
    with open(os.path.join(RESULTS_DIR, 'ekf_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    # Plot 1: Simulation data (Figure 1 from MATLAB)
    fig1, axs = plt.subplots(3, 1, figsize=(10, 7))
    fig1.suptitle('Nonlinear System Simulation', fontsize=13)
    axs[0].plot(u);  axs[0].set_title('u');  axs[0].grid(True, alpha=0.3)
    axs[1].plot(x);  axs[1].set_title('x');  axs[1].grid(True, alpha=0.3)
    axs[2].plot(y);  axs[2].set_title('y');  axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(RESULTS_DIR, 'ekf_simulation.png'), dpi=150)

    # Plot 2: EKF estimates and residuals (Figure 2 from MATLAB)
    fig2, axs = plt.subplots(3, 2, figsize=(12, 9))
    fig2.suptitle('Extended Kalman Filter Results', fontsize=13)
    axs[0, 0].plot(x, label='x');  axs[0, 0].plot(XHM, '--', label='XHM');  axs[0, 0].plot(XHP, ':', label='XHP')
    axs[0, 0].set_title('x XHM XHP');  axs[0, 0].legend(fontsize=8);  axs[0, 0].grid(True, alpha=0.3)
    axs[0, 1].plot(xtm);  axs[0, 1].set_title('x-XHM');  axs[0, 1].grid(True, alpha=0.3)
    axs[1, 0].plot(xtp);  axs[1, 0].set_title('x-XHP');  axs[1, 0].grid(True, alpha=0.3)
    axs[1, 1].plot(ytm);  axs[1, 1].set_title('y-YHM');  axs[1, 1].grid(True, alpha=0.3)
    axs[2, 0].axis('off');  axs[2, 1].axis('off')
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, 'ekf_estimates.png'), dpi=150)

    # Plot 3: Kalman gain and covariances (Figure 3 from MATLAB)
    fig3, axs = plt.subplots(3, 1, figsize=(10, 7))
    fig3.suptitle('EKF Internal Variables', fontsize=13)
    axs[0].plot(K_log);   axs[0].set_title('K');   axs[0].grid(True, alpha=0.3)
    axs[1].plot(Pm_log);  axs[1].set_title('Pm');  axs[1].grid(True, alpha=0.3)
    axs[2].plot(Pp_log);  axs[2].set_title('Pp');  axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, 'ekf_gains.png'), dpi=150)

    print(f"\nResults saved to: {RESULTS_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
