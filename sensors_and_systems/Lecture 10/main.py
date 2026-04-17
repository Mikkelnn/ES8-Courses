import matplotlib.pyplot as plt
import numpy as np


def lsim(n=200, a=0.9, b_sys=1.0, c=1.0,
         x0=0.0, sigmax0=0.1, sigmaw=0.1, sigmav=0.1, seed=42):
    """Simulate the linear system x(i)=a*x(i-1)+b*u(i-1)+w(i-1), y(i)=c*x(i)+v(i)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0 + sigmax0 * rng.standard_normal()
    y[0] = c * x[0] + sigmav * rng.standard_normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + b_sys * u[i - 1] + sigmaw * rng.standard_normal()
        y[i] = c * x[i] + sigmav * rng.standard_normal()
    return x, y, u


def ekf_par_est():
    """Extended Kalman filter for parameter estimation.

    System:
        x(i)  = a(i-1)*x(i-1) + b*u(i-1) + w(i-1)
        a(i)  = aa*a(i-1)     + wa(i-1)
        y(i)  = c*x(i)        + v(i)

    The state is augmented with the unknown parameter a so that
    the EKF jointly estimates x and a.
    """
    # --- Simulation parameters (LSim equivalent) ---
    n = 200
    a_true = 0.9
    b_sys = 1.0
    c = 1.0
    x0 = 0.0
    sigmax0 = 0.1
    sigmaw = 0.1
    sigmav = 0.1

    x, y, u = lsim(n, a_true, b_sys, c, x0, sigmax0, sigmaw, sigmav)

    # --- EKF parameters ---
    a0 = 0.5;      sigmaa0 = 0.1
    aa = 1.0;      sigmawa = 0.0
    # aa = 0.9;    sigmawa = 0.5 * np.sqrt(1 - aa**2)  # alternative

    xh0 = np.array([x0, a0])
    P0  = np.diag([sigmax0**2, sigmaa0**2])
    R   = sigmav**2
    Q   = np.diag([sigmaw**2, sigmawa**2])
    H   = np.array([[c, 0.0]])       # measurement Jacobian (linear in x, not a)

    # Storage
    XHM = np.zeros((2, n))   # prior  (time-update) estimates
    XHP = np.zeros((2, n))   # posterior (meas-update) estimates
    YHM = np.zeros(n)
    K_log  = np.zeros((2, n))
    Pm_log = np.zeros((2, n))
    Pp_log = np.zeros((2, n))

    xhm = xh0.copy()
    Pm  = P0.copy()

    for i in range(n):
        # --- collect prior ---
        XHM[:, i] = xhm
        YHM[i]    = (H @ xhm)[0]

        # --- measurement update ---
        S   = (H @ Pm @ H.T).item() + R  # scalar
        K   = (Pm @ H.T).flatten() / S   # shape (2,)
        xhp = xhm + K * (y[i] - YHM[i])
        Pp  = (np.eye(2) - np.outer(K, H)) @ Pm @ (np.eye(2) - np.outer(K, H)).T \
              + np.outer(K, K) * R        # Joseph form for numerical stability

        XHP[:, i]  = xhp
        K_log[:, i]  = K
        Pm_log[:, i] = np.diag(Pm)
        Pp_log[:, i] = np.diag(Pp)

        # --- time update (nonlinear f, linearised Phi) ---
        xhm_new    = np.empty(2)
        xhm_new[0] = xhp[1] * xhp[0] + b_sys * u[i]   # a_est * x_est + b*u
        xhm_new[1] = aa * xhp[1]
        Phi        = np.array([[xhp[1], xhp[0]],        # df/d[x, a]
                               [0.0,    aa      ]])
        Pm  = Phi @ Pp @ Phi.T + Q
        xhm = xhm_new

    # --- plots ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle('Extended Kalman filter for parameter estimation')

    axes[0, 0].plot(x,            label='x true')
    axes[0, 0].plot(XHM[0],       label='XHM (prior)')
    axes[0, 0].plot(XHP[0],       label='XHP (posterior)')
    axes[0, 0].set_title('x, XHM, XHP');  axes[0, 0].legend()

    axes[0, 1].plot(x - XHM[0])
    axes[0, 1].set_title('x - XHM (prior error)')

    axes[0, 2].plot(x - XHP[0])
    axes[0, 2].set_title('x - XHP (posterior error)')

    axes[1, 0].plot(y - YHM)
    axes[1, 0].set_title('y - YHM (output residual)')

    axes[1, 1].plot(XHM[1],                   label='a est (prior)')
    axes[1, 1].axhline(a_true, color='r', ls='--', label=f'a true = {a_true}')
    axes[1, 1].set_title('Parameter a estimate');  axes[1, 1].legend()

    axes[1, 2].plot(Pm_log[1], label='Pm(2,2)')
    axes[1, 2].set_title('Var on a estimate (Pm diag)')

    plt.tight_layout()
    plt.savefig('ekf_par_est.png')
    plt.clf()

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7))
    fig2.suptitle('Kalman gain and state covariances')
    axes2[0, 0].plot(K_log[0]);   axes2[0, 0].set_title('K(1)')
    axes2[0, 1].plot(K_log[1]);   axes2[0, 1].set_title('K(2)')
    axes2[1, 0].plot(Pm_log.T);   axes2[1, 0].set_title('diag(Pm)')
    axes2[1, 1].plot(Pp_log.T);   axes2[1, 1].set_title('diag(Pp)')
    plt.tight_layout()
    plt.savefig('ekf_par_est_gains.png')
    plt.clf()


def main():
    ekf_par_est()


if __name__ == "__main__":
    main()
