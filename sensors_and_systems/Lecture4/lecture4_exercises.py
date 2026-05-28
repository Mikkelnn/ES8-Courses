#!/usr/bin/env python3
"""
UKF solution for exercises 1, 2, and 3, updated with the uploaded MATLAB files.

This mirrors NLSim.m as closely as possible, then runs a scalar UKF instead
of the EKF from EKF.m.

Default:
    CASE = "nlsim"

This matches the active NLSim.m file, including:
    if 1; c = 10; end

Available cases:
    "nlsim"   -> active NLSim.m settings, c = 10
    "initial" -> exercise initial settings, c = 1
    "phi_h"   -> phi_h = pi/16, c = 1
    "phi_f"   -> phi_f = pi/16, c = 1
    "c10"     -> c = 10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

CASE = "nlsim"
SEED = None          # Use None to mimic unseeded MATLAB randn; use e.g. 1 for repeatable results
SHOW_PLOTS = True


# ============================================================
# Parameters and simulation
# ============================================================

@dataclass
class Params:
    # NLSim.m values
    n: int = 100
    a: float = 0.95
    k_gain: float = 1.0
    c: float = 10.0
    phi_f: float = 0.0
    phi_h: float = 0.0
    fu: float = 0.02
    x0: float = 0.0
    seed: Optional[int] = None

    # From NLSim.m:
    # sigmaw = 0.05*sqrt(1-a^2)
    # sigmav = 0.1
    # sigmax0 = sigmaw
    sigma_w: Optional[float] = None
    sigma_v: float = 0.1
    sigma_x0: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sigma_w is None:
            self.sigma_w = 0.05 * np.sqrt(1.0 - self.a**2)
        if self.sigma_x0 is None:
            self.sigma_x0 = self.sigma_w

    @property
    def b(self) -> float:
        return self.k_gain * (1.0 - self.a)

    @property
    def Q(self) -> float:
        return float(self.sigma_w**2)

    @property
    def R(self) -> float:
        return float(self.sigma_v**2)

    @property
    def P0(self) -> float:
        return float(self.sigma_x0**2)


def make_params(case: str, seed: Optional[int]) -> Params:
    p = Params(seed=seed)

    if case == "nlsim":
        # Active NLSim.m settings
        p.c = 10.0
        p.phi_f = 0.0
        p.phi_h = 0.0
    elif case == "initial":
        p.c = 1.0
        p.phi_f = 0.0
        p.phi_h = 0.0
    elif case == "phi_h":
        p.c = 1.0
        p.phi_f = 0.0
        p.phi_h = np.pi / 16.0
    elif case == "phi_f":
        p.c = 1.0
        p.phi_f = np.pi / 16.0
        p.phi_h = 0.0
    elif case == "c10":
        p.c = 10.0
        p.phi_f = 0.0
        p.phi_h = 0.0
    else:
        raise ValueError(f"Unknown CASE: {case}")

    p.__post_init__()
    return p


def matlab_square(theta: np.ndarray) -> np.ndarray:
    """
    Replacement for MATLAB square(theta), without scipy.
    """
    return np.where(np.sin(theta) >= 0.0, 1.0, -1.0)


def simulate_nonlinear_system(p: Params):
    """
    Simulate the nonlinear system as in NLSim.m.
    """
    rng = np.random.default_rng(p.seed)

    w = p.sigma_w * rng.standard_normal(p.n)
    v = p.sigma_v * rng.standard_normal(p.n)

    k = np.arange(1, p.n + 1)
    u = matlab_square(k * p.fu * 2.0 * np.pi)

    x = np.zeros(p.n)
    y = np.zeros(p.n)

    x[0] = p.sigma_x0 * rng.standard_normal() + p.x0
    y[0] = np.sin(p.c * x[0] + p.phi_h) + v[0]

    for i in range(1, p.n):
        x[i] = p.a * np.sin(x[i - 1] + p.phi_f) + p.b * u[i - 1] + w[i - 1]
        y[i] = np.sin(p.c * x[i] + p.phi_h) + v[i]

    return x, y, u


# ============================================================
# UKF
# ============================================================

def sigma_points_scalar(mean: float, variance: float, scale: float) -> np.ndarray:
    variance = max(float(variance), 1e-14)
    spread = np.sqrt(scale * variance)
    return np.array([mean, mean + spread, mean - spread])


def ukf_scalar(y: np.ndarray, u: np.ndarray, p: Params):
    """
    Scalar UKF for:

        x_k = a*sin(x_{k-1} + phi_f) + b*u_{k-1} + w_{k-1}
        y_k = sin(c*x_k + phi_h) + v_k
    """
    N = len(y)

    n_state = 1
    alpha = 1.0
    kappa = 2.0
    beta = 0.0

    lam = alpha**2 * (n_state + kappa) - n_state
    scale = n_state + lam

    Wm = np.zeros(2 * n_state + 1)
    Wc = np.zeros(2 * n_state + 1)

    Wm[0] = lam / scale
    Wc[0] = lam / scale + 1.0 - alpha**2 + beta
    Wm[1:] = 1.0 / (2.0 * scale)
    Wc[1:] = 1.0 / (2.0 * scale)

    XHM = np.zeros(N)      # prior state estimate
    YHM = np.zeros(N)      # predicted measurement
    XHP = np.zeros(N)      # posterior state estimate
    PM = np.zeros(N)       # prior variance
    PP = np.zeros(N)       # posterior variance
    K = np.zeros(N)        # Kalman gain
    S_arr = np.zeros(N)    # innovation variance

    xhm = p.x0
    Pm = p.P0

    for i in range(N):
        XHM[i] = xhm
        PM[i] = Pm

        # Measurement update
        Xsig = sigma_points_scalar(xhm, Pm, scale)
        Ysig = np.sin(p.c * Xsig + p.phi_h)

        yhm = float(np.sum(Wm * Ysig))
        YHM[i] = yhm

        S = p.R
        Cxy = 0.0

        for j in range(2 * n_state + 1):
            dy = Ysig[j] - yhm
            dx = Xsig[j] - xhm
            S += Wc[j] * dy * dy
            Cxy += Wc[j] * dx * dy

        K[i] = Cxy / S

        xhp = xhm + K[i] * (y[i] - yhm)
        Pp = Pm - K[i] * S * K[i]
        Pp = max(float(Pp), 1e-14)

        XHP[i] = xhp
        PP[i] = Pp
        S_arr[i] = S

        # Time update
        Xsig = sigma_points_scalar(xhp, Pp, scale)
        Xnext = p.a * np.sin(Xsig + p.phi_f) + p.b * u[i]

        xhm = float(np.sum(Wm * Xnext))

        Pm_next = p.Q
        for j in range(2 * n_state + 1):
            dx = Xnext[j] - xhm
            Pm_next += Wc[j] * dx * dx

        Pm = max(float(Pm_next), 1e-14)

    return {
        "XHM": XHM,
        "YHM": YHM,
        "XHP": XHP,
        "PM": PM,
        "PP": PP,
        "K": K,
        "S": S_arr,
    }


# ============================================================
# Diagnostics and plotting
# ============================================================

def print_parameters(p: Params) -> None:
    print("Parameters")
    print("----------")
    print(f"CASE      = {CASE}")
    print(f"n         = {p.n}")
    print(f"a         = {p.a}")
    print(f"k_gain    = {p.k_gain}")
    print(f"b         = {p.b}")
    print(f"c         = {p.c}")
    print(f"phi_f     = {p.phi_f}")
    print(f"phi_h     = {p.phi_h}")
    print(f"fu        = {p.fu}")
    print(f"sigma_w   = {p.sigma_w}")
    print(f"sigma_v   = {p.sigma_v}")
    print(f"sigma_x0  = {p.sigma_x0}")
    print(f"Q         = {p.Q}")
    print(f"R         = {p.R}")
    print(f"x0        = {p.x0}")
    print(f"P0        = {p.P0}")
    print(f"seed      = {p.seed}")


def print_simulation_summary(u: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    print("\nSimulation summary, like NLSim.m")
    print("--------------------------------")
    print(f"{'':>8}  {'u':>12}  {'x':>12}  {'y':>12}")
    print(f"{'Mean':>8}  {np.mean(u):12.6g}  {np.mean(x):12.6g}  {np.mean(y):12.6g}")
    print(f"{'Std':>8}  {np.std(u):12.6g}  {np.std(x):12.6g}  {np.std(y):12.6g}")


def print_filter_summary(x: np.ndarray, y: np.ndarray, out: dict) -> None:
    ytm = y - out["YHM"]
    xtm = x - out["XHM"]
    xtp = x - out["XHP"]

    print("\nUKF residual summary, like EKF.m")
    print("--------------------------------")
    print(f"{'':>8}  {'ytm':>12}  {'xtm':>12}  {'xtp':>12}")
    print(f"{'Mean':>8}  {np.mean(ytm):12.6g}  {np.mean(xtm):12.6g}  {np.mean(xtp):12.6g}")
    print(f"{'Std':>8}  {np.std(ytm):12.6g}  {np.std(xtm):12.6g}  {np.std(xtp):12.6g}")
    print(
        f"{'RMSE':>8}  "
        f"{np.sqrt(np.mean(ytm ** 2)):12.6g}  "
        f"{np.sqrt(np.mean(xtm ** 2)):12.6g}  "
        f"{np.sqrt(np.mean(xtp ** 2)):12.6g}"
    )


def simple_autocorrelation(x: np.ndarray, max_lag: int = 24) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.dot(x, x)

    if denom <= 0.0:
        return np.zeros(max_lag + 1)

    rho = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        rho[lag] = np.dot(x[: len(x) - lag], x[lag:]) / denom

    return rho


def plot_results(x: np.ndarray, y: np.ndarray, u: np.ndarray, out: dict, p: Params) -> None:
    k = np.arange(1, p.n + 1)

    plt.figure(1)
    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(k, u)
    plt.title("u")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(k, x)
    plt.title("x")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(k, y)
    plt.title("y")
    plt.grid(True)

    plt.tight_layout()

    plt.figure(2)
    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(k, x, label="x")
    plt.plot(k, out["XHM"], label="XHM prior")
    plt.plot(k, out["XHP"], label="XHP filtered")
    plt.title("x, XHM, XHP")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(k, x - out["XHM"])
    plt.title("x - XHM")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(k, x - out["XHP"])
    plt.title("x - XHP")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(k, y - out["YHM"])
    plt.title("y - YHM")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    max_lag = min(24, p.n - 1)
    rho = simple_autocorrelation(y - out["YHM"], max_lag=max_lag)
    lags = np.arange(max_lag + 1)
    conf = 1.96 / np.sqrt(p.n)
    plt.plot(lags, rho, "-x")
    plt.axhline(conf, linestyle="--")
    plt.axhline(-conf, linestyle="--")
    plt.title("Residual autocorrelation")
    plt.xlabel("lag")
    plt.grid(True)

    plt.tight_layout()

    plt.figure(3)
    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(k, out["K"])
    plt.title("K")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(k, out["PM"])
    plt.title("Pm = prior variance")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(k, out["PP"])
    plt.title("Pp = posterior variance")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    p = make_params(CASE, SEED)

    print_parameters(p)

    x, y, u = simulate_nonlinear_system(p)
    out = ukf_scalar(y, u, p)

    print_simulation_summary(u, x, y)
    print_filter_summary(x, y, out)

    if SHOW_PLOTS:
        plot_results(x, y, u, out, p)


if __name__ == "__main__":
    main()