# Lecture 9 — Kalman Filter, Extended Kalman Filter & UKF Exercises

All five exercises are in a single file: `lecture9_all_exercises.py`.
KF, EKF, and UKF are implemented as plain functions (no classes), ported from Lecture 4.

---

## How to run

From the `sensors_and_systems/` root (where `pyproject.toml` lives):

```sh
uv sync
uv run python Lecture_9/lecture9_all_exercises.py
```

All plots and text summaries are saved to `Lecture_9/results/`.

---

## Exercises overview

| Exercise | Topic | Key insight |
|----------|-------|-------------|
| **Exercise 1** | Whiteness test | Innovations of a well-tuned filter are uncorrelated (white noise) |
| **Exercise 2** | Normality test | KF innovations are Gaussian; EKF/UKF may deviate due to nonlinearity |
| **Exercise 3** | Uniform vs Normal noise | KF is BLUE for any noise; normality fails when noise is not Gaussian |
| **Exercise 4** | Effect of sample size N | CI width shrinks as 1/sqrt(N); larger N detects smaller correlations |
| **Exercise 5** | EKF joint state + parameter estimation | Augment state [x; a] so EKF learns the true 'a' online from measurements |

### Filter implementations (ported from Lecture 4)

| Filter | Source function | System |
|--------|----------------|--------|
| KF  | `Lecture_4/main.py` → `linear_kalman_filter()` | Linear: `y = c*x + v` |
| EKF | `Lecture_4/main.py` → `extended_kalman_filter()` | Nonlinear: `y = sin(c*x + phi_h) + v` |
| UKF | `Lecture_4/lecture4_exercises.py` → `ukf_scalar()` | Nonlinear (same as EKF) |

---

## System Models

### Linear system (used in KF exercises)

```
x[i] = a*x[i-1] + b*u[i-1] + w[i-1]     process equation
y[i] = c*x[i] + v[i]                      measurement equation

w ~ N(0, Q),  v ~ N(0, R)
```

| Symbol   | Value  | Meaning |
|----------|--------|---------|
| `a`      | 0.95   | State transition — slow AR(1), time constant ~19 steps |
| `b`      | 0.05   | Input gain = 1 - a (DC gain = 1) |
| `c`      | 1.0    | Direct state observation |
| `sigmaw` | 0.0156 | = 0.05 * sqrt(1 - 0.95^2) |
| `sigmav` | 0.01   | Small measurement noise |

### Nonlinear system (used in EKF / UKF exercises)

```
x[i] = a*sin(x[i-1] + phi_f) + b*u[i-1] + w[i-1]    process equation
y[i] = sin(c*x[i] + phi_h) + v[i]                     measurement equation
```

| Symbol   | Value  | Meaning |
|----------|--------|---------|
| `c`      | 10.0   | Steep measurement nonlinearity — oscillates rapidly |
| `phi_f`  | 0.0    | Phase offset in state transition |
| `phi_h`  | 0.0    | Phase offset in measurement |
| `sigmav` | 0.1    | Larger noise than linear case |

---

## Kalman Filter Algorithm (KF)

### Initialisation

```
xhm = x0,   Pm = P0,   yhm = c * xhm
```

### One cycle (i = 0, 1, ..., n-1)

**Measurement update:**

```
K   = Pm * c / (c * Pm * c + R)
xhp = xhm + K * (y[i] - yhm)
Pp  = (1-K*c)*Pm*(1-K*c) + K*R*K    (Joseph form)
```

**Time update:**

```
xhm = a * xhp + b * u[i]
Pm  = a * Pp * a + Q
yhm = c * xhm
```

#### Numerical example (i=0, a=0.95, c=1, P0=2.44e-4, R=1e-4, y[0]=0.05, u[0]=1)

```
K   = 2.44e-4 / (2.44e-4 + 1e-4)  = 0.709
xhp = 0 + 0.709 * (0.05 - 0)      = 0.0355
Pp  = (1-0.709)^2 * 2.44e-4 + 0.709^2 * 1e-4 = 7.1e-5
xhm = 0.95 * 0.0355 + 0.05 * 1    = 0.0837
Pm  = 0.95^2 * 7.1e-5 + 2.44e-4   = 3.08e-4
```

K = 0.71 means the filter trusts the measurement more than the prediction (R < Pm).
Pp drops from 2.44e-4 to 7.1e-5 — uncertainty reduced by 3x after the update.

---

## Extended Kalman Filter Algorithm (EKF)

Replaces constant Jacobians with ones computed at the current estimate each step.

| Jacobian | Formula | Evaluated at |
|----------|---------|--------------|
| `H`  (measurement)       | `c * cos(c * xhm + phi_h)` | `xhm` (before update) |
| `Phi` (state transition) | `a * cos(xhp + phi_f)`     | `xhp` (after update)  |

### One cycle

**Measurement update:**

```
H   = c * cos(c * xhm + phi_h)
K   = Pm * H / (H * Pm * H + R)
xhp = xhm + K * (y[i] - yhm)
Pp  = (1-K*H)*Pm*(1-K*H) + K*R*K
```

**Time update:**

```
Phi = a * cos(xhp + phi_f)
xhm = a * sin(xhp + phi_f) + b * u[i]
Pm  = Phi * Pp * Phi + Q
yhm = sin(c * xhm + phi_h)
```

#### Numerical example (i=0, c=10, P0=2.44e-4, R=0.01, xhm=0, y[0]=0.4, u[0]=1)

```
H   = 10 * cos(0) = 10.0
K   = 2.44e-4 * 10 / (10 * 2.44e-4 * 10 + 0.01) = 0.0718
xhp = 0 + 0.0718 * (0.4 - 0) = 0.0287
Pp  = (1 - 0.718)^2 * 2.44e-4 + 0.0718^2 * 0.01 = 9.5e-5
Phi = 0.95 * cos(0.0287) = 0.9496
xhm = 0.95 * sin(0.0287) + 0.05 = 0.0772
Pm  = 0.9496^2 * 9.5e-5 + 2.44e-4 = 3.3e-4
```

---

## Unscented Kalman Filter Algorithm (UKF)

Avoids Jacobians by propagating 3 sigma points through the exact nonlinear functions.

### Parameters (alpha=1, kappa=2, beta=0, n_state=1)

```
lambda = alpha^2 * (n_state + kappa) - n_state = 2
scale  = n_state + lambda = 3

Sigma points:  X[0] = xhm
               X[1] = xhm + sqrt(scale * Pm)
               X[2] = xhm - sqrt(scale * Pm)

Weights:  Wm[0] = 2/3,  Wm[1] = Wm[2] = 1/6
          Wc[0] = 2/3,  Wc[1] = Wc[2] = 1/6
```

### One cycle

**Measurement update:**

```
X_sigma = sigma_points(xhm, Pm)
Y_sigma = sin(c * X_sigma + phi_h)

z_hat = sum(Wm * Y_sigma)
S     = R + sum(Wc * (Y_sigma - z_hat)^2)
Cxy   = sum(Wc * (X_sigma - xhm) * (Y_sigma - z_hat))

K   = Cxy / S
xhp = xhm + K * (y[i] - z_hat)
Pp  = Pm - K * S * K
```

**Time update:**

```
X_sigma = sigma_points(xhp, Pp)
X_next  = a * sin(X_sigma + phi_f) + b * u[i]

xhm = sum(Wm * X_next)
Pm  = Q + sum(Wc * (X_next - xhm)^2)
```

#### Numerical example (i=0, c=10, xhm=0, Pm=2.44e-4, R=0.01)

```
X_sigma = [0.0,  +0.02708,  -0.02708]
Y_sigma = sin(10 * X_sigma) = [0.0,  0.2703,  -0.2703]

z_hat = 0.0
S     = 0.01 + 2*(1/6)*0.2703^2 = 0.0344
Cxy   = 2*(1/6)*0.02708*0.2703  = 0.00244
K     = 0.00244 / 0.0344        = 0.0710
```

---

## KF vs EKF vs UKF comparison

| Aspect | KF | EKF | UKF |
|--------|-----|-----|-----|
| State transition | `a * xhp` (linear) | `a * sin(xhp + phi_f)` | `a * sin(xhp + phi_f)` |
| Measurement function | `c * xhm` (linear) | `sin(c * xhm + phi_h)` | `sin(c * xhm + phi_h)` |
| Covariance propagation | Exact (Phi = a, constant) | 1st-order Jacobian | Unscented transform |
| Jacobians needed | Yes (trivially constant) | Yes (derived analytically) | No |
| Accuracy | Optimal for linear + Gaussian | Degrades with strong nonlinearity | Better than EKF for highly nonlinear systems |

---

## Output files (`results/`)

| File | Content |
|------|---------|
| `ex1_whiteness.png` | Autocorrelation — 4 panels: KF, EKF, UKF, KF mismatch |
| `ex1_whiteness.txt` | Chi-squared p-values and whiteness verdict |
| `ex2_normality.png` | Q-Q plots — 3 panels: KF, EKF, UKF |
| `ex2_normality.txt` | Shapiro-Wilk W and p-values |
| `ex3_noise_types.png` | 2x4 grid: autocorrelation + Q-Q for 4 noise combinations |
| `ex3_noise_types.txt` | p_white and p_normal for each noise type |
| `ex4_n_effect.png` | Autocorrelation + histogram for N = 10, 100, 1000, 10000 |
| `ex4_n_effect.txt` | RMSE, p-value, CI width for each N |
| `ex5_ekf_states.png`      | x state estimates: true x, XHM, XHP, innovations |
| `ex5_ekf_parameter.png`   | parameter a convergence + variance on a over time |
| `ex5_ekf_gains.png`       | Kalman gains K[0], K[1] and diagonal covariances  |
| `ex5_ekf_results.txt`     | Final a estimate, error, Pm[1,1], method summary  |
