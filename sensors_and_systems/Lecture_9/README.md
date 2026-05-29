# Lecture 9 — Kalman Filter & Extended Kalman Filter

Python conversion of the original MATLAB files (`LSim.m + KF.m` → `KF.py`, `NLSim.m + EKF.m` → `EKF.py`).

---

## How to run

From the `sensors_and_systems/` root (where `pyproject.toml` lives):

```sh
uv sync
uv run python Lecture_9/KF.py     # runs Kalman Filter on the linear system
uv run python Lecture_9/EKF.py    # runs Extended Kalman Filter on the nonlinear system
```

Plots and a text summary are saved automatically to `Lecture_9/results/`.

---

## System Models

### Linear system — `KF.py`

```
x[i] = a*x[i-1] + b*u[i-1] + w[i-1]     process equation
y[i] = c*x[i] + v[i]                      measurement equation

w ~ N(0, Q),  v ~ N(0, R)
```

Parameters used (from `LSim.m`):

| Symbol   | Value   | Meaning |
|----------|---------|---------|
| `a`      | 0.95    | State transition — slow AR(1), time constant ≈ 19 steps |
| `b`      | 0.05    | Input gain = k*(1-a) with k=1, so DC gain = b/(1-a) = 1 |
| `c`      | 1.0     | Direct state observation |
| `fu`     | 0.02    | Normalized frequency of square-wave input → period = 50 steps |
| `sigmaw` | 0.0156  | = 0.05 * sqrt(1 − 0.95²), keeps Var(x_ss) = 0.0025 |
| `sigmav` | 0.01    | Small measurement noise (active branch in `LSim.m`) |

### Nonlinear system — `EKF.py`

```
x[i] = a*sin(x[i-1] + phi_f) + b*u[i-1] + w[i-1]    process equation
y[i] = sin(c*x[i] + phi_h) + v[i]                     measurement equation
```

Parameters used (from `NLSim.m`, active branches only):

| Symbol   | Value   | Meaning |
|----------|---------|---------|
| `a`      | 0.95    | Amplitude of sin in state transition |
| `c`      | 10.0    | Steep measurement nonlinearity (active: `if 1; c=10; end`) |
| `phi_f`  | 0.0     | Phase offset in state transition |
| `phi_h`  | 0.0     | Phase offset in measurement |
| `sigmav` | 0.1     | Larger noise than linear case |

---

## Kalman Filter Algorithm (KF)

### Initialisation

```
xhm = x0,   Pm = P0,   yhm = c * xhm
```

Numerical example with `a=0.95, c=1, Q=2.44e-4, R=1e-4, P0=2.44e-4, x0=0`:
- `xhm = 0`,  `Pm = 2.44e-4`,  `yhm = 0`

### One cycle (repeated for i = 0, 1, …, n-1)

**Step 1 — Measurement update (correct with y[i]):**

```
K   = Pm * c / (c * Pm * c + R)          Kalman gain
xhp = xhm + K * (y[i] - yhm)             posterior state
Pp  = (1-K*c)*Pm*(1-K*c) + K*R*K         posterior covariance (Joseph form)
```

**Step 2 — Time update (predict next step):**

```
xhm = a * xhp + b * u[i]
Pm  = a * Pp * a + Q
yhm = c * xhm
```

#### Concrete walk-through (step i=0)

Given `xhm=0, Pm=2.44e-4, y[0]=0.05, u[0]=1`:

```
K   = 2.44e-4 * 1 / (1 * 2.44e-4 * 1 + 1e-4) = 2.44e-4 / 3.44e-4 = 0.709
xhp = 0 + 0.709 * (0.05 - 0)                  = 0.0355
Pp  = (1-0.709)^2 * 2.44e-4 + 0.709^2 * 1e-4  = 2.07e-5 + 5.03e-5 = 7.1e-5
xhm = 0.95 * 0.0355 + 0.05 * 1                 = 0.0837  (next step prior)
Pm  = 0.95^2 * 7.1e-5 + 2.44e-4                = 6.41e-5 + 2.44e-4 = 3.08e-4
```

K ≈ 0.71 means the filter trusts the measurement more than the prediction (R < Pm).  
After the update Pp drops from 2.44e-4 to 7.1e-5 — uncertainty reduced by 3×.

---

## Extended Kalman Filter Algorithm (EKF)

The EKF handles nonlinear functions by replacing them with their first-order
Taylor (Jacobian) approximations evaluated at the current estimate.

### Jacobians

| Jacobian | Formula | Where evaluated |
|----------|---------|-----------------|
| `H`  (measurement)         | `c * cos(c * xhm + phi_h)` | at `xhm` (before measurement update) |
| `Phi` (state transition)   | `a * cos(xhp + phi_f)`     | at `xhp` (after measurement update)  |

### One cycle

**Step 1 — Measurement update:**

```
H   = c * cos(c * xhm + phi_h)
K   = Pm * H / (H * Pm * H + R)
xhp = xhm + K * (y[i] - yhm)
Pp  = (1-K*H)*Pm*(1-K*H) + K*R*K
```

**Step 2 — Time update:**

```
xhm = a * sin(xhp + phi_f) + b * u[i]    (nonlinear propagation)
Phi = a * cos(xhp + phi_f)               (linearized covariance propagation)
Pm  = Phi * Pp * Phi + Q
yhm = sin(c * xhm + phi_h)
```

#### Concrete walk-through (step i=0)

Given `c=10, a=0.95, xhm=0.0, Pm=2.44e-4, y[0]=0.4, u[0]=1, R=0.01`:

```
H   = 10 * cos(10 * 0.0 + 0) = 10 * cos(0) = 10.0
K   = 2.44e-4 * 10 / (10 * 2.44e-4 * 10 + 0.01) = 2.44e-3 / (2.44e-2 + 0.01) = 0.0718
xhp = 0.0 + 0.0718 * (0.4 - sin(0)) = 0.0287
Pp  = (1 - 0.0718*10)^2 * 2.44e-4 + 0.0718^2 * 0.01 = 4.3e-5 + 5.15e-5 ≈ 9.5e-5
Phi = 0.95 * cos(0.0287 + 0) ≈ 0.9496
xhm = 0.95 * sin(0.0287) + 0.05 * 1 ≈ 0.0272 + 0.05 = 0.0772
Pm  = 0.9496^2 * 9.5e-5 + 2.44e-4 ≈ 8.57e-5 + 2.44e-4 = 3.3e-4
```

With `c=10`, the measurement Jacobian H=10 amplifies the innovation significantly —
the filter sees the measurement as very informative about x.

---

## Using the classes directly

```python
from KF import KalmanFilter, simulate_linear

# Simulate data
u, x, y, params = simulate_linear(n=200, seed=0)

# Create and run filter
kf = KalmanFilter(a=params['a'], b=params['b'], c=params['c'],
                  Q=params['Q'], R=params['R'],
                  x0=params['x0'], P0=params['P0'])
results = kf.run(y, u)

# results is a dict with keys: XHM, YHM, XHP, K, Pm, Pp (each np.ndarray of shape (n,))
print(results['XHP'])   # posterior state estimates
```

```python
from EKF import ExtendedKalmanFilter, simulate_nonlinear

u, x, y, params = simulate_nonlinear(n=200, seed=0)

ekf = ExtendedKalmanFilter(a=params['a'], b=params['b'], c=params['c'],
                           Q=params['Q'], R=params['R'],
                           phi_f=params['phi_f'], phi_h=params['phi_h'],
                           x0=params['x0'], P0=params['P0'])

# Run step by step
ekf.reset()
for i in range(len(y)):
    r = ekf.step(y[i], u[i])
    print(f"i={i}  xhm={r['xhm']:.4f}  K={r['K']:.4f}  xhp={r['xhp']:.4f}")
```

---

## Output files

After running, `results/` contains:

| File | Content |
|------|---------|
| `kf_simulation.png`  | Input u, true state x, measurement y |
| `kf_estimates.png`   | x vs XHM/XHP, prior/posterior errors, innovations |
| `kf_gains.png`       | Kalman gain K, covariances Pm and Pp over time |
| `kf_results.txt`     | Mean, Std, RMSE of ytm, xtm, xtp |
| `ekf_simulation.png` | Same plots for the nonlinear system |
| `ekf_estimates.png`  | EKF state estimates and residuals |
| `ekf_gains.png`      | EKF gain K and covariances |
| `ekf_results.txt`    | EKF residual statistics |

---

## Key differences: KF vs EKF

| Aspect | KF | EKF |
|--------|-----|-----|
| State transition | `a * xhp` (linear) | `a * sin(xhp + phi_f)` (nonlinear) |
| Measurement function | `c * xhm` (linear) | `sin(c * xhm + phi_h)` (nonlinear) |
| Covariance propagation | exact (`Phi = a` constant) | approximate (`Phi = a*cos(xhp+phi_f)` varies) |
| Measurement update | H = c (constant) | H = c*cos(c*xhm+phi_h) (varies per step) |
| Accuracy | Optimal for Gaussian linear systems | Approximate; degrades with strong nonlinearity |
