# README - Nonlinear State Estimation using Kalman Filter and Extended Kalman Filter

## 1. Introduction

This project compares a standard linear Kalman Filter (KF) and an Extended Kalman Filter (EKF) on a nonlinear dynamical system.

The true system contains nonlinear state dynamics and nonlinear measurements:

```math
x_k = a \sin(x_{k-1} + \phi_f) + b u_{k-1} + w_{k-1}
```

```math
z_k = \sin(c x_k + \phi_h) + v_k
```

where:

* $x_k$ = system state
* $z_k$ = measurement
* $u_k$ = control input
* $w_k \sim \mathcal N(0,Q)$ = process noise
* $v_k \sim \mathcal N(0,R)$ = measurement noise

---

# 2. System Model

## 2.1 State Equation

The state evolves according to

```math
x_k = f(x_{k-1},u_{k-1}) + w_{k-1}
```

with

```math
f(x,u)=a\sin(x+\phi_f)+bu
```

The parameter $a$ controls the system dynamics while $b$ scales the external input.

---

## 2.2 Measurement Equation

Measurements are generated from

```math
z_k = h(x_k)+v_k
```

where

```math
h(x)=\sin(c x+\phi_h)
```

The parameter $c$ controls the nonlinearity of the sensor.

---

## 2.3 Input Signal

The system is excited using a square wave:

```math
u_k = \mathrm{square}(2\pi f_u k)
```

where $f_u$ is the input frequency.

---

# 3. Noise Statistics

## 3.1 Process Noise

Process noise is Gaussian:

```math
w_k \sim \mathcal N(0,Q)
```

with variance

```math
Q=\sigma_w^2
```

---

## 3.2 Measurement Noise

Measurement noise is Gaussian:

```math
v_k \sim \mathcal N(0,R)
```

with variance

```math
R=\sigma_v^2
```

---

## 3.3 Initial State Uncertainty

Initial conditions are

```math
x_0 \sim \mathcal N(\hat{x}_0,P_0)
```

where

```math
P_0=\sigma_{x_0}^2
```

---

# 4. Linear Kalman Filter

The linear KF assumes the system can be approximated as

```math
x_k = \Phi x_{k-1} + b u_{k-1} + w_{k-1}
```

```math
z_k = H x_k + v_k
```

using

```math
\Phi = a
```

```math
H = c
```

This approximation is only accurate near the origin.

---

## 4.1 Prediction Step

State prediction:

```math
\hat{x}_{k|k-1}
=
\Phi \hat{x}_{k-1|k-1}
+
b u_{k-1}
```

Covariance prediction:

```math
P_{k|k-1}
=
\Phi P_{k-1|k-1}\Phi^T
+
Q
```

---

## 4.2 Update Step

Predicted measurement:

```math
\hat{z}_k
=
H\hat{x}_{k|k-1}
```

Innovation:

```math
r_k
=
z_k-\hat{z}_k
```

Innovation covariance:

```math
S_k
=
H P_{k|k-1}H^T + R
```

Kalman gain:

```math
K_k
=
P_{k|k-1}H^TS_k^{-1}
```

State update:

```math
\hat{x}_{k|k}
=
\hat{x}_{k|k-1}
+
K_k r_k
```

Covariance update:

```math
P_{k|k}
=
(I-K_kH)P_{k|k-1}
```

---

# 5. Extended Kalman Filter (EKF)

The EKF handles nonlinear systems by linearizing around the current estimate.

---

## 5.1 Nonlinear Prediction

State prediction:

```math
\hat{x}_{k|k-1}
=
a\sin(\hat{x}_{k-1|k-1}+\phi_f)
+
b u_{k-1}
```

Measurement prediction:

```math
\hat{z}_k
=
\sin(c\hat{x}_{k|k-1}+\phi_h)
```

---

## 5.2 Jacobian of State Function

The EKF requires the derivative of the state equation:

```math
\Phi_k
=
\frac{\partial f}{\partial x}
```

For

```math
f(x)=a\sin(x+\phi_f)
```

the Jacobian becomes

```math
\Phi_k
=
a\cos(\hat{x}_{k|k}+\phi_f)
```

---

## 5.3 Jacobian of Measurement Function

The measurement Jacobian is

```math
H_k
=
\frac{\partial h}{\partial x}
```

For

```math
h(x)=\sin(c x+\phi_h)
```

we obtain

```math
H_k
=
c\cos(c\hat{x}_{k|k-1}+\phi_h)
```

---

## 5.4 EKF Covariance Prediction

```math
P_{k|k-1}
=
\Phi_k
P_{k-1|k-1}
\Phi_k^T
+
Q
```

---

## 5.5 EKF Update

Innovation:

```math
r_k
=
z_k-\hat{z}_k
```

Innovation covariance:

```math
S_k
=
H_kP_{k|k-1}H_k^T+R
```

Kalman gain:

```math
K_k
=
P_{k|k-1}
H_k^T
S_k^{-1}
```

State update:

```math
\hat{x}_{k|k}
=
\hat{x}_{k|k-1}
+
K_k r_k
```

Covariance update:

```math
P_{k|k}
=
(I-K_kH_k)
P_{k|k-1}
```

---

# 6. Performance Metrics

Three performance measures are used.

---

## 6.1 Prior State Error

Prediction error:

```math
e^-_k
=
x_k-\hat{x}_{k|k-1}
```

Root Mean Square Error (RMSE):

```math
RMSE_{prior}
=
\sqrt{
\frac1N
\sum_{k=1}^{N}
(e^-_k)^2
}
```

---

## 6.2 Posterior State Error

Estimation error:

```math
e^+_k
=
x_k-\hat{x}_{k|k}
```

RMSE:

```math
RMSE_{posterior}
=
\sqrt{
\frac1N
\sum_{k=1}^{N}
(e^+_k)^2
}
```

---

## 6.3 Residual Error

Residual:

```math
r_k
=
z_k-\hat{z}_k
```

RMSE:

```math
RMSE_{residual}
=
\sqrt{
\frac1N
\sum_{k=1}^{N}
r_k^2
}
```

---

# 7. Task 4 Results

## 7.1 Task 4a

Parameters:

```math
c=1,\quad \phi_f=0,\quad \phi_h=0
```

Both filters perform well.

The EKF achieves lower RMSE because it uses the true nonlinear model.

---

## 7.2 Task 4b

Parameters:

```math
c=1,\quad \phi_f=0,\quad \phi_h=\frac{\pi}{16}
```

The measurement model is phase shifted.

The KF assumes a linear measurement model, while the true measurement contains a nonlinear phase offset. The EKF compensates through the correct measurement function.

---

## 7.3 Task 4c

Parameters:

```math
c=1,\quad \phi_f=\frac{\pi}{16},\quad \phi_h=0
```

The state dynamics are phase shifted.

The KF uses an incorrect process model and accumulates prediction error. The EKF continuously linearizes around the current estimate and remains accurate.

---

## 7.4 Task 4d

Parameters:

```math
c=10,\quad \phi_f=0,\quad \phi_h=0
```

The measurement function becomes

```math
z_k=\sin(10x_k)
```

which oscillates rapidly.

The fixed linear approximation used by the KF becomes invalid. The EKF still performs better because the Jacobian is recomputed at every time step, although performance degrades compared to the simpler cases.

---

# 8. Conclusion

The linear Kalman Filter assumes fixed linear dynamics and measurements. This works only when the nonlinear system operates close to the linearization point.

The Extended Kalman Filter updates its linearization at every iteration using the Jacobians

```math
\Phi_k
=
a\cos(\hat{x}_{k|k}+\phi_f)
```

and

```math
H_k
=
c\cos(c\hat{x}_{k|k-1}+\phi_h)
```

allowing it to track nonlinear behavior significantly more accurately than the standard KF.
