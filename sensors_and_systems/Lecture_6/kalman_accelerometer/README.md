# Kalman filter for accelerometer-based positioning

Arduino Nano 33 IoT slides on a table. Only the x-axis accelerometer is available.
Goal: estimate acceleration, velocity and position from that one noisy signal.
Three methods run in parallel and stream labeled data at 13 Hz for comparison in MATLAB.

---

## The core problem

Integrating acceleration to get position sounds simple. The issue is bias — a
small constant offset in the accelerometer output that every MEMS sensor has.
Integrate a 0.05 m/s² bias once and you get a velocity that grows at 0.05 m/s
per second. Integrate that again and position drifts as 0.025t² meters. After
ten seconds that is 2.5 meters of error even though the Arduino barely moved.

The Kalman filter handles this by building a probabilistic model of the
physics and using it to constrain what the filter will believe.

---

## Code structure

**IMUExtended** inherits from the library class to add sample rate control.
The base class hardcodes 104 Hz; we write directly to CTRL1\_XL (0x10) and
CTRL2\_G (0x11) to set 13 Hz, which matches Tₛ = 1/13 s used throughout.

**Parameters** at the top of the file are the only things that need tuning.
Collect ~1 minute of stationary data and compute mean and standard deviation
of the x-axis readings. The mean is the bias (individual to each Arduino),
the standard deviation is σᵥ. See the parameter section below.

**benchmarkUpdate** is the naive double-integration baseline.
No filtering, just forward Euler. Useful to see how badly bias accumulates.

**kalmanUpdate** runs the 3-state filter: states are acceleration,
velocity, position. Bias is not modeled, so the filter will drift similarly
to the benchmark if the Arduino has significant bias.

**kalmanBiasUpdate** adds bias as a fourth state. The filter actively
estimates and subtracts the bias.

---

## State space model

### Acceleration: AR(1) process

Acceleration is treated as a random process — each sample is a scaled version
of the previous one plus noise:

```
a(k+1) = φ · a(k) + wₐ
```

φ controls how quickly acceleration returns toward zero between pushes.
It comes from a desired bandwidth ωb:

```
φ = exp(−ωb · Tₛ)
```

Low ωb (e.g. 0.1 Hz) → acceleration assumed to change slowly.
High ωb → filter expects quick changes in direction.

The noise σqa is set so the long-run standard deviation of acceleration
equals σa (reasonable max acceleration):

```
σa² = σqa² / (1 − φ²)
  ⟹  σqa = σa · √(1 − φ²)
```

### Kinematics: forward Euler

```
v(k+1) = v(k) + Tₛ · a(k)
p(k+1) = p(k) + Tₛ · v(k)
```

### Bias: random walk

Bias changes very slowly, modeled as integration of tiny random steps:

```
b(k+1) = b(k) + wb
```

σqb controls how fast the bias is allowed to drift. Keep it small —
MEMS bias at room temperature barely moves between samples.

### State transition matrix Φ

Both filters use **x(k+1) = Φ · x(k)**:

3-state `x = [a, v, p]ᵀ` (no bias):
```
Φ = [ φ    0    0  ]    ← AR(1) acceleration decay
    [ Tₛ   1    0  ]    ← Euler velocity update
    [ 0    Tₛ   1  ]    ← Euler position update
```

4-state `x = [a, v, p, b]ᵀ` (with bias):
```
Φ = [ φ    0    0   0 ]    ← same as above
    [ Tₛ   1    0   0 ]
    [ 0    Tₛ   1   0 ]
    [ 0    0    0   1 ]    ← bias carries forward unchanged
```

### Process noise Q

Only acceleration and bias are driven by noise:

3-state:
```
Q = [ σqa²   0   0 ]
    [ 0       0   0 ]
    [ 0       0   0 ]
```

4-state:
```
Q = [ σqa²   0   0   0    ]
    [ 0       0   0   0    ]
    [ 0       0   0   0    ]
    [ 0       0   0   σqb² ]
```

---

## Measurement model

The accelerometer measures acceleration (plus bias in the 4-state case).
H picks out the relevant states from x:

```
kalman:      H = [ 1  0  0  ]      y = a + noise
kalmanBias:  H = [ 1  0  0  1 ]    y = a + b + noise
```

Measurement noise variance **R = σᵥ²**.

The 4-state filter separates acceleration from bias because they have different
dynamics — a decays (AR(1), φ < 1) while b persists (random walk, φ = 1).

---

## Kalman filter algorithm

Both `kalmanUpdate` and `kalmanBiasUpdate` are the same algorithm at different sizes.

**Predict** — roll state forward before seeing new data:

```
x̂⁻ = Φ · x̂
P⁻  = Φ · P · Φᵀ + Q
```

P is the covariance matrix tracking filter uncertainty. Adding Q accounts for
process noise injected at this step.

**Update** — correct prediction using new measurement y:

```
ν  = y − H · x̂⁻              (innovation: how wrong the prediction was)
S  = H · P⁻ · Hᵀ + R          (innovation covariance)
K  = P⁻ · Hᵀ · S⁻¹            (Kalman gain)
x̂  = x̂⁻ + K · ν              (corrected state)
P  = (I − K · H) · P⁻         (corrected covariance)
```

When P⁻ is large (model uncertain), K is large → lean on measurement.
When R is large (sensor noisy), K is small → trust the prediction.
The filter finds the optimal balance automatically.

---

## Parameters

Measure from stationary data before running the experiment:

**σᵥ** — standard deviation of x-axis readings while the Arduino sits still.
Typical: 0.03–0.08 m/s².

**σa** — rough upper bound on acceleration during the slide.
Gentle push: ~1–2 m/s², hard push: ~4 m/s². Start at 1.5.

**ωb** — bandwidth of the AR(1) model. A 1 s back-and-forth ≈ 6 rad/s.
Start at 0.5 Hz (≈ 3.14 rad/s), increase if the filter reacts too slowly.

**σqb** — keep very small (0.001–0.005). If the bias estimate jumps, reduce it.

**Initial covariance P₀** — start at rest so velocity and position uncertainty
can be small. The bias diagonal entry P₀(3,3) should reflect prior uncertainty
about the bias — σa² is a reasonable conservative choice.

---

## Output columns

Connect via Serial Monitor at 115200 baud.

| Label       | Description                                    |
|-------------|------------------------------------------------|
| t_ms        | Timestamp [ms]                                 |
| accel       | Raw x-axis accelerometer [m/s²]                |
| bench_v     | Benchmark velocity [m/s]                       |
| bench_p     | Benchmark position [m]                         |
| kf_a        | kalman: estimated acceleration [m/s²]          |
| kf_v        | kalman: estimated velocity [m/s]               |
| kf_p        | kalman: estimated position [m]                 |
| kf_stdv     | kalman: velocity 1σ uncertainty [m/s]          |
| kf_stdp     | kalman: position 1σ uncertainty [m]            |
| kfb_a       | kalmanBias: estimated acceleration [m/s²]      |
| kfb_v       | kalmanBias: estimated velocity [m/s]           |
| kfb_p       | kalmanBias: estimated position [m]             |
| kfb_b       | kalmanBias: estimated sensor bias [m/s²]       |
| kfb_stdv    | kalmanBias: velocity 1σ uncertainty [m/s]      |
| kfb_stdp    | kalmanBias: position 1σ uncertainty [m]        |

Plot each state with ±2σ bounds shaded. True position starts and ends at zero —
use that as a visual sanity check on filter performance.
