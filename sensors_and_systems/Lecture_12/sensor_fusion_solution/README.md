# Asynchronous Sensor Fusion with a Kalman Filter

This project solves the attached sensor-fusion exercise in Python. Two sensors measure a two-state discrete-time process, but their measurements do **not** arrive at the same sample times. The implementation uses one centralized Kalman filter and performs a measurement update only when a sensor value is available.

## 1. Exercise model

The process is

$$
x_{k+1} = A x_k + Q^{1/2}w_k,
\qquad
w_k \sim \mathcal{N}(0, I),
$$

with

$$
A =
\begin{bmatrix}
0.7360 & 0.2560 \\
0.3200 & 0.6480
\end{bmatrix},
\qquad
Q = 0.4 I.
$$

The complete measurement model is

$$
y_k = Hx_k + R^{1/2}v_k,
\qquad
v_k \sim \mathcal{N}(0, I),
$$

where

$$
H =
\begin{bmatrix}
0.50 & 0.50 \\
0 & 1
\end{bmatrix},
\qquad
R = 0.2 I.
$$

The individual sensors are therefore

$$
y_{1,k} = 0.5x_{1,k} + 0.5x_{2,k} + \nu_{1,k},
$$

$$
y_{2,k} = x_{2,k} + \nu_{2,k}.
$$

Their schedules are asynchronous:

- sensor 1 is available at $k = 0, 5, 10, 15, \ldots$;
- sensor 2 is available at $k = 1, 6, 11, 16, \ldots$;
- at the remaining samples, no sensor update is possible.

## 2. Kalman-filter equations

The filter stores a **prior** estimate $\hat{x}_{k|k-1}$ before processing any measurement received at sample $k$, and a **posterior** estimate $\hat{x}_{k|k}$ afterward.

### Time update

After processing sample $k$, the filter predicts the next state:

$$
\hat{x}_{k+1|k} = A\hat{x}_{k|k},
$$

$$
P_{k+1|k} = AP_{k|k}A^\mathsf{T} + Q.
$$

There is no deterministic input in this exercise.

### Measurement update with an available sensor

At each sample, the implementation selects the row of $H$ and the variance from $R$ that belong to the sensor that actually delivered a value. Calling the selected quantities $H_k$, $R_k$, and $y_k$, the innovation is

$$
\tilde{y}_k = y_k - H_k\hat{x}_{k|k-1}.
$$

The innovation covariance and Kalman gain are

$$
S_k = H_kP_{k|k-1}H_k^\mathsf{T} + R_k,
$$

$$
K_k = P_{k|k-1}H_k^\mathsf{T}S_k^{-1}.
$$

The posterior state estimate is

$$
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k\tilde{y}_k.
$$

The covariance is updated with the Joseph form

$$
P_{k|k} = (I-K_kH_k)P_{k|k-1}(I-K_kH_k)^\mathsf{T} + K_kR_kK_k^\mathsf{T}.
$$

The Joseph form is algebraically equivalent to the usual covariance update under exact arithmetic, but it is less likely to lose symmetry or positive-semidefiniteness because of floating-point roundoff.

### Samples without measurements

At samples where no sensor value arrives, the filter skips the measurement correction:

$$
\hat{x}_{k|k} = \hat{x}_{k|k-1},
\qquad
P_{k|k} = P_{k|k-1}.
$$

The next time update is then performed normally. Selecting only the available measurement row is equivalent to the non-concurrent-sampling construction in the lecture material, but avoids introducing padded zero measurements in the Python implementation.

## 3. Files

- `sensor_fusion_kalman.py`: readable simulation, filter, plotting, and metric calculation code.
- `results/state_estimation.png`: true states, state estimates, and sampled sensor outputs.
- `results/gain_and_covariance.png`: Kalman-gain and posterior covariance elements as functions of $k$.
- `results/metrics.json`: numerical summary for the default reproducible run.
- `requirements.txt`: minimal package requirements.

## 4. Running the solution

Create an environment with Python 3.10 or newer, install the dependencies, and run:

```bash
python -m pip install -r requirements.txt
python sensor_fusion_kalman.py
```

Useful options:

```bash
python sensor_fusion_kalman.py --num-steps 201 --seed 12 --output-dir results
python sensor_fusion_kalman.py --show
```

## 5. How to interpret the figures

### State and measurement figure

The first two panels compare the true state trajectories with the posterior Kalman-filter estimates. The last two panels show the noisy samples delivered by each sensor against its corresponding noiseless output. Sensor 1 is **not** a direct measurement of $x_1$; it measures the average $0.5x_1 + 0.5x_2$. For that reason, its samples are plotted in a separate panel rather than overlaid on the $x_1$ trajectory.

The estimates are corrected at $k \bmod 5 = 0$ by sensor 1 and at $k \bmod 5 = 1$ by sensor 2. During the next three samples, the estimate is propagated using the model only.

### Gain and covariance figure

The implementation stores an effective $2\times2$ Kalman-gain matrix for plotting. Only the column associated with the sensor that is available can be nonzero. Therefore:

- $K_{11}$ and $K_{21}$ appear only when sensor 1 is sampled;
- $K_{12}$ and $K_{22}$ appear only when sensor 2 is sampled;
- all plotted gain elements are zero during prediction-only samples.

The posterior covariance shows a repeating pattern. Process noise increases uncertainty during time propagation. A sensor arrival reduces uncertainty in the directions observed by that sensor. Since the same two-sensor schedule repeats every five samples, the covariance and gain approach a periodic steady regime rather than a single constant value.

## 6. Conclusions

1. A standard Kalman filter handles asynchronous sensor fusion cleanly: the measurement-update matrices are selected from the sensor values that are actually available.
2. Sensor 2 directly measures $x_2$, so its updates strongly reduce uncertainty in the second state. Through the coupled dynamics matrix $A$, those updates also improve the estimate of $x_1$.
3. Sensor 1 measures a combination of both states, so its update influences both components of the estimate.
4. The three prediction-only samples in every five-sample cycle allow the covariance to grow before the next measurements arrive.
5. Because the measurement schedule is periodic, the gain and covariance become periodic after the initial transient.

## 7. Reproducibility assumptions

The original exercise does not specify an initial true state, an initial prior, or a simulation length. The default run uses:

$$
x_0 = [2, -1]^\mathsf{T},
\qquad
\hat{x}_{0|-1} = [0, 0]^\mathsf{T},
\qquad
P_{0|-1} = I,
$$

with 121 samples and random-number seed 7. These values can be changed in `simulate_and_filter()` or through the command-line arguments where applicable.
