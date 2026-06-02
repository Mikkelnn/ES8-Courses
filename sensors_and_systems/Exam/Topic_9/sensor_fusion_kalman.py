"""Kalman filter sensor-fusion example with asynchronous measurements.

Exercise system
---------------
x[k+1] = A x[k] + Q^(1/2) w[k]
y[k]   = H x[k] + R^(1/2) v[k]

Sensor 1 is available at k = 0, 5, 10, ...
Sensor 2 is available at k = 1, 6, 11, ...

The script simulates the system, runs a centralized Kalman filter that only
uses the measurements available at each sample, and creates the requested
plots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Model:
    """Matrices defining the linear Gaussian state-space model."""

    A: FloatArray
    H: FloatArray
    Q: FloatArray
    R: FloatArray


@dataclass(frozen=True)
class SimulationResult:
    """Outputs from simulation and asynchronous Kalman filtering."""

    x_true: FloatArray
    y_full: FloatArray
    y_available: FloatArray
    x_prior: FloatArray
    x_posterior: FloatArray
    P_prior: FloatArray
    P_posterior: FloatArray
    K_effective: FloatArray
    sensor_used: NDArray[np.int64]


def build_model() -> Model:
    """Return the matrices from the exercise."""
    A = np.array(
        [
            [0.7360, 0.2560],
            [0.3200, 0.6480],
        ],
        dtype=float,
    )
    H = np.array(
        [
            [0.50, 0.50],
            [0.00, 1.00],
        ],
        dtype=float,
    )
    Q = 0.4 * np.eye(2)
    R = 0.2 * np.eye(2)
    return Model(A=A, H=H, Q=Q, R=R)


def available_sensor(k: int) -> int | None:
    """Return the available sensor index at sample k, or None.

    Sensor numbering in the explanation is one-based, but Python indices are
    zero-based:
      - return 0 for sensor 1 at k = 0, 5, 10, ...
      - return 1 for sensor 2 at k = 1, 6, 11, ...
      - return None for prediction-only samples
    """
    remainder = k % 5
    if remainder == 0:
        return 0
    if remainder == 1:
        return 1
    return None


def joseph_covariance_update(
    P_prior: FloatArray,
    K: FloatArray,
    H_active: FloatArray,
    R_active: FloatArray,
) -> FloatArray:
    """Perform the numerically robust Joseph-form covariance update."""
    identity = np.eye(P_prior.shape[0])
    correction = identity - K @ H_active
    P_posterior = correction @ P_prior @ correction.T + K @ R_active @ K.T

    # Remove tiny asymmetries introduced by floating-point arithmetic.
    return 0.5 * (P_posterior + P_posterior.T)


def simulate_and_filter(
    model: Model,
    num_steps: int = 121,
    seed: int = 7,
    x0_true: FloatArray | None = None,
    x0_prior: FloatArray | None = None,
    P0_prior: FloatArray | None = None,
) -> SimulationResult:
    """Simulate the exercise and run an asynchronous centralized Kalman filter."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")

    rng = np.random.default_rng(seed)

    x0_true = np.array([2.0, -1.0]) if x0_true is None else np.asarray(x0_true, dtype=float)
    x0_prior = np.zeros(2) if x0_prior is None else np.asarray(x0_prior, dtype=float)
    P0_prior = np.eye(2) if P0_prior is None else np.asarray(P0_prior, dtype=float)

    x_true = np.zeros((num_steps, 2))
    y_full = np.zeros((num_steps, 2))
    y_available = np.full((num_steps, 2), np.nan)
    x_prior = np.zeros((num_steps, 2))
    x_posterior = np.zeros((num_steps, 2))
    P_prior = np.zeros((num_steps, 2, 2))
    P_posterior = np.zeros((num_steps, 2, 2))

    # K_effective is stored as a 2x2 matrix for easy visualization.  At any
    # sample, only the column associated with the active sensor can be nonzero.
    K_effective = np.zeros((num_steps, 2, 2))
    sensor_used = np.full(num_steps, -1, dtype=np.int64)

    sqrt_Q = np.linalg.cholesky(model.Q)
    sqrt_R = np.linalg.cholesky(model.R)

    # Simulate truth and complete sensor outputs. The filter only receives the
    # subset represented in y_available.
    x_true[0] = x0_true
    for k in range(num_steps):
        measurement_noise = sqrt_R @ rng.standard_normal(2)
        y_full[k] = model.H @ x_true[k] + measurement_noise

        sensor_index = available_sensor(k)
        if sensor_index is not None:
            y_available[k, sensor_index] = y_full[k, sensor_index]

        if k < num_steps - 1:
            process_noise = sqrt_Q @ rng.standard_normal(2)
            x_true[k + 1] = model.A @ x_true[k] + process_noise

    # Initial prior p(x[0] | y[-1]).
    x_prior[0] = x0_prior
    P_prior[0] = P0_prior

    for k in range(num_steps):
        sensor_index = available_sensor(k)

        if sensor_index is None:
            # No measurement is received. The posterior equals the prior.
            x_posterior[k] = x_prior[k]
            P_posterior[k] = P_prior[k]
        else:
            sensor_used[k] = sensor_index
            H_active = model.H[[sensor_index], :]  # shape: (1, 2)
            R_active = model.R[np.ix_([sensor_index], [sensor_index])]  # (1, 1)
            measurement = y_available[k, sensor_index : sensor_index + 1]

            innovation = measurement - H_active @ x_prior[k]
            innovation_covariance = H_active @ P_prior[k] @ H_active.T + R_active
            K_active = P_prior[k] @ H_active.T @ np.linalg.inv(innovation_covariance)

            x_posterior[k] = x_prior[k] + (K_active @ innovation).ravel()
            P_posterior[k] = joseph_covariance_update(
                P_prior=P_prior[k],
                K=K_active,
                H_active=H_active,
                R_active=R_active,
            )
            K_effective[k, :, sensor_index] = K_active.ravel()

        if k < num_steps - 1:
            x_prior[k + 1] = model.A @ x_posterior[k]
            P_prior[k + 1] = model.A @ P_posterior[k] @ model.A.T + model.Q

    return SimulationResult(
        x_true=x_true,
        y_full=y_full,
        y_available=y_available,
        x_prior=x_prior,
        x_posterior=x_posterior,
        P_prior=P_prior,
        P_posterior=P_posterior,
        K_effective=K_effective,
        sensor_used=sensor_used,
    )


def compute_metrics(result: SimulationResult) -> dict[str, object]:
    """Compute a few concise diagnostics for the README and console output."""
    error = result.x_true - result.x_posterior
    rmse_per_state = np.sqrt(np.mean(error**2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(error**2)))

    sensor1_samples = int(np.sum(result.sensor_used == 0))
    sensor2_samples = int(np.sum(result.sensor_used == 1))
    prediction_only_samples = int(np.sum(result.sensor_used == -1))

    return {
        "rmse_per_state": rmse_per_state.tolist(),
        "overall_rmse": overall_rmse,
        "sensor_1_samples": sensor1_samples,
        "sensor_2_samples": sensor2_samples,
        "prediction_only_samples": prediction_only_samples,
        "final_posterior_covariance": result.P_posterior[-1].tolist(),
    }


def plot_states_and_measurements(
    model: Model,
    result: SimulationResult,
    output_path: Path,
) -> None:
    """Create the figure with states, estimates, and asynchronous measurements."""
    k = np.arange(result.x_true.shape[0])
    noiseless_outputs = result.x_true @ model.H.T

    figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)

    axes[0].plot(k, result.x_true[:, 0], label="true $x_1$")
    axes[0].plot(k, result.x_posterior[:, 0], "--", label="estimated $x_1$")
    axes[0].set_ylabel("state value")
    axes[0].set_title("State $x_1$: truth and asynchronous Kalman-filter estimate")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k, result.x_true[:, 1], label="true $x_2$")
    axes[1].plot(k, result.x_posterior[:, 1], "--", label="estimated $x_2$")
    axes[1].set_ylabel("state value")
    axes[1].set_title("State $x_2$: truth and asynchronous Kalman-filter estimate")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    sensor1_mask = ~np.isnan(result.y_available[:, 0])
    axes[2].plot(k, noiseless_outputs[:, 0], label="true $0.5x_1 + 0.5x_2$")
    axes[2].scatter(k[sensor1_mask], result.y_available[sensor1_mask, 0], marker="o", label="sensor 1 samples")
    axes[2].set_ylabel("measurement")
    axes[2].set_title("Sensor 1: available only at $k = 0, 5, 10, \\ldots$")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    sensor2_mask = ~np.isnan(result.y_available[:, 1])
    axes[3].plot(k, noiseless_outputs[:, 1], label="true $x_2$")
    axes[3].scatter(k[sensor2_mask], result.y_available[sensor2_mask, 1], marker="o", label="sensor 2 samples")
    axes[3].set_xlabel("sample $k$")
    axes[3].set_ylabel("measurement")
    axes[3].set_title("Sensor 2: available only at $k = 1, 6, 11, \\ldots$")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    figure.suptitle("Sensor fusion with non-concurrent measurements", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_gain_and_covariance(result: SimulationResult, output_path: Path) -> None:
    """Plot effective Kalman-gain and posterior covariance elements over time."""
    k = np.arange(result.x_true.shape[0])

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, constrained_layout=True)

    axes[0].plot(k, result.K_effective[:, 0, 0], label="$K_{11}$: state 1 from sensor 1")
    axes[0].plot(k, result.K_effective[:, 1, 0], label="$K_{21}$: state 2 from sensor 1")
    axes[0].plot(k, result.K_effective[:, 0, 1], label="$K_{12}$: state 1 from sensor 2")
    axes[0].plot(k, result.K_effective[:, 1, 1], label="$K_{22}$: state 2 from sensor 2")
    axes[0].set_ylabel("gain value")
    axes[0].set_title("Effective Kalman-gain elements (zero when a sensor is unavailable)")
    axes[0].legend(loc="upper right", ncols=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k, result.P_posterior[:, 0, 0], label="$P_{11}$")
    axes[1].plot(k, result.P_posterior[:, 0, 1], label="$P_{12}$")
    axes[1].plot(k, result.P_posterior[:, 1, 0], "--", label="$P_{21}$")
    axes[1].plot(k, result.P_posterior[:, 1, 1], label="$P_{22}$")
    axes[1].set_xlabel("sample $k$")
    axes[1].set_ylabel("covariance value")
    axes[1].set_title("Posterior estimation-error covariance $P_{k|k}$")
    axes[1].legend(loc="upper right", ncols=4)
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("Kalman gain and covariance for asynchronous sensor fusion", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=121, help="Number of simulated samples.")
    parser.add_argument("--seed", type=int, default=7, help="Random-number seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory in which plots and metrics are written.",
    )
    parser.add_argument("--show", action="store_true", help="Display generated figures after saving them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    result = simulate_and_filter(model=model, num_steps=args.num_steps, seed=args.seed)
    metrics = compute_metrics(result)

    state_plot = args.output_dir / "state_estimation.png"
    gain_plot = args.output_dir / "gain_and_covariance.png"
    metrics_file = args.output_dir / "metrics.json"

    plot_states_and_measurements(model=model, result=result, output_path=state_plot)
    plot_gain_and_covariance(result=result, output_path=gain_plot)
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("Simulation complete.")
    print(f"State and measurement figure: {state_plot}")
    print(f"Gain and covariance figure:   {gain_plot}")
    print(f"Metrics file:                 {metrics_file}")
    print(json.dumps(metrics, indent=2))

    if args.show:
        for image_path in (state_plot, gain_plot):
            image = plt.imread(image_path)
            plt.figure(figsize=(11, 8))
            plt.imshow(image)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
