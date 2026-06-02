import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Simulation settings
np.random.seed(4)

sampling_period_seconds       = 0.1
simulation_end_time_seconds   = 10.0
time_axis_seconds             = np.arange(0, simulation_end_time_seconds + sampling_period_seconds,
                                          sampling_period_seconds)
num_time_steps                = len(time_axis_seconds)

complementary_filter_time_constant = 0.5   # tau: sets crossover frequency between sensors

# Continuous-time state-space model
# State vector: x = [position y, velocity v]^T
#
#   dy/dt =  v
#   dv/dt = -0.015*v + 0.5*u
#
# Chosen dynamics: light damping (-0.015) and moderate input gain (0.5)
continuous_state_matrix = np.array([
    [0.0,  1.0],
    [0.0, -0.015]
])

continuous_input_matrix = np.array([
    [0.0],
    [0.5]
])

continuous_output_matrix    = np.eye(2)
continuous_feedthrough_matrix = np.zeros((2, 1))

# Discretize using zero-order hold: Phi = e^(A*Ts), Gamma = integral(e^(A*s)*B, 0, Ts)
discrete_state_transition_matrix, discrete_input_vector, _, _, _ = cont2discrete(
    (continuous_state_matrix, continuous_input_matrix,
     continuous_output_matrix, continuous_feedthrough_matrix),
    sampling_period_seconds)
discrete_input_vector = discrete_input_vector.ravel()   # flatten to 1-D for x_next = Phi@x + Gamma*u


def compute_input_at_time(time_value):
    # Smooth excitation signal: cos(t) + 0.1*sin(t/2)
    return np.cos(time_value) + 0.1 * np.sin(time_value / 2)


input_signal_sequence = compute_input_at_time(time_axis_seconds)

# Generate deterministic (noise-free) system response
# x[k+1] = Phi * x[k] + Gamma * u[k]
true_state_trajectory = np.zeros((num_time_steps, 2))

for time_index in range(num_time_steps - 1):
    true_state_trajectory[time_index + 1] = (
        discrete_state_transition_matrix @ true_state_trajectory[time_index]
        + discrete_input_vector * input_signal_sequence[time_index])

true_position_sequence = true_state_trajectory[:, 0]
true_velocity_sequence = true_state_trajectory[:, 1]

# Add independent Gaussian measurement noise
# Var(position noise) = 0.05  ->  sigma_y = sqrt(0.05) ≈ 0.224
# Var(velocity noise) = 0.10  ->  sigma_v = sqrt(0.10) ≈ 0.316
position_measurement_variance = 0.05
velocity_measurement_variance = 0.10

noisy_position_measurement = (true_position_sequence
                               + np.sqrt(position_measurement_variance) * np.random.randn(num_time_steps))
noisy_velocity_measurement = (true_velocity_sequence
                               + np.sqrt(velocity_measurement_variance) * np.random.randn(num_time_steps))

# 1. Complementary filter for position
#
# Derived from the bilinear (Tustin) transform of the ideal complementary transfer:
#   Y_cf(z) = (tau*s / (tau*s + 1)) * Y_sensor + (1 / (tau*s + 1)) * integral(V_sensor)
#
# With tau=0.5, Ts=0.1:
#   alpha1 = 2*tau/Ts + 1 = 2*0.5/0.1 + 1 = 11
#   alpha0 = 1 - 2*tau/Ts = 1 - 10 = -9
#
# Recurrence:
#   y_cf[k] = (-alpha0*y_cf[k-1] + tau*(v[k]+v[k-1]) + (y[k]+y[k-1])) / alpha1
complementary_filter_coeff_alpha1 = 2.0 * complementary_filter_time_constant / sampling_period_seconds + 1.0
complementary_filter_coeff_alpha0 = 1.0 - 2.0 * complementary_filter_time_constant / sampling_period_seconds

complementary_filter_position_estimate    = np.zeros(num_time_steps)
complementary_filter_position_estimate[0] = noisy_position_measurement[0]

for time_index in range(1, num_time_steps):
    complementary_filter_position_estimate[time_index] = (
        - complementary_filter_coeff_alpha0 * complementary_filter_position_estimate[time_index - 1]
        + complementary_filter_time_constant * (noisy_velocity_measurement[time_index]
                                                + noisy_velocity_measurement[time_index - 1])
        + (noisy_position_measurement[time_index] + noisy_position_measurement[time_index - 1])
    ) / complementary_filter_coeff_alpha1

# 2. Sensor-fused Kalman filter
#
# Both position and velocity are measured directly:
#   z[k] = [y_meas[k], v_meas[k]]^T = H * x[k] + noise
#   H = I (identity — full state measurement)
#
# Noise covariances:
#   R = diag(0.05, 0.10)   — from sensor specs
#   Q = diag(1e-6, 1e-5)   — small: model is trusted more than sensors
#
# Kalman equations (Joseph form for numerical stability):
#   Innovation:  ỹ = z - H*x̂
#   Innovation covariance: S = H*P*H' + R
#   Gain:        K = P*H'*inv(S)
#   Update:      x̂ = x̂ + K*ỹ
#   Covariance:  P = (I-KH)*P*(I-KH)' + K*R*K'
#   Predict:     x̂ = Phi*x̂ + Gamma*u,  P = Phi*P*Phi' + Q
measurement_matrix = np.eye(2)   # H: directly observe both position and velocity

measurement_noise_covariance = np.array([
    [position_measurement_variance, 0.0],
    [0.0,                           velocity_measurement_variance]
])

process_noise_covariance = np.array([
    [1e-6, 0.0],
    [0.0,  1e-5]
])

state_estimate_vector  = np.zeros(2)
state_error_covariance = np.eye(2)
identity_matrix        = np.eye(2)

kalman_filter_state_history = np.zeros((num_time_steps, 2))

for time_index in range(num_time_steps):
    stacked_measurement_vector = np.array([noisy_position_measurement[time_index],
                                           noisy_velocity_measurement[time_index]])

    # Measurement update
    innovation_vector          = stacked_measurement_vector - measurement_matrix @ state_estimate_vector
    innovation_covariance_matrix = (measurement_matrix @ state_error_covariance @ measurement_matrix.T
                                    + measurement_noise_covariance)
    kalman_gain_matrix         = (state_error_covariance @ measurement_matrix.T
                                  @ np.linalg.inv(innovation_covariance_matrix))

    state_estimate_vector = state_estimate_vector + kalman_gain_matrix @ innovation_vector

    # Joseph form: (I-KH)*P*(I-KH)' + K*R*K'  — numerically safer than P=(I-KH)*P
    innovation_weighting_matrix = identity_matrix - kalman_gain_matrix @ measurement_matrix
    state_error_covariance      = (innovation_weighting_matrix @ state_error_covariance
                                   @ innovation_weighting_matrix.T
                                   + kalman_gain_matrix @ measurement_noise_covariance @ kalman_gain_matrix.T)

    kalman_filter_state_history[time_index] = state_estimate_vector

    # Time update (predict)
    if time_index < num_time_steps - 1:
        state_estimate_vector  = (discrete_state_transition_matrix @ state_estimate_vector
                                  + discrete_input_vector * input_signal_sequence[time_index])
        state_error_covariance = (discrete_state_transition_matrix @ state_error_covariance
                                  @ discrete_state_transition_matrix.T + process_noise_covariance)

kalman_filter_position_estimate = kalman_filter_state_history[:, 0]
kalman_filter_velocity_estimate = kalman_filter_state_history[:, 1]

# Plot and save results
fig, axs = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle('Lecture 11 — Complementary Filter vs Kalman Filter', fontsize=13)

axs[0].plot(time_axis_seconds, true_position_sequence,              'r-',  label='True position')
axs[0].plot(time_axis_seconds, noisy_position_measurement,          'b.',  label='Noisy position measurement')
axs[0].plot(time_axis_seconds, complementary_filter_position_estimate, 'k-',
            linewidth=2, label='Complementary filter position')
axs[0].plot(time_axis_seconds, kalman_filter_position_estimate,     'g--', linewidth=2,
            label='Kalman filter position')
axs[0].set_ylabel('Position y [m]')
axs[0].set_xlabel('Time [s]')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time_axis_seconds, true_velocity_sequence,     'r-',  label='True velocity')
axs[1].plot(time_axis_seconds, noisy_velocity_measurement, 'b.',  label='Noisy velocity measurement')
axs[1].plot(time_axis_seconds, kalman_filter_velocity_estimate, 'g--', linewidth=2,
            label='Kalman filter velocity')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Velocity v [m/s]')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'lecture11_cf_vs_kf.png'), dpi=150)
plt.show()
