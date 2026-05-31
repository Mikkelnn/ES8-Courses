import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Discrete-Time Kalman Filter with Multi-Rate Sensor Fusion
# ============================================================

np.random.seed(7)

# ------------------------------------------------------------
# System Definition
# ------------------------------------------------------------

A = np.array([
    [0.7360, 0.2560],
    [0.3200, 0.6480]
])

C = np.array([
    [0.50, 0.50],
    [0.00, 1.00]
])

Q = 0.4 * np.eye(2)
R = 0.2 * np.eye(2)

# Square roots for simulation noise generation
Q_sqrt = np.linalg.cholesky(Q)
R_sqrt = np.linalg.cholesky(R)

# ------------------------------------------------------------
# Simulation Parameters
# ------------------------------------------------------------

N = 100                      # number of time steps
nx = 2                      # number of states

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

x_true = np.zeros((nx, N))
x_hat  = np.zeros((nx, N))

P_store = np.zeros((nx, nx, N))
K_store = np.zeros((nx, 2, N))   # store full 2-column gain

y_meas = np.full((2, N), np.nan)

# ------------------------------------------------------------
# Initial Conditions
# ------------------------------------------------------------

x_true[:, 0] = np.array([2.0, -1.0])

x_hat[:, 0] = np.array([0.0, 0.0])

P = 5 * np.eye(nx)

# ============================================================
# MAIN LOOP
# ============================================================

for k in range(N - 1):

    # --------------------------------------------------------
    # TRUE SYSTEM
    # x(k+1) = A x(k) + w(k)
    # --------------------------------------------------------

    w = Q_sqrt @ np.random.randn(2)

    x_true[:, k + 1] = A @ x_true[:, k] + w

    # --------------------------------------------------------
    # SENSOR MEASUREMENTS
    # --------------------------------------------------------

    # Sensor 1 available at k = 0,5,10,...
    sensor1_available = (k % 5 == 0)

    # Sensor 2 available at k = 1,6,11,...
    sensor2_available = ((k - 1) % 5 == 0)

    v = R_sqrt @ np.random.randn(2)

    y_full = C @ x_true[:, k] + v

    # Save only available measurements
    if sensor1_available:
        y_meas[0, k] = y_full[0]

    if sensor2_available:
        y_meas[1, k] = y_full[1]

    # --------------------------------------------------------
    # PREDICTION STEP
    # --------------------------------------------------------

    x_pred = A @ x_hat[:, k]
    P_pred = A @ P @ A.T + Q

    # --------------------------------------------------------
    # UPDATE STEP (Sensor Fusion)
    # --------------------------------------------------------

    # Determine active measurements
    H_list = []
    z_list = []
    R_list = []

    if sensor1_available:
        H_list.append(C[0:1, :])
        z_list.append([y_full[0]])
        R_list.append([R[0, 0]])

    if sensor2_available:
        H_list.append(C[1:2, :])
        z_list.append([y_full[1]])
        R_list.append([R[1, 1]])

    # If measurements exist
    if len(H_list) > 0:

        H = np.vstack(H_list)
        z = np.array(z_list).reshape(-1, 1)
        Rk = np.diag(np.array(R_list).flatten())

        x_pred_col = x_pred.reshape(-1, 1)

        innovation = z - H @ x_pred_col

        S = H @ P_pred @ H.T + Rk

        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred_col + K @ innovation

        P = (np.eye(nx) - K @ H) @ P_pred

        x_hat[:, k + 1] = x_upd.flatten()

        # Store gain in consistent 2-column format
        if sensor1_available and sensor2_available:
            K_store[:, :, k] = K

        elif sensor1_available:
            K_store[:, 0, k] = K.flatten()

        elif sensor2_available:
            K_store[:, 1, k] = K.flatten()

    else:
        # No measurements
        x_hat[:, k + 1] = x_pred
        P = P_pred

    P_store[:, :, k] = P

# ============================================================
# PLOTS
# ============================================================

time = np.arange(N)

# ------------------------------------------------------------
# Figure 1: States and Estimates
# ------------------------------------------------------------

fig1, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ------------------------------------------------------------
# Average state
# ------------------------------------------------------------

x_true_avg = 0.5 * (x_true[0, :] + x_true[1, :])
x_hat_avg  = 0.5 * (x_hat[0, :] + x_hat[1, :])

# Optional averaged measurements
y_avg = np.full(N, np.nan)

for k in range(N):

    vals = []

    if not np.isnan(y_meas[0, k]):
        vals.append(y_meas[0, k])

    if not np.isnan(y_meas[1, k]):
        vals.append(y_meas[1, k])

    if len(vals) > 0:
        y_avg[k] = np.mean(vals)

# ----- x1 -----
axs[0].plot(time, x_true[0, :], 'k', linewidth=2, label='True $x_1$')
axs[0].plot(time, x_hat[0, :], 'b--', linewidth=2, label='KF Estimate')

axs[0].scatter(time, y_meas[0, :],
               color='red',
               marker='o',
               label='Sensor 1')

axs[0].set_ylabel('$x_1$')
axs[0].grid(True)
axs[0].legend()

# ----- x2 -----
axs[1].plot(time, x_true[1, :], 'k', linewidth=2, label='True $x_2$')
axs[1].plot(time, x_hat[1, :], 'b--', linewidth=2, label='KF Estimate')

axs[1].scatter(time, y_meas[1, :],
               color='green',
               marker='s',
               label='Sensor 2')

axs[1].set_ylabel('$x_2$')
axs[1].grid(True)
axs[1].legend()

# ----- Average state -----
axs[2].plot(time, x_true_avg,
            'k',
            linewidth=2,
            label='True Average State')

axs[2].plot(time, x_hat_avg,
            'm--',
            linewidth=2,
            label='Estimated Average State')

axs[2].scatter(time,
               y_avg,
               color='orange',
               marker='d',
               label='Average Measurement')

axs[2].set_ylabel('Average')
axs[2].set_xlabel('Time Step k')
axs[2].grid(True)
axs[2].legend()

fig1.suptitle('True States, Measurements, and Kalman Filter Estimates')

# ------------------------------------------------------------
# Figure 2: Kalman Gain Elements
# ------------------------------------------------------------

fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

labels = [['K11', 'K12'],
          ['K21', 'K22']]

for i in range(2):
    for j in range(2):
        axs2[i, j].plot(time, K_store[i, j, :], linewidth=2)
        axs2[i, j].set_title(labels[i][j])
        axs2[i, j].grid(True)

axs2[1, 0].set_xlabel('Time Step k')
axs2[1, 1].set_xlabel('Time Step k')

fig2.suptitle('Kalman Gain Elements')

# ------------------------------------------------------------
# Figure 3: Covariance Matrix Elements
# ------------------------------------------------------------

fig3, axs3 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

P_labels = [['P11', 'P12'],
            ['P21', 'P22']]

for i in range(2):
    for j in range(2):
        axs3[i, j].plot(time, P_store[i, j, :], linewidth=2)
        axs3[i, j].set_title(P_labels[i][j])
        axs3[i, j].grid(True)

axs3[1, 0].set_xlabel('Time Step k')
axs3[1, 1].set_xlabel('Time Step k')

fig3.suptitle('Covariance Matrix Elements')

plt.tight_layout()
plt.show()

