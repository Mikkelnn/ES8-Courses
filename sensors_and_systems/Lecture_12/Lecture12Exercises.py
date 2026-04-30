import numpy as np
import matplotlib.pyplot as plt

#Just got chat to cook something

# -----------------------------
# 1) SYSTEM SETUP
# -----------------------------
Ts = 0.01
T = 10
t = np.arange(0, T, Ts)
N = len(t)

# Input
u = np.cos(t) + 0.1*np.sin(t/2)

# Continuous system
A = np.array([[0, 1],
              [0, -0.015]])
B = np.array([[0],
              [0.5]])

# Discretization (Euler)
Ad = np.eye(2) + Ts*A
Bd = Ts*B

# -----------------------------
# TRUE SYSTEM SIMULATION
# -----------------------------
x = np.zeros((2, N))
for k in range(N-1):
    x[:,k+1] = x[:,k] + Ts*(A @ x[:,k] + B.flatten()*u[k])

y_true = x[0]
v_true = x[1]

# -----------------------------
# ADD MEASUREMENT NOISE
# -----------------------------
np.random.seed(0)

y_meas = y_true + np.random.normal(0, np.sqrt(0.05), N)
v_meas = v_true + np.random.normal(0, np.sqrt(0.1), N)

# =========================================================
# 1) COMPLEMENTARY FILTER
# =========================================================
tau = 0.5

alpha1 = tau*Ts/2 + 1
alpha0 = 1 - tau*Ts/2
beta = alpha1 - 1

y_comp = np.zeros(N)

for k in range(1, N):
    y_comp[k] = (1/alpha1)*(
        -alpha0*y_comp[k-1]
        + beta*(Ts/2)*v_meas[k]
        + beta*(Ts/2)*v_meas[k-1]
        + y_meas[k] + y_meas[k-1]
    )

# =========================================================
# 2) KALMAN FILTER
# =========================================================
H = np.eye(2)
R = np.diag([0.05, 0.1])
Q = 1e-4 * np.eye(2)

x_hat = np.zeros((2, N))
P = np.eye(2)

for k in range(N-1):
    # Prediction
    x_pred = Ad @ x_hat[:,k] + Bd.flatten()*u[k]
    P_pred = Ad @ P @ Ad.T + Q

    # Measurement
    yk = np.array([y_meas[k], v_meas[k]])

    # Gain
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)

    # Update
    x_hat[:,k+1] = x_pred + K @ (yk - H @ x_pred)
    P = (np.eye(2) - K @ H) @ P_pred

y_kalman = x_hat[0]
v_kalman = x_hat[1]

# =========================================================
# 3) VERIFY DISCRETE-TIME EXPRESSIONS (TUSTIN)
# =========================================================

# Integrate velocity using Tustin (should resemble position)
y_tustin = np.zeros(N)
for k in range(1, N):
    y_tustin[k] = y_tustin[k-1] + (Ts/2)*(v_meas[k] + v_meas[k-1])

# =========================================================
# PLOTS (MATCH SLIDE 14 STYLE)
# =========================================================

plt.figure(figsize=(12,5))

# --- POSITION ---
plt.subplot(1,2,1)
plt.plot(t, y_true, 'r', linewidth=2, label='True')
plt.plot(t, y_meas, 'b', alpha=0.4, label='Measured')

plt.plot(t, y_comp, 'y--', alpha=0.5, linewidth=1.2, label='Complementary')
plt.plot(t[::10], y_comp[::10], 'yo', markersize=3)

plt.plot(t, y_kalman, 'g--', linewidth=1.5, label='Kalman')

plt.title("Position")
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.legend()
plt.grid()

# --- VELOCITY ---
plt.subplot(1,2,2)
plt.plot(t, v_true, 'r', linewidth=2, label='True')
plt.plot(t, v_meas, 'b', alpha=0.4, label='Measured')
plt.plot(t, v_kalman, 'g--', linewidth=1.5, label='Kalman')

plt.title("Velocity")
plt.xlabel("Time [s]")
plt.ylabel("v")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()