import matplotlib.pyplot as plt
import numpy as np

# Implement a Kalman filter for a simple linear system
phi = 0.9
SD_a = 0.1
SD_b = 0.1
SD_v = 0.1
b = -0.2
Time_step = 0.001

x, v, a = 0, 0, 0 # Initial state of the system

phi_matrix = np.array(([1 , Time_step, 0, 0],
                        [0, 1, Time_step, 0], 
                        [0, 0, phi, 0],
                        [0, 0, 0, 1]))

state_matrix = np.array(([x], [v], [a], [b]), dtype=float) # state matrix of the system

H = np.array([[0, 0, 1, 1]]) # measurement matrix, shape (1,4)

# implement a Kalman filter
Q = np.diag([0, 0, SD_a**2, SD_b**2]) # process noise covariance
R = np.array([[SD_v**2]]) # measurement noise covariance
P = np.eye(4) # initial estimate error covariance


# make the system
def system_update(state_matrix):
    '''Are the w_a and w_b correlated?'''
    w_a = np.random.randn() # driving noise input for acceleration
    w_b = np.random.randn() # driving noise input for bias
    
    # update the state matrix via dynamics
    new_state_matrix = phi_matrix @ state_matrix
    
    # add process noise to acceleration and bias after propagation
    new_state_matrix[2] += SD_a * w_a
    new_state_matrix[3] += SD_b * w_b
    
    return new_state_matrix

def measurement_update(state_matrix):
    '''Is the v measurement noise correlated with the a and b noise?'''
    v = np.random.randn() # measurement noise
    z = H @ state_matrix + SD_v * v # measurement of acceleration + bias with noise
    
    return z

def kalman_filter(true_state, state_estimate, P):
    # Prediction step
    predicted_state = phi_matrix @ state_estimate
    predicted_P = phi_matrix @ P @ phi_matrix.T + Q
    
    # Measurement update step - measure the TRUE state, not the prediction
    z = measurement_update(true_state)
    y = z - H @ predicted_state # measurement residual
    S = H @ predicted_P @ H.T + R # residual covariance, shape (1,1)
    K = predicted_P @ H.T @ np.linalg.inv(S) # Kalman gain, shape (4,1)
    
    updated_state = predicted_state + K @ y
    updated_P = (np.eye(len(P)) - K @ H) @ predicted_P
    
    return updated_state, updated_P

def main():
    global state_matrix, P

    # Kalman filter implementation
    num_iterations = 1000

    filtered_states = []
    true_states = []

    state_estimate = state_matrix.copy() # separate estimate tracked by the filter

    for i in range(num_iterations):
        new_state_matrix = system_update(state_matrix) # propagate the true state

        # filter takes the true measurement and the previous estimate
        filtered_state_matrix, P = kalman_filter(new_state_matrix, state_estimate, P)

        filtered_states.append(filtered_state_matrix)
        true_states.append(new_state_matrix)

        state_matrix = new_state_matrix          # advance the true state
        state_estimate = filtered_state_matrix   # advance the filter estimate

    # plot the results
    true_accelerations = [state[2, 0] for state in true_states]
    estimated_accelerations = [state[2, 0] for state in filtered_states]
    plt.plot(true_accelerations, label='True Acceleration')
    plt.plot(estimated_accelerations, label='Estimated Acceleration')
    plt.legend()
    plt.title('True vs Estimated Acceleration')
    plt.savefig('acceleration.png')
    plt.clf()

    true_biases = [state[3, 0] for state in true_states]
    estimated_biases = [state[3, 0] for state in filtered_states]
    plt.plot(true_biases, label='True Bias')
    plt.plot(estimated_biases, label='Estimated Bias')
    plt.legend()
    plt.title('True vs Estimated Bias')
    plt.savefig('bias.png')
    plt.clf()


if __name__ == "__main__":
    main()
