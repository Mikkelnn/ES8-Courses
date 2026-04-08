import numpy as np
from scipy import linalg as sclin
import matplotlib.pyplot as plt

lambdaX = 0.001
lambdaY = 0.002
muX = 0.05
muY = muX




def plot_probabilities(res):
    """Plot state probabilities over time with different colors for each state."""
    res_array = np.array(res)
    time = np.arange(len(res))
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['State 0', 'State 1', 'State 2', 'State 3']
    
    plt.figure(figsize=(10, 6))
    for state in range(4):
        plt.plot(time, res_array[:, state], color=colors[state], label=labels[state], linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('State Probabilities Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    Q = np.array([[-lambdaX-lambdaY, lambdaY, lambdaX, 0],
                  [muY, -lambdaX-muY, 0, lambdaX],
                  [muX, 0, -muX-lambdaY, lambdaY],
                  [0, muX, muY, -muX-muY]])
    P0 = np.array([1, 0, 0, 0])
    
    res = [P0]
    for t in range(1,200):
        res.append(P0 @ sclin.expm(Q*t))
        # print(f"P({t})= {res[t]}")
    
    plot_probabilities(res)



    

if __name__ == "__main__":
    main()