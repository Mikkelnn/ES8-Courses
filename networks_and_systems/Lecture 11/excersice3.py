"""
Exercise 3: HMM-based detection system for network congestion.

This exercise applies HMM filtering to detect network faults in a 3-state
birth-death Markov chain model. Compares two detection strategies:
1. Threshold-based detector: predicts fault if observation bin >= threshold
2. HMM-based detector: predicts fault if P(fault_state | observations) >= threshold

The HMM states represent RTT quality:
- State 0: Good condition (low RTT)
- State 1: Degraded condition (medium RTT)
- State 2: Fault condition (high RTT)

Analysis includes ROC curves and filtered posterior probabilities.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    stationary_distribution,
    simulate_hmm,
    forward,
    confusion_rates,
    threshold_detector,
    roc_from_thresholds,
    hmm_detector_point,
    setup_plot_theme,
)


# ============================================================
# HMM / congestion model definition
# ============================================================

def build_transition_matrix(
    p_up: float = 0.2,
    p_down: float = 0.2,
    p_stay: float = 0.6
) -> np.ndarray:
    """
    Build transition matrix for 3-state birth-death Markov chain.

    States {0, 1, 2} represent quality degradation levels.
    - State 0: Good condition
    - State 1: Degraded
    - State 2: Fault (bad)

    Transition probabilities:
    - Interior (state 1): down→0, stay→1, up→2 with given probabilities
    - Boundary states: redistribute impossible transitions to stay

    Parameters
    ----------
    p_up : float, default 0.2
        Probability of degradation (move to worse state).
    p_down : float, default 0.2
        Probability of improvement (move to better state).
    p_stay : float, default 0.6
        Probability of remaining in same state.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Row-stochastic transition matrix P[i, j] = P(X_t=j | X_{t-1}=i).

    Notes
    -----
    Satisfies: each row sums to 1.0
    """
    P = np.array([
        [p_stay + p_down, p_up,           0.0],   # state 0: can't go down
        [p_down,          p_stay,         p_up],  # state 1: interior
        [0.0,             p_down,         p_stay + p_up]  # state 2: can't go up
    ], dtype=float)

    return P


def build_emission_matrix() -> np.ndarray:
    """
    Build emission matrix for discretized RTT observations.

    RTT is discretized into 7 bins (0-6), each representing 100 ms interval:
    - Bin 0: [0, 100) ms
    - Bin 1: [100, 200) ms
    - ...
    - Bin 6: [600, 700) ms

    Emission probabilities B[i, k] = P(observation=k | hidden_state=i).

    Returns
    -------
    np.ndarray, shape (3, 7)
        Row-stochastic emission matrix.
        - State 0 (good): concentrated at low bins
        - State 1 (degraded): spread across middle bins
        - State 2 (fault): concentrated at high bins
    """
    B = np.array([
        [0.2, 0.6, 0.1, 0.05, 0.05, 0.0, 0.0],   # State 0 (good)
        [0.1, 0.2, 0.3, 0.2,  0.1,  0.05, 0.05], # State 1 (degraded)
        [0.0, 0.1, 0.1, 0.1,  0.3,  0.3,  0.1],  # State 2 (fault)
    ], dtype=float)

    return B


def bin_to_rtt_ms(bin_idx: int) -> str:
    """
    Convert bin index to RTT interval label.

    Parameters
    ----------
    bin_idx : int
        Bin index in {0, ..., 6}.

    Returns
    -------
    str
        Interval label, e.g., '[0,100) ms' for bin 0.
    """
    lo = 100 * bin_idx
    hi = 100 * (bin_idx + 1)
    return f"[{lo},{hi}) ms"



def main() -> None:
    """
    Main analysis: threshold-based vs HMM-based fault detection.

    Simulates 50,000 samples from 3-state HMM and compares two detectors:
    1. Threshold-based: predict fault if observation >= threshold
    2. HMM-based: predict fault if P(fault|obs) >= decision_threshold

    Generates ROC curve and prints confusion metrics for both approaches.
    """
    setup_plot_theme(figsize=(8, 6))

    rng = np.random.default_rng(7)

    # Model
    P = build_transition_matrix()
    B = build_emission_matrix()
    pi = stationary_distribution(P)

    print("Transition matrix P:")
    print(P)
    print("\nEmission matrix B:")
    print(B)
    print("\nStationary distribution pi:")
    print(pi)

    # Simulate a sample sequence
    T = 50000
    states, obs = simulate_hmm(T, P, B, pi=pi, rng=rng)

    # True label: state 2 is fault state
    true_fault = (states == 2).astype(int)

    # --------------------------------------------------------
    # Threshold ROC
    # --------------------------------------------------------
    roc_points = roc_from_thresholds(obs, true_fault, thresholds=np.arange(0, 8))

    print("\nThreshold-detector ROC points:")
    for fpr, tpr, th in roc_points:
        if th <= 6:
            th_label = f"predict fault if bin >= {th} ({bin_to_rtt_ms(th)})"
        else:
            th_label = "predict fault never"
        print(f"threshold={th}: FPR={fpr:.4f}, TPR={tpr:.4f}   {th_label}")

    # --------------------------------------------------------
    # HMM-based detector point
    # --------------------------------------------------------
    hmm_tpr, hmm_fpr, fault_posterior, hmm_pred = hmm_detector_point(
        obs, true_fault, P, B, pi=pi, decision_threshold=0.5
    )

    print("\nHMM-based detector point:")
    print(f"FPR = {hmm_fpr:.4f}")
    print(f"TPR = {hmm_tpr:.4f}")

    # --------------------------------------------------------
    # Plot ROC + HMM point
    # --------------------------------------------------------
    fprs = [p[0] for p in roc_points]
    tprs = [p[1] for p in roc_points]
    ths = [p[2] for p in roc_points]

    plt.figure()
    plt.plot(fprs, tprs, marker='o', label="Threshold detector ROC")

    for fpr, tpr, th in roc_points:
        plt.annotate(str(th), (fpr, tpr), textcoords="offset points", xytext=(5, 5))

    plt.scatter([hmm_fpr], [hmm_tpr], s=120, marker='x', label="HMM detector")
    plt.plot([0, 1], [0, 1], linestyle='--', alpha=0.6)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve: threshold detector vs HMM-based detector")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Optional: inspect first few samples
    # --------------------------------------------------------
    print("\nFirst 20 samples:")
    print("t | state | obs_bin | obs_interval | true_fault | posterior_fault")
    for t in range(20):
        print(
            f"{t:2d} | "
            f"{states[t] + 1:5d} | "
            f"{obs[t]:7d} | "
            f"{bin_to_rtt_ms(obs[t]):>12s} | "
            f"{true_fault[t]:10d} | "
            f"{fault_posterior[t]:15.4f}"
        )


if __name__ == "__main__":
    main()