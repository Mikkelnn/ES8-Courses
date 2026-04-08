"""
Exercise 2: Hidden Markov Model fundamentals.

This exercise explores core HMM algorithms:
- (a) Sequence generation from an HMM
- (b) Forward algorithm: computing likelihood P(observations)
- (c) Viterbi algorithm: finding most likely state path

Models a coin flipping scenario with 3 hidden states and binary observations (H/T).

Parameters
----------
π = [0.2, 0.3, 0.5]         Initial state distribution
P                            Transition matrix (3×3)
B                            Emission matrix (3×2, observations H/T)
Test sequence: S = 'HHTHTTTHT'
"""

from typing import Tuple, Optional
import numpy as np
from utils import sample_categorical, simulate_hmm, forward, viterbi

pi = np.array([0.2, 0.3, 0.5], dtype=float)

P = np.array([
    [0.2, 0.4, 0.4],
    [0.5, 0.1, 0.4],
    [0.2, 0.2, 0.6]
], dtype=float)

# Emission matrix B:
# rows = states 1..3
# cols = observations [H, T]
B = np.array([
    [0.8, 0.2],  # state 1
    [0.5, 0.5],  # state 2
    [0.1, 0.9]   # state 3
], dtype=float)

obs_map = {'H': 0, 'T': 1}
inv_obs_map = {0: 'H', 1: 'T'}


def encode_sequence(seq_str: str) -> np.ndarray:
    """
    Encode observation sequence from string to indices.

    Parameters
    ----------
    seq_str : str
        String of observations, e.g., 'HHTHT' for heads/tails.

    Returns
    -------
    np.ndarray, shape (len(seq_str),)
        Integer indices corresponding to obs_map.

    Examples
    --------
    >>> encode_sequence('HT')
    array([0, 1])
    """
    return np.array([obs_map[ch] for ch in seq_str], dtype=int)


def decode_observations(obs: np.ndarray) -> str:
    """
    Decode observation indices to string representation.

    Parameters
    ----------
    obs : np.ndarray
        Observation indices {0, 1}.

    Returns
    -------
    str
        Decoded observation sequence, e.g., 'HHTHT'.

    Examples
    --------
    >>> decode_observations(np.array([0, 1, 0]))
    'HTO'
    """
    return ''.join(inv_obs_map[o] for o in obs)


def simulation_estimate(
    target_obs: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    n_sim: int = 200000,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Estimate P(target_obs) by Monte Carlo simulation.

    Generates n_sim HMM sequences and counts how many match target observation sequence.

    Estimate: P̂(obs) = (# matches) / n_sim

    Parameters
    ----------
    target_obs : np.ndarray
        Target observation sequence (indices).
    pi : np.ndarray
        Initial state distribution.
    P : np.ndarray
        Transition matrix.
    B : np.ndarray
        Emission matrix.
    n_sim : int, default 200000
        Number of simulation trials.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Estimated probability P̂(obs).

    Notes
    -----
    Convergence: error ~ O(1/√n_sim) by law of large numbers.
    Only useful for short, more likely sequences; rare sequences may have zero matches.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(target_obs)
    matches = 0

    for _ in range(n_sim):
        _, obs = simulate_hmm(T, P, B, pi=pi, rng=rng)
        if np.array_equal(obs, target_obs):
            matches += 1

    return matches / n_sim


def main() -> None:
    """
    Main analysis for Exercise 2: HMM fundamentals.

    (a) Generate example sequence from HMM
    (b) Compute likelihood using forward algorithm
    (c) Estimate likelihood via Monte Carlo simulation
    (d) Find most likely state path using Viterbi algorithm
    """
    S = "HHTHTTTHT"
    obs = encode_sequence(S)

    print("Observation sequence:", S)

    # (a) Example generation
    rng = np.random.default_rng(42)
    states_gen, obs_gen = simulate_hmm(len(S), P, B, pi=pi, rng=rng)
    print("\nGenerated example:")
    print("States (1-based):", states_gen + 1)
    print("Observations:", decode_observations(obs_gen))

    # (b) Forward probability
    prob, alpha = forward(obs, pi, P, B, normalized=False)
    print("\nForward algorithm:")
    print(f"P({S}) = {prob:.10f}")

    # Simulation estimate
    est = simulation_estimate(obs, pi, P, B, n_sim=200000, rng=rng)
    print("\nSimulation estimate:")
    print(f"Estimated P({S}) ≈ {est:.10f}")

    # (c) Viterbi path
    best_path, best_prob, delta, psi = viterbi(obs, pi, P, B)
    print("\nViterbi:")
    print("Most likely states (1-based):", best_path + 1)
    print(f"Best path probability = {best_prob:.10f}")


if __name__ == "__main__":
    main()