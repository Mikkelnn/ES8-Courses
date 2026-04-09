"""
Utilities for Hidden Markov Model exercises.

This module consolidates common HMM utilities, detection metrics, and plotting
configurations for improved code reuse and academic quality across exercises.

Functions
---------
sample_categorical()
    Sample from a categorical distribution.
stationary_distribution()
    Compute stationary distribution of a Markov chain.
confusion_rates()
    Compute TPR and FPR from true/predicted labels.
setup_plot_theme()
    Configure matplotlib for consistent academic styling.
simulate_hmm()
    Generate observation and state sequences from HMM.
forward()
    Forward algorithm for HMM: compute likelihood or filtered posteriors.
viterbi()
    Viterbi algorithm: find most likely state sequence.
threshold_detector()
    Binary classification via threshold on observations.
roc_from_thresholds()
    Compute ROC points for threshold-based detector.
hmm_detector_point()
    Evaluate HMM-based detector: predict fault via posterior threshold.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Sampling and probability utilities
# ============================================================

def sample_categorical(probs: np.ndarray, rng: Optional[np.random.Generator] = None) -> int:
    """
    Sample a category from a categorical distribution.

    Parameters
    ----------
    probs : np.ndarray, shape (K,)
        Probability mass function. Must sum to approximately 1.0.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default (not recommended for reproducibility).

    Returns
    -------
    int
        Sampled category index in {0, 1, ..., K-1}.

    Notes
    -----
    Uses numpy's discrete choice with replacement.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(len(probs), p=probs)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a finite-state Markov chain.

    Solves the eigenvalue equation:
        π P = π    (row vector convention)
        π · 1 = 1  (normalization)

    Equivalently, π is the left eigenvector of P with eigenvalue 1.

    Parameters
    ----------
    P : np.ndarray, shape (n, n)
        Row-stochastic transition matrix where P[i,j] = P(X_{t+1}=j | X_t=i).

    Returns
    -------
    np.ndarray, shape (n,)
        Stationary distribution π ≥ 0, normalized to sum to 1.

    Notes
    -----
    For irreducible, aperiodic Markov chains, this distribution is unique and
    the chain converges to it from any initial distribution.
    """
    # Solve π = π P, equivalent to π (P^T - I) = 0
    # Find left eigenvector (π P = λ π) with λ ≈ 1
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = np.maximum(pi, 0)  # Ensure non-negative
    pi = pi / np.sum(pi)     # Normalize
    return pi


# ============================================================
# Evaluation metrics
# ============================================================

def confusion_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Compute True Positive Rate (sensitivity) and False Positive Rate (1-specificity).

    These are standard metrics for evaluating binary classifiers.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Ground truth binary labels: 1 if positive class, 0 if negative class.
    y_pred : np.ndarray, shape (n,)
        Predicted binary labels: 1 if predicted positive, 0 if predicted negative.

    Returns
    -------
    tpr : float
        True Positive Rate = TP / (TP + FN)
        Probability of predicting positive given true positive.
    fpr : float
        False Positive Rate = FP / (FP + TN)
        Probability of predicting positive given true negative.

    Notes
    -----
    Formulas:
        TP = Σ 1{y_pred[i]=1 ∧ y_true[i]=1}
        FP = Σ 1{y_pred[i]=1 ∧ y_true[i]=0}
        FN = Σ 1{y_pred[i]=0 ∧ y_true[i]=1}
        TN = Σ 1{y_pred[i]=0 ∧ y_true[i]=0}
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

    Returns (0, 0) if no positive or negative samples exist.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pos = (y_true == 1)
    neg = (y_true == 0)

    tp = np.sum((y_pred == 1) & pos)
    fn = np.sum((y_pred == 0) & pos)
    fp = np.sum((y_pred == 1) & neg)
    tn = np.sum((y_pred == 0) & neg)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr


# ============================================================
# Plotting configuration
# ============================================================

def setup_plot_theme(
    figsize: Tuple[float, float] = (8, 6),
    grid: bool = True,
    font_size: Optional[int] = None
) -> None:
    """
    Configure matplotlib for consistent academic plotting style.

    Applies settings globally to all subsequent figures created in this session.

    Parameters
    ----------
    figsize : tuple[float, float], default (8, 6)
        Default figure size (width, height) in inches.
    grid : bool, default True
        Enable grid on all plots.
    font_size : int, optional
        Font size in points. If None, uses matplotlib default (10).

    Notes
    -----
    Affects matplotlib.pyplot global rcParams. Used at start of main() in each exercise.

    Examples
    --------
    >>> setup_plot_theme(figsize=(8, 5), grid=True, font_size=10)
    >>> plt.figure()
    >>> plt.plot([1, 2, 3], [1, 2, 3])
    >>> plt.show()  # Figure uses configured style
    """
    plt.rcParams['figure.figsize'] = figsize
    if font_size is not None:
        plt.rcParams['font.size'] = font_size
    if grid:
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3


# ============================================================
# HMM core algorithms
# ============================================================

def simulate_hmm(
    T: int,
    P: np.ndarray,
    B: np.ndarray,
    pi: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a sequence of hidden states and observations from an HMM.

    Procedure:
        1. Sample initial state: X_1 ~ Categorical(π)
        2. For t = 2, ..., T:
           - Sample transition: X_t ~ Categorical(P[X_{t-1}, :])
           - Sample observation: Y_t ~ Categorical(B[X_t, :])

    Parameters
    ----------
    T : int
        Length of sequence.
    P : np.ndarray, shape (n_states, n_states)
        Transition matrix: P[i, j] = P(X_t=j | X_{t-1}=i).
    B : np.ndarray, shape (n_states, n_obs)
        Emission matrix: B[i, k] = P(Y_t=k | X_t=i).
    pi : np.ndarray, shape (n_states,), optional
        Initial state distribution. If None, uses stationary distribution of P.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    states : np.ndarray, shape (T,)
        Hidden state sequence, indices in {0, ..., n_states-1}.
    obs : np.ndarray, shape (T,)
        Observation sequence, indices in {0, ..., n_obs-1}.

    Examples
    --------
    >>> P = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> B = np.array([[0.8, 0.2], [0.3, 0.7]])
    >>> states, obs = simulate_hmm(100, P, B, pi=np.array([0.5, 0.5]))
    >>> states.shape, obs.shape
    ((100,), (100,))
    """
    if rng is None:
        rng = np.random.default_rng()

    n_states = P.shape[0]

    if pi is None:
        pi = stationary_distribution(P)

    states = np.zeros(T, dtype=int)
    obs = np.zeros(T, dtype=int)

    # Initial state
    states[0] = sample_categorical(pi, rng)
    obs[0] = sample_categorical(B[states[0]], rng)

    # Subsequent states and observations
    for t in range(1, T):
        states[t] = sample_categorical(P[states[t - 1]], rng)
        obs[t] = sample_categorical(B[states[t]], rng)

    return states, obs


def forward(
    obs: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    normalized: bool = False
) -> Tuple[Optional[float], np.ndarray]:
    """
    Forward algorithm for Hidden Markov Models.

    Computes forward variables α_t(j) using dynamic programming:

    **Unnormalized** (normalized=False):
        α_t(j) = b_j(y_t) · Σ_i α_{t-1}(i) P_{ij}

        Returns: P(Y_1:T), unnormalized forward variables

    **Normalized** (normalized=True):
        α_t(j) = [b_j(y_t) · Σ_i α_{t-1}(i) P_{ij}] / c_t

        where c_t = Σ_j [b_j(y_t) · Σ_i α_{t-1}(i) P_{ij}]

        Returns: filtered posterior distribution P(X_t | Y_1:t), normalized alphas

    Parameters
    ----------
    obs : np.ndarray, shape (T,)
        Observation sequence (indices).
    pi : np.ndarray, shape (n_states,)
        Initial state distribution.
    P : np.ndarray, shape (n_states, n_states)
        Transition matrix: P[i, j] = P(X_t=j | X_{t-1}=i).
    B : np.ndarray, shape (n_states, n_obs)
        Emission matrix: B[i, k] = P(Y_t=k | X_t=i).
    normalized : bool, default False
        If False: return (likelihood, unnormalized_alphas).
        If True: return (None, normalized_alphas).

    Returns
    -------
    likelihood : float or None
        P(Y_1:T) if normalized=False, else None.
    alpha : np.ndarray, shape (T, n_states)
        Forward variables (unnormalized if normalized=False, normalized if True).

    Notes
    -----
    Unnormalized version: preferred for computing likelihood P(observations).
    Normalized version: preferred for filtering (computing posteriors at each time step).

    Both versions have O(T · n_states²) time complexity.

    References
    ----------
    Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proc. IEEE, 77(2), 257-286.
    """
    T = len(obs)
    n_states = len(pi)

    alpha = np.zeros((T, n_states), dtype=float)

    # Initialization: α_1(j) = π_j · b_j(y_1)
    alpha[0] = pi * B[:, obs[0]]

    if normalized:
        c = np.sum(alpha[0])
        if c > 0:
            alpha[0] /= c

    # Recursion: α_t(j) = b_j(y_t) · Σ_i α_{t-1}(i) P_{ij}
    for t in range(1, T):
        for j in range(n_states):
            alpha[t, j] = B[j, obs[t]] * np.sum(alpha[t - 1] * P[:, j])

        if normalized:
            c = np.sum(alpha[t])
            if c > 0:
                alpha[t] /= c
            else:
                alpha[t] = np.ones(n_states) / n_states

    # Termination: P(Y_1:T) = Σ_i α_T(i)
    if normalized:
        return None, alpha
    else:
        likelihood = np.sum(alpha[T - 1])
        return likelihood, alpha


def viterbi(
    obs: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Viterbi algorithm: find the most likely hidden state sequence given observations.

    Finds: X̂* = argmax_{x_1:T} P(x_1:T, y_1:T)

    Uses dynamic programming with backtracking:

        δ_t(j) = max_{x_1:t-1} P(x_1:t-1, X_t=j, y_1:t)
        ψ_t(j) = argmax_i δ_{t-1}(i) P_{ij}

    Parameters
    ----------
    obs : np.ndarray, shape (T,)
        Observation sequence (indices).
    pi : np.ndarray, shape (n_states,)
        Initial state distribution.
    P : np.ndarray, shape (n_states, n_states)
        Transition matrix: P[i, j] = P(X_t=j | X_{t-1}=i).
    B : np.ndarray, shape (n_states, n_obs)
        Emission matrix: B[i, k] = P(Y_t=k | X_t=i).

    Returns
    -------
    best_path : np.ndarray, shape (T,)
        Most likely state sequence (indices in {0, ..., n_states-1}).
    best_prob : float
        Probability of best path and observations: P(X*, Y).
    delta : np.ndarray, shape (T, n_states)
        Viterbi variable δ_t(j) = max probability of path to state j at time t.
    psi : np.ndarray, shape (T, n_states)
        Backpointer: ψ_t(j) = argmax predecessor state for state j at time t.

    Notes
    -----
    Time complexity: O(T · n_states²).
    Space complexity: O(T · n_states).

    References
    ----------
    Viterbi, A. (1967). Error bounds for convolutional codes and an
    asymptotically optimum decoding algorithm. IEEE Trans. Inf. Theory, 13(2), 260-269.
    """
    T = len(obs)
    n_states = len(pi)

    delta = np.zeros((T, n_states), dtype=float)
    psi = np.zeros((T, n_states), dtype=int)

    # Initialization: δ_1(j) = π_j · b_j(y_1)
    delta[0] = pi * B[:, obs[0]]
    psi[0] = 0

    # Recursion: δ_t(j) = max_i [δ_{t-1}(i) P_{ij}] · b_j(y_t)
    for t in range(1, T):
        for j in range(n_states):
            scores = delta[t - 1] * P[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = np.max(scores) * B[j, obs[t]]

    # Termination: find best final state
    best_last_state = np.argmax(delta[T - 1])
    best_prob = delta[T - 1, best_last_state]

    # Backtracking: recover state sequence from ψ
    best_path = np.zeros(T, dtype=int)
    best_path[T - 1] = best_last_state
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    return best_path, best_prob, delta, psi


# ============================================================
# Detection utilities
# ============================================================

def threshold_detector(obs: np.ndarray, threshold_bin: int) -> np.ndarray:
    """
    Binary threshold detector: predict positive if observation >= threshold.

    Parameters
    ----------
    obs : np.ndarray, shape (n,)
        Observations (typically bin indices or continuous values).
    threshold_bin : int or float
        Threshold value. Predictions are 1 if obs[i] >= threshold_bin, else 0.

    Returns
    -------
    predictions : np.ndarray, shape (n,), dtype int
        Binary predictions {0, 1}.

    Examples
    --------
    >>> obs = np.array([1, 3, 2, 5, 4])
    >>> threshold_detector(obs, 3)
    array([0, 1, 0, 1, 1])
    """
    return (obs >= threshold_bin).astype(int)


def roc_from_thresholds(
    obs: np.ndarray,
    true_labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> list:
    """
    Compute ROC curve points by sweeping over threshold values.

    For each threshold, computes (FPR, TPR) using the threshold detector and confusion_rates.

    Parameters
    ----------
    obs : np.ndarray, shape (n,)
        Observations.
    true_labels : np.ndarray, shape (n,)
        Ground truth binary labels {0, 1}.
    thresholds : np.ndarray, optional
        Array of thresholds to evaluate. If None, sweeps from 0 to max(obs)+1.

    Returns
    -------
    roc_points : list of tuples (fpr, tpr, threshold)
        ROC points sorted by increasing FPR (left to right on plot).
        Each tuple contains:
        - fpr: False positive rate at this threshold
        - tpr: True positive rate at this threshold
        - threshold: Threshold value used

    Notes
    -----
    ROC curve traditionally plotted with FPR on x-axis, TPR on y-axis.
    Points range from (0, 0) [never predict positive] to (1, 1) [always predict positive].

    Examples
    --------
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> labels = np.array([0, 0, 1, 1, 1])
    >>> roc_pts = roc_from_thresholds(obs, labels, thresholds=np.arange(0, 7))
    >>> [(round(f, 2), round(t, 2)) for f, t, _ in roc_pts]
    [(0.0, 1.0), (0.5, 1.0), (1.0, 0.67), (1.0, 0.33), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    """
    if thresholds is None:
        thresholds = np.arange(0, int(np.max(obs)) + 2)

    roc_points = []

    for threshold in thresholds:
        predictions = threshold_detector(obs, threshold)
        tpr, fpr = confusion_rates(true_labels, predictions)
        roc_points.append((fpr, tpr, threshold))

    # Sort by FPR for plotting (left to right)
    roc_points.sort(key=lambda x: x[0])
    return roc_points


def hmm_detector_point(
    obs: np.ndarray,
    true_labels: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    pi: Optional[np.ndarray] = None,
    decision_threshold: float = 0.5
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate HMM-based detector on a sequence.

    Decision rule: predict positive if P(X_t = fault_state | Y_1:t) >= decision_threshold.

    Uses forward filtering to compute posterior probabilities of hidden states.

    Parameters
    ----------
    obs : np.ndarray, shape (T,)
        Observation sequence (indices).
    true_labels : np.ndarray, shape (T,)
        Ground truth binary labels {0, 1}.
    P : np.ndarray, shape (n_states, n_states)
        Transition matrix.
    B : np.ndarray, shape (n_states, n_obs)
        Emission matrix.
    pi : np.ndarray, optional
        Initial state distribution. If None, uses stationary distribution.
    decision_threshold : float, default 0.5
        Decision threshold for posterior probability. Predict positive if
        posterior >= decision_threshold.

    Returns
    -------
    tpr : float
        True positive rate at this decision threshold.
    fpr : float
        False positive rate at this decision threshold.
    fault_posterior : np.ndarray, shape (T,)
        Posterior probability of fault state at each time: P(X_t = 2 | Y_1:t).
    predictions : np.ndarray, shape (T,)
        Binary predictions {0, 1} from HMM detector.

    Notes
    -----
    Assumes fault state is index 2 (third state) in 3-state HMM.
    Computes forward-filtered posteriors (normalized forward variables).

    Examples
    --------
    >>> P = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
    >>> B = np.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
    >>> obs = np.array([0, 1, 1, 1])
    >>> labels = np.array([0, 0, 1, 1])
    >>> tpr, fpr, post, pred = hmm_detector_point(obs, labels, P, B, decision_threshold=0.5)
    >>> (tpr, fpr)
    (1.0, 0.0)  # Depends on specific P, B, pi
    """
    if pi is None:
        pi = stationary_distribution(P)

    # Forward filtering: compute filtered posteriors
    _, alpha = forward(obs, pi, P, B, normalized=True)

    # Extract posterior for fault state (index 2)
    fault_posterior = alpha[:, 2]

    # Make predictions based on posterior threshold
    predictions = (fault_posterior >= decision_threshold).astype(int)

    # Evaluate
    tpr, fpr = confusion_rates(true_labels, predictions)

    return tpr, fpr, fault_posterior, predictions
