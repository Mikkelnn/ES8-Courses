"""
Exercise 1: Threshold-based congestion detection.

This exercise analyzes statistical detection of network congestion using
exponential models of Round-Trip Time (RTT) under good and congested conditions.

Models:
    - Good condition:       RTT ~ Exponential(mean = 500 ms)
    - Congestion condition: RTT ~ Exponential(mean = 800 ms)

Decision rule: Predict congestion if RTT >= threshold

Analysis includes:
    - Threshold optimization for different prior probabilities
    - ROC curve for threshold-based detector
    - Accuracy curves under different assumptions
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils import setup_plot_theme

MEAN_GOOD = 500.0
MEAN_CONGESTED = 800.0


def exp_pdf(x: np.ndarray, mean: float) -> np.ndarray:
    """
    Exponential probability density function.

    PDF: f(x; λ) = λ exp(-λ x) where λ = 1/mean.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate PDF (x ≥ 0).
    mean : float
        Mean of the exponential distribution (μ = 1/λ > 0).

    Returns
    -------
    np.ndarray
        PDF values at x.
    """
    lam = 1.0 / mean
    return lam * np.exp(-lam * x)


def exp_survival(threshold: float, mean: float) -> float:
    """
    Exponential survival (tail) probability.

    Tail probability: P(X ≥ threshold) = exp(-threshold / mean).

    Parameters
    ----------
    threshold : float
        Threshold value (x_0 ≥ 0).
    mean : float
        Mean of exponential distribution.

    Returns
    -------
    float
        P(X ≥ threshold).

    Notes
    -----
    For detector using rule "predict congestion if RTT >= threshold":
    - Under good condition: P(X ≥ threshold | λ_good) = false positive rate
    - Under congestion: P(X ≥ threshold | λ_cong) = true positive rate
    """
    return np.exp(-threshold / mean)


def true_positive_rate(threshold: float) -> float:
    """
    True Positive Rate (TPR) = P(predict congestion | actually congested).

    Probability of detecting congestion when system is genuinely congested.

    Parameters
    ----------
    threshold : float
        Decision threshold for RTT (milliseconds).

    Returns
    -------
    float
        TPR ∈ [0, 1].
    """
    return exp_survival(threshold, MEAN_CONGESTED)


def false_positive_rate(threshold: float) -> float:
    """
    False Positive Rate (FPR) = P(predict congestion | good condition).

    Probability of raising false alarm when system is healthy.

    Parameters
    ----------
    threshold : float
        Decision threshold for RTT (milliseconds).

    Returns
    -------
    float
        FPR ∈ [0, 1].
    """
    return exp_survival(threshold, MEAN_GOOD)


def true_negative_rate(threshold: float) -> float:
    """
    True Negative Rate (TNR) = 1 - FPR.

    Probability of correctly identifying good condition.

    Parameters
    ----------
    threshold : float
        Decision threshold for RTT (milliseconds).

    Returns
    -------
    float
        TNR ∈ [0, 1].
    """
    return 1.0 - false_positive_rate(threshold)


def false_negative_rate(threshold: float) -> float:
    """
    False Negative Rate (FNR) = 1 - TPR.

    Probability of missing (not detecting) actual congestion.

    Parameters
    ----------
    threshold : float
        Decision threshold for RTT (milliseconds).

    Returns
    -------
    float
        FNR ∈ [0, 1].
    """
    return 1.0 - true_positive_rate(threshold)


def accuracy(threshold: float, p_good: float) -> float:
    """
    Overall classification accuracy for a given prior probability.

    Accuracy = P(correct) = p_good · TNR + p_cong · TPR
    where p_cong = 1 - p_good.

    Parameters
    ----------
    threshold : float
        Decision threshold for RTT (milliseconds).
    p_good : float
        Prior probability of good condition (0 ≤ p_good ≤ 1).

    Returns
    -------
    float
        Overall accuracy ∈ [0, 1].

    Notes
    -----
    For optimal threshold: maximize over all possible thresholds.
    Different priors may lead to different optimal thresholds.
    """
    p_cong = 1.0 - p_good
    tnr = true_negative_rate(threshold)
    tpr = true_positive_rate(threshold)
    return p_good * tnr + p_cong * tpr



def main() -> None:
    """
    Main analysis: compute detection metrics and plot results for exercises.

    Analyzes threshold-based detector for exponential RTT model under two scenarios:
    1. Prior P(good) = 0.7 - optimize for likely good conditions
    2. Prior P(good) = 0.4 - optimize for likely congestion

    Generates 5 plots:
    - Plot 1: PDF density comparison (good vs congested)
    - Plot 2: True Positive Rate (sensitivity) vs threshold
    - Plot 3: False Positive Rate (1-specificity) vs threshold
    - Plot 4: ROC curve
    - Plot 5: Accuracy vs threshold for two priors
    """
    setup_plot_theme(figsize=(8, 5))

    # Range for density plot
    x = np.linspace(0, 4000, 1000)

    # Threshold range
    thresholds = np.linspace(0, 4000, 1000)

    # PDFs
    pdf_good = exp_pdf(x, MEAN_GOOD)
    pdf_cong = exp_pdf(x, MEAN_CONGESTED)

    # Threshold curves
    tpr = true_positive_rate(thresholds)
    fpr = false_positive_rate(thresholds)

    # Accuracy curves for two priors
    acc_70_good = accuracy(thresholds, p_good=0.7)
    acc_40_good = accuracy(thresholds, p_good=0.4)

    # Best thresholds
    idx_70 = np.argmax(acc_70_good)
    idx_40 = np.argmax(acc_40_good)

    best_th_70 = thresholds[idx_70]
    best_acc_70 = acc_70_good[idx_70]

    best_th_40 = thresholds[idx_40]
    best_acc_40 = acc_40_good[idx_40]

    print("Best threshold for p(good)=0.7:")
    print(f"  threshold = {best_th_70:.2f} ms")
    print(f"  accuracy  = {best_acc_70:.4f}")

    print("\nBest threshold for p(good)=0.4:")
    print(f"  threshold = {best_th_40:.2f} ms")
    print(f"  accuracy  = {best_acc_40:.4f}")

    # --------------------------------------------------------
    # Plot 1: densities
    # --------------------------------------------------------
    plt.figure()
    plt.plot(x, pdf_good, label="Good: Exp(mean=500 ms)")
    plt.plot(x, pdf_cong, label="Congested: Exp(mean=800 ms)")
    plt.xlabel("RTT (ms)")
    plt.ylabel("Density")
    plt.title("Density of RTT distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 2: TPR over threshold
    # --------------------------------------------------------
    plt.figure()
    plt.plot(thresholds, tpr, label="TPR")
    plt.xlabel("Threshold (ms)")
    plt.ylabel("True Positive Rate")
    plt.title("TPR over threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 3: FPR over threshold
    # --------------------------------------------------------
    plt.figure()
    plt.plot(thresholds, fpr, label="FPR")
    plt.xlabel("Threshold (ms)")
    plt.ylabel("False Positive Rate")
    plt.title("FPR over threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 4: ROC curve
    # --------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 5: Accuracy over threshold for two priors
    # --------------------------------------------------------
    plt.figure()
    plt.plot(thresholds, acc_70_good, label="Accuracy, P(good)=0.7")
    plt.plot(thresholds, acc_40_good, label="Accuracy, P(good)=0.4")

    plt.scatter([best_th_70], [best_acc_70], marker="o")
    plt.scatter([best_th_40], [best_acc_40], marker="o")

    plt.annotate(f"max @ {best_th_70:.0f} ms",
                 (best_th_70, best_acc_70),
                 textcoords="offset points", xytext=(8, 8))
    plt.annotate(f"max @ {best_th_40:.0f} ms",
                 (best_th_40, best_acc_40),
                 textcoords="offset points", xytext=(8, -15))

    plt.xlabel("Threshold (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()