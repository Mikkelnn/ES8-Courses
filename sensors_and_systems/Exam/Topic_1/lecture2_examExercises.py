import math
import random
import numpy as np
from scipy.integrate import quad



# Define the PDF
def f(x):
    if 0 < x <= 1:
        return (3 / 5)*x

    elif 1 < x <= 3:
        return ((math.sqrt(3)*x - math.sqrt(12)) ** 2) / 5

    elif 3 < x <= 4:
        return 12 / 5 - (3 / 5) * x

    return 0.0


# Check whether this is a valid PDF
def is_f_valid_pdf():
    integral, _ = quad(f, -np.inf, np.inf)

    print("Integral of f(x) over R =", integral)

    if abs(integral - 1.0) < 1e-8:
        print("f(x) IS a valid probability density function.\n")
    else:
        print("f(x) is NOT a valid probability density function.\n")


# Compute exact expectation E[X]
def calculate_expectation():
    def x_f(x):
        return x * f(x)

    EX, _ = quad(x_f, -np.inf, np.inf)

    print("Exact expectation E[X] =", EX)
    return EX



# Compute exact variance Var(X)
def calculate_variance(EX):
    def x2_f(x):
        return x**2 * f(x) 

    EX2, _ = quad(x2_f, -np.inf, np.inf)

    var = EX2 - EX**2

    print("Exact variance Var(X) =", var)
    print()
    return var


def sample_pdf():
    while True:
        # proposal sample
        x = random.uniform(0, 4)
        y = random.uniform(0, 1)

        # upper bound on f(x) for x in [0,4]
        max_pdf_value = 0.6 

        # accept/reject
        if y <= f(x) / max_pdf_value:
            return x



# Estimate mean and variace
def estimate_mean_and_variance(N = 100000):
    samples = np.array([sample_pdf() for _ in range(N)])

    sample_mean = np.mean(samples)
    sample_variance = np.var(samples)

    print(f"Generated {N} samples")
    print("Estimated sample mean     =", sample_mean)
    print("Estimated sample variance =", sample_variance)


def main():
    is_f_valid_pdf()
    calc_mean = calculate_expectation()
    calc_var = calculate_variance(calc_mean)

    estimate_mean_and_variance()

if __name__ == "__main__":    
    main()