"""
Probability Density Function Exercise

f(x) =
    (3/5)*x                                 for 0 < x <= 1
    ((sqrt(3x) - sqrt(12))^2) / 5       for 1 < x <= 3
    12/5 - (3/5)x                       for 3 < x <= 4
    0                                   otherwise

This script:
1. Verifies that f(x) is a valid PDF
2. Computes the exact expectation and variance
3. Samples the random variable using rejection sampling
4. Estimates the sample mean and variance
"""

import math
import random
import numpy as np
from scipy.integrate import quad


# ------------------------------------------------------------
# Define the PDF
# ------------------------------------------------------------

def f(x):
    if 0 < x <= 1:
        return (3 / 5)*x

    elif 1 < x <= 3:
        return ((math.sqrt(3)*x - math.sqrt(12)) ** 2) / 5

    elif 3 < x <= 4:
        return 12 / 5 - (3 / 5) * x

    return 0.0


# ------------------------------------------------------------
# Check whether this is a valid PDF
# ------------------------------------------------------------

integral, _ = quad(f, -np.inf, np.inf)

print("Integral of f(x) over R =", integral)

if abs(integral - 1.0) < 1e-8:
    print("f(x) IS a valid probability density function.\n")
else:
    print("f(x) is NOT a valid probability density function.\n")


# ------------------------------------------------------------
# Compute exact expectation E[X]
# ------------------------------------------------------------

def x_f(x):
    return x * f(x)

EX, _ = quad(x_f, -np.inf, np.inf)

print("Exact expectation E[X] =", EX)


# ------------------------------------------------------------
# Compute exact variance Var(X)
# ------------------------------------------------------------

def x2_f(x):
    return x**2 * f(x) 

EX2, _ = quad(x2_f, -np.inf, np.inf)

VarX = EX2 - EX**2

print("Exact variance Var(X) =", VarX)
print()


# ------------------------------------------------------------
# Rejection Sampling
# ------------------------------------------------------------
#
# Support is on [0, 4]
#
# Maximum of f(x) occurs at x=1:
# f(1)=3/5 = 0.6
#
# Use proposal:
#   X ~ Uniform(0,4)
#
# Proposal density:
#   g(x)=1/4
#
# Need M such that:
#   f(x) <= C g(x)
#
# Since max f = 0.6:
#
#   C >= 0.6 / (1/4) = 2.4
#
# We'll use C = 2.4
# ------------------------------------------------------------

C = 2.4

def sample_pdf_v2():
    while True:
        x = random.uniform(0, 4)
        y = random.uniform(0, 0.6)
        if y <= f(x):
            return x

def sample_pdf():
    while True:
        # proposal sample
        x = random.uniform(0, 4)
        
        # acceptance probability
        acceptance = f(x) / (M * (1/4)) 

        # accept/reject
        if random.random() <= acceptance:
            return x


# ------------------------------------------------------------
# Generate samples
# ------------------------------------------------------------

N = 100000

samples = np.array([sample_pdf_v2() for _ in range(N)])

sample_mean = np.mean(samples)
sample_variance = np.var(samples)

print(f"Generated {N} samples")
print("Estimated sample mean     =", sample_mean)
print("Estimated sample variance =", sample_variance)