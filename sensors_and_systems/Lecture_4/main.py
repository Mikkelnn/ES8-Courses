import numpy as np

rand = np.random.default_rng()
steps = 100

w = rand.standard_normal(steps)
v = rand.standard_normal(steps)

def ut_alg(x, P, alpha = 1, kappa = 2, beta = 0):
    x_i = [0]
    w_i = [0]

    n = x.size
    lambda_ = alpha**2 * (n + kappa) - n
    k = np.sqrt(n + lambda_)

    l, u = np.linalg.eig(P)
    for i in range(l+1):
        if i == 0:
            x_i.append(np.mean(x))
        elif 1 <= i <= n:
            x_i.append(np.mean(x) + k*u[i]*np.sqrt(l[i]))
        elif n+1 <= i <= 2*n:
            x_i.append(np.mean(x) - k*u[i]*np.sqrt(l[i]))


if __name__ == "__main__":
    for _ in range(steps):
        pass