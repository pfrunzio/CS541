import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return A * B + C.T

def problem_1c (x, y):
    return x.T @ y # inner product

def problem_1d (A, j):
    return np.sum(A[::2, j])

def problem_1e (A, c, d):
    return np.mean(A[np.nonzero((A >= c) & (A <= d))])

def problem_1f (x, k, m, s):
    return ...

def problem_1g (A):
    return ...

def problem_1h (x):
    return ...

def problem_1i (x, k):
    return ...



def linear_regression (X_tr, y_tr):
    ...


def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...


def question4a():
    #load data
    data = np.load('PoissonX.npy')

    #plot empirical distribution
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, density=True, alpha=0.6, color='g', label='Empirical Distribution')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Empirical Probability Distribution')
    # plt.legend()
    # plt.show()

    #set alt rate parameters
    rate_params = [2.5, 3.1, 3.7, 4.3]
    x_values = np.arange(0, int(np.max(data)) + 1)

    #plot poisson distributions
    for l in rate_params:
        poisson_pmf = stats.poisson.pmf(x_values, l)
        plt.plot(x_values, poisson_pmf, label=f'Poisson λ={l}', lw=2)

    plt.xlabel('Value')
    plt.ylabel('Density')
    # plt.title('Poisson Distributions')

    plt.title('Comparison of Empirical and Poisson Distributions')
    plt.legend()
    plt.show()

def question4b():
    mu = 1
    sigma = 1.268857754

    #calculate cumulative distribution function at 0
    probY = stats.norm.cdf(0, loc=mu, scale=sigma)

    #P(Y > 0)
    probY_gt_0 = 1 - probY

    print(probY_gt_0)