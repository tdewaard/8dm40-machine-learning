import numpy as np
import scipy.stats as ss

def cond_prop(X, y):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # prior probability of class variable
    # Y = 1
    PY_pos = np.count_nonzero(y)/X.shape[0]
    # Y = 0
    PY_neg = 1 - PY_pos

    # calculate matrix P storing all P(X = x) probabilities
    P = ss.norm.pdf(X, mu, sigma)
