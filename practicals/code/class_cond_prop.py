import numpy as np
import scipy.stats as ss


def cond_prop(X, y):
    """
    Calculates class conditional probability for attribute values stored in X,
    class values stored in y.
    :param X: N x p input matrix
    :param y: N x 1 output vector (class, binary)
    :return: CP: N x p matrix storing the calculated class conditional probabilities
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # prior probability of class variable
    # Y = 1
    PY_pos = np.count_nonzero(y)/X.shape[0]
    # Y = 0
    PY_neg = 1 - PY_pos

    # Create vector PY storing all prior probabilities P(Y|X)
    PY = (y > 0).astype(int) * PY_pos
    PY[PY == 0] = PY_neg

    # calculate matrix P storing all P(X = x) probabilities
    PX = ss.norm.pdf(X, mu, sigma)

    # compute conditional probabilities: P(X = x | Y = y) = (P(X=x)*P(Y=y|X=x))/P(Y=y)
    # P(Y = y) = sum_x(P(Y|x)*P(x))

    # prior matrix stores every attribute's P(X)*P(Y|x) calculations
    prior = PX * PY

    # compute P(y) for all attributes
    norm = np.sum(prior, axis=1)

    # compute all conditional probabilities P(X=x|Y)
    CP = (PX * PY)/norm[:, np.newaxis]

    return CP