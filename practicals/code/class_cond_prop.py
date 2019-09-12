import numpy as np
import scipy.stats as ss


def cond_prop(X, y):
    """
    Calculate class conditional probabilities assuming a gaussian distribution.
    :param X: Feature matrix (N x p) with N number of instances and p number of parameters
    :param y: Class vector N x 1 (binary) vector
    :return: CP: cond. prop. not separated by class
             CP_pos: normalised cond. prop. pos class
             CP_neg: normalised cond. prop. neg class
             PXpos: gaussian of pos class
             PXneg: gaussian of neg class
             priorPos: prior probabilty pos class
             priorNeg: prior probability neg class
    """
    i = y[:, 0] > 0
    j = y[:, 0] == 0

    # separate data between class values
    posX =  X[i, :]
    posMu = np.mean(posX, axis=0)
    posSigma = np.std(posX, axis=0)

    negX =  X[j, :]
    negMu = np.mean(negX, axis=0)
    negSigma = np.std(negX, axis=0)

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
    PXpos = ss.norm.pdf(posX, posMu, posSigma)
    PXneg = ss.norm.pdf(negX, negMu, negSigma)

    # compute conditional probabilities: P(X = x | Y = y) = (P(X=x)*P(Y=y|X=x))/P(Y=y)
    # P(Y = y) = sum_x(P(Y|x)*P(x))

    # prior matrix stores every attribute's P(X)*P(Y|x) calculations
    prior = PX * PY
    priorPos = PXpos * PY[i, :]
    priorNeg = PXneg * PY[j, :]

    # compute P(y) for all attributes
    norm = np.sum(prior, axis=1)
    normPos = np.sum(priorPos, axis=1)
    normNeg = np.sum(priorNeg, axis=1)

    # compute all conditional probabilities P(X=x|Y)
    CP = prior/norm[:, np.newaxis]
    CP_pos = priorPos/normPos[:, np.newaxis]
    CP_neg = priorNeg/normNeg[:, np.newaxis]

    return CP, CP_pos, CP_neg, PXpos, PXneg, priorPos, priorNeg