import numpy as np


def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta


def eval_lin_model(beta, X_test, y_test):
    """
    Evaluates the linear model with coefficient vector beta on testing data,
    by calculating the mean squared error
    :param beta: coefficient vector
    :param X: the testing data X vector/matrix (attribute values)
    :param y_test: the testing data Y vector (output, class variable)
    :return: MSE: Mean Squared Error of the model
    """
    w = beta.T[:, 1:]
    y = beta.T[:, 0] + np.dot(X_test, w.T)
    MSE = np.sum((y - y_test)**2)/y_test.shape[0]

    return MSE, y
