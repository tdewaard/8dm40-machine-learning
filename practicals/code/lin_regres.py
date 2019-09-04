import numpy as np

def lst_sqr_reg(x_train, y_train):
    line = np.ones(1, x_train.shape[1])
    np.concatenate([x_train, line], axis=0)
    w_hat = np.linalg.inv(np.transpose(x_train)*x_train)*np.transpose(x_train)*y_train
    return beta