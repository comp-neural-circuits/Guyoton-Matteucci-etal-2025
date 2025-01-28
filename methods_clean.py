import numpy as np
from numpy.linalg import inv
import scipy.ndimage as image
from scipy.stats import norm

def thresh(x):
    #assumes matrix/array input
    x[x < 0] = 0
    return x

def sub_rand_weights(dim1, dim2, p):
    '''
    returns 2*dim1 x 2*dim2 block matrix with off-diagonal blocks zero
    dim1: num. projection targets
    dim2: num. projections
    p: probability of connection
    '''
    W00 = np.matrix(np.random.rand(dim1, dim2)<p)
    W11 = np.matrix(np.random.rand(dim1, dim2)<p)
    W01 = np.matrix(np.zeros((dim1, dim2)))
    W10 = np.matrix(np.zeros((dim1, dim2)))
    return np.hstack((np.vstack((W00, W01)), np.vstack((W10, W11))))

