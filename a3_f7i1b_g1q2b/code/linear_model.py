import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        self.w = solve(X.T @ z @ X, X.T @ z @ y)     # From the formula 2.2.3.



class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        # Calculate the function value
        f = np.sum(np.log(np.exp(w.T*X - y) + np.exp(y - w.T*X)))
        # Calculate the gradient value
        g = np.sum(X * ((np.exp(w.T * X - y) - np.exp(-w.T * X + y))/(np.exp(w.T * X - y) + np.exp(-w.T * X + y))),axis = 0)
        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        X_bias = np.concatenate((X,np.ones((X.shape[0],1))), axis=1)
        self.w = solve(X_bias.T@X_bias, X_bias.T@y)

    def predict(self, X):
        X_bias = np.concatenate((X,np.ones((X.shape[0],1))), axis=1)
        return X_bias@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z = np.ones((X.shape[0],0)) # initialize Z matrix with shape (N,0)
        for i in range((self.p+1)):
            Z = np.concatenate((Z,np.power(X,i)), axis=1) # loop through [0,1,2,...,p] and add column of X^i
        return Z
