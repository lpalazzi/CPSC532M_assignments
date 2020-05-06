import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape
        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals


    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1
        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue
                selected_new = selected | {i} # tentatively add feature "i" to the selected set
                # Calculate the minimum values of w for the current selected indices. 
                temp_w = np.zeros(d)
                temp_w[list(selected_new)], _ = minimize(list(selected_new))
                # Calculate the loss function. 
                f = self.funObj(temp_w, X, y)[0] + len(selected_new)
                if f < minLoss: 
                    minLoss = f
                    bestFeature = i               
                
            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class logRegL1():

    def __init__(self, L1_lambda, maxEvals, verbose=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.L1_lambda = L1_lambda

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape
        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda, self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL2():

    def __init__(self, lammy, maxEvals, verbose=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.lammy = lammy

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy/2)*w.dot(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy*w

        return f, g

    def fit(self,X, y):
        n, d = X.shape
        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
        print(self.w)
    def predict(self, X):
        return np.sign(X@self.w)





class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

# Code for q 3.2
class logLinearClassifier():

    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))
        
        for i in range(self.n_classes):

            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i], self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class softmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w,[k,d])

        # obtain indicator function I(y_i = c)
        indicator = np.zeros((n,k)).astype(bool)
        indicator[np.arange(n), y] = 1

        XW = np.dot(X, W.T)             # (w_c')^T * x_i
        Z = np.sum(np.exp(XW), axis=1)  # sum of exp(XW)

        # compute function
        f = np.sum(-XW[indicator] + np.log(Z))

        # compute gradient
        g = ((( np.exp(XW) / Z[:, None] ) - indicator).T@X).flatten()

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        k = self.n_classes

        self.W = np.zeros(d*k)
        self.w = self.W

        #utils.check_gradient(self, X, y)
        (self.W, f) = findMin.findMin(self.funObj, self.W, self.maxEvals, X, y, verbose=self.verbose)

        self.W = np.reshape(self.W, [k,d])

    def predict(self, X):
        print ("k = " + str(self.n_classes))
        print ("size of X = " + str(X.shape))
        print ("size of W^T = " + str(self.W.T.shape))
        print ("size of X*W^T = " + str((X@self.W.T).shape))
        return np.argmax(X@self.W.T, axis=1)