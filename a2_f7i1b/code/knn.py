"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y 

    def predict(self, Xtest):

        # compute distance between all pairs of points in Xtest and self.X
        dist = utils.euclidean_dist_squared(self.X, Xtest)    # returns N by T matrix

        # loop through each test data point and determine predicted class labels
        ypred = np.zeros(Xtest.shape[0])
        for t,y in enumerate(ypred):
            knn = np.argsort(dist[:,t])[:self.k]    # gets indices of k nearest neighbours
            knn_labels = self.y[knn]                # array of labels of the k nearest neighbours
            ypred[t] = utils.mode(knn_labels)       # saves most common label (mode)
            
        return ypred
