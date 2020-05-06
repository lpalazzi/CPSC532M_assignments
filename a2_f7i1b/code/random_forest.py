import numpy as np
import utils
from random_tree import RandomTree

class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

        # Create array of size num_trees with RandomTree models
        self.rand_trees = []
        for i in range(self.num_trees):
            self.rand_trees.append(RandomTree(max_depth=self.max_depth))

    def fit(self, X, y):
        for rand_tree in self.rand_trees:
            rand_tree.fit(X,y)
    
    def predict(self, X):
        y_all = np.zeros((X.shape[0],self.num_trees))   # label predictions, each row holding the predicted labels from each RandomTree in forest
        y_pred = np.zeros(X.shape[0])                   # 1D label prediction array, each row holds the mode of each row in y_all
        
        # loop through each random tree (y_all columns) and store predicted labels
        for j, y_col in enumerate(y_all.T):
            y_all.T[j] = self.rand_trees[j].predict(X)
            
        # loop through each data point's set of predicted labels (y_all rows) and choose most common one (mode)
        for i, y_row in enumerate(y_all):
            y_pred[i] = utils.mode(y_row)
        
        return y_pred
