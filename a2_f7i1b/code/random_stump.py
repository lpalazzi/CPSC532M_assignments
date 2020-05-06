import numpy as np
import utils
from decision_stump import DecisionStumpInfoGain


class RandomStumpInfoGain(DecisionStumpInfoGain):
    
    def fit(self, X, y):
        # overwrites fit() function from DecisionStumpInfoGain
        # randomly chooses features to fit to
        
        # Randomly select k features.
        # This can be done by randomly permuting
        # the feature indices and taking the first k
        D = X.shape[1]  # returns number of features D
        k = int(np.floor(np.sqrt(D)))   # k is number of features to sample
        
        chosen_features = np.random.choice(D, k, replace=False) # sample k numbers from range [0, D-1] 
                
        DecisionStumpInfoGain.fit(self, X, y, split_features=chosen_features)
