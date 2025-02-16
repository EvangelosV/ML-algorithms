import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    # X: features vector list, y: category list
    def fit(self, X, y):
        n_samples = len(X)
        self.trees = []
        for i in range(self.n_trees):
            # bootstrap sampling me replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            #selecting new samples for the new tree based on the indices
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            # Etoimo Decision Tree apo sklearn
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth, max_features=self.max_features)
            #recursion
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # recursion gia kathe tree
        tree_preds = [tree.predict(X) for tree in self.trees]
        tree_preds = np.array(tree_preds) # tree_preds = (n_trees, n_samples) 1d array 
        # array me ena prediction/review
        predictions = []
        for i in range(len(X)):
            # selects all the trees for the i-th sample
            votes = tree_preds[:, i] 
            # selecting the most common prediction 
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        return predictions
