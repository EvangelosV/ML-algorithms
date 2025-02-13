import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples = len(X)
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Συλλογή προβλέψεων από κάθε δέντρο
        tree_preds = [tree.predict(X) for tree in self.trees]
        tree_preds = np.array(tree_preds)
        # Κλειδώνουμε με πλειοψηφία ψήφων για κάθε δείγμα
        predictions = []
        for i in range(len(X)):
            votes = tree_preds[:, i]
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        return predictions
