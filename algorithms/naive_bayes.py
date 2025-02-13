import math
import numpy as np

class BernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.feature_log_prob_neg_ = {}
        self.classes_ = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.classes_ = sorted(set(y))
        for c in self.classes_:
            X_c = [x for x, label in zip(X, y) if label == c]
            n_c = len(X_c)
            feature_count = [sum(x[i] for x in X_c) for i in range(n_features)]
            prob = [(count + self.alpha) / (n_c + 2 * self.alpha) for count in feature_count]
            self.feature_log_prob_[c] = [math.log(p) for p in prob]
            self.feature_log_prob_neg_[c] = [math.log(1 - p) for p in prob]
            self.class_log_prior_[c] = math.log(n_c / n_samples)

    def predict(self, X):
        predictions = []
        for x in X:
            scores = {}
            for c in self.classes_:
                score = self.class_log_prior_[c]
                for i, xi in enumerate(x):
                    if xi:
                        score += self.feature_log_prob_[c][i]
                    else:
                        score += self.feature_log_prob_neg_[c][i]
                scores[c] = score
            predictions.append(max(scores, key=scores.get))
        return predictions
