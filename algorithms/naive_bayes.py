import math
import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.feature_log_prob = {}
        self.feature_log_prob_neg = {}
        self.classes = None

    # X: features vector list, y: category list
    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.classes = sorted(set(y))
        for c in self.classes:
            #X_c : features vector list gia kathe category
            X_c = [x for x, label in zip(X, y) if label == c]
            n_c = len(X_c)
            # Kanei sum poses fores emfanizetai kathe word sto category
            feature_count = [sum(x[i] for x in X_c) for i in range(n_features)]
            # typos NB: P(x_i|c) = (plithos(x_i|c) + alpha) / (plithos(c) + 2*alpha) binary ara k=2
            prob = [(count + self.alpha) / (n_c + 2 * self.alpha) for count in feature_count]
            #P(x_i=1|c)
            self.feature_log_prob[c] = [math.log(p) for p in prob]
            #P(x_i=0|c)
            self.feature_log_prob_neg[c] = [math.log(1 - p) for p in prob]
            # P(c) = plithos(reviews gia c) / plithos(reviews)
            self.class_log_prior[c] = math.log(n_c / n_samples)
            
    def predict(self, X):
        # array me ena prediction/review
        predictions = []
        # gia kathe review
        for x in X:
            scores = {} # 2 scores gia kathe category
            # gia kathe category
            for c in self.classes:
                score = self.class_log_prior[c]
                for i, xi in enumerate(x):
                    if xi: #xi=1
                        score += self.feature_log_prob[c][i]
                    else: #xi=0
                        score += self.feature_log_prob_neg[c][i]
                scores[c] = score
            # add sto predictions to category me megalytero score
            predictions.append(max(scores, key=scores.get))
        return predictions
