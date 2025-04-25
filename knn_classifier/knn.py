import numpy as np
from collections import Counter
from collections import defaultdict


def compute_all_pairwise_distances(X_test, X_train):
    X_test_sq = np.sum(X_test**2, axis=1).reshape(-1, 1)        # shape: (num_test, 1)
    X_train_sq = np.sum(X_train**2, axis=1).reshape(1, -1)       # shape: (1, num_train)
    dists = np.sqrt(np.maximum(X_test_sq + X_train_sq - 2 * np.dot(X_test, X_train.T), 0.0))
    return dists

class KNN_Custom:
    
    def __init__(self, k = 5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        
    def predict(self, X_test):
        print("making prediction:")
        
        distances = compute_all_pairwise_distances(X_test, self.X_train)
        
        print("finish calculating")
    
        epsilon = 1e-5
        final_predictions = []

        epsilon = 1e-5
        num_test = X_test.shape[0]
        predictions = []

        for i in range(num_test):
            k_idx = np.argpartition(distances[i], self.k)[:self.k]
            k_labels = self.y_train[k_idx]
            k_distances = distances[i][k_idx]

            weights = 1 / (k_distances + epsilon)

            label_weights = defaultdict(float)
            for label, weight in zip(k_labels, weights):
                label_weights[label] += weight

            predicted_label = max(label_weights.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_label)

        return np.array(predictions)