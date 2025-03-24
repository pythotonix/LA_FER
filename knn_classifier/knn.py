import numpy as np
from collections import Counter

class KNN:
    
    def __init__(self, num_neighbors: int = 5):
        self.num_neighbors = num_neighbors
        
    def fit(self, X: np.array, y: np.array):
        """
        Memorize training data.
        """
        self.X = X
        self.y = y
        
    def get_distance(self, a: np.array, b: np.array):
        """
        Calculate Euclidean distance between two examples.
        """
        return np.sqrt(np.sum((a - b) ** 2))
    
    def get_neighbors(self, example: np.array):
        """
        Find and rank nearest neighbors of an example.
        """
        distances = []
        # Calculate distances as tuples (index, distance)
        for i in range(len(self.X)):
            distances.append((i, self.get_distance(self.X[i], example)))
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        # Return IDs and distances of top neighbors
        return distances[:self.num_neighbors]
    
    def predict(self, X: np.array):
        """
        Predict class labels for given examples using majority vote.
        """
        predictions = []
        for idx in range(len(X)):
            example = X[idx]
            k_neighbors = self.get_neighbors(example)
            k_y_values = [self.y[item[0]] for item in k_neighbors]
            # Determine the most common label among the neighbors
            prediction = Counter(k_y_values).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)
    
    def score(self, X: np.array, y: np.array) -> float:
        """
        Compute the accuracy of the classifier.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
