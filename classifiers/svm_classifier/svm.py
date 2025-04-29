import numpy as np

class CustomSVM:
    def __init__(self, C=1.0, gamma=0.1, max_iter=1000, lr=0.001):
        self.C = C                    # Regularization parameter
        self.gamma = gamma            # RBF kernel gamma
        self.max_iter = max_iter      # Number of training iterations
        self.lr = lr                  # Learning rate

    def rbf_kernel(self, X1, X2):
        # Compute the RBF (Gaussian) kernel matrix
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}

        for cls in self.classes:
            # Create binary labels for one-vs-rest
            binary_y = np.where(y == cls, 1, -1)
            n_samples = X.shape[0]
            alpha = np.zeros(n_samples)
            K = self.rbf_kernel(X, X)

            # Gradient descent to optimize alpha
            for _ in range(self.max_iter):
                for i in range(n_samples):
                    margin = np.dot(alpha * binary_y, K[:, i])
                    grad = 1 - binary_y[i] * margin
                    alpha[i] += self.lr * grad
                    alpha[i] = min(max(alpha[i], 0), self.C)

            # Store support vectors and parameters
            sv = alpha > 1e-5
            self.models[cls] = {
                'X': X[sv],
                'y': binary_y[sv],
                'alpha': alpha[sv]
            }

    def project(self, X, model):
        # Project test samples onto the decision boundary using the RBF kernel
        K = self.rbf_kernel(X, model['X'])
        return np.dot(K, model['alpha'] * model['y'])

    def predict(self, X):
        # Compute decision values for each class and take the class with the highest value
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            scores[:, idx] = self.project(X, self.models[cls])
        return self.classes[np.argmax(scores, axis=1)]
