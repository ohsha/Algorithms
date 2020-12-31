from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GMM:

    def __init__(self, K):
        self.K = K
        self.mean = None
        self.covariance = None
        self.n_features = None
        self.n_instances = None
        self.W = None

    def _initialize_parameters(self, X):
        self.W = [1. / self.K] * self.K

        random_means = []
        for i in range(self.n_features):
            min_location = np.min(X[:, i])
            max_location = np.max(X[:, i])

            random_feature_location = np.random.uniform(min_location, max_location, self.K)
            random_means.append(random_feature_location)

        covariance = [np.eye(self.n_features)] * self.K
        random_means = np.array(random_means).T

        return random_means, covariance

    def  _multivariate_gaussian(self, x, mean, cov):
        frac = 1. / (np.power((2 * np.pi), mean.shape[0] / 2) * np.sqrt(np.linalg.det(cov)))
        expo = np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))

        return frac * expo


    def _responsibilities_matrix(self, X):
        r_matrix = np.zeros((self.n_instances, self.K))
        for n in range(self.n_instances):
            sum_row = 0
            for c in range(self.K):
                # calculating the likelihood.
                gauss = self._multivariate_gaussian(X[n], self.mean[c], self.covariance[c])
                r_matrix[n, c] = self.W[c] * gauss

                sum_row += r_matrix[n, c]

            for j in range(self.K):
                # calculating the posterior
                r_matrix[n, j] /= sum_row

        return r_matrix


    def fit(self,X, epochs=50):
        self.n_instances, self.n_features = X.shape

        self.mean, self.covariance = self._initialize_parameters(X)

        for epoch in range(epochs):

            ### Expectation ###
            self.r_matrix = self._responsibilities_matrix(X)
            N = np.sum(self.r_matrix, axis=0)

            ### Maximization ###
            for c in range(self.K):
                # mean:
                self.mean[c] = np.sum([self.r_matrix[n][c] * X[n] for n in range(self.n_instances)], axis=0)
                self.mean[c] *= (1. / N[c])
                # covariance:
                b_k = self.r_matrix[:, c].reshape(self.n_instances, 1)
                self.covariance[c] = (b_k * (X - self.mean[c])).T @ (X - self.mean[c])
                self.covariance[c] /= N[c]

                # W - probabilities
                self.W[c] = N[c] / self.n_instances

            if epoch % 10 == 0 :
                checks = np.argmax(self.r_matrix, axis=1)
                count_per_clusters = [np.count_nonzero(checks == c) for c in range(self.K)]
                print(f' epoch #{epoch} \t {count_per_clusters}\n')

    def predict(self, X):

        return np.argmax(self._responsibilities_matrix(X), axis=1)



if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=500, n_features=4, shuffle=True, random_state=42)

    raw_clusters = [np.count_nonzero(y == c) for c in range(3)]
    print(f'raw_clusters: {raw_clusters}')
    model = GMM(K=3)
    model.fit(X, epochs=100)
    y_pred = model.predict(X)
    print(accuracy_score(y, y_pred))

    print('f')
