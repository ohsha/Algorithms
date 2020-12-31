import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TSNE:

    def __init__(self, n_components=3,  perplexity=30, alpha = 0.1):
        self.n_features = None
        self.n_instances = None
        self.p_table = None
        self.q_table = None
        self.y = None
        self.sigma = None
        self.n_components = n_components
        self.perplexity = perplexity
        self.loss = None
        self.alpha = alpha


    def initialize_parameters(self, X):
        # used for binary search
        self.max = [1e3] * self.n_instances
        self.min = [1e-3] * self.n_instances

        self.y = np.random.ranf((self.n_instances, self.n_components)) * np.max(X)
        self.sigma = np.ones(self.n_instances) * 1e-3



    def _p_distance_calculation(self, X, i):
        p_i = np.exp((np.linalg.norm(X[i] - X, axis=1) ** 2 ) / (-2 * (self.sigma[i] ** 2)))
        p_i /= np.sum(p_i)

        return p_i


    def _perplexity_calculation(self, p_i):
        # perplexity = 2 ^ H_p  when:  H_p = -SUM(p_ij * log2(P_ij)) Shannon entropy
        H_p = np.dot(p_i.T, np.nan_to_num(np.log2(p_i)))

        return 2 ** -H_p


    def _binary_search_perplexity(self, X, i):
        times = 0
        continue_searching = True
        while(continue_searching):
            times += 1
            p_i = self._p_distance_calculation(X, i)
            perplexity_flag = self._perplexity_calculation(p_i)

            delta = self.perplexity - perplexity_flag
            # if delta is positive >>> we need to increase the sigma
            threshold = 0.1
            if abs(delta) < threshold or times > 30:
                return p_i

            else:
                if delta > 0:
                    self.min[i] = self.sigma[i]
                    self.sigma[i] = (self.min[i] + self.max[i]) / 2

                else:
                    self.max[i] = self.sigma[i]
                    self.sigma[i] = (self.min[i] + self.max[i]) / 2


    def _get_p_table(self, X):
        p_table = np.array([self._binary_search_perplexity(X, i) for i in range(self.n_instances)])

        return (p_table + p_table.T ) / 2


    def _q_distance_calculation(self, i):
        q_i = (1 + np.linalg.norm(self.y[i] - self.y, axis=1) ** 2) ** -1
        sum_ = np.sum(q_i)
        q_i /= sum_

        return q_i


    def _get_q_table(self):
        return np.array([self._q_distance_calculation(i) for i in range(self.n_instances)])


    def _gradient_descent(self):
        for i in range(self.n_instances):
            gd = self.p_table[i] - self.q_table[i]
            gd = gd * 1. / (1 + np.linalg.norm(self.y[i] - self.y, axis=1))
            gradient = (self.y[i] - self.y).T @ gd

            self.y[i] = self.y[i] - (4 * self.alpha * gradient)

    def _loss_function(self):
        # applying Kullback-Leibler divergence.
        return np.sum(self.p_table * np.log(self.p_table / self.q_table))


    def fit(self, X, epochs=2000):

        self.n_instances, self.n_features = X.shape
        self.initialize_parameters(X)

        self.p_table = self._get_p_table(X)
        for epoch in range(epochs):
            self.q_table = self._get_q_table()
            self._gradient_descent()
            self.loss = self._loss_function()

            if epoch % 200 == 0:
                print('epoch: #{}, loss: {}'.format(epoch, self.loss))
                print(' ')
                plt.show()

        return self.y



if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=150, n_features=8, shuffle=False, random_state=42)
    colors = ['red'] * 50 + ['blue'] * 50 + ['green'] * 50
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()
    plt.close()

    model = TSNE(n_components=2)
    X_embedded = model.fit(X, epochs=3000)

    colors = ['red'] * 50 + ['blue'] * 50 + ['green'] * 50
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
    plt.show()
    plt.close()