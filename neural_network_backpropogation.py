import numpy as np

class NeuralNetwork:

    def __init__(self, layers, alpha=0.01, W=None):
        self.layers = layers
        self.alpha = alpha
        self.n_instances = None
        self.n_features = None
        self.W = W


    def _get_initial_weights(self):
        W = []
        for i in range(len(self.layers) - 2):
            # adding +1 dimension for the bias vector.
            w = np.random.randn(self.layers[i] + 1, self.layers[i+1] + 1) # (n+1 X m+1)
            W.append(w / np.sqrt(self.layers[i]))
        # without the bias vector.
        w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        W.append(w / np.sqrt(self.layers[-2]))

        return W


    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-x))


    def _sigmoid_deriv(self, z):
        return z * (1. - z)


    def fit(self, X, y, epochs=1000, verbose=100):
        self.n_instances, self.n_features = X.shape
        if self.W is None:
            self.W = self._get_initial_weights()
        # bias trick
        X = np.c_[X, np.ones((self.n_instances))]

        for epoch in range(epochs):
            # Z equal to ACTIVATION(WiXi)
            Z = [np.atleast_2d(X)]

            # FEEDFORWARD
            for layer in range(0, len(self.W)):
                wx = Z[layer] @ self.W[layer]
                act_wx = self._sigmoid(wx)
                Z.append(act_wx)

            # BACKPROPOGATION
            error = Z[-1] - y

            C = [error * self._sigmoid_deriv(Z[-1])]
            for layer in range(len(Z) - 2, 0, -1):
                delta = C[-1] @ self.W[layer].T
                delta *= self._sigmoid_deriv(Z[layer])
                C.append(delta)

            # reversing the matrix
            C = C[::-1]
            for layer in range(len(self.W)):
                # updating the weight matrix
                self.W[layer] += -self.alpha * (Z[layer].T @ C[layer])

            if epoch == 0 or (epoch + 1) % verbose  == 0 :
                loss = self._loss_function(X, y)
                print("  Epoch: #{},    Loss: {:.7f}".format(epoch+1, loss))


    def predict(self, X, add_bias=True):
        P = np.atleast_2d(X)
        if add_bias:
            P = np.c_[P, np.ones((P.shape[0]))]

        for layer in range(len(self.W)):
            P = self._sigmoid(P.dot(self.W[layer]))

        return P


    def _loss_function(self, X, y):
        y = np.atleast_2d(y)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - y)**2)

        return loss


if __name__ == '__main__':

    # XOR
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[1], [0], [0], [1]])

    model = NeuralNetwork(layers=[2,3,3,1], alpha=0.1)
    model.fit(X, y, epochs=20000)

    predictions = model.predict(X)
    preds = np.around(predictions)
    for i in range(4):
        print('[INFO] data={}, ground_truth={}, prediction={}, label={}'.format(X[i], y[i], predictions[i], preds[i]))

