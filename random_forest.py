
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



class Node:

    def __init__(self, predicted_class, n_instances, n_samples_per_class):
        self.predicted_class = predicted_class
        self.n_instances = n_instances
        self.n_samples_per_class = n_samples_per_class
        self.threshold = 0
        self.feature_index = 0
        self.left = None
        self.right = None


class DecisionTree:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth


    def _gini_prob(self, y):
        size_node = y.size

        p_j = np.array(y[0].value_counts(sort=False))
        P = (p_j / (size_node + 1e-5)) ** 2
        P_k = 1 - np.sum(P)

        return P_k


    def _gini(self,X, y, value):
        n_instances = X.shape[0]
        if n_instances == 0:
            return 0

        left = self._gini_prob(y.loc[X <= value])
        sum_of_left = len(X[X <= value])

        right = self._gini_prob( y.loc[X > value])
        sum_of_right = len(X[X > value])

        gini_left = left * sum_of_left / (n_instances )
        gini_right = right * sum_of_right / (n_instances)

        return gini_left + gini_right, value


    def _best_split(self, X, y):
        best_idx, best_thr = None, None
        gini_list = np.array([self._gini(X.iloc[:,i], y, np.mean(X.iloc[:,i])) for i in range(self.n_features)])
        best_idx = np.argmin(gini_list[:,0])
        best_thr = gini_list[best_idx, 1]

        return X.columns[best_idx], best_thr


    def _build_tree(self,X, y,depth=0):
        n_samples_per_class = y[0].value_counts(sort=False)
        predicted_class = n_samples_per_class.idxmax()

        node = Node(
            predicted_class=predicted_class,
            n_instances = X.shape[0],
            n_samples_per_class=n_samples_per_class
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_idx = X.loc[:, idx] <= thr
                X_left, y_left = X.loc[left_idx], y.loc[left_idx]
                X_right, y_right = X.loc[~left_idx], y.loc[~left_idx]

                node.feature_index = idx
                node.threshold = thr

                if X_left.shape[0]:
                    node.left = self._build_tree(X_left, y_left, depth + 1)
                if X_right.shape[0]:
                    node.right = self._build_tree(X_right, y_right, depth + 1)

        return node


    def _predict(self, x, node):
        if x.loc[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right

        if node.left == None or node.right == None:
            return node.predicted_class
        else:
            return self._predict(x, node)


    def fit(self, X, y, n_classes=None):
        if n_classes is not None:
            self.n_classes = n_classes
        else:
            self.n_classes = len(y[0].unique())

        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)


    def predict(self, X_test):
        X_pred = pd.Series(index=X_test.index, dtype=np.int)
        predictions = [self._predict(X_test.iloc[i, :], self.tree) for i in range(X_test.shape[0])]
        X_pred.iloc[:] = predictions

        return X_pred



class RandomForest():

    def __init__(self, max_tree=4, tree_depth=2):
        self.max_tree = max_tree
        self.tree_depth = tree_depth


    def _build_forest(self, X, y, subset=0.4):
        trees_list = np.array([])
        for k in range(self.max_tree):
            n_instances_samples = int(self.n_instances * subset)
            n_features_samples = int(np.sqrt(self.n_features))

            random_instances = X.index[np.random.choice(self.n_instances, n_instances_samples, replace=False)].values
            random_features = X.columns[np.random.choice(self.n_features, n_features_samples, replace=False)].values

            X_rand= X.loc[random_instances,:]
            X_rand = X_rand.loc[:, random_features]
            y_rand = y.loc[random_instances]

            dt = DecisionTree(max_depth=self.tree_depth)
            dt.fit(X_rand, y_rand, self.n_classes)

            trees_list = np.append(trees_list, dt)

        return trees_list


    def _get_majority_decision(self, votes):
        return votes.value_counts().idxmax()


    def fit(self, X, y):
        self.n_classes = len(y[0].unique())
        self.n_features = X.shape[1]
        self.n_instances = X.shape[0]
        self.forest = self._build_forest(X, y)


    def predict(self, X_test):
        X_preds = pd.DataFrame([tree.predict(X_test) for tree in self.forest]).T
        votes = [self._get_majority_decision(X_preds.iloc[i, :]) for i in range(X_test.shape[0])]
        votes = pd.Series(votes, X_test.index)

        return votes


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=500, n_features=7, shuffle=False, random_state=42)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = RandomForest(max_tree=30, tree_depth=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc *100.}%')
