import pytest
import rusty_learning as rl
import numpy as np
import timeit
from sklearn.linear_model import Perceptron as SKPerceptron

TIMEIT_NUMBER = 10


class BasicPerceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


def test_perceptron_predict(separable_data):
    weights = np.array([[10.0], [2], [3]])
    X, y = separable_data
    res = rl.predict(weights, X)


def test_perceptron_train(separable_data):
    X, y = separable_data

    weights, accuracy = rl.train(X, y.reshape((-1, 1)).astype(float), 0.01, 100)
    y_hat = rl.predict(weights, X).flatten()

    np.testing.assert_almost_equal(
        weights, np.array([[3.53], [0.83434904], [1.59407104]])
    )
    assert 1 - ((y - y_hat.flatten()) ** 2).mean() == accuracy


def test_benchmark_perceptron_train(separable_data):
    X, y = separable_data

    # Calculate rust implemted time
    rl_time = np.round(
        timeit.timeit(
            lambda: rl.train(
                X, y.reshape((-1, 1)).astype(float), alpha=0.01, n_epoch=1000
            ),
            number=TIMEIT_NUMBER,
        ),
        decimals=2,
    )

    # Calculate pure python numpy time
    p = BasicPerceptron(learning_rate=0.01, n_iters=1000)
    python_time = np.round(
        timeit.timeit(lambda: p.fit(X, y), number=TIMEIT_NUMBER), decimals=2
    )

    print(
        f"Rust implementation: {rl_time}, Numpy implementation: {python_time}\nMultiple: {python_time / rl_time}\n"
    )

    # Calculate pure python numpy time
    p = SKPerceptron(alpha=0.01, max_iter=1000)
    python_time = np.round(
        timeit.timeit(lambda: p.fit(X, y), number=TIMEIT_NUMBER), decimals=2
    )

    print(
        f"Rust implementation: {rl_time}, Sklearn implementation: {python_time}\nMultiple: {python_time / rl_time}"
    )
