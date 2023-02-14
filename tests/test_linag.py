import pytest
import rusty_learning as rl
import numpy as np


def test_perceptron_predict(linearly_related_data):
    weights = np.array([[10.0], [2], [3]])
    x, y = linearly_related_data
    res = rl.predict(weights, x)
    print(res)


def test_perceptron_predict(linearly_related_data):
    x = linearly_related_data[0]
    y = linearly_related_data[1]
    weights, accuracy = rl.train(x, y, 0.1, 10)
    y_hat = rl.predict(
        np.array([[-0.1], [0.20653640140000007], [-0.23418117710000003]]), x
    )
    print(weights.flatten(), accuracy)
    print(y.flatten())
    print(y_hat.flatten())
    print((y - y_hat).flatten())
