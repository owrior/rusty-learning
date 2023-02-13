import pytest
import rusty_learning as rl
import numpy as np


def test_perceptron_predict(linearly_related_data):
    weights = np.array([10.0, 2, 3])
    res = rl.predict(weights, linearly_related_data)
    print(res)


def test_perceptron_predict(linearly_related_data):
    y = np.array([1.0, 1, 0, 0, 0])
    res = rl.train(linearly_related_data, y, 0.01, 100)
    print(res)
