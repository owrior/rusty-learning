import pytest
import rusty_learning as rl
import numpy as np


def test_perceptron_predict(linearly_related_data):
    weights = np.array([[10.0], [2], [3]])
    res = rl.predict(weights, linearly_related_data)
    print(res)


def test_perceptron_predict(linearly_related_data):
    x = linearly_related_data[0]
    y = linearly_related_data[1]
    res = rl.train(x, y, 0.01, 10)
    print(res)
