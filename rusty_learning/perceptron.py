import rusty_learning.rusty_learning as _rl
import numpy as np


class Perceptron:
    def __init__(self, alpha: float, n_epoch: int, num_features: int) -> None:
        self.alpha = alpha
        self.n_epoch = n_epoch
        self._p = _rl.Perceptron(alpha, n_epoch, num_features)
        self.weights = self._p.get_weights()

    def train(self, x, y) -> float:
        accuracy = self._p.train(x, y)
        self.weights = self._p.get_weights()
        return accuracy

    def predict(self, x):
        return self._p.predict(x)

    def set_weights(self, weights) -> None:
        self._p.set_weights(weights)
        self.weights = self._p.get_weights()
