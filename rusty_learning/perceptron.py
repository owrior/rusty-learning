import rusty_learning as _rl
import numpy as np


class Perceptron:
    def __init__(self, alpha: float, n_epoch: int, num_features: int) -> None:
        self.alpha = alpha
        self.n_epoch = n_epoch
        self._p = _rl.Perceptron(alpha, n_epoch, num_features)
        self.weights = self._p.get_weights()

    def train(self, x: np._ArrayFloat_co, y: np._ArrayFloat_co) -> float:
        accuracy = self._p.train(x, y)
        self.weights = self._p.get_weights()
        return accuracy

    def predict(self, x: np._ArrayFloat_co) -> np._ArrayFloat_co:
        return self._p.predict(x)
