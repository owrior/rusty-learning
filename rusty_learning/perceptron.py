import rusty_learning.rusty_learning as _rl
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]


class Perceptron:
    def __init__(self, alpha: float, n_epoch: int, num_features: int) -> None:
        self.alpha = alpha
        self.n_epoch = n_epoch
        self._p = _rl.Perceptron(alpha, n_epoch, num_features)
        self.weights = self._p.get_weights()

    def train(self, x: FloatArray, y: FloatArray) -> float:
        accuracy = self._p.train(x, y)
        self.weights = self._p.get_weights()
        return accuracy

    def predict(self, x: FloatArray) -> FloatArray:
        return self._p.predict(x)

    def set_weights(self, weights: FloatArray) -> None:
        self._p.set_weights(weights)
        self.weights = self._p.get_weights()
