import rusty_learning.rusty_learning as _rl
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]


class Perceptron:
    def __init__(self, alpha: float, n_epoch: int) -> None:
        self.alpha = alpha
        self.n_epoch = n_epoch
        self.weights = None

    def train(self, x: FloatArray, y: FloatArray) -> None:
        self.weights = _rl.perceptron.train(self.alpha, self.n_epoch, x, y)

    def predict(self, x: FloatArray) -> FloatArray:
        return _rl.perceptron.predict(self.weights, x)
