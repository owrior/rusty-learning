import pytest
import numpy as np 

@pytest.feature
def linearly_related_data():
    return np.array(
        [1, 1.1],
        [2, 2.4],
        [3, 3.2],
        [4, 4.7],
        [5, 5.8],
    )