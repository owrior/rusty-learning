import pytest
from sklearn import datasets


@pytest.fixture
def separable_data():
    return datasets.make_blobs(
        n_features=2, n_samples=150, centers=2, cluster_std=1.05, random_state=2
    )
