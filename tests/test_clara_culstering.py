from src.clara_clustering import clara_clustering
import numpy as np

def test_clara_dummy():
    x = np.array([[4, 7], [6, 5], [6, 7], [1, 1], [1, 3], [3, 1]])
    start_medoids = [3, 5]
    labels, medoids = clara_clustering(x, 2, start_medoids, sample_size=6)

    assert (labels == [[2], [2], [2], [3], [3], [3]]).all()
    assert sorted(medoids) == [2, 3]
