import numpy as np
from src.pam_clustering import pam_clustering


def test_pam_for_two_classes_and_few_points():
    x = np.array([[4, 7, 2], [6, 5, 2], [6, 7, 2], [1, 1, 1], [1, 3, 1], [3, 1, 1]])
    start_medoids = [3, 5]
    labels, medoids = pam_clustering(x, 2, start_medoids)

    assert (labels == [[2], [2], [2], [3], [3], [3]]).all()
    assert medoids == [2, 3] or medoids == [3, 2]
