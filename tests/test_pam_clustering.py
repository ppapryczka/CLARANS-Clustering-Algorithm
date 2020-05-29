import numpy as np
from src.pam_clustering import pam_clustering


def test_pam_for_two_classes_and_few_points():
    x = np.array([[4, 7], [6, 5], [6, 7], [1, 1], [1, 3], [3, 1]])
    start_medoids = [3, 5]
    labels, medoids = pam_clustering(x, 2, start_medoids)

    assert (labels == [[2], [2], [2], [3], [3], [3]]).all()
    assert medoids == [2, 3] or medoids == [3, 2]


def test_pam_for_two_classes_and_few_points_random_medoids():
    x = np.array([[4, 7], [6, 5], [6, 7], [1, 1], [1, 3], [3, 1]])
    labels, medoids = pam_clustering(x, 2)

    assert (labels == [[2], [2], [2], [3], [3], [3]]).all()
    assert medoids == [2, 3] or medoids == [3, 2]


def test_pam_simple_three_points():
    x = np.array([[1, 0], [1, 1], [1, 2], [50, 49], [50, 50], [50, 51]])

    labels, medoids = pam_clustering(x, k=2)

    print(labels)

    assert sorted(medoids) == [1, 4]
    assert (labels == [[1], [1], [1], [4], [4], [4]]).all()
