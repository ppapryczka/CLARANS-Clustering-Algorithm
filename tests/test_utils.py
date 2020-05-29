from src.utils import (
    count_silhouette_score,
    generate_random_uniform_points_clouds,
    generate_random_normal_points_clouds,
    count_dissimilarity_for_samples,
)
import numpy as np


def test_count_generate_random_uniform_points_clouds_check_labels_and_elements_number():
    points = generate_random_uniform_points_clouds(
        [[0, 0], [5, 5]], [[2, 2], [7, 7]], [10, 10], 2
    )

    labels = np.unique(points[:, -1])
    assert sorted(labels) == [0, 1]
    assert points.shape == (20, 3)


def test_count_generate_random_uniform_points_clouds_check_elements_number_for_labels():
    points = generate_random_uniform_points_clouds(
        [[0, 0], [5, 5]], [[2, 2], [7, 7]], [5, 25], 2
    )

    labels = np.unique(points[:, -1])

    labels = sorted(labels)

    assert points.shape == (30, 3)
    assert points[points[:, -1] == labels[0], :].shape == (5, 3)
    assert points[points[:, -1] == labels[1], :].shape == (25, 3)


def test_count_generate_random_uniform_points_clouds_check_if_correct_values():
    points = generate_random_uniform_points_clouds([[0, 5]], [[2, 7]], [5], 2)

    for p in range(points.shape[0]):
        assert 0 <= points[p, 0] <= 2
        assert 5 <= points[p, 1] <= 7


def test_count_generate_random_normal_points_clouds_check_labels_and_elements_number():
    points = generate_random_normal_points_clouds(
        [[0, 0], [5, 5]], [[2, 2], [7, 7]], [10, 10], 2
    )

    labels = np.unique(points[:, -1])
    assert sorted(labels) == [0, 1]
    assert points.shape == (20, 3)


def test_count_generate_random_normal_points_clouds_check_elements_number_for_labels():
    points = generate_random_normal_points_clouds(
        [[0, 0], [5, 5]], [[2, 2], [7, 7]], [5, 25], 2
    )

    labels = np.unique(points[:, -1])

    labels = sorted(labels)

    assert points.shape == (30, 3)
    assert points[points[:, -1] == labels[0], :].shape == (5, 3)
    assert points[points[:, -1] == labels[1], :].shape == (25, 3)


def test_count_dissimilarity_for_samples_simple_test():
    data = np.array([[1, 1], [1, 2]])
    labels = np.array([[1], [1]])
    dissimilarity = count_dissimilarity_for_samples(data, labels)

    assert dissimilarity == 1.0
