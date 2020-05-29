from src.pam_clustering import pam_clustering, count_labels_pam_clustering
import numpy as np
from typing import Union, List, Tuple
from src.utils import DIST_FUNCTION, count_average_dissimilarity_for_samples
from random import sample

DEFAULT_ITERATIONS: int = 5
DEFAULT_SAMPLE_SIZE: int = 40


def clara_clustering(
    x: np.ndarray,
    k: int,
    m: Union[List, None] = None,
    dist_function: DIST_FUNCTION = lambda a, b: np.linalg.norm(a - b),
    iterations: int = DEFAULT_ITERATIONS,
    sample_size: Union[int, None] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Use CLARA algorithm to cluster ``x``. Start algorithm with
    ``m`` medoids if given, else generate random.

    Args:
        x: Numpy data matrix.
        k: Number of clusters.
        m: Starting medoids, optional.
        dist_function: Function to count distance between points.
        iterations: Number of iteration of algorithm.
        sample_size: Number of samples draw outside of data.

    Returns:
        Label for each row given in ``x`` and medoids list.
    """
    # allocate column of labels for x
    x_label: np.ndarray = np.zeros(shape=[x.shape[0], 1])
    # allocate list of medoids
    medoids: List[int]
    # get number of rows for data x
    points_len: int = x.shape[0]

    if m is not None:  # get medoids from argument
        medoids = m
    else:  # generate random, unique indexes as medoids
        medoids = sample(range(points_len), k)

    # get historical sample size
    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZE + 2 * k

    current_average_dissimilarity = count_average_dissimilarity_for_samples(x, x_label)

    x_label = count_labels_pam_clustering(x, x_label, medoids)

    for i in range(iterations):
        random_samples_indexes = sample(range(points_len), sample_size)
        random_samples_indexes = sorted(random_samples_indexes)
        random_samples = x[random_samples_indexes, :]

        _, sample_medoids = pam_clustering(
            random_samples, k, dist_function=dist_function
        )

        # allocate column of labels for x and count labels
        new_x_label: np.ndarray = np.zeros(shape=[x.shape[0], 1])
        new_x_label = count_labels_pam_clustering(
            x, new_x_label, sample_medoids, dist_function
        )

        # count dissimilarity
        new_average_dissimilarity = count_average_dissimilarity_for_samples(
            x, new_x_label, dist_function
        )

        if current_average_dissimilarity > new_average_dissimilarity:
            medoids = sample_medoids
            current_average_dissimilarity = new_average_dissimilarity
            x_label = new_x_label

    return x_label, medoids
