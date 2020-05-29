import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Callable, Sequence

DIST_FUNCTION = Callable[[Sequence, Sequence], float]


def generate_random_uniform_points_clouds(
    low_boundaries: List[List[float]],
    high_boundaries: List[List[float]],
    points: List[int],
    dimensions: int,
) -> np.ndarray:
    """
    Generate clouds of points using uniform distribution.

    Args:
        low_boundaries: List of low boundaries. Each element of list is list of ints,
        each represent low boundary for dimension, example [[0, 10]] - one points cloud,
        `x` low boundary is 0, `y` low boundary is 10.
        high_boundaries: List of high boundaries. Each element of list is list of ints,
        each represent high boundary for dimension, example [[10, 20]] - one points cloud,
        `x` high boundary is 10, `y` high boundary is 20.
        points: Number of points for each cloud.
        dimensions: Points dimensions.

    Returns:
        Numpy array with points clouds, last column represent label - original cloud ID.
    """

    result_array = np.empty(shape=[0, dimensions + 1])

    for cloud_idx in range(len(low_boundaries)):
        cloud = np.random.uniform(
            low_boundaries[cloud_idx],
            high_boundaries[cloud_idx],
            [points[cloud_idx], dimensions],
        )
        cloud_label = np.ones(shape=[cloud.shape[0], 1]) * cloud_idx
        cloud = np.append(cloud, cloud_label, axis=1)
        result_array = np.concatenate((result_array, cloud))
    return result_array


def generate_random_normal_points_clouds(
    means: List[List[float]],
    widths: List[List[float]],
    points: List[int],
    dimensions: int,
):
    """
    Generate clouds of points using normal distribution.

    Args:
        means: List of means for each cloud. Each element represent means for one cloud,
        each element of such list represent mean of normal distribution
        for dimension, example [[0, 10]] - one points cloud,
        `x` mean is 0, `y` mean is 10.
        widths: List of widths for each cloud. Each element represent widths for one cloud,
        each element of such list represent width of normal distribution
        for dimension, example [[10, 20]] - one points cloud,
        `x` width is 10, `y` width is 20.
        points: Number of points for each cloud.
        dimensions: Points dimensions.

    Returns:
        Numpy array with points clouds, last column represent label - original cloud ID.
    """
    result_array = np.empty(shape=[0, dimensions + 1])

    for cloud_idx in range(len(means)):
        cloud = np.random.normal(
            means[cloud_idx], widths[cloud_idx], [points[cloud_idx], dimensions]
        )
        cloud_label = np.ones(shape=[cloud.shape[0], 1]) * cloud_idx
        cloud = np.append(cloud, cloud_label, axis=1)
        result_array = np.concatenate((result_array, cloud))
    return result_array


def plot_points(x: np.ndarray, first_idx: int = 0, second_idx: int = 1) -> None:
    """
    Plot ``x``, last column represent ID.

    Returns:
        x: Numpy data matrix.
        first_idx: Index of column to plot as x axis.
        second_idx: Index of column to plot as y axis.
    """

    fig, ax = plt.subplots()
    for cloud in np.unique(x[:, -1]):
        ax.scatter(x[x[:, -1] == cloud, first_idx], x[x[:, -1] == cloud, second_idx])
    plt.show()


def count_silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Rotate ``labels`` to [1, rows], if given shape is [rows, 1]
    and calculate silhouette score.

    Args:
        x: Numpy data matrix.
        labels: Labels for probes in ``x``.

    Returns:
        Silhouette score for data.
    """

    if labels.shape == (x.shape[0], 1):
        l = labels.T
        l = l[0, :]
    else:
        l = labels

    return silhouette_score(x, l)


def count_silhouette_score_for_samples(x: np.ndarray, labels: np.ndarray):
    """
    Rotate ``labels`` to [1, rows], if given shape is [rows, 1]
    and calculate silhouette score.

    Args:
        x: Numpy data matrix.
        labels: Labels for probes in ``x``.

    Returns:
        Silhouette score for all samples.
    """

    if labels.shape == (x.shape[0], 1):
        l = labels.T
        l = l[0, :]
    else:
        l = labels

    return silhouette_samples(x, l)


def count_dissimilarity_for_samples(
    x: np.ndarray,
    labels: np.ndarray,
    dist_function: DIST_FUNCTION = lambda a, b: np.linalg.norm(a - b),
) -> float:
    if x.shape[0] != labels.shape[0]:
        raise Exception(
            f"Data and labels have different size! Data: {x.shape}, labels: {labels.shape}"
        )

    dissimilarity = 0.0
    for i in range(x.shape[0]):
        sample_medoid = x[labels[i, 0], :]
        dissimilarity += dist_function(x[i, :], sample_medoid)

    return dissimilarity


def count_average_dissimilarity_for_samples(
    x: np.ndarray,
    labels: np.ndarray,
    dist_function: DIST_FUNCTION = lambda a, b: np.linalg.norm(a - b),
) -> float:
    dissimilarity = count_dissimilarity_for_samples(x, labels, dist_function)
    return dissimilarity / x.shape[0]
