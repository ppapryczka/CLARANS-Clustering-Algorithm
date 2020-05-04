import numpy as np
import matplotlib.pyplot as plt
from typing import List


def generate_random_uniform_points_clouds(
    low_boundaries: List[List[float]],
    high_boundaries: List[List[float]],
    points: List[int],
    dimensions: int,
) -> np.ndarray:
    result_array = np.empty(shape=[0, dimensions+1])

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
    result_array = np.empty(shape=[0, dimensions+1])

    for cloud_idx in range(len(means)):
        cloud = np.random.normal(
            means[cloud_idx], widths[cloud_idx], [points[cloud_idx], dimensions]
        )
        cloud_label = np.ones(shape=[cloud.shape[0], 1]) * cloud_idx
        cloud = np.append(cloud, cloud_label, axis=1)
        result_array = np.concatenate((result_array, cloud))
    return result_array


def plot_points(points: np.ndarray) -> None:
    fig, ax = plt.subplots()
    for cloud in np.unique(points[:, -1]):
        ax.scatter(points[points[:, -1] == cloud, 0], points[points[:, -1] == cloud, 1])
    plt.show()


if __name__ == "__main__":
    points = generate_random_uniform_points_clouds([[0, 10], [10, 20], [10, 10]], [[5, 20], [21, 31], [20, 20]], [100, 100, 100], 2)

    plot_points(points)