import numpy as np
from typing import List, Sequence, Tuple, Union, Callable
from src.utils import plot_points
from random import sample
from timeit import default_timer as timer
from scipy.spatial.distance import euclidean

DIST_FUNCTION = Callable[[Sequence, Sequence], float]


def pam_clustering(
    x: np.ndarray,
    k: int,
    m: Union[List, None] = None,
    dist_function: DIST_FUNCTION = lambda a, b: np.linalg.norm(a - b),
) -> Tuple[np.ndarray, List[int]]:
    """
    Use PAM algorithm to cluster ``points``. Start algorithm with
    ``m`` medoids if given, else generate random.

    Args:
        x: Numpy data matrix.
        k: Number of clusters.
        m: Starting medoids, optional.
        dist_function: Function to count distance between points.

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

    # produce init labels
    x_label = count_labels_pam_clustering(x, x_label, medoids, dist_function)

    # infinite loop
    while 1:
        # init cost dict
        costs_dict = {}

        # search for better medoids
        for medoid_idx, i in enumerate(medoids):
            for h in range(points_len):
                if h not in medoids:
                    # print(i, h) # for debug

                    # init cost with 0
                    costs_dict[(i, h, medoid_idx)] = 0

                    for j in range(points_len):
                        if j not in [h] + medoids:
                            # print(i, h, j) # for debug

                            # count cost for row j if is in group i
                            if x_label[j, 0] == i:
                                costs_dict[
                                    (i, h, medoid_idx)
                                ] += count_cost_for_point_from_medoid_i(
                                    x, medoids, j, i, h, dist_function
                                )
                            # count cost for row j if is in other group than i
                            else:
                                costs_dict[
                                    (i, h, medoid_idx)
                                ] += count_cost_for_point_from_medoid_different_than_i(
                                    x, x_label, j, h, dist_function
                                )

        # find minimal cost
        min_cost = min(costs_dict, key=costs_dict.get)
        # print(min_cost, costs_dict[min_cost], medoids) # for debug

        # if minimal cost is lower than 0 there is improvement
        if costs_dict[min_cost] < 0:
            medoids[min_cost[2]] = min_cost[1]
            x_label = count_labels_pam_clustering(x, x_label, medoids, dist_function)
        else:
            break

    return x_label, medoids


def count_labels_pam_clustering(
    x: np.ndarray,
    x_label: np.ndarray,
    medoids: List[int],
    dist_function: DIST_FUNCTION = lambda a, b: np.linalg.norm(a - b),
) -> np.ndarray:
    """
    Count labels for ``x`` using ``medoids`` and save them to ``x_label``.

    Args:
        x: Numpy matrix data.
        x_label: Column of labels for ``x`` data.
        medoids: List of medoids as indexes of rows in ``x``.
        dist_function: Function to count distance between points.

    Returns:
        New labels as column.
    """

    for idx, row in enumerate(x):
        medoids_iter = iter(medoids)

        # "zero" iteration
        current_medoid = next(medoids_iter)
        min_distance = dist_function(x[current_medoid, :], row)
        x_label[idx, 0] = current_medoid

        for current_medoid in medoids_iter:
            dist = dist_function(x[current_medoid, :], row)
            if dist < min_distance:
                min_distance = dist
                x_label[idx, 0] = current_medoid
    return x_label


def count_cost_for_point_from_medoid_i(
    x: np.ndarray, medoids: List, j: int, i: int, h: int, dist_function: DIST_FUNCTION
) -> float:
    """
    Count cost of replacement ``i`` by ``h`` for as medoid for row ``j``.
    Row ``j`` must belong to meoid ``i``.

    Args:
        x: Numpy data matrix.
        medoids: List of medoids as indexes of rows in ``x``.
        j: Row to count cost of replacement ``i`` by ``h``.
        i: Current medoid.
        h: Possible medoid.
        dist_function: Function to count distance between points.

    Returns:
         Cost of replacement.
    """

    # find second best distance
    second_best_distance = None

    for m in medoids:
        if m != i:  # j belongs to i - we want find second best distance
            dist = dist_function(x[m, :], x[j, :])
            if second_best_distance is None or dist < second_best_distance:
                second_best_distance = dist

    # compute distance between j and h points
    distance_to_h_medoid = dist_function(x[j, :], x[h, :])

    # count cost and return
    if distance_to_h_medoid >= second_best_distance:
        return second_best_distance - dist_function(x[j, :], x[i, :])
    else:
        return distance_to_h_medoid - dist_function(x[j, :], x[i, :])


def count_cost_for_point_from_medoid_different_than_i(
    x: np.ndarray, x_label: np.ndarray, j: int, h: int, dist_function: DIST_FUNCTION
) -> float:
    """
     Args:
        x: Numpy data matrix.
        x_label:
        j: Row to count cost of replacement ``i`` by ``h``.
        h: Possible medoid.
        dist_function: Function to count distance between points.

    Returns:
         Cost of replacement.
    """
    # compute distance to h as medoid
    distance_to_h_medoid = dist_function(x[j, :], x[h, :])
    # compute distance to current medoid
    distance_to_current_medoid = dist_function(x[j, :], x[int(x_label[j, 0]), :])

    # count cost of replacement and return
    if distance_to_current_medoid <= distance_to_h_medoid:
        return 0
    else:
        return distance_to_h_medoid - distance_to_current_medoid


"""
#points = generate_random_uniform_points_clouds([[0, 10], [10, 20], [10, 10]], [[5, 20], [21, 31], [20, 20]],
#                                               [100, 100, 100], 2)
points = generate_random_uniform_points_clouds([[0, 10], [8, 10], [16, 10]], [[10, 20], [18, 20], [26, 20]],
                                               [100, 100, 100], 2)
# points = np.array([[4, 7, 2], [6, 5, 2], [6, 7, 2], [1, 1, 1], [1,  3, 1], [3, 1, 1]])

plot_points(points)

start = timer()
points_label, medoids = pam_clustering(points[:, 0:points.shape[1]-1],  3)
end = timer()
print(end - start)


#points_label = pam_clustering(points, k=2, m=[1, 2])
points_2 = points
#print(points_label)

points_2[:, -1] = points_label.T
for m in medoids:
    points_2[m, -1] = 5

plot_points(points_2)
"""
