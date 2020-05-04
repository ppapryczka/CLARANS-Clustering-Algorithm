import numpy as np
from typing import List, Sequence
from src.utils import generate_random_uniform_points_clouds
from random import sample
import math


def distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    """
    Count euclidean distance between ``point1`` and ``point2``.
    Args:
        point1: First point as a (x, y).
        point2: Second point as a (x, y).
    Returns:
        Distance as a float number.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def case_1(points, points_label, medoids, j, i, h) -> float:
    cost = 0

    if j not in [i, h] and points_label[j, 0] == i:
        # find second best distance
        second_best_distance = None

        for m in medoids:
            if m != i: # j belongs to i - we want find second best
                dist = distance(points[m, :], points[j, :])
                if second_best_distance is None or dist < second_best_distance:
                    min_distance = dist

        distance_to_h_medoid = distance(points[j, :], points[h, :])

        if distance_to_h_medoid >= second_best_distance:
            print("aaa")
            cost = second_best_distance - distance(points[j, :], points[i, :])
    return cost


def case_2(points, points_label, medoids, j, i, h) -> float:
    cost = 0

    if j not in [i, h] and points_label[j, 0] == i:
        # find second best distance
        second_best_distance = None

        for m in medoids:
            if m != i:  # j belongs to i - we want find second best
                dist = distance(points[m, :], points[j, :])
                if second_best_distance is None or dist < second_best_distance:
                    min_distance = dist

        distance_to_h_medoid = distance(points[j, :], points[h, :])

        if distance_to_h_medoid < second_best_distance:
            print("aaa")
            cost = distance_to_h_medoid - distance(points[j, :], points[i, :])
    return cost


def case_3(points, points_label, medoids, j, i, h) -> float:
    cost = 0
    if j not in [i, h] and points_label[j, 0] != i:
        distance_to_h_medoid = distance(points[j, :], points[h, :])
        distance_to_current_medoid = distance(points[j, :], points[int(points_label[j, 0]), :])

        if distance_to_current_medoid <= distance_to_h_medoid:
            print("aaa")
            cost = 0
    return cost


def case_4(points, points_label, medoids, j, i, h) -> float:
    cost = 0

    if j not in [i, h] and points_label[j, 0] != i:
        distance_to_h_medoid = distance(points[j, :], points[h, :])
        distance_to_current_medoid = distance(points[j, :], points[int(points_label[j, 0]), :])

        if distance_to_current_medoid > distance_to_h_medoid:
            print("aaa")
            cost = distance_to_h_medoid - distance_to_current_medoid

    return cost

def pam_clustering(points: np.ndarray, k: int, m: List = None) -> List:
    points_label = np.zeros(shape=[points.shape[0], 1])

    medoids: List
    points_len: int = points.shape[0]

    if m is not None:
        medoids = m
    else:
        medoids = sample(range(points_len), k)

    iteration = 0
    while 1:
        # produce current labels
        for idx, row in enumerate(points):
            medoids_iter = iter(medoids)

            # "zero" iteration
            current_medoid = next(medoids_iter)
            min_distance = distance(points[current_medoid, :], row)
            points_label[idx, 0] = current_medoid

            for current_medoid in medoids_iter:
                dist = distance(points[current_medoid, :], row)
                if dist < min_distance:
                    min_distance = dist
                    points_label[idx, 0] = current_medoid

        costs_dict = {}

        for i in medoids:
            for h in range(points_len):
                if h not in medoids:
                    costs_dict[(i, h)] = 0

                    for j in range(points_len):

                        case_1_cost = case_1(points, points_label, medoids, i, h, j)
                        case_2_cost = case_2(points, points_label, medoids, i, h, j)
                        case_3_cost = case_3(points, points_label, medoids, i, h, j)
                        case_4_cost = case_4(points, points_label, medoids, i, h, j)
                        print(case_1_cost, case_2_cost, case_3_cost, case_4_cost)
                        costs_dict[(i, h)] += sum([case_1_cost, case_2_cost, case_3_cost, case_4_cost])
        print(costs_dict)
        iteration = iteration + 1
        break
    print(points_label)
if __name__ == "__main__":
    points = generate_random_uniform_points_clouds([[0, 10], [10, 20], [10, 10]], [[5, 20], [21, 31], [20, 20]],
                                                   [5, 5, 5], 2)
    pam_clustering(points[:, 0:points.shape[1]-1],  2)
