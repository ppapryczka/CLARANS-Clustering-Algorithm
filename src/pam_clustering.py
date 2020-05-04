import numpy as np
from typing import List, Sequence
from src.utils import generate_random_uniform_points_clouds, plot_points
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

'''
def case_1(points, points_label, medoids, j, i, h) -> float:
    cost = 0

    if j not in [i, h] and points_label[j, 0] == i:
        print("jesteś tu 1?")
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
        print("jesteś tu2?")
        # find second best distance
        second_best_distance = None

        for m in medoids:
            if m != i:  # j belongs to i - we want find second best
                dist = distance(points[m, :], points[j, :])
                if second_best_distance is None or dist < second_best_distance:
                    min_distance = dist

        distance_to_h_medoid = distance(points[j, :], points[h, :])

        if distance_to_h_medoid < second_best_distance:
            cost = distance_to_h_medoid - distance(points[j, :], points[i, :])
            print("cost", cost)
    return cost
"""

def case_3(points, points_label, medoids, j, i, h) -> float:
    cost = 0
    if j not in [i, h] and points_label[j, 0] != i:
        distance_to_h_medoid = distance(points[j, :], points[h, :])
        distance_to_current_medoid = distance(points[j, :], points[int(points_label[j, 0]), :])

        if distance_to_current_medoid <= distance_to_h_medoid:
            cost = 0
    return cost


def case_4(points, points_label, medoids, j, i, h) -> float:
    cost = 0

    if j not in [i, h] and points_label[j, 0] != i:
        distance_to_h_medoid = distance(points[j, :], points[h, :])
        distance_to_current_medoid = distance(points[j, :], points[int(points_label[j, 0]), :])

        if distance_to_current_medoid > distance_to_h_medoid:
            cost = distance_to_h_medoid - distance_to_current_medoid
            print("cost", cost)

    return cost
'''


def case_1_2(points, points_label, medoids, j, i, h) -> float:
    cost = 0
    if points_label[j, 0] == i:
        #print("case 1_2")
        # find second best distance
        second_best_distance = None

        for m in medoids:
            if m != i:  # j belongs to i - we want find second best
                dist = distance(points[m, :], points[j, :])
                if second_best_distance is None or dist < second_best_distance:
                    second_best_distance = dist

        distance_to_h_medoid = distance(points[j, :], points[h, :])

        if distance_to_h_medoid >= second_best_distance:
            cost = second_best_distance - distance(points[j, :], points[i, :])
        else:
            cost = distance_to_h_medoid - distance(points[j, :], points[i, :])
    return cost


def case_3_4(points, points_label, medoids, j, i, h) -> float:
    cost = 0
    if points_label[j, 0] != i:
        #print("case 3_4")
        distance_to_h_medoid = distance(points[j, :], points[h, :])
        #print("j", j, "label", int(points_label[j, 0]))
        distance_to_current_medoid = distance(points[j, :], points[int(points_label[j, 0]), :])

        #print("distance_to_h_medoid",distance_to_h_medoid, "distance_to_current_medoid", distance_to_current_medoid)
        if distance_to_current_medoid <= distance_to_h_medoid:
            cost = 0
        else:
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
    while iteration < 10:
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
                    print(i, h)
                    costs_dict[(i, h)] = 0
                    for j in range(points_len):
                        if j not in [h] + medoids:
                            print(i, h, j)

                            costs_dict[(i, h)] += case_1_2(points, points_label, medoids, j, i, h)
                            costs_dict[(i, h)] += case_3_4(points, points_label, medoids, j, i, h)

        print(costs_dict, min(costs_dict, key=costs_dict.get))
        min_cost = min(costs_dict, key=costs_dict.get)

        if costs_dict[min_cost]<0:
            for idx, m in enumerate(medoids):
                if m == min_cost[0]:
                    medoids[idx] = min_cost[1]
        else:
            break
        iteration = iteration + 1

    return points_label

if __name__ == "__main__":
    points = generate_random_uniform_points_clouds([[0, 10], [10, 20], [10, 10]], [[5, 20], [21, 31], [20, 20]],
                                                   [10, 10, 10], 2)
    # points = np.array([[4, 7, 2], [6, 5, 2], [6, 7, 2], [1, 1, 1], [1,  3, 1], [3, 1, 1]])

    plot_points(points)
    points_label = pam_clustering(points[:, 0:points.shape[1]-1],  3)

    #points_label = pam_clustering(points, k=2, m=[1, 2])
    points_2 = points
    #print(points_label)

    points_2[:, -1] = points_label.T

    plot_points(points_2)
