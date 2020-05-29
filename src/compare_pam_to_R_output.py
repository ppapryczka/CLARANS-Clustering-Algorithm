import os
import numpy as np
from src.utils import (
    generate_random_uniform_points_clouds,
    plot_points,
    generate_random_normal_points_clouds,
    count_silhouette_score,
)
import pandas as pd
from src.pam_clustering import pam_clustering
from scipy.spatial.distance import euclidean
from typing import Tuple

R_SCRIPT_PROGRAM: str = "Rscript"
PAM_R_SCRIPT_NAME: str = "../r_src/pam_test.R"
DEFAULT_CSV_DATA_NAME: str = "test_data.csv"
DEFAULT_MEDOIDS_CSV: str = "medoids.csv"
DEFAULT_LABELS_CSV: str = "labels.csv"


def compare_pam_results(labels1, medoids1, labels2, medoids2) -> bool:
    if sorted(medoids1) != sorted(medoids2):
        return False

    if list(labels1) != list(labels2):
        return False

    return True


def label_r_clustering(labels, medoids):
    for m in medoids:
        value = labels[int(m)]
        labels[labels[:] == value] = int(m)
    return labels


def run_r_pam_script(x: np.ndarray, medoids_number: int) -> None:
    # save csv file
    pd.DataFrame(x).to_csv(DEFAULT_CSV_DATA_NAME, header=False, index=False)

    # run R script
    r_script_output = os.popen(
        f"{R_SCRIPT_PROGRAM} {PAM_R_SCRIPT_NAME} {DEFAULT_CSV_DATA_NAME} {medoids_number}"
    )

    output = r_script_output.read()
    if output != "":
        raise Exception(f"R script error! Full r script output:\n {output}")


def load_r_script_result() -> Tuple[np.ndarray, np.ndarray]:
    medoids = np.genfromtxt(fname=DEFAULT_MEDOIDS_CSV, skip_header=1)
    medoids = medoids - 1
    labels = np.genfromtxt(fname=DEFAULT_LABELS_CSV, skip_header=1)
    labels = labels - 1

    return labels, medoids


if __name__ == "__main__":

    for _ in range(100):
        try:
            points = generate_random_uniform_points_clouds(
                [[0, 0], [20, 20]], [[10, 10], [30, 30]], [10, 10], 2
            )

            run_r_pam_script(points[:, 0 : points.shape[1] - 1], 2)

            labels_R, medoids_R = load_r_script_result()
            labels_R = label_r_clustering(labels_R, medoids_R)
            labels_PY, medoids_PY = pam_clustering(
                points[:, 0 : points.shape[1] - 1], 2, dist_function=euclidean
            )

            if compare_pam_results(labels_PY.T[0, :], medoids_PY, labels_R, medoids_R):
                pass
                # print("Correct")
            else:
                print(medoids_R, medoids_PY)
                print("*** FAILED")
        finally:
            os.remove(DEFAULT_CSV_DATA_NAME)
            os.remove(DEFAULT_MEDOIDS_CSV)
            os.remove(DEFAULT_LABELS_CSV)
