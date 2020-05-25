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
import copy
from scipy.spatial.distance import euclidean


R_SCRIPT_PROGRAM = "Rscript"
PAM_R_SCRIPT_NAME = "../r_src/pam_test.R"
DEFAULT_CSV_DATA_NAME = "test_data.csv"
DEFAULT_CSV_DATA_PATH = ".."


def compare_r_output_to_python_output() -> None:
    pass


def parse_r_script_output(r_script_output: str):
    lines = r_script_output.split("\n")

    if len(lines) != 3 and lines[2] != "":
        raise Exception(f"R script error! Full output:\n {r_script_output}")

    medoids = lines[0].split()[1:]
    medoids = [int(i) - 1 for i in medoids]

    labels = lines[1].split()[1:]
    labels = [int(i) - 1 for i in labels]

    return labels, medoids


def run_r_pam_script(x: np.ndarray, medoids_number: int) -> str:
    # get path to csv file
    csv_file_path: str = os.path.join(DEFAULT_CSV_DATA_PATH, DEFAULT_CSV_DATA_NAME)

    # save csv file
    pd.DataFrame(x).to_csv(csv_file_path, header=False, index=False)

    # run R script
    r_script_output = os.popen(
        f"{R_SCRIPT_PROGRAM} {PAM_R_SCRIPT_NAME} {csv_file_path} {medoids_number}"
    )

    # return output
    return r_script_output.read()


if __name__ == "__main__":
    """
    while 1:
        points = generate_random_uniform_points_clouds([[0, 0], [20, 20]], [[10, 10], [30, 30]], [10, 10], 2)
        #print(points)

        output = run_r_pam_script(points[:, 0:points.shape[1]-1], 2)

        labels_R, medoids_R = parse_r_script_output(output)
        labels_PY, medoids_PY = pam_clustering(points[:, 0:points.shape[1]-1], 2, dist_function=euclidean)

        print(medoids_R, medoids_PY)
        #print(labels_R )
        #print(labels_PY.T)

        #plot_points(points)

        points_2 = copy.copy(points)

        for m in medoids_R:
            points_2[m, -1] = -1

        #plot_points(points_2)

        for m in medoids_PY:
            points[m, -1] = -1

        #plot_points(points)

        #r_script_output = os.popen("Rscript ../r_src/pam_test.R ../test_data.csv 2")
    
        print("We are on a mission from God!")
    """

    points = generate_random_uniform_points_clouds(
        [[0, 0], [20, 20]], [[10, 10], [30, 30]], [10, 10], 2
    )
    labels_PY, medoids_PY = pam_clustering(
        points[:, 0 : points.shape[1] - 1], 2, dist_function=euclidean
    )
    print(count_silhouette_score(points[:, 0 : points.shape[1] - 1], labels_PY))
