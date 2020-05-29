from mpl_toolkits import basemap
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import (
    count_silhouette_score,
    count_avg_silhouette_score_for_cls,
    count_silhouette_score_for_samples,
)
import numpy as np
from src.pam_clustering import pam_clustering
import os
import seaborn as sns
import time


def map_points_clustering(
    data_path: str, map_margin, area_threshold, report_dir_pref
) -> None:
    # load data
    data = pd.read_csv(data_path, sep=";")
    data = np.array(data)

    for k in range(2, 20):
        dir_name = f"{k}_{report_dir_pref}"
        os.mkdir(dir_name)

        start_time = time.time()
        # process clustering
        labels, medoids = pam_clustering(data, k)
        execution_time = time.time() - start_time

        silhouette_scores = []
        for m in medoids:
            silhouette_scores.append(
                count_avg_silhouette_score_for_cls(data, labels, m)
            )
        full_silhouette_score = count_silhouette_score(data, labels)

        # get longitudes and latitudes
        longitude = data[:, 0]
        latitude = data[:, 1]

        # get maximal an minimal values
        max_long = max(longitude)
        min_long = min(longitude)
        max_lati = max(latitude)
        min_lati = min(latitude)

        # init plot
        fig, ax = plt.subplots()

        # create map
        map_fragment = basemap.Basemap(
            projection="merc",
            llcrnrlon=min_long - map_margin,
            llcrnrlat=min_lati - map_margin,
            urcrnrlon=max_long + map_margin,
            urcrnrlat=max_lati + map_margin,
            resolution="h",
            area_thresh=area_threshold,
        )

        map_fragment.drawcoastlines()
        map_fragment.drawcountries()
        map_fragment.drawstates()

        # convert position to map position
        lons, lats = map_fragment(longitude, latitude)

        # draw points clouds
        for cloud in np.unique(labels[:, -1]):
            ax.scatter(lons[labels[:, -1] == cloud], lats[labels[:, -1] == cloud], s=2)
        plt.savefig(fname=os.path.join(dir_name, "map.png"), dpi=300)
        plt.clf()

        with open(os.path.join(dir_name, "report.txt"), "w") as report_f:
            report_f.write(f"Medoids: {medoids}\n")
            report_f.write(f"Silhouette scores: {silhouette_scores}\n")
            report_f.write(f"Full silhouette score: {full_silhouette_score}\n")
            report_f.write(f"Clustering execution: {execution_time}")

            sns.distplot(silhouette_scores, kde=False, bins=20)
            plt.xlabel("Wartość silhouette")
            plt.ylabel("Liczba grup")
            plt.savefig(fname=os.path.join(dir_name, "hist.png"), dpi=300)
            plt.clf()


if __name__ == "__main__":
    SHOP_DATA = "../data/shops.csv"
    SHOP_MAP_MARGIN = 0.5
    SHOP_MAP_WATER_AREA_THRESHOLD = 200
    SHOP_MAP_FNAME = "shops.png"

    map_points_clustering(
        SHOP_DATA, SHOP_MAP_MARGIN, SHOP_MAP_WATER_AREA_THRESHOLD, "shop"
    )
    """
    plot_points(BAR_DATA, BAR_MAP_MARGIN, BAR_MAP_WATER_AREA_THRESHOLD, BAR_MAP_FNAME)
    plot_points(
        CHURCH_DATA,
        CHURCH_MAP_MARGIN,
        CHURCH_MAP_WATER_AREA_THRESHOLD,
        CHURCH_MAP_FNAME,
    )
    plot_points(
        SHOP_DATA, SHOP_MAP_MARGIN, SHOP_MAP_WATER_AREA_THRESHOLD, SHOP_MAP_FNAME
    )
    """
