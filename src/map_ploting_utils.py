from mpl_toolkits import basemap
import pandas as pd
import matplotlib.pyplot as plt


SHOP_DATA = "data/shops.csv"
SHOP_MAP_MARGIN = 0.5
SHOP_MAP_WATER_AREA_THRESHOLD = 200
SHOP_MAP_FNAME = "shops.png"

BAR_DATA = "data/bars.csv"
BAR_MAP_MARGIN = 1
BAR_MAP_WATER_AREA_THRESHOLD = 200
BAR_MAP_FNAME = "bars.png"

CHURCH_DATA = "data/churches.csv"
CHURCH_MAP_MARGIN = 0.5
CHURCH_MAP_WATER_AREA_THRESHOLD = 200
CHURCH_MAP_FNAME = "churches.png"


def plot_points(data_path: str, map_margin, area_threshold, fname) -> None:
    # load data
    data = pd.read_csv(data_path, sep=";")

    # get longitudes and latitudes
    longitude = data.iloc[:, 0].values
    latitude = data.iloc[:, 1].values

    # get maxmial an minimal values
    max_long = max(longitude)
    min_long = min(longitude)
    max_lati = max(latitude)
    min_lati = min(latitude)

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

    lons, lats = map_fragment(longitude, latitude)

    plt.scatter(lons, lats, marker="o", color="Red", s=1)
    plt.savefig(fname=fname, dpi=300)
    plt.clf()


if __name__ == "__main__":
    plot_points(BAR_DATA, BAR_MAP_MARGIN, BAR_MAP_WATER_AREA_THRESHOLD, BAR_MAP_FNAME)
    plot_points(
        CHURCH_DATA,
        CHURCH_MAP_MARGIN,
        CHURCH_MAP_WATER_AREA_THRESHOLD,
        CHURCH_MAP_FNAME,
    )
    plot_points(SHOP_DATA, SHOP_MAP_MARGIN, SHOP_MAP_WATER_AREA_THRESHOLD, SHOP_MAP_FNAME)
