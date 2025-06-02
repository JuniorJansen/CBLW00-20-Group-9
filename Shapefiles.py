import os
import geopandas as gpd
import pandas as pd

folder = ("boundaries/london")


shapefiles = [f for f in os.listdir(folder) if f.endswith('.shp')]

gdfs = []
for shp in shapefiles:
    full_path = os.path.join(folder, shp)
    gdf = gpd.read_file(full_path)
    gdfs.append(gdf)

gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

output_path ="boundaries/london_lsoa.geojson"

gdf_all.to_file(output_path, driver="GeoJSON")