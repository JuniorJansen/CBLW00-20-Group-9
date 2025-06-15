import os
import geopandas as gpd
import pandas as pd

folder = "boundaries/london"
shapefiles = [f for f in os.listdir(folder) if f.endswith('.shp')]

gdfs = []
for shp in shapefiles:
    full_path = os.path.join(folder, shp)
    gdf = gpd.read_file(full_path)
    gdfs.append(gdf)

# Combine all into one GeoDataFrame
gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Output path (Shapefile must be a directory without file extension)
output_folder = "boundaries/london_lsoa_shapefile"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "london_lsoa.shp")

# Save as ESRI Shapefile
gdf_all.to_file(output_path, driver="ESRI Shapefile")
