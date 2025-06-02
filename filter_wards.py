import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Load shapefile
gdf = gpd.read_file("boundaries/WD_MAY_2024_UK_BGC.shp")

# Ensure LAT and LONG are numeric
gdf["LAT"] = pd.to_numeric(gdf["LAT"], errors="coerce")
gdf["LONG"] = pd.to_numeric(gdf["LONG"], errors="coerce")

# Bounding box for Greater London
lat_min, lat_max = 51.3, 51.7
lon_min, lon_max = -0.5, 0.3

# Filter only by coordinates
london_wards = gdf[
    (gdf["LAT"] >= lat_min) & (gdf["LAT"] <= lat_max) &
    (gdf["LONG"] >= lon_min) & (gdf["LONG"] <= lon_max)
].copy()

print(f"✅ Filtered wards count (bounding box only): {len(london_wards)}")

# Save filtered shapefile
output_path = "boundaries/London_Wards_2024.shp"
london_wards.to_file(output_path)
print(f"✅ Filtered shapefile saved to: {output_path}")

# Plot
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.1, label="All UK Wards")
london_wards.plot(ax=ax, color="cornflowerblue", edgecolor="black", linewidth=0.4, label="London Wards")

ax.set_title("2024 Greater London Wards (Filtered by Coordinates Only)")
ax.axis("off")
plt.legend()
plt.show()
