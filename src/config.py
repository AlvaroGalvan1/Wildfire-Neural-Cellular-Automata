# src/config.py

# Study area coordinates (validated Colorado fire zone)
# Format: (minx, miny, maxx, maxy) in a projected CRS like Web Mercator (EPSG:3857)
# These are approximate coordinates for the original bbox.
STUDY_AREA_COORDS = (-11733136.85, 4879292.05, -11715331.01, 4890659.13)

# Target resolution for the final grid in meters
TARGET_RESOLUTION = 30