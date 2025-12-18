'''This script is for applying functional rules
to each hazard dataset in order to classify the raw data into actionable risk factors'''

### LOAD LIBRARIES
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from src.cvt.data_cleaning import clip_to_boundary, convert_point_to_grid, explode_to_polygons, read_boundary_path

### FILE PATHS
RAW_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "raw inputs"
MODEL_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model inputs"

INFRASTRUCTURE_IN = RAW_INPUT_PATH / "Infrastructure"

OTHER_IN = INFRASTRUCTURE_IN / "Other"


### GENERAL FUNCTIONS

def spatial_smooth_zero_grids(gdf, variables):
    neighbours = gpd.sjoin(gdf, gdf, how='left',predicate='touches') # Find neighbouring grids

    # Calculate the average value of the neighbouring grids
    neighbours_avg = neighbours.groupby(neighbours.index)[[f"{var}_right" for var in variables]].mean()

    for var in variables:
        # Condition: variable is NA
        na_condition = (gdf[var].isna())

        # Fill with neighbour average
        gdf.loc[na_condition, var] = gdf.loc[na_condition].index.map(neighbours_avg[f"{var}_right"])

    return gdf

def iterative_spatial_smoothing(gdf, variables, max_iterations=10):
    prev_na_count = None

    for i in range(max_iterations):
        # Count current NA values
        current_na_count = gdf[variables].isna().sum().sum()

        # Stop if all filled
        if current_na_count == 0:
            print(f"All NA values filled after {i} iterations.")
            return gdf


        # Stop if no improvement
        if prev_na_count is not None and current_na_count == prev_na_count:
            print("No further improvement. Switching to nearest join.")
            break

        prev_na_count = current_na_count

        gdf = spatial_smooth_zero_grids(gdf, variables)


    # Fallback: nearest join for remaining NAs
    remaining_na = gdf[gdf[variables].isna().any(axis=1)]
    if not remaining_na.empty:
        nearest = gpd.sjoin_nearest(remaining_na, gdf.drop(remaining_na.index), how='left')

        # Calculate the average value of the neighbouring grids
        nearest_avg = nearest.groupby(nearest.index)[[f"{var}_right" for var in variables]].mean()
        for var in variables:
            na_condition = (gdf[var].isna())
            # Fill with neighbour average
            gdf.loc[na_condition, var] = gdf.loc[na_condition].index.map(nearest_avg[f"{var}_right"])

    return gdf

def create_grid(bounds, cell_size):
    '''Takes bounds and a cell size and returns a grid of the given size within the bounds'''
    xmin, ymin, xmax, ymax = bounds
    rows = int(np.ceil((ymax - ymin) / cell_size))
    cols = int(np.ceil((xmax - xmin) / cell_size))
    grid_cells = []
    for i in range(cols):
        for j in range(rows):
            x0 = xmin + i * cell_size
            y0 = ymin + j * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(box(x0, y0, x1, y1))
    return gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:27700")

def min_max_scaling_pair(gdf, pairs, feature_range=(0,1)):
    # Use min-max scaling on the column pairs
    for col_c, col_f in pairs:
        # Combine both columns into one array for global min/max
        combined_values = gdf[[col_c, col_f]].values.flatten().reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(combined_values)

        # Transform each column using the same scaler
        gdf[col_c] = scaler.transform(gdf[[col_c]].values)
        gdf[col_f] = scaler.transform(gdf[[col_f]].values)

    return gdf

def area_weighted_flood_assignment(grid, flood_gdf, risk_column):
    # Spatial join to find intersecting polygons
    flood_risk_join = gpd.sjoin(
        grid,
        flood_gdf[[risk_column, 'geometry']],
        how='left',
        predicate='intersects'
    )

    # Retrieve flood polygon geometry using index_right
    flood_risk_join = flood_risk_join.merge(
        flood_gdf[[risk_column, 'geometry']],
        left_on='index_right',
        right_index=True,
        suffixes=('', '_flood')
    )

    # Get geometry of each intersection between grid and flood polygon
    flood_risk_join['intersection'] = flood_risk_join.apply(
        lambda row: row['geometry'].intersection(row['geometry_flood']), axis=1
    )

    # Calculate area of each intersection
    flood_risk_join['area'] = flood_risk_join['intersection'].area

    # Compute area weighted average flood risk per grid cell
    weighted_avg_flood = flood_risk_join.groupby(flood_risk_join.index).apply(
        lambda group: (group[risk_column] * group['area']).sum() / group['area'].sum()
    )

    # Assign weighted average flood risk back to the original grid
    grid[risk_column] = weighted_avg_flood

    # Fill missing values with 0 (no risk)
    grid[risk_column] = grid[risk_column].fillna(0)

    return grid

def overlay_normalise(gdf1, gdf2, risk_cols, combined_risk_name, weights):
    # Overlay two gdf's
    gdf1 = gdf1.to_crs(gdf2.crs)
    composite_gdf = gpd.overlay(gdf2, gdf1, how="union")

    composite_gdf = explode_to_polygons(composite_gdf)

    # Fill NA values with 0, indicating no risk
    composite_gdf[risk_cols] = composite_gdf[risk_cols].fillna(0)

    # Normalise risk values
    scaler = MinMaxScaler(feature_range=(0,100))
    normalised_values = scaler.fit_transform(composite_gdf[risk_cols])

    # Compute composite risk score
    composite_gdf[combined_risk_name] = (
        normalised_values[:, 0] * weights[risk_cols[0]] +
        normalised_values[:, 1] * weights[risk_cols[1]]
    )

    return composite_gdf

# FUNCTIONAL RULES

def apply_functional_rules():
    tfn_boundary = read_boundary_path(
        OTHER_IN / "TfN Boundary" / "Transport_for_the_north_boundary_2020_generalised.shp")

## HAZARDS

### EXTREME WEATHER


