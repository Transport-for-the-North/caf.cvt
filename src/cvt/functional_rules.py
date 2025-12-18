'''This script is for applying functional rules
to each hazard dataset in order to classify the raw data into actionable risk factors'''

### LOAD LIBRARIES
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from functools import reduce

from src.cvt.data_cleaning import clip_to_boundary, convert_point_to_grid, explode_to_polygons, read_boundary_path, \
    tfn_drought

### FILE PATHS
RAW_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "raw inputs"
MODEL_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model inputs"

INFRASTRUCTURE_RAW_IN = RAW_INPUT_PATH / "Infrastructure"

OTHER_RAW_IN = INFRASTRUCTURE_RAW_IN / "Other"

HAZARDS_MODEL_IN = MODEL_INPUT_PATH / "Hazards"

EXTREME_WEATHER_MODEL_IN = HAZARDS_MODEL_IN / "Extreme Weather"


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

def merge_on_key(dfs, grid, key):
    merged = reduce(lambda left, right: pd.merge(left, right, on=key), dfs)
    merged_df = pd.merge(merged, grid, on=key)
    merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=grid.crs)
    return merged_gdf

def calculate_risk_threshold(df, base_col, output_col, threshold, invert=False):
    for tp in ['c', 'f']:
        col_name = f'{base_col}_{tp}'
        out_name = f'{output_col}_{tp}'

        if invert:
            df[out_name] = np.where(df[col_name] > threshold, 0, -df[col_name])
        else:
            df[out_name] = np.where(df[col_name] < threshold, 0, df[col_name])

        return df

def calculate_composite_score(df, weights, output_col):
    # Compute composite scores
    w = pd.Series(weights, dtype=float)

    for tp in ['c', 'f']:
        df[f'{output_col}_{tp}'] = sum(
            df[f'{col}_{tp}'] * weight for col, weight in weights.items()
        )

    return df

def min_max_scaling_pair(gdf, pairs, feature_range=(0,100)):
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
        OTHER_RAW_IN / "TfN Boundary" / "Transport_for_the_north_boundary_2020_generalised.shp")

## HAZARDS

### EXTREME WEATHER

def extreme_weather_index():
    tfn_common_grid = gpd.read_file(MODEL_INPUT_PATH / "Other" / "TfN Common Grid" / "tfn_common_grid.shp")

    tfn_extreme_heat = extreme_heat_index(tfn_common_grid)
    tfn_extreme_cold = extreme_cold_index(tfn_common_grid)
    tfn_drought = drought_index()
    tfn_storm = storm_index()


#### EXTREME HEAT

def extreme_heat_index(common_grid):
    tfn_temp_max = pd.read_csv(
        EXTREME_WEATHER_MODEL_IN / "TfN Summer Max Temperature Change Projections" / "tfn_temp_max.csv")
    tfn_hsd = pd.read_csv(EXTREME_WEATHER_MODEL_IN / "TfN Hot Summer Days Projections" / "tfn_hot_days.csv")
    tfn_esd = pd.read_csv(EXTREME_WEATHER_MODEL_IN / "TfN Extreme Summer Days Projections" / "tfn_extr_days.csv")

    tfn_extreme_heat = merge_on_key([tfn_temp_max, tfn_hsd, tfn_esd], common_grid, 'grid_id')

    tfn_extreme_heat = calculate_risk_threshold(
        tfn_extreme_heat, 'tasmax_s', 'tasmax_risk', 30)

    tfn_extreme_heat = min_max_scaling_pair(
        tfn_extreme_heat,
        [('tasmax_risk_c', 'tasmax_risk_f'), ('hsd_c', 'hsd_future'),('esd_c', 'esd_future')]
    )

    tfn_extreme_heat = calculate_composite_score(
        tfn_extreme_heat, {'tasmax_risk': 0.5, 'hsd': 0.25,'esd': 0.25,}, 'heat_risk')

    tfn_extreme_heat = min_max_scaling_pair(tfn_extreme_heat, [('heat_risk_c', 'heat_risk_f')])

    tfn_extreme_heat = gpd.GeoDataFrame(tfn_extreme_heat, geometry='geometry')

    return tfn_extreme_heat

### EXTREME COLD

def extreme_cold_index(common_grid):
    tfn_temp_min = pd.read_csv(
        EXTREME_WEATHER_MODEL_IN / "TfN Winter Min Temperature Change Projections" / "tfn_temp_min.csv")
    tfn_frost = pd.read_csv(EXTREME_WEATHER_MODEL_IN / "TfN Frost Days Projections" / "tfn_frost_days.csv")
    tfn_icing = pd.read_csv(EXTREME_WEATHER_MODEL_IN / "TfN Icing Days Projections" / "tfn_ice_days.csv")

    tfn_extreme_cold = merge_on_key([tfn_temp_min, tfn_frost, tfn_icing], common_grid, 'grid_id')

    tfn_extreme_cold = calculate_risk_threshold(
        tfn_extreme_cold, 'tasmin_w', 'tasmin_risk', 0, invert=True)

    tfn_extreme_cold = min_max_scaling_pair(
        tfn_extreme_cold, [('tasmin_risk_c', 'tasmin_risk_f'),('frost_d_c', 'frost_d_f'),('ice_d_c', 'ice_d_f')])

    tfn_extreme_cold = calculate_composite_score(
        tfn_extreme_cold, {'tasmin_risk': 0.5,  'frost_d': 0.25,  'ice_d': 0.25}, 'cold_risk')

    tfn_extreme_cold = min_max_scaling_pair(tfn_extreme_cold, [('cold_risk_c', 'cold_risk_f')])

    tfn_extreme_cold = gpd.GeoDataFrame(tfn_extreme_cold, geometry='geometry')

    return tfn_extreme_cold

### DROUGHT

def drought_index():
    tfn_drought = gpd.read_file(EXTREME_WEATHER_MODEL_IN / "TfN Drought Severity Index" / "tfn_drought_index.shp")
    tfn_precip_sum = gpd.read_file(
        EXTREME_WEATHER_MODEL_IN / "TfN Summer Precipitation Change Projections" / "tfn_precip_sum.shp")

    tfn_drought = tfn_drought.to_crs("EPSG:27700")
    tfn_precip_sum = tfn_precip_sum.to_crs("EPSG:27700")

    tfn_drought_overlay = gpd.overlay(tfn_precip_sum, tfn_drought, how='union')
    tfn_drought_overlay = iterative_spatial_smoothing(tfn_drought_overlay,
                                                      ['pr_s_c', 'pr_s_f', 'dsi_c', 'dsi_f'])

    tfn_drought_overlay = explode_to_polygons(tfn_drought_overlay)
    tfn_drought_risk = tfn_drought_overlay[['dsi_c', 'dsi_f', 'pr_s_c', 'pr_s_f', 'geometry']]

    tfn_drought_risk = min_max_scaling_pair(tfn_drought_risk, [('dsi_c', 'dsi_f'), ('pr_s_c', 'pr_s_f')])

    # Reverse the polarity for precipitation
    tfn_drought_risk['pr_s_c'] = 100 - tfn_drought_risk['pr_s_c']
    tfn_drought_risk['pr_s_f'] = 100 - tfn_drought_risk['pr_s_f']

    tfn_drought_risk = calculate_composite_score(
        tfn_drought_risk, {'dsi': 0.75, 'pr_s': 0.25}, 'drought_risk')

    tfn_drought_risk = min_max_scaling_pair(tfn_drought_risk, [('drought_risk_c', 'drought_risk_f')])

    tfn_drought_risk = gpd.GeoDataFrame(tfn_drought_risk, geometry='geometry')

    return tfn_drought_risk

### STORMS

def storm_index():
    tfn_precip_win = gpd.read_file(
        EXTREME_WEATHER_MODEL_IN / "TfN Winter Precipitation Change Projections" / "tfn_precip_win.shp")
    tfn_rain_days = gpd.read_file(EXTREME_WEATHER_MODEL_IN / "TfN 10mm Rain Days 1991-2020" / "tfn_rain_days.shp")
    tfn_wind_spd = gpd.read_file(EXTREME_WEATHER_MODEL_IN / "TfN Wind Speed Projections" / "tfn_windspd.shp")
    tfn_wdr = gpd.read_file(EXTREME_WEATHER_MODEL_IN / "TfN Wind Driven Rain Index" / "tfn_wdr.shp")

    tfn_wind_spd = tfn_wind_spd.to_crs("EPSG:27700")
    tfn_rain_days = tfn_rain_days.to_crs("EPSG:27700")
    tfn_precip_win = tfn_precip_win.to_crs("EPSG:27700")
    tfn_wdr = tfn_wdr.to_crs("EPSG:27700")

    tfn_storm_overlay = gpd.overlay(tfn_wind_spd, tfn_rain_days, how='union')
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_precip_win, how='union')
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_wdr, how='union')

    tfn_storm_overlay = tfn_storm_overlay[['rain_days_c', 'rain_days_f','pr_w_c', 'pr_w_f','p99_c', 'p99_f','avg_excd_c',
                                           'avg_excd_f','wdr_c', 'wdr_f','geometry']]

    tfn_storm_overlay['area'] = tfn_storm_overlay.geometry.area
    threshold = tfn_storm_overlay['area'].median() * 0.01  # Threshold: 1% of median
    tfn_storm_overlay = tfn_storm_overlay[tfn_storm_overlay['area'] > threshold]  # Filter out tiny geometries

    tfn_storm_overlay = iterative_spatial_smoothing(
        tfn_storm_overlay,
        ['rain_days_c', 'rain_days_f', 'pr_w_c', 'pr_w_f', 'p99_c', 'p99_f','avg_excd_c', 'avg_excd_f', 'wdr_c', 'wdr_f'])

    tfn_storm_overlay = explode_to_polygons(tfn_storm_overlay)

    tfn_storm_risk = tfn_storm_overlay.drop(columns=['area'])

    tfn_storm_risk['wind_spd_risk_c'] = tfn_storm_risk['p99_c'].apply(wind_risk_scaled)
    tfn_storm_risk['wind_spd_risk_f'] = tfn_storm_risk['p99_f'].apply(wind_risk_scaled)

    tfn_storm_risk = min_max_scaling_pair(tfn_storm_risk,[
        ('wind_spd_risk_c', 'wind_spd_risk_f'),('pr_w_c', 'pr_w_f'),('avg_excd_c', 'avg_excd_f'),('wdr_c', 'wdr_f'),
        ('rain_days_c', 'rain_days_f'),])


    tfn_storm_risk = calculate_composite_score(
        tfn_storm_risk, {'wind_spd_risk': 0.3,'avg_excd': 0.2,'pr_w': 0.15,'rain_days': 0.15, 'wdr': 0.2})

    tfn_storm_risk = min_max_scaling_pair(tfn_storm_risk, [('storm_risk_c', 'storm_risk_f')])

    tfn_storm_risk = gpd.GeoDataFrame(tfn_storm_risk, geometry='geometry')

    return tfn_storm_risk


def wind_risk_scaled(speed_mps):
    if speed_mps < 13.4: # Below 30 mph
        return 0
    elif speed_mps <= 20.1: # between 30 and 45 mph
        return (speed_mps - 13.4) / (20.1 - 13.4) # Scale to 0 - 1
    else:
        return 1 + (speed_mps - 20.1) / (25 - 20.1) # Scale beyond 1
