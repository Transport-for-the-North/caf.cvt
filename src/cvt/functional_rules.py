"""
Apply functional rules to hazard and impact datasets, and normalise, in order to classify into actionable risk factors
"""

### LOAD LIBRARIES
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from sklearn.preprocessing import MinMaxScaler
from functools import reduce

from data_cleaning import explode_to_polygons, clip_to_boundary, write_to_file


### GENERAL FUNCTIONS

def spatial_smooth_zero_grids(gdf, variables):
    """Applies spatial smoothing to GeoDataFrame on given variables"""
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
    """Iteratively applies spatial smoothing to GeoDataFrame on given variables"""
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
    """Takes bounds and a cell size and returns a grid of the given size within the bounds"""
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
    """Merges a list of dataframes into a single dataframe on a given common key, then merges onto a common grid"""
    merged = reduce(lambda left, right: pd.merge(left, right, on=key, how='outer'), dfs)
    merged_df = pd.merge(merged, grid, on=key, how='left', validate='one_to_many')
    merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=grid.crs)
    return merged_gdf

def calculate_risk_threshold(df, base_col, output_col, threshold, invert=False):
    """Calculates risk level of a given column based on a threshold"""
    for tp in ['c', 'f']:
        col_name = f'{base_col}_{tp}'
        out_name = f'{output_col}_{tp}'

        if invert:
            df[out_name] = np.where(df[col_name] > threshold, 0, -df[col_name])
        else:
            df[out_name] = np.where(df[col_name] < threshold, 0, df[col_name])

    return df

def calculate_composite_score(df, weights, output_col):
    """Calculates composite score given a dataframe with variables and corresponding weights"""
    # Compute composite scores
    w = pd.Series(weights, dtype=float)

    for tp in ['c', 'f']:
        df[f'{output_col}_{tp}'] = sum(
            df[f'{col}_{tp}'] * weight for col, weight in weights.items()
        )

    return df

def min_max_scaling_pair(gdf, pairs, feature_range=(0,100)):
    """Takes a list of paired variables, and normalises them between 0 and 100 using Min-Max scaling"""
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
    """Assigns flood risk to grid squares using an area-weighted average"""
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
    """Overlays two GeoDataFrames, then normalises and calculates a combined risk score"""
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

def filter_out_small_geometries(gdf, pct_of_median):
    """Filters out geometries that have an area less than a given percentage of the median area"""
    gdf['area'] = gdf.geometry.area
    threshold = gdf['area'].median() * pct_of_median # Threshold: 3.5% of median
    gdf = gdf[gdf['area'] > threshold] # Filter out tiny geometries
    gdf.drop(columns=['area'], inplace=True)
    return gdf

# FUNCTIONAL RULES

def apply_functional_rules(cfg):
    """Applies functional rules to model input data"""
    boundary = gpd.read_file(cfg.paths.boundary_path)

    #extreme_weather_index(cfg)
    flooding_index(cfg, boundary)
    #ground_stability_index(cfg)
    #coastal_erosion_index(cfg)

## HAZARDS

### EXTREME WEATHER

def extreme_weather_index(cfg):
    """Combines extreme heat, extreme cold, drought and storm indexes into a single index"""
    tfn_common_grid = gpd.read_file(cfg.paths.model_input / "Other" / "TfN Common Grid" / "tfn_common_grid.shp")

    tfn_extreme_heat = extreme_heat_index(cfg, tfn_common_grid)
    tfn_extreme_cold = extreme_cold_index(cfg, tfn_common_grid)
    tfn_drought = drought_index(cfg, tfn_common_grid)
    tfn_storm = storm_index(cfg, tfn_common_grid)


    tfn_extreme_weather_merge = pd.merge(
        tfn_extreme_heat[['grid_id', 'part', 'heat_risk_c', 'heat_risk_f']],
        tfn_extreme_cold[['grid_id', 'part', 'cold_risk_c', 'cold_risk_f', 'geometry']],
        on=['grid_id', 'part'],
        how='inner'
    )

    tfn_extreme_weather_merge = gpd.GeoDataFrame(tfn_extreme_weather_merge, geometry='geometry', crs="EPSG:3857")
    tfn_extreme_weather_merge.drop(columns=['grid_id', 'part'], inplace=True)

    tfn_extreme_weather_merge = tfn_extreme_weather_merge.to_crs("EPSG:27700")
    tfn_drought = tfn_drought.to_crs("EPSG:27700")
    tfn_storm = tfn_storm.to_crs("EPSG:27700")

    tfn_extreme_weather_overlay = gpd.overlay(tfn_extreme_weather_merge,
                                              tfn_drought[['drought_risk_c', 'drought_risk_f', 'geometry']],
                                              how='union'
                                              )

    tfn_extreme_weather_overlay = gpd.overlay(tfn_extreme_weather_overlay,
                                              tfn_storm[['storm_risk_c', 'storm_risk_f', 'geometry']],
                                              how='union'
                                              )

    tfn_extreme_weather_overlay = filter_out_small_geometries(tfn_extreme_weather_overlay, 0.035)

    tfn_extreme_weather_risk = iterative_spatial_smoothing(
        tfn_extreme_weather_overlay, ['heat_risk_c', 'heat_risk_f', 'cold_risk_c', 'cold_risk_f',
                                      'drought_risk_c', 'drought_risk_f', 'storm_risk_c', 'storm_risk_f'])

    tfn_extreme_weather_risk = explode_to_polygons(tfn_extreme_weather_risk)
    tfn_extreme_weather_risk.drop(columns=['part'], inplace=True)

    tfn_extreme_weather_risk = calculate_composite_score(
        tfn_extreme_weather_risk,
        {'heat_risk': 0.25, 'cold_risk': 0.25, 'drought_risk': 0.25, 'storm_risk': 0.25},
        'extreme_weather_risk'
    )

    tfn_extreme_weather_risk = min_max_scaling_pair(
        tfn_extreme_weather_risk, [('extreme_weather_risk_c', 'extreme_weather_risk_f')])

    tfn_extreme_weather_risk = gpd.GeoDataFrame(tfn_extreme_weather_risk, geometry='geometry')

    write_to_file(tfn_extreme_weather_risk,
                  cfg.paths.model_interim_output / "TfN Extreme Weather Risk" / "tfn_extreme_weather_risk.shp")

#### EXTREME HEAT

def extreme_heat_index(cfg, common_grid):
    """Combines several datasets into a single extreme heat index by merging on their common grid"""
    tfn_temp_max = pd.read_csv(
        cfg.paths.model_input / "Hazards" / "Extreme Weather" / "TfN Summer Max Temperature Change Projections" /
        "tfn_temp_max.csv")
    tfn_hsd = pd.read_csv(
        cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Hot Summer Days Projections" / "tfn_hot_days.csv")
    tfn_esd = pd.read_csv(
        cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Extreme Summer Days Projections" / "tfn_extr_days.csv")


    tfn_extreme_heat = merge_on_key([tfn_temp_max, tfn_hsd, tfn_esd], common_grid, 'grid_id')


    tfn_extreme_heat = calculate_risk_threshold(
        tfn_extreme_heat, 'tasmax_s', 'tasmax_risk', 30)

    tfn_extreme_heat = min_max_scaling_pair(
        tfn_extreme_heat,
        [('tasmax_risk_c', 'tasmax_risk_f'), ('hsd_c', 'hsd_f'),('esd_c', 'esd_f')]
    )

    tfn_extreme_heat = calculate_composite_score(
        tfn_extreme_heat, {'tasmax_risk': 0.5, 'hsd': 0.25,'esd': 0.25,}, 'heat_risk')

    tfn_extreme_heat = min_max_scaling_pair(tfn_extreme_heat, [('heat_risk_c', 'heat_risk_f')])

    tfn_extreme_heat = gpd.GeoDataFrame(tfn_extreme_heat, geometry='geometry')

    return tfn_extreme_heat

#### EXTREME COLD

def extreme_cold_index(cfg, common_grid):
    """Combines several datasets into a single extreme cold index by merging on their common grid"""
    tfn_temp_min = pd.read_csv(cfg.paths.model_input / "Hazards" / "Extreme Weather"  /
                               "TfN Winter Min Temperature Change Projections" / "tfn_temp_min.csv")
    tfn_frost = pd.read_csv(cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Frost Days Projections" /
                            "tfn_frost_days.csv")
    tfn_icing = pd.read_csv(cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Icing Days Projections" /
                            "tfn_ice_days.csv")

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

#### DROUGHT

def drought_index(cfg, common_grid):
    """Combines several datasets into a single drought index using a spatial overlay and spatial smoothing, before
    normalising and calculating the composite score"""
    tfn_drought = gpd.read_file(cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Drought Severity Index" /
                                "tfn_drought_index.shp")
    tfn_precip_sum = pd.read_csv(cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                                   "TfN Summer Precipitation Change Projections" / "tfn_precip_sum.csv")

    tfn_precip_sum_grid = pd.merge(tfn_precip_sum, common_grid, on='grid_id')
    tfn_precip_sum_gdf = gpd.GeoDataFrame(tfn_precip_sum_grid, geometry='geometry', crs=common_grid.crs)
    tfn_precip_sum_gdf = tfn_precip_sum_gdf[['pr_s_c', 'pr_s_f', 'geometry']]

    tfn_drought = tfn_drought.to_crs("EPSG:27700")
    tfn_precip_sum_gdf = tfn_precip_sum_gdf.to_crs("EPSG:27700")

    tfn_drought_overlay = gpd.overlay(tfn_precip_sum_gdf, tfn_drought, how='union')
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

#### STORMS

def storm_index(cfg, common_grid):
    """Combines several datasets into a single storm index using a spatial overlay and spatial smoothing, before
    normalising and calculating the composite score"""
    tfn_precip_win = pd.read_csv(cfg.paths.model_input / "Hazards" / "Extreme Weather"  /
                                 "TfN Winter Precipitation Change Projections" / "tfn_precip_win.csv")
    tfn_rain_days = gpd.read_file(cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN 10mm Rain Days 1991-2020" /
                                  "tfn_rain_days.shp")
    tfn_wind_spd = gpd.read_file(cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Wind Speed Projections" /
                                 "tfn_windspd.shp")
    tfn_wdr = gpd.read_file(cfg.paths.model_input / "Hazards" / "Extreme Weather"  / "TfN Wind Driven Rain Index" /
                            "tfn_wdr.shp")

    tfn_precip_win_grid = pd.merge(tfn_precip_win, common_grid, on='grid_id', how='left', validate='one_to_many')
    tfn_precip_win_gdf = gpd.GeoDataFrame(tfn_precip_win_grid, geometry='geometry', crs=common_grid.crs)
    tfn_precip_win_gdf = tfn_precip_win_gdf[['pr_w_c', 'pr_w_f', 'geometry']]

    tfn_wind_spd = tfn_wind_spd.to_crs("EPSG:27700")
    tfn_rain_days = tfn_rain_days.to_crs("EPSG:27700")
    tfn_precip_win_gdf = tfn_precip_win_gdf.to_crs("EPSG:27700")
    tfn_wdr = tfn_wdr.to_crs("EPSG:27700")

    tfn_storm_overlay = gpd.overlay(tfn_wind_spd, tfn_rain_days, how='union')
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_precip_win_gdf, how='union')
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_wdr, how='union')

    tfn_storm_overlay = tfn_storm_overlay[['rain_d_c', 'rain_d_f','pr_w_c', 'pr_w_f','p99_c', 'p99_f','avg_excd_c',
                                           'avg_excd_f','wdr_c', 'wdr_f','geometry']]

    tfn_storm_overlay = filter_out_small_geometries(tfn_storm_overlay, 0.01)

    tfn_storm_overlay = iterative_spatial_smoothing(
        tfn_storm_overlay,
        ['rain_d_c', 'rain_d_f', 'pr_w_c', 'pr_w_f', 'p99_c', 'p99_f','avg_excd_c', 'avg_excd_f', 'wdr_c', 'wdr_f'])

    tfn_storm_risk = explode_to_polygons(tfn_storm_overlay)
    tfn_storm_risk.drop(columns=['part'], inplace=True)

    tfn_storm_risk['wind_spd_risk_c'] = tfn_storm_risk['p99_c'].apply(wind_risk_scaled)
    tfn_storm_risk['wind_spd_risk_f'] = tfn_storm_risk['p99_f'].apply(wind_risk_scaled)

    tfn_storm_risk = min_max_scaling_pair(tfn_storm_risk,[
        ('wind_spd_risk_c', 'wind_spd_risk_f'),('pr_w_c', 'pr_w_f'),('avg_excd_c', 'avg_excd_f'),('wdr_c', 'wdr_f'),
        ('rain_d_c', 'rain_d_f'),])


    tfn_storm_risk = calculate_composite_score(
        tfn_storm_risk, {'wind_spd_risk': 0.3,'avg_excd': 0.2,'pr_w': 0.15,'rain_d': 0.15, 'wdr': 0.2}, 'storm_risk')

    tfn_storm_risk = min_max_scaling_pair(tfn_storm_risk, [('storm_risk_c', 'storm_risk_f')])

    tfn_storm_risk = gpd.GeoDataFrame(tfn_storm_risk, geometry='geometry')

    return tfn_storm_risk

def wind_risk_scaled(speed_mps):
    """Calculates a wind risk value given a wind speed, based on classification rule"""
    if speed_mps < 13.4: # Below 30 mph
        return 0
    elif speed_mps <= 20.1: # between 30 and 45 mph
        return (speed_mps - 13.4) / (20.1 - 13.4) # Scale to 0 - 1
    else:
        return 1 + (speed_mps - 20.1) / (25 - 20.1) # Scale beyond 1

### FLOODING

def flooding_index(cfg, boundary):
    """Combines RoFRS and RoFSW indexes into a single risk index by upscaling them to a common grid, normalising, and
    calculating a composite risk score"""
    risk_score_map = {'Unavailable': 0, 'Very low': 0, 'Low': 1, 'Medium': 2, 'High': 3}

    if cfg.create_flood_grid:
        flood_grid = create_flood_grid(cfg, 1000, boundary)
    else:
        flood_grid = gpd.read_file(cfg.paths.model_interim_output / "Other" / "flood_grid.shp")

    tfn_flood_risk_c, tfn_flood_risk_f = upscale_to_grid(cfg, risk_score_map, flood_grid)

    tfn_flood_risk = pd.merge(tfn_flood_risk_c[['FID', 'rofrs_risk_c', 'rofsw_risk_c']],
                              tfn_flood_risk_f, on='FID', how='left')

    tfn_flood_risk = min_max_scaling_pair(
        tfn_flood_risk, [('rofrs_risk_c', 'rofrs_risk_f'),('rofsw_risk_c', 'rofsw_risk_f')])

    tfn_flood_risk = calculate_composite_score(tfn_flood_risk, {'rofrs_risk': 0.5, 'rofsw_risk': 0.5}, 'flood_risk')

    tfn_flood_risk = min_max_scaling_pair(tfn_flood_risk, [('flood_risk_c', 'flood_risk_f')])

    write_to_file(tfn_flood_risk, cfg.paths.model_interim_output / "TfN Flood Risk" / "tfn_flood_risk.gpkg", "GPKG")

def create_flood_grid(cfg, size_m, boundary):
    """Creates a grid of a given size in metres, within a given boundary"""
    bounds = boundary.total_bounds
    grid = create_grid(bounds, size_m)
    flood_grid = clip_to_boundary(grid, boundary)
    write_to_file(flood_grid, cfg.paths.model_interim_output / "Other" / "flood_grid.shp")
    return flood_grid

def process_flood_layer(flood_grid, file_path, risk_column, risk_score_map):
    """Reads a flood layer, assigns its risk, and returns its area-weighted flood risk GeoDataFrame"""
    layer = gpd.read_file(file_path)
    layer[risk_column] = layer['Risk_band'].map(risk_score_map)
    return area_weighted_flood_assignment(flood_grid, layer, risk_column)

def upscale_to_grid(cfg, risk_score_map, flood_grid):
    """Upscales each flood layer to the common grid and writes to file"""
    scenarios = {
        "current": [("TfN RoFRS", "tfn_rofrs.gpkg", "rofrs_risk_c"),
                    ("TfN RoFSW", "tfn_rofsw.gpkg", "rofsw_risk_c")],
        "forecast": [("TfN RoFRS_CC", "tfn_rofrs_cc.gpkg", "rofrs_risk_f"),
                   ("TfN RoFSW CC", "tfn_rofsw_cc.gpkg", "rofsw_risk_f")]
    }

    results = []
    for scenario, layers in scenarios.items():
        result = flood_grid
        for folder, file, risk_col in layers:
            result = process_flood_layer(
                result, cfg.paths.model_input / "Hazards" / "Flooding" / folder / file, risk_col, risk_score_map)

        write_to_file(
            result, cfg.paths.model_interim_output / "TfN Flood Risk" / f"tfn_flood_risk_{scenario[0]}.gpkg", "GPKG")
        results.append(result)

    return results[0], results[1]

### GROUND STABILITY

def ground_stability_index(cfg):
    """Combines GeoSure and GeoClimate ground stability risk into a single index, using a spatial overlay, before
    normalising and calculating a composite risk score"""
    risk_scores = {  # Map risk scores to normalised values (0-100)
        'Probable': 100,
        'Possible': 66,
        'Improbable': 33,
        'Unavailable': 50  # Assign neutral value
    }


    tfn_geosure = gpd.read_file(cfg.paths.model_input / "Hazards" / "Ground Stability" / "TfN Geosure" / "tfn_geosure.shp")
    tfn_geosure = tfn_geosure.to_crs("EPSG:27700")

    tfn_ss = {}
    ground_stability = {}
    for year, tp in {'2030': 'c', '2070': 'f'}.items():
        tfn_ss[year] = gpd.read_file(
            cfg.model_input / "Hazards" / "Ground Stability" / "BGS Shrink Swell" / year / f"tfn_bgs_ss_{year}.shp")
        tfn_ss[year]['ss_geo_risk'] = tfn_ss[year]['ss_geo_ris'].map(risk_scores)
        tfn_ss[year] = tfn_ss[year][['ss_geo_risk', 'geometry']]
        tfn_ss[year] = tfn_ss[year].to_crs("EPSG:27700")
        ground_stability[tp] = gpd.overlay(tfn_geosure, tfn_ss[year], how='union')
        ground_stability[tp] = ground_stability[tp].rename(
            columns={col: f"{col}_{tp}" for col in ground_stability[tp].columns if col != 'geometry'}
        )
        ground_stability[tp] = ground_stability[tp].to_crs("EPSG:27700")

    tfn_ground_stability = gpd.overlay(ground_stability['c'], ground_stability['f'], how='union')

    tfn_ground_stability = explode_to_polygons(tfn_ground_stability)

    risk_cols = ['cd_risk_c', 'cg_risk_c', 'ls_risk_c', 'rs_risk_c', 'ss_risk_c', 'sr_risk_c', 'ss_geo_risk_c',
                 'cd_risk_f', 'cg_risk_f', 'ls_risk_f', 'rs_risk_f', 'ss_risk_f', 'sr_risk_f', 'ss_geo_risk_f']

    for col in risk_cols:
        tfn_ground_stability[col] = pd.to_numeric(tfn_ground_stability[col], errors='coerce')

    tfn_ground_stability = iterative_spatial_smoothing(tfn_ground_stability, risk_cols)

    tfn_ground_stability = min_max_scaling_pair(tfn_ground_stability,
                                                [('cd_risk_c', 'cd_risk_f'),
                                                ('cg_risk_c', 'cg_risk_f'),
                                                ('ls_risk_c', 'ls_risk_f'),
                                                ('rs_risk_c', 'rs_risk_f'),
                                                ('ss_risk_c', 'ss_risk_f'),
                                                ('sr_risk_c', 'sr_risk_f'),
                                                ('ss_geo_risk_c', 'ss_geo_risk_f'),
                                                ])

    tfn_ground_stability = calculate_composite_score(
        tfn_ground_stability,
        {'ss_geo_risk': 0.40, 'ls_risk': 0.10, 'ss_risk': 0.10, 'cg_risk': 0.10,
         'cd_risk': 0.10, 'rs_risk': 0.10, 'sr_risk': 0.10}, 'ground_stability_risk')

    tfn_ground_stability = min_max_scaling_pair(
        tfn_ground_stability, [('ground_stability_risk_c', 'ground_stability_risk_f')])

    write_to_file(
        tfn_ground_stability,
        cfg.paths.model_interim_output / "TfN Ground Stability Risk" / "tfn_ground_stability_risk.gpkg", "GPKG"
    )

### COASTAL EROSION

def coastal_erosion_index(cfg):
    """Combines erosion and ground instability risk from NCERM into a single index using a spatial overlay, before
    normalising and calculating a composite risk score"""
    tfn_ncerm_giz = gpd.read_file(
        cfg.paths.model_input / "Hazards" / "Coastal Erosion" / "NCERM" / "Ground Instability Zones" / "tfn_ncerm_giz.shp")
    tfn_ncerm_giz['risk_giz'] = 1

    tfn_ncerm = {}
    tfn_erosion_risk = {}
    for year, tp in {'2055': 'c', '2105': 'f'}.items():
        tfn_ncerm[year] = (
            gpd.read_file(
                cfg.model_input / "Hazards" / "Coastal Erosion" / "NCERM" / f"SMP_{year}_70CC" /
                f"tfn_ncerm_smp_{year}_70CC.shp"))
        tfn_ncerm[year]['risk_erosion'] = 1
        tfn_erosion_risk[tp] = overlay_normalise(
            tfn_ncerm_giz, tfn_ncerm[year], ['risk_erosion', 'risk_giz'],
            'erosion', {'risk_erosion': 0.9, 'risk_giz': 0.1})
        tfn_erosion_risk[tp].rename(
            columns={'erosion': f'erosion_{tp}'}, inplace=True)

    tfn_coastal_erosion_risk = gpd.overlay(tfn_erosion_risk['c'], tfn_erosion_risk['f'], how="union")

    tfn_coastal_erosion_risk = tfn_coastal_erosion_risk.fillna(0)
    tfn_coastal_erosion_risk = gpd.GeoDataFrame(tfn_coastal_erosion_risk, geometry='geometry')
    tfn_coastal_erosion_risk = tfn_coastal_erosion_risk[['erosion_c', 'erosion_f', 'geometry']]

    write_to_file(tfn_coastal_erosion_risk,
                  cfg.paths.model_interim_output / "TfN Coastal Erosion Risk" / "tfn_coastal_erosion_risk.shp")




