import os

import geopandas as gpd
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import contextily as ctx

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from data_cleaning import write_to_file
from functional_rules import min_max_scaling_pair


# FILE PATHS

MODEL_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model inputs"
MODEL_OUTPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model outputs"
MODEL_INTERIM_OUTPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model intermim outputs"

INTERIM_HAZARD_PATH = MODEL_INTERIM_OUTPUT_PATH / "Hazards"

INFRASTRUCTURE_IN_PATH = MODEL_INPUT_PATH / "Infrastructure"

IMPACT_IN_PATH = MODEL_INPUT_PATH / "Impact"

# GENERAL FUNCTIONS

def infrastructure_risk_overlay(gdf, hazards_dict):
    gdf_with_risk = gdf.copy()

    for hazard, gdf in hazards_dict.items():
        # Spatial join to find intersections with hazards
        gdf = gdf.to_crs(gdf_with_risk.crs) # Match CRS
        intersections = gpd.sjoin(gdf_with_risk, gdf, how="left", predicate="intersects")

        # Identify risk columns
        risk_columns = gdf.columns[gdf.columns.str.contains("risk", case=False)]

        # Calculate hazard risk score per road segment as max value of intersection
        agg = intersections.groupby(intersections.index)[risk_columns].max()

        # Merge back into main DataFrame
        gdf_with_risk = gdf_with_risk.join(agg, how="left")

    gdf_with_risk.fillna(0, inplace=True)

    return gdf_with_risk

def reshape_for_current_forecast(gdf, id_col, risk_cols_order):
    # Identify risk and descriptive columns
    risk_cols = [col for col in gdf.columns if col.endswith('_c') or col.endswith('_f')]
    id_cols = [id_col]
    descriptive_cols = [col for col in gdf.columns if col not in risk_cols and col not in id_cols and col != 'geometry']

    # Separate geometry for later
    geometry = gdf[[id_col, 'geometry']].copy()

    # Melt only risk columns
    melted = gdf.melt(id_vars=id_cols + descriptive_cols, value_vars=risk_cols, var_name='variable', value_name='value')

    # Extract scenario and clean variable names
    melted['current_or_forecast'] = melted['variable'].str.extract(r'_(c|f)$')[0].map({'c': 'Current', 'f': 'Forecast'})
    melted['variable'] = melted['variable'].str.replace(r'_(c|f)$', '', regex=True)

    # Pivot back so each risk variable becomes a column
    reshaped = melted.pivot_table(index=id_cols + ['current_or_forecast'] + descriptive_cols, columns='variable', values='value').reset_index()

    # Reorder risk columns based on original order
    reshaped = reshaped[id_cols + ['current_or_forecast'] + descriptive_cols + risk_cols_order]

    # Merge geometry back
    reshaped_gdf = pd.merge(reshaped, geometry, on=id_col)
    reshaped_gdf = gpd.GeoDataFrame(reshaped_gdf, geometry='geometry', crs=gdf.crs)

    return reshaped_gdf

def prepare_model_output(gdf, drop_cols, desc_cols, rename_map, risk_cols_order):
    gdf.drop(columns=drop_cols, inplace=True)
    gdf.drop_duplicates(subset=['geometry'], inplace=True)
    gdf.rename(columns=rename_map, inplace=True)
    gdf[desc_cols] = gdf[desc_cols].replace(0, 'N/A')
    gdf = gdf.to_crs(epsg=27700)
    gdf = reshape_for_current_forecast(gdf, 'id', risk_cols_order)
    gdf[risk_cols_order] = gdf[risk_cols_order].round(1)
    gdf.rename(columns={col: f"{col}_score" for col in risk_cols_order}, inplace=True)
    return gdf

def split_csv_shapefile(gdf, id_col, folder, filename):
    # Separate spatial and attribute data
    spatial_gdf = gdf[[id_col, 'geometry']].copy()
    attribute_gdf = gdf.drop(columns=['geometry'])

    # Save to file
    write_to_file(spatial_gdf, MODEL_OUTPUT_PATH / folder / f"{filename}.shp")
    write_to_file(attribute_gdf, MODEL_OUTPUT_PATH / folder / f"{filename}.csv", csv=True)

# LAYERING

def layering():
    hazard_layers = read_hazard_layers()

    risk_cols = [col for col in hazard_layers.keys.columns] # Extract all columns with 'risk' in title

    impact_weights = {
        'demand': 0.5,  # Weight demand as half of impact score
        'flood': 0.125,  # Weight hazards as 0.125 each to make up half
        'extreme_weather': 0.125,
        'ground_stability': 0.125,
        'erosion': 0.125
    }

    infrastructure_layering(hazard_layers, risk_cols)

## HAZARD LAYERS

def read_hazard_layers():
    hazard_layers = {
        'Extreme Weather': gpd.read_file(
            INTERIM_HAZARD_PATH / "TfN Extreme Weather Risk" / "tfn_extreme_weather_risk.shp"),
        'Flooding': gpd.read_file(
            INTERIM_HAZARD_PATH / "TfN Flood Risk" / "tfn_flood_risk.gpkg"),
        'Ground Stability': gpd.read_file(
            INTERIM_HAZARD_PATH / "TfN Ground Stability Risk" / "tfn_ground_stability_risk.gpkg"),
        'Coastal Erosion':  gpd.read_file(
            INTERIM_HAZARD_PATH / "TfN Coastal Erosion Risk" / "tfn_coastal_erosion_risk.shp"),
    }

    hazard_layers['Extreme Weather'].rename(columns={'heat_risk_': 'heat_risk_c', 'heat_ris_1': 'heat_risk_f',
                                         'cold_risk_': 'cold_risk_c', 'cold_ris_1': 'cold_risk_f',
                                         'drought_ri': 'drought_risk_c', 'drought__1': 'drought_risk_f',
                                         'storm_risk': 'storm_risk_c', 'storm_ri_1': 'storm_risk_f',
                                         'extreme_we': 'extreme_weather_risk_c',
                                         'extreme__1': 'extreme_weather_risk_f'}, inplace=True)

    hazard_layers['Ground Stability'].rename(columns=
                {'cd_risk_c': 'collapsible_deposits_risk_c', 'cd_risk_f': 'collapsible_deposits_risk_f',
                 'cg_risk_c': 'compressible_ground_risk_c', 'cg_risk_f': 'compressible_ground_risk_f',
                 'ls_risk_c': 'landslides_risk_c', 'ls_risk_f': 'landslides_risk_f',
                 'rs_risk_c': 'running_sand_risk_c', 'rs_risk_f': 'running_sand_risk_f',
                 'ss_risk_c': 'shrink_swell_risk_c', 'ss_risk_f': 'shrink_swell_risk_f',
                 'sr_risk_c': 'soluble_rocks_risk_c', 'sr_risk_f': 'soluble_rocks_risk_f',
                 'ss_geo_risk_c': 'shrink_swell_geoclimate_risk_c', 'ss_geo_risk_f': 'shrink_swell_geoclimate_risk_f'},
                inplace=True)

    hazard_layers['Coastal Erosion'].rename(columns=
                                            {'erosion_c': 'erosion_risk_c','erosion_f': 'erosion_risk_f'}, inplace=True)

    return hazard_layers

## INFRASTRUCTURE-HAZARD LAYERING

def infrastructure_layering(hazard_layers, risk_cols):
    get_road_risk(hazard_layers, risk_cols)


### ROADS

def get_road_risk(hazard_layers, risk_cols):
    os_open_road_risk(hazard_layers, risk_cols)

#### OS Open Roads

def os_open_road_risk(hazard_layers, risk_cols):
    tfn_os_road = gpd.read_file(INFRASTRUCTURE_IN_PATH / "Road" / "TfN OS Road" / "tfn_os_road.shp")

    tfn_os_road_risk = infrastructure_risk_overlay(tfn_os_road, hazard_layers)

    tfn_os_road_risk = prepare_model_output(
        gdf=tfn_os_road_risk,
        drop_cols=['class', 'name2', 'formOfWay', 'primary', 'structure'],
        desc_cols=['road_number', 'name', 'function'],
        rename_map={'identifier': 'id', 'name1': 'name', 'roadNumber': 'road_number'},
        risk_cols_order=risk_cols
    )

    split_csv_shapefile(tfn_os_road_risk, 'id', 'OS Roads', 'tfn_os_road_risk')

def noham_road_risk(hazard_layers, risk_cols):
    tfn_noham_c = gpd.read_file(IMPACT_IN_PATH / "TfN NoHAM Flows" / "2023" / "tfn_noham_net_flows_c.gpkg")
    tfn_noham_f = gpd.read_file(IMPACT_IN_PATH / "TfN NoHAM Flows" / "2048" / "tfn_noham_net_flows_f.gpkg")

    tfn_noham_risk_c = infrastructure_risk_overlay(tfn_noham_c, hazard_layers)
    tfn_noham_risk_f = infrastructure_risk_overlay(tfn_noham_f, hazard_layers)

    user_classes = ["uc1", "uc2", "uc3", "uc4", "uc5"]

    # Normalise user class totals together
    uc_total_cols = [f"{uc}_total" for uc in user_classes]

    # Combine current and future values for global min/max
    combined_values = np.vstack([
        tfn_noham_risk_c[uc_total_cols].values,
        tfn_noham_risk_f[uc_total_cols].values
    ])

    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_values = scaler.fit_transform(combined_values)

    # Assign scaled values to dataframe
    scaled_c = scaler.transform(tfn_noham_risk_c[uc_total_cols].values)
    scaled_f = scaler.transform(tfn_noham_risk_f[uc_total_cols].values)

    # Assign scaled values back with proper column names
    tfn_noham_risk_c[[f"{uc}_demand_c" for uc in user_classes]] = scaled_c
    tfn_noham_risk_f[[f"{uc}_demand_f" for uc in user_classes]] = scaled_f

    # Normalise all vehicles total separately
    combined_values = np.vstack([
        tfn_noham_risk_c['all_vehs_total'].values.reshape(-1, 1),
        tfn_noham_risk_f['all_vehs_total'].values.reshape(-1, 1)
    ])

    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_values = scaler.fit_transform(combined_values)

    # Assign scaled values to dataframe
    scaled_c = scaler.transform(tfn_noham_risk_c['all_vehs_total'].values.reshape(-1, 1))
    scaled_f = scaler.transform(tfn_noham_risk_f['all_vehs_total'].values.reshape(-1, 1))

    # Assign scaled values back with proper column names
    tfn_noham_risk_c['demand_c'] = scaled_c
    tfn_noham_risk_f['demand_f'] = scaled_f

    # Calculate impact metric for each user class
    for uc in user_classes:
        tfn_noham_risk_c[f'{uc}_impact_c'] = (
                tfn_noham_risk_c[f'{uc}_demand_c'] * impact_weights['demand'] +
                tfn_noham_risk_c['flood_risk_c'] * impact_weights['flood'] +
                tfn_noham_risk_c['extreme_weather_risk_c'] * impact_weights['extreme_weather'] +
                tfn_noham_risk_c['ground_stability_risk_c'] * impact_weights['ground_stability'] +
                tfn_noham_risk_c['erosion_risk_c'] * impact_weights['erosion']
        )

        tfn_noham_risk_f[f'{uc}_impact_f'] = (
                tfn_noham_risk_f[f'{uc}_demand_f'] * impact_weights['demand'] +
                tfn_noham_risk_f['flood_risk_f'] * impact_weights['flood'] +
                tfn_noham_risk_f['extreme_weather_risk_f'] * impact_weights['extreme_weather'] +
                tfn_noham_risk_f['ground_stability_risk_f'] * impact_weights['ground_stability'] +
                tfn_noham_risk_f['erosion_risk_f'] * impact_weights['erosion']
        )

        tfn_noham_risk_c['impact_c'] = (
                tfn_noham_risk_c['demand_c'] * impact_weights['demand'] +
                tfn_noham_risk_c['flood_risk_c'] * impact_weights['flood'] +
                tfn_noham_risk_c['extreme_weather_risk_c'] * impact_weights['extreme_weather'] +
                tfn_noham_risk_c['ground_stability_risk_c'] * impact_weights['ground_stability'] +
                tfn_noham_risk_c['erosion_risk_c'] * impact_weights['erosion']
        )

        tfn_noham_risk_f['impact_f'] = (
                tfn_noham_risk_f['demand_f'] * impact_weights['demand'] +
                tfn_noham_risk_f['flood_risk_f'] * impact_weights['flood'] +
                tfn_noham_risk_f['extreme_weather_risk_f'] * impact_weights['extreme_weather'] +
                tfn_noham_risk_f['ground_stability_risk_f'] * impact_weights['ground_stability'] +
                tfn_noham_risk_f['erosion_risk_f'] * impact_weights['erosion']
        )

        impact_cols_c = [f"{uc}_impact_c" for uc in user_classes] + ['impact_c']
        impact_cols_f = [f"{uc}_impact_f" for uc in user_classes] + ['impact_f']

        combined_values = np.vstack([
            tfn_noham_risk_c[impact_cols_c].values,
            tfn_noham_risk_f[impact_cols_f].values
        ])

        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(combined_values)

        scaled_c = scaler.transform(tfn_noham_risk_c[impact_cols_c].values)
        scaled_f = scaler.transform(tfn_noham_risk_f[impact_cols_f].values)

        tfn_noham_risk_c[impact_cols_c] = scaled_c
        tfn_noham_risk_f[impact_cols_f] = scaled_f

        cols_to_round = [col for col in tfn_noham_risk_c.columns if col not in ['link_id', 'geometry']]
        tfn_noham_risk_c[cols_to_round] = tfn_noham_risk_c[cols_to_round].round(1)

        # Drop all raw demand and future hazards
        tfn_noham_risk_c.drop(
            columns=['uc1_total', 'uc2_total', 'uc3_total', 'uc4_total', 'uc5_total', 'all_vehs_total',
                     'uc1_demand_c', 'uc2_demand_c', 'uc3_demand_c', 'uc4_demand_c', 'uc5_demand_c', 'demand_c',
                     'heat_risk_f', 'cold_risk_f', 'drought_risk_f', 'storm_risk_f', 'extreme_weather_risk_f',
                     'rofrs_risk_f', 'rofsw_risk_f', 'flood_risk_f', 'collapsible_deposits_risk_f',
                     'compressible_ground_risk_f',
                     'landslides_risk_f', 'running_sand_risk_f', 'shrink_swell_risk_f', 'soluble_rocks_risk_f',
                     'shrink_swell_geoclimate_risk_f', 'ground_stability_risk_f', 'erosion_risk_f'], inplace=True)

        tfn_noham_risk_c.drop_duplicates(subset=['geometry'], inplace=True)

        tfn_noham_risk_c = tfn_noham_risk_c.to_crs(epsg=27700)

        cols_to_round = [col for col in tfn_noham_risk_f.columns if col not in ['link_id', 'geometry']]
        tfn_noham_risk_f[cols_to_round] = tfn_noham_risk_f[cols_to_round].round(1)

        # Drop all raw demand and future hazards
        tfn_noham_risk_f.drop(
            columns=['uc1_total', 'uc2_total', 'uc3_total', 'uc4_total', 'uc5_total', 'all_vehs_total',
                     'uc1_demand_f', 'uc2_demand_f', 'uc3_demand_f', 'uc4_demand_f', 'uc5_demand_f', 'demand_f',
                     'heat_risk_c', 'cold_risk_c', 'drought_risk_c', 'storm_risk_c', 'extreme_weather_risk_c',
                     'rofrs_risk_c', 'rofsw_risk_c', 'flood_risk_c', 'collapsible_deposits_risk_c',
                     'compressible_ground_risk_c',
                     'landslides_risk_c', 'running_sand_risk_c', 'shrink_swell_risk_c', 'soluble_rocks_risk_c',
                     'shrink_swell_geoclimate_risk_c', 'ground_stability_risk_c', 'erosion_risk_c'], inplace=True)

        tfn_noham_risk_f.drop_duplicates(subset=['geometry'], inplace=True)

        tfn_noham_risk_f = tfn_noham_risk_f.to_crs(epsg=27700)

        # Remove suffixes from risk columns
        tfn_noham_risk_c.columns = [col.replace('_c', '') for col in tfn_noham_risk_c.columns]
        tfn_noham_risk_f.columns = [col.replace('_f', '') for col in tfn_noham_risk_f.columns]

        # Add scenario column
        tfn_noham_risk_c['current_or_forecast'] = 'Current'
        tfn_noham_risk_f['current_or_forecast'] = 'Forecast'

        # Concatenate
        tfn_noham_risk = pd.concat([tfn_noham_risk_c, tfn_noham_risk_f], ignore_index=True)

        tfn_noham_risk = tfn_noham_risk[['link_id', 'current_or_forecast', 'geometry'] + risk_cols]

        tfn_noham_risk.rename(columns={'link_id': 'id'}, inplace=True)
        tfn_noham_risk.rename(columns={col: f"{col}_score" for col in risk_cols + ['impact']}, inplace=True)

        split_csv_shapefile(tfn_noham_risk, 'id', 'NoHAM', 'tfn_noham_risk')



