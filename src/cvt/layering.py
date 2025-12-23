import geopandas as gpd
import pandas as pd

import numpy as np

from pathlib import Path

from data_cleaning import write_to_file

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

    risk_cols = [
        # Extreme Weather risk columns
        'heat_risk', 'cold_risk', 'drought_risk', 'storm_risk', 'extreme_weather_risk',

        # Flooding risk columns
        'rofrs_risk', 'rofsw_risk', 'flood_risk',

        # Ground Stability risk columns
        'collapsible_deposits_risk', 'compressible_ground_risk', 'landslides_risk', 'running_sand_risk',
        'shrink_swell_risk',
        'soluble_rocks_risk', 'shrink_swell_geoclimate_risk', 'ground_stability_risk',

        # Coastal Erosion risk columns
        'erosion_risk'
    ]

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
    noham_road_risk(hazard_layers, risk_cols)

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
    tfn_noham = {}
    tfn_noham_risk = {}
    for tp in ['c', 'f']:
        tfn_noham[tp] = gpd.read_file(
            MODEL_INTERIM_OUTPUT_PATH / "Impact" / "TfN NoHAM Flows" / f"tfn_noham_net_flows_{tp}.gpkg")
        tfn_noham_risk[tp] = infrastructure_risk_overlay(tfn_noham[tp], hazard_layers)
        other_tp = 'f' if tp == 'c' else 'c'
        drop_cols = [col for col in tfn_noham_risk[tp].columns if col.endswith(f'_{other_tp}')]
        tfn_noham_risk[tp].drop(columns=drop_cols, inplace=True)
        tfn_noham_risk[tp].drop_duplicates(subset=['geometry'], inplace=True)
        tfn_noham_risk[tp].columns = [col.replace(f'_{tp}', '') for col in tfn_noham_risk[tp].columns]
        if tp == 'c':
            tfn_noham_risk[tp]['current_or_forecast'] = 'Current'
        else:
            tfn_noham_risk[tp]['current_or_forecast'] = 'Forecast'

    # Concatenate
    tfn_noham_risk = pd.concat([tfn_noham_risk['c'], tfn_noham_risk['f']], ignore_index=True)

    cols_to_round = [col for col in tfn_noham_risk.columns if col not in ['link_id', 'geometry']]
    tfn_noham_risk[cols_to_round] = tfn_noham_risk[cols_to_round].round(1)
    tfn_noham_risk = tfn_noham_risk.to_crs(epsg=27700)
    tfn_noham_risk.rename(columns={'link_id': 'id'}, inplace=True)
    tfn_noham_risk.rename(columns={col: f"{col}_score" for col in cols_to_round}, inplace=True)

    split_csv_shapefile(tfn_noham_risk, 'id', 'NoHAM', 'tfn_noham_risk')



