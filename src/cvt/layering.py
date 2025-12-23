import geopandas as gpd
import pandas as pd

from data_cleaning import write_to_file
from file_paths import (MODEL_OUTPUT, HAZARD_INTERIM_OUT, IMPACT_INTERIM_OUT,
                        ROAD_MODEL_IN, RAIL_MODEL_IN, OTHER_MODEL_IN)


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
    write_to_file(spatial_gdf, MODEL_OUTPUT / folder / f"{filename}.shp")
    write_to_file(attribute_gdf, MODEL_OUTPUT / folder / f"{filename}.csv", csv=True)

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
            HAZARD_INTERIM_OUT / "TfN Extreme Weather Risk" / "tfn_extreme_weather_risk.shp"),
        'Flooding': gpd.read_file(
            HAZARD_INTERIM_OUT / "TfN Flood Risk" / "tfn_flood_risk.gpkg"),
        'Ground Stability': gpd.read_file(
            HAZARD_INTERIM_OUT / "TfN Ground Stability Risk" / "tfn_ground_stability_risk.gpkg"),
        'Coastal Erosion':  gpd.read_file(
            HAZARD_INTERIM_OUT / "TfN Coastal Erosion Risk" / "tfn_coastal_erosion_risk.shp"),
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

    hazard_layers['Coastal Erosion'].rename(
        columns={'erosion_c': 'erosion_risk_c','erosion_f': 'erosion_risk_f'}, inplace=True)

    return hazard_layers

## INFRASTRUCTURE-HAZARD LAYERING

def infrastructure_layering(hazard_layers, risk_cols):
    get_road_risk(hazard_layers, risk_cols)
    get_rail_risk(hazard_layers, risk_cols)
    get_other_risk(hazard_layers, risk_cols)


### ROAD

def get_road_risk(hazard_layers, risk_cols):
    os_open_road_risk(hazard_layers, risk_cols)
    noham_road_risk(hazard_layers, risk_cols)

#### OS Open Roads

def os_open_road_risk(hazard_layers, risk_cols):
    tfn_os_road = gpd.read_file(ROAD_MODEL_IN / "TfN OS Road" / "tfn_os_road.shp")

    tfn_os_road_risk = infrastructure_risk_overlay(tfn_os_road, hazard_layers)

    tfn_os_road_risk = prepare_model_output(
        gdf=tfn_os_road_risk,
        drop_cols=['class', 'name2', 'formOfWay', 'primary', 'structure'],
        desc_cols=['road_number', 'name', 'function'],
        rename_map={'identifier': 'id', 'name1': 'name', 'roadNumber': 'road_number'},
        risk_cols_order=risk_cols
    )

    split_csv_shapefile(tfn_os_road_risk, 'id', 'OS Roads', 'tfn_os_road_risk')

#### NoHAM

def noham_road_risk(hazard_layers, risk_cols):
    tfn_noham = {}
    tfn_noham_risk = {}
    for tp in ['c', 'f']:
        tfn_noham[tp] = gpd.read_file(
            IMPACT_INTERIM_OUT / "TfN NoHAM Flows" / f"tfn_noham_net_flows_{tp}.gpkg")
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

### RAIL

def get_rail_risk(hazard_layers, risk_cols):
    passenger_rail_risk(hazard_layers, risk_cols)
    freight_rail_risk(hazard_layers, risk_cols)

#### Passenger Rail

def passenger_rail_risk(hazard_layers, risk_cols):
    tfn_rail_network = gpd.read_file(
        RAIL_MODEL_IN / "TfN OS Passenger Rail" / "tfn_pass_rail_links.shp")

    tfn_rail_network_risk = infrastructure_risk_overlay(tfn_rail_network, hazard_layers)

    tfn_rail_network_risk = prepare_model_output(
        gdf=tfn_rail_network_risk,
        drop_cols=['geometry_l', 'gauge', 'direction', 'operationa', 'name1_text', 'startnode', 'endnode', 'rgn17cd'],
        desc_cols=['description', 'structure', 'physical_level', 'railway_use', 'track_representation'],
        rename_map={'osid': 'id', 'descriptio': 'description', 'physicalle': 'physical_level',
                    'railwayuse': 'railway_use', 'trackrepre': 'track_representation'},
        risk_cols_order=risk_cols
    )

    split_csv_shapefile(tfn_rail_network_risk, "id", "Passenger Rail", "tfn_passenger_rail_network_risk")

#### Freight Rail

def freight_rail_risk(hazard_layers, risk_cols):
    tfn_freight_network = gpd.read_file(
        IMPACT_INTERIM_OUT / "TfN Freight Flows" / "tfn_freight_network_impact.gpkg")

    tfn_freight_network_risk = infrastructure_risk_overlay(tfn_freight_network, hazard_layers)

    tfn_freight_network_risk = prepare_model_output(
        gdf=tfn_freight_network_risk,
        drop_cols=['geometry_l', 'gauge', 'direction', 'operationa', 'name1_text', 'startnode', 'endnode', 'rgn17cd',
                   'dij_id', 'distance', 'demand_c', 'demand_f'],
        desc_cols=['description', 'structure', 'physical_level', 'railway_use', 'track_representation'],
        rename_map={'osid': 'id', 'descriptio': 'description', 'physicalle': 'physical_level',
                    'railwayuse': 'railway_use', 'trackrepre': 'track_representation'},
        risk_cols_order=risk_cols + ['impact']
    )

    split_csv_shapefile(tfn_freight_network_risk, 'id', "Freight Rail", "tfn_freight_rail_network_risk")

### OTHER

def get_other_risk(hazard_layers, risk_cols):
    train_stations_risk(hazard_layers, risk_cols)
    ev_charging_sites_risk(hazard_layers, risk_cols)
    airports_risk(hazard_layers, risk_cols)
    bus_coach_stations_risk(hazard_layers, risk_cols)
    bus_stops_risk(hazard_layers, risk_cols)
    tram_stations_risk(hazard_layers, risk_cols)
    rapid_transport_stations_risk(hazard_layers, risk_cols)
    ferry_terminals_risk(hazard_layers, risk_cols)
    petrol_stations_risk(hazard_layers, risk_cols)
    ncn_risk(hazard_layers, risk_cols)
    tram_network_risk(hazard_layers, risk_cols)
    rapid_transport_network_risk(hazard_layers, risk_cols)

def buffer_geometry(gdf, buffer_size_m):
    gdf = gdf.to_crs(epsg=27700)
    gdf['geometry'] = gdf.buffer(buffer_size_m)
    return gdf

#### Train Stations

def train_stations_risk(hazard_layers, risk_cols):
    tfn_train_stations = gpd.read_file(OTHER_MODEL_IN / "TfN OS Train Stations" / "tfn_train_stations.shp")

    tfn_train_stations = buffer_geometry(tfn_train_stations, 100)

    tfn_train_stations_risk = infrastructure_risk_overlay(tfn_train_stations, hazard_layers)

    tfn_train_stations_risk = prepare_model_output(
        gdf=tfn_train_stations_risk,
        drop_cols=['os_parenti', 'os_nodetyp', 'name'],
        desc_cols=[],
        rename_map={'nodeid': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_train_stations_risk, 'id', "Train Stations", "tfn_train_stations_risk")

#### EV Charging Sites

def ev_charging_sites_risk(hazard_layers, risk_cols):
    tfn_chg_sites = gpd.read_file(OTHER_MODEL_IN / "TfN EV Charging Sites" / "tfn_chg_sites.shp")

    tfn_chg_sites = buffer_geometry(tfn_chg_sites, 25)

    tfn_chg_sites_risk = infrastructure_risk_overlay(tfn_chg_sites, hazard_layers)

    tfn_chg_sites_risk = prepare_model_output(
        gdf=tfn_chg_sites_risk,
        drop_cols=[],
        desc_cols=['name', 'speed'],
        rename_map={'devices': 'installed_devices'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_chg_sites_risk, 'id', "EV Charging Sites", "tfn_chg_sites_risk")

#### Airports

def airports_risk(hazard_layers, risk_cols):
    tfn_airports = gpd.read_file(OTHER_MODEL_IN / "TfN Airports" / "tfn_airports.shp")

    tfn_airports_risk = infrastructure_risk_overlay(tfn_airports, hazard_layers)

    tfn_airports_risk = prepare_model_output(
        gdf=tfn_airports_risk,
        drop_cols=[],
        desc_cols=['name'],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_airports_risk, 'id', "Airports", "tfn_airports_risk")

#### Bus and Coach Stations

def bus_coach_stations_risk(hazard_layers, risk_cols):
    tfn_bus_coach_stations = gpd.read_file(OTHER_MODEL_IN / "TfN OS Bus Coach Stations" / "tfn_bus_coach_stations.shp")

    tfn_bus_coach_stations = buffer_geometry(tfn_bus_coach_stations, 50)

    tfn_bus_coach_stations_risk = infrastructure_risk_overlay(tfn_bus_coach_stations, hazard_layers)

    tfn_bus_coach_stations_risk = prepare_model_output(
        gdf=tfn_bus_coach_stations_risk,
        drop_cols=['os_parenti', 'os_nodetyp', 'name'],
        desc_cols=[],
        rename_map={'nodeid': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_bus_coach_stations_risk, 'id', "Bus and Coach Stations", "tfn_bus_coach_stations_risk")

#### Bus Stops

def bus_stops_risk(hazard_layers, risk_cols):
    tfn_bus_stops = gpd.read_file(OTHER_MODEL_IN / "TfN Bus Stops" / "tfn_bus_stops.shp")

    tfn_bus_stops_risk = infrastructure_risk_overlay(tfn_bus_stops, hazard_layers)

    tfn_bus_stops_risk = prepare_model_output(
        gdf=tfn_bus_stops_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={'stop_id': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_bus_stops_risk, 'id', "Bus Stops", "tfn_bus_stops_risk")

#### Tram Stations

def tram_stations_risk(hazard_layers, risk_cols):
    tfn_tram_stations = gpd.read_file(OTHER_MODEL_IN / "TfN OS Tram Stations" / "tfn_tram_stations.shp")

    tfn_tram_stations = buffer_geometry(tfn_tram_stations, 25)

    tfn_tram_stations_risk = infrastructure_risk_overlay(tfn_tram_stations, hazard_layers)

    tfn_tram_stations_risk = prepare_model_output(
        gdf=tfn_tram_stations_risk,
        drop_cols=['os_parenti', 'os_nodetyp', 'name'],
        desc_cols=[],
        rename_map={'nodeid': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_tram_stations_risk, 'id', "Tram Stations", "tfn_tram_stations_risk")

#### Rapid Transport Stations

def rapid_transport_stations_risk(hazard_layers, risk_cols):
    tfn_metro_stations = gpd.read_file(OTHER_MODEL_IN / "TfN OS Metro Stations" / "tfn_metro_stations.shp")

    tfn_metro_stations = buffer_geometry(tfn_metro_stations, 50)

    tfn_metro_stations_risk = infrastructure_risk_overlay(tfn_metro_stations, hazard_layers)

    tfn_metro_stations_risk = prepare_model_output(
        gdf=tfn_metro_stations_risk,
        drop_cols=['os_parenti', 'os_nodetyp', 'name'],
        desc_cols=[],
        rename_map={'nodeid': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_metro_stations_risk, 'id', 'Rapid Transport Stations', 'tfn_rapid_transport_stations_risk')

#### Ferry Terminals

def ferry_terminals_risk(hazard_layers, risk_cols):
    tfn_ferry_stations = gpd.read_file(OTHER_MODEL_IN / "TfN OS Ferry Stations" / "tfn_ferry_stations.shp")

    tfn_ferry_stations = buffer_geometry(tfn_ferry_stations, 50)

    tfn_ferry_stations_risk = infrastructure_risk_overlay(tfn_ferry_stations, hazard_layers)

    tfn_ferry_stations_risk = prepare_model_output(
        gdf=tfn_ferry_stations_risk,
        drop_cols=['os_parenti', 'os_nodetyp', 'name'],
        desc_cols=[],
        rename_map={'nodeid': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_ferry_stations_risk, 'id', 'Ferry Stations', 'tfn_ferry_stations_risk')

#### Petrol Stations

def petrol_stations_risk(hazard_layers, risk_cols):
    tfn_petrol_stations = gpd.read_file(OTHER_MODEL_IN / "TfN Petrol Stations" / "tfn_petrol_stations.shp")

    tfn_petrol_stations = buffer_geometry(tfn_petrol_stations, 50)

    tfn_petrol_stations_risk = infrastructure_risk_overlay(tfn_petrol_stations, hazard_layers)

    tfn_petrol_stations_risk = prepare_model_output(
        gdf=tfn_petrol_stations_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_petrol_stations_risk, 'id', 'Petrol Stations', 'tfn_petrol_stations_risk')

#### National Cycle Network

def ncn_risk(hazard_layers, risk_cols):
    tfn_ncn = gpd.read_file(OTHER_MODEL_IN / "TfN NCN" / "tfn_ncn.shp")

    tfn_ncn_risk = infrastructure_risk_overlay(tfn_ncn, hazard_layers)

    tfn_ncn_risk = prepare_model_output(
        gdf=tfn_ncn_risk,
        drop_cols=['RouteCat', 'OpenStatus', 'GlobalID'],
        desc_cols=['description', 'greenway', 'route_type', 'route_number', 'link_number', 'surface', 'quality',
                   'lighting', 'road_class'],
        rename_map={'Desc_': 'description', 'Greenway': 'greenway', 'RouteType': 'route_type',
                    'RouteNo': 'route_number', 'LinkNo': 'link_number', 'Surface': 'surface', 'Quality': 'quality',
                    'Lighting': 'lighting', 'RoadClass': 'road_class', 'SegmentID': 'id'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_ncn_risk, 'id', 'National Cycle Network', 'tfn_ncn_risk')

#### Tram Network

def tram_network_risk(hazard_layers, risk_cols):
    tfn_tram_network = gpd.read_file(OTHER_MODEL_IN / "TfN OS Tram Links" / "tfn_os_tram_links.shp")

    tfn_tram_risk = infrastructure_risk_overlay(tfn_tram_network, hazard_layers)

    tfn_tram_risk = prepare_model_output(
        gdf=tfn_tram_risk,
        drop_cols=['geometry_l', 'gauge', 'direction', 'operationa', 'startnode', 'endnode', 'rgn17cd'],
        desc_cols=['description', 'structure', 'physical_level', 'railway_use', 'track_representation', 'name'],
        rename_map={'osid': 'id', 'descriptio': 'description', 'physicalle': 'physical_level',
                    'railwayuse': 'railway_use', 'trackrepre': 'track_representation', 'name1_text': 'name'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_tram_risk, 'id', 'Tram Network', 'tfn_tram_links_risk')

#### Rapid Transport Network

def rapid_transport_network_risk(hazard_layers, risk_cols):
    tfn_rapid_transport = gpd.read_file(OTHER_MODEL_IN / "TfN Rapid Transport" / "tfn_rapid_transport_links.shp")

    tfn_rapid_transport_risk = infrastructure_risk_overlay(tfn_rapid_transport, hazard_layers)

    tfn_rapid_transport_risk = prepare_model_output(
        gdf=tfn_rapid_transport_risk,
        drop_cols=['geometry_l', 'gauge', 'direction', 'operationa', 'startnode', 'endnode', 'rgn17cd'],
        desc_cols=['description', 'structure', 'physical_level', 'railway_use', 'track_representation', 'name'],
        rename_map={'osid': 'id', 'descriptio': 'description', 'physicalle': 'physical_level',
                    'railwayuse': 'railway_use', 'trackrepre': 'track_representation', 'name1_text': 'name'},
        risk_cols_order=risk_cols,
    )

    split_csv_shapefile(tfn_rapid_transport_risk, 'id', 'Rapid Transport Network', 'tfn_rapid_transport_risk')



