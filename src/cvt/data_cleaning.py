'''This script will take raw input data and clean it ready for model input'''

### LOAD LIBRARIES
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
import xarray as xr
import h5py
import py7zr
from pathlib import Path
import zipfile


### FILE PATHS
RAW_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "raw inputs"
MODEL_INPUT_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model inputs"

INFRASTRUCTURE_IN = RAW_INPUT_PATH / "Infrastructure"
INFRASTRUCTURE_OUT = MODEL_INPUT_PATH / "Infrastructure"

ROAD_IN = INFRASTRUCTURE_IN / "Road"
RAIL_IN = INFRASTRUCTURE_IN / "Rail"
OTHER_IN = INFRASTRUCTURE_IN / "Other"

ROAD_OUT = INFRASTRUCTURE_OUT / "Road"
RAIL_OUT = INFRASTRUCTURE_OUT / "Rail"
OTHER_OUT = INFRASTRUCTURE_OUT / "Other"

HAZARD_IN = RAW_INPUT_PATH / "Hazard"
HAZARD_OUT = MODEL_INPUT_PATH / "Hazard"

EXTREME_WEATHER_IN = HAZARD_IN / "Extreme Weather"
FLOODING_IN = HAZARD_IN / "Flooding"
GROUND_STABILITY_IN = HAZARD_IN / "Ground Stability"
COASTAL_EROSION_IN = HAZARD_IN / "Coastal Erosion"

EXTREME_WEATHER_OUT = HAZARD_OUT / "Extreme Weather"
FLOODING_OUT = HAZARD_OUT / "Flooding"
GROUND_STABILITY_OUT = HAZARD_OUT / "Ground Stability"
COASTAL_EROSION_OUT = HAZARD_OUT / "Coastal Erosion"

IMPACT_IN = RAW_INPUT_PATH / "Impact"
IMPACT_OUT = MODEL_INPUT_PATH / "Impact"

# Main function, replace this later to be in the main script
def main():
    data_cleaning(OTHER_IN / "TfN Boundary" / "Transport_for_the_north_boundary_2020_generalised.shp")


### GENERAL FUNCTIONS

def clip_to_boundary(gdf, boundary):
    '''Takes a GeoDataFrame and a boundary and returns a GeoDataFrame clipped to that boundary'''
    boundary = boundary.to_crs(gdf.crs) # Match CRS
    gdf_boundary = gpd.clip(gdf, boundary) # Clip GDF to boundary
    return gdf_boundary

def write_to_file(gdf, output_path, driver=None, csv=False):
    '''Function to write a GeoDataFrame to a file'''
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists, make one if not

    if gdf.empty:
        raise ValueError(f"GeoDataFrame is empty. Nothing written to {output_path}")

    if csv == True:
        gdf.to_csv(output_path, index=False)
    else:
        gdf.to_file(output_path, driver=driver)

def df_to_gdf(df, x_col, y_col, crs):
    '''Takes a DataFrame and converts it to a GeoDataFrame using spatial columns'''
    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]  # Create geometry from lat/lon
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs) # Convert to GeoDataFrame
    return gdf

def convert_point_to_grid(x, y, size):
    '''Takes a point and converts it to a grid of the given size'''
    return Polygon([
        (x - size, y - size),
        (x + size, y - size),
        (x + size, y + size),
        (x - size, y + size),
    ])

def extract_poly_from_geomcollection(gdf):
    '''Takes a GeoDataFrame and extracts polygons from the GeomCollection objects and turns them into new rows'''
    rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # If it's already Polygon or MultiPolygon, keep as is
        if isinstance(geom, (Polygon, MultiPolygon)):
            rows.append(row)

     # If it's a GeometryCollection, extract polygons
        elif isinstance(geom, GeometryCollection):
            for sub_geom in geom.geoms:
                if isinstance(sub_geom, (Polygon, MultiPolygon)):
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    rows.append(new_row)

    # Create new GeoDataFrame from expanded rows
    return gpd.GeoDataFrame(rows, crs=gdf.crs)

def explode_to_polygons(gdf, id_col='grid_id'):
    '''Takes a GeoDataFrame and explodes the MultiPolygons and GeomCollections into Polygons'''
    rows = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        original_id = row[id_col] if id_col in row else idx  # fallback to index if no ID
        if geom.geom_type == 'Polygon':
            new_row = row.copy()
            new_row[id_col] = f"{original_id}_0"
            rows.append(new_row)
        elif geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(geom.geoms):
                new_row = row.copy()
                new_row.geometry = poly
                new_row[id_col] = f"{original_id}_{i}"
                rows.append(new_row)
        elif geom.geom_type == 'GeometryCollection':
            poly_count = 0
            for part in geom.geoms:
                if part.geom_type == 'Polygon':
                    new_row = row.copy()
                    new_row.geometry = part
                    new_row[id_col] = f"{original_id}_{poly_count}"
                    rows.append(new_row)
                    poly_count += 1
    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)

def nearest_centroids(gdf1, gdf2):
    '''Takes two GeoDataFrames and merges them on their nearest centroids'''
    # Ensure both GeoDataFrames are in the same projected CRS
    gdf1 = gdf1.to_crs("EPSG:27700")
    gdf2 = gdf2.to_crs("EPSG:27700")

    # Convert both to centroids
    gdf1_centroids = gdf1.copy()
    gdf1_centroids['geometry'] = gdf1_centroids.geometry.centroid
    gdf2_centroids = gdf2.copy()
    gdf2_centroids['geometry'] = gdf2_centroids.geometry.centroid

    nearest = gpd.sjoin_nearest(gdf1_centroids, gdf2_centroids, how='left')

    # Merge back with original gdf1 to restore original geometry
    result = gdf1.merge(nearest.drop(columns='geometry'), left_index=True, right_index=True)

    return result

def calculate_exceedance(threshold, gdf, variable, timescale):
    gdf['exceedance'] = gdf[variable] > threshold

    # Group by grid square and year, and count exceedance days
    exceedance_counts = (
      gdf[gdf['exceedance']]
     .groupby(['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude', 'year'])
     .size()
     .reset_index(name='exceedance_days')
    )

    # Calculate the average exceedance days per year for each grid square
    average_exceedance = (
        exceedance_counts
        .groupby(['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude'])['exceedance_days']
        .mean()
        .reset_index(name=f'avg_excd_{timescale}')
    )

    return average_exceedance

def calculate_percentile(gdf, quantiles, variable):
    percentiles = (
        gdf.groupby(['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude'])[variable]
        .quantile(quantiles)
        .unstack()
        .reset_index()
    )

    return percentiles

# DATA CLEANING

def data_cleaning(boundary_path):
    '''Function to clean all datasets ready for analysis'''
    boundary = read_boundary_path(boundary_path)

    clean_infrastruture(boundary)
    clean_hazards(boundary)
    clean_impact(boundary)


## Geographic Boundary
def read_boundary_path(boundary_path):
    tfn_boundary = gpd.read_file(boundary_path)
    return tfn_boundary

## INFRASTRUCTURE

def clean_infrastruture(boundary):
    '''Function to clean all infrastrcuture datasets ready for analysis'''
    clean_roads(boundary)
    clean_rail()
    clean_other(boundary)

### ROAD

def clean_roads(boundary):
    '''Function to clean all roads datasets ready for analysis'''
    clean_os_roads(ROAD_IN / "TfN OS Open Roads" / "os_open_gb_road_links_tfn.shp", boundary)
    clean_noham_roads(ROAD_IN / "NoHAM 2023" / "NoHAM_Decarb_DM_Core_2023_carbon.shp",
                      ROAD_IN / "NoHAM 2048" / "NoHAM_Decarb_DM_Core_2048_carbon.shp", boundary)

def clean_os_roads(os_road_path, boundary):
    os_road = gpd.read_file(os_road_path)
    os_road.drop_duplicates(subset=['identifier', 'geometry'], inplace=True)
    os_road = os_road[['identifier', 'roadNumber', 'name1', 'function', 'geometry']]
    os_road.rename(columns={'name1': 'name', 'roadNumber': 'road_number'}, inplace=True)
    os_road[['road_number', 'name', 'function']] = (
        os_road[['road_number', 'name', 'function']].replace(0, pd.NA))
    os_road = os_road[~os_road.geometry.is_empty]
    os_road = os_road[os_road.geometry.notnull()]
    tfn_os_road = clip_to_boundary(os_road, boundary)
    write_to_file(tfn_os_road, ROAD_OUT / "TfN OS Road" / "tfn_os_road.shp")

def clean_noham_roads(noham_roads_path_2023, noham_roads_path_2048, boundary):
    noham = {
        '2023': gpd.read_file(noham_roads_path_2023),
        '2048': gpd.read_file(noham_roads_path_2048)
    }

    for year, noham_network in noham.items():
        noham_network.drop_duplicates(subset=['link_id', 'geometry'], inplace=True)
        noham_network[['a', 'b']] = noham_network['link_id'].str.split('_', expand=True).astype(int)
        noham_network = noham_network[(noham_network['a'] >= 10000) & (noham_network['b'] >= 10000)]
        noham_network = noham_network[['link_id', 'geometry']]
        noham_network = noham_network[~noham_network.geometry.is_empty]
        noham_network = noham_network[noham_network.geometry.notnull()]
        tfn_noham_network = clip_to_boundary(noham_network, boundary)
        write_to_file(tfn_noham_network, ROAD_OUT / f"TfN NoHAM {year}" / f"tfn_noham_{year}.shp")

### RAIL

def clean_rail():
    '''Function to clean all rail datasets ready for analysis'''
    tfn_rail_links = get_rail_links(RAIL_IN / "OS Rail Network" / "TfN_Area_tfn_ntwk_railwaylink.gpkg")
    clean_passenger_rail(tfn_rail_links)
    clean_freight_rail(tfn_rail_links)

    # Other network data
    clean_tram_network(tfn_rail_links)
    clean_rapid_transport_network(tfn_rail_links)

def get_rail_links(os_rail_path):
    tfn_rail_links = gpd.read_file(os_rail_path)
    tfn_rail_links = tfn_rail_links[tfn_rail_links['operationalstatus'] == 'Active']  # Exclude inactive links
    tfn_rail_links = tfn_rail_links[['osid', 'description', 'structure', 'physicallevel', 'railwayuse',
                                     'trackrepresentation', 'geometry']]
    tfn_rail_links = tfn_rail_links[
        ~tfn_rail_links['description'].isin(['Preserved', 'Funicular', 'Mineral', 'Static Museum'])]
    tfn_rail_links = tfn_rail_links.drop_duplicates(subset=['osid', 'geometry'])
    tfn_rail_links[['description', 'structure', 'physicallevel', 'railwayuse', 'trackrepresentation']] = (
        tfn_rail_links[['description', 'structure', 'physicallevel', 'railwayuse', 'trackrepresentation']]
        .replace(0,'N/A'))
    return tfn_rail_links

def clean_passenger_rail(tfn_rail_links):
    tfn_pass_rail = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_pass_rail = tfn_pass_rail[tfn_pass_rail['description'].isin(['Main Line', 'Main Line And Tram',
                                                                     'Main Line And Rapid Transport System'])]
    write_to_file(tfn_pass_rail, RAIL_OUT / "TfN OS Passenger Rail" / "tfn_pass_rail_links.shp")

def clean_freight_rail(tfn_rail_links):
    tfn_freight_rail = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Freight'])]
    write_to_file(tfn_freight_rail, RAIL_OUT / "TfN OS Freight Rail" / "tfn_freight_rail_links.shp")

### OTHER

def clean_other(boundary):
    '''Function to clean all other datasets ready for analysis'''
    clean_bus_stops(OTHER_IN / "Bus Stops", boundary)
    clean_petrol_stations(f"zip://{OTHER_IN / "poi_uk.zip"}!poi_uk.gpkg", boundary)
    clean_charging_points(OTHER_IN / "Zap Map Data - full data set Oct 25.csv", boundary)
    clean_ncn(OTHER_IN / "NCN Sustrans" / "National_Cycle_Network_Public.shp", boundary)

    os_mm_net_node = read_os_mm_node_network(
        OTHER_IN / "OS Multi-Modal Routing Network" / "OSMulti-modalRoutingNetwork.gpkg")
    clean_train_stations(os_mm_net_node, boundary)
    clean_tram_stations(os_mm_net_node, boundary)
    clean_metro_stations(os_mm_net_node, boundary)
    clean_ferry_terminals(os_mm_net_node, boundary)
    clean_bus_coach_stations(os_mm_net_node, boundary)

def clean_bus_stops(path, boundary):
    '''Function to clean all bus stops datasets ready for analysis'''
    bus_stops_ne = pd.read_csv(path / "bus_stops_ne.csv")  # North East
    bus_stops_nw = pd.read_csv(path / "bus_stops_nw.csv")  # North West
    bus_stops_ys = pd.read_csv(path / "bus_stops_ys.csv")  # Yorkshire

    bus_stops = pd.concat([bus_stops_ne, bus_stops_nw, bus_stops_ys], ignore_index=True)  # Combine bus stops
    bus_stops_gdf = df_to_gdf(bus_stops, 'stop_lon', 'stop_lat', "EPSG:4326")
    bus_stops_gdf = bus_stops_gdf[['stop_id', 'stop_name', 'geometry']]  # Filter out columns
    bus_stops_gdf = bus_stops_gdf.drop_duplicates(subset=['stop_id', 'geometry'])  # Remove duplicate rows
    tfn_bus_stops = clip_to_boundary(bus_stops_gdf, boundary)  # Clip to TfN boundary
    write_to_file(tfn_bus_stops, OTHER_OUT / "TfN Bus Stops" / "tfn_bus_stops.shp")

def clean_petrol_stations(path, boundary):
    '''Function to clean POI data to get petrol stations ready for analysis'''
    poi_uk = gpd.read_file(path)
    petrol_stations = poi_uk[poi_uk['main_category'] == 'gas_station']
    petrol_stations = petrol_stations[['id', 'geometry']]
    petrol_stations = petrol_stations.drop_duplicates(subset=['id', 'geometry'])
    tfn_petrol = clip_to_boundary(petrol_stations, boundary)
    write_to_file(tfn_petrol, OTHER_OUT / "TfN Petrol Stations" / "tfn_petrol_stations.shp")

def read_os_mm_node_network(path):
    os_mm_net_node = gpd.read_file(path, layer="mrn_ntwk_transportnode")
    os_mm_net_node.drop(columns=['os_parentid', 'name'], inplace=True)
    os_mm_net_node.drop_duplicates(subset=['nodeid', 'geometry'], inplace=True)
    return os_mm_net_node

def clean_train_stations(os_mm_net_node, boundary):
    train_stations = os_mm_net_node[
        (os_mm_net_node['os_nodetype'] == 'Railway Station;Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station;Railway Station (Underground System);Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station;Tram Station;Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station (Non Public Accessible);Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station (Principal);Tram Station;Modal Change')
        ]
    train_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_train_stations = clip_to_boundary(train_stations, boundary)
    write_to_file(tfn_train_stations, OTHER_OUT / "TfN OS Train Stations" / "tfn_train_stations.shp")

def clean_tram_stations(os_mm_net_node, boundary):
    tram_stations = os_mm_net_node[os_mm_net_node['os_nodetype'].str.contains('Tram Station', case=False, na=False)]
    tram_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_tram_stations = clip_to_boundary(tram_stations, boundary)
    write_to_file(tfn_tram_stations, OTHER_OUT / "TfN OS Tram Stations" / "tfn_tram_stations.shp")

def clean_metro_stations(os_mm_net_node, boundary):
    metro_stations = os_mm_net_node[
        os_mm_net_node['os_nodetype'].str.contains('Underground System', case=False, na=False)]
    metro_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_metro_stations = clip_to_boundary(metro_stations, boundary)
    write_to_file(tfn_metro_stations, OTHER_OUT / "TfN OS Metro Stations" / "tfn_metro_stations.shp")

def clean_ferry_terminals(os_mm_net_node, boundary):
    ferry_stations = os_mm_net_node[os_mm_net_node['os_nodetype'].str.contains('Ferry', case=False, na=False)]
    ferry_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_ferry_stations = clip_to_boundary(ferry_stations, boundary)
    write_to_file(tfn_ferry_stations, OTHER_OUT / "TfN OS Ferry Stations" / "tfn_ferry_stations.shp")

def clean_bus_coach_stations(os_mm_net_node, boundary):
    bus_coach_stations = os_mm_net_node[
        (os_mm_net_node['os_nodetype'].str.contains('Bus Station', case=False, na=False)) |
        (os_mm_net_node['os_nodetype'] == 'Coach Station;Modal Change')
        ]
    bus_coach_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_bus_coach_stations = clip_to_boundary(bus_coach_stations, boundary)
    write_to_file(tfn_bus_coach_stations, OTHER_OUT / "TfN OS Bus Coach Stations" / "tfn_bus_coach_stations.shp")

def clean_tram_network(tfn_rail_links):
    tfn_tram_links = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_tram_links = tfn_tram_links[tfn_tram_links['description'].isin(['Tram', 'Main Line And Tram'])]
    write_to_file(tfn_tram_links, OTHER_OUT / "TfN OS Tram Links" / "tfn_os_tram_links.shp")

def clean_rapid_transport_network(tfn_rail_links):
    tfn_rapid_transport = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_rapid_transport = tfn_rapid_transport[tfn_rapid_transport['description'].isin(
        ['Rapid Transport System', 'Main Line And Rapid Transport System'])]
    write_to_file(tfn_rapid_transport, OTHER_OUT / "TfN Rapid Transport" / "tfn_rapid_transport_links.shp")

def clean_charging_points(path, boundary):
    chg_pts = pd.read_csv(path)

    chg_pts_gdf = gpd.GeoDataFrame(chg_pts, geometry=[Point(xy) for xy in zip(chg_pts['lon'], chg_pts['lat'])], crs="EPSG:4326")
    chg_pts_gdf = chg_pts_gdf[['identifier', 'name', 'speed', 'value', 'geometry']]
    chg_pts_gdf.rename(columns={'identifier': 'id', 'value': 'devices'}, inplace=True)
    chg_pts_gdf = chg_pts_gdf.drop_duplicates(subset=['geometry'])
    chg_pts_gdf = chg_pts_gdf[~chg_pts_gdf.geometry.is_empty]
    chg_pts_gdf = chg_pts_gdf[chg_pts_gdf.geometry.notnull()]
    tfn_chg_pts = clip_to_boundary(chg_pts_gdf, boundary)
    write_to_file(tfn_chg_pts, OTHER_OUT / "TfN EV Charging Sites" / "tfn_chg_sites.shp")

def clean_ncn(path, boundary):
    ncn = gpd.read_file(path)

    ncn.drop(columns=['RouteCat', 'OpenStatus', 'GlobalID'], inplace=True)
    ncn.drop_duplicates(subset=['SegmentID', 'geometry'], inplace=True)
    ncn_cols_replace = ['Desc_', 'Greenway', 'RouteType', 'RouteNo', 'LinkNo', 'Surface',
                        'Quality', 'Lighting', 'RoadClass']
    ncn[ncn_cols_replace] = ncn[ncn_cols_replace].replace(0, 'N/A')
    tfn_ncn = clip_to_boundary(ncn, boundary)
    write_to_file(tfn_ncn, OTHER_OUT / "TfN NCN" / "tfn_ncn.shp")

## HAZARDS

def clean_hazards(boundary):
    clean_extreme_weather(boundary)
    clean_flooding(boundary)
    clean_ground_stability(boundary)
    clean_coastal_erosion(boundary)



### EXTREME WEATHER

def clean_extreme_weather(boundary):
    '''Function to clean all extreme weather variables'''
    tfn_common_grid = clean_common_grid(f"zip://{EXTREME_WEATHER_IN / "Summer_Maximum_Temperature_Change___Projections_12km_grid.zip"}"
        "!summer_maximum_temperature_change_projections_12km.shp", boundary)

    clean_temp_max(f"zip://{EXTREME_WEATHER_IN / "Summer_Maximum_Temperature_Change___Projections_12km_grid.zip"}"
        "!summer_maximum_temperature_change_projections_12km.shp", tfn_common_grid)
    clean_temp_min(f"zip://{EXTREME_WEATHER_IN / "Winter_Minimum_Temperature_Change___Projections_12km_grid.zip"}"
                   "!winter_minimum_temperature_change_projections_12km.shp", tfn_common_grid)
    clean_summer_precip(f"zip://{EXTREME_WEATHER_IN / "Summer_Precipitation_Change___Projections_12km_grid.zip"}"
                        "!summer_precipitation_change_projections_12km.shp", tfn_common_grid)
    clean_winter_precip(f"zip://{EXTREME_WEATHER_IN / "Winter_Precipitation_Change___Projections_12km_grid.zip"}"
                        "!winter_precipitation_change_projections_12km.shp", tfn_common_grid)
    clean_rain_days(f"zip://{EXTREME_WEATHER_IN / "Annual_Count_of_10mm_Rain_Days_1991_2020.zip"}"
                    "!Annual_Count_of_10mm_Rain_Days_1991-2020.shp", boundary)
    clean_drought_index(f"zip://{EXTREME_WEATHER_IN / "Drought_Severity_Index_12_Month_Accumulations.zip"}"
              "!Drought_Severity_Index_12_Month_Accumulations_-_Projections.shp", boundary)
    clean_hot_summer_days(f"zip://{EXTREME_WEATHER_IN / "Annual_Count_of_Hot_Days___Projections__12km_grid.zip"}"
                          "!annual_count_of_hot_summer_days_projections_12km.shp", tfn_common_grid)
    clean_extreme_summer_days(f"zip://{EXTREME_WEATHER_IN 
                                       / "Annual_Count_of_Extreme_Summer_Days_Projections_12km_Grid.zip"}"
                              "!annual_count_of_extreme_summer_days_projections_12km.shp", tfn_common_grid)
    clean_frost_days(f"zip://{EXTREME_WEATHER_IN / "Annual_Count_of_Frost_Days_Projections_12km_Grid.zip"}"
                     "!annual_count_of_frost_days_projections_12km.shp", tfn_common_grid)
    clean_icing_days(f"zip://{EXTREME_WEATHER_IN / "Annual_Count_of_Icing_Days___Projections__12km_grid.zip"}"
                     "!annual_count_of_icing_days_projections_12km.shp", tfn_common_grid)
    clean_wind_speed(EXTREME_WEATHER_IN / "CEDA_Max_Wind_Speed", boundary)
    clean_wind_driven_rain(f"zip://{EXTREME_WEATHER_IN / "Annual_Index_of_Wind_Driven_Rain_Projections_5km.zip"}"
                           "!Annual_Index_of_Wind_Driven_Rain_-_Projections_(5km).shp", boundary)

def clean_common_grid(path, boundary):
    temp_max = gpd.read_file(path)
    temp_max['grid_id'] = range(1, len(temp_max) + 1)

    common_grid = temp_max[['grid_id', 'geometry']]
    tfn_common_grid = clip_to_boundary(common_grid, boundary)
    tfn_common_grid = explode_to_polygons(tfn_common_grid)
    write_to_file(tfn_common_grid, MODEL_INPUT_PATH / "Other" / "TfN Common Grid" / "tfn_common_grid.shp")
    return tfn_common_grid

def clean_temp_max(path, grid):
    temp_max = gpd.read_file(path)
    temp_max['grid_id'] = range(1, len(temp_max) + 1)
    temp_max = temp_max[['grid_id', 'tasmax_s_4', 'tasmax__22']]
    temp_max.rename(columns={'tasmax_s_4': 'tasmax_s_c', 'tasmax__22': 'tasmax_s_f'}, inplace=True)
    temp_max['tasmax_s_f'] = temp_max['tasmax_s_c'] + temp_max['tasmax_s_f']
    tfn_temp_max = temp_max[temp_max['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_temp_max,
                  EXTREME_WEATHER_OUT / "TfN Summer Max Temperature Change Projections" / "tfn_temp_max.csv", csv=True)

def clean_temp_min(path, grid):
    temp_min = gpd.read_file(path)
    temp_min['grid_id'] = range(1, len(temp_min) + 1)
    temp_min = temp_min[['grid_id', 'tasmin_w_4', 'tasmin__22']]
    temp_min.rename(columns={'tasmin_w_4': 'tasmin_w_c', 'tasmin__22': 'tasmin_w_f'}, inplace=True)
    temp_min['tasmin_w_f'] = temp_min['tasmin_w_c'] + temp_min['tasmin_w_f']
    tfn_temp_min = temp_min[temp_min['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_temp_min,
                  EXTREME_WEATHER_OUT / "TfN Winter Min Temperature Change Projections" / "tfn_temp_min.csv", csv=True)

def clean_summer_precip(path, grid):
    precip_sum = gpd.read_file(path)
    precip_sum['grid_id'] = range(1, len(precip_sum) + 1)
    precip_sum = precip_sum[['grid_id', 'pr_summe_3', 'pr_summ_21']]
    precip_sum.rename(columns={'pr_summe_3': 'pr_s_c','pr_summ_21': 'pr_s_pct_f'}, inplace=True)
    precip_sum['pr_s_f'] = precip_sum['pr_s_c'] * (1 + (precip_sum['pr_s_pct_f'] / 100))
    precip_sum.drop(columns=['pr_s_pct_f'], inplace=True)
    tfn_precip_sum = precip_sum[precip_sum['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_precip_sum,
                  EXTREME_WEATHER_OUT / "TfN Summer Precipitation Change Projections" / "tfn_precip_sum.csv", csv=True)

def clean_winter_precip(path, grid):
    precip_win = gpd.read_file(path)
    precip_win['grid_id'] = range(1, len(precip_win) + 1)
    precip_win = precip_win[['grid_id', 'pr_winte_3', 'pr_wint_21']]
    precip_win.rename(columns={'pr_winte_3': 'pr_w_c','pr_wint_21': 'pr_w_pct_f',}, inplace=True)
    precip_win['pr_w_f'] = precip_win['pr_w_c'] * (1 + (precip_win['pr_w_pct_f'] / 100))
    precip_win.drop(columns=['pr_w_pct_f'], inplace=True)
    tfn_precip_win = precip_win[precip_win['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_precip_win,
                  EXTREME_WEATHER_OUT / "TfN Winter Precipitation Change Projections" / "tfn_precip_win.csv", csv=True)

def clean_rain_days(path, boundary):
    rain_days = gpd.read_file(path)
    tfn_rain_days = clip_to_boundary(rain_days, boundary)
    tfn_rain_days = explode_to_polygons(tfn_rain_days)
    tfn_rain_days.rename(columns={'Rain10mmDa': 'rain_days_c'}, inplace=True)
    tfn_rain_days['rain_days_f'] = tfn_rain_days['rain_days_c'] # Duplicate rain days column
    write_to_file(tfn_rain_days, EXTREME_WEATHER_OUT / "TfN 10mm Rain Days 1991-2020" / "tfn_rain_days.shp")

def clean_drought_index(path, boundary):
    drought_index = gpd.read_file(path)
    drought_index = drought_index[['DSI12_ba_4', 'DSI12_40_m', 'geometry']]
    drought_index.rename(columns={'DSI12_ba_4': 'dsi_c', 'DSI12_40_m': 'dsi_f'}, inplace=True)
    tfn_drought = clip_to_boundary(drought_index, boundary)
    tfn_drought = explode_to_polygons(tfn_drought)
    write_to_file(tfn_drought, EXTREME_WEATHER_OUT / "TfN Drought Severity Index" / "tfn_drought_index.shp")

def clean_hot_summer_days(path, grid):
    hot_days = gpd.read_file(path)
    hot_days['grid_id'] = range(1, len(hot_days) + 1)
    hot_days = hot_days[['grid_id', 'HSD_base_4', 'HSD_40_med']]
    hot_days.rename(columns={'HSD_base_4': 'hsd_c', 'HSD_40_med': 'hsd_f'}, inplace=True)
    tfn_hot_days = hot_days[hot_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_hot_days, EXTREME_WEATHER_OUT / "TfN Hot Summer Days Projections" / "tfn_hot_days.csv", csv=True)

def clean_extreme_summer_days(path, grid):
    extr_days = gpd.read_file(path)
    extr_days['grid_id'] = range(1, len(extr_days) + 1)
    extr_days = extr_days[['grid_id', 'ESD_base_4', 'ESD_40_med']]
    extr_days.rename(columns={'ESD_base_4': 'esd_c','ESD_40_med': 'esd_f'}, inplace=True)
    tfn_extr_days = extr_days[extr_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_extr_days,
                  EXTREME_WEATHER_OUT / "TfN Extreme Summer Days Projections" / "tfn_extr_days.csv", csv=True)

def clean_frost_days(path, grid):
    frost_days = gpd.read_file(path)
    frost_days['grid_id'] = range(1, len(frost_days) + 1)
    frost_days = frost_days[['grid_id', 'FrostDay_3', 'FrostDa_18']]
    frost_days.rename(columns={'FrostDay_3': 'frost_d_c', 'FrostDa_18': 'frost_d_f'}, inplace=True)
    tfn_frost_days = frost_days[frost_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_frost_days, EXTREME_WEATHER_IN / "TfN Frost Days Projections" / "tfn_frost_days.csv", csv=True)

def clean_icing_days(path, grid):
    ice_days = gpd.read_file(path)
    ice_days['grid_id'] = range(1, len(ice_days) + 1)
    ice_days = ice_days[['grid_id', 'IcingDay_3', 'IcingDa_18']]
    ice_days.rename(columns={'IcingDay_3': 'ice_d_c','IcingDa_18': 'ice_d_f'}, inplace=True)
    tfn_ice_days = ice_days[ice_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_ice_days, EXTREME_WEATHER_OUT / "TfN Icing Days Projections" / "tfn_ice_days.csv", csv=True)

def clean_wind_speed(path, boundary):
    windspd_c = xr.open_dataset(path / "wsgmax10m_rcp85_land-cpm_uk_5km_01_day_20701201-20801130.nc").to_dataframe()
    windspd_f = xr.open_dataset(path / "wsgmax10m_rcp85_land-cpm_uk_5km_01_day_19901201-20001130.nc").to_dataframe()

    exc_c = calculate_exceedance(20, windspd_c, 'wsgmax10m', 'c')
    exc_f = calculate_exceedance(20, windspd_f, 'wsgmax10m', 'f')
    pct_c = calculate_percentile(windspd_c, [0.95, 0.99], 'wsgmax10m')
    pct_f = calculate_percentile(windspd_f, [0.95, 0.99], 'wsgmax10m')

    pct_c.columns = ['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude', 'p95_c', 'p99_c']
    pct_f.columns = ['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude', 'p95_f','p99_f']

    def merge_and_fill(df1, df2, fill_col):
        return pd.merge(df1, df2, on=['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude'],
                        how='outer').fillna({fill_col: 0})

    windspd_c_combined = merge_and_fill(pct_c, exc_c, 'avg_excd_c')
    windspd_f_combined = merge_and_fill(pct_f, exc_f, 'avg_excd_f')
    windspd_combined = merge_and_fill(windspd_c_combined, windspd_f_combined, 'avg_excd_f')

    windspd_combined['geometry'] = [convert_point_to_grid(x, y, 2500)
        for x, y in zip(windspd_combined['projection_x_coordinate'], windspd_combined['projection_y_coordinate'])]
    windspd_combined = gpd.GeoDataFrame(windspd_combined, geometry='geometry', crs="EPSG:27700")
    windspd_combined = windspd_combined[['p95_c', 'p99_c', 'avg_excd_c','p95_f', 'p99_f', 'avg_excd_f','geometry']]
    tfn_windspd = clip_to_boundary(windspd_combined, boundary)
    tfn_windspd = explode_to_polygons(tfn_windspd)
    write_to_file(tfn_windspd, EXTREME_WEATHER_OUT / "TfN Wind Speed Projections" / "tfn_windspd.shp")

def clean_wind_driven_rain(path, boundary):
    wdr = gpd.read_file(path)

    wdr_agg = (
        wdr.groupby(['x_coord', 'y_coord'])
        .agg({
            'WDR_base_1': 'mean',
            'WDR_40_Med': 'mean',
            'geometry': 'first'  # keep the first geometry for each group
        })
        .reset_index()
    )

    wdr_agg = wdr_agg[['WDR_base_1', 'WDR_40_Med', 'geometry']]
    wdr_agg.rename(columns={'WDR_base_1': 'wdr_c', 'WDR_40_Med': 'wdr_f'}, inplace=True)
    wdr_agg = gpd.GeoDataFrame(wdr_agg, geometry='geometry', crs='EPSG:3857')
    tfn_wdr = clip_to_boundary(wdr_agg, boundary)
    tfn_wdr = explode_to_polygons(tfn_wdr)
    write_to_file(tfn_wdr, EXTREME_WEATHER_OUT / "TfN Wind Driven Rain Index" / "tfn_wdr.shp")

### FLOODING

def clean_flooding(boundary):
    code_number_map = {'NT': ['50', '55'],
                       'NU': ['00', '05'],
                       'NX': ['50'],
                       'NY': ['00', '05', '50', '55'],
                       'NZ': ['00', '05', '50'],
                       'OV': ['00'],
                       'SD': ['00', '05', '50', '55'],
                       'SE': ['00', '05', '50', '55'],
                       'SJ': ['00', '05', '50', '55'],
                       'SK': ['00', '05', '50', '55'],
                       'TA': ['00', '05'],
                       'TF': ['00', '05', '50', '55'],
                       }

    extract_flood_data(code_number_map ,extract=False)

    clean_flood("RoFRS", "RoFRS", "v202501", boundary,
                 Path("TfN RoFRS CC") / "tfn_rofrs_cc.gpkg", True, code_number_map)
    clean_flood("RoFRS", "RoFRS", "v202501", boundary,
                 Path("TfN RoFRS") / "tfn_rofrs.gpkg", False, code_number_map)
    clean_flood("RoFSW CC", "RoFSW", "v202509", boundary,
                Path("TfN RoFSW CC") / "tfn_rofsw_cc.gpkg", True, code_number_map)
    clean_flood("RoFSW", "RoFSW", "v202509", boundary,
                Path("TfN RoFSW") / "tfn_rofsw.gpkg", False, code_number_map)

def extract_gdb_file(code, number, flood_data, version, cc):
    try:
        base_path = FLOODING_IN / flood_data
        if cc == True:
            zip_path = base_path / code / f"{flood_data}_Climate_Change_01_{code}{number}_{version}.zip"
            extract_to = base_path / code
            gdb_path = extract_to / f"{flood_data}_Climate_Change_01_{code}{number}_{version}.gdb"
        else:
            zip_path = base_path / code / f"{flood_data}_{code}{number}_{version}.zip"
            extract_to = base_path / code
            gdb_path = extract_to / f"{flood_data}_{code}{number}_{version}.gdb"

        # Check if zip file exists
        if not zip_path.exists():
          print(f"Zip file not found: {zip_path}")
          return None

        # Extract the contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # Check if GDB folder exists
        if not gdb_path.exists():
            print(f"GDB folder not found: {gdb_path}")
            return None

        layers = fiona.listlayers(gdb_path)
        if not layers:
            print("No layers found in:", gdb_path)
            return None

        print("Available layers:", layers)
        gdf = gpd.read_file(gdb_path, layer=layers[0])

        return gdf
    except Exception as e:
        print(f"Error processing {code}{number}: {e}")
        return None

def read_gdb(code, number, file_name, flood_data, version, cc):
    base_path = FLOODING_IN / flood_data / code

    if cc == True:
        gdb_path = base_path / f"{flood_data}_Climate_Change_01_{code}{number}_{version}.gdb"
    else:
        gdb_path = base_path / f"{flood_data}_{code}{number}_{version}.gdb"

    # Check if GDB folder exists
    if not gdb_path.exists():
        print(f"GDB folder not found: {gdb_path}")
        return None

    layers = fiona.listlayers(gdb_path)
    if not layers:
        print("No layers found in:", gdb_path)
        return None

    print("Available layers:", layers)
    gdf = gpd.read_file(gdb_path, layer=layers[0])

    return gdf

def extract_flood_data(code_number_map, extract):
    if extract == False:
        return

    for code in code_number_map.keys():
        for number in code_number_map[code]:
            # Climate Change Datasets
             extract_gdb_file(code, number, "RoFRS", "v202501", True)
             extract_gdb_file(code, number, "RoFSW", "v202509", True)

            # Present Day Datasets
             extract_gdb_file(code, number, "RoFRS", "v202501", False)
             extract_gdb_file(code, number, "RoFSW", "v202509", False)

def clean_flood(file_name, flood_type, version, boundary, out_path, cc, code_number_map):
    gdfs = []
    for code in code_number_map.keys():
        for number in code_number_map[code]:
            gdf = read_gdb(code, number, file_name, flood_type, version, cc)  # Read file
            tfn_gdf = clip_to_boundary(gdf, boundary)
            gdfs.append(tfn_gdf)  # Add to list
            print(code, number)
            del gdf, tfn_gdf

    flood_data = pd.concat(gdfs, ignore_index=True)
    flood_data = extract_poly_from_geomcollection(flood_data)
    flood_data = flood_data[['Risk_band', 'geometry']]
    write_to_file(flood_data, FLOODING_OUT / out_path, driver="GPKG")

### GROUND STABILITY

def clean_ground_stability(boundary):
    clean_geosure(f"zip://{GROUND_STABILITY_IN / "GeoSureHexGrids.zip"}!/GeoSureHexGrids/Data", boundary)
    clean_geoclimate(GROUND_STABILITY_IN / "GeoClimateUKCP18OpenData" / "GeoclimateUKCP18_Open", boundary)

def clean_geosure(path, boundary):
    geosure_layers = {
        'cd': gpd.read_file(path + "/GB_Hex_5km_GS_CollapsibleDeposits_v8.shp"),
        'cg': gpd.read_file(path + "/GB_Hex_5km_GS_CompressibleGround_v8.shp"),
        'ls': gpd.read_file(path + "/GB_Hex_5km_GS_Landslides_v8.shp"),
        'rs': gpd.read_file(path + "/GB_Hex_5km_GS_RunningSand_v8.shp"),
        'ss': gpd.read_file(path + "/GB_Hex_5km_GS_ShrinkSwell_v8.shp"),
        'sr': gpd.read_file(path + "/GB_Hex_5km_GS_SolubleRocks_v8.shp")
    }

    tfn_geosure_layers = {}
    for code, df in geosure_layers.items():
        df.rename(columns={'CLASS': f'{code}_risk'}, inplace=True)
        tfn_geosure_layers[code] = clip_to_boundary(df, boundary)
        tfn_geosure_layers[code] = explode_to_polygons(tfn_geosure_layers[code])

    # Merge layers based on nearest centroids
    base_code = 'cd'
    tfn_geosure = tfn_geosure_layers[base_code][['cd_risk', 'geometry']].copy()
    for code, layer in tfn_geosure_layers.items():
        if code == base_code:
            continue  # skip the base layer
        layer_subset = layer[[f'{code}_risk', 'geometry']] # Select only the relevant class and geometry columns
        matched = nearest_centroids(tfn_geosure, layer_subset) # Apply nearest centroid matching
        tfn_geosure[f'{code}_risk'] = matched[f'{code}_risk'] # Add the matched CLASS column to the base dataframe

    tfn_geosure = tfn_geosure[['cd_risk', 'cg_risk', 'ls_risk', 'rs_risk', 'ss_risk', 'sr_risk', 'geometry']]
    write_to_file(tfn_geosure, GROUND_STABILITY_OUT / "TfN GeoSure" / "tfn_geosure.shp")

def clean_geoclimate(path, boundary):
    for year in ['2030', '2070']:
        gdf = gpd.read_file(path / f"GeoClimateUKCP18_ShrinkSwell_{year}_Average_Open.shp")
        gdf.rename(columns={'CLASS': 'ss_geo_risk'}, inplace=True)
        gdf = gdf[['ss_geo_risk', 'geometry']]
        tfn_gdf = clip_to_boundary(gdf, boundary)
        tfn_gdf = explode_to_polygons(tfn_gdf)
        write_to_file(tfn_gdf, GROUND_STABILITY_OUT / "BGS Shrink Swell" / year / f"tfn_bgs_ss_{year}.shp")

### COASTAL EROSION

def clean_coastal_erosion(boundary):
    clean_giz(f"zip://{COASTAL_EROSION_IN / "National_Coastal_Erosion_Risk_Mapping_NCERM_National_2024.shp.zip"}"
              "!NCERM_Ground_Instability_Zone.shp", boundary)
    clean_ncerm(f"zip://{COASTAL_EROSION_IN / "National_Coastal_Erosion_Risk_Mapping_NCERM_National_2024.shp.zip"}!",
                boundary)

def clean_giz(path, boundary):
    ncerm_giz = gpd.read_file(path)
    ncerm_giz = ncerm_giz[['smp_no', 'geometry']]
    tfn_ncerm_giz = clip_to_boundary(ncerm_giz, boundary)
    tfn_ncerm_giz = explode_to_polygons(tfn_ncerm_giz)
    write_to_file(tfn_ncerm_giz, COASTAL_EROSION_OUT / "NCERM" / "Ground Instability Zones" / "tfn_ncerm_giz.shp")

def clean_ncerm(path, boundary):
    for year in ['2055', '2105']:
        gdf = gpd.read_file(path + f"NCERM_SMP_{year}_70CC.shp")
        gdf = gdf[['smp_name', 'geometry']]
        tfn_gdf = clip_to_boundary(gdf, boundary)
        tfn_gdf = explode_to_polygons(tfn_gdf)
        write_to_file(tfn_gdf, COASTAL_EROSION_OUT / "NCERM" / f"SMP_{year}_70CC" / f"tfn_ncerm_smp_{year}_70CC.shp")

## IMPACT

def clean_impact(boundary):
    clean_freight_demand(boundary)
    clean_noham_flows()

### FREIGHT

def clean_freight_demand(boundary):
    tfn_freight_network_demand = read_freight_demand(IMPACT_IN / "Freight" / "rail_freight_network_demand.gpkg", boundary)
    tfn_os_freight_network_demand = map_freight_networks(tfn_freight_network_demand,
                         INFRASTRUCTURE_OUT / "Rail" / "TfN OS Freight Rail" / "tfn_freight_rail_links.shp")
    write_to_file(tfn_os_freight_network_demand, IMPACT_OUT / "TfN Freight Flows" / "tfn_freight_network_demand.gpkg")

def read_freight_demand(path, boundary):
    freight_network_demand = gpd.read_file(path)
    freight_network_demand = freight_network_demand[['dij_id', '2022_23_total', '2050_51 sc2_total', 'geometry']]
    tfn_freight_network_demand = clip_to_boundary(freight_network_demand, boundary)
    return tfn_freight_network_demand

def map_freight_networks(tfn_freight_network_demand, os_path):
    tfn_os_freight_rail = gpd.read_file(os_path)
    tfn_os_freight_rail = tfn_os_freight_rail.to_crs(tfn_freight_network_demand.crs)
    tfn_os_freight_network_demand = gpd.sjoin_nearest(
        tfn_os_freight_rail,
        tfn_freight_network_demand,
        how="left",
        max_distance=500,
        distance_col="distance"
    )
    tfn_os_freight_network_demand[['2022_23_total', '2050_51 sc2_total']] = tfn_os_freight_network_demand[
        ['2022_23_total', '2050_51 sc2_total']].fillna(0)
    tfn_os_freight_network_demand.drop(columns=['index_right'], inplace=True)
    return tfn_os_freight_network_demand

### NoHAM

def clean_noham_flows():
    link_flows = aggregate_link_flows_year(IMPACT_IN / "NoHAM Link Flows" / "input h5s.7z",
                              IMPACT_IN / "NoHAM Link Flows" / "h5 files")

    tfn_noham_flows = link_flows['2023'].merge(
        link_flows['2048'],
        on='link_id',
        suffixes=('_c', '_f'),  # _c for current, _f for future
        how='outer'  # Keep all rows, fill missing with NA
    )

    tfn_noham_flows_c = link_flows['2023']
    tfn_noham_flows_f = link_flows['2048']

    tfn_noham_net_flows_c = merge_noham_flow_network(tfn_noham_flows_c,
                                                     ROAD_OUT / "TfN NoHAM 2023" / "tfn_noham_2023.shp")
    tfn_noham_net_flows_f = merge_noham_flow_network(tfn_noham_flows_f,
                                                     ROAD_OUT / "TfN NoHAM 2023" / "tfn_noham_2048.shp")

    write_to_file(tfn_noham_net_flows_c,
                  IMPACT_OUT /  "TfN NoHAM Flows" / "2023" / "tfn_noham_net_flows_c.gpkg", driver="GPKG")
    write_to_file(tfn_noham_net_flows_f,
                  IMPACT_OUT / "TfN NoHAM Flows" / "2048" / "tfn_noham_net_flows_f.gpkg", driver="GPKG")

def read_noham_h5(year, time_period, user_class, noham_path, output_path, extract):
    '''Reads NoHAM h5 files and extracts the link, routes, and od's DataFrames'''
    if extract == True:
        with py7zr.SevenZipFile(noham_path, mode='r') as archive:
            archive.extract(
                path=output_path,
                targets=[f"input h5s/{year}/NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5"])

    with h5py.File(output_path / "input h5s" / year /
                   f"NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5", 'r') as f:
        # Get OD's
        od_columns = [x.decode() for x in f['data/OD/block0_items'][:]]
        od_values = f['data/OD/block0_values'][:]

        od_labels = [
            f['data/OD/axis1_label0'][:],
            f['data/OD/axis1_label1'][:],
            f['data/OD/axis1_label2'][:],
            f['data/OD/axis1_label3'][:],
            f['data/OD/axis1_label4'][:]
        ]

        # Get routes
        route_col = [x.decode() for x in f['data/Route/block0_items']][:][0]
        route_values = f['data/Route/block0_values'][:]

        # MultiIndex labels
        route_label0 = f['data/Route/axis1_label0'][:]
        route_label1 = f['data/Route/axis1_label1'][:]

        # Get links
        link_values = f['data/link/block0_values'][:]
        link_columns = [x.decode() for x in f['data/link/block0_items'][:]]

    # Build OD Dataframe
    od_multi_index = pd.MultiIndex.from_arrays(od_labels, names=['o', 'd', 'route', 'uc', 'total_links'])
    od_df = pd.DataFrame(od_values, index = od_multi_index, columns = od_columns)

    # Build Route Dataframe
    route_multi_index = pd.MultiIndex.from_arrays([route_label0, route_label1], names=['route', 'link_id'])
    route_df = pd.DataFrame(route_values, index=route_multi_index, columns=[route_col])

    # Build Link DataFrame
    link_df = pd.DataFrame(link_values, columns=link_columns)

    return od_df, route_df, link_df

def aggregate_link_flows(ods, routes, links):
    '''Takes NoHAM ODs, Routes, and Links to create aggregated link flows DataFrame'''
    # Flatten OD and Route data
    od_flat = ods.reset_index()[['route', 'abs_demand']]
    route_flat = routes.reset_index()[['route', 'link_id']]

    # Merge OD demand with route links
    od_links = od_flat.merge(route_flat, on='route')

    # Aggregate demand per link_id
    link_demand = od_links.groupby('link_id')['abs_demand'].sum().reset_index()

    link_flows = pd.merge(link_demand, links, left_on='link_id', right_index=True)

    return link_flows

def aggregate_link_flows_year(noham_path, output_path):
    years = ["2023", "2048"]
    time_periods = ["TS1", "TS2", "TS3"]
    user_classes = ["uc1", "uc2", "uc3", "uc4", "uc5"]

    link_flows = {}
    for year in years:
        ts_dfs = []
        print(year)
        for time_period in time_periods:
            print(time_period)
            uc_dfs = []
            for user_class in user_classes:
                print(user_class)
                od_df, route_df, link_df = read_noham_h5(year, time_period, user_class, noham_path, output_path, False)
                link_demand = aggregate_link_flows(od_df, route_df, link_df)  # Get link based demand
                link_demand = link_demand.rename(
                    columns={'abs_demand': f'{user_class}_{time_period}'})  # Rename demand column
                link_demand['link_id'] = link_demand['a'].astype(str) + '_' + link_demand['b'].astype(
                    str)  # Create unique noham link id
                link_demand = link_demand[['link_id', f'{user_class}_{time_period}']]  # Keep relevant columns
                uc_dfs.append(link_demand)  # Add to list of df's

            # Merge all user class dataframes
            combined_uc_df = uc_dfs[0]
            for df_uc in uc_dfs[1:]:
                combined_uc_df = combined_uc_df.merge(df_uc, on='link_id', how='outer')

            # Compute total demand for all vehicles for each time period
            combined_uc_df[f'all_vehs_{time_period}'] = combined_uc_df[
                [f"{uc}_{time_period}" for uc in user_classes]].sum(axis=1)

            # Store result
            ts_dfs.append(combined_uc_df)

        # Merge all time period dataframes
        combined_ts_df = ts_dfs[0]
        for df_ts in ts_dfs[1:]:
            combined_ts_df = combined_ts_df.merge(df_ts, on='link_id', how='outer')

        # Compute totals for each user class across all time periods
        for uc in user_classes:
            combined_ts_df[f"{uc}_total"] = combined_ts_df[[f"{uc}_{tp}" for tp in time_periods]].sum(axis=1)

        # Compute total of each user class across all time periods
        combined_ts_df['all_vehs_total'] = combined_ts_df[[f'all_vehs_{tp}' for tp in time_periods]].sum(axis=1)

        # Add to data dictionary
        link_flows[year] = combined_ts_df

        return link_flows

def merge_noham_flow_network(tfn_noham_flows, noham_path):
    tfn_noham_link = gpd.read_file(noham_path)
    tfn_noham_net_flows = pd.merge(
        tfn_noham_link,
        tfn_noham_flows,
        on='link_id',
        how='left'  # Keep all network, adding flows where available
    )
    return tfn_noham_net_flows

















