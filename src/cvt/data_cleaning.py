'''This script will take raw input data and clean it ready for model input'''

### LOAD LIBRARIES
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, box
from shapely import wkt
import numpy as np

import xarray as xr
import h5py
import py7zr
import io

import matplotlib.pyplot as plt
import contextily as ctx

from pathlib import Path
import zipfile

### FILE PATHS
RAW_INPUT_PATH = Path("F:/") / "4. Climate Datasets" / "Climate Vulnerability Tool" / "raw inputs"
OUTPUT_PATH = Path("F:/") / "4. Climate Datasets" / "Climate Vulnerability Tool" / "model inputs"

INFRASTRUCTURE_IN = RAW_INPUT_PATH / "Infrastructure"
INFRASTRUCTURE_OUT = OUTPUT_PATH / "Infrastructure"

ROAD_IN = INFRASTRUCTURE_IN / "Road"
RAIL_IN = INFRASTRUCTURE_IN / "Rail"
OTHER_IN = INFRASTRUCTURE_IN / "Other"

ROAD_OUT = INFRASTRUCTURE_OUT / "Road"
RAIL_OUT = INFRASTRUCTURE_OUT / "Rail"
OTHER_OUT = INFRASTRUCTURE_OUT / "Other"

HAZARD_IN = RAW_INPUT_PATH / "Hazard"
HAZARD_OUT = OUTPUT_PATH / "Hazard"

EXTREME_WEATHER_IN = HAZARD_IN / "Extreme Weather"

EXTREME_WEATHER_OUT = HAZARD_OUT / "Extreme Weather"

def main():
    data_cleaning(OTHER_IN / "TfN_Boundary" / "Transport_for_the_north_boundary_2020_generalised.shp")
### DEFINE FUNCTIONS

def clip_to_boundary(gdf, boundary):
    '''Takes a GeoDataFrame and a boundary and returns a GeoDataFrame clipped to that boundary'''
    boundary = boundary.to_crs(gdf.crs) # Match CRS
    gdf_boundary = gpd.clip(gdf, boundary) # Clip GDF to boundary
    return gdf_boundary

def standard_cleaning(gdf, boundary, columns_to_keep, rename_map, replace_na_cols, out_path):
    '''Takes a GeoDataFrame, performs standard cleaning operations, and returns a GeoDataFrame'''
    # Attribute cleaning
    gdf = gdf.drop_duplicates(subset=['geometry']) # Drop duplicate rows
    gdf = gdf[columns_to_keep] # Filter for relevant columns
    gdf = gdf.rename(columns=rename_map) # Rename columns
    gdf[replace_na_cols] = gdf[replace_na_cols].replace(0, pd.NA) # Replace 0s with NA

    # Spatial cleaning
    gdf = gdf[~gdf.geometry.is_empty] # Remove empty geometries
    gdf = gdf[gdf.geometry.notnull()] # Remove null geometries
    tfn_gdf = clip_to_boundary(gdf, boundary) # Clip to spatial boundary
    tfn_gdf.to_file(out_path) # Write to file

    return tfn_gdf

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

def read_noham_h5(year, time_period, user_class, NOHAM_FLOWS_PATH, OUTPUT_DIR):
    '''Reads NoHAM h5 files and extracts the link, routes, and od's DataFrames'''
    with py7zr.SevenZipFile(NOHAM_FLOWS_PATH, mode='r') as archive:
        archive.extract(path=OUTPUT_DIR, targets=[f"input h5s/{year}/NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5"])

    with h5py.File(OUTPUT_DIR / "input h5s" / year / f"NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5", 'r') as f:
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

# DATA CLEANING

def data_cleaning(boundary_path):
    '''Function to clean all datasets ready for analysis'''
    boundary = read_boundary_path(boundary_path)

    clean_infrastruture(boundary)
    clean_hazards(boundary)


## TfN Boundary
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
    os_road.drop_duplicates(subset=['identifier', 'geometry'], inplace=True)  # Drop duplicate rows
    os_road = os_road[['identifier', 'roadNumber', 'name1', 'function', 'geometry']]  # Filter for relevant columns
    os_road.rename(columns={'name1': 'name', 'roadNumber': 'road_number'}, inplace=True)  # Rename columns
    os_road[['road_number', 'name', 'function']] = (
        os_road[['road_number', 'name', 'function']].replace(0, pd.NA))  # Replace 0s with NA
    os_road = os_road[~os_road.geometry.is_empty]  # Remove empty geometries
    os_road = os_road[os_road.geometry.notnull()]  # Remove null geometries
    tfn_os_road = clip_to_boundary(os_road, boundary)  # Clip to spatial boundary
    tfn_os_road.to_file(ROAD_OUT / "TfN OS Road" / "tfn_os_road.shp")  # Write to file

def clean_noham_roads(noham_roads_path_2023, noham_roads_path_2048, boundary):
    noham_2023 = gpd.read_file(noham_roads_path_2023)
    noham_2048 = gpd.read_file(noham_roads_path_2048)

    noham = {'2023': noham_2023, '2048': noham_2048}
    for year, noham_network in noham.items():
        noham_network.drop_duplicates(subset=['link_id', 'geometry'], inplace=True)
        noham_network = noham_network[['link_id', 'geometry']]
        noham_network = noham_network[~noham_network.geometry.is_empty]
        noham_network = noham_network[noham_network.geometry.notnull()]
        tfn_noham_network = clip_to_boundary(noham_network, boundary)
        tfn_noham_network.to_file(ROAD_OUT / f"TfN NoHAM {year}" / f"tfn_noham_{year}.shp")

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
    tfn_pass_rail.to_file(RAIL_OUT / "TfN OS Passenger Rail" / "tfn_pass_rail_links.shp")

def clean_freight_rail(tfn_rail_links):
    tfn_freight_rail = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Freight'])]
    tfn_freight_rail.to_file(RAIL_OUT / "TfN OS Freight Rail" / "tfn_freight_rail_links.shp")

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
    tfn_bus_stops.to_file(OTHER_OUT / "TfN Bus Stops" / "tfn_bus_stops.shp")

def clean_petrol_stations(path, boundary):
    '''Function to clean POI data to get petrol stations ready for analysis'''
    poi_uk = gpd.read_file(path)
    petrol_stations = poi_uk[poi_uk['main_category'] == 'gas_station']
    petrol_stations = petrol_stations[['id', 'geometry']]
    petrol_stations = petrol_stations.drop_duplicates(subset=['id', 'geometry'])
    tfn_petrol = clip_to_boundary(petrol_stations, boundary)
    tfn_petrol.to_file(OTHER_OUT / "TfN Petrol Stations" / "tfn_petrol_stations.shp")

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
    tfn_train_stations.to_file(OTHER_OUT / "TfN OS Train Stations" / "tfn_train_stations.shp")

def clean_tram_stations(os_mm_net_node, boundary):
    tram_stations = os_mm_net_node[os_mm_net_node['os_nodetype'].str.contains('Tram Station', case=False, na=False)]
    tram_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_tram_stations = clip_to_boundary(tram_stations, boundary)
    tfn_tram_stations.to_file(OTHER_OUT / "TfN OS Tram Stations" / "tfn_tram_stations.shp")

def clean_metro_stations(os_mm_net_node, boundary):
    metro_stations = os_mm_net_node[
        os_mm_net_node['os_nodetype'].str.contains('Underground System', case=False, na=False)]
    metro_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_metro_stations = clip_to_boundary(metro_stations, boundary)
    tfn_metro_stations.to_file(OTHER_OUT / "TfN OS Metro Stations" / "tfn_metro_stations.shp")

def clean_ferry_terminals(os_mm_net_node, boundary):
    ferry_stations = os_mm_net_node[os_mm_net_node['os_nodetype'].str.contains('Ferry', case=False, na=False)]
    ferry_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_ferry_stations = clip_to_boundary(ferry_stations, boundary)
    tfn_ferry_stations.to_file(OTHER_OUT / "TfN OS Ferry Stations" / "tfn_ferry_stations.shp")

def clean_bus_coach_stations(os_mm_net_node, boundary):
    bus_coach_stations = os_mm_net_node[
        (os_mm_net_node['os_nodetype'].str.contains('Bus Station', case=False, na=False)) |
        (os_mm_net_node['os_nodetype'] == 'Coach Station;Modal Change')
        ]
    bus_coach_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_bus_coach_stations = clip_to_boundary(bus_coach_stations, boundary)
    tfn_bus_coach_stations.to_file(OTHER_OUT / "TfN OS Bus Coach Stations" / "tfn_bus_coach_stations.shp")

def clean_tram_network(tfn_rail_links):
    tfn_tram_links = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_tram_links = tfn_tram_links[tfn_tram_links['description'].isin(['Tram', 'Main Line And Tram'])]
    tfn_tram_links.to_file(OTHER_OUT / "TfN OS Tram Links" / "tfn_os_tram_links.shp")

def clean_rapid_transport_network(tfn_rail_links):
    tfn_rapid_transport = tfn_rail_links[tfn_rail_links['railwayuse'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_rapid_transport = tfn_rapid_transport[tfn_rapid_transport['description'].isin(
        ['Rapid Transport System', 'Main Line And Rapid Transport System'])]
    tfn_rapid_transport.to_file(OTHER_OUT / "TfN Rapid Transport" / "tfn_rapid_transport_links.shp")

def clean_charging_points(path, boundary):
    chg_pts = pd.read_csv(path)

    chg_pts["geom"] = chg_pts["geom"].apply(wkt.loads)
    chg_pts_gdf = gpd.GeoDataFrame(chg_pts, geometry=chg_pts.geom, crs="EPSG:4326")
    chg_pts_gdf = chg_pts_gdf[['zapmap_device_uid', 'charge_device_name', 'geometry']]
    chg_pts_gdf.rename(columns={'zapmap_device_uid': 'zapmap_id', 'charge_device_name': 'name', })
    chg_pts_gdf.drop_duplicates(subset=['zapmap_id', 'geometry'], inplace=True)
    chg_pts_gdf = chg_pts_gdf[~chg_pts_gdf.geometry.is_empty]
    chg_pts_gdf = chg_pts_gdf[chg_pts_gdf.geometry.notnull()]
    tfn_chg_pts = clip_to_boundary(chg_pts_gdf, boundary)
    tfn_chg_pts.to_file(OTHER_OUT / "TfN Charging Points" / "tfn_chg_pts.shp")

def clean_ncn(path, boundary):
    ncn = gpd.read_file(path)

    ncn.drop(columns=['RouteCat', 'OpenStatus', 'GlobalID'], inplace=True)
    ncn.drop_duplicates(subset=['SegmentID', 'geometry'], inplace=True)
    ncn_cols_replace = ['Desc_', 'Greenway', 'RouteType', 'RouteNo', 'LinkNo', 'Surface',
                        'Quality', 'Lighting', 'RoadClass']
    ncn[ncn_cols_replace] = ncn[ncn_cols_replace].replace(0, 'N/A')
    tfn_ncn = clip_to_boundary(ncn, boundary)
    tfn_ncn.to_file(OTHER_OUT / "TfN NCN" / "tfn_ncn.shp")

## HAZARDS

def clean_hazards(boundary):
    clean_extreme_weather(boundary)

### EXTREME WEATHER

def clean_extreme_weather(boundary):
    '''Function to clean all extreme weather variables'''
    clean_temp_max(f"zip://{EXTREME_WEATHER_IN / "Summer_Maximum_Temperature_Change___Projections_12km_grid.zip"}"
        "!summer_maximum_temperature_change_projections_12km.shp", boundary)
    clean_temp_min(f"zip://{EXTREME_WEATHER_IN / "Winter_Minimum_Temperature_Change___Projections_12km_grid.zip"}"
                   "!winter_minimum_temperature_change_projections_12km.shp", boundary)



def clean_temp_max(path, boundary):
    temp_max = gpd.read_file(path)
    temp_max['grid_id'] = range(1, len(temp_max) + 1)
    temp_max = temp_max[['grid_id', 'tasmax_s_4', 'tasmax__22', 'geometry']]  # Select relevant columns
    temp_max.rename(columns={'tasmax_s_4': 'tasmax_s_b', 'tasmax__22': 'tasmax_s_f'}, inplace=True)
    temp_max['tasmax_s_f'] = temp_max['tasmax_s_b'] + temp_max['tasmax_s_f']
    tfn_temp_max = clip_to_boundary(temp_max, boundary)
    tfn_temp_max = explode_to_polygons(tfn_temp_max)
    tfn_temp_max.to_file(EXTREME_WEATHER_OUT / "TfN Summer Max Temperature Change Projections" / "tfn_temp_max.shp")

def clean_temp_min(path, boundary):
    temp_min = gpd.read_file(path)
    temp_min['grid_id'] = range(1, len(temp_min) + 1)
    temp_min = temp_min[['grid_id', 'tasmin_w_4', 'tasmin__22', 'geometry']]
    temp_min.rename(columns={'tasmin_w_4': 'tasmin_w_b', 'tasmin__22': 'tasmin_w_f'}, inplace=True)
    temp_min['tasmin_w_f'] = temp_min['tasmin_w_b'] + temp_min['tasmin_w_f']
    tfn_temp_min = clip_to_boundary(temp_min, boundary)
    tfn_temp_min = explode_to_polygons(tfn_temp_min)
    tfn_temp_min.to_file(EXTREME_WEATHER_OUT / "TfN Winter Min Temperature Change Projections" / "tfn_temp_min.shp")



