"""
Cleans raw input data to prepare it for input into the model
"""

### LOAD LIBRARIES
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
import xarray as xr
import h5py
import py7zr
from zipfile import ZipFile

### GENERAL FUNCTIONS

def clip_to_boundary(gdf, boundary):
    """Clip a GeoDataFrame to a spatial boundary"""
    boundary = boundary.to_crs(gdf.crs) # Match CRS
    gdf_boundary = gpd.clip(gdf, boundary) # Clip GDF to boundary
    return gdf_boundary

def write_to_file(df, output_path, driver=None, csv=False):
    """Write data to file, creating directory if necessary"""
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists, make one if not

    if df.empty:
        raise ValueError(f"GeoDataFrame is empty. Nothing written to {output_path}")

    if csv:
        df.to_csv(output_path, index=False)
    else:
        df.to_file(output_path, driver=driver)

def df_to_gdf(df, x_col, y_col, crs):
    """Takes a DataFrame and converts it to a GeoDataFrame using spatial columns"""
    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]  # Create geometry
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs) # Convert to GeoDataFrame
    return gdf

def convert_point_to_grid(x, y, size):
    """Takes a point and converts it to a grid of the given size"""
    return Polygon([
        (x - size, y - size),
        (x + size, y - size),
        (x + size, y + size),
        (x - size, y + size),
    ])

def extract_poly_from_geomcollection(gdf):
    """Extracts polygons from GeomCollection objects in a GeoDataFrame and turns them into new rows"""
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

def explode_to_polygons(gdf):
    """Explodes the MultiPolygons and GeomCollections in a GeoDataFrame into Polygons"""
    rows = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            new_row = row.copy()
            new_row['part'] = 0
            rows.append(new_row)
        elif geom.geom_type == 'MultiPolygon':
            for i, poly in enumerate(geom.geoms):
                new_row = row.copy()
                new_row.geometry = poly
                new_row['part'] = i
                rows.append(new_row)
        elif geom.geom_type == 'GeometryCollection':
            poly_count = 0
            for part in geom.geoms:
                if part.geom_type == 'Polygon':
                    new_row = row.copy()
                    new_row.geometry = part
                    new_row['part'] = poly_count
                    rows.append(new_row)
                    poly_count += 1

    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)

def nearest_centroids(gdf1, gdf2):
    """Takes two GeoDataFrames and merges them on their nearest centroids"""
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
    """Counts the number of exceedance days of a variable over a given threshold, then calculates the average over the
    timescale, per geometry"""

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
    """Calculates the percentiles of a given variable in a GeoDataFrame per geometry"""
    percentiles = (
        gdf.groupby(['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude'])[variable]
        .quantile(quantiles)
        .unstack()
        .reset_index()
    )

    return percentiles

# DATA CLEANING

def data_cleaning(cfg):
    """Clean all input datasets ready for analysis"""
    boundary = gpd.read_file(cfg.paths.boundary_path)

    clean_infrastructure(cfg, boundary)
    clean_hazards(cfg, boundary)
    clean_impact(cfg, boundary)


## INFRASTRUCTURE

def clean_infrastructure(cfg, boundary):
    """Clean all infrastructure datasets ready for analysis"""
    tfn_rail_links = get_rail_links(boundary, cfg.rail.tfn_rail_links)

    clean_roads(cfg, boundary)
    clean_rail(cfg,tfn_rail_links)
    clean_other(cfg, boundary, tfn_rail_links)

### ROAD

def clean_roads(cfg, boundary):
    """Clean all roads datasets ready for analysis"""
    clean_os_roads(cfg, boundary)
    clean_noham_roads(cfg, boundary)

def clean_os_roads(cfg, boundary):
    """Reads and cleans OS Open Roads dataset, then writes to file"""
    os_road = gpd.read_file(cfg.infrastructure.road.os_road)
    os_road.drop_duplicates(subset=['identifier', 'geometry'], inplace=True)
    os_road = os_road[['identifier', 'roadNumber', 'name1', 'function', 'geometry']]
    os_road.rename(columns={'name1': 'name', 'roadNumber': 'road_number'}, inplace=True)
    os_road[['road_number', 'name', 'function']] = (
        os_road[['road_number', 'name', 'function']].replace(0, 'N/A'))
    os_road = os_road[~os_road.geometry.is_empty]
    os_road = os_road[os_road.geometry.notnull()]
    tfn_os_road = clip_to_boundary(os_road, boundary)
    write_to_file(tfn_os_road, cfg.paths.model_input / "Infrastructure" / "Road" / "TfN OS Road" / "tfn_os_road.shp")

def clean_noham_roads(cfg, boundary):
    """Reads and cleans 2023 and 2048 NoHAM network datasets, then writes to file"""
    noham = {
        '2023': gpd.read_file(cfg.infrastructure.road.noham_2023),
        '2048': gpd.read_file(cfg.infrastructure.road.noham_2048),
    }

    for year, noham_network in noham.items():
        noham_network.drop_duplicates(subset=['link_id', 'geometry'], inplace=True)
        noham_network[['a', 'b']] = noham_network['link_id'].str.split('_', expand=True).astype(int)
        noham_network = noham_network[(noham_network['a'] >= 10000) & (noham_network['b'] >= 10000)]
        noham_network = noham_network[['link_id', 'geometry']]
        noham_network = noham_network[~noham_network.geometry.is_empty]
        noham_network = noham_network[noham_network.geometry.notnull()]
        tfn_noham_network = clip_to_boundary(noham_network, boundary)
        write_to_file(tfn_noham_network,
                      cfg.paths.model_input / "Infrastructure" / "Road" / f"TfN NoHAM {year}" / f"tfn_noham_{year}.shp")

### RAIL

def clean_rail(cfg, rail_links):
    """Clean all rail datasets ready for analysis"""
    clean_passenger_rail(cfg, rail_links)
    clean_freight_rail(cfg, rail_links)

def get_rail_links(boundary, os_rail_path):
    """Reads and cleans OS Rail Network data"""
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
    tfn_rail_links.rename(columns={'description': 'desc', 'physicallevel': 'phys_level', 'railwayuse': 'rail_use',
                                   'trackrepresentation': 'track_rep'}, inplace=True)
    tfn_rail_links = clip_to_boundary(tfn_rail_links, boundary)
    return tfn_rail_links

def clean_passenger_rail(cfg, tfn_rail_links):
    """Filters OS rail data to passenger rail network, then writes to file"""
    tfn_pass_rail = tfn_rail_links[tfn_rail_links['rail_use'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_pass_rail = tfn_pass_rail[tfn_pass_rail['desc'].isin(['Main Line', 'Main Line And Tram',
                                                                     'Main Line And Rapid Transport System'])]
    write_to_file(tfn_pass_rail,
                  cfg.paths.model_input / "Infrastructure" / "Rail" / "TfN OS Passenger Rail" / "tfn_pass_rail_links.shp")

def clean_freight_rail(cfg, tfn_rail_links):
    """Filters OS rail data to freight rail network, then writes to file"""
    tfn_freight_rail = tfn_rail_links[tfn_rail_links['rail_use'].isin(['Freight And Passenger', 'Freight'])]
    write_to_file(tfn_freight_rail,
                  cfg.paths.model_input / "Infrastructure" / "Rail" / "TfN OS Freight Rail" / "tfn_freight_rail_links.shp")

### OTHER

def clean_other(cfg, boundary, rail_links):
    """Cleans all other datasets ready for analysis"""
    clean_bus_stops(cfg, boundary)
    clean_petrol_stations(cfg, boundary)
    clean_charging_sites(cfg, boundary)
    clean_ncn(cfg, boundary)

    os_mm_net_node = read_os_mm_node_network(cfg.infrastructure.other.os_mmrn)
    clean_train_stations(cfg, os_mm_net_node, boundary)
    clean_tram_stations(cfg, os_mm_net_node, boundary)
    clean_rapid_transport_stations(cfg, os_mm_net_node, boundary)
    clean_ferry_terminals(cfg, os_mm_net_node, boundary)
    clean_bus_coach_stations(cfg, os_mm_net_node, boundary)
    clean_tram_network(cfg, rail_links)
    clean_rapid_transport_network(cfg, rail_links)

def clean_bus_stops(cfg, boundary):
    """Reads, combines and cleans regional bus stops datasets, then writes to file"""
    bus_stops_ne = pd.read_csv(cfg.infrastructure.other.bus_stops.ne)  # North East
    bus_stops_nw = pd.read_csv(cfg.infrastructure.other.bus_stops.nw)  # North West
    bus_stops_ys = pd.read_csv(cfg.infrastructure.other.bus_stops.ys)  # Yorkshire

    bus_stops = pd.concat([bus_stops_ne, bus_stops_nw, bus_stops_ys], ignore_index=True)  # Combine bus stops
    bus_stops_gdf = df_to_gdf(bus_stops, 'stop_lon', 'stop_lat', "EPSG:4326")
    bus_stops_gdf = bus_stops_gdf[['stop_id', 'stop_name', 'geometry']]  # Filter out columns
    bus_stops_gdf = bus_stops_gdf.drop_duplicates(subset=['stop_id', 'geometry'])  # Remove duplicate rows
    tfn_bus_stops = clip_to_boundary(bus_stops_gdf, boundary)  # Clip to TfN boundary
    write_to_file(tfn_bus_stops, cfg.paths.model_input / "Infrastructure" / "Other" / "TfN Bus Stops" /
                  "tfn_bus_stops.shp")

def clean_petrol_stations(cfg, boundary):
    """Reads and cleans POI data, filters for petrol stations, and writes to file"""
    poi_uk = gpd.read_file(
        f"zip://{cfg.infrastructure.other.poi_uk.zip_path}!{cfg.infrastructure.other.poi_uk.internal_path}")
    petrol_stations = poi_uk[poi_uk['main_category'] == 'gas_station']
    petrol_stations = petrol_stations[['id', 'geometry']]
    petrol_stations = petrol_stations.drop_duplicates(subset=['id', 'geometry'])
    tfn_petrol = clip_to_boundary(petrol_stations, boundary)
    write_to_file(tfn_petrol,
                  cfg.paths.model_input / "Infrastructure" / "Other" / "TfN Petrol Stations" / "tfn_petrol_stations.shp")

def read_os_mm_node_network(path):
    """Reads and cleans OS Multi-Modal Routing Network (OS MMRN) dataset to prepare for further filtering"""
    os_mm_net_node = gpd.read_file(path, layer="mrn_ntwk_transportnode")
    os_mm_net_node.drop(columns=['os_parentid', 'name'], inplace=True)
    os_mm_net_node.drop_duplicates(subset=['nodeid', 'geometry'], inplace=True)
    return os_mm_net_node

def clean_train_stations(cfg, os_mm_net_node, boundary):
    """Filters OS MMRN for train stations, then clips to boundary and writes to file"""
    train_stations = os_mm_net_node[
        (os_mm_net_node['os_nodetype'] == 'Railway Station;Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station;Railway Station (Underground System);Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station;Tram Station;Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station (Non Public Accessible);Modal Change') |
        (os_mm_net_node['os_nodetype'] == 'Railway Station (Principal);Tram Station;Modal Change')
        ]
    train_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_train_stations = clip_to_boundary(train_stations, boundary)
    write_to_file(tfn_train_stations,
                  cfg.paths.model_input / "Infrastructure" / "Other" / "TfN OS Train Stations" / "tfn_train_stations.shp")

def clean_tram_stations(cfg, os_mm_net_node, boundary):
    """Filters OS MMRN for tram stations, then clips to boundary and writes to file"""
    tram_stations = os_mm_net_node[os_mm_net_node['os_nodetype'].str.contains('Tram Station', case=False, na=False)]
    tram_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_tram_stations = clip_to_boundary(tram_stations, boundary)
    write_to_file(tfn_tram_stations,
                  cfg.paths.model_input / "Infrastructure" / "Other" / "TfN OS Tram Stations" / "tfn_tram_stations.shp")

def clean_rapid_transport_stations(cfg, os_mm_net_node, boundary):
    """Filters OS MMRN for rapid transport stations, then clips to boundary and writes to file"""
    rapid_transport_stations = os_mm_net_node[
        os_mm_net_node['os_nodetype'].str.contains('Underground System', case=False, na=False)]
    rapid_transport_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_rapid_transport_stations = clip_to_boundary(rapid_transport_stations, boundary)
    write_to_file(tfn_rapid_transport_stations,
                  cfg.paths.model_input / "Infrastructure" / "Other" /
                  "TfN OS Rapid Transport Stations" / "tfn_rapid_transport_stations.shp")

def clean_ferry_terminals(cfg, os_mm_net_node, boundary):
    """Filters OS MMRN for ferry terminals, then clips to boundary and writes to file"""
    ferry_terminals = os_mm_net_node[os_mm_net_node['os_nodetype'].str.contains('Ferry', case=False, na=False)]
    ferry_terminals.drop(columns=['os_nodetype'], inplace=True)
    tfn_ferry_terminals = clip_to_boundary(ferry_terminals, boundary)
    write_to_file(tfn_ferry_terminals,
                  cfg.paths.model_input / "Infrastructure" / "Other" / "TfN OS Ferry Terminals" / "tfn_ferry_terminals.shp")

def clean_bus_coach_stations(cfg, os_mm_net_node, boundary):
    """Filters OS MMRN for bus and coach stations, then clips to boundary and writes to file"""
    bus_coach_stations = os_mm_net_node[
        (os_mm_net_node['os_nodetype'].str.contains('Bus Station', case=False, na=False)) |
        (os_mm_net_node['os_nodetype'] == 'Coach Station;Modal Change')
        ]
    bus_coach_stations.drop(columns=['os_nodetype'], inplace=True)
    tfn_bus_coach_stations = clip_to_boundary(bus_coach_stations, boundary)
    write_to_file(tfn_bus_coach_stations,
                  cfg.paths.model_input / "Infrastructure" / "Other" /
                  "TfN OS Bus Coach Stations" / "tfn_bus_coach_stations.shp")

def clean_tram_network(cfg, tfn_rail_links):
    """Filters OS rail links for tram network, then writes to file"""
    tfn_tram_links = tfn_rail_links[tfn_rail_links['rail_use'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_tram_links = tfn_tram_links[tfn_tram_links['desc'].isin(['Tram', 'Main Line And Tram'])]
    write_to_file(tfn_tram_links,
                  cfg.paths.model_input / "Infrastructure" / "Other" / "TfN OS Tram Network" / "tfn_os_tram_links.shp")

def clean_rapid_transport_network(cfg, tfn_rail_links):
    """Filters OS rail links for rapid transport network, then writes to file"""
    tfn_rapid_transport = tfn_rail_links[tfn_rail_links['rail_use'].isin(['Freight And Passenger', 'Passenger'])]
    tfn_rapid_transport = tfn_rapid_transport[tfn_rapid_transport['desc'].isin(
        ['Rapid Transport System', 'Main Line And Rapid Transport System'])]
    write_to_file(tfn_rapid_transport,
                  cfg.paths.model_input / "Infrastructure" / "Other" /
                  "TfN Rapid Transport Network" / "tfn_rapid_transport_links.shp")

def clean_charging_sites(cfg, boundary):
    """Reads and cleans ZapMap charging sites data, then writes to file"""
    chg_sites = pd.read_csv(cfg.infrastructure.other.zapmap)
    chg_sites_gdf = gpd.GeoDataFrame(chg_sites, geometry=[Point(xy) for xy in zip(chg_sites['lon'], chg_sites['lat'])],
                                     crs="EPSG:4326")
    chg_sites_gdf = chg_sites_gdf[['identifier', 'name', 'speed', 'value', 'geometry']]
    chg_sites_gdf.rename(columns={'identifier': 'id', 'value': 'devices'}, inplace=True)
    chg_sites_gdf = chg_sites_gdf.drop_duplicates(subset=['geometry'])
    chg_sites_gdf = chg_sites_gdf[~chg_sites_gdf.geometry.is_empty]
    chg_sites_gdf = chg_sites_gdf[chg_sites_gdf.geometry.notnull()]
    tfn_chg_sites = clip_to_boundary(chg_sites_gdf, boundary)
    write_to_file(tfn_chg_sites,
                  cfg.paths.model_input / "Infrastructure" / "Other" / "TfN EV Charging Sites" / "tfn_chg_sites.shp")

def clean_ncn(cfg, boundary):
    """Reads and cleans National Cycle Network data, then writes to file"""
    ncn = gpd.read_file(cfg.infrastructure.other.ncn_sustrans)
    ncn.drop(columns=['RouteCat', 'OpenStatus', 'GlobalID'], inplace=True)
    ncn.drop_duplicates(subset=['SegmentID', 'geometry'], inplace=True)
    ncn_cols_replace = ['Desc_', 'Greenway', 'RouteType', 'RouteNo', 'LinkNo', 'Surface',
                        'Quality', 'Lighting', 'RoadClass']
    ncn[ncn_cols_replace] = ncn[ncn_cols_replace].replace(0, 'N/A')
    tfn_ncn = clip_to_boundary(ncn, boundary)
    write_to_file(tfn_ncn, cfg.paths.model_input / "Infrastructure" / "Other" / "TfN NCN" / "tfn_ncn.shp")

## HAZARDS

def clean_hazards(cfg, boundary):
    """Cleans hazard data ready for analysis"""
    clean_extreme_weather(cfg, boundary)
    clean_flooding(cfg, boundary)
    clean_ground_stability(cfg, boundary)
    clean_coastal_erosion(cfg, boundary)

### EXTREME WEATHER

def clean_extreme_weather(cfg, boundary):
    """Clean all extreme weather datasets ready for analysis"""
    tfn_common_grid = clean_common_grid(cfg, boundary)

    clean_temp_max(cfg, tfn_common_grid)
    clean_temp_min(cfg, tfn_common_grid)
    clean_summer_precip(cfg, tfn_common_grid)
    clean_winter_precip(cfg, tfn_common_grid)
    clean_rain_days(cfg, boundary)
    clean_drought_index(cfg, boundary)
    clean_hot_summer_days(cfg, tfn_common_grid)
    clean_extreme_summer_days(cfg, tfn_common_grid)
    clean_frost_days(cfg, tfn_common_grid)
    clean_icing_days(cfg, tfn_common_grid)
    clean_wind_speed(cfg, boundary)
    clean_wind_driven_rain(cfg, boundary)

def clean_common_grid(cfg, boundary):
    """Creates and prepares common grid DataFrame for variables on same 12km British National Grid"""
    temp_max = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.max_temp_summer.zip_path}!"
                            f"{cfg.hazards.extreme_weather.max_temp_summer.internal_path}")
    temp_max['grid_id'] = range(1, len(temp_max) + 1)
    common_grid = temp_max[['grid_id', 'geometry']]
    tfn_common_grid = clip_to_boundary(common_grid, boundary)
    tfn_common_grid = explode_to_polygons(tfn_common_grid)
    write_to_file(tfn_common_grid, cfg.paths.model_input / "Other" / "TfN Common Grid" / "tfn_common_grid.shp")
    return tfn_common_grid

def clean_temp_max(cfg, grid):
    """Reads and cleans max summer temperature change projections, then writes to file"""
    temp_max = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.max_temp_summer.zip_path}!"
                             f"{cfg.hazards.extreme_weather.max_temp_summer.internal_path}")
    temp_max['grid_id'] = range(1, len(temp_max) + 1)
    temp_max = temp_max[['grid_id', 'tasmax_s_4', 'tasmax__22']]
    temp_max.rename(columns={'tasmax_s_4': 'tasmax_s_c', 'tasmax__22': 'tasmax_s_f'}, inplace=True)
    temp_max['tasmax_s_f'] = temp_max['tasmax_s_c'] + temp_max['tasmax_s_f']
    tfn_temp_max = temp_max[temp_max['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_temp_max,
                  cfg.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Summer Max Temperature Change Projections" / "tfn_temp_max.csv",
                  csv=True)

def clean_temp_min(cfg, grid):
    """Reads and cleans min winter temperature change projections, then writes to file"""
    temp_min = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.min_temp_winter.zip_path}!"
                             f"{cfg.hazards.extreme_weather.min_temp_winter.internal_path}")
    temp_min['grid_id'] = range(1, len(temp_min) + 1)
    temp_min = temp_min[['grid_id', 'tasmin_w_4', 'tasmin__22']]
    temp_min.rename(columns={'tasmin_w_4': 'tasmin_w_c', 'tasmin__22': 'tasmin_w_f'}, inplace=True)
    temp_min['tasmin_w_f'] = temp_min['tasmin_w_c'] + temp_min['tasmin_w_f']
    tfn_temp_min = temp_min[temp_min['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_temp_min,
                  cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Winter Min Temperature Change Projections" / "tfn_temp_min.csv",
                  csv=True)

def clean_summer_precip(cfg, grid):
    """Reads and cleans summer precipitation change projections, then writes to file"""
    precip_sum = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.precip_summer.zip_path}!"
                               f"{cfg.hazards.extreme_weather.precip_summer.internal_path}")
    precip_sum['grid_id'] = range(1, len(precip_sum) + 1)
    precip_sum = precip_sum[['grid_id', 'pr_summe_3', 'pr_summ_21']]
    precip_sum.rename(columns={'pr_summe_3': 'pr_s_c','pr_summ_21': 'pr_s_pct_f'}, inplace=True)
    precip_sum['pr_s_f'] = precip_sum['pr_s_c'] * (1 + (precip_sum['pr_s_pct_f'] / 100))
    precip_sum.drop(columns=['pr_s_pct_f'], inplace=True)
    tfn_precip_sum = precip_sum[precip_sum['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_precip_sum,
                  cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Summer Precipitation Change Projections" / "tfn_precip_sum.csv",
                  csv=True)

def clean_winter_precip(cfg, grid):
    """Reads and cleans winter precipitation change projections, then writes to file"""
    precip_win = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.precip_winter.zip_path}!"
                               f"{cfg.hazards.extreme_weather.precip_winter.internal_path}")
    precip_win['grid_id'] = range(1, len(precip_win) + 1)
    precip_win = precip_win[['grid_id', 'pr_winte_3', 'pr_wint_21']]
    precip_win.rename(columns={'pr_winte_3': 'pr_w_c','pr_wint_21': 'pr_w_pct_f',}, inplace=True)
    precip_win['pr_w_f'] = precip_win['pr_w_c'] * (1 + (precip_win['pr_w_pct_f'] / 100))
    precip_win.drop(columns=['pr_w_pct_f'], inplace=True)
    tfn_precip_win = precip_win[precip_win['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_precip_win,
                  cfg.paths.model_input / "Hazards" / "Extreme Weather"  /
                  "TfN Winter Precipitation Change Projections" / "tfn_precip_win.csv",
                  csv=True)

def clean_rain_days(cfg, boundary):
    """Reads and cleans 10mm rain days observations, then writes to file"""
    rain_days = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.rain_days.zip_path}!"
                              f"{cfg.hazards.extreme_weather.rain_days.internal_path}")
    tfn_rain_days = clip_to_boundary(rain_days, boundary)
    tfn_rain_days = explode_to_polygons(tfn_rain_days)
    tfn_rain_days.rename(columns={'Rain10mmDa': 'rain_d_c'}, inplace=True)
    tfn_rain_days['rain_d_f'] = tfn_rain_days['rain_d_c'] # Duplicate rain days column
    tfn_rain_days.drop(columns=['part'], inplace=True)
    write_to_file(tfn_rain_days, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN 10mm Rain Days 1991-2020" / "tfn_rain_days.shp")

def clean_drought_index(cfg, boundary):
    """Reads and cleans drought severity index data, then writes to file"""
    drought_index = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.drought_index.zip_path}!"
                                  f"{cfg.hazards.extreme_weather.drought_index.internal_path}")
    drought_index = drought_index[['DSI12_ba_4', 'DSI12_40_m', 'geometry']]
    drought_index.rename(columns={'DSI12_ba_4': 'dsi_c', 'DSI12_40_m': 'dsi_f'}, inplace=True)
    tfn_drought = clip_to_boundary(drought_index, boundary)
    tfn_drought = explode_to_polygons(tfn_drought)
    tfn_drought.drop(columns=['part'], inplace=True)
    write_to_file(tfn_drought, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Drought Severity Index" / "tfn_drought_index.shp")

def clean_hot_summer_days(cfg, grid):
    """Reads and cleans hot summer days projections, then writes to file"""
    hot_days = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.hot_days.zip_path}!"
                             f"{cfg.hazards.extreme_weather.hot_days.internal_path}")
    hot_days['grid_id'] = range(1, len(hot_days) + 1)
    hot_days = hot_days[['grid_id', 'HSD_base_4', 'HSD_40_med']]
    hot_days.rename(columns={'HSD_base_4': 'hsd_c', 'HSD_40_med': 'hsd_f'}, inplace=True)
    tfn_hot_days = hot_days[hot_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_hot_days, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Hot Summer Days Projections" / "tfn_hot_days.csv",
                  csv=True)

def clean_extreme_summer_days(cfg, grid):
    """Reads and cleans extreme summer days projections, then writes to file"""
    extr_days = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.extreme_summer_days.zip_path}!"
                              f"{cfg.hazards.extreme_weather.extreme_summer_days.internal_path}")
    extr_days['grid_id'] = range(1, len(extr_days) + 1)
    extr_days = extr_days[['grid_id', 'ESD_base_4', 'ESD_40_med']]
    extr_days.rename(columns={'ESD_base_4': 'esd_c','ESD_40_med': 'esd_f'}, inplace=True)
    tfn_extr_days = extr_days[extr_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_extr_days,
                  cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Extreme Summer Days Projections" / "tfn_extr_days.csv", csv=True)

def clean_frost_days(cfg, grid):
    """Reads and cleans frost days projections, then writes to file"""
    frost_days = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.frost_days.zip_path}!"
                               f"{cfg.hazards.extreme_weather.frost_days.internal_path}")
    frost_days['grid_id'] = range(1, len(frost_days) + 1)
    frost_days = frost_days[['grid_id', 'FrostDay_3', 'FrostDa_18']]
    frost_days.rename(columns={'FrostDay_3': 'frost_d_c', 'FrostDa_18': 'frost_d_f'}, inplace=True)
    tfn_frost_days = frost_days[frost_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_frost_days, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Frost Days Projections" / "tfn_frost_days.csv",
                  csv=True)

def clean_icing_days(cfg, grid):
    """Reads and cleans icing days projections, then writes to file"""
    ice_days = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.icing_days.zip_path}!"
                             f"{cfg.hazards.extreme_weather.icing_days.internal_path}")
    ice_days['grid_id'] = range(1, len(ice_days) + 1)
    ice_days = ice_days[['grid_id', 'IcingDay_3', 'IcingDa_18']]
    ice_days.rename(columns={'IcingDay_3': 'ice_d_c','IcingDa_18': 'ice_d_f'}, inplace=True)
    tfn_ice_days = ice_days[ice_days['grid_id'].isin(grid['grid_id'])]
    write_to_file(tfn_ice_days, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Icing Days Projections" / "tfn_ice_days.csv", csv=True)

def clean_wind_speed(cfg, boundary):
    """Reads wind speed projections, calculates exceedance and percentiles, cleans, then writes to file"""
    windspd_c_combined = read_wind_speed_reduce(cfg.hazards.extreme_weather.wind_spd_current, 'c')
    windspd_f_combined = read_wind_speed_reduce(cfg.hazards.extreme_weather.wind_spd_forecast, 'f')

    windspd_combined = windspd_merge_and_fill(windspd_c_combined, windspd_f_combined, 'avg_excd_f')

    windspd_combined['geometry'] = [convert_point_to_grid(x, y, 2500)
        for x, y in zip(windspd_combined['projection_x_coordinate'], windspd_combined['projection_y_coordinate'])]
    windspd_combined = gpd.GeoDataFrame(windspd_combined, geometry='geometry', crs="EPSG:27700")
    windspd_combined = windspd_combined[['p99_c', 'avg_excd_c', 'p99_f', 'avg_excd_f','geometry']]
    tfn_windspd = clip_to_boundary(windspd_combined, boundary)
    tfn_windspd = explode_to_polygons(tfn_windspd)
    tfn_windspd.drop(columns=['part'], inplace=True)
    write_to_file(tfn_windspd, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Wind Speed Projections" / "tfn_windspd.shp")

def read_wind_speed_reduce(xr_path, tp):
    """Reads wind speed projections, calculates exceedance and percentile measures and returns a reduced dataframe"""
    windspd = xr.open_dataset(xr_path).to_dataframe()
    exc = calculate_exceedance(20, windspd, 'wsgmax10m', tp)
    pct = calculate_percentile(windspd, [0.99], 'wsgmax10m')
    pct.columns = ['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude', f'p99_{tp}']
    windspd_combined = windspd_merge_and_fill(pct, exc, f'avg_excd_{tp}')
    return windspd_combined

def windspd_merge_and_fill(df1, df2, fill_col):
    """Merges two dataframes on common coordinates, filling a given column with 0 values"""
    return pd.merge(df1, df2, on=['projection_y_coordinate', 'projection_x_coordinate', 'latitude', 'longitude'],
                    how='outer').fillna({fill_col: 0})

def clean_wind_driven_rain(cfg, boundary):
    """Reads and cleans wind driven rain index data, then writes to file"""
    wdr = gpd.read_file(f"zip://{cfg.hazards.extreme_weather.wdr_index.zip_path}!"
                        f"{cfg.hazards.extreme_weather.wdr_index.internal_path}")

    # Aggregate by wind direction to calculate mean wind speed
    wdr_agg = (
        wdr.groupby(['x_coord', 'y_coord'])
        .agg({
            'WDR_base_1': 'mean',
            'WDR_40_Med': 'mean',
            'geometry': 'first'
        })
        .reset_index()
    )

    wdr_agg = wdr_agg[['WDR_base_1', 'WDR_40_Med', 'geometry']]
    wdr_agg.rename(columns={'WDR_base_1': 'wdr_c', 'WDR_40_Med': 'wdr_f'}, inplace=True)
    wdr_agg = gpd.GeoDataFrame(wdr_agg, geometry='geometry', crs='EPSG:3857')
    tfn_wdr = clip_to_boundary(wdr_agg, boundary)
    tfn_wdr = explode_to_polygons(tfn_wdr)
    tfn_wdr.drop(columns=['part'], inplace=True)
    write_to_file(tfn_wdr, cfg.paths.model_input / "Hazards" / "Extreme Weather" /
                  "TfN Wind Driven Rain Index" / "tfn_wdr.shp")

### FLOODING

def clean_flooding(cfg, boundary):
    """Clean flooding data ready for analysis"""
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

    if cfg.switches.flood_zip_extract:
        extract_flood_data(cfg, code_number_map)

    clean_flood(cfg, "RoFRS", "RoFRS", "v202501", boundary,
                 "TfN RoFRS CC/tfn_rofrs_cc.gpkg", True, code_number_map)
    clean_flood(cfg, "RoFRS", "RoFRS", "v202501", boundary,
                 "TfN RoFRS/tfn_rofrs.gpkg", False, code_number_map)
    clean_flood(cfg, "RoFSW CC", "RoFSW", "v202509", boundary,
                "TfN RoFSW CC/tfn_rofsw_cc.gpkg", True, code_number_map)
    clean_flood(cfg, "RoFSW", "RoFSW", "v202509", boundary,
                "TfN RoFSW/tfn_rofsw.gpkg", False, code_number_map)

def extract_gdb_file(cfg, code, number, flood_data, version, cc):
    """Extracts a flood gdb file from a zip file given its BNG code and number, and version"""
    try:
        base_path = cfg.paths.raw_input / "Hazards" / "Flooding" / flood_data
        if cc:
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
        with ZipFile(zip_path, 'r') as zip_ref:
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

def read_gdb(cfg, code, number, file_name, flood_data, version, cc):
    """Reads first layer of flood gdb file"""
    base_path = cfg.paths.model_input / "Hazards" / "Flooding" / file_name / code

    if cc:
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

def extract_flood_data(cfg, code_number_map):
    """Extract geodatabase files from raw RoFRS and RoFSW flood data"""
    for code in code_number_map.keys():
        for number in code_number_map[code]:
            # Forecast (Climate Change) data
            extract_gdb_file(cfg, code, number, "RoFRS", "v202501", True)
            extract_gdb_file(cfg, code, number, "RoFSW", "v202509", True)

            # Current data
            extract_gdb_file(cfg, code, number, "RoFRS", "v202501", False)
            extract_gdb_file(cfg, code, number, "RoFSW", "v202509", False)

def clean_flood(cfg, file_name, flood_type, version, boundary, out_path, cc, code_number_map):
    """Reads and cleans flood data, then writes to file"""
    gdfs = []
    for code in code_number_map.keys():
        for number in code_number_map[code]:
            gdf = read_gdb(cfg, code, number, file_name, flood_type, version, cc)  # Read file
            tfn_gdf = clip_to_boundary(gdf, boundary)
            gdfs.append(tfn_gdf)  # Add to list
            print(code, number)
            del gdf, tfn_gdf

    flood_data = pd.concat(gdfs, ignore_index=True)
    flood_data = extract_poly_from_geomcollection(flood_data)
    flood_data = flood_data[['Risk_band', 'geometry']]
    write_to_file(flood_data, cfg.paths.model_input / "Hazards" / "Flooding" / out_path, driver="GPKG")

### GROUND STABILITY

def clean_ground_stability(cfg, boundary):
    """Cleans ground stability data ready for analysis"""
    clean_geosure(cfg, boundary)
    clean_geoclimate(cfg, boundary)

def clean_geosure(cfg, boundary):
    """Cleans GeoSureHexGrids data, merges by nearest centroids, then writes to file"""
    geosure_layers = {
        'cd': gpd.read_file(f"zip://{cfg.hazards.ground_stability.geosure.zip_path}!"
                            f"/" + cfg.hazards.ground_stability.geosure.collapsible_deposits),
        'cg': gpd.read_file(f"zip://{cfg.hazards.ground_stability.geosure.zip_path}!"
                            f"/" + cfg.hazards.ground_stability.geosure.compressible_ground),
        'ls': gpd.read_file(f"zip://{cfg.hazards.ground_stability.geosure.zip_path}!"
                            f"/" + cfg.hazards.ground_stability.geosure.landslides),
        'rs': gpd.read_file(f"zip://{cfg.hazards.ground_stability.geosure.zip_path}!"
                            f"/" + cfg.hazards.ground_stability.geosure.running_sand),
        'ss': gpd.read_file(f"zip://{cfg.hazards.ground_stability.geosure.zip_path}!"
                            f"/" + cfg.hazards.ground_stability.geosure.shrink_swell),
        'sr': gpd.read_file(f"zip://{cfg.hazards.ground_stability.geosure.zip_path}!"
                            f"/" + cfg.hazards.ground_stability.geosure.soluble_rocks)
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
    write_to_file(tfn_geosure, cfg.paths.model_input / "Hazards" / "Ground Stability" / "TfN GeoSure" / "tfn_geosure.shp")

def clean_geoclimate(cfg, boundary):
    """Reads and cleans GeoClimate Shrink-Swell data, then writes to file"""
    for year in ['2030', '2070']:
        gdf = gpd.read_file(cfg.hazards.ground_stability.geo_shrink_swell[year])
        gdf.rename(columns={'CLASS': 'ss_geo_risk'}, inplace=True)
        gdf = gdf[['ss_geo_risk', 'geometry']]
        tfn_gdf = clip_to_boundary(gdf, boundary)
        tfn_gdf = explode_to_polygons(tfn_gdf)
        write_to_file(tfn_gdf, cfg.paths.model_input / "Hazards" / "Ground Stability" /
                      "BGS Shrink Swell" / year / f"tfn_bgs_ss_{year}.shp")

### COASTAL EROSION

def clean_coastal_erosion(cfg, boundary):
    """Cleans coastal erosion data ready for analysis"""
    clean_giz(cfg, boundary)
    clean_ncerm(cfg, boundary)

def clean_giz(cfg, boundary):
    """Cleans Ground Instability Zones data from NCERM, then writes to file"""
    ncerm_giz = gpd.read_file(f"zip://{cfg.hazards.coastal_erosion.zip_path}!{cfg.hazards.coastal_erosion.giz}")
    ncerm_giz = ncerm_giz[['smp_no', 'geometry']]
    tfn_ncerm_giz = clip_to_boundary(ncerm_giz, boundary)
    tfn_ncerm_giz = explode_to_polygons(tfn_ncerm_giz)
    write_to_file(tfn_ncerm_giz, cfg.paths.model_input / "Hazards" / "Coastal Erosion" /
                  "NCERM" / "Ground Instability Zones" / "tfn_ncerm_giz.shp")

def clean_ncerm(cfg, boundary):
    """Cleans erosion data from NCERM for 2055, and 2105, then writes to file"""
    for year in ['2055', '2105']:
        gdf = gpd.read_file(f"zip://{cfg.hazards.coastal_erosion.zip_path}!/" + cfg.hazards.coastal_erosion.smp[year])
        gdf = gdf[['smp_name', 'geometry']]
        tfn_gdf = clip_to_boundary(gdf, boundary)
        tfn_gdf = explode_to_polygons(tfn_gdf)
        write_to_file(tfn_gdf,
                      cfg.paths.model_input / "Hazards" / "Coastal Erosion" /
                      "NCERM" / f"SMP_{year}_70CC" / f"tfn_ncerm_smp_{year}_70CC.shp")

## IMPACT

def clean_impact(cfg, boundary):
    """Cleans impact datasets ready for analysis"""
    clean_freight_demand(cfg, boundary)
    clean_noham_flows(cfg)

### FREIGHT

def clean_freight_demand(cfg, boundary):
    """Cleans freight demand data ready for analysis"""
    tfn_freight_network_demand = read_freight_demand(cfg.impact.freight_demand, boundary)

    tfn_os_freight_network_demand = map_freight_networks(
        tfn_freight_network_demand,
        cfg.model_input / "Infrastructure" / "Rail" / "TfN OS Freight Rail" / "tfn_freight_rail_links.shp")

    write_to_file(tfn_os_freight_network_demand,
                  cfg.paths.model_input / "Impact" / "TfN Freight Flows" / "tfn_freight_network_demand.gpkg")

def read_freight_demand(path, boundary):
    """Reads and cleans freight demand data, and returns as GeoDataFrame"""
    freight_network_demand = gpd.read_file(path)
    freight_network_demand = freight_network_demand[['dij_id', '2022_23_total', '2050_51 sc2_total', 'geometry']]
    tfn_freight_network_demand = clip_to_boundary(freight_network_demand, boundary)
    return tfn_freight_network_demand

def map_freight_networks(tfn_freight_network_demand, os_path):
    """Maps freight demand data onto the OS freight network using nearest spatial join, then cleans and returns"""
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

def clean_noham_flows(cfg):
    """Cleans NoHAM flows data, aggregates link flows by year, merges with the network, then write to file"""
    link_flows = aggregate_link_flows_year(cfg)

    for year, tp in {'2023': 'c', '2048': 'f'}.items():
        tfn_noham_flows = link_flows[year]
        tfn_noham_net_flows = merge_noham_flow_network(
            tfn_noham_flows,
            cfg.paths.model_input / "Infrastructure" / "Road" / f"TfN NoHAM {year}" / f"tfn_noham_{year}.shp"
        )
        write_to_file(tfn_noham_net_flows,
                      cfg.paths.model_input / "Impact" / "TfN NoHAM Flows" / year / f"tfn_noham_net_flows_{tp}.gpkg",
                      driver="GPKG")

def read_noham_h5(year, time_period, user_class, noham_path, output_path, extract):
    """Reads NoHAM h5 files and extracts the link, routes, and od's DataFrames"""
    if extract:
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
    """Takes NoHAM od's, routes, and links to create aggregated link flows DataFrame"""
    # Flatten OD and Route data
    od_flat = ods.reset_index()[['route', 'abs_demand']]
    route_flat = routes.reset_index()[['route', 'link_id']]

    # Merge OD demand with route links
    od_links = od_flat.merge(route_flat, on='route')

    # Aggregate demand per link_id
    link_demand = od_links.groupby('link_id')['abs_demand'].sum().reset_index()

    link_flows = pd.merge(link_demand, links, left_on='link_id', right_index=True)

    return link_flows

def aggregate_link_flows_year(cfg):
    """Aggregates link flows for each year, time period, and user class"""
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
                od_df, route_df, link_df = read_noham_h5(
                    year, time_period, user_class, cfg.impact.noham_demand.zip_path, cfg.impact.noham_demand.output_path,
                    cfg.switches.noham_zip_extract)
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
    """Merges NoHAM flows onto road network, then returns as GeoDataFrame"""
    tfn_noham_link = gpd.read_file(noham_path)
    tfn_noham_net_flows = pd.merge(
        tfn_noham_link,
        tfn_noham_flows,
        on='link_id',
        how='left'  # Keep all network, adding flows where available
    )
    return tfn_noham_net_flows

















