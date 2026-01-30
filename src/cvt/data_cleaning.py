"""Cleans raw input data to prepare it for input into the model."""

### LOAD LIBRARIES
import logging
import os
import pathlib
from zipfile import ZipFile

import fiona
import geopandas as gpd
import h5py
import pandas as pd
import py7zr
import xarray as xr
from shapely import geometry

from cvt import model_config

LOG = logging.getLogger(__name__)

### ENVIRONMENT VARIABLES ###

 # Minimum a and b values for NoHAM road links to keep
_NOHAM_ROAD_THRESHOLD = int(os.getenv("_NOHAM_ROAD_THRESHOLD", "10000"))

###  MODULE CONSTANTS ###
MMRN_NODE_TYPES = {
    'Train Stations': [
        "Railway Station;Modal Change",
        "Railway Station;Railway Station (Underground System);Modal Change",
        "Railway Station;Tram Station;Modal Change",
        "Railway Station (Non Public Accessible);Modal Change",
        "Railway Station (Principal);Tram Station;Modal Change"
    ]
}

### GENERAL FUNCTIONS


def clip_to_boundary(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to a specified spatial boundary.

    This function reprojects the boundary to match the CRS of the input GeoDataFrame and then
    clips the geometries in the GeoDataFrame so that only features within the boundary are
    retained.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        The input GeoDataFrame containing geometries to be clipped.
    boundary: geopandas.GeoDataFrame
        The GeoDataFrame representing the clipping boundary.

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame containing only the geometries from 'gdf' that fall within the
        specified boundary.
    """
    boundary = boundary.to_crs(gdf.crs)  # Match CRS
    return gpd.clip(gdf, boundary)  # Clip GDF to boundary


def write_to_file(
    data: pd.DataFrame | gpd.GeoDataFrame,
    output_path: pathlib.Path,
) -> None:
    """
    Write a DataFrame or GeoDataFrame to a file.

    This function ensures the output directory exists, checks that the input DataFrame is not
    empty, and writes the data to the given path. Supports formats CSV, GPKG, and SHP.

    Parameters
    ----------
    data: pandas.DataFrame or geopandas.GeoDataFrame
        The input data to write. Must be a GeoDataFrame for GIS formats.
    output_path: pathlib.Path
        Full path including filename and extension where the file will be saved.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the directory exists, make one if not

    if data.empty:
        raise ValueError(f"GeoDataFrame is empty. Nothing written to {output_path}")

    ext = pathlib.Path(output_path).suffix.lower()

    driver_map = {".gpkg": "GPKG", ".shp": "ESRI Shapefile"}

    if ext == ".csv":
        data.to_csv(output_path, index=False)
    elif ext in driver_map:
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError(f"{ext} requires a GeoDataFrame, got {type(data)}")
        data.to_file(output_path, driver=driver_map[ext])
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def explode_to_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Explodes the MultiPolygons and GeomCollections in a GeoDataFrame into Polygons.

    Converts rows in a GeoDataFrame with a MultiPolygon or GeometryCollection geometry into
    multiple rows with Polygon geometries, adding a 'part' column to keep track of how many new
    rows have been created for each. Leaves other geometries as they are, and preserves the
    original CRS.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        The input GeoDataFrame that needs to be cleaned

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing only Polygon geometries, with a new column 'part' to keep
        track.
    """
    rows = []
    for _idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Polygon":
            new_row = row.copy()
            new_row["part"] = 0
            rows.append(new_row)
        elif geom.geom_type == "MultiPolygon":
            for i, poly in enumerate(geom.geoms):
                new_row = row.copy()
                new_row.geometry = poly
                new_row["part"] = i
                rows.append(new_row)
        elif geom.geom_type == "GeometryCollection":
            poly_count = 0
            for part in geom.geoms:
                if part.geom_type == "Polygon":
                    new_row = row.copy()
                    new_row.geometry = part
                    new_row["part"] = poly_count
                    rows.append(new_row)
                    poly_count += 1

    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)


def _df_to_gdf(df: pd.DataFrame, x_col: str, y_col: str, crs: str) -> gpd.GeoDataFrame:
    """Take a DataFrame and convert it to a GeoDataFrame using spatial columns."""
    geometry = [geometry.Point(xy) for xy in zip(df[x_col], df[y_col], strict=False)]  # Create geometry
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)  # Convert to GeoDataFrame


def _convert_point_to_grid(x: int, y: int, size: int) -> geometry.Polygon:
    """Take a point and convert it to a grid of the given size."""
    return geometry.Polygon(
        [
            (x - size, y - size),
            (x + size, y - size),
            (x + size, y + size),
            (x - size, y + size),
        ]
    )


def _extract_poly_from_geomcollection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract polygons from GeomCollection objects in and turn into new rows."""
    rows = []

    for _idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # If it's already Polygon or MultiPolygon, keep as is
        if isinstance(geom, (geometry.Polygon, geometry.MultiPolygon)):
            rows.append(row)

        # If it's a GeometryCollection, extract polygons
        elif isinstance(geom, geometry.GeometryCollection):
            for sub_geom in geom.geoms:
                if isinstance(sub_geom, (geometry.Polygon, geometry.MultiPolygon)):
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    rows.append(new_row)

    # Create new GeoDataFrame from expanded rows
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def _nearest_centroids(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Take two GeoDataFrames and merge them on their nearest centroids."""
    bng_crs = "EPSG:27700"

    # Ensure both GeoDataFrames are in the same projected CRS
    gdf1 = gdf1.to_crs(bng_crs)
    gdf2 = gdf2.to_crs(bng_crs)

    # Convert both to centroids
    gdf1_centroids = gdf1.copy()
    gdf1_centroids["geometry"] = gdf1_centroids.geometry.centroid
    gdf2_centroids = gdf2.copy()
    gdf2_centroids["geometry"] = gdf2_centroids.geometry.centroid

    nearest = gpd.sjoin_nearest(gdf1_centroids, gdf2_centroids, how="left")

    # Merge back with original gdf1 to restore original geometry
    return gdf1.merge(nearest.drop(columns="geometry"), left_index=True, right_index=True)


# DATA CLEANING


def data_cleaning(config: model_config.Config) -> None:
    """
    Clean all input data for CVT.

    Clean infrastructure, hazard, and impact data according to the configurations provided.

    Parameters
    ----------
    config : Config
        Main config for the model, containing paths and settings.
    """
    boundary = gpd.read_file(config.other_input.boundary_path)

    _clean_infrastructure(config, boundary)
    _clean_hazards(config, boundary)
    _clean_impact(config, boundary)


## INFRASTRUCTURE


def _clean_infrastructure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all infrastructure datasets ready for analysis."""
    tfn_rail_links = _get_rail_links(boundary, config.infrastructure.rail.tfn_rail_links)

    _clean_roads(config, boundary)
    _clean_rail(config, tfn_rail_links)
    _clean_other(config, boundary, tfn_rail_links)


### ROAD


def _clean_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all roads datasets ready for analysis."""
    _clean_os_roads(config, boundary)
    _clean_noham_roads(config, boundary)


def _clean_os_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean OS Open Roads dataset, then write to file."""
    os_road = gpd.read_file(config.infrastructure.road.os_road)
    os_road = os_road.drop_duplicates(subset=["identifier", "geometry"])
    os_road = os_road[["identifier", "roadNumber", "name1", "function", "geometry"]]
    os_road = os_road.rename(columns={"name1": "name", "roadNumber": "road_number"})
    os_road[["road_number", "name", "function"]] = os_road[
        ["road_number", "name", "function"]
    ].replace(0, "N/A")
    len_before_filter = len(os_road)
    os_road = os_road[~os_road.geometry.is_empty]
    os_road = os_road[os_road.geometry.notna()]
    tfn_os_road = clip_to_boundary(os_road, boundary)
    filter_removed = len_before_filter - len(tfn_os_road)
    LOG.info("OS roads filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_os_road,
        config.paths.model_input / "Infrastructure" / "Road" / "TfN OS Road" / "tfn_os_road.gpkg",
    )


def _clean_noham_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean 2023 and 2048 NoHAM network datasets, then write to file."""
    for scenario, noham_entry in config.infrastructure.road.noham.items():
        year = noham_entry.year
        file_path = noham_entry.file_path
        noham_network = gpd.read_file(file_path)
        len_before_filter = len(noham_network)
        noham_network_clean = noham_network.drop_duplicates(subset=["link_id", "geometry"])
        noham_network_clean[["a", "b"]] = (
            noham_network_clean["link_id"].str.split("_", expand=True).astype(int)
        )
        # Filter out links with a or b less than 10,000 (zone connectors)
        noham_network_clean = noham_network_clean[
            (noham_network_clean["a"] >= _NOHAM_ROAD_THRESHOLD)
            & (noham_network_clean["b"] >= _NOHAM_ROAD_THRESHOLD)
        ]
        noham_network_clean = noham_network_clean[["link_id", "geometry"]]
        noham_network_clean = noham_network_clean[~noham_network_clean.geometry.is_empty]
        noham_network_clean = noham_network_clean[noham_network_clean.geometry.notna()]
        tfn_noham_network = clip_to_boundary(noham_network_clean, boundary)
        filter_removed = len_before_filter - len(tfn_noham_network)
        LOG.info("NoHAM network %s filtered - %s of %s (%s percent) rows removed", year, filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
        write_to_file(
            tfn_noham_network,
            config.paths.model_input
            / "Infrastructure"
            / "Road"
            / f"TfN NoHAM {year}"
            / f"tfn_noham_{year}.gpkg",
        )


### RAIL


def _clean_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Clean all rail datasets ready for analysis."""
    _clean_passenger_rail(config, rail_links)
    _clean_freight_rail(config, rail_links)


def _get_rail_links(boundary: gpd.GeoDataFrame, os_rail_path: pathlib.Path) -> gpd.GeoDataFrame:
    """Read and clean OS Rail Network data."""
    tfn_rail_links = gpd.read_file(os_rail_path)
    len_before_filter = len(tfn_rail_links)
    tfn_rail_links = tfn_rail_links[
        tfn_rail_links["operationalstatus"] == "Active"
    ]  # Exclude inactive links
    tfn_rail_links = tfn_rail_links[
        [
            "osid",
            "description",
            "structure",
            "physicallevel",
            "railwayuse",
            "trackrepresentation",
            "geometry",
        ]
    ]
    tfn_rail_links = tfn_rail_links[
        ~tfn_rail_links["description"].isin(
            ["Preserved", "Funicular", "Mineral", "Static Museum"]
        )
    ]
    tfn_rail_links = tfn_rail_links.drop_duplicates(subset=["osid", "geometry"])
    tfn_rail_links[
        ["description", "structure", "physicallevel", "railwayuse", "trackrepresentation"]
    ] = tfn_rail_links[
        ["description", "structure", "physicallevel", "railwayuse", "trackrepresentation"]
    ].replace(0, "N/A")
    tfn_rail_links = tfn_rail_links.rename(
        columns={
            "description": "desc",
            "physicallevel": "phys_level",
            "railwayuse": "rail_use",
            "trackrepresentation": "track_rep",
        },
    )

    tfn_rail_links = clip_to_boundary(tfn_rail_links, boundary)
    filter_removed = len_before_filter - len(tfn_rail_links)
    LOG.info("OS MMRN Rail links filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)

    return tfn_rail_links


def _clean_passenger_rail(config: model_config.Config, tfn_rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to passenger rail network, then write to file."""
    len_before_filter = len(tfn_rail_links)
    tfn_pass_rail = tfn_rail_links[
        tfn_rail_links["rail_use"].isin(["Freight And Passenger", "Passenger"])
    ]
    tfn_pass_rail = tfn_pass_rail[
        tfn_pass_rail["desc"].isin(
            ["Main Line", "Main Line And Tram", "Main Line And Rapid Transport System"]
        )
    ]
    filter_removed = len_before_filter - len(tfn_pass_rail)
    LOG.info("Passenger rail links filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_pass_rail,
        config.paths.model_input
        / "Infrastructure"
        / "Rail"
        / "TfN OS Passenger Rail"
        / "tfn_pass_rail_links.gpkg",
    )


def _clean_freight_rail(config: model_config.Config, tfn_rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to freight rail network, then write to file."""
    len_before_filter = len(tfn_rail_links)
    tfn_freight_rail = tfn_rail_links[
        tfn_rail_links["rail_use"].isin(["Freight And Passenger", "Freight"])
    ]
    filter_removed = len_before_filter - len(tfn_freight_rail)
    LOG.info("Freight rail links filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_freight_rail,
        config.paths.model_input
        / "Infrastructure"
        / "Rail"
        / "TfN OS Freight Rail"
        / "tfn_freight_rail_links.gpkg",
    )


### OTHER


def _clean_other(
    config: model_config.Config, boundary: gpd.GeoDataFrame, rail_links: gpd.GeoDataFrame
) -> None:
    """Clean all other datasets ready for analysis."""
    _clean_airports(config)
    _clean_bus_stops(config, boundary)
    _clean_petrol_stations(config, boundary)
    _clean_charging_sites(config, boundary)
    _clean_ncn(config, boundary)

    os_mm_net_node = _read_os_mm_node_network(config.infrastructure.other.os_mmrn)
    _clean_train_stations(config, os_mm_net_node, boundary)
    _clean_tram_stations(config, os_mm_net_node, boundary)
    _clean_rapid_transport_stations(config, os_mm_net_node, boundary)
    _clean_ferry_terminals(config, os_mm_net_node, boundary)
    _clean_bus_coach_stations(config, os_mm_net_node, boundary)
    _clean_tram_network(config, rail_links)
    _clean_rapid_transport_network(config, rail_links)


def _clean_airports(config: model_config.Config) -> None:
    """Read TfN airports dataset, then write to file in new directory."""
    airports = gpd.read_file(config.infrastructure.other.uk_airports)
    write_to_file(
        airports,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Airports"
        / "tfn_airports.gpkg",
    )


def _clean_bus_stops(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read, combine and clean regional bus stops datasets, then write to file."""
    bus_stops_ne = pd.read_csv(config.infrastructure.other.bus_stops['north_east'])  # North East
    bus_stops_nw = pd.read_csv(config.infrastructure.other.bus_stops['north_west'])  # North West
    bus_stops_ys = pd.read_csv(config.infrastructure.other.bus_stops['yorkshire'])  # Yorkshire

    bus_stops = pd.concat(
        [bus_stops_ne, bus_stops_nw, bus_stops_ys], ignore_index=True
    )  # Combine bus stops
    bus_stops_gdf = _df_to_gdf(bus_stops, "stop_lon", "stop_lat", "EPSG:4326")
    bus_stops_gdf = bus_stops_gdf[["stop_id", "stop_name", "geometry"]]  # Filter out columns
    len_before_filter = len(bus_stops_gdf)
    bus_stops_gdf = bus_stops_gdf.drop_duplicates(
        subset=["stop_id", "geometry"]
    )  # Remove duplicate rows
    tfn_bus_stops = clip_to_boundary(bus_stops_gdf, boundary)  # Clip to TfN boundary
    filter_removed = len_before_filter - len(tfn_bus_stops)
    LOG.info("Bus stops filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_bus_stops,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Bus Stops"
        / "tfn_bus_stops.gpkg",
    )


def _clean_petrol_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean POI data, filter for petrol stations, and write to file."""
    poi_uk = gpd.read_file(
        f"zip://{config.infrastructure.other.poi_uk.zip_path}!{config.infrastructure.other.poi_uk.file_path}"
    )
    len_before_filter = len(poi_uk)
    petrol_stations = poi_uk[poi_uk["main_category"] == "gas_station"]
    petrol_stations = petrol_stations[["id", "geometry"]]
    petrol_stations = petrol_stations.drop_duplicates(subset=["id", "geometry"])
    tfn_petrol = clip_to_boundary(petrol_stations, boundary)
    filter_removed = len_before_filter - len(tfn_petrol)
    LOG.info("Petrol stations filtered from POIs - %s of %s (% percent)", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_petrol,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Petrol Stations"
        / "tfn_petrol_stations.gpkg",
    )


def _read_os_mm_node_network(path: pathlib.Path) -> gpd.GeoDataFrame:
    """Read and clean OS MMRN dataset to prepare for further filtering."""
    os_mm_net_node = gpd.read_file(path, layer="mrn_ntwk_transportnode")
    len_before_filter = len(os_mm_net_node)
    os_mm_net_node = os_mm_net_node.drop(columns=["os_parentid", "name"])
    os_mm_net_node = os_mm_net_node.drop_duplicates(subset=["nodeid", "geometry"])
    filter_removed = len_before_filter - len(os_mm_net_node)
    LOG.info("OS MMRN filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    return os_mm_net_node


def _clean_train_stations(
    config: model_config.Config, os_mm_net_node: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for train stations, then clip to boundary and write to file."""
    len_before_filter = len(os_mm_net_node)
    train_stations = os_mm_net_node[
        os_mm_net_node["os_nodetype"].isin(MMRN_NODE_TYPES['Train Stations'])
    ]
    train_stations = train_stations.drop(columns=["os_nodetype"])
    tfn_train_stations = clip_to_boundary(train_stations, boundary)
    filter_removed = len_before_filter - len(tfn_train_stations)
    LOG.info("Train stations filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_train_stations,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Train Stations"
        / "tfn_train_stations.gpkg",
    )


def _clean_tram_stations(
    config: model_config.Config, os_mm_net_node: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for tram stations, then clip to boundary and write to file."""
    len_before_filter = len(os_mm_net_node)
    tram_stations = os_mm_net_node[
        os_mm_net_node["os_nodetype"].str.contains("Tram Station", case=False, na=False)
    ]
    tram_stations = tram_stations.drop(columns=["os_nodetype"])
    tfn_tram_stations = clip_to_boundary(tram_stations, boundary)
    filter_removed = len_before_filter - len(tfn_tram_stations)
    LOG.info("Tram stations filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_tram_stations,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Tram Stations"
        / "tfn_tram_stations.gpkg",
    )


def _clean_rapid_transport_stations(
    config: model_config.Config, os_mm_net_node: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for rapid transport stations, then clip to boundary and write to file."""
    len_before_filter = len(os_mm_net_node)
    rapid_transport_stations = os_mm_net_node[
        os_mm_net_node["os_nodetype"].str.contains("Underground System", case=False, na=False)
    ]
    rapid_transport_stations = rapid_transport_stations.drop(columns=["os_nodetype"])
    tfn_rapid_transport_stations = clip_to_boundary(rapid_transport_stations, boundary)
    filter_removed = len_before_filter - len(tfn_rapid_transport_stations)
    LOG.info("Rapid transport stations filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_rapid_transport_stations,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Rapid Transport Stations"
        / "tfn_rapid_transport_stations.gpkg",
    )


def _clean_ferry_terminals(
    config: model_config.Config, os_mm_net_node: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for ferry terminals, then clip to boundary and write to file."""
    len_before_filter = len(os_mm_net_node)
    ferry_terminals = os_mm_net_node[
        os_mm_net_node["os_nodetype"].str.contains("Ferry", case=False, na=False)
    ]
    ferry_terminals = ferry_terminals.drop(columns=["os_nodetype"])
    tfn_ferry_terminals = clip_to_boundary(ferry_terminals, boundary)
    filter_removed = len_before_filter - len(tfn_ferry_terminals)
    LOG.info("Ferry terminals filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_ferry_terminals,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Ferry Terminals"
        / "tfn_ferry_terminals.gpkg",
    )


def _clean_bus_coach_stations(
    config: model_config.Config, os_mm_net_node: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for bus and coach stations, then clip to boundary and write to file."""
    len_before_filter = len(os_mm_net_node)
    bus_coach_stations = os_mm_net_node[
        (os_mm_net_node["os_nodetype"].str.contains("Bus Station", case=False, na=False))
        | (os_mm_net_node["os_nodetype"] == "Coach Station;Modal Change")
    ]
    bus_coach_stations = bus_coach_stations.drop(columns=["os_nodetype"])
    tfn_bus_coach_stations = clip_to_boundary(bus_coach_stations, boundary)
    filter_removed = len_before_filter - len(tfn_bus_coach_stations)
    LOG.info("Bus and coach stations filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_bus_coach_stations,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Bus Coach Stations"
        / "tfn_bus_coach_stations.gpkg",
    )


def _clean_tram_network(config: model_config.Config, tfn_rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail links for tram network, then write to file."""
    len_before_filter = len(tfn_rail_links)
    tfn_tram_links = tfn_rail_links[
        tfn_rail_links["rail_use"].isin(["Freight And Passenger", "Passenger"])
    ]
    tfn_tram_links = tfn_tram_links[
        tfn_tram_links["desc"].isin(["Tram", "Main Line And Tram"])
    ]
    filter_removed = len(tfn_tram_links)
    LOG.info("Tram network links filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_tram_links,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Tram Network"
        / "tfn_os_tram_links.gpkg",
    )


def _clean_rapid_transport_network(
    config: model_config.Config, tfn_rail_links: gpd.GeoDataFrame
) -> None:
    """Filter OS rail links for rapid transport network, then write to file."""
    len_before_filter = len(tfn_rail_links)
    tfn_rapid_transport = tfn_rail_links[
        tfn_rail_links["rail_use"].isin(["Freight And Passenger", "Passenger"])
    ]
    tfn_rapid_transport = tfn_rapid_transport[
        tfn_rapid_transport["desc"].isin(
            ["Rapid Transport System", "Main Line And Rapid Transport System"]
        )
    ]
    filter_removed = len_before_filter - len(tfn_rapid_transport)
    LOG.info("Rapid transport links filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_rapid_transport,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Rapid Transport Network"
        / "tfn_rapid_transport_links.gpkg",
    )


def _clean_charging_sites(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean ZapMap charging sites data, then write to file."""
    chg_sites = pd.read_csv(config.infrastructure.other.zapmap)
    chg_sites_gdf = gpd.GeoDataFrame(
        chg_sites,
        geometry=[geometry.Point(xy) for xy in zip(chg_sites["lon"], chg_sites["lat"], strict=False)],
        crs="EPSG:4326",
    )
    chg_sites_gdf = chg_sites_gdf[["identifier", "name", "speed", "value", "geometry"]]
    chg_sites_gdf = chg_sites_gdf.rename(columns={"identifier": "id", "value": "devices"})
    len_before_filter = len(chg_sites_gdf)
    chg_sites_gdf = chg_sites_gdf.drop_duplicates(subset=["geometry"])
    chg_sites_gdf = chg_sites_gdf[~chg_sites_gdf.geometry.is_empty]
    chg_sites_gdf = chg_sites_gdf[chg_sites_gdf.geometry.notna()]
    tfn_chg_sites = clip_to_boundary(chg_sites_gdf, boundary)
    filter_removed = len_before_filter - len(tfn_chg_sites)
    LOG.info("Charging sites filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_chg_sites,
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN EV Charging Sites"
        / "tfn_chg_sites.gpkg",
    )


def _clean_ncn(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean National Cycle Network data, then write to file."""
    ncn = gpd.read_file(config.infrastructure.other.ncn_sustrans)
    ncn = ncn.drop(columns=["RouteCat", "OpenStatus", "GlobalID"])
    len_before_filter = len(ncn)
    ncn = ncn.drop_duplicates(subset=["SegmentID", "geometry"])
    ncn_cols_replace = [
        "Desc_",
        "Greenway",
        "RouteType",
        "RouteNo",
        "LinkNo",
        "Surface",
        "Quality",
        "Lighting",
        "RoadClass",
    ]
    ncn[ncn_cols_replace] = ncn[ncn_cols_replace].replace(0, "N/A")
    tfn_ncn = clip_to_boundary(ncn, boundary)
    filter_removed = len_before_filter - len(tfn_ncn)
    LOG.info("NCN filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_ncn,
        config.paths.model_input / "Infrastructure" / "Other" / "TfN NCN" / "tfn_ncn.gpkg",
    )


## HAZARDS


def _clean_hazards(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean hazard data ready for analysis."""
    _clean_extreme_weather(config, boundary)
    _clean_flooding(config, boundary)
    _clean_ground_stability(config, boundary)
    _clean_coastal_erosion(config, boundary)


### EXTREME WEATHER


def _clean_extreme_weather(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all extreme weather datasets ready for analysis."""
    tfn_common_grid = _clean_common_grid(config, boundary)

    _clean_temp_max(config, tfn_common_grid)
    _clean_temp_min(config, tfn_common_grid)
    _clean_summer_precip(config, tfn_common_grid)
    _clean_winter_precip(config, tfn_common_grid)
    _clean_rain_days(config, boundary)
    _clean_drought_index(config, boundary)
    _clean_hot_summer_days(config, tfn_common_grid)
    _clean_extreme_summer_days(config, tfn_common_grid)
    _clean_frost_days(config, tfn_common_grid)
    _clean_icing_days(config, tfn_common_grid)
    _clean_wind_speed(config, boundary)
    _clean_wind_driven_rain(config, boundary)


def _clean_common_grid(config: model_config.Config, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create and prepare common grid DataFrame for variables on same 12km BNG."""
    temp_max = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.max_temp_summer.zip_path}!"
        f"{config.hazards.extreme_weather.max_temp_summer.file_path}"
    )
    temp_max["grid_id"] = range(1, len(temp_max) + 1)
    common_grid = temp_max[["grid_id", "geometry"]]
    tfn_common_grid = clip_to_boundary(common_grid, boundary)
    tfn_common_grid = explode_to_polygons(tfn_common_grid)
    write_to_file(
        tfn_common_grid,
        config.paths.model_input / "Other" / "TfN Common Grid" / "tfn_common_grid.gpkg",
    )
    return tfn_common_grid


def _clean_temp_max(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean max summer temperature change projections, then write to file."""
    temp_max = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.max_temp_summer.zip_path}!"
        f"{config.hazards.extreme_weather.max_temp_summer.file_path}"
    )
    temp_max["grid_id"] = range(1, len(temp_max) + 1)
    temp_max = temp_max[["grid_id", "tasmax_s_4", "tasmax__22"]]
    temp_max = temp_max.rename(
        columns={"tasmax_s_4": "tasmax_s_c", "tasmax__22": "tasmax_s_f"}
    )
    temp_max["tasmax_s_f"] = temp_max["tasmax_s_c"] + temp_max["tasmax_s_f"]
    len_before_filter = len(temp_max)
    tfn_temp_max = temp_max[temp_max["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_temp_max)
    LOG.info("Summer max temperature change projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_temp_max,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Summer Max Temperature Change Projections"
        / "tfn_temp_max.csv",
    )


def _clean_temp_min(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean min winter temperature change projections, then write to file."""
    temp_min = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.min_temp_winter.zip_path}!"
        f"{config.hazards.extreme_weather.min_temp_winter.file_path}"
    )
    temp_min["grid_id"] = range(1, len(temp_min) + 1)
    temp_min = temp_min[["grid_id", "tasmin_w_4", "tasmin__22"]]
    temp_min = temp_min.rename(
        columns={"tasmin_w_4": "tasmin_w_c", "tasmin__22": "tasmin_w_f"}
    )
    temp_min["tasmin_w_f"] = temp_min["tasmin_w_c"] + temp_min["tasmin_w_f"]
    len_before_filter = len(temp_min)
    tfn_temp_min = temp_min[temp_min["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_temp_min)
    LOG.info("Winter minimum temperature change projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_temp_min,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Winter Min Temperature Change Projections"
        / "tfn_temp_min.csv",
    )


def _clean_summer_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean summer precipitation change projections, then write to file."""
    precip_sum = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.precip_summer.zip_path}!"
        f"{config.hazards.extreme_weather.precip_summer.file_path}"
    )
    precip_sum["grid_id"] = range(1, len(precip_sum) + 1)
    precip_sum = precip_sum[["grid_id", "pr_summe_3", "pr_summ_21"]]
    precip_sum = precip_sum.rename(
        columns={"pr_summe_3": "pr_s_c", "pr_summ_21": "pr_s_pct_f"}
    )
    precip_sum["pr_s_f"] = precip_sum["pr_s_c"] * (1 + (precip_sum["pr_s_pct_f"] / 100))
    precip_sum = precip_sum.drop(columns=["pr_s_pct_f"])
    len_before_filter = len(precip_sum)
    tfn_precip_sum = precip_sum[precip_sum["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_precip_sum)
    LOG.info("Summer precipitation change projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_precip_sum,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Summer Precipitation Change Projections"
        / "tfn_precip_sum.csv",
    )


def _clean_winter_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean winter precipitation change projections, then write to file."""
    precip_win = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.precip_winter.zip_path}!"
        f"{config.hazards.extreme_weather.precip_winter.file_path}"
    )
    precip_win["grid_id"] = range(1, len(precip_win) + 1)
    precip_win = precip_win[["grid_id", "pr_winte_3", "pr_wint_21"]]
    precip_win = precip_win.rename(
        columns={
            "pr_winte_3": "pr_w_c",
            "pr_wint_21": "pr_w_pct_f",
        },
    )
    precip_win["pr_w_f"] = precip_win["pr_w_c"] * (1 + (precip_win["pr_w_pct_f"] / 100))
    precip_win = precip_win.drop(columns=["pr_w_pct_f"])
    len_before_filter = len(precip_win)
    tfn_precip_win = precip_win[precip_win["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_precip_win)
    LOG.info("Winter precipitation change projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_precip_win,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Winter Precipitation Change Projections"
        / "tfn_precip_win.csv",
    )


def _clean_rain_days(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean 10mm rain days observations, then write to file."""
    rain_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.rain_days.zip_path}!"
        f"{config.hazards.extreme_weather.rain_days.file_path}"
    )
    len_before_filter = len(rain_days)
    tfn_rain_days = clip_to_boundary(rain_days, boundary)
    tfn_rain_days = explode_to_polygons(tfn_rain_days)
    tfn_rain_days = tfn_rain_days.rename(columns={"Rain10mmDa": "rain_d_c"})
    tfn_rain_days = tfn_rain_days.drop(columns=["part"])
    filter_removed = len_before_filter - len(tfn_rain_days)
    LOG.info("10mm rain days 1991-2020 filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_rain_days,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN 10mm Rain Days 1991-2020"
        / "tfn_rain_days.gpkg",
    )


def _clean_drought_index(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean drought severity index data, then write to file."""
    drought_index = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.drought_index.zip_path}!"
        f"{config.hazards.extreme_weather.drought_index.file_path}"
    )
    drought_index = drought_index[["DSI12_ba_4", "DSI12_40_m", "geometry"]]
    drought_index = drought_index.rename(
        columns={"DSI12_ba_4": "dsi_c", "DSI12_40_m": "dsi_f"}
    )
    len_before_filter = len(drought_index)
    tfn_drought = clip_to_boundary(drought_index, boundary)
    tfn_drought = explode_to_polygons(tfn_drought)
    tfn_drought = tfn_drought.drop(columns=["part"])
    filter_removed = len_before_filter - len(tfn_drought)
    LOG.info("Drought severity index filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_drought,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Drought Severity Index"
        / "tfn_drought_index.gpkg",
    )


def _clean_hot_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean hot summer days projections, then write to file."""
    hot_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.hot_days.zip_path}!"
        f"{config.hazards.extreme_weather.hot_days.file_path}"
    )
    hot_days["grid_id"] = range(1, len(hot_days) + 1)
    hot_days = hot_days[["grid_id", "HSD_base_4", "HSD_40_med"]]
    hot_days = hot_days.rename(columns={"HSD_base_4": "hsd_c", "HSD_40_med": "hsd_f"})
    len_before_filter = len(hot_days)
    tfn_hot_days = hot_days[hot_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_hot_days)
    LOG.info("Hot summer days projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_hot_days,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Hot Summer Days Projections"
        / "tfn_hot_days.csv",
    )


def _clean_extreme_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean extreme summer days projections, then write to file."""
    extr_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.extreme_summer_days.zip_path}!"
        f"{config.hazards.extreme_weather.extreme_summer_days.file_path}"
    )
    extr_days["grid_id"] = range(1, len(extr_days) + 1)
    extr_days = extr_days[["grid_id", "ESD_base_4", "ESD_40_med"]]
    extr_days = extr_days.rename(columns={"ESD_base_4": "esd_c", "ESD_40_med": "esd_f"})
    len_before_filter = len(extr_days)
    tfn_extr_days = extr_days[extr_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_extr_days)
    LOG.info("Extreme summer days projections - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_extr_days,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Extreme Summer Days Projections"
        / "tfn_extr_days.csv",
    )


def _clean_frost_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean frost days projections, then write to file."""
    frost_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.frost_days.zip_path}!"
        f"{config.hazards.extreme_weather.frost_days.file_path}"
    )
    frost_days["grid_id"] = range(1, len(frost_days) + 1)
    frost_days = frost_days[["grid_id", "FrostDay_3", "FrostDa_18"]]
    frost_days = frost_days.rename(
        columns={"FrostDay_3": "frost_d_c", "FrostDa_18": "frost_d_f"}
    )
    len_before_filter = len(frost_days)
    tfn_frost_days = frost_days[frost_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_frost_days)
    LOG.info("Frost days projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_frost_days,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Frost Days Projections"
        / "tfn_frost_days.csv",
    )


def _clean_icing_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean icing days projections, then write to file."""
    ice_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.icing_days.zip_path}!"
        f"{config.hazards.extreme_weather.icing_days.file_path}"
    )
    ice_days["grid_id"] = range(1, len(ice_days) + 1)
    ice_days = ice_days[["grid_id", "IcingDay_3", "IcingDa_18"]]
    ice_days = ice_days.rename(columns={"IcingDay_3": "ice_d_c", "IcingDa_18": "ice_d_f"})
    len_before_filter = len(ice_days)
    tfn_ice_days = ice_days[ice_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_ice_days)
    LOG.info("Icing days projections filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_ice_days,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Icing Days Projections"
        / "tfn_ice_days.csv",
    )


def _clean_wind_speed(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read wind speed projections, calculate metrics, clean, then write to file."""
    windspd_c_combined, len_before_filter_current = _read_wind_speed_reduce(
        config.hazards.extreme_weather.wind_spd_current, "c"
    )
    windspd_f_combined, len_before_filter_future = _read_wind_speed_reduce(
        config.hazards.extreme_weather.wind_spd_forecast, "f"
    )

    len_before_filter = len_before_filter_current + len_before_filter_future

    windspd_combined = _windspd_merge_and_fill(
        windspd_c_combined, windspd_f_combined, "avg_excd_f"
    )

    windspd_combined["geometry"] = [
        _convert_point_to_grid(x, y, 2500)
        for x, y in zip(
            windspd_combined["projection_x_coordinate"],
            windspd_combined["projection_y_coordinate"],
            strict=False,
        )
    ]
    windspd_combined = gpd.GeoDataFrame(
        windspd_combined, geometry="geometry", crs="EPSG:27700"
    )
    windspd_combined = windspd_combined[
        ["p99_c", "avg_excd_c", "p99_f", "avg_excd_f", "geometry"]
    ]
    tfn_windspd = clip_to_boundary(windspd_combined, boundary)
    tfn_windspd = explode_to_polygons(tfn_windspd)
    tfn_windspd = tfn_windspd.drop(columns=["part"])
    filter_removed = len_before_filter - len(tfn_windspd)
    LOG.info("Wind speed projections filtered - %s of %s (%s percent) rows removed", len_before_filter, filter_removed, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_windspd,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Wind Speed Projections"
        / "tfn_windspd.gpkg",
    )


def _read_wind_speed_reduce(xr_path: pathlib.Path, tp: str) -> pd.DataFrame:
    """Read wind speed projections, calculate metrics and return a merged dataframe."""
    windspd = xr.open_dataset(xr_path).to_dataframe()
    len_before_filter = len(windspd)
    exc = _calculate_exceedance(20, windspd, "wsgmax10m", tp)
    pct = _calculate_percentile(windspd, 0.99, "wsgmax10m")
    pct.columns = [
        "projection_y_coordinate",
        "projection_x_coordinate",
        "latitude",
        "longitude",
        f"p99_{tp}",
    ]
    return _windspd_merge_and_fill(pct, exc, f"avg_excd_{tp}"), len_before_filter


def _windspd_merge_and_fill(
    df1: pd.DataFrame, df2: pd.DataFrame, fill_col: str
) -> pd.DataFrame:
    """Merge two dataframes on common coordinates, fill a given column with 0 values."""
    return df1.merge(
        df2,
        on=["projection_y_coordinate", "projection_x_coordinate", "latitude", "longitude"],
        how="outer",
    ).fillna({fill_col: 0})


def _calculate_exceedance(
    threshold: int, df: pd.DataFrame, variable: str, timescale: str
) -> pd.DataFrame:
    """Compute average exceedance days per geometry for values above a threshold."""
    df["exceedance"] = df[variable] > threshold

    # Group by grid square and year, and count exceedance days
    exceedance_counts = (
        df[df["exceedance"]]
        .groupby(
            [
                "projection_y_coordinate",
                "projection_x_coordinate",
                "latitude",
                "longitude",
                "year",
            ]
        )
        .size()
        .reset_index(name="exceedance_days")
    )

    # Calculate the average exceedance days per year for each grid square
    return (
        exceedance_counts.groupby(
            ["projection_y_coordinate", "projection_x_coordinate", "latitude", "longitude"]
        )["exceedance_days"]
        .mean()
        .reset_index(name=f"avg_excd_{timescale}")
    )


def _calculate_percentile(df: pd.DataFrame, quantile: float, variable: str) -> pd.DataFrame:
    """Calculate the percentiles per geometry for a given variable."""
    return df.pivot_table(
        index=[
            "projection_y_coordinate",
            "projection_x_coordinate",
            "latitude",
            "longitude",
            "time",
        ],
        values=variable,
        aggfunc=lambda x: x.quantile(quantile),
    ).reset_index()


def _clean_wind_driven_rain(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean wind driven rain index data, then write to file."""
    wdr = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.wdr_index.zip_path}!"
        f"{config.hazards.extreme_weather.wdr_index.file_path}"
    )
    len_before_filter = len(wdr)
    # Aggregate by wind direction to calculate mean wind speed
    wdr_agg = (
        wdr.groupby(["x_coord", "y_coord"])
        .agg({"WDR_base_1": "mean", "WDR_40_Med": "mean", "geometry": "first"})
        .reset_index()
    )

    wdr_agg = wdr_agg[["WDR_base_1", "WDR_40_Med", "geometry"]]
    wdr_agg = wdr_agg.rename(columns={"WDR_base_1": "wdr_c", "WDR_40_Med": "wdr_f"})
    wdr_agg = gpd.GeoDataFrame(wdr_agg, geometry="geometry", crs="EPSG:3857")
    tfn_wdr = clip_to_boundary(wdr_agg, boundary)
    tfn_wdr = explode_to_polygons(tfn_wdr)
    tfn_wdr = tfn_wdr.drop(columns=["part"])
    filter_removed = len_before_filter - len(tfn_wdr)
    LOG.info("Wind driven rain index filtered - %s of %s (%s percent) rows removed", filter_removed, len_before_filter, (filter_removed/len_before_filter)*100)
    write_to_file(
        tfn_wdr,
        config.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Wind Driven Rain Index"
        / "tfn_wdr.gpkg",
    )


### FLOODING


def _clean_flooding(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean flooding data ready for analysis."""
    code_number_map = {
        "NT": ["50", "55"],
        "NU": ["00", "05"],
        "NX": ["50"],
        "NY": ["00", "05", "50", "55"],
        "NZ": ["00", "05", "50"],
        "OV": ["00"],
        "SD": ["00", "05", "50", "55"],
        "SE": ["00", "05", "50", "55"],
        "SJ": ["00", "05", "50", "55"],
        "SK": ["00", "05", "50", "55"],
        "TA": ["00", "05"],
        "TF": ["00", "05", "50", "55"],
    }

    if config.switches.flood_zip_extract:
        _extract_flood_data(config, code_number_map)

    _clean_flood(
        config,
        "RoFRS",
        "RoFRS",
        "v202501",
        boundary,
        "TfN RoFRS CC/tfn_rofrs_cc.gpkg",
        True,
        code_number_map,
    )
    _clean_flood(
        config,
        "RoFRS",
        "RoFRS",
        "v202501",
        boundary,
        "TfN RoFRS/tfn_rofrs.gpkg",
        False,
        code_number_map,
    )
    _clean_flood(
        config,
        "RoFSW CC",
        "RoFSW",
        "v202509",
        boundary,
        "TfN RoFSW CC/tfn_rofsw_cc.gpkg",
        True,
        code_number_map,
    )
    _clean_flood(
        config,
        "RoFSW",
        "RoFSW",
        "v202509",
        boundary,
        "TfN RoFSW/tfn_rofsw.gpkg",
        False,
        code_number_map,
    )


def _extract_flood_gdb_file(
    config: model_config.Config, code: str, number: str, flood_data: str, version: str, cc: bool
) -> gpd.GeoDataFrame | None:
    """Extract a flood gdb file from a zip file given its BNG code and number, and version."""
    try:
        base_path = config.paths.raw_input / "Hazards" / "Flooding" / flood_data
        if cc:
            zip_path = (
                base_path
                / code
                / f"{flood_data}_Climate_Change_01_{code}{number}_{version}.zip"
            )
            extract_to = base_path / code
            gdb_path = (
                extract_to / f"{flood_data}_Climate_Change_01_{code}{number}_{version}.gdb"
            )
        else:
            zip_path = base_path / code / f"{flood_data}_{code}{number}_{version}.zip"
            extract_to = base_path / code
            gdb_path = extract_to / f"{flood_data}_{code}{number}_{version}.gdb"

        if not zip_path.exists():
            LOG.warning("Zip file not found: %s", zip_path)
            return None

        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        if not gdb_path.exists():
            LOG.warning("GDB folder not found: %s", gdb_path)
            return None

        layers = fiona.listlayers(gdb_path)
        if not layers:
            LOG.warning("No layers found in: %s", gdb_path)
            return None

        LOG.info("Available layers: %s", layers)
        return gpd.read_file(gdb_path, layer=layers[0])

    except Exception:
        LOG.exception("Error processing %s%s", code, number)
        return None


def _read_flood_gdb(
    config: model_config.Config,
    code: str,
    number: str,
    file_name: str,
    flood_data: str,
    version: str,
    climate_change_switch: bool,
) -> gpd.GeoDataFrame | None:
    """Read first layer of flood gdb file."""
    base_path = config.paths.raw_input / "Hazards" / "Flooding" / file_name / code

    if climate_change_switch:
        gdb_path = base_path / f"{flood_data}_Climate_Change_01_{code}{number}_{version}.gdb"
    else:
        gdb_path = base_path / f"{flood_data}_{code}{number}_{version}.gdb"

    # Check if GDB folder exists
    if not gdb_path.exists():
        raise FileNotFoundError(f"GBD folder not found: {gdb_path}")

    layers = fiona.listlayers(gdb_path)
    if not layers:
        raise ValueError(f"No layers found in GDB: {gdb_path}")

    LOG.info("Available layers: %s", layers)
    return gpd.read_file(gdb_path, layer=layers[0])


def _extract_flood_data(config: model_config.Config, code_number_map: dict[str, list[str]]) -> None:
    """Extract geodatabase files from raw RoFRS and RoFSW flood data."""
    for code, num_list in code_number_map.items():
        for number in num_list:
            # Forecast (Climate Change) data
            _extract_flood_gdb_file(config, code, number, "RoFRS", "v202501", True)
            _extract_flood_gdb_file(config, code, number, "RoFSW", "v202509", True)

            # Current data
            _extract_flood_gdb_file(config, code, number, "RoFRS", "v202501", False)
            _extract_flood_gdb_file(config, code, number, "RoFSW", "v202509", False)


def _clean_flood(
    config: model_config.Config,
    file_name: str,
    flood_type: str,
    version: str,
    boundary: gpd.GeoDataFrame,
    out_path: str,
    climate_change_switch: bool,
    code_number_map: dict[str, list[str]],
) -> None:
    """Read and clean flood data, then write to file."""
    flood_datasets = []
    for code, num_list in code_number_map.items():
        for number in num_list:
            LOG.info("Processing: %s%s", code, number)
            flood_data = _read_flood_gdb(config, code, number, file_name, flood_type, version, climate_change_switch)  # Read file
            tfn_flood_data = clip_to_boundary(flood_data, boundary)
            flood_datasets.append(tfn_flood_data)  # Add to list

    flood_data_combined = gpd.GeoDataFrame(
        pd.concat(flood_datasets, ignore_index=True), geometry="geometry", crs=flood_datasets[0].crs
    )
    flood_data_combined = _extract_poly_from_geomcollection(flood_data_combined)
    flood_data_combined = flood_data_combined[["Risk_band", "geometry"]]
    write_to_file(flood_data_combined, config.paths.model_input / "Hazards" / "Flooding" / out_path)


### GROUND STABILITY


def _clean_ground_stability(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean ground stability data ready for analysis."""
    _clean_geosure(config, boundary)
    _clean_geoclimate(config, boundary)


def _clean_geosure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean GeoSureHexGrids data, merge by nearest centroids, then write to file."""
    geosure_layers = {
        "collapsible_deposits": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.collapsible_deposits}"
        ),
        "compressible_ground": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.compressible_ground}"
        ),
        "landslides": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.landslides}"
        ),
        "running_sand": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.running_sand}"
        ),
        "shrink_swell": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.shrink_swell}"
        ),
        "soluble_rocks": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.soluble_rocks}"
        ),
    }

    tfn_geosure_layers = {}
    for code, df in geosure_layers.items():
        df_clean = df.rename(columns={"CLASS": f"{code}_risk"})
        tfn_geosure_layers[code] = clip_to_boundary(df_clean, boundary)
        tfn_geosure_layers[code] = explode_to_polygons(tfn_geosure_layers[code])

    # Merge layers based on nearest centroids
    base_code = next(iter(geosure_layers.keys()))
    tfn_geosure = tfn_geosure_layers[base_code][
        [f"{base_code}_risk", "geometry"]
    ].copy()
    for code, layer in tfn_geosure_layers.items():
        if code == base_code:
            continue  # skip the base layer
        layer_subset = layer[
            [f"{code}_risk", "geometry"]
        ]  # Select only the relevant class and geometry columns
        matched = _nearest_centroids(
            tfn_geosure, layer_subset
        )  # Apply nearest centroid matching
        tfn_geosure[f"{code}_risk"] = matched[
            f"{code}_risk"
        ]  # Add the matched CLASS column to the base dataframe

    tfn_geosure = tfn_geosure[
        [
            "collapsible_deposits_risk",
            "compressible_ground_risk",
            "landslides_risk",
            "running_sand_risk",
            "shrink_swell_risk",
            "soluble_rocks_risk",
            "geometry",
        ]
    ]
    write_to_file(
        tfn_geosure,
        config.paths.model_input
        / "Hazards"
        / "Ground Stability"
        / "TfN GeoSure"
        / "tfn_geosure.gpkg",
    )


def _clean_geoclimate(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean GeoClimate Shrink-Swell data, then write to file."""
    for year, filepath in config.hazards.ground_stability.geo_shrink_swell:
        gdf = gpd.read_file(filepath)
        gdf = gdf.rename(columns={"CLASS": "shrink_swell_geoclimate_risk"})
        gdf = gdf[["shrink_swell_geoclimate_risk", "geometry"]]
        tfn_gdf = clip_to_boundary(gdf, boundary)
        tfn_gdf = explode_to_polygons(tfn_gdf)
        write_to_file(
            tfn_gdf,
            config.paths.model_input
            / "Hazards"
            / "Ground Stability"
            / "BGS Shrink Swell"
            / year
            / f"tfn_bgs_ss_{year}.gpkg",
        )


### COASTAL EROSION


def _clean_coastal_erosion(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean coastal erosion data ready for analysis."""
    _clean_giz(config, boundary)
    _clean_ncerm(config, boundary)


def _clean_giz(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean Ground Instability Zones data from NCERM, then write to file."""
    ncerm_giz = gpd.read_file(
        f"zip://{config.hazards.coastal_erosion.zip_path}!{config.hazards.coastal_erosion.giz}"
    )
    ncerm_giz = ncerm_giz[["smp_no", "geometry"]]
    tfn_ncerm_giz = clip_to_boundary(ncerm_giz, boundary)
    tfn_ncerm_giz = explode_to_polygons(tfn_ncerm_giz)
    write_to_file(
        tfn_ncerm_giz,
        config.paths.model_input
        / "Hazards"
        / "Coastal Erosion"
        / "NCERM"
        / "Ground Instability Zones"
        / "tfn_ncerm_giz.gpkg",
    )


def _clean_ncerm(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean erosion data from NCERM for 2055, and 2105, then write to file."""
    for year in ["2055", "2105"]:
        gdf = gpd.read_file(
            f"zip://{config.hazards.coastal_erosion.zip_path}!/"
            + config.hazards.coastal_erosion.smp[year]
        )
        gdf = gdf[["smp_name", "geometry"]]
        tfn_gdf = clip_to_boundary(gdf, boundary)
        tfn_gdf = explode_to_polygons(tfn_gdf)
        write_to_file(
            tfn_gdf,
            config.paths.model_input
            / "Hazards"
            / "Coastal Erosion"
            / "NCERM"
            / f"SMP_{year}_70CC"
            / f"tfn_ncerm_smp_{year}_70CC.gpkg",
        )


## IMPACT


def _clean_impact(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean impact datasets ready for analysis."""
    _clean_freight_demand(config, boundary)
    _clean_noham_flows(config)


### FREIGHT


def _clean_freight_demand(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean freight demand data ready for analysis."""
    tfn_freight_network_demand = _read_freight_demand(config.impact.freight_demand, boundary)

    tfn_os_freight_network_demand = _map_freight_networks(
        tfn_freight_network_demand,
        config.paths.model_input
        / "Infrastructure"
        / "Rail"
        / "TfN OS Freight Rail"
        / "tfn_freight_rail_links.gpkg",
    )

    write_to_file(
        tfn_os_freight_network_demand,
        config.paths.model_input
        / "Impact"
        / "TfN Freight Flows"
        / "tfn_freight_network_demand.gpkg",
    )


def _read_freight_demand(path: pathlib.Path, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Read and clean freight demand data, and return as GeoDataFrame."""
    freight_network_demand = gpd.read_file(path)
    freight_network_demand = freight_network_demand[
        ["dij_id", "2022_23_total", "2050_51 sc2_total", "geometry"]
    ]
    return clip_to_boundary(freight_network_demand, boundary)


def _map_freight_networks(
    tfn_freight_network_demand: gpd.GeoDataFrame, os_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Map freight demand data onto OS network, then clean and return."""
    tfn_os_freight_rail = gpd.read_file(os_path)
    tfn_os_freight_rail = tfn_os_freight_rail.to_crs(tfn_freight_network_demand.crs)
    tfn_os_freight_network_demand = gpd.sjoin_nearest(
        tfn_os_freight_rail,
        tfn_freight_network_demand,
        how="left",
        max_distance=500,
        distance_col="distance",
    )
    tfn_os_freight_network_demand[["2022_23_total", "2050_51 sc2_total"]] = (
        tfn_os_freight_network_demand[["2022_23_total", "2050_51 sc2_total"]].fillna(0)
    )
    return tfn_os_freight_network_demand.drop(columns=["index_right"])


### NoHAM


def _clean_noham_flows(config: model_config.Config) -> None:
    """Clean NoHAM flows data, aggregate link flows, merge with network, then write to file."""
    link_flows = _aggregate_link_flows_year(config)

    for scenario, noham_entry in config.infrastructure.road.noham.items():
        year = noham_entry.year
        tfn_noham_flows = link_flows[year]
        tfn_noham_net_flows = _merge_noham_flow_network(
            tfn_noham_flows,
            config.paths.model_input
            / "Infrastructure"
            / "Road"
            / f"TfN NoHAM {year}"
            / f"tfn_noham_{year}.gpkg",
        )
        write_to_file(
            tfn_noham_net_flows,
            config.paths.model_input
            / "Impact"
            / "TfN NoHAM Flows"
            / year
            / f"tfn_noham_net_flows_{scenario}.gpkg",
        )


def _read_noham_h5(
    year: str,
    time_period: str,
    user_class: str,
    noham_path: pathlib.Path,
    output_path: pathlib.Path,
    extract: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and clean NoHAM h5 files and extract the link, routes, and od's DataFrames."""
    if extract:
        with py7zr.SevenZipFile(noham_path, mode="r") as archive:
            archive.extract(
                path=output_path,
                targets=[
                    (
                        f"input h5s/"
                        f"{year}/"
                        f"NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5"
                    )
                ],
            )

    with h5py.File(
        output_path
        / "input h5s"
        / year
        / f"NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5",
        "r",
    ) as f:
        # Get OD's
        od_columns = [x.decode() for x in f["data/OD/block0_items"][:]]
        od_values = f["data/OD/block0_values"][:]

        od_labels = [
            f["data/OD/axis1_label0"][:],
            f["data/OD/axis1_label1"][:],
            f["data/OD/axis1_label2"][:],
            f["data/OD/axis1_label3"][:],
            f["data/OD/axis1_label4"][:],
        ]

        # Get routes
        route_col = [x.decode() for x in f["data/Route/block0_items"]][:][0]
        route_values = f["data/Route/block0_values"][:]

        # MultiIndex labels
        route_label0 = f["data/Route/axis1_label0"][:]
        route_label1 = f["data/Route/axis1_label1"][:]

        # Get links
        link_values = f["data/link/block0_values"][:]
        link_columns = [x.decode() for x in f["data/link/block0_items"][:]]

    # Build OD Dataframe
    od_multi_index = pd.MultiIndex.from_arrays(
        od_labels, names=["o", "d", "route", "uc", "total_links"]
    )
    od_df = pd.DataFrame(od_values, index=od_multi_index, columns=od_columns)

    # Build Route Dataframe
    route_multi_index = pd.MultiIndex.from_arrays(
        [route_label0, route_label1], names=["route", "link_id"]
    )
    route_df = pd.DataFrame(route_values, index=route_multi_index, columns=[route_col])

    # Build Link DataFrame
    link_df = pd.DataFrame(link_values, columns=link_columns)

    return od_df, route_df, link_df


def _aggregate_link_flows(
    ods: pd.DataFrame, routes: pd.DataFrame, links: pd.DataFrame
) -> pd.DataFrame:
    """Take NoHAM od's, routes, and link to create aggregated link flows DataFrame."""
    # Flatten OD and Route data
    od_flat = ods.reset_index()[["route", "abs_demand"]]
    route_flat = routes.reset_index()[["route", "link_id"]]

    # Merge OD demand with route links
    od_links = od_flat.merge(route_flat, on="route")

    # Aggregate demand per link_id
    link_demand = od_links.groupby("link_id")["abs_demand"].sum().reset_index()

    return link_demand.merge(links, left_on="link_id", right_index=True)


def _aggregate_link_flows_year(config: model_config.Config) -> dict[str, pd.DataFrame]:
    """Aggregate link flows for each year, time period, and user class."""
    years = [
        config.infrastructure.road.noham['current'].year,
        config.infrastructure.road.noham['forecast'].year
        ]
    time_periods = ["TS1", "TS2", "TS3"]
    user_classes = ["uc1", "uc2", "uc3", "uc4", "uc5"]

    link_flows = {}
    for year in years:
        ts_dfs = []
        for time_period in time_periods:
            uc_dfs = []
            for user_class in user_classes:
                LOG.info("Processing: %s %s %s", year, time_period, user_class)
                od_df, route_df, link_df = _read_noham_h5(
                    year,
                    time_period,
                    user_class,
                    config.impact.noham_demand.zip_path,
                    config.impact.noham_demand.output_path,
                    config.switches.noham_zip_extract,
                )
                link_demand = _aggregate_link_flows(
                    od_df, route_df, link_df
                )  # Get link based demand
                link_demand = link_demand.rename(
                    columns={"abs_demand": f"{user_class}_{time_period}"}
                )  # Rename demand column
                link_demand["link_id"] = (
                    link_demand["a"].astype(str) + "_" + link_demand["b"].astype(str)
                )  # Create unique noham link id
                link_demand = link_demand[
                    ["link_id", f"{user_class}_{time_period}"]
                ]  # Keep relevant columns
                uc_dfs.append(link_demand)  # Add to list of df's

            # Merge all user class dataframes
            combined_uc_df = uc_dfs[0]
            for df_uc in uc_dfs[1:]:
                combined_uc_df = combined_uc_df.merge(df_uc, on="link_id", how="outer")

            # Compute total demand for all vehicles for each time period
            combined_uc_df[f"all_vehs_{time_period}"] = combined_uc_df[
                [f"{uc}_{time_period}" for uc in user_classes]
            ].sum(axis=1)

            # Store result
            ts_dfs.append(combined_uc_df)

        # Merge all time period dataframes
        combined_ts_df = ts_dfs[0]
        for df_ts in ts_dfs[1:]:
            combined_ts_df = combined_ts_df.merge(df_ts, on="link_id", how="outer")

        # Compute totals for each user class across all time periods
        for uc in user_classes:
            combined_ts_df[f"{uc}_total"] = combined_ts_df[
                [f"{uc}_{tp}" for tp in time_periods]
            ].sum(axis=1)

        # Compute total of each user class across all time periods
        combined_ts_df["all_vehs_total"] = combined_ts_df[
            [f"all_vehs_{tp}" for tp in time_periods]
        ].sum(axis=1)

        # Add to data dictionary
        link_flows[year] = combined_ts_df

    return link_flows


def _merge_noham_flow_network(
    tfn_noham_flows: pd.DataFrame, noham_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Merge NoHAM flows onto road network, then return as GeoDataFrame."""
    tfn_noham_link = gpd.read_file(noham_path)
    tfn_noham_net_flows = tfn_noham_link.merge(
        tfn_noham_flows,
        on="link_id",
        how="left",  # Keep all network, adding flows where available
    )
    return gpd.GeoDataFrame(tfn_noham_net_flows, geometry="geometry", crs=tfn_noham_link.crs)
