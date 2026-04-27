"""Cleans raw input data to prepare it for input into the model."""

### LOAD LIBRARIES
import gc
import logging
import os
import pathlib
from zipfile import ZipFile

import fiona
import geopandas as gpd
import pandas as pd
import py7zr
import xarray as xr
from shapely import geometry

from cvt import file_paths, model_config

LOG = logging.getLogger(__name__)

### ENVIRONMENT VARIABLES ###

# Minimum a and b values for NoHAM road links to keep
_NOHAM_ROAD_THRESHOLD = int(os.getenv("NOHAM_ROAD_THRESHOLD", "10000"))

_NOHAM_TIME_PERIODS = ["TS1", "TS2", "TS3"]
_NOHAM_USER_CLASSES = ["uc1", "uc2", "uc3", "uc4", "uc5"]

# British National Grid CRS, for use in spatially merging datasets
BNG_CRS = os.getenv("BNG_CRS", "EPSG:27700")

_FREIGHT_DEMAND_NETWORK_MAP_MAX_DISTANCE = int(
    os.getenv("_FREIGHT_DEMAND_NETWORK_MAP_MAX_DISTANCE", "500")
)

### MODULE CONSTANTS ###
MMRN_NODE_TYPES = {
    "Train Stations": [
        "Railway Station;Modal Change",
        "Railway Station;Railway Station (Underground System);Modal Change",
        "Railway Station (Principal);Modal Change",
        "Railway Station (Principal);Railway Station (Underground System);Modal Change",
        "Railway Station;Tram Station;Modal Change",
        "Railway Station (Non Public Accessible);Modal Change",
        "Railway Station (Principal);Tram Station;Modal Change",
    ]
}


_FLOOD_CODE_NUMBER_MAP = {
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


_WIND_SPEED_EXCEEDANCE_THRESHOLD = 20
_WIND_SPEED_PERCENTILE = 0.99

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
    mode: str = "w",
    layer: str | None = None,
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
    mode: str, optional
        File writing mode, "w" for write (default) or "a" for append.
    layer: str or None, optional
        Name of the layer to write to for GIS formats. Required for GPKG, ignored for CSV.

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
        data.to_csv(output_path, index=False, mode=mode, header=mode == "w")
    elif ext in driver_map:
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError(f"{ext} requires a GeoDataFrame, got {type(data)}")
        data.to_file(output_path, layer=layer, driver=driver_map[ext], mode=mode)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def explode_to_polygons(gdf: gpd.GeoDataFrame, track_part: bool = False) -> gpd.GeoDataFrame:
    """
    Explodes the MultiPolygons and GeomCollections in a GeoDataFrame into Polygons.

    Converts rows in a GeoDataFrame with a MultiPolygon or GeometryCollection geometry into
    multiple rows with Polygon geometries, adding a 'part' column to keep track of how many new
    rows have been created for each if specified. Leaves other geometries as they are, and
    preserves the original CRS.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        The input GeoDataFrame that needs to be cleaned
    track_part: bool = False
        True if tracking the parts of each extracted polygon is necessary, False if not.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing only Polygon geometries, with a new column 'part' to keep
        track.
    """
    counts = {
        "multipolygon": 0,
        "multipolygon_polygon": 0,
        "geometry_collection": 0,
        "geometry_collection_polygon": 0,
    }

    rows = []
    for _idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Polygon":
            new_row = row.copy()
            new_row["part"] = 0
            rows.append(new_row)
        elif geom.geom_type == "MultiPolygon":
            counts["multipolygon"] += 1
            for i, poly in enumerate(geom.geoms):
                counts["multipolygon_polygon"] += 1
                new_row = row.copy()
                new_row.geometry = poly
                new_row["part"] = i
                rows.append(new_row)
        elif geom.geom_type == "GeometryCollection":
            counts["geometry_collection"] += 1
            poly_count = 0
            for part in geom.geoms:
                if part.geom_type == "Polygon":
                    counts["geometry_collection_polygon"] += 1
                    new_row = row.copy()
                    new_row.geometry = part
                    new_row["part"] = poly_count
                    rows.append(new_row)
                    poly_count += 1

    LOG.info(
        "Exploded %s MultiPolygons and %s GeometryCollections into %s Polygons.",
        counts["multipolygon"],
        counts["geometry_collection"],
        counts["multipolygon_polygon"] + counts["geometry_collection_polygon"],
    )

    exploded_gdf = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)

    if track_part:
        return exploded_gdf

    return exploded_gdf.drop(columns=["part"])


def _df_to_gdf(df: pd.DataFrame, x_col: str, y_col: str, crs: str) -> gpd.GeoDataFrame:
    """Take a DataFrame and convert it to a GeoDataFrame using spatial columns."""
    return gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=crs
    )


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
    geometry_collection_count = 0
    polygon_count = 0
    multipolygon_count = 0

    for _idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # If it's already Polygon or MultiPolygon, keep as is
        if isinstance(geom, (geometry.Polygon, geometry.MultiPolygon)):
            rows.append(row)
        # If it's a GeometryCollection, extract polygons
        elif isinstance(geom, geometry.GeometryCollection):
            geometry_collection_count += 1
            for sub_geom in geom.geoms:
                if isinstance(sub_geom, geometry.Polygon):
                    polygon_count += 1
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    rows.append(new_row)
                elif isinstance(sub_geom, geometry.MultiPolygon):
                    multipolygon_count += 1
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    rows.append(new_row)

        else:
            raise TypeError(
                "Unexpected geometry type when extracting Polygons from GeometryCollections: "
                f"{geom}"
            )

    LOG.info(
        "Extracted %s Polygons and %s MultiPolygons from %s GeometryCollections",
        polygon_count,
        multipolygon_count,
        geometry_collection_count,
    )

    # Create new GeoDataFrame from expanded rows
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def _nearest_centroids(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Take two GeoDataFrames and merge them on their nearest centroids."""
    # Ensure both GeoDataFrames are in the same projected CRS
    gdf1 = gdf1.to_crs(BNG_CRS)
    gdf2 = gdf2.to_crs(BNG_CRS)

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
    LOG.info("Cleaning infrastructure data...")
    _clean_roads(config, boundary)

    tfn_rail_links = _get_rail_links(boundary, config.infrastructure.rail.tfn_rail_links)
    _clean_rail(config, tfn_rail_links)
    _clean_other(config, boundary, tfn_rail_links)

    LOG.info("Finished cleaning infrastructure data.")


### ROAD


def _clean_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all roads datasets ready for analysis."""
    LOG.info("Cleaning roads data...")
    _clean_os_roads(config, boundary)
    _clean_noham_roads(config, boundary)
    LOG.info("Finished cleaning roads data.")


def _clean_os_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean OS Open Roads dataset, then write to file."""
    os_road = gpd.read_file(
        config.infrastructure.road.os_road,
        mask=boundary,
        columns=["identifier", "roadNumber", "name1", "function"],
    )
    os_road = os_road.drop_duplicates(subset=["identifier", "geometry"])
    os_road = os_road.rename(columns={"name1": "name", "roadNumber": "road_number"})
    os_road[["road_number", "name", "function"]] = os_road[
        ["road_number", "name", "function"]
    ].replace(0, "N/A")
    len_before_filter = len(os_road)
    os_road = os_road[~os_road.geometry.is_empty]
    os_road = os_road[os_road.geometry.notna()]
    tfn_os_road = clip_to_boundary(os_road, boundary)
    filter_removed = len_before_filter - len(tfn_os_road)
    LOG.info(
        "OS roads filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_os_road,
        config.paths.model_input / file_paths.OS_ROAD_MODEL_INPUT_PATH,
    )


def _clean_noham_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean 2023 and 2048 NoHAM network datasets, then write to file."""
    for scenario, noham_entry in config.infrastructure.road.noham.items():
        noham_network = gpd.read_file(
            noham_entry.file_path, mask=boundary, columns=["link_id"]
        )
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
        noham_network_clean = noham_network_clean.drop(columns=["a", "b"])
        noham_network_clean = noham_network_clean[~noham_network_clean.geometry.is_empty]
        noham_network_clean = noham_network_clean[noham_network_clean.geometry.notna()]
        tfn_noham_network = clip_to_boundary(noham_network_clean, boundary)
        filter_removed = len_before_filter - len(tfn_noham_network)
        LOG.info(
            "NoHAM network %s filtered - %s of %s (%.1f percent) rows removed",
            noham_entry.year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        write_to_file(
            tfn_noham_network,
            config.paths.model_input
            / file_paths.NOHAM_NETWORK_MODEL_INPUT_PATH
            / f"tfn_noham_{scenario}.gpkg",
        )


### RAIL


def _clean_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Clean all rail datasets ready for analysis."""
    LOG.info("Cleaning rail data...")
    _clean_passenger_rail(config, rail_links)
    _clean_freight_rail(config, rail_links)
    LOG.info("Finished cleaning rail data.")


def _get_rail_links(
    boundary: gpd.GeoDataFrame, os_rail_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Read and clean OS Rail Network data from the Mutli-Modal Routing Network."""
    tfn_rail_links = gpd.read_file(
        os_rail_path,
        mask=boundary,
        columns=[
            "osid",
            "description",
            "structure",
            "physicallevel",
            "railwayuse",
            "trackrepresentation",
            "operationalstatus",
        ],
    )
    len_before_filter = len(tfn_rail_links)
    tfn_rail_links = tfn_rail_links[tfn_rail_links["operationalstatus"] == "Active"]
    tfn_rail_links = tfn_rail_links.drop(columns="operationalstatus")
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
    LOG.info(
        "OS MMRN Rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    return tfn_rail_links


def _clean_passenger_rail(
    config: model_config.Config, tfn_rail_links: gpd.GeoDataFrame
) -> None:
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
    LOG.info(
        "Passenger rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_pass_rail,
        config.paths.model_input / file_paths.PASSENGER_RAIL_MODEL_INPUT_PATH,
    )


def _clean_freight_rail(config: model_config.Config, tfn_rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to freight rail network, then write to file."""
    len_before_filter = len(tfn_rail_links)
    tfn_freight_rail = tfn_rail_links[
        tfn_rail_links["rail_use"].isin(["Freight And Passenger", "Freight"])
    ]
    filter_removed = len_before_filter - len(tfn_freight_rail)
    LOG.info(
        "Freight rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_freight_rail,
        config.paths.model_input / file_paths.FREIGHT_RAIL_MODEL_INPUT_PATH,
    )


### OTHER


def _clean_other(
    config: model_config.Config, boundary: gpd.GeoDataFrame, rail_links: gpd.GeoDataFrame
) -> None:
    """Clean all other datasets ready for analysis."""
    LOG.info("Cleaning other infrastructure data...")
    _clean_airports(config)
    _clean_bus_stops(config, boundary)
    _clean_petrol_stations(config, boundary)
    _clean_charging_sites(config, boundary)
    _clean_ncn(config, boundary)

    _clean_train_stations(config, boundary)
    _clean_tram_stations(config, boundary)
    _clean_rapid_transport_stations(config, boundary)
    _clean_ferry_terminals(config, boundary)
    _clean_bus_coach_stations(config, boundary)
    _clean_tram_network(config, rail_links)
    _clean_rapid_transport_network(config, rail_links)
    LOG.info("Finished cleaning other infrastructure data.")


def _clean_airports(config: model_config.Config) -> None:
    """Read TfN airports dataset, then write to file in new directory."""
    airports = gpd.read_file(config.infrastructure.other.airports)
    write_to_file(airports, config.paths.model_input / file_paths.AIRPORTS_MODEL_INPUT_PATH)
    LOG.info("Cleaned airports data.")


def _clean_bus_stops(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read, combine and clean regional bus stops datasets, then write to file."""
    bus_stops_ne = pd.read_csv(
        config.infrastructure.other.bus_stops["north_east"]
    )  # North East
    bus_stops_nw = pd.read_csv(
        config.infrastructure.other.bus_stops["north_west"]
    )  # North West
    bus_stops_ys = pd.read_csv(config.infrastructure.other.bus_stops["yorkshire"])  # Yorkshire

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
    LOG.info(
        "Bus stops filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_bus_stops,
        config.paths.model_input / file_paths.BUS_STOPS_MODEL_INPUT_PATH,
    )


def _clean_petrol_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean POI data, filter for petrol stations, and write to file."""
    petrol_stations = gpd.read_file(
        f"zip://{config.infrastructure.other.poi_uk.zip_path}!{config.infrastructure.other.poi_uk.file_path}",
        columns=["id"],
        where="main_category = 'gas_station'",
    )
    len_before_filter = len(petrol_stations)
    petrol_stations = petrol_stations.drop_duplicates()
    tfn_petrol = clip_to_boundary(petrol_stations, boundary)
    filter_removed = len_before_filter - len(tfn_petrol)
    LOG.info(
        "Petrol stations filtered from POIs - %s of %s (%.1f percent)",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_petrol,
        config.paths.model_input / file_paths.PETROL_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_train_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for train stations, then clip to boundary and write to file."""
    os_mmrn_railway_stations = gpd.read_file(
        config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Railway Station%'",
        columns=["nodeid", "os_nodetype"],
    )
    len_before_filter = len(os_mmrn_railway_stations)
    train_stations = os_mmrn_railway_stations[
        os_mmrn_railway_stations["os_nodetype"].isin(MMRN_NODE_TYPES["Train Stations"])
    ]
    train_stations = train_stations.drop(columns=["os_nodetype"])
    train_stations = train_stations.drop_duplicates()
    tfn_train_stations = clip_to_boundary(train_stations, boundary)
    filter_removed = len_before_filter - len(tfn_train_stations)
    LOG.info(
        "Train stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_train_stations,
        config.paths.model_input / file_paths.TRAIN_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_tram_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for tram stations, then clip to boundary and write to file."""
    tram_stations = gpd.read_file(
        config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Tram Station%'",
        columns=["nodeid"],
    )
    len_before_filter = len(tram_stations)
    tram_stations = tram_stations.drop_duplicates()
    tfn_tram_stations = clip_to_boundary(tram_stations, boundary)
    filter_removed = len_before_filter - len(tfn_tram_stations)
    LOG.info(
        "Tram stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_tram_stations, config.paths.model_input / file_paths.TRAM_STATIONS_MODEL_INPUT_PATH
    )


def _clean_rapid_transport_stations(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for rapid transport stations, then clip to boundary and write to file."""
    rapid_transport_stations = gpd.read_file(
        config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Underground System%'",
        columns=["nodeid"],
    )
    len_before_filter = len(rapid_transport_stations)
    rapid_transport_stations = rapid_transport_stations.drop_duplicates()
    tfn_rapid_transport_stations = clip_to_boundary(rapid_transport_stations, boundary)
    filter_removed = len_before_filter - len(tfn_rapid_transport_stations)
    LOG.info(
        "Rapid transport stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_rapid_transport_stations,
        config.paths.model_input / file_paths.RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_ferry_terminals(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for ferry terminals, then clip to boundary and write to file."""
    ferry_terminals = gpd.read_file(
        config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Ferry%'",
        columns=["nodeid"],
    )
    len_before_filter = len(ferry_terminals)
    ferry_terminals = ferry_terminals.drop_duplicates()
    tfn_ferry_terminals = clip_to_boundary(ferry_terminals, boundary)
    filter_removed = len_before_filter - len(tfn_ferry_terminals)
    LOG.info(
        "Ferry terminals filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_ferry_terminals,
        config.paths.model_input / file_paths.FERRY_TERMINALS_MODEL_INPUT_PATH,
    )


def _clean_bus_coach_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for bus and coach stations, then clip to boundary and write to file."""
    bus_coach_stations = gpd.read_file(
        config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Bus Station%' OR os_nodetype LIKE '%Coach Station%'",
        columns=["nodeid"],
    )
    len_before_filter = len(bus_coach_stations)
    bus_coach_stations = bus_coach_stations.drop_duplicates()
    tfn_bus_coach_stations = clip_to_boundary(bus_coach_stations, boundary)
    filter_removed = len_before_filter - len(tfn_bus_coach_stations)
    LOG.info(
        "Bus and coach stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_bus_coach_stations,
        config.paths.model_input / file_paths.BUS_COACH_STATIONS_MODEL_INPUT_PATH,
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
    LOG.info(
        "Tram network links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_tram_links, config.paths.model_input / file_paths.TRAM_NETWORK_MODEL_INPUT_PATH
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
    LOG.info(
        "Rapid transport links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_rapid_transport,
        config.paths.model_input / file_paths.RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH,
    )


def _clean_charging_sites(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean ZapMap charging sites data, then write to file."""
    chg_sites = pd.read_csv(
        config.infrastructure.other.zapmap,
        usecols=["identifier", "name", "speed", "value", "lon", "lat"],
    )
    chg_sites_gdf = gpd.GeoDataFrame(
        chg_sites,
        geometry=[
            geometry.Point(xy) for xy in zip(chg_sites["lon"], chg_sites["lat"], strict=False)
        ],
        crs="EPSG:4326",
    )
    chg_sites_gdf = chg_sites_gdf.drop(columns=["lon", "lat"])
    chg_sites_gdf = chg_sites_gdf.rename(columns={"identifier": "id", "value": "devices"})
    len_before_filter = len(chg_sites_gdf)
    chg_sites_gdf = chg_sites_gdf.drop_duplicates(subset=["geometry"])
    chg_sites_gdf = chg_sites_gdf[~chg_sites_gdf.geometry.is_empty]
    chg_sites_gdf = chg_sites_gdf[chg_sites_gdf.geometry.notna()]
    tfn_chg_sites = clip_to_boundary(chg_sites_gdf, boundary)
    filter_removed = len_before_filter - len(tfn_chg_sites)
    LOG.info(
        "Charging sites filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_chg_sites, config.paths.model_input / file_paths.CHARGING_SITES_MODEL_INPUT_PATH
    )


def _clean_ncn(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean National Cycle Network data, then write to file."""
    ncn = gpd.read_file(
        config.infrastructure.other.ncn_sustrans,
        mask=boundary,
        columns=[
            "SegmentID",
            "Desc_",
            "Greenway",
            "RouteType",
            "RouteNo",
            "LinkNo",
            "Surface",
            "Quality",
            "Lighting",
            "RoadClass",
        ],
    )
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
    LOG.info(
        "NCN filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_ncn,
        config.paths.model_input / file_paths.NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH,
    )


## HAZARDS


def _clean_hazards(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean hazard data ready for analysis."""
    LOG.info("Cleaning hazard data...")
    _clean_extreme_weather(config, boundary)
    _clean_flooding(config, boundary)
    _clean_ground_stability(config, boundary)
    _clean_coastal_erosion(config, boundary)
    LOG.info("Finished cleaning hazard data.")


### EXTREME WEATHER


def _clean_extreme_weather(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all extreme weather datasets ready for analysis."""
    LOG.info("Cleaning extreme weather data...")
    tfn_hazard_grid = _clean_hazard_grid(config, boundary)

    _clean_temp_max(config, tfn_hazard_grid)
    _clean_temp_min(config, tfn_hazard_grid)
    _clean_summer_precip(config, tfn_hazard_grid)
    _clean_winter_precip(config, tfn_hazard_grid)
    _clean_rain_days(config, boundary)
    _clean_drought_index(config, boundary)
    _clean_hot_summer_days(config, tfn_hazard_grid)
    _clean_extreme_summer_days(config, tfn_hazard_grid)
    _clean_frost_days(config, tfn_hazard_grid)
    _clean_icing_days(config, tfn_hazard_grid)
    _clean_wind_speed(config, boundary)
    _clean_wind_driven_rain(config, boundary)
    LOG.info("Finished cleaning extreme weather data.")


def _clean_hazard_grid(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Create and prepare common hazard grid DataFrame for variables on same 12km BNG."""
    hazard_grid = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.max_temp_summer.zip_path}!"
        f"{config.hazards.extreme_weather.max_temp_summer.file_path}",
        mask=boundary,
        columns=[],
    )
    hazard_grid["grid_id"] = range(1, len(hazard_grid) + 1)
    len_before_filter = len(hazard_grid)
    tfn_hazard_grid = clip_to_boundary(hazard_grid, boundary)
    filter_removed = len_before_filter - len(tfn_hazard_grid)
    LOG.info(
        "Extreme weather grid filtered - % s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    tfn_hazard_grid = explode_to_polygons(tfn_hazard_grid, track_part=True)
    write_to_file(
        tfn_hazard_grid,
        config.paths.model_input / file_paths.HAZARD_GRID_MODEL_INPUT_PATH,
    )
    return tfn_hazard_grid


def _clean_temp_max(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean max summer temperature change projections, then write to file."""
    temp_max = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.max_temp_summer.zip_path}!"
        f"{config.hazards.extreme_weather.max_temp_summer.file_path}",
        columns=["tasmax_s_4", "tasmax__22"],
    )
    temp_max["grid_id"] = range(1, len(temp_max) + 1)
    temp_max = temp_max.drop(columns=["geometry"])
    temp_max = temp_max.rename(
        columns={
            "tasmax_s_4": "max_temp_summer_current",
            "tasmax__22": "max_temp_summer_forecast",
        }
    )
    temp_max["max_temp_summer_forecast"] = (
        temp_max["max_temp_summer_current"] + temp_max["max_temp_summer_forecast"]
    )
    len_before_filter = len(temp_max)
    tfn_temp_max = temp_max[temp_max["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_temp_max)
    LOG.info(
        "Summer max temperature change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_temp_max, config.paths.model_input / file_paths.TEMP_MAX_MODEL_INPUT_PATH
    )


def _clean_temp_min(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean min winter temperature change projections, then write to file."""
    temp_min = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.min_temp_winter.zip_path}!"
        f"{config.hazards.extreme_weather.min_temp_winter.file_path}",
        columns=["tasmin_w_4", "tasmin__22"],
    )
    temp_min["grid_id"] = range(1, len(temp_min) + 1)
    temp_min = temp_min.drop(columns=["geometry"])
    temp_min = temp_min.rename(
        columns={
            "tasmin_w_4": "min_temp_winter_current",
            "tasmin__22": "min_temp_winter_forecast",
        }
    )
    temp_min["min_temp_winter_forecast"] = (
        temp_min["min_temp_winter_current"] + temp_min["min_temp_winter_forecast"]
    )
    len_before_filter = len(temp_min)
    tfn_temp_min = temp_min[temp_min["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_temp_min)
    LOG.info(
        "Winter minimum temperature change projections filtered - %s of %s (%.1f percent) "
        "rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_temp_min,
        config.paths.model_input / file_paths.TEMP_MIN_MODEL_INPUT_PATH,
    )


def _clean_summer_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean summer precipitation change projections, then write to file."""
    precip_sum = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.precip_summer.zip_path}!"
        f"{config.hazards.extreme_weather.precip_summer.file_path}",
        columns=["pr_summe_3", "pr_summ_21"],
    )
    precip_sum["grid_id"] = range(1, len(precip_sum) + 1)
    precip_sum = precip_sum.drop(columns=["geometry"])
    precip_sum = precip_sum.rename(
        columns={
            "pr_summe_3": "precip_summer_current",
            "pr_summ_21": "precip_summer_pct_chg_forecast",
        }
    )
    precip_sum["precip_summer_forecast"] = precip_sum["precip_summer_current"] * (
        1 + (precip_sum["precip_summer_pct_chg_forecast"] / 100)
    )
    precip_sum = precip_sum.drop(columns=["precip_summer_pct_chg_forecast"])
    len_before_filter = len(precip_sum)
    tfn_precip_sum = precip_sum[precip_sum["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_precip_sum)
    LOG.info(
        "Summer precipitation change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_precip_sum, config.paths.model_input / file_paths.SUMMER_PRECIP_MODEL_INPUT_PATH
    )


def _clean_winter_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean winter precipitation change projections, then write to file."""
    precip_win = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.precip_winter.zip_path}!"
        f"{config.hazards.extreme_weather.precip_winter.file_path}",
        columns=["pr_winte_3", "pr_wint_21"],
    )
    precip_win["grid_id"] = range(1, len(precip_win) + 1)
    precip_win = precip_win.drop(columns=["geometry"])
    precip_win = precip_win.rename(
        columns={
            "pr_winte_3": "precip_winter_current",
            "pr_wint_21": "precip_winter_pct_chg_forecast",
        },
    )
    precip_win["precip_winter_forecast"] = precip_win["precip_winter_current"] * (
        1 + (precip_win["precip_winter_pct_chg_forecast"] / 100)
    )
    precip_win = precip_win.drop(columns=["precip_winter_pct_chg_forecast"])
    len_before_filter = len(precip_win)
    tfn_precip_win = precip_win[precip_win["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_precip_win)
    LOG.info(
        "Winter precipitation change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_precip_win, config.paths.model_input / file_paths.WINTER_PRECIP_MODEL_INPUT_PATH
    )


def _clean_rain_days(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean 10mm rain days observations, then write to file."""
    rain_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.rain_days.zip_path}!"
        f"{config.hazards.extreme_weather.rain_days.file_path}",
        mask=boundary,
    )
    len_before_filter = len(rain_days)
    tfn_rain_days = clip_to_boundary(rain_days, boundary)
    tfn_rain_days = tfn_rain_days.rename(columns={"Rain10mmDa": "10mm_rain_days_current"})
    filter_removed = len_before_filter - len(tfn_rain_days)
    LOG.info(
        "10mm rain days 1991-2020 filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    tfn_rain_days = explode_to_polygons(tfn_rain_days)
    write_to_file(
        tfn_rain_days, config.paths.model_input / file_paths.RAIN_DAYS_MODEL_INPUT_PATH
    )


def _clean_drought_index(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean drought severity index data, then write to file."""
    drought_index = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.drought_index.zip_path}!"
        f"{config.hazards.extreme_weather.drought_index.file_path}",
        mask=boundary,
        columns=["DSI12_ba_4", "DSI12_40_m"],
    )
    drought_index = drought_index.rename(
        columns={
            "DSI12_ba_4": "drought_severity_index_current",
            "DSI12_40_m": "drought_severity_index_forecast",
        }
    )
    len_before_filter = len(drought_index)
    tfn_drought = clip_to_boundary(drought_index, boundary)
    filter_removed = len_before_filter - len(tfn_drought)
    LOG.info(
        "Drought severity index filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    tfn_drought = explode_to_polygons(tfn_drought)
    write_to_file(
        tfn_drought, config.paths.model_input / file_paths.DROUGHT_INDEX_MODEL_INPUT_PATH
    )


def _clean_hot_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean hot summer days projections, then write to file."""
    hot_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.hot_days.zip_path}!"
        f"{config.hazards.extreme_weather.hot_days.file_path}",
        columns=["HSD_base_4", "HSD_40_med"],
    )
    hot_days["grid_id"] = range(1, len(hot_days) + 1)
    hot_days = hot_days.drop(columns=["geometry"])
    hot_days = hot_days.rename(
        columns={
            "HSD_base_4": "hot_summer_days_current",
            "HSD_40_med": "hot_summer_days_forecast",
        }
    )
    len_before_filter = len(hot_days)
    tfn_hot_days = hot_days[hot_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_hot_days)
    LOG.info(
        "Hot summer days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_hot_days, config.paths.model_input / file_paths.HOT_SUMMER_DAYS_MODEL_INPUT_PATH
    )


def _clean_extreme_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean extreme summer days projections, then write to file."""
    extr_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.extreme_summer_days.zip_path}!"
        f"{config.hazards.extreme_weather.extreme_summer_days.file_path}",
        columns=["ESD_base_4", "ESD_40_med"],
    )
    extr_days["grid_id"] = range(1, len(extr_days) + 1)
    extr_days = extr_days.drop(columns=["geometry"])
    extr_days = extr_days.rename(
        columns={
            "ESD_base_4": "extreme_summer_days_current",
            "ESD_40_med": "extreme_summer_days_forecast",
        }
    )
    len_before_filter = len(extr_days)
    tfn_extr_days = extr_days[extr_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_extr_days)
    LOG.info(
        "Extreme summer days projections - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_extr_days,
        config.paths.model_input / file_paths.EXTREME_SUMMER_DAYS_MODEL_INPUT_PATH,
    )


def _clean_frost_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean frost days projections, then write to file."""
    frost_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.frost_days.zip_path}!"
        f"{config.hazards.extreme_weather.frost_days.file_path}",
        columns=["FrostDay_3", "FrostDa_18"],
    )
    frost_days["grid_id"] = range(1, len(frost_days) + 1)
    frost_days = frost_days.drop(columns=["geometry"])
    frost_days = frost_days.rename(
        columns={"FrostDay_3": "frost_days_current", "FrostDa_18": "frost_days_forecast"}
    )
    len_before_filter = len(frost_days)
    tfn_frost_days = frost_days[frost_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_frost_days)
    LOG.info(
        "Frost days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_frost_days, config.paths.model_input / file_paths.FROST_DAYS_MODEL_INPUT_PATH
    )


def _clean_icing_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean icing days projections, then write to file."""
    ice_days = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.icing_days.zip_path}!"
        f"{config.hazards.extreme_weather.icing_days.file_path}",
        columns=["IcingDay_3", "IcingDa_18"],
    )
    ice_days["grid_id"] = range(1, len(ice_days) + 1)
    ice_days = ice_days.drop(columns=["geometry"])
    ice_days = ice_days.rename(
        columns={"IcingDay_3": "icing_days_current", "IcingDa_18": "icing_days_forecast"}
    )
    len_before_filter = len(ice_days)
    tfn_ice_days = ice_days[ice_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(tfn_ice_days)
    LOG.info(
        "Icing days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tfn_ice_days, config.paths.model_input / file_paths.ICING_DAYS_MODEL_INPUT_PATH
    )


def _clean_wind_speed(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read wind speed projections, calculate metrics, clean, then write to file."""
    windspd_current_agg, len_before_filter_current = _read_wind_speed_reduce(
        config.hazards.extreme_weather.wind_spd_current, "current"
    )
    windspd_forecast_agg, len_before_filter_forecast = _read_wind_speed_reduce(
        config.hazards.extreme_weather.wind_spd_forecast, "forecast"
    )

    metric_cols = [
        "wind_speed_99th_percentile_current",
        "avg_exceedance_days_current",
        "wind_speed_99th_percentile_forecast",
        "avg_exceedance_days_forecast",
    ]

    len_before_filter = len_before_filter_current + len_before_filter_forecast

    windspd_combined = windspd_current_agg.merge(
        windspd_forecast_agg,
        on=["projection_y_coordinate", "projection_x_coordinate", "latitude", "longitude"],
        how="outer",
    ).fillna({col: 0 for col in metric_cols})

    windspd_combined["geometry"] = [
        _convert_point_to_grid(x, y, 2500)
        for x, y in zip(
            windspd_combined["projection_x_coordinate"],
            windspd_combined["projection_y_coordinate"],
            strict=False,
        )
    ]
    windspd_combined = gpd.GeoDataFrame(windspd_combined, geometry="geometry", crs=BNG_CRS)
    windspd_combined = windspd_combined[[*metric_cols, "geometry"]]
    tfn_windspd = clip_to_boundary(windspd_combined, boundary)
    filter_removed = len_before_filter - len(tfn_windspd)
    LOG.info(
        "Wind speed projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    tfn_windspd = explode_to_polygons(tfn_windspd)
    write_to_file(
        tfn_windspd, config.paths.model_input / file_paths.WIND_SPEED_MODEL_INPUT_PATH
    )


def _read_wind_speed_reduce(xr_path: pathlib.Path, scenario: str) -> tuple[pd.DataFrame, int]:
    """Read wind speed projections, calculate metrics and return a merged dataframe."""
    windspd_dataset = xr.open_dataset(xr_path)
    var = "wsgmax10m"

    # Compute exceedance and percentile aggregations
    exceedance_dataset = _calculate_windspd_exceedance(
        windspd_dataset, _WIND_SPEED_EXCEEDANCE_THRESHOLD, var, scenario
    )
    percentile_dataset = _calculate_windspd_percentile(
        windspd_dataset, _WIND_SPEED_PERCENTILE, var, scenario=scenario
    )

    windspd_aggregations = xr.merge(
        [exceedance_dataset, percentile_dataset], compat="override"
    )
    windspd_dataframe = windspd_aggregations.to_dataframe().reset_index()
    len_before_filter = windspd_dataset.sizes["time"]

    return windspd_dataframe, len_before_filter


def _calculate_windspd_exceedance(
    windspd_data: xr.Dataset, threshold: int, variable: str, scenario: str
) -> xr.Dataset:
    """Compute average exceedance days per geometry for values above a given threshold."""
    # boolean exceedance mask
    exceedance = windspd_data[variable] > threshold

    # count exceedance days per year, per grid cell
    exceedance_per_year = exceedance.groupby("year").sum(dim="time")

    # average exceedance across years
    avg_exc = exceedance_per_year.mean(dim="year")

    # store in a Dataset with a named variable
    return avg_exc.to_dataset(name=f"avg_exceedance_days_{scenario}")


def _calculate_windspd_percentile(
    windspd_data: xr.Dataset, quantile: float, variable: str, scenario: str
) -> xr.Dataset:
    """Calculate the wind speed percentiles per geometry for a given variable."""
    pct = windspd_data[variable].quantile(quantile, dim="time")
    return pct.to_dataset(name=f"wind_speed_{int(quantile * 100)}th_percentile_{scenario}")


def _clean_wind_driven_rain(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean wind driven rain index data, then write to file."""
    wind_driven_rain = gpd.read_file(
        f"zip://{config.hazards.extreme_weather.wdr_index.zip_path}!"
        f"{config.hazards.extreme_weather.wdr_index.file_path}",
        mask=boundary,
        columns=["WDR_base_1", "WDR_40_Med", "x_coord", "y_coord"],
    )
    len_before_filter = len(wind_driven_rain)
    # Aggregate by wind direction to calculate mean wind speed
    wind_driven_rain_agg = (
        wind_driven_rain.groupby(["x_coord", "y_coord"])
        .agg({"WDR_base_1": "mean", "WDR_40_Med": "mean", "geometry": "first"})
        .reset_index()
    )

    wind_driven_rain_agg = wind_driven_rain_agg.drop(columns=["x_coord", "y_coord"])
    wind_driven_rain_agg = wind_driven_rain_agg.rename(
        columns={
            "WDR_base_1": "wind_driven_rain_index_current",
            "WDR_40_Med": "wind_driven_rain_index_forecast",
        }
    )
    wind_driven_rain_agg = gpd.GeoDataFrame(
        wind_driven_rain_agg, geometry="geometry", crs="EPSG:3857"
    )
    tfn_wind_driven_rain = clip_to_boundary(wind_driven_rain_agg, boundary)
    filter_removed = len_before_filter - len(tfn_wind_driven_rain)
    LOG.info(
        "Wind driven rain index filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    tfn_wind_driven_rain = explode_to_polygons(tfn_wind_driven_rain)
    write_to_file(
        tfn_wind_driven_rain,
        config.paths.model_input / file_paths.WIND_DRIVEN_RAIN_MODEL_INPUT_PATH,
    )


### FLOODING


def _clean_flooding(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean flooding data ready for analysis."""
    LOG.info("Cleaning flooding data...")

    if config.switches.flood_zip_extract:
        _extract_flood_data(config, _FLOOD_CODE_NUMBER_MAP)

    LOG.info("Cleaning climate change river and sea flooding data...")
    _clean_flood(
        config,
        file_name="RoFRS",
        flood_type="RoFRS",
        version="v202501",
        boundary=boundary,
        out_path=file_paths.FLOOD_RIVERS_SEA_CLIMATE_CHANGE_MODEL_INPUT_PATH,
        climate_change_switch=True,
        code_number_map=_FLOOD_CODE_NUMBER_MAP,
        rename_risk_col="rivers_sea_flood_risk_forecast",
    )
    LOG.info("Finished cleaning climate change river and sea flooding data.")

    LOG.info("Cleaning river and sea flooding data...")
    _clean_flood(
        config,
        file_name="RoFRS",
        flood_type="RoFRS",
        version="v202501",
        boundary=boundary,
        out_path=file_paths.FLOOD_RIVERS_SEA_MODEL_INPUT_PATH,
        climate_change_switch=False,
        code_number_map=_FLOOD_CODE_NUMBER_MAP,
        rename_risk_col="rivers_sea_flood_risk_current",
    )
    LOG.info("Finished cleaning river and sea flooding data.")

    LOG.info("Cleaning climate change surface water flooding data...")
    _clean_flood(
        config,
        file_name="RoFSW CC",
        flood_type="RoFSW",
        version="v202509",
        boundary=boundary,
        out_path=file_paths.FLOOD_SURFACE_WATER_CLIMATE_CHANGE_MODEL_INPUT_PATH,
        climate_change_switch=True,
        code_number_map=_FLOOD_CODE_NUMBER_MAP,
        rename_risk_col="surface_water_flood_risk_forecast",
    )
    LOG.info("Finished cleaning climate change surface water flooding data.")

    LOG.info("Cleaning surface water flooding data...")
    _clean_flood(
        config,
        file_name="RoFSW",
        flood_type="RoFSW",
        version="v202509",
        boundary=boundary,
        out_path=file_paths.FLOOD_SURFACE_WATER_MODEL_INPUT_PATH,
        climate_change_switch=False,
        code_number_map=_FLOOD_CODE_NUMBER_MAP,
        rename_risk_col="surface_water_flood_risk_current",
    )
    LOG.info("Finished cleaning surface water flooding data.")
    LOG.info("Finished cleaning flooding data.")


def _extract_flood_gdb_file(
    config: model_config.Config,
    *,
    code: str,
    number: str,
    flood_data: str,
    version: str,
    cc: bool,
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
    *,
    code: str,
    number: str,
    file_name: str,
    flood_type: str,
    version: str,
    climate_change_switch: bool,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Read first layer of flood gdb file."""
    base_path = config.hazards.flooding.flood_path / file_name / code

    if climate_change_switch:
        gdb_path = base_path / f"{flood_type}_Climate_Change_01_{code}{number}_{version}.gdb"
    else:
        gdb_path = base_path / f"{flood_type}_{code}{number}_{version}.gdb"

    # Check if GDB folder exists
    if not gdb_path.exists():
        raise FileNotFoundError(f"GBD folder not found: {gdb_path}")

    layers = fiona.listlayers(gdb_path)
    if not layers:
        raise ValueError(f"No layers found in GDB: {gdb_path}")

    return gpd.read_file(
        gdb_path,
        layer=layers[0],
        mask=boundary,
        columns=["Risk_band"],
        engine="pyogrio",
        use_arrow=True,
    )


def _extract_flood_data(
    config: model_config.Config, code_number_map: dict[str, list[str]]
) -> None:
    """Extract geodatabase files from raw RoFRS and RoFSW flood data."""
    for code, num_list in code_number_map.items():
        for number in num_list:
            # Forecast (Climate Change) data
            _extract_flood_gdb_file(
                config,
                code=code,
                number=number,
                flood_data="RoFRS",
                version="v202501",
                cc=True,
            )
            _extract_flood_gdb_file(
                config,
                code=code,
                number=number,
                flood_data="RoFSW",
                version="v202509",
                cc=True,
            )

            # Current data
            _extract_flood_gdb_file(
                config,
                code=code,
                number=number,
                flood_data="RoFRS",
                version="v202501",
                cc=False,
            )
            _extract_flood_gdb_file(
                config,
                code=code,
                number=number,
                flood_data="RoFSW",
                version="v202509",
                cc=False,
            )


def _process_flood_layer(
    config: model_config.Config,
    *,
    code: str,
    number: str,
    file_name: str,
    flood_type: str,
    version: str,
    boundary: gpd.GeoDataFrame,
    climate_change_switch: bool,
    rename_risk_col: str,
) -> gpd.GeoDataFrame | None:
    """Read, clip, clean, and prepare a single flood layer. Helper function for clean_flood."""
    LOG.info("Processing flood data: %s%s", code, number)

    flood_data = _read_flood_gdb(
        config,
        code=code,
        number=number,
        file_name=file_name,
        flood_type=flood_type,
        version=version,
        climate_change_switch=climate_change_switch,
        boundary=boundary,
    )  # Read file
    len_before_filter = len(flood_data)
    tfn_flood_data = clip_to_boundary(flood_data, boundary)
    if tfn_flood_data.empty:
        LOG.info("%s%s layer empty. Continuing.", code, number)
        return None
    filter_removed = len_before_filter - len(tfn_flood_data)

    LOG.info(
        "%s %s %s filtered - %s of %s (%.1f percent) rows removed",
        flood_type,
        code,
        number,
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    tfn_flood_data = _extract_poly_from_geomcollection(tfn_flood_data)
    tfn_flood_data = tfn_flood_data[["Risk_band", "geometry"]]
    return tfn_flood_data.rename(columns={"Risk_band": rename_risk_col})


def _clean_flood(
    config: model_config.Config,
    *,
    file_name: str,
    flood_type: str,
    version: str,
    boundary: gpd.GeoDataFrame,
    out_path: pathlib.Path,
    climate_change_switch: bool,
    code_number_map: dict[str, list[str]],
    rename_risk_col: str,
) -> None:
    """Read and clean flood data, then write to file."""
    first_write = True
    for code, num_list in code_number_map.items():
        for number in num_list:
            tfn_flood_data = _process_flood_layer(
                config,
                code=code,
                number=number,
                file_name=file_name,
                flood_type=flood_type,
                version=version,
                boundary=boundary,
                climate_change_switch=climate_change_switch,
                rename_risk_col=rename_risk_col,
            )
            if first_write:
                write_to_file(tfn_flood_data, config.paths.model_input / out_path, mode="w")
                first_write = False
            else:
                write_to_file(tfn_flood_data, config.paths.model_input / out_path, mode="a")

            del tfn_flood_data
            gc.collect()


### GROUND STABILITY


def _clean_ground_stability(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean ground stability data ready for analysis."""
    LOG.info("Cleaning ground stability data...")
    _clean_geosure(config, boundary)
    _clean_geoclimate(config, boundary)
    LOG.info("Finished cleaning ground stability data.")


def _clean_geosure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean GeoSureHexGrids data, merge by nearest centroids, then write to file."""
    geosure_layers = {
        "collapsible_deposits": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.collapsible_deposits}",
            mask=boundary,
            columns=["CLASS"],
        ),
        "compressible_ground": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.compressible_ground}",
            mask=boundary,
            columns=["CLASS"],
        ),
        "landslides": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.landslides}",
            mask=boundary,
            columns=["CLASS"],
        ),
        "running_sand": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.running_sand}",
            mask=boundary,
            columns=["CLASS"],
        ),
        "shrink_swell": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.shrink_swell}",
            mask=boundary,
            columns=["CLASS"],
        ),
        "soluble_rocks": gpd.read_file(
            f"zip://{config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.soluble_rocks}",
            mask=boundary,
            columns=["CLASS"],
        ),
    }

    tfn_geosure_layers = {}
    for code, geosure_data in geosure_layers.items():
        len_before_filter = len(geosure_data)
        geosure_data_clean = geosure_data.rename(columns={"CLASS": f"{code}_risk"})
        tfn_geosure_layers[code] = clip_to_boundary(geosure_data_clean, boundary)
        filter_removed = len_before_filter - len(tfn_geosure_layers[code])
        LOG.info(
            "GeoSure %s filtered - %s of %s (%.1f percent) rows removed",
            code.replace("_", " ").title(),
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        tfn_geosure_layers[code] = explode_to_polygons(tfn_geosure_layers[code])

    # Merge layers based on nearest centroids
    base_code = next(iter(geosure_layers.keys()))
    tfn_geosure = tfn_geosure_layers[base_code][[f"{base_code}_risk", "geometry"]].copy()
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

    filter_removed = len_before_filter - len(tfn_geosure)
    LOG.info(
        "GeoSure layers filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

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
        config.paths.model_input / file_paths.GEOSURE_MODEL_INPUT_PATH,
    )


def _clean_geoclimate(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean GeoClimate Shrink-Swell data, then write to file."""
    for year, filepath in config.hazards.ground_stability.geo_shrink_swell.items():
        geoclimate_data = gpd.read_file(filepath, mask=boundary, columns=["CLASS"])
        geoclimate_data = geoclimate_data.rename(
            columns={"CLASS": "shrink_swell_geoclimate_risk"}
        )
        len_before_filter = len(geoclimate_data)
        tfn_geoclimate_data = clip_to_boundary(geoclimate_data, boundary)
        filter_removed = len_before_filter - len(tfn_geoclimate_data)
        LOG.info(
            "GeoClimate %s filtered - %s of %s (%.1f percent) rows removed",
            year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        tfn_geoclimate_data = explode_to_polygons(tfn_geoclimate_data)

        write_to_file(
            tfn_geoclimate_data,
            config.paths.model_input
            / file_paths.GEOCLIMATE_SHRINK_SWELL_MODEL_INPUT_PATH
            / f"tfn_bgs_ss_{year}.gpkg",
        )


### COASTAL EROSION


def _clean_coastal_erosion(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean coastal erosion data ready for analysis."""
    LOG.info("Cleaning coastal erosion data...")
    _clean_ground_instability_zones(config, boundary)
    _clean_ncerm(config, boundary)
    LOG.info("Finished cleaning coastal erosion data.")


def _clean_ground_instability_zones(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> None:
    """Clean Ground Instability Zones data from NCERM, then write to file."""
    ncerm_giz = gpd.read_file(
        f"zip://{config.hazards.coastal_erosion.zip_path}!{config.hazards.coastal_erosion.giz}",
        mask=boundary,
        columns=["smp_no"],
    )
    len_before_filter = len(ncerm_giz)
    tfn_ncerm_giz = clip_to_boundary(ncerm_giz, boundary)
    filter_removed = len_before_filter - len(tfn_ncerm_giz)
    LOG.info(
        "Ground Instability Zones filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    tfn_ncerm_giz = explode_to_polygons(tfn_ncerm_giz)

    write_to_file(
        tfn_ncerm_giz,
        config.paths.model_input / file_paths.GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH,
    )


def _clean_ncerm(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean erosion data from NCERM for 2055, and 2105, then write to file."""
    for year in ["2055", "2105"]:
        erosion_data = gpd.read_file(
            f"zip://{config.hazards.coastal_erosion.zip_path}!/"
            + config.hazards.coastal_erosion.smp[year],
            mask=boundary,
            columns=["smp_name"],
        )
        len_before_filter = len(erosion_data)
        tfn_erosion_data = clip_to_boundary(erosion_data, boundary)
        filter_removed = len_before_filter - len(tfn_erosion_data)
        LOG.info(
            "NCERM %s filtered - %s of %s (%.1f percent) rows removed",
            year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        tfn_erosion_data = explode_to_polygons(tfn_erosion_data)

        write_to_file(
            tfn_erosion_data,
            config.paths.model_input
            / file_paths.NCERM_MODEL_INPUT_PATH
            / f"tfn_ncerm_smp_{year}_70CC.gpkg",
        )


## IMPACT


def _clean_impact(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean impact datasets ready for analysis."""
    LOG.info("Cleaning impact data...")
    _clean_freight_demand(config, boundary)
    _clean_noham_flows(config)
    LOG.info("Finished cleaning impact data.")


### FREIGHT


def _clean_freight_demand(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean freight demand data ready for analysis."""
    tfn_freight_network_demand = _read_freight_demand(config.impact.freight_demand, boundary)

    tfn_os_freight_network_demand = _map_freight_networks(
        tfn_freight_network_demand,
        config.paths.model_input / file_paths.FREIGHT_RAIL_MODEL_INPUT_PATH,
    )

    write_to_file(
        tfn_os_freight_network_demand,
        config.paths.model_input / file_paths.FREIGHT_DEMAND_MODEL_INPUT_PATH,
    )


def _read_freight_demand(path: pathlib.Path, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Read and clean freight demand data, and return as GeoDataFrame."""
    freight_network_demand = gpd.read_file(
        path, mask=boundary, columns=["dij_id", "2022_23_total", "2050_51 sc2_total"]
    )
    freight_network_demand = freight_network_demand.rename(
        columns={"2022_23_total": "demand_current", "2050_51 sc2_total": "demand_forecast"}
    )
    len_before_filter = len(freight_network_demand)
    tfn_freight_network_demand = clip_to_boundary(freight_network_demand, boundary)
    filter_removed = len_before_filter - len(tfn_freight_network_demand)
    LOG.info(
        "Freight demand filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    return tfn_freight_network_demand


def _map_freight_networks(
    tfn_freight_network_demand: gpd.GeoDataFrame, os_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Map freight demand data onto OS network, then clean and return."""
    tfn_os_freight_rail = gpd.read_file(os_path)
    tfn_os_freight_rail = tfn_os_freight_rail.to_crs(tfn_freight_network_demand.crs)
    len_before_mapping = len(tfn_freight_network_demand)
    tfn_os_freight_network_demand = gpd.sjoin_nearest(
        tfn_os_freight_rail,
        tfn_freight_network_demand,
        how="left",
        max_distance=_FREIGHT_DEMAND_NETWORK_MAP_MAX_DISTANCE,
        distance_col="distance",
    )
    len_after_mapping = len(tfn_os_freight_network_demand)
    LOG.info(
        "Freight demand mapped to OS network - %s segments mapped onto %s OS network segments",
        len_before_mapping,
        len_after_mapping,
    )

    tfn_os_freight_network_demand[["demand_current", "demand_forecast"]] = (
        tfn_os_freight_network_demand[["demand_current", "demand_forecast"]].fillna(0)
    )
    return tfn_os_freight_network_demand.drop(columns=["index_right"])


### NoHAM


def _clean_noham_flows(config: model_config.Config) -> None:
    """Clean NoHAM flows data, aggregate link flows, merge with network, then write to file."""
    for scenario, _ in config.infrastructure.road.noham.items():
        tfn_noham_flows = _aggregate_link_flows_year(config, scenario)
        tfn_noham_net_flows = _merge_noham_flow_network(
            tfn_noham_flows,
            config.paths.model_input
            / file_paths.NOHAM_NETWORK_MODEL_INPUT_PATH
            / f"tfn_noham_{scenario}.gpkg",
        )
        write_to_file(
            tfn_noham_net_flows,
            config.paths.model_input
            / file_paths.NOHAM_FLOWS_MODEL_INPUT_PATH
            / f"tfn_noham_net_flows_{scenario}.gpkg",
        )


def _read_noham_h5(
    *,
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    year: str,
    time_period: str,
    user_class: str,
    noham_path: pathlib.Path,
    output_path: pathlib.Path | None,
    extract: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and clean NoHAM h5 files and extract the link, routes, and od's DataFrames."""
    if output_path is None:
        raise ValueError("NoHAM output path must be provided.")
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

    noham_demand_path = (
        output_path
        / "input h5s"
        / year
        / f"NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5"
    )

    if (year, time_period) not in route_links_store:
        noham_routes = pd.read_hdf(noham_demand_path, key="/data/Route")
        noham_routes = noham_routes.reset_index()[["route", "link_id"]]
        noham_links = pd.read_hdf(noham_demand_path, key="/data/link")
        noham_links = noham_links[["a", "b"]]
        route_links_store[(year, time_period)] = (noham_routes, noham_links)
    else:
        noham_routes, noham_links = route_links_store[(year, time_period)]

    noham_ods = pd.read_hdf(noham_demand_path, key="/data/OD")
    noham_ods = noham_ods.reset_index()[["route", "abs_demand"]]

    return noham_ods, noham_routes, noham_links


def _aggregate_link_flows(
    ods: pd.DataFrame, routes: pd.DataFrame, links: pd.DataFrame
) -> pd.DataFrame:
    """Take NoHAM od's, routes, and link to create aggregated link flows DataFrame."""
    # Merge OD demand onto routes
    od_routes = routes.merge(ods[["route", "abs_demand"]], on="route", how="inner")

    # Aggregate demand per link_id
    link_demand = od_routes.groupby("link_id")["abs_demand"].sum().reset_index()

    return link_demand.merge(links[["a", "b"]], left_on="link_id", right_index=True)


def _process_single_noham_layer(
    config: model_config.Config,
    *,
    year: str,
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    time_period: str,
    user_class: str,
) -> pd.DataFrame:
    LOG.info("Processing NoHAM demand: %s %s %s", year, time_period, user_class)

    noham_ods, noham_routes, noham_links = _read_noham_h5(
        route_links_store=route_links_store,
        year=year,
        time_period=time_period,
        user_class=user_class,
        noham_path=config.impact.noham_demand.zip_path,
        output_path=config.impact.noham_demand.output_path,
        extract=config.switches.noham_zip_extract,
    )
    link_demand = _aggregate_link_flows(
        noham_ods, noham_routes, noham_links
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
    LOG.info(
        "%s ODs, %s Routes and %s Links aggregated to %s link flows",
        len(noham_ods),
        len(noham_routes),
        len(noham_links),
        len(link_demand),
    )

    return link_demand


def _aggregate_link_flows_year(
    config: model_config.Config, scenario: str
) -> dict[str, pd.DataFrame]:
    """Aggregate link flows for each year, time period, and user class."""
    year = config.infrastructure.road.noham[scenario].year

    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    ts_dfs = []
    for time_period in _NOHAM_TIME_PERIODS:
        uc_dfs = []
        for user_class in _NOHAM_USER_CLASSES:
            uc_dfs.append(
                _process_single_noham_layer(
                    config,
                    year=year,
                    route_links_store=route_links_store,
                    time_period=time_period,
                    user_class=user_class,
                )
            )

        # Merge all user class dataframes
        combined_uc_df = uc_dfs[0]
        for df_uc in uc_dfs[1:]:
            combined_uc_df = combined_uc_df.merge(df_uc, on="link_id", how="outer")

        # Compute total demand for all vehicles for each time period
        combined_uc_df[f"all_vehs_{time_period}"] = combined_uc_df[
            [f"{uc}_{time_period}" for uc in _NOHAM_USER_CLASSES]
        ].sum(axis=1)

        # Store result
        ts_dfs.append(combined_uc_df)

    # Merge all time period dataframes
    combined_ts_df = ts_dfs[0]
    for df_ts in ts_dfs[1:]:
        combined_ts_df = combined_ts_df.merge(df_ts, on="link_id", how="outer")

    # Compute totals for each user class across all time periods
    for uc in _NOHAM_USER_CLASSES:
        combined_ts_df[f"{uc}_total"] = combined_ts_df[
            [f"{uc}_{tp}" for tp in _NOHAM_TIME_PERIODS]
        ].sum(axis=1)

    # Compute total of each user class across all time periods
    combined_ts_df["all_vehs_total"] = combined_ts_df[
        [f"all_vehs_{tp}" for tp in _NOHAM_TIME_PERIODS]
    ].sum(axis=1)

    return combined_ts_df


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
