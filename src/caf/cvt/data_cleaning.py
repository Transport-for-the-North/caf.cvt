"""Cleans raw input data to prepare it for input into the model."""

### LOAD LIBRARIES
import gc
import logging
import os
import pathlib

import fiona
import geopandas as gpd
import osbng
import pandas as pd
import py7zr
import xarray as xr
from shapely import geometry

from caf.cvt import file_paths, model_config
from caf.cvt.definitions import (
    DroughtCols,
    ExtremeColdCols,
    ExtremeHeatCols,
    GroundStabilityRiskCols,
    NoHAMTimePeriods,
    NoHAMUserClasses,
    Scenarios,
    StormCols,
)

LOG = logging.getLogger(__name__)

### ENVIRONMENT VARIABLES ###


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


_WIND_SPEED_EXCEEDANCE_THRESHOLD = 20
_WIND_SPEED_PERCENTILE = 0.99


### GENERAL FUNCTIONS


def clip_to_boundary(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to a specified spatial boundary.

    This function re-projects the boundary to match the CRS of the input GeoDataFrame and then
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


def get_boundary(config: model_config.Config) -> gpd.GeoDataFrame:
    """Get the boundary to use for clipping and filtering datasets based on the config."""
    if config.other_input.boundary_path is not None and config.other_input.boundary_path != "":
        LOG.info("Using specific boundary file: %s", config.other_input.boundary_path)
        return gpd.read_file(config.paths.raw_input / config.other_input.boundary_path)

    if config.parameters.stb is not None:
        LOG.info("Using boundary for STB: %s", config.parameters.stb)
        stb_boundaries = gpd.read_file(config.paths.raw_input / config.other_input.stb_path)
        stb_boundary = stb_boundaries[stb_boundaries["stb_name"] == config.parameters.stb]
        stb_boundary = stb_boundary[["stb_name", "geometry"]]
        if stb_boundary.empty:
            raise ValueError(f"No boundary found for STB: '{config.parameters.stb}'")
        return stb_boundary

    if config.parameters.ca is not None:
        LOG.info("Using boundary for CA: %s", config.parameters.ca)
        ca_boundaries = gpd.read_file(config.paths.raw_input / config.other_input.ca_path)
        ca_boundary = ca_boundaries[ca_boundaries["CAUTH25NM"] == config.parameters.ca]
        ca_boundary = ca_boundary[["CAUTH25NM", "geometry"]]
        if ca_boundary.empty:
            raise ValueError(f"No boundary found for CA: '{config.parameters.ca}'")
        return ca_boundary

    raise ValueError(
        "No valid boundary specified in config. "
        "You must provide one of: "
        "`boundary_path`, `stb`, or `ca`."
    )


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


def _get_bng_codes(boundary: gpd.GeoDataFrame) -> list[str]:
    """Clip the 100km BNG to the boundary."""
    bng_100km = gpd.GeoDataFrame.from_features(osbng.grids.bng_grid_100km, crs=BNG_CRS)
    boundary_bng = clip_to_boundary(bng_100km, boundary)
    return list(boundary_bng["bng_ref"])


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate geometries in a GeoDataFrame, removing invalid ones."""
    if gdf.geometry.is_empty.sum() > 0:
        LOG.warning("Found %s empty geometries, removing them.", gdf.geometry.is_empty.sum())
        gdf = gdf[~gdf.geometry.is_empty]
    if gdf.geometry.notna().sum() < len(gdf):
        LOG.warning(
            "Found %s invalid geometries, removing them.",
            len(gdf) - gdf.geometry.notna().sum(),
        )
        gdf = gdf[gdf.geometry.notna()]
    return gdf


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
    boundary = get_boundary(config)

    _clean_infrastructure(config, boundary)
    _clean_hazards(config, boundary)
    _clean_impact(config, boundary)


## INFRASTRUCTURE


def _clean_infrastructure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all infrastructure datasets ready for analysis."""
    LOG.info("Cleaning infrastructure data...")
    _clean_roads(config, boundary)

    rail_links = _get_rail_links(
        boundary, config.paths.raw_input / config.infrastructure.rail.rail_links
    )
    _clean_rail(config, rail_links)
    _clean_other(config, boundary, rail_links)

    LOG.info("Finished cleaning infrastructure data.")


### ROAD


def _clean_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all roads datasets ready for analysis."""
    LOG.info("Cleaning roads data...")
    if config.switches.all_roads:
        LOG.info("Cleaning all roads data...")
        _clean_os_roads(config, boundary)
    if config.switches.noham_roads:
        LOG.info("Cleaning NoHAM roads data...")
        _clean_noham_roads(config, boundary)
    LOG.info("Finished cleaning roads data.")


def _clean_os_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean OS Open Roads dataset, then write to file."""
    os_road = gpd.read_file(
        f"zip://{config.paths.raw_input / config.infrastructure.road.os_road.zip_path}!"
        f"{config.infrastructure.road.os_road.file_path.as_posix()}",
        mask=boundary,
        columns=[
            "id",
            "road_classification",
            "road_function",
            "form_of_way",
            "road_classification_number",
            "name_1",
            "road_structure",
            "primary_route",
            "trunk_road",
            "geometry",
        ],
        layer="road_link",
    )
    len_before_filter = len(os_road)
    os_road = os_road.drop_duplicates(subset=["id", "geometry"])
    os_road = os_road.rename(columns={"name_1": "name"})
    os_road = os_road.replace(0, "N/A")
    os_road = validate_geometries(os_road)
    os_road = clip_to_boundary(os_road, boundary)
    filter_removed = len_before_filter - len(os_road)
    LOG.info(
        "OS roads filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        os_road,
        config.paths.model_input / file_paths.OS_ROAD_MODEL_INPUT_PATH,
    )


def _clean_noham_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean NoHAM network dataset, then write to file."""
    year = config.infrastructure.road.noham.year
    noham_network = gpd.read_file(
        config.paths.raw_input / config.infrastructure.road.noham.file_path,
        mask=boundary,
        columns=["link_id"],
    )
    len_before_filter = len(noham_network)
    noham_network_clean = noham_network.drop_duplicates(subset=["link_id", "geometry"])
    noham_network_clean[["a", "b"]] = (
        noham_network_clean["link_id"].str.split("_", expand=True).astype(int)
    )
    # Filter out links with a or b less than 10,000 (zone connectors)
    noham_network_clean = noham_network_clean[
        (noham_network_clean["a"] >= config.constants.noham_road_id_threshold)
        & (noham_network_clean["b"] >= config.constants.noham_road_id_threshold)
    ]
    noham_network_clean = noham_network_clean.drop(columns=["a", "b"])
    noham_network_clean = validate_geometries(noham_network_clean)
    noham_network_clipped = clip_to_boundary(noham_network_clean, boundary)
    noham_network_clipped = noham_network_clipped.reset_index(drop=True)
    filter_removed = len_before_filter - len(noham_network_clipped)
    LOG.info(
        "NoHAM network filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        noham_network_clipped,
        config.paths.model_input
        / file_paths.NOHAM_NETWORK_MODEL_INPUT_PATH
        / f"noham_{year}.gpkg",
    )


### RAIL


def _clean_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Clean all rail datasets ready for analysis."""
    LOG.info("Cleaning rail data...")
    if config.switches.passenger_rail:
        LOG.info("Cleaning passenger rail data...")
        _clean_passenger_rail(config, rail_links)
    if config.switches.freight_rail:
        LOG.info("Cleaning freight rail data...")
        _clean_freight_rail(config, rail_links)
    LOG.info("Finished cleaning rail data.")


def _get_rail_links(
    boundary: gpd.GeoDataFrame, os_rail_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Read and clean OS Rail Network data from the Multi-Modal Routing Network."""
    rail_links = gpd.read_file(
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
    len_before_filter = len(rail_links)
    rail_links = rail_links[rail_links["operationalstatus"] == "Active"]
    rail_links = rail_links.drop(columns="operationalstatus")
    rail_links = rail_links[
        ~rail_links["description"].isin(["Preserved", "Funicular", "Mineral", "Static Museum"])
    ]
    rail_links = rail_links.drop_duplicates(subset=["osid", "geometry"])
    rail_links[
        ["description", "structure", "physicallevel", "railwayuse", "trackrepresentation"]
    ] = rail_links[
        ["description", "structure", "physicallevel", "railwayuse", "trackrepresentation"]
    ].replace(0, "N/A")
    rail_links = rail_links.rename(
        columns={
            "description": "desc",
            "physicallevel": "phys_level",
            "railwayuse": "rail_use",
            "trackrepresentation": "track_rep",
        },
    )
    rail_links = validate_geometries(rail_links)
    rail_links = clip_to_boundary(rail_links, boundary)
    filter_removed = len_before_filter - len(rail_links)
    LOG.info(
        "OS MMRN Rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    return rail_links


def _clean_passenger_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to passenger rail network, then write to file."""
    len_before_filter = len(rail_links)
    passenger_rail = rail_links[
        rail_links["rail_use"].isin(["Freight And Passenger", "Passenger"])
    ]
    passenger_rail = passenger_rail[
        passenger_rail["desc"].isin(
            ["Main Line", "Main Line And Tram", "Main Line And Rapid Transport System"]
        )
    ]
    filter_removed = len_before_filter - len(passenger_rail)
    LOG.info(
        "Passenger rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        passenger_rail,
        config.paths.model_input / file_paths.PASSENGER_RAIL_MODEL_INPUT_PATH,
    )


def _clean_freight_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to freight rail network, then write to file."""
    len_before_filter = len(rail_links)
    freight_rail = rail_links[
        rail_links["rail_use"].isin(["Freight And Passenger", "Freight"])
    ]
    filter_removed = len_before_filter - len(freight_rail)
    LOG.info(
        "Freight rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        freight_rail,
        config.paths.model_input / file_paths.FREIGHT_RAIL_MODEL_INPUT_PATH,
    )


### OTHER


def _clean_other(  # noqa: C901
    config: model_config.Config,
    boundary: gpd.GeoDataFrame,
    rail_links: gpd.GeoDataFrame,
) -> None:
    """Clean all other datasets ready for analysis."""
    LOG.info("Cleaning other infrastructure data...")
    if config.switches.airports:
        LOG.info("Cleaning airports data...")
        _clean_airports(config, boundary)
    if config.switches.bus_stops:
        LOG.info("Cleaning bus stops data...")
        _clean_bus_stops(config, boundary)
    if config.switches.petrol_stations:
        LOG.info("Cleaning petrol stations data...")
        _clean_petrol_stations(config, boundary)
    if config.switches.charging_sites:
        LOG.info("Cleaning charging sites data...")
        _clean_charging_sites(config, boundary)
    if config.switches.national_cycle_network:
        LOG.info("Cleaning NCN data...")
        _clean_ncn(config, boundary)
    if config.switches.train_stations:
        LOG.info("Cleaning train stations data...")
        _clean_train_stations(config, boundary)
    if config.switches.tram_stations:
        LOG.info("Cleaning tram stations data...")
        _clean_tram_stations(config, boundary)
    if config.switches.rapid_transport_stations:
        LOG.info("Cleaning rapid transport stations data...")
        _clean_rapid_transport_stations(config, boundary)
    if config.switches.ferry_terminals:
        LOG.info("Cleaning ferry terminals data...")
        _clean_ferry_terminals(config, boundary)
    if config.switches.bus_coach_stations:
        LOG.info("Cleaning bus coach stations data...")
        _clean_bus_coach_stations(config, boundary)
    if config.switches.tram_network:
        LOG.info("Cleaning tram network data...")
        _clean_tram_network(config, rail_links)
    if config.switches.rapid_transport_network:
        LOG.info("Cleaning rapid transport network data...")
        _clean_rapid_transport_network(config, rail_links)
    LOG.info("Finished cleaning other infrastructure data.")


def _clean_airports(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read airports dataset, then write to file in new directory."""
    airports = gpd.read_file(config.paths.raw_input / config.infrastructure.other.airports)
    airports = clip_to_boundary(airports, boundary)
    write_to_file(airports, config.paths.model_input / file_paths.AIRPORTS_MODEL_INPUT_PATH)


def _clean_bus_stops(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read, combine and clean regional bus stops datasets, then write to file."""
    bus_stops_ne = pd.read_csv(
        config.paths.raw_input / config.infrastructure.other.bus_stops["north_east"]
    )  # North East
    bus_stops_nw = pd.read_csv(
        config.paths.raw_input / config.infrastructure.other.bus_stops["north_west"]
    )  # North West
    bus_stops_ys = pd.read_csv(
        config.paths.raw_input / config.infrastructure.other.bus_stops["yorkshire"]
    )  # Yorkshire

    bus_stops = pd.concat(
        [bus_stops_ne, bus_stops_nw, bus_stops_ys], ignore_index=True
    )  # Combine bus stops
    bus_stops = _df_to_gdf(bus_stops, "stop_lon", "stop_lat", "EPSG:4326")
    bus_stops = bus_stops[["stop_id", "stop_name", "geometry"]]
    len_before_filter = len(bus_stops)
    bus_stops = bus_stops.drop_duplicates(
        subset=["stop_id", "geometry"]
    )  # Remove duplicate rows
    bus_stops = validate_geometries(bus_stops)
    bus_stops = clip_to_boundary(bus_stops, boundary)
    filter_removed = len_before_filter - len(bus_stops)
    LOG.info(
        "Bus stops filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        bus_stops,
        config.paths.model_input / file_paths.BUS_STOPS_MODEL_INPUT_PATH,
    )


def _clean_petrol_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean POI data, filter for petrol stations, and write to file."""
    petrol_stations = gpd.read_file(
        f"zip://{config.paths.raw_input / config.infrastructure.other.poi_uk.zip_path}!"
        f"{config.infrastructure.other.poi_uk.file_path}",
        columns=["id"],
        where="main_category = 'gas_station'",
    )
    len_before_filter = len(petrol_stations)
    petrol_stations = petrol_stations.drop_duplicates()
    petrol_stations = validate_geometries(petrol_stations)
    petrol_stations = clip_to_boundary(petrol_stations, boundary)
    filter_removed = len_before_filter - len(petrol_stations)
    LOG.info(
        "Petrol stations filtered from POIs - %s of %s (%.1f percent)",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        petrol_stations,
        config.paths.model_input / file_paths.PETROL_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_train_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for train stations, then clip to boundary and write to file."""
    os_mmrn_railway_stations = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.os_mmrn,
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
    train_stations = validate_geometries(train_stations)
    train_stations = clip_to_boundary(train_stations, boundary)
    filter_removed = len_before_filter - len(train_stations)
    LOG.info(
        "Train stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        train_stations,
        config.paths.model_input / file_paths.TRAIN_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_tram_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for tram stations, then clip to boundary and write to file."""
    tram_stations = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Tram Station%'",
        columns=["nodeid"],
    )
    len_before_filter = len(tram_stations)
    tram_stations = tram_stations.drop_duplicates()
    tram_stations = validate_geometries(tram_stations)
    tram_stations = clip_to_boundary(tram_stations, boundary)
    filter_removed = len_before_filter - len(tram_stations)
    LOG.info(
        "Tram stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tram_stations,
        config.paths.model_input / file_paths.TRAM_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_rapid_transport_stations(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> None:
    """Filter OS MMRN for rapid transport stations, then clip to boundary and write to file."""
    rapid_transport_stations = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Underground System%'",
        columns=["nodeid"],
    )
    len_before_filter = len(rapid_transport_stations)
    rapid_transport_stations = rapid_transport_stations.drop_duplicates()
    rapid_transport_stations = validate_geometries(rapid_transport_stations)
    rapid_transport_stations = clip_to_boundary(rapid_transport_stations, boundary)
    filter_removed = len_before_filter - len(rapid_transport_stations)
    LOG.info(
        "Rapid transport stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        rapid_transport_stations,
        config.paths.model_input / file_paths.RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_ferry_terminals(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for ferry terminals, then clip to boundary and write to file."""
    ferry_terminals = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Ferry%'",
        columns=["nodeid"],
    )
    len_before_filter = len(ferry_terminals)
    ferry_terminals = ferry_terminals.drop_duplicates()
    ferry_terminals = validate_geometries(ferry_terminals)
    ferry_terminals = clip_to_boundary(ferry_terminals, boundary)
    filter_removed = len_before_filter - len(ferry_terminals)
    LOG.info(
        "Ferry terminals filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        ferry_terminals,
        config.paths.model_input / file_paths.FERRY_TERMINALS_MODEL_INPUT_PATH,
    )


def _clean_bus_coach_stations(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Filter OS MMRN for bus and coach stations, then clip to boundary and write to file."""
    bus_coach_stations = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.os_mmrn,
        layer="mrn_ntwk_transportnode",
        where="os_nodetype LIKE '%Bus Station%' OR os_nodetype LIKE '%Coach Station%'",
        columns=["nodeid"],
    )
    len_before_filter = len(bus_coach_stations)
    bus_coach_stations = bus_coach_stations.drop_duplicates()
    bus_coach_stations = validate_geometries(bus_coach_stations)
    bus_coach_stations = clip_to_boundary(bus_coach_stations, boundary)
    filter_removed = len_before_filter - len(bus_coach_stations)
    LOG.info(
        "Bus and coach stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        bus_coach_stations,
        config.paths.model_input / file_paths.BUS_COACH_STATIONS_MODEL_INPUT_PATH,
    )


def _clean_tram_network(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail links for tram network, then write to file."""
    len_before_filter = len(rail_links)
    tram_links = rail_links[
        rail_links["rail_use"].isin(["Freight And Passenger", "Passenger"])
    ]
    tram_links = tram_links[tram_links["desc"].isin(["Tram", "Main Line And Tram"])]
    filter_removed = len_before_filter - len(tram_links)
    LOG.info(
        "Tram network links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        tram_links, config.paths.model_input / file_paths.TRAM_NETWORK_MODEL_INPUT_PATH
    )


def _clean_rapid_transport_network(
    config: model_config.Config, rail_links: gpd.GeoDataFrame
) -> None:
    """Filter OS rail links for rapid transport network, then write to file."""
    len_before_filter = len(rail_links)
    rapid_transport_links = rail_links[
        rail_links["rail_use"].isin(["Freight And Passenger", "Passenger"])
    ]
    rapid_transport_links = rapid_transport_links[
        rapid_transport_links["desc"].isin(
            ["Rapid Transport System", "Main Line And Rapid Transport System"]
        )
    ]
    filter_removed = len_before_filter - len(rapid_transport_links)
    LOG.info(
        "Rapid transport links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        rapid_transport_links,
        config.paths.model_input / file_paths.RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH,
    )


def _clean_charging_sites(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean ZapMap charging sites data, then write to file."""
    chg_sites = pd.read_csv(
        config.paths.raw_input / config.infrastructure.other.zapmap,
        usecols=["identifier", "name", "speed", "value", "lon", "lat"],
    )
    chg_sites = gpd.GeoDataFrame(
        chg_sites,
        geometry=[
            geometry.Point(xy) for xy in zip(chg_sites["lon"], chg_sites["lat"], strict=False)
        ],
        crs="EPSG:4326",
    )
    chg_sites = chg_sites.drop(columns=["lon", "lat"])
    chg_sites = chg_sites.rename(columns={"identifier": "id", "value": "devices"})
    len_before_filter = len(chg_sites)
    chg_sites = chg_sites.drop_duplicates(subset=["geometry"])
    chg_sites = validate_geometries(chg_sites)
    chg_sites = clip_to_boundary(chg_sites, boundary)
    filter_removed = len_before_filter - len(chg_sites)
    LOG.info(
        "Charging sites filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        chg_sites, config.paths.model_input / file_paths.CHARGING_SITES_MODEL_INPUT_PATH
    )


def _clean_ncn(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean National Cycle Network data, then write to file."""
    ncn = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.ncn_sustrans,
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
    ncn = validate_geometries(ncn)
    ncn = clip_to_boundary(ncn, boundary)
    filter_removed = len_before_filter - len(ncn)
    LOG.info(
        "NCN filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        ncn,
        config.paths.model_input / file_paths.NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH,
    )


## HAZARDS


def _clean_hazards(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean hazard data ready for analysis."""
    LOG.info("Cleaning hazard data...")
    if config.switches.extreme_weather:
        _clean_extreme_weather(config, boundary)
    if config.switches.flooding:
        _clean_flooding(config, boundary)
    if config.switches.ground_stability:
        _clean_ground_stability(config, boundary)
    if config.switches.coastal_erosion:
        _clean_coastal_erosion(config, boundary)
    LOG.info("Finished cleaning hazard data.")


### EXTREME WEATHER


def _clean_extreme_weather(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all extreme weather datasets ready for analysis."""
    LOG.info("Cleaning extreme weather data...")
    hazard_grid = _clean_hazard_grid(config, boundary)

    _clean_temp_max(config, hazard_grid)
    _clean_temp_min(config, hazard_grid)
    _clean_summer_precip(config, hazard_grid)
    _clean_winter_precip(config, hazard_grid)
    _clean_rain_days(config, boundary)
    _clean_drought_index(config, boundary)
    _clean_hot_summer_days(config, hazard_grid)
    _clean_extreme_summer_days(config, hazard_grid)
    _clean_frost_days(config, hazard_grid)
    _clean_icing_days(config, hazard_grid)
    _clean_wind_speed(config, boundary)
    _clean_wind_driven_rain(config, boundary)
    LOG.info("Finished cleaning extreme weather data.")


def _clean_hazard_grid(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Create and prepare common hazard grid DataFrame for variables on same 12km BNG."""
    hazard_grid = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.max_temp_summer,
        mask=boundary,
        columns=[],
    )
    hazard_grid["grid_id"] = range(1, len(hazard_grid) + 1)
    len_before_filter = len(hazard_grid)
    hazard_grid = clip_to_boundary(hazard_grid, boundary)
    filter_removed = len_before_filter - len(hazard_grid)
    LOG.info(
        "Extreme weather grid filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    hazard_grid = explode_to_polygons(hazard_grid, track_part=True)
    write_to_file(
        hazard_grid,
        config.paths.model_input / file_paths.HAZARD_GRID_MODEL_INPUT_PATH,
    )
    return hazard_grid


def _clean_temp_max(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean max summer temperature change projections, then write to file."""
    temp_max = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.max_temp_summer,
        columns=["tasmax_summer_01_20_median", "tasmax_summer_change_40_median"],
    )
    temp_max["grid_id"] = range(1, len(temp_max) + 1)
    temp_max = temp_max.drop(columns=["geometry"])
    temp_max = temp_max.rename(
        columns={
            "tasmax_summer_01_20_median": (
                f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.CURRENT}",
            ),
            "tasmax_summer_change_40_median": (
                f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.FORECAST}",
            ),
        }
    )
    temp_max[f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.FORECAST}"] = (
        temp_max[f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.CURRENT}"]
        + temp_max[f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.FORECAST}"]
    )
    len_before_filter = len(temp_max)
    temp_max = temp_max[temp_max["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(temp_max)
    LOG.info(
        "Summer max temperature change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(temp_max, config.paths.model_input / file_paths.TEMP_MAX_MODEL_INPUT_PATH)


def _clean_temp_min(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean min winter temperature change projections, then write to file."""
    temp_min = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.min_temp_winter,
        columns=["tasmin_winter_01_20_median", "tasmin_winter_change_40_median"],
    )
    temp_min["grid_id"] = range(1, len(temp_min) + 1)
    temp_min = temp_min.drop(columns=["geometry"])
    temp_min = temp_min.rename(
        columns={
            "tasmin_winter_01_20_median": (
                f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.CURRENT}",
            ),
            "tasmin_winter_change_40_median": (
                f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.FORECAST}",
            ),
        }
    )
    temp_min[f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.FORECAST}"] = (
        temp_min[f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.CURRENT}"]
        + temp_min[f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.FORECAST}"]
    )
    len_before_filter = len(temp_min)
    temp_min = temp_min[temp_min["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(temp_min)
    LOG.info(
        "Winter minimum temperature change projections filtered - %s of %s (%.1f percent) "
        "rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        temp_min,
        config.paths.model_input / file_paths.TEMP_MIN_MODEL_INPUT_PATH,
    )


def _clean_summer_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean summer precipitation change projections, then write to file."""
    precip_sum = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.precip_summer,
        columns=["pr_summer_01_20_median", "pr_summer_change_40_median"],
    )
    precip_sum["grid_id"] = range(1, len(precip_sum) + 1)
    precip_sum = precip_sum.drop(columns=["geometry"])
    precip_sum = precip_sum.rename(
        columns={
            "pr_summer_01_20_median": (f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}",),
            "pr_summer_change_40_median": (
                f"{DroughtCols.PRECIP_SUMMER}_pct_chg_{Scenarios.FORECAST}",
            ),
        }
    )
    precip_sum[f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}"] = precip_sum[
        f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}"
    ] * (1 + (precip_sum[f"{DroughtCols.PRECIP_SUMMER}_pct_chg_{Scenarios.FORECAST}"] / 100))
    precip_sum = precip_sum.drop(
        columns=[f"{DroughtCols.PRECIP_SUMMER}_pct_chg_{Scenarios.FORECAST}"]
    )
    len_before_filter = len(precip_sum)
    precip_sum = precip_sum[precip_sum["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(precip_sum)
    LOG.info(
        "Summer precipitation change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        precip_sum, config.paths.model_input / file_paths.SUMMER_PRECIP_MODEL_INPUT_PATH
    )


def _clean_winter_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean winter precipitation change projections, then write to file."""
    precip_win = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.precip_winter,
        columns=["pr_winter_01_20_median", "pr_winter_change_40_median"],
    )
    precip_win["grid_id"] = range(1, len(precip_win) + 1)
    precip_win = precip_win.drop(columns=["geometry"])
    precip_win = precip_win.rename(
        columns={
            "pr_winter_01_20_median": (f"{StormCols.PRECIP_WINTER}_{Scenarios.CURRENT}",),
            "pr_winter_change_40_median": (
                f"{StormCols.PRECIP_WINTER}_pct_chg_{Scenarios.FORECAST}",
            ),
        },
    )
    precip_win[f"{StormCols.PRECIP_WINTER}_{Scenarios.FORECAST}"] = precip_win[
        f"{StormCols.PRECIP_WINTER}_{Scenarios.CURRENT}"
    ] * (1 + (precip_win[f"{StormCols.PRECIP_WINTER}_pct_chg_{Scenarios.FORECAST}"] / 100))
    precip_win = precip_win.drop(
        columns=[f"{StormCols.PRECIP_WINTER}_pct_chg_{Scenarios.FORECAST}"]
    )
    len_before_filter = len(precip_win)
    precip_win = precip_win[precip_win["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(precip_win)
    LOG.info(
        "Winter precipitation change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        precip_win, config.paths.model_input / file_paths.WINTER_PRECIP_MODEL_INPUT_PATH
    )


def _clean_rain_days(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean 10mm rain days observations, then write to file."""
    rain_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.rain_days,
        mask=boundary,
    )
    len_before_filter = len(rain_days)
    rain_days = clip_to_boundary(rain_days, boundary)
    rain_days = rain_days.rename(
        columns={"Rain10mmDays": f"{StormCols.RAIN_DAYS}_{Scenarios.CURRENT}"}
    )
    filter_removed = len_before_filter - len(rain_days)
    LOG.info(
        "10mm rain days 1991-2020 filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    rain_days = explode_to_polygons(rain_days)
    write_to_file(rain_days, config.paths.model_input / file_paths.RAIN_DAYS_MODEL_INPUT_PATH)


def _clean_drought_index(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean drought severity index data, then write to file."""
    drought_index = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.drought_index,
        mask=boundary,
        columns=["DSI12_baseline_00_17_median", "DSI12_40_median"],
    )
    drought_index = drought_index.rename(
        columns={
            "DSI12_baseline_00_17_median": (
                f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.CURRENT}",
            ),
            "DSI12_40_median": (f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.FORECAST}",),
        }
    )
    len_before_filter = len(drought_index)
    drought_index = clip_to_boundary(drought_index, boundary)
    filter_removed = len_before_filter - len(drought_index)
    LOG.info(
        "Drought severity index filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    drought_index = explode_to_polygons(drought_index)
    write_to_file(
        drought_index, config.paths.model_input / file_paths.DROUGHT_INDEX_MODEL_INPUT_PATH
    )


def _clean_hot_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean hot summer days projections, then write to file."""
    hot_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.hot_days,
        columns=["HSD_baseline_01_20_median", "HSD_40_median"],
    )
    hot_days["grid_id"] = range(1, len(hot_days) + 1)
    hot_days = hot_days.drop(columns=["geometry"])
    hot_days = hot_days.rename(
        columns={
            "HSD_baseline_01_20_median": (
                f"{ExtremeHeatCols.HOT_SUMMER_DAYS}_{Scenarios.CURRENT}",
            ),
            "HSD_40_median": (f"{ExtremeHeatCols.HOT_SUMMER_DAYS}_{Scenarios.FORECAST}",),
        }
    )
    len_before_filter = len(hot_days)
    hot_days = hot_days[hot_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(hot_days)
    LOG.info(
        "Hot summer days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        hot_days, config.paths.model_input / file_paths.HOT_SUMMER_DAYS_MODEL_INPUT_PATH
    )


def _clean_extreme_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean extreme summer days projections, then write to file."""
    extr_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.extreme_summer_days,
        columns=["ESD_baseline_01_20_median", "ESD_40_median"],
    )
    extr_days["grid_id"] = range(1, len(extr_days) + 1)
    extr_days = extr_days.drop(columns=["geometry"])
    extr_days = extr_days.rename(
        columns={
            "ESD_baseline_01_20_median": (
                f"{ExtremeHeatCols.EXTREME_SUMMER_DAYS}_{Scenarios.CURRENT}",
            ),
            "ESD_40_median": (f"{ExtremeHeatCols.EXTREME_SUMMER_DAYS}_{Scenarios.FORECAST}",),
        }
    )
    len_before_filter = len(extr_days)
    extr_days = extr_days[extr_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(extr_days)
    LOG.info(
        "Extreme summer days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        extr_days,
        config.paths.model_input / file_paths.EXTREME_SUMMER_DAYS_MODEL_INPUT_PATH,
    )


def _clean_frost_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean frost days projections, then write to file."""
    frost_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.frost_days,
        columns=["FrostDays_baseline_01_20_median", "FrostDays_40_median"],
    )
    frost_days["grid_id"] = range(1, len(frost_days) + 1)
    frost_days = frost_days.drop(columns=["geometry"])
    frost_days = frost_days.rename(
        columns={
            "FrostDays_baseline_01_20_median": (
                f"{ExtremeColdCols.FROST_DAYS}_{Scenarios.CURRENT}",
            ),
            "FrostDays_40_median": (f"{ExtremeColdCols.FROST_DAYS}_{Scenarios.FORECAST}",),
        }
    )
    len_before_filter = len(frost_days)
    frost_days = frost_days[frost_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(frost_days)
    LOG.info(
        "Frost days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(
        frost_days, config.paths.model_input / file_paths.FROST_DAYS_MODEL_INPUT_PATH
    )


def _clean_icing_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean icing days projections, then write to file."""
    ice_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.icing_days,
        columns=["IcingDays_baseline_01_20_median", "IcingDays_40_median"],
    )
    ice_days["grid_id"] = range(1, len(ice_days) + 1)
    ice_days = ice_days.drop(columns=["geometry"])
    ice_days = ice_days.rename(
        columns={
            "IcingDays_baseline_01_20_median": (
                f"{ExtremeColdCols.ICING_DAYS}_{Scenarios.CURRENT}",
            ),
            "IcingDays_40_median": (f"{ExtremeColdCols.ICING_DAYS}_{Scenarios.FORECAST}",),
        }
    )
    len_before_filter = len(ice_days)
    ice_days = ice_days[ice_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(ice_days)
    LOG.info(
        "Icing days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    write_to_file(ice_days, config.paths.model_input / file_paths.ICING_DAYS_MODEL_INPUT_PATH)


def _clean_wind_speed(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read wind speed projections, calculate metrics, clean, then write to file."""
    windspd_current_agg, len_before_filter_current = _read_wind_speed_reduce(
        config.paths.raw_input / config.hazards.extreme_weather.wind_speed["1990_2000"],
        Scenarios.CURRENT,
    )
    windspd_forecast_agg, len_before_filter_forecast = _read_wind_speed_reduce(
        config.paths.raw_input / config.hazards.extreme_weather.wind_speed["2070_2080"],
        Scenarios.FORECAST,
    )

    metric_cols = [
        f"{StormCols.WIND_SPEED}_{Scenarios.CURRENT}",
        f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.CURRENT}",
        f"{StormCols.WIND_SPEED}_{Scenarios.FORECAST}",
        f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.FORECAST}",
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
    windspd_combined = clip_to_boundary(windspd_combined, boundary)
    filter_removed = len_before_filter - len(windspd_combined)
    LOG.info(
        "Wind speed projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    windspd_combined = explode_to_polygons(windspd_combined)
    write_to_file(
        windspd_combined, config.paths.model_input / file_paths.WIND_SPEED_MODEL_INPUT_PATH
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
    return avg_exc.to_dataset(name=f"{StormCols.EXCEEDANCE_DAYS}_{scenario}")


def _calculate_windspd_percentile(
    windspd_data: xr.Dataset, quantile: float, variable: str, scenario: str
) -> xr.Dataset:
    """Calculate the wind speed percentiles per geometry for a given variable."""
    pct = windspd_data[variable].quantile(quantile, dim="time")
    return pct.to_dataset(name=f"{StormCols.WIND_SPEED}_{scenario}")


def _clean_wind_driven_rain(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean wind driven rain index data, then write to file."""
    wind_driven_rain = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.wdr_index,
        mask=boundary,
        columns=["WDR_baseline_Median", "WDR_40_Median", "x_coord", "y_coord"],
    )
    len_before_filter = len(wind_driven_rain)
    # Aggregate by wind direction to calculate mean wind speed
    wind_driven_rain = (
        wind_driven_rain.groupby(["x_coord", "y_coord"])
        .agg({"WDR_baseline_Median": "mean", "WDR_40_Median": "mean", "geometry": "first"})
        .reset_index()
    )

    wind_driven_rain = wind_driven_rain.drop(columns=["x_coord", "y_coord"])
    wind_driven_rain = wind_driven_rain.rename(
        columns={
            "WDR_baseline_Median": f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.CURRENT}",
            "WDR_40_Median": f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.FORECAST}",
        }
    )
    wind_driven_rain = gpd.GeoDataFrame(wind_driven_rain, geometry="geometry", crs="EPSG:3857")
    wind_driven_rain = clip_to_boundary(wind_driven_rain, boundary)
    filter_removed = len_before_filter - len(wind_driven_rain)
    LOG.info(
        "Wind driven rain index filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    wind_driven_rain = explode_to_polygons(wind_driven_rain)
    write_to_file(
        wind_driven_rain,
        config.paths.model_input / file_paths.WIND_DRIVEN_RAIN_MODEL_INPUT_PATH,
    )


### FLOODING


def _clean_flooding(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean flooding data ready for analysis."""
    LOG.info("Cleaning flooding data...")
    bng_codes = _get_bng_codes(boundary)

    for flooding_type in config.hazards.flooding:
        for scenario in Scenarios:
            LOG.info("Cleaning %s flooding data for %s scenario...", flooding_type, scenario)
            _clean_flooding_layer(
                config=config,
                flooding_type=flooding_type,
                boundary=boundary,
                out_path=file_paths.FLOODING_MODEL_INPUT_PATH
                / flooding_type
                / scenario
                / f"{flooding_type}_{scenario}.gpkg",
                rename_risk_col=f"{flooding_type}_flooding_risk_{scenario}",
                climate_change=(scenario == Scenarios.FORECAST),
                bng_codes=bng_codes,
            )

    LOG.info("Finished cleaning flooding data.")


def _clean_flooding_layer(
    config: model_config.Config,
    flooding_type: str,
    boundary: gpd.GeoDataFrame,
    *,
    out_path: pathlib.Path,
    rename_risk_col: str,
    climate_change: bool,
    bng_codes: list[str],
) -> None:
    """Clean flooding data and write to file."""
    zip_files = _get_flooding_zip_files(config, flooding_type)
    first_write = True

    for zip_path in zip_files:
        metadata = _parse_flooding_metadata(zip_path)

        if metadata["tile"][:2] not in bng_codes:
            continue
        if metadata["climate_change"] != climate_change:
            continue
        LOG.info("Processing tile: %s...", metadata["tile"])

        flooding_data = _read_flooding_zip(zip_path, boundary)

        flooding_data = clip_to_boundary(flooding_data, boundary)

        if flooding_data.empty:
            LOG.info("%s layer empty. Continuing.", metadata["tile"])
            continue

        flooding_data = _extract_poly_from_geomcollection(flooding_data)

        flooding_data = flooding_data.rename(columns={"Risk_band": rename_risk_col})

        write_to_file(
            flooding_data,
            config.paths.model_input / out_path,
            mode="w" if first_write else "a",
        )

        first_write = False

        del flooding_data
        gc.collect()


def _get_flooding_zip_files(
    config: model_config.Config,
    flooding_type: str,
) -> list[pathlib.Path]:
    """Return all flooding zip files."""
    flooding_root = config.hazards.flooding[flooding_type]
    return sorted(flooding_root.rglob("*.zip"))


def _parse_flooding_metadata(zip_path: pathlib.Path) -> dict:
    """Extract metadata from flooding filename."""
    name = zip_path.stem
    climate_change = "Climate_Change" in name

    try:
        if climate_change:
            flooding_type, _, _, _, tile, version = name.split("_")
        else:
            flooding_type, tile, version = name.split("_")
    except ValueError as exc:
        raise ValueError(f"Unexpected flooding zip filename format: '{name}'") from exc

    return {
        "flooding_type": flooding_type,
        "tile": tile,
        "version": version,
        "climate_change": climate_change,
    }


def _read_flooding_zip(
    zip_path: pathlib.Path,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """Read flooding gdb file from zip."""
    vsi_path = f"zip://{zip_path}!/{zip_path.stem}.gdb"
    layers = fiona.listlayers(vsi_path)

    if not layers:
        raise ValueError(f"No layers found in zip file: {zip_path}")

    return gpd.read_file(
        vsi_path,
        layer=layers[0],
        mask=boundary,
        columns=["Risk_band"],
        engine="pyogrio",
        use_arrow=True,
    )


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
        GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.file_path}",
            layer="GB_Hex_5km_GS_CollapsibleDeposits_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        GroundStabilityRiskCols.COMPRESSIBLE_GROUND: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.file_path}",
            layer="GB_Hex_5km_GS_CompressibleGround_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        GroundStabilityRiskCols.LANDSLIDES: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.file_path}",
            layer="GB_Hex_5km_GS_Landslides_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        GroundStabilityRiskCols.RUNNING_SAND: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.file_path}",
            layer="GB_Hex_5km_GS_RunningSand_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        GroundStabilityRiskCols.SHRINK_SWELL: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.file_path}",
            layer="GB_Hex_5km_GS_ShrinkSwell_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        GroundStabilityRiskCols.SOLUBLE_ROCKS: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.zip_path}!"
            f"{config.hazards.ground_stability.geosure.file_path}",
            layer="GB_Hex_5km_GS_SolubleRocks_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
    }

    for code, geosure_data in geosure_layers.items():
        if geosure_data.empty:
            raise ValueError(
                f"GeoSure {code} layer is empty after reading. "
                f"Check the source file and boundary."
            )
        len_before_filter = len(geosure_data)
        geosure_data_clean = geosure_data.rename(columns={"CLASS": code})
        geosure_layers[code] = clip_to_boundary(geosure_data_clean, boundary)
        filter_removed = len_before_filter - len(geosure_layers[code])
        LOG.info(
            "GeoSure %s filtered - %s of %s (%.1f percent) rows removed",
            code.replace("_", " ").title(),
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        geosure_layers[code] = explode_to_polygons(geosure_layers[code])

    # Merge layers based on nearest centroids
    base_code = next(iter(geosure_layers.keys()))
    geosure = geosure_layers[base_code][[base_code, "geometry"]].copy()
    for code, layer in geosure_layers.items():
        if code == base_code:
            continue  # skip the base layer
        layer_subset = layer[
            [code, "geometry"]
        ]  # Select only the relevant class and geometry columns
        matched = _nearest_centroids(geosure, layer_subset)  # Apply nearest centroid matching
        geosure[code] = matched[code]  # Add the matched CLASS column to the base dataframe

    geosure_risk_cols = [col for col in geosure.columns if col.endswith("_risk")]
    geosure = geosure[[*geosure_risk_cols, "geometry"]]
    write_to_file(
        geosure,
        config.paths.model_input / file_paths.GEOSURE_MODEL_INPUT_PATH,
    )


def _clean_geoclimate(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean GeoClimate Shrink-Swell data, then write to file."""
    for year, filepath in config.hazards.ground_stability.geo_shrink_swell.items():
        geoclimate_data = gpd.read_file(
            config.paths.raw_input / filepath, mask=boundary, columns=["CLASS"]
        )
        if geoclimate_data.empty:
            LOG.info("GeoClimate shrink-swell %s layer empty. Continuing.", year)
            continue
        geoclimate_data = geoclimate_data.rename(
            columns={"CLASS": GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE}
        )
        len_before_filter = len(geoclimate_data)
        geoclimate_data = clip_to_boundary(geoclimate_data, boundary)
        filter_removed = len_before_filter - len(geoclimate_data)
        LOG.info(
            "GeoClimate %s filtered - %s of %s (%.1f percent) rows removed",
            year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        geoclimate_data = explode_to_polygons(geoclimate_data)

        write_to_file(
            geoclimate_data,
            config.paths.model_input
            / file_paths.GEOCLIMATE_SHRINK_SWELL_MODEL_INPUT_PATH
            / f"bgs_ss_{year}.gpkg",
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
        f"zip://{config.paths.raw_input / config.hazards.coastal_erosion.zip_path}"
        f"!{config.hazards.coastal_erosion.file_path}",
        layer="NCERM_Ground_Instability_Zone",
        mask=boundary,
        columns=["smp_no"],
    )
    if ncerm_giz.empty:
        LOG.info("Ground Instability Zones layer empty after filtering. Writing empty file.")
        write_to_file(
            gpd.GeoDataFrame(columns=["smp_no", "geometry"], geometry="geometry", crs=BNG_CRS),
            config.paths.model_input / file_paths.GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH,
        )
        return
    len_before_filter = len(ncerm_giz)
    ncerm_giz = clip_to_boundary(ncerm_giz, boundary)
    filter_removed = len_before_filter - len(ncerm_giz)
    LOG.info(
        "Ground Instability Zones filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    ncerm_giz = explode_to_polygons(ncerm_giz)

    write_to_file(
        ncerm_giz,
        config.paths.model_input / file_paths.GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH,
    )


def _clean_ncerm(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean erosion data from NCERM for 2055, and 2105, then write to file."""
    for year in ["2055", "2105"]:
        erosion_data = gpd.read_file(
            f"zip://{config.paths.raw_input / config.hazards.coastal_erosion.zip_path}!"
            f"{config.hazards.coastal_erosion.file_path}",
            layer=f"NCERM_SMP_{year}_70CC",
            mask=boundary,
            columns=["smp_name"],
        )
        if erosion_data.empty:
            LOG.info("NCERM %s layer empty after filtering. Writing empty file.", year)
            write_to_file(
                gpd.GeoDataFrame(
                    columns=["smp_name", "geometry"], geometry="geometry", crs=BNG_CRS
                ),
                config.paths.model_input
                / file_paths.NCERM_MODEL_INPUT_PATH
                / f"ncerm_smp_{year}_70CC.gpkg",
            )
            if year == "2055":
                continue
            return
        len_before_filter = len(erosion_data)
        erosion_data = clip_to_boundary(erosion_data, boundary)
        filter_removed = len_before_filter - len(erosion_data)
        LOG.info(
            "NCERM %s filtered - %s of %s (%.1f percent) rows removed",
            year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        erosion_data = explode_to_polygons(erosion_data)

        write_to_file(
            erosion_data,
            config.paths.model_input
            / file_paths.NCERM_MODEL_INPUT_PATH
            / f"ncerm_smp_{year}_70CC.gpkg",
        )


## IMPACT


def _clean_impact(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean impact datasets ready for analysis."""
    LOG.info("Cleaning impact data...")
    if config.switches.freight_rail:
        _clean_freight_demand(config, boundary)
    if config.switches.noham_roads:
        _clean_noham_flows(config)
    LOG.info("Finished cleaning impact data.")


### FREIGHT


def _clean_freight_demand(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean freight demand data ready for analysis."""
    freight_network_demand = _read_freight_demand(
        config.paths.raw_input / config.impact.freight_demand, boundary
    )

    os_freight_network_demand = _map_freight_networks(
        freight_network_demand,
        config.paths.model_input / file_paths.FREIGHT_RAIL_MODEL_INPUT_PATH,
    )

    write_to_file(
        os_freight_network_demand,
        config.paths.model_input / file_paths.FREIGHT_DEMAND_MODEL_INPUT_PATH,
    )


def _read_freight_demand(path: pathlib.Path, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Read and clean freight demand data, and return as GeoDataFrame."""
    freight_network_demand = gpd.read_file(
        path, mask=boundary, columns=["dij_id", "2022_23_total", "2050_51 sc2_total"]
    )
    freight_network_demand = freight_network_demand.rename(
        columns={
            "2022_23_total": f"demand_{Scenarios.CURRENT}",
            "2050_51 sc2_total": f"demand_{Scenarios.FORECAST}",
        }
    )
    len_before_filter = len(freight_network_demand)
    freight_network_demand = clip_to_boundary(freight_network_demand, boundary)
    filter_removed = len_before_filter - len(freight_network_demand)
    LOG.info(
        "Freight demand filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    return freight_network_demand


def _map_freight_networks(
    freight_network_demand: gpd.GeoDataFrame, os_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Map freight demand data onto OS network, then clean and return."""
    os_freight_rail = gpd.read_file(os_path)
    os_freight_rail = os_freight_rail.to_crs(freight_network_demand.crs)
    len_before_mapping = len(freight_network_demand)
    os_freight_network_demand = gpd.sjoin_nearest(
        os_freight_rail,
        freight_network_demand,
        how="left",
        max_distance=_FREIGHT_DEMAND_NETWORK_MAP_MAX_DISTANCE,
        distance_col="distance",
    )
    len_after_mapping = len(os_freight_network_demand)
    LOG.info(
        "Freight demand mapped to OS network - %s segments mapped onto %s OS network segments",
        len_before_mapping,
        len_after_mapping,
    )

    os_freight_network_demand[
        [f"demand_{Scenarios.CURRENT}", f"demand_{Scenarios.FORECAST}"]
    ] = os_freight_network_demand[
        [f"demand_{Scenarios.CURRENT}", f"demand_{Scenarios.FORECAST}"]
    ].fillna(0)
    return os_freight_network_demand.drop(columns=["index_right"])


### NoHAM


def _clean_noham_flows(config: model_config.Config) -> None:
    """Clean NoHAM flows data, aggregate link flows, merge with network, then write to file."""
    noham_network = gpd.read_file(
        config.paths.model_input
        / file_paths.NOHAM_NETWORK_MODEL_INPUT_PATH
        / f"noham_{config.infrastructure.road.noham.year}.gpkg"
    )
    network_link_ids = set(noham_network["link_id"])

    scenario_flows = {}
    for year_label, year in config.impact.noham_years.items():
        if year_label == "baseline":
            scenario = Scenarios.CURRENT
        elif year_label == "future":
            scenario = Scenarios.FORECAST
        else:
            raise ValueError(
                f"Unexpected year label: {year_label}, expects 'baseline' or 'future'."
            )
        flows = _aggregate_link_flows_year(config, year, network_link_ids)

        flows = flows.rename(
            columns={col: f"{col}_{scenario}" for col in flows.columns if col != "link_id"}
        )

        scenario_flows[scenario] = flows

    current_ids = set(scenario_flows[Scenarios.CURRENT]["link_id"])
    forecast_ids = set(scenario_flows[Scenarios.FORECAST]["link_id"])
    common_ids = current_ids & forecast_ids
    current_only_ids = current_ids - forecast_ids
    forecast_only_ids = forecast_ids - current_ids

    noham_flows = scenario_flows[Scenarios.CURRENT].merge(
        scenario_flows[Scenarios.FORECAST], on="link_id", how="inner"
    )

    LOG.info(
        "NoHAM flows merged: \n"
        "Current links: %s, Forecast links: %s \n"
        "Common links: %s, Current only links: %s, Forecast only links: %s",
        len(current_ids),
        len(forecast_ids),
        len(common_ids),
        len(current_only_ids),
        len(forecast_only_ids),
    )

    noham_net_flows = noham_network.merge(noham_flows, on="link_id", how="left")

    noham_net_flows = gpd.GeoDataFrame(
        noham_net_flows, geometry="geometry", crs=noham_network.crs
    )

    write_to_file(
        noham_net_flows, config.paths.model_input / file_paths.NOHAM_FLOWS_MODEL_INPUT_PATH
    )


def _read_noham_h5(
    *,
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    year: int,
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

    noham_demand_path = (
        output_path
        / "input h5s"
        / str(year)
        / f"NoHAM_Decarb_DM_Core_{year!s}_{time_period}_v107_SatPig_{user_class}.h5"
    )

    if (str(year), time_period) not in route_links_store:
        noham_routes = pd.read_hdf(noham_demand_path, key="/data/Route")
        noham_routes = noham_routes.reset_index()[["route", "link_id"]]
        noham_links = pd.read_hdf(noham_demand_path, key="/data/link")
        noham_links = noham_links[["a", "b"]]
        route_links_store[(str(year), time_period)] = (noham_routes, noham_links)
    else:
        noham_routes, noham_links = route_links_store[(str(year), time_period)]

    noham_ods = pd.read_hdf(noham_demand_path, key="/data/OD")
    noham_ods = noham_ods.reset_index()[["route", "abs_demand"]]

    return noham_ods, noham_routes, noham_links


def _aggregate_link_flows(
    ods: pd.DataFrame, routes: pd.DataFrame, links: pd.DataFrame
) -> pd.DataFrame:
    """Take NoHAM od's, routes, and link to create aggregated link flows DataFrame."""
    # Merge OD demand onto routes
    od_routes = routes.merge(ods[["route", "abs_demand"]], on="route", how="inner")

    link_demand = od_routes.groupby("link_id")["abs_demand"].sum().reset_index()

    return link_demand.merge(links[["a", "b"]], left_on="link_id", right_index=True)


def _process_single_noham_layer(
    config: model_config.Config,
    *,
    year: int,
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    time_period: str,
    user_class: str,
    link_ids: set[str],
) -> pd.DataFrame:
    LOG.info("Processing NoHAM demand: %s %s %s", year, time_period, user_class)

    noham_ods, noham_routes, noham_links = _read_noham_h5(
        route_links_store=route_links_store,
        year=year,
        time_period=time_period,
        user_class=user_class,
        noham_path=config.paths.raw_input / config.impact.noham_demand,
        output_path=config.paths.raw_input / file_paths.NOHAM_ZIP_EXTRACT_OUTPUT_PATH,
        extract=config.switches.noham_zip_extract,
    )
    noham_links["noham_link_id"] = (
        noham_links["a"].astype(str) + "_" + noham_links["b"].astype(str)
    )
    noham_links = noham_links[noham_links["noham_link_id"].isin(link_ids)]
    noham_routes = noham_routes[noham_routes["link_id"].isin(noham_links.index)]

    link_demand = _aggregate_link_flows(
        noham_ods,
        noham_routes,
        noham_links,
    )

    link_demand["link_id"] = link_demand["a"].astype(str) + "_" + link_demand["b"].astype(str)

    link_demand = link_demand[["link_id", "abs_demand"]]
    link_demand = link_demand.rename(
        columns={"abs_demand": f"{user_class}_{time_period}"}
    )  # Rename demand column

    LOG.info(
        "%s ODs, %s Routes and %s Links aggregated to %s link flows",
        len(noham_ods),
        len(noham_routes),
        len(noham_links),
        len(link_demand),
    )

    return link_demand


def _aggregate_link_flows_year(
    config: model_config.Config, year: int, network_link_ids: set[str]
) -> pd.DataFrame:
    """Aggregate link flows for each year, time period, and user class."""
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    ts_dfs = []
    for time_period in NoHAMTimePeriods:
        uc_dfs = []
        for user_class in NoHAMUserClasses:
            uc_dfs.append(
                _process_single_noham_layer(
                    config,
                    year=year,
                    route_links_store=route_links_store,
                    time_period=time_period,
                    user_class=user_class,
                    link_ids=network_link_ids,
                )
            )

        # Merge all user class dataframes
        combined_uc_df = uc_dfs[0]
        for df_uc in uc_dfs[1:]:
            combined_uc_df = combined_uc_df.merge(df_uc, on="link_id", how="outer")

        # Compute total demand for all vehicles for each time period
        combined_uc_df[f"all_vehs_{time_period}"] = combined_uc_df[
            [f"{uc}_{time_period}" for uc in NoHAMUserClasses]
        ].sum(axis=1)

        # Store result
        ts_dfs.append(combined_uc_df)

    # Merge all time period dataframes
    combined_ts_df = ts_dfs[0]
    for df_ts in ts_dfs[1:]:
        combined_ts_df = combined_ts_df.merge(df_ts, on="link_id", how="outer")

    # Compute totals for each user class across all time periods
    for uc in NoHAMUserClasses:
        combined_ts_df[f"{uc}_total"] = combined_ts_df[
            [f"{uc}_{tp}" for tp in NoHAMTimePeriods]
        ].sum(axis=1)

    # Compute total of each user class across all time periods
    combined_ts_df["all_vehs_total"] = combined_ts_df[
        [f"all_vehs_{tp}" for tp in NoHAMTimePeriods]
    ].sum(axis=1)

    return combined_ts_df[
        ["link_id", "all_vehs_total"] + [f"{uc}_total" for uc in NoHAMUserClasses]
    ]
