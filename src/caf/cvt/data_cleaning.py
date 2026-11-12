"""Cleans raw input data to prepare it for input into the model."""

### LOAD LIBRARIES
import logging
import pathlib

import geopandas as gpd
import osbng
import pandas as pd
import shapely

from caf.cvt import (
    definitions,
    demand_cleaning,
    hazard_cleaning,
    infrastructure_cleaning,
    model_config,
)

# LOGGING

LOG = logging.getLogger(__name__)


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

    LOG.debug(
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
    """
    Get the boundary to use for clipping and filtering datasets based on the config.

    Validate the boundary inputs in the config and reads the appropriate boundary file, which
    can either be a bespoke boundary, an STB boundary, or a CA boundary.

    Parameters
    ----------
    config : model_config.Config
        The configuration object containing paths and parameters for the model.

    Returns
    -------
    gpd.GeoDataFrame
        The boundary GeoDataFrame to use for clipping and filtering datasets.
    """
    if config.other_input.boundary_path is not None and config.other_input.boundary_path != "":
        LOG.info("Using specific boundary file: %s", config.other_input.boundary_path)
        return gpd.read_file(config.paths.raw_input / config.other_input.boundary_path)

    if config.parameters.stb is not None:
        LOG.info("Using boundary for STB: %s", config.parameters.stb)
        stb_boundaries = gpd.read_file(config.paths.raw_input / config.other_input.stb_path)
        stb_boundary = stb_boundaries[stb_boundaries["stb_name"] == config.parameters.stb]
        stb_boundary = stb_boundary[["stb_name", definitions.GEOMETRY_COL]]
        if stb_boundary.empty:
            raise ValueError(f"No boundary found for STB: '{config.parameters.stb}'")
        return stb_boundary

    if config.parameters.ca is not None:
        LOG.info("Using boundary for CA: %s", config.parameters.ca)
        ca_boundaries = gpd.read_file(config.paths.raw_input / config.other_input.ca_path)
        ca_boundary = ca_boundaries[ca_boundaries["CAUTH25NM"] == config.parameters.ca]
        ca_boundary = ca_boundary[["CAUTH25NM", definitions.GEOMETRY_COL]]
        if ca_boundary.empty:
            raise ValueError(f"No boundary found for CA: '{config.parameters.ca}'")
        return ca_boundary

    raise ValueError(
        "No valid boundary specified in config. "
        "You must provide one of: "
        "`boundary_path`, `stb`, or `ca`."
    )


def convert_point_to_grid(x: int, y: int, size: int) -> shapely.geometry.Polygon:
    """
    Take a point and convert it to a grid of the given size.

    Converts a point (x, y) into a square grid of the specified size, centered on the point.

    Parameters
    ----------
    x : int
        The x-coordinate of the point.
    y : int
        The y-coordinate of the point.
    size : int
        The size of the grid to create around the point.

    Returns
    -------
    shapely.geometry.Polygon
        A square polygon centered on the point with the specified size.
    """
    return shapely.geometry.Polygon(
        [
            (x - size, y - size),
            (x + size, y - size),
            (x + size, y + size),
            (x - size, y + size),
        ]
    )


def extract_poly_from_geomcollection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract polygons from GeomCollection objects in and turn into new rows.

    Takes a GeoDataFrame and extracts all Polygon and MultiPolygon geometries from any
    GeometryCollection objects, returning a new GeoDataFrame with the extracted polygons
    as new rows.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing geometries to extract polygons from.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame with the extracted polygons as rows.
    """
    rows = []
    geometry_collection_count = 0
    polygon_count = 0
    multipolygon_count = 0

    for _idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # If it's already Polygon or MultiPolygon, keep as is
        if isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            rows.append(row)
        # If it's a GeometryCollection, extract polygons
        elif isinstance(geom, shapely.geometry.GeometryCollection):
            geometry_collection_count += 1
            for sub_geom in geom.geoms:
                if isinstance(sub_geom, shapely.geometry.Polygon):
                    polygon_count += 1
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    rows.append(new_row)
                elif isinstance(sub_geom, shapely.geometry.MultiPolygon):
                    multipolygon_count += 1
                    new_row = row.copy()
                    new_row.geometry = sub_geom
                    rows.append(new_row)

        else:
            raise TypeError(
                "Unexpected geometry type when extracting Polygons from GeometryCollections: "
                f"{geom}"
            )

    LOG.debug(
        "Extracted %s Polygons and %s MultiPolygons from %s GeometryCollections",
        polygon_count,
        multipolygon_count,
        geometry_collection_count,
    )

    # Create new GeoDataFrame from expanded rows
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def get_bng_codes(boundary: gpd.GeoDataFrame) -> list[str]:
    """
    Clip the 100km BNG to the boundary.

    Extracts the 100km BNG codes that intersect with the given boundary using the osbng
    package. These codes are used by the input flooding data.

    Parameters
    ----------
    boundary : gpd.GeoDataFrame
        The boundary to clip the 100km BNG grid to.

    Returns
    -------
    list[str]
        A list of 100km BNG codes that intersect with the boundary.
    """
    bng_100km = gpd.GeoDataFrame.from_features(
        osbng.grids.bng_grid_100km, crs=definitions.BNG_CRS
    )
    boundary_bng = clip_to_boundary(bng_100km, boundary)
    return list(boundary_bng["bng_ref"])


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate geometries in a GeoDataFrame, removing invalid ones.

    Takes a GeoDataFrame and removes any empty or invalid geometries, logging
    a warning message for any removed geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to validate.

    Returns
    -------
    gpd.GeoDataFrame
        The validated GeoDataFrame with empty and invalid geometries removed.

    Raises
    ------
    ValueError
        If all geometries are empty.
    """
    if gdf.geometry.is_empty.sum() == len(gdf):
        raise ValueError("All geometries are empty, returning empty GeoDataFrame.")
    if gdf.geometry.is_empty.sum() > 0:
        LOG.debug(
            "Found %s empty geometries out of %s, removing them.",
            gdf.geometry.is_empty.sum(),
            len(gdf),
        )
        gdf = gdf[~gdf.geometry.is_empty]
    if gdf.geometry.notna().sum() < len(gdf):
        LOG.warning(
            "Found %s invalid geometries, removing them.",
            len(gdf) - gdf.geometry.notna().sum(),
        )
        gdf = gdf[gdf.geometry.notna()]
    return gdf


def nearest_centroids(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Take two GeoDataFrames and merge them on their nearest centroids.

    Finds the centroids of geometries in both GeoDataFrames and performs a nearest spatial join
    to associate each geometry in gdf1 with the nearest geometry in gdf2. Returns the merge of
    gdf1 with the nearest geometries from gdf2.

    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        The first GeoDataFrame.
    gdf2 : gpd.GeoDataFrame
        The second GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        The merged GeoDataFrame with geometries from gdf1 and the nearest geometries from gdf2.
    """
    # Ensure both GeoDataFrames are in the same projected CRS
    gdf1 = gdf1.to_crs(definitions.BNG_CRS)
    gdf2 = gdf2.to_crs(definitions.BNG_CRS)

    # Convert both to centroids
    gdf1_centroids = gdf1.copy()
    gdf1_centroids[definitions.GEOMETRY_COL] = gdf1_centroids.geometry.centroid
    gdf2_centroids = gdf2.copy()
    gdf2_centroids[definitions.GEOMETRY_COL] = gdf2_centroids.geometry.centroid

    nearest = gpd.sjoin_nearest(gdf1_centroids, gdf2_centroids, how="left")

    # Merge back with original gdf1 to restore original geometry
    return gdf1.merge(
        nearest.drop(columns=definitions.GEOMETRY_COL), left_index=True, right_index=True
    )


def normalise_user_class_label(value: str | int | float) -> str:
    """
    Normalise a user class label to the format 'ucX', where X is an integer.

    Parameters
    ----------
    value : str | int | float
        The user class value to normalise.

    Returns
    -------
    str
        The normalised user class label in the format 'ucX'.
    """
    value_str = str(value).strip().lower()
    value_str = value_str.removeprefix("uc")
    try:
        return f"uc{int(float(value_str))}"
    except ValueError as exc:
        raise ValueError(f"Invalid model road userclass value '{value}'.") from exc


def get_boundary_name(config: model_config.Config) -> str:
    """
    Get the name of the boundary area from the config.

    Parameters
    ----------
    config : Config
        Main config for the model, containing boundary information.

    Returns
    -------
    str
        The name of the boundary area, either the STB, CA, or "model area" if neither is
        specified.
    """
    if config.parameters.stb:
        return config.parameters.stb
    if config.parameters.ca:
        return config.parameters.ca
    return "model area"


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

    infrastructure_cleaning.clean_infrastructure(config, boundary)
    hazard_cleaning.clean_hazards(config, boundary)
    demand_cleaning.clean_demand(config, boundary)
