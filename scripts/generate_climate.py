"""Module for generating climate data using DARe Weather Generator tool."""

import logging
import pathlib
import subprocess
import sys
import tempfile

import caf.toolkit as ctk
import geopandas as gpd
import osbng
import yaml

from caf import cvt

_NAME = pathlib.Path(__file__).stem

# LOGGING

LOG = logging.getLogger(_NAME)


# CONSTANTS

BOUNDARY_NAME = "Liverpool City Region"
BNG_GRID_SIZE_KM = 5

# PATHS

DATA_PATH = pathlib.Path("D:/Climate Vulnerability Tool/Localisation/v3 - NPR/Data")

LOG_PATH = DATA_PATH / "logging"

BOUNDARY_PATH = (
    DATA_PATH / "raw input" / "Other" /
    "Combined_Authorities_December_2025_Boundaries_EN_BFE.gpkg"
)

GRID_PATH = DATA_PATH / "raw input" / "Other" / f"BNG_{BNG_GRID_SIZE_KM}km_grid.gpkg"



# MAIN FUNCTIONS

# 1. Generate grid of appropriate resolution, BNG 5km grid size
# 2. Pass list of grid centroids to the config of the weather generator
# 3. Generate climate data for precipitation, temperature, and other weather variables when
#    they are available
# 4. Save generated climate data to raw input directory

def _generate_climate_data(
        boundary_path: pathlib.Path,
        country: str,
        grid_path: pathlib.Path
) -> None:
    """Generate climate data using the DARe Weather Generator tool."""
    LOG.info("Generating climate data using DARe Weather Generator tool.")
    boundaries = gpd.read_file(boundary_path)
    country_boundary = boundaries[boundaries["CAUTH25NM"] == country]
    _generate_grid_centroids(
        grid_size=BNG_GRID_SIZE_KM,
        boundary=country_boundary,
        output_path=GRID_PATH
    )
    # TEMPORARY READ TO SAVE TIME
    grid = gpd.read_file(grid_path)
    grid_centroids = [
        (int(geom.centroid.x), int(geom.centroid.y))
        for geom in grid.geometry
    ]

    LOG.info("Generated grid centroids.")

    with open("D:/GitHub/WeatherGenerator/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["spatial"]["coordinates"] = grid_centroids

    with tempfile.NamedTemporaryFile(
        suffix=".yaml",
        mode="w",
        delete=False
    ) as f:
        yaml.safe_dump(config, f)
        temp_config = f.name

    LOG.info("Running Weather Generator tool for %s: %s grids.", country, len(grid_centroids))
    subprocess.run(
        [
            sys.executable,
            "D:/GitHub/WeatherGenerator/src/WeatherGeneratorApp/main.py",
            "--config",
            temp_config
        ],
        check=True
    )

    LOG.info("Weather Generator tool run completed.")



def _generate_grid_centroids(
        grid_size: int,
        boundary: gpd.GeoDataFrame,
        output_path: pathlib.Path
    )-> list[tuple[float, float]]:
    """Generate BNG grid centroids with specified grid size."""
    LOG.info("Create BNG centroids with %dkm grid size.", grid_size)
    bbox = boundary.total_bounds
    grid = list(osbng.grids.bbox_to_bng_iterfeatures(*bbox, f"{grid_size}km"))
    grid_gdf = gpd.GeoDataFrame.from_features(grid, crs="EPSG:27700")

    # Filter grid to squares that intersect the boundary
    boundary_geometry = boundary.geometry.iloc[0]
    grid_gdf = grid_gdf[
        grid_gdf.geometry.intersects(boundary_geometry)
    ]

    grid_gdf.to_file(
        output_path,
        driver="GPKG"
    )

    return [(geom.centroid.x, geom.centroid.y) for geom in grid_gdf.geometry]



# RUN SCRIPT

# pylint: disable=duplicate-code
if __name__ == "__main__":
    with ctk.LogHelper(
        _NAME,
        ctk.ToolDetails(
            cvt.__package__,
            cvt.__version__,
        ),
        log_file=LOG_PATH / "weather_generator.log",
    ) as log:
        _generate_climate_data(
            boundary_path=BOUNDARY_PATH,
            country=BOUNDARY_NAME,
            grid_path=GRID_PATH
        )
