"""Apply functional rules to hazard and impact datasets, and normalise."""

### LOAD LIBRARIES
import logging
import pathlib
from functools import reduce

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xyzservices
from shapely.geometry import Polygon, box

from caf.cvt import data_cleaning, file_paths, model_config

LOG = logging.getLogger(__name__)

SCENARIO_NAMES = ["current", "forecast"]
_SCENARIO_SUFFIXES = ["_current", "_forecast"]

_EXTREME_HEAT_RISK_THRESHOLD = 30
_EXTREME_HEAT_WEIGHTS = {
    "max_temp_summer_risk": 0.5,
    "hot_summer_days": 0.25,
    "extreme_summer_days": 0.25,
}

_EXTREME_COLD_RISK_THRESHOLD = 0
_EXTREME_COLD_WEIGHTS = {"min_temp_winter_risk": 0.5, "frost_days": 0.25, "icing_days": 0.25}

_WIND_SPEED_RISK_THRESHOLD_LOWER = 13.4  # 30 mph in m/s (should not exceed upper threshold)
_WIND_SPEED_RISK_THRESHOLD_UPPER = 20.1  # 45 mph in m/s (should not exceed 25)
_EXTREME_WIND_MAX = 25

_DROUGHT_NEAREST_JOIN_MAX_DISTANCE = 10000
_DROUGHT_WEIGHTS = {"drought_severity_index": 0.75, "precip_summer": 0.25}

_STORM_NEAREST_JOIN_MAX_DISTANCE = 5000
_STORM_WEIGHTS = {
    "wind_speed_risk": 0.3,
    "avg_exceedance_days": 0.2,
    "precip_winter": 0.15,
    "10mm_rain_days": 0.15,
    "wind_driven_rain_index": 0.2,
}

_EXTREME_WEATHER_NEAREST_JOIN_MAX_DISTANCE = 10000
_EXTREME_WEATHER_WEIGHTS = {
    "heat_risk": 0.25,
    "cold_risk": 0.25,
    "drought_risk": 0.25,
    "storm_risk": 0.25,
}

_GROUND_STABILITY_NEAREST_JOIN_MAX_DISTANCE = 1000
_GROUND_STABILITY_RISK_SCORE_MAP = {  # Map risk scores to normalised values (0-100)
    "Probable": 100,
    "Possible": 66,
    "Improbable": 33,
    "Unavailable": 50,  # Assign neutral value
}
_GEOSURE_HAZARDS = [
    "collapsible_deposits",
    "compressible_ground",
    "landslides",
    "running_sand",
    "shrink_swell",
    "soluble_rocks",
    "shrink_swell_geoclimate",
]
_GEOCLIMATE_YEAR_SCENARIO_MAP = {"2030": "current", "2070": "forecast"}
_GROUND_STABILITY_WEIGHTS = {
    "shrink_swell_geoclimate_risk": 0.40,
    "landslides_risk": 0.10,
    "shrink_swell_risk": 0.10,
    "compressible_ground_risk": 0.10,
    "collapsible_deposits_risk": 0.10,
    "running_sand_risk": 0.10,
    "soluble_rocks_risk": 0.10,
}

_COASTAL_EROSION_NEAREST_JOIN_MAX_DISTANCE = 500
_COASTAL_EROSION_YEAR_SCENARIO_MAP = {"2055": "current", "2105": "forecast"}
_COASTAL_EROSION_WEIGHTS = {"erosion_risk": 0.9, "giz_risk": 0.1}

_FLOOD_GRID_SIZE_M = 1000
_FLOOD_TILE_SIZE_M = 10000
_FLOOD_RISK_SCORE_MAP = {"Unavailable": 0, "Very low": 0, "Low": 1, "Medium": 2, "High": 3}
_FLOOD_WEIGHTS = {"rivers_sea_flood_risk": 0.5, "surface_water_flood_risk": 0.5}


### GENERAL FUNCTIONS


def min_max_scaling_pair(
    risk_data: pd.DataFrame,
    pairs: list[tuple[str, str]],
    feature_range: tuple[int, int] = (0, 100),
) -> pd.DataFrame:
    """Scale paired columns jointly using Min-Max scaling.

    For each tuple (col_current, col_forecast) in `pairs`, this function computes a
    single minimum and maximum across the combined values of both columns and
    applies a shared `sklearn.preprocessing.MinMaxScaler` with the provided
    `feature_range`. This ensures the two columns in each pair are scaled
    using the same mapping so they are directly comparable.

    Parameters
    ----------
        risk_data: pd.DataFrame
            DataFrame containing the columns to be scaled.
        pairs: list[Tuple[str, str]])
            List of 2-tuples of column names. Each tuple specifies a pair of
            columns that will share a single scaler.
        feature_range: Tuple[int, int], optional
             Desired range of the transformed data. Defaults to ``(0, 100)``.

    Returns
    -------
    pd.DataFrame:
        The original DataFrame with the specified columns scaled in-place.
    """
    for col_current, col_forecast in pairs:
        # Combine both columns into one array for global min/max
        combined_values = (
            risk_data[[col_current, col_forecast]].to_numpy().flatten().reshape(-1, 1)
        )

        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range)
        scaler.fit(combined_values)

        # Transform each column using the same scaler
        risk_data[col_current] = scaler.transform(risk_data[[col_current]].values)
        risk_data[col_forecast] = scaler.transform(risk_data[[col_forecast]].values)

    return risk_data


def _spatial_infill_na_grids(
    risk_grid: gpd.GeoDataFrame, variables: list[str]
) -> gpd.GeoDataFrame:
    """Apply spatial infilling to GeoDataFrame on given variables."""
    neighbours = gpd.sjoin(
        risk_grid, risk_grid, how="left", predicate="touches"
    )  # Find neighbouring grids

    # Calculate the average value of the neighbouring grids
    neighbours_avg = neighbours.groupby(neighbours.index)[
        [f"{var}_right" for var in variables]
    ].mean()

    for var in variables:
        # Set condition as variable is NA
        na_condition = risk_grid[var].isna()

        # Fill with neighbour average
        risk_grid.loc[na_condition, var] = risk_grid.loc[na_condition].index.map(
            neighbours_avg[f"{var}_right"]
        )

    return risk_grid


def _nearest_join_infilling(
    risk_grid: gpd.GeoDataFrame,
    variables: list[str],
    max_distance: int,
    prev_na_count: int,
) -> gpd.GeoDataFrame:
    """Fill remaining NA values using nearest-join spatial infilling."""
    remaining_na = risk_grid[risk_grid[variables].isna().any(axis=1)]
    if remaining_na.empty:
        return risk_grid

    nearest = gpd.sjoin_nearest(
        remaining_na,
        risk_grid.drop(remaining_na.index),
        how="left",
        max_distance=max_distance,
    )

    # Calculate the average value of the neighbouring grids
    nearest_avg = nearest.groupby(nearest.index)[[f"{var}_right" for var in variables]].mean()

    for var in variables:
        na_condition = risk_grid[var].isna()
        risk_grid.loc[na_condition, var] = risk_grid.loc[na_condition].index.map(
            nearest_avg[f"{var}_right"]
        )

    final_remaining = risk_grid[variables].isna().sum().sum()
    filled_nearest = prev_na_count - final_remaining

    LOG.info(
        "Nearest-join infilling with a %sm max distance filled %s NA values; %s remain",
        int(max_distance),
        int(filled_nearest),
        int(final_remaining),
    )

    return risk_grid


def _iterative_spatial_infilling(
    risk_grid: gpd.GeoDataFrame,
    variables: list[str],
    max_iterations: int = 10,
    nearest_join_max_distance: int = 5000,
) -> gpd.GeoDataFrame:
    """Apply spatial infilling iteratively to GeoDataFrame on given variables."""
    prev_na_count = None
    total_na_count = risk_grid[variables].isna().sum().sum()
    LOG.info("Spatial infilling %s NA values.", total_na_count)

    for i in range(max_iterations):
        # Count current NA values
        current_na_count = risk_grid[variables].isna().sum().sum()

        # Stop if all filled
        if current_na_count == 0:
            LOG.info("All NA values filled after %s iterations.", i)
            return risk_grid

        # Stop if no improvement
        if prev_na_count is not None and current_na_count == prev_na_count:
            LOG.info(
                "No further improvement after %s iterations using spatial infilling. "
                "Switching to nearest join to fill remaining %s NA values.",
                i,
                current_na_count,
            )
            break

        prev_na_count = current_na_count

        risk_grid = _spatial_infill_na_grids(risk_grid, variables)

        new_na_count = risk_grid[variables].isna().sum().sum()
        filled_this_iter = prev_na_count - new_na_count
        LOG.info(
            "Iteration %s: filled %s NA values (%s remaining)",
            i + 1,
            int(filled_this_iter),
            int(new_na_count),
        )

    # Fallback: nearest join for remaining NAs
    return _nearest_join_infilling(
        risk_grid=risk_grid,
        variables=variables,
        max_distance=nearest_join_max_distance,
        prev_na_count=prev_na_count or 0,
    )


def _make_grid_cell(xmin: float, ymin: float, i: int, j: int, cell_size: int) -> Polygon:
    """Create a single grid cell polygon."""
    x0 = xmin + i * cell_size
    y0 = ymin + j * cell_size
    return box(x0, y0, x0 + cell_size, y0 + cell_size)


def _create_grid(
    xmin: float, ymin: float, xmax: float, ymax: float, cell_size: int
) -> gpd.GeoDataFrame:
    """Take bounds and a cell size and return a grid of the given size within the bounds."""
    rows = int(np.ceil((ymax - ymin) / cell_size))
    cols = int(np.ceil((xmax - xmin) / cell_size))

    grid_cells = [
        _make_grid_cell(xmin, ymin, i, j, cell_size) for i in range(cols) for j in range(rows)
    ]
    grid_ids = list(range(len(grid_cells)))
    return gpd.GeoDataFrame(
        {"grid_id": grid_ids, "geometry": grid_cells}, crs=data_cleaning.BNG_CRS
    )


def _merge_on_key(
    df_list: list[pd.DataFrame], grid: gpd.GeoDataFrame, key: str
) -> gpd.GeoDataFrame:
    """Merge dataframes into single dataframe on common key, then merge onto a common grid."""
    # Check whether the dataframes list is empty:
    if not df_list:
        raise ValueError("dfs must contain at least one DataFrame")
    # Check whether the merge key is present in the DataFrames
    missing = [i for i, df in enumerate(df_list) if key not in df.columns]
    if missing:
        raise KeyError(f"Merge key '{key}' missing in dfs at positions {missing}.")
    # Check whether the merge key is present in the grid to merge onto
    if key not in grid.columns:
        raise KeyError(f"Merge key '{key}' missing in grid.")
    merged = reduce(lambda left, right: left.merge(right, on=key, how="outer"), df_list)
    merged_df = merged.merge(grid, on=key, how="left", validate="one_to_many")
    return gpd.GeoDataFrame(merged_df, geometry="geometry", crs=grid.crs)


def _calculate_risk_threshold(
    risk_data: pd.DataFrame,
    base_col: str,
    output_col: str,
    threshold: int,
    invert: bool = False,
) -> pd.DataFrame:
    """Calculate risk level of a given column based on a threshold."""
    for scenario in SCENARIO_NAMES:
        col_name = f"{base_col}_{scenario}"
        out_name = f"{output_col}_{scenario}"

        if invert:
            risk_data[out_name] = np.where(
                risk_data[col_name] > threshold, 0, threshold - risk_data[col_name]
            )
        else:
            risk_data[out_name] = np.where(
                risk_data[col_name] < threshold, 0, risk_data[col_name] - threshold
            )

    return risk_data


def _calculate_composite_score(
    risk_data: pd.DataFrame, weights: dict[str, float], output_col: str
) -> pd.DataFrame:
    """Calculate composite score given a dataframe with variables and corresponding weights."""
    for scenario in SCENARIO_NAMES:
        risk_data[f"{output_col}_{scenario}"] = sum(
            risk_data[f"{col}_{scenario}"] * weight for col, weight in weights.items()
        )

    return risk_data


def _overlay_and_clean(
    *hazard_layers: gpd.GeoDataFrame,
    target_crs: str,
    how: str = "union",
) -> gpd.GeoDataFrame:
    """Overlay any number of GeoDataFrames with a union (default), then clean to polygons."""
    clean_layers: list[gpd.GeoDataFrame] = []
    for i, hazard_data in enumerate(hazard_layers):
        if hazard_data is None or len(hazard_data) == 0:
            continue
        if hazard_data.crs is None:
            raise ValueError(f"Layer {i} has no CRS. Set one first.")
        if hazard_data.crs != target_crs:
            hazard_data_clean = hazard_data.to_crs(target_crs)
        else:
            hazard_data_clean = hazard_data

        clean_layers.append(hazard_data_clean)

    if not clean_layers:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    # Sequential union overlay (reduce) keeping all geometry data
    hazard_overlay = clean_layers[0]
    for clean_hazard in clean_layers[1:]:
        hazard_overlay = gpd.overlay(
            hazard_overlay, clean_hazard, how=how, keep_geom_type=False
        )

        # Drop non-area geometries
        geom_types = hazard_overlay.geometry.type
        num_points = (geom_types == "Point").sum() + (geom_types == "MultiPoint").sum()
        num_lines = (geom_types == "LineString").sum() + (
            geom_types == "MultiLineString"
        ).sum()
        hazard_overlay = hazard_overlay[
            geom_types.isin(["Polygon", "MultiPolygon", "GeometryCollection"])
        ].reset_index(drop=True)
        LOG.info(
            "Overlay dropped %s points and %s lines, and kept %s area geometries",
            int(num_points),
            int(num_lines),
            len(hazard_overlay),
        )

        # Standardise to polygons
        hazard_overlay = data_cleaning.explode_to_polygons(hazard_overlay)

        # Ensure correct CRS
        hazard_overlay = hazard_overlay.set_crs(target_crs, allow_override=True)

    return hazard_overlay


def _plot_choropleth_current_and_forecast(
        risk_data: gpd.GeoDataFrame,
        column: str,
        title: str,
        out_path: pathlib.Path,
        cmap: str = "Reds",
        linewidth: float = 0.1,
        basemap_source: xyzservices.TileProvider | None = ctx.providers.CartoDB.Positron,
    ) -> None:
    """Plot a choropleth map of the given column in the risk data."""
    _fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    if basemap_source is not None:
        risk_data = risk_data.to_crs(epsg=3857)
        ctx.add_basemap(ax, source=basemap_source)

    risk_data.plot(
        column=f"{column}_current",
        cmap=cmap,
        linewidth=linewidth,
        ax=ax[0],
        edgecolor="black",
        legend=True
    )

    risk_data.plot(
        column=f"{column}_forecast",
        cmap=cmap,
        linewidth=linewidth,
        ax=ax[1],
        edgecolor="black",
        legend=True
    )

    ax[0].set_title(f"{title} (Current)")
    ax[1].set_title(f"{title} (Forecast)")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _validate_index(
        index: gpd.GeoDataFrame, index_vars: list[str]) -> None:
    """Validate a given index."""
    for scenario in SCENARIO_NAMES:
        for var in index_vars:
            col = f"{var}_{scenario}"
            if col not in index.columns:
                raise ValueError(f"Missing column: {col}")
            if not index[f"{var}_{scenario}"].between(0, 100).all():
                raise ValueError(
                    f"{var.replace('_', ' ').title()} for {scenario} "
                    f"contains values outside 0-100."
                )

    if index.isna().any().any():
        raise ValueError(
            "Index contains NA values."
        )


def _audit_index(
        index: gpd.GeoDataFrame,
        index_vars: list[str],
        out_path: pathlib.Path,
        cmap: str = "Reds"
    ) -> None:
    """Audit a given index."""
    out_path.mkdir(parents=True, exist_ok=True)

    for var in index_vars:
        # Plot Choropleth Maps for each variable
        _plot_choropleth_current_and_forecast(
            index,
            var,
            f"{var.replace('_', ' ').title()}",
            out_path / f"{var}_choropleth.png",
            cmap=cmap,
        )


# FUNCTIONAL RULES


def apply_functional_rules(config: model_config.Config) -> None:
    """
    Apply functional rules to hazard datasets and spatially combine to generate risk indices.

    Create extreme weather, flooding, ground stability, and coastal erosion risk indices by
    combining spatially, either on a shared grid or via spatial overlay. Uses functional rules
    to normalise and classify risk levels.

    Parameters
    ----------
    config : Config
        Main config for the model, containing paths and settings.
    """
    boundary = data_cleaning.get_boundary(config)
    audit_path = config.paths.audit_path / "Functional Rules"

    _extreme_weather_index(config, audit_path)
    if config.switches.flood_overlay_direct:
        _flooding_index_direct(config, boundary, audit_path)
    else:
        _flooding_index(config, boundary, audit_path)
    _ground_stability_index(config, audit_path)
    _coastal_erosion_index(config, audit_path)


## HAZARDS

### EXTREME WEATHER


def _extreme_weather_index(config: model_config.Config, audit_path: pathlib.Path) -> None:
    """Combine extreme heat, extreme cold, drought and storm indexes into a single index."""
    LOG.info("Calculating extreme weather risk index...")
    hazard_grid = gpd.read_file(
        config.paths.model_input / file_paths.HAZARD_GRID_MODEL_INPUT_PATH
    )

    extreme_heat = _extreme_heat_index(config, hazard_grid, audit_path)
    extreme_cold = _extreme_cold_index(config, hazard_grid, audit_path)
    drought = _drought_index(config, hazard_grid, audit_path)
    storm = _storm_index(config, hazard_grid, audit_path)

    LOG.info("Combining extreme heat, extreme cold, drought and storm indexes.")
    extreme_heat_cold = extreme_heat[
        ["grid_id", "part", "heat_risk_current", "heat_risk_forecast"]
    ].merge(
        extreme_cold[
            ["grid_id", "part", "cold_risk_current", "cold_risk_forecast", "geometry"]
        ],
        on=["grid_id", "part"],
        how="inner",
    )

    extreme_heat_cold = gpd.GeoDataFrame(
        extreme_heat_cold, geometry="geometry", crs="EPSG:3857"
    )
    extreme_heat_cold = extreme_heat_cold.drop(columns=["grid_id", "part"])

    extreme_weather_risk = _overlay_and_clean(
        extreme_heat_cold,
        drought[["drought_risk_current", "drought_risk_forecast", "geometry"]],
        storm[["storm_risk_current", "storm_risk_forecast", "geometry"]],
        target_crs=data_cleaning.BNG_CRS,
    )

    extreme_weather_risk = _iterative_spatial_infilling(
        extreme_weather_risk,
        [
            "heat_risk_current",
            "heat_risk_forecast",
            "cold_risk_current",
            "cold_risk_forecast",
            "drought_risk_current",
            "drought_risk_forecast",
            "storm_risk_current",
            "storm_risk_forecast",
        ],
        _EXTREME_WEATHER_NEAREST_JOIN_MAX_DISTANCE,
    )

    extreme_weather_risk = _calculate_composite_score(
        extreme_weather_risk,
        _EXTREME_WEATHER_WEIGHTS,
        "extreme_weather_risk",
    )

    extreme_weather_risk = min_max_scaling_pair(
        extreme_weather_risk,
        [("extreme_weather_risk_current", "extreme_weather_risk_forecast")],
    )

    extreme_weather_risk = gpd.GeoDataFrame(extreme_weather_risk, geometry="geometry")

    _validate_index(
        extreme_weather_risk,
        ["heat_risk", "cold_risk", "drought_risk", "storm_risk", "extreme_weather_risk"],
    )

    _audit_index(
        extreme_weather_risk,
        ["heat_risk", "cold_risk", "drought_risk", "storm_risk", "extreme_weather_risk"],
        audit_path / "Extreme Weather Risk Index"
    )

    data_cleaning.write_to_file(
        extreme_weather_risk,
        config.paths.model_interim_output
        / file_paths.EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Extreme weather risk index calculation complete.")


#### EXTREME HEAT


def _extreme_heat_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame, audit_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Combine several datasets into extreme heat index by merging on their hazard grid."""
    LOG.info("Calculating extreme heat index...")
    temp_max = pd.read_csv(config.paths.model_input / file_paths.TEMP_MAX_MODEL_INPUT_PATH)
    hsd = pd.read_csv(
        config.paths.model_input / file_paths.HOT_SUMMER_DAYS_MODEL_INPUT_PATH
    )
    esd = pd.read_csv(
        config.paths.model_input / file_paths.EXTREME_SUMMER_DAYS_MODEL_INPUT_PATH
    )

    extreme_heat = _merge_on_key([temp_max, hsd, esd], hazard_grid, "grid_id")

    extreme_heat = _calculate_risk_threshold(
        extreme_heat,
        "max_temp_summer",
        "max_temp_summer_risk",
        _EXTREME_HEAT_RISK_THRESHOLD,
    )

    extreme_heat = min_max_scaling_pair(
        extreme_heat,
        [
            ("max_temp_summer_risk_current", "max_temp_summer_risk_forecast"),
            ("hot_summer_days_current", "hot_summer_days_forecast"),
            ("extreme_summer_days_current", "extreme_summer_days_forecast"),
        ],
    )

    extreme_heat = _calculate_composite_score(
        extreme_heat,
        _EXTREME_HEAT_WEIGHTS,
        "heat_risk",
    )

    extreme_heat = min_max_scaling_pair(
        extreme_heat, [("heat_risk_current", "heat_risk_forecast")]
    )

    LOG.info("Extreme heat index calculation complete.")
    extreme_heat = gpd.GeoDataFrame(extreme_heat, geometry="geometry")

    _validate_index(
        extreme_heat,
        ["max_temp_summer_risk", "hot_summer_days", "extreme_summer_days", "heat_risk"],
    )

    _audit_index(
        extreme_heat,
        ["max_temp_summer_risk", "hot_summer_days", "extreme_summer_days", "heat_risk"],
        audit_path / "Extreme Heat Index",
        cmap="Reds"
    )

    return extreme_heat


#### EXTREME COLD


def _extreme_cold_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame, audit_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Combine several datasets into extreme cold index by merging on their hazard grid."""
    LOG.info("Calculating extreme cold index...")
    temp_min = pd.read_csv(config.paths.model_input / file_paths.TEMP_MIN_MODEL_INPUT_PATH)
    frost = pd.read_csv(config.paths.model_input / file_paths.FROST_DAYS_MODEL_INPUT_PATH)
    icing = pd.read_csv(config.paths.model_input / file_paths.ICING_DAYS_MODEL_INPUT_PATH)

    extreme_cold = _merge_on_key(
        [temp_min, frost, icing], hazard_grid, "grid_id"
    )

    extreme_cold = _calculate_risk_threshold(
        extreme_cold,
        "min_temp_winter",
        "min_temp_winter_risk",
        _EXTREME_COLD_RISK_THRESHOLD,
        invert=True,
    )

    extreme_cold = min_max_scaling_pair(
        extreme_cold,
        [
            ("min_temp_winter_risk_current", "min_temp_winter_risk_forecast"),
            ("frost_days_current", "frost_days_forecast"),
            ("icing_days_current", "icing_days_forecast"),
        ],
    )

    extreme_cold = _calculate_composite_score(
        extreme_cold,
        _EXTREME_COLD_WEIGHTS,
        "cold_risk",
    )

    extreme_cold = min_max_scaling_pair(
        extreme_cold, [("cold_risk_current", "cold_risk_forecast")]
    )

    extreme_cold = gpd.GeoDataFrame(extreme_cold, geometry="geometry")

    _validate_index(
        extreme_cold,
        ["min_temp_winter_risk", "frost_days", "icing_days", "cold_risk"],
    )

    _audit_index(
        extreme_cold,
        ["min_temp_winter_risk", "frost_days", "icing_days", "cold_risk"],
        audit_path / "Extreme Cold Index",
        cmap="Blues"
    )

    LOG.info("Extreme cold index calculation complete.")
    return extreme_cold


#### DROUGHT


def _drought_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame, audit_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Combine several datasets into single drought index with spatial overlay."""
    LOG.info("Calculating drought index...")
    drought = gpd.read_file(
        config.paths.model_input / file_paths.DROUGHT_INDEX_MODEL_INPUT_PATH
    )
    precip_sum = pd.read_csv(
        config.paths.model_input / file_paths.SUMMER_PRECIP_MODEL_INPUT_PATH
    )

    precip_sum_grid = precip_sum.merge(hazard_grid, on="grid_id")
    precip_sum_gdf = gpd.GeoDataFrame(
        precip_sum_grid, geometry="geometry", crs=hazard_grid.crs
    )
    precip_sum_gdf = precip_sum_gdf[
        ["precip_summer_current", "precip_summer_forecast", "geometry"]
    ]

    drought_risk = _overlay_and_clean(
        precip_sum_gdf, drought, target_crs=data_cleaning.BNG_CRS
    )

    drought_risk = _iterative_spatial_infilling(
        drought_risk,
        [
            "precip_summer_current",
            "precip_summer_forecast",
            "drought_severity_index_current",
            "drought_severity_index_forecast",
        ],
        _DROUGHT_NEAREST_JOIN_MAX_DISTANCE,
    )

    drought_risk = drought_risk[
        [
            "drought_severity_index_current",
            "drought_severity_index_forecast",
            "precip_summer_current",
            "precip_summer_forecast",
            "geometry",
        ]
    ]

    drought_risk = min_max_scaling_pair(
        drought_risk,
        [
            ("drought_severity_index_current", "drought_severity_index_forecast"),
            ("precip_summer_current", "precip_summer_forecast"),
        ],
    )

    # Reverse the polarity for precipitation
    drought_risk["precip_summer_current"] = 100 - drought_risk["precip_summer_current"]
    drought_risk["precip_summer_forecast"] = (
        100 - drought_risk["precip_summer_forecast"]
    )

    drought_risk = _calculate_composite_score(
        drought_risk,
        _DROUGHT_WEIGHTS,
        "drought_risk",
    )

    drought_risk = min_max_scaling_pair(
        drought_risk, [("drought_risk_current", "drought_risk_forecast")]
    )

    drought_risk = gpd.GeoDataFrame(drought_risk, geometry="geometry")

    _validate_index(
        drought_risk,
        ["drought_severity_index", "precip_summer", "drought_risk"],
    )

    _audit_index(
        drought_risk,
        ["drought_severity_index", "precip_summer", "drought_risk"],
        audit_path / "Drought Risk Index",
        cmap="Oranges"
    )

    LOG.info("Drought index calculation complete.")
    return drought_risk


#### STORMS


def _storm_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame, audit_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Combine several datasets into a single storm index with a spatial overlay."""
    LOG.info("Calculating storm index...")
    precip_win = pd.read_csv(
        config.paths.model_input / file_paths.WINTER_PRECIP_MODEL_INPUT_PATH
    )
    rain_days = gpd.read_file(
        config.paths.model_input / file_paths.RAIN_DAYS_MODEL_INPUT_PATH
    )
    wind_spd = gpd.read_file(
        config.paths.model_input / file_paths.WIND_SPEED_MODEL_INPUT_PATH
    )
    wdr = gpd.read_file(
        config.paths.model_input / file_paths.WIND_DRIVEN_RAIN_MODEL_INPUT_PATH
    )

    precip_win_grid = precip_win.merge(
        hazard_grid, on="grid_id", how="left", validate="one_to_many"
    )
    precip_win_gdf = gpd.GeoDataFrame(
        precip_win_grid, geometry="geometry", crs=hazard_grid.crs
    )
    precip_win_gdf = precip_win_gdf[
        ["precip_winter_current", "precip_winter_forecast", "geometry"]
    ]

    storm_risk = _overlay_and_clean(
        wind_spd,
        rain_days,
        precip_win_gdf,
        wdr,
        target_crs=data_cleaning.BNG_CRS,
    )

    storm_risk = storm_risk[
        [
            "10mm_rain_days_current",
            "precip_winter_current",
            "precip_winter_forecast",
            "wind_speed_99th_percentile_current",
            "wind_speed_99th_percentile_forecast",
            "avg_exceedance_days_current",
            "avg_exceedance_days_forecast",
            "wind_driven_rain_index_current",
            "wind_driven_rain_index_forecast",
            "geometry",
        ]
    ]

    storm_risk = _iterative_spatial_infilling(
        storm_risk,
        [
            "10mm_rain_days_current",
            "precip_winter_current",
            "precip_winter_forecast",
            "wind_speed_99th_percentile_current",
            "wind_speed_99th_percentile_forecast",
            "avg_exceedance_days_current",
            "avg_exceedance_days_forecast",
            "wind_driven_rain_index_current",
            "wind_driven_rain_index_forecast",
        ],
        _STORM_NEAREST_JOIN_MAX_DISTANCE,
    )

    storm_risk["wind_speed_risk_current"] = storm_risk[
        "wind_speed_99th_percentile_current"
    ].apply(_wind_risk_scaled)
    storm_risk["wind_speed_risk_forecast"] = storm_risk[
        "wind_speed_99th_percentile_forecast"
    ].apply(_wind_risk_scaled)

    storm_risk = min_max_scaling_pair(
        storm_risk,
        [
            ("wind_speed_risk_current", "wind_speed_risk_forecast"),
            ("precip_winter_current", "precip_winter_forecast"),
            ("avg_exceedance_days_current", "avg_exceedance_days_forecast"),
            ("wind_driven_rain_index_current", "wind_driven_rain_index_forecast"),
        ],
    )

    # Scale rain days on its own, then duplicate
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
    storm_risk["10mm_rain_days_current"] = scaler.fit_transform(
        storm_risk[["10mm_rain_days_current"]]
    )
    storm_risk["10mm_rain_days_forecast"] = storm_risk["10mm_rain_days_current"]

    storm_risk = _calculate_composite_score(
        storm_risk,
        _STORM_WEIGHTS,
        "storm_risk",
    )

    storm_risk = min_max_scaling_pair(
        storm_risk, [("storm_risk_current", "storm_risk_forecast")]
    )

    storm_risk = gpd.GeoDataFrame(storm_risk, geometry="geometry")

    _validate_index(
        storm_risk,
        [
            "10mm_rain_days",
            "precip_winter",
            "avg_exceedance_days",
            "wind_driven_rain_index",
            "wind_speed_risk",
            "storm_risk",
        ],
    )

    _audit_index(
        storm_risk,
        [
            "10mm_rain_days",
            "precip_winter",
            "avg_exceedance_days",
            "wind_driven_rain_index",
            "wind_speed_risk",
            "storm_risk",
        ],
        audit_path / "Storm Risk Index",
        cmap="Blues"
    )

    LOG.info("Storm index calculation complete.")
    return gpd.GeoDataFrame(storm_risk, geometry="geometry")


def _wind_risk_scaled(speed_metres_per_second: float) -> float:
    """Calculate wind risk value given a wind speed, based on classification rule."""
    if speed_metres_per_second < _WIND_SPEED_RISK_THRESHOLD_LOWER:
        return 0
    if speed_metres_per_second <= _WIND_SPEED_RISK_THRESHOLD_UPPER:
        return (speed_metres_per_second - _WIND_SPEED_RISK_THRESHOLD_LOWER) / (
            _WIND_SPEED_RISK_THRESHOLD_UPPER - _WIND_SPEED_RISK_THRESHOLD_LOWER
        )  # Scale to 0 - 1
    return 1 + (speed_metres_per_second - _WIND_SPEED_RISK_THRESHOLD_UPPER) / (
        _EXTREME_WIND_MAX - _WIND_SPEED_RISK_THRESHOLD_UPPER
    )  # Scale beyond 1


### FLOODING


def _flooding_index(
        config: model_config.Config,
        boundary: gpd.GeoDataFrame,
        audit_path: pathlib.Path
    ) -> None:
    """Combine RoFRS & RoFSW into a single risk score by upscaling them to a common grid."""
    LOG.info("Calculating flood risk index...")
    # Ensure grid exists
    if config.switches.create_flood_grid:
        flood_grid = _create_flood_grid(config, _FLOOD_GRID_SIZE_M, boundary)
    else:
        flood_grid = gpd.read_file(
            config.paths.model_interim_output / file_paths.FLOOD_GRID_MODEL_INTERIM_OUTPUT_PATH
        )

    current_flood_scenario_map = [
        (file_paths.FLOOD_RIVERS_SEA_MODEL_INPUT_PATH, "rivers_sea_flood_risk_current"),
        (file_paths.FLOOD_SURFACE_WATER_MODEL_INPUT_PATH, "surface_water_flood_risk_current"),
    ]

    forecast_flood_scenario_map = [
        (
            file_paths.FLOOD_RIVERS_SEA_CLIMATE_CHANGE_MODEL_INPUT_PATH,
            "rivers_sea_flood_risk_forecast",
        ),
        (
            file_paths.FLOOD_SURFACE_WATER_CLIMATE_CHANGE_MODEL_INPUT_PATH,
            "surface_water_flood_risk_forecast",
        ),
    ]

    LOG.info("Processing current flood risk...")
    flood_risk_c = _upscale_to_grid(
        config, flood_grid, current_flood_scenario_map, "current"
    )
    LOG.info("Current flood risk processing complete.")

    LOG.info("Processing forecast flood risk...")
    flood_risk_f = _upscale_to_grid(
        config, flood_grid, forecast_flood_scenario_map, "forecast"
    )
    LOG.info("Forecast flood risk processing complete.")

    # Merge on grid ID
    flood_risk = flood_risk_c.merge(flood_risk_f, on="grid_id", how="left")

    # Merge with grid to get geometries
    flood_risk = flood_risk.merge(flood_grid, on="grid_id", how="left")

    # Convert to GeoDataFrame
    flood_risk = gpd.GeoDataFrame(
        flood_risk, geometry="geometry", crs=data_cleaning.BNG_CRS
    )

    flood_risk = flood_risk.drop(columns=["grid_id"])

    flood_risk = min_max_scaling_pair(
        flood_risk,
        [
            ("rivers_sea_flood_risk_current", "rivers_sea_flood_risk_forecast"),
            ("surface_water_flood_risk_current", "surface_water_flood_risk_forecast"),
        ],
    )

    flood_risk = _calculate_composite_score(
        flood_risk,
        _FLOOD_WEIGHTS,
        "flood_risk",
    )

    flood_risk = min_max_scaling_pair(
        flood_risk, [("flood_risk_current", "flood_risk_forecast")]
    )

    _validate_index(
        flood_risk,
        ["rivers_sea_flood_risk", "surface_water_flood_risk", "flood_risk"],
    )

    _audit_index(
        flood_risk,
        ["rivers_sea_flood_risk", "surface_water_flood_risk", "flood_risk"],
        audit_path / "Flood Risk Index",
        cmap="Blues"
    )

    data_cleaning.write_to_file(
        flood_risk,
        config.paths.model_interim_output / file_paths.FLOOD_RISK_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Flood risk index calculation complete.")


def _create_flood_grid(
    config: model_config.Config, size_m: int, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Create a grid of a given size in metres, within a given boundary."""
    LOG.info("Creating flood grid with cell size %s m2.", size_m)
    xmin, ymin, xmax, ymax = boundary.total_bounds
    grid = _create_grid(xmin, ymin, xmax, ymax, size_m)
    flood_grid = data_cleaning.clip_to_boundary(grid, boundary)
    data_cleaning.write_to_file(
        flood_grid,
        config.paths.model_interim_output / file_paths.FLOOD_GRID_MODEL_INTERIM_OUTPUT_PATH,
    )
    return flood_grid


def _upscale_to_grid(
    config: model_config.Config,
    flood_grid: gpd.GeoDataFrame,
    scenario_map: list[tuple[pathlib.Path, str]],
    scenario: str,
) -> gpd.GeoDataFrame:
    """Upscale each flood layer to the common grid and writes to file."""
    flood_upscaled = {}
    for path, risk_col in scenario_map:
        LOG.info("Upscaling flood layer %s to a %sm grid", risk_col, _FLOOD_GRID_SIZE_M)
        flood_upscaled[risk_col] = _area_weighted_flood_assignment(
            config,
            flood_grid,
            config.paths.model_input / path,
            scenario,
            risk_col,
        )

    flood_risk_scenario = flood_upscaled[scenario_map[0][1]].merge(
        flood_upscaled[scenario_map[1][1]], on="grid_id", how="inner"
    )

    data_cleaning.write_to_file(
        flood_risk_scenario,
        config.paths.model_interim_output
        / file_paths.FLOOD_RISK_SCENARIO_MODEL_INTERIM_OUTPUT_PATH
        / f"flood_risk_{scenario}.csv",
    )

    return flood_risk_scenario


def _area_weighted_flood_assignment(
    config: model_config.Config,
    grid: gpd.GeoDataFrame,
    flood_path: pathlib.Path,
    scenario: str,
    risk_column: str,
) -> gpd.GeoDataFrame:
    """Assign flood risk to grid squares using an area-weighted average."""
    # Perform chunked overlay to get intersections
    flood_intersections, len_before_upscale = _chunked_grid_polygon_flood_overlay(
        config,
        flood_path=flood_path,
        flood_grid=grid,
        scenario=scenario,
        risk_column=risk_column,
    )

    # Compute weighted risk contribution
    flood_intersections["weighted"] = (
        flood_intersections[risk_column] * flood_intersections["area"]
    )

    # Compute aggregated weighted sum per grid cell (total exposure)
    exposure_agg = flood_intersections.groupby("grid_id").agg(weighted_sum=("weighted", "sum"))

    # Assign weighted average flood risk back to the original grid
    flood_result = grid[["grid_id", "geometry"]].copy().set_index("grid_id")
    flood_result["grid_area"] = flood_result.geometry.area
    flood_result[risk_column] = 0.0
    flood_result.loc[exposure_agg.index, risk_column] = (
        exposure_agg["weighted_sum"] / flood_result.loc[exposure_agg.index, "grid_area"]
    )
    flood_result = flood_result.drop(columns=["geometry", "grid_area"])
    flood_result = flood_result.reset_index()

    # Fill missing values with 0 (no risk) since no data means no risk in the underlying data
    num_na_rows = flood_result[risk_column].isna().sum()
    pct_na_rows = (num_na_rows / len(flood_result)) * 100
    flood_result[risk_column] = flood_result[risk_column].fillna(0.0)
    LOG.info(
        "Filled %s NA values (%s percent of data) in flood data column %s with 0.",
        num_na_rows,
        pct_na_rows,
        risk_column,
    )

    len_after_upscale = len(flood_result)
    LOG.info(
        "Upscaled flood layer %s from %s geometries to %s grid cells "
        "using area-weighted average.",
        risk_column,
        len_before_upscale,
        len_after_upscale,
    )

    return flood_result


def _process_flood_grid_tile(
    *,
    flood_path: pathlib.Path,
    grid_square: gpd.GeoSeries,
    risk_column: str,
) -> tuple[pd.DataFrame, int]:
    """Process the flood overlay for a single grid tile."""
    grid_geom = grid_square.geometry
    grid_bbox = grid_geom.bounds

    grid_square_gdf = gpd.GeoDataFrame(
        {"grid_id": [grid_square["grid_id"]]}, geometry=[grid_geom], crs=data_cleaning.BNG_CRS
    )

    flood_layer = gpd.read_file(flood_path, bbox=grid_bbox)
    rows_before = len(flood_layer)

    if flood_layer.empty:
        return pd.DataFrame(), rows_before

    flood_layer[risk_column] = flood_layer[risk_column].map(_FLOOD_RISK_SCORE_MAP)

    flood_chunk = gpd.overlay(flood_layer, grid_square_gdf, how="intersection")

    if flood_chunk.empty:
        return pd.DataFrame, rows_before

    flood_chunk["area"] = flood_chunk.geometry.area
    flood_chunk = flood_chunk.drop(columns=["geometry"])

    return flood_chunk, rows_before


def _chunked_grid_polygon_flood_overlay(
    config: model_config.Config,
    *,
    flood_path: pathlib.Path,
    flood_grid: gpd.GeoDataFrame,
    scenario: str,
    risk_column: str,
    log_every: int = 1000,
) -> gpd.GeoDataFrame:
    """Chunked polygon-grid overlay."""
    # Prepare output csv
    output_path = (
        config.paths.model_interim_output
        / file_paths.FLOOD_RISK_SCENARIO_MODEL_INTERIM_OUTPUT_PATH
        / f"flood_risk_overlay_{scenario}.csv"
    )
    first_write = True

    total_rows_read = 0
    rows_written = 0

    # For each tile, do spatial filtering and run overlay and clean
    for i, grid_square in flood_grid.iterrows():
        if i % log_every == 0:
            LOG.info(
                "Overlay progress: %s/%s grid squares (%.1f%%), "
                "%s intersections written to file",
                i,
                len(flood_grid),
                (i / len(flood_grid) * 100),
                rows_written,
            )

        flood_chunk, rows_before = _process_flood_grid_tile(
            flood_path=flood_path, grid_square=grid_square, risk_column=risk_column
        )

        total_rows_read += rows_before

        # If there are no resulting geometries, skip to the next tile
        if flood_chunk.empty:
            continue

        data_cleaning.write_to_file(
            flood_chunk,
            output_path,
            mode="w" if first_write else "a",
        )
        first_write = False

        rows_written += len(flood_chunk)

    LOG.info("Chunked overlay completed. Output written to %s", output_path)

    return pd.read_csv(output_path), total_rows_read


def _flooding_index_direct(
    config: model_config.Config, boundary: gpd.GeoDataFrame, audit_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Overlay all four flood datasets using a tiled chunking method."""
    LOG.info("Combining all four flood datasets...")

    # If the direct tiled overlay hasn't been done yet, do it.
    if config.switches.compute_flood_overlay:
        _tile_polygon_flood_overlay(
            config,
            boundary,
            [
                config.paths.model_input / file_paths.FLOOD_RIVERS_SEA_MODEL_INPUT_PATH,
                config.paths.model_input
                / file_paths.FLOOD_RIVERS_SEA_CLIMATE_CHANGE_MODEL_INPUT_PATH,
                config.paths.model_input / file_paths.FLOOD_SURFACE_WATER_MODEL_INPUT_PATH,
                config.paths.model_input
                / file_paths.FLOOD_SURFACE_WATER_CLIMATE_CHANGE_MODEL_INPUT_PATH,
            ],
            crs=data_cleaning.BNG_CRS,
            tile_size_m=_FLOOD_TILE_SIZE_M,
        )

    # Read the direct overlay result, and filter to region
    flood_risk = gpd.read_file(
        config.paths.model_interim_output
        / file_paths.FLOOD_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH,
        mask=boundary,
        layer="flood_overlay",
    )

    # Map original risk categories to numeric scores
    for col in ["rivers_sea_flood_risk_current", "rivers_sea_flood_risk_forecast",
                "surface_water_flood_risk_current", "surface_water_flood_risk_forecast"]:
        flood_risk[col] = flood_risk[col].map(_FLOOD_RISK_SCORE_MAP)

    # Fill NA values with 0 (no risk) since no data means no risk in the underlying data
    flood_risk = flood_risk.fillna(0)

    flood_risk = min_max_scaling_pair(
        flood_risk,
        [
            ("rivers_sea_flood_risk_current", "rivers_sea_flood_risk_forecast"),
            ("surface_water_flood_risk_current", "surface_water_flood_risk_forecast"),
        ],
    )

    flood_risk = _calculate_composite_score(
        flood_risk,
        _FLOOD_WEIGHTS,
        "flood_risk",
    )

    flood_risk = min_max_scaling_pair(
        flood_risk, [("flood_risk_current", "flood_risk_forecast")]
    )

    _validate_index(
        flood_risk,
        ["rivers_sea_flood_risk", "surface_water_flood_risk", "flood_risk"],
    )

    _audit_index(
        flood_risk,
        ["rivers_sea_flood_risk", "surface_water_flood_risk", "flood_risk"],
        audit_path / "Flood Risk Index (Direct Overlay)",
        cmap="Blues"
    )

    data_cleaning.write_to_file(
        flood_risk,
        config.paths.model_interim_output
        / file_paths.FLOOD_RISK_DIRECT_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Flood risk index calculation complete.")
    return flood_risk


def _create_flood_tiles(
    config: model_config.Config, boundary: gpd.GeoDataFrame, tile_size_m: int
) -> gpd.GeoDataFrame:
    """Create flood tiles to be used in chunked overlay."""
    xmin, ymin, xmax, ymax = boundary.total_bounds
    tiles = _create_grid(xmin, ymin, xmax, ymax, tile_size_m)
    tiles = tiles[tiles.geometry.intersects(boundary.geometry.iloc[0])].copy()
    tiles = tiles.reset_index(drop=True)
    tiles["tile_id"] = range(len(tiles))
    data_cleaning.write_to_file(
        tiles,
        config.paths.model_interim_output / file_paths.TILE_GRID_MODEL_INTERIM_OUTPUT_PATH,
    )
    return tiles


def _process_flood_overlay_tile(
    *, tile: gpd.GeoSeries, layer_paths: list[pathlib.Path], crs: str
) -> gpd.GeoDataFrame | None:
    """Process flood overlay for a single tile."""
    tile_geom = tile.geometry
    tile_bbox = tile_geom.bounds
    tile_gdf = gpd.GeoDataFrame(geometry=[tile_geom], crs=crs)

    layer_subsets: list[gpd.GeoDataFrame] = []

    for layer_path in layer_paths:
        layer_sub = gpd.read_file(layer_path, bbox=tile_bbox)
        if layer_sub.empty:
            continue

        layer_sub = gpd.overlay(layer_sub, tile_gdf, how="intersection")
        if layer_sub.empty:
            return None

        layer_subsets.append(layer_sub)

    if not layer_subsets:
        return None

    tile_overlay = _overlay_and_clean(*layer_subsets, target_crs=crs, how="union")

    if tile_overlay.empty:
        return None

    return tile_overlay


def _tile_polygon_flood_overlay(
    config: model_config.Config,
    boundary: gpd.GeoDataFrame,
    layer_paths: list[gpd.GeoDataFrame],
    crs: str,
    tile_size_m: int = 5000,
) -> gpd.GeoDataFrame:
    """Chunked polygon-polygon overlay using a tile grid."""
    # Create tiles
    if config.switches.create_flood_tiles:
        tiles = _create_flood_tiles(config, boundary, tile_size_m)
    else:
        tiles = gpd.read_file(
            config.paths.model_interim_output / file_paths.TILE_GRID_MODEL_INTERIM_OUTPUT_PATH
        )

    # Prepare output GPKG
    output_path = (
        config.paths.model_interim_output
        / file_paths.FLOOD_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH
    )
    layer_name = "flood_overlay"
    first_write = True

    # For each tile, do spatial filtering and run overlay and clean
    for tile_idx, tile in tiles.iterrows():
        LOG.info("Tile %s/%s starting overlay", tile_idx + 1, len(tiles))

        tile_overlay = _process_flood_overlay_tile(tile=tile, layer_paths=layer_paths, crs=crs)

        # If there are no resulting geometries, skip to the next tile
        if tile_overlay is None:
            continue

        data_cleaning.write_to_file(
            tile_overlay, output_path, mode="w" if first_write else "a", layer=layer_name
        )
        first_write = False

        LOG.info("Tile %s wrote %s geometries.", tile_idx + 1, len(tile_overlay))

    LOG.info("Chunked overlay completed. Output written to %s", output_path)


### GROUND STABILITY


def _ground_stability_index(config: model_config.Config, audit_path: pathlib.Path) -> None:
    """Combine GeoSure & GeoClimate risk into a single index, using a spatial overlay."""
    LOG.info("Calculating ground stability risk index...")
    geosure = gpd.read_file(config.paths.model_input / file_paths.GEOSURE_MODEL_INPUT_PATH)
    geosure = geosure.to_crs(data_cleaning.BNG_CRS)

    shrink_swell = {}
    ground_stability = {}
    for year, scenario in _GEOCLIMATE_YEAR_SCENARIO_MAP.items():
        shrink_swell[year] = gpd.read_file(
            config.paths.model_input
            / file_paths.GEOCLIMATE_SHRINK_SWELL_MODEL_INPUT_PATH
            / f"bgs_ss_{year}.gpkg"
        )
        shrink_swell[year]["shrink_swell_geoclimate_risk"] = shrink_swell[year][
            "shrink_swell_geoclimate_risk"
        ].map(_GROUND_STABILITY_RISK_SCORE_MAP)
        shrink_swell[year] = shrink_swell[year][["shrink_swell_geoclimate_risk", "geometry"]]
        ground_stability[scenario] = _overlay_and_clean(
            geosure, shrink_swell[year], target_crs=data_cleaning.BNG_CRS
        )
        ground_stability[scenario] = ground_stability[scenario].rename(
            columns={
                col: f"{col}_{scenario}"
                for col in ground_stability[scenario].columns
                if col != "geometry"
            }
        )

    ground_stability = _overlay_and_clean(
        ground_stability["current"],
        ground_stability["forecast"],
        target_crs=data_cleaning.BNG_CRS,
    )

    risk_cols = [
        f"{hazard}_risk{suffix}"
        for hazard in _GEOSURE_HAZARDS
        for suffix in _SCENARIO_SUFFIXES
    ]

    for col in risk_cols:
        ground_stability[col] = pd.to_numeric(ground_stability[col], errors="coerce")

    ground_stability = _iterative_spatial_infilling(
        ground_stability, risk_cols, _GROUND_STABILITY_NEAREST_JOIN_MAX_DISTANCE
    )

    gs_pairs = [(f"{col}_risk_current", f"{col}_risk_forecast") for col in _GEOSURE_HAZARDS]

    ground_stability = min_max_scaling_pair(ground_stability, gs_pairs)

    ground_stability = _calculate_composite_score(
        ground_stability,
        _GROUND_STABILITY_WEIGHTS,
        "ground_stability_risk",
    )

    ground_stability = min_max_scaling_pair(
        ground_stability,
        [("ground_stability_risk_current", "ground_stability_risk_forecast")],
    )

    _validate_index(
        ground_stability,
        [f"{col}_risk" for col in _GEOSURE_HAZARDS] + ["ground_stability_risk"],
    )

    _audit_index(
        ground_stability,
        [f"{col}_risk" for col in _GEOSURE_HAZARDS] + ["ground_stability_risk"],
        audit_path / "Ground Stability Risk Index",
        cmap="Oranges"
    )

    data_cleaning.write_to_file(
        ground_stability,
        config.paths.model_interim_output
        / file_paths.GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Ground stability risk index calculation complete.")


### COASTAL EROSION


def _coastal_erosion_index(config: model_config.Config, audit_path: pathlib.Path) -> None:
    """Combine erosion and ground stability risk into single index using a spatial overlay."""
    LOG.info("Calculating coastal erosion risk index...")
    ncerm_giz = gpd.read_file(
        config.paths.model_input / file_paths.GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH
    )
    ncerm_giz["giz_risk"] = 100

    ncerm = {}
    erosion_risk = {}
    for year, scenario in _COASTAL_EROSION_YEAR_SCENARIO_MAP.items():
        ncerm[year] = gpd.read_file(
            config.paths.model_input
            / file_paths.NCERM_MODEL_INPUT_PATH
            / f"ncerm_smp_{year}_70CC.gpkg"
        )
        if ncerm_giz.empty and ncerm[year].empty:
            LOG.warning(
                "Both NCERM and GIZ layers are empty for scenario %s. " \
                "Coastal erosion risk will be 0 everywhere.",
                scenario,
            )
            if year == "2055":
                continue
            data_cleaning.write_to_file(
                gpd.GeoDataFrame(
                    columns=["coastal_erosion_risk_current", "coastal_erosion_risk_forecast",
                             "geometry"],  # Empty GeoDataFrame
                    geometry="geometry",
                    crs=data_cleaning.BNG_CRS
                ),
                config.paths.model_interim_output
                / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH,
            )
            return

        ncerm[year]["erosion_risk"] = 100

        erosion_risk[scenario] = _overlay_and_clean(
            ncerm_giz, ncerm[year], target_crs=data_cleaning.BNG_CRS
        )

        # Compute composite risk score
        erosion_risk[scenario]["coastal_erosion_risk"] = (
            erosion_risk[scenario]["erosion_risk"]
            * _COASTAL_EROSION_WEIGHTS["erosion_risk"]
            + erosion_risk[scenario]["giz_risk"] * _COASTAL_EROSION_WEIGHTS["giz_risk"]
        )

        erosion_risk[scenario] = erosion_risk[scenario].rename(
            columns={"coastal_erosion_risk": f"coastal_erosion_risk_{scenario}"}
        )

    coastal_erosion_risk = _overlay_and_clean(
        erosion_risk["current"],
        erosion_risk["forecast"],
        target_crs=data_cleaning.BNG_CRS,
    )

    coastal_erosion_risk = _iterative_spatial_infilling(
        coastal_erosion_risk,
        ["coastal_erosion_risk_current", "coastal_erosion_risk_forecast"],
        nearest_join_max_distance=_COASTAL_EROSION_NEAREST_JOIN_MAX_DISTANCE,
    )
    coastal_erosion_risk = gpd.GeoDataFrame(coastal_erosion_risk, geometry="geometry")
    coastal_erosion_risk = coastal_erosion_risk[
        ["coastal_erosion_risk_current", "coastal_erosion_risk_forecast", "geometry"]
    ]

    _validate_index(
        coastal_erosion_risk,
        ["coastal_erosion_risk"],
    )

    _audit_index(
        coastal_erosion_risk,
        ["coastal_erosion_risk"],
        audit_path / "Coastal Erosion Risk Index",
        cmap="Purples"
    )

    data_cleaning.write_to_file(
        coastal_erosion_risk,
        config.paths.model_interim_output
        / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Coastal erosion risk index calculation complete.")
