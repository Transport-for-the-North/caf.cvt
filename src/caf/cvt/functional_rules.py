"""Apply functional rules to hazard and impact datasets, and normalise."""

### LOAD LIBRARIES
import logging
import pathlib
from functools import reduce

import contextily as ctx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")  # Use non-interactive backend for plotting
import numpy as np
import pandas as pd
import sklearn
import xyzservices
from shapely.geometry import Polygon, box

from caf.cvt import data_cleaning, file_paths, model_config
from caf.cvt.definitions import (
    DroughtCols,
    ExtremeColdCols,
    ExtremeHeatCols,
    ExtremeWeatherCols,
    FloodingCols,
    GroundStabilityCols,
    MainHazardCols,
    Scenarios,
    StormCols,
)

LOG = logging.getLogger(__name__)


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
    "extreme_heat_risk": 0.25,
    "extreme_cold_risk": 0.25,
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

_GEOCLIMATE_YEAR_SCENARIO_MAP = {"2030": Scenarios.CURRENT, "2070": Scenarios.FORECAST}
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
_COASTAL_EROSION_YEAR_SCENARIO_MAP = {"2055": Scenarios.CURRENT, "2105": Scenarios.FORECAST}
_COASTAL_EROSION_WEIGHTS = {"erosion_risk": 0.9, "giz_risk": 0.1}

_FLOODING_GRID_SIZE_M = 1000
_FLOODING_TILE_SIZE_M = 10000
_FLOODING_RISK_SCORE_MAP = {"Unavailable": 0, "Very low": 0, "Low": 1, "Medium": 2, "High": 3}
_FLOODING_WEIGHTS = {"rivers_sea_flooding_risk": 0.5, "surface_water_flooding_risk": 0.5}


### GENERAL FUNCTIONS


def min_max_scaling_pair(
    data: pd.DataFrame,
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
        data: pd.DataFrame
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
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range)
    for col_current, col_forecast in pairs:
        # Combine both columns into one array for global min/max
        combined_values = data[[col_current, col_forecast]].to_numpy().flatten().reshape(-1, 1)

        scaler.fit(combined_values)

        # Transform each column using the same scaler
        data[col_current] = scaler.transform(data[[col_current]].values).clip(*feature_range)
        data[col_forecast] = scaler.transform(data[[col_forecast]].values).clip(*feature_range)

    return data


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
    for scenario in Scenarios.all():
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
    for scenario in Scenarios.all():
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
            (geom_types.isin(["Polygon", "MultiPolygon", "GeometryCollection"]))
            & (~hazard_overlay.geometry.is_empty)
            & (hazard_overlay.geometry.notna())
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


def _get_cmap_for_column(col: str) -> str:
    """Return the appropriate colormap for a given hazard column."""
    base_col = col.replace("_risk", "")
    cmap = None

    if base_col in MainHazardCols.all():
        cmap = MainHazardCols.get_cmap(MainHazardCols(base_col))

    if base_col in ExtremeWeatherCols.all():
        cmap = ExtremeWeatherCols.get_cmap(ExtremeWeatherCols(base_col))
    if base_col in FloodingCols.all():
        cmap = FloodingCols.get_cmap(FloodingCols(base_col))
    if base_col in GroundStabilityCols.all():
        cmap = GroundStabilityCols.get_cmap(GroundStabilityCols(base_col))

    if base_col in ExtremeHeatCols.all():
        cmap = ExtremeWeatherCols.get_cmap(ExtremeWeatherCols.EXTREME_HEAT)
    if base_col in ExtremeColdCols.all():
        cmap = ExtremeWeatherCols.get_cmap(ExtremeWeatherCols.EXTREME_COLD)
    if base_col in DroughtCols.all():
        cmap = ExtremeWeatherCols.get_cmap(ExtremeWeatherCols.DROUGHT)
    if base_col in StormCols.all():
        cmap = ExtremeWeatherCols.get_cmap(ExtremeWeatherCols.STORM)

    return cmap if cmap is not None else "Reds"  # Default colormap


def plot_choropleth_current_and_forecast(
    risk_data: gpd.GeoDataFrame,
    column: str,
    title: str,
    out_path: pathlib.Path,
    linewidth: float = 0.1,
    edgecolor: str | None = "black",
    basemap_source: xyzservices.TileProvider | None = ctx.providers.CartoDB.Positron,
) -> None:
    """Plot a choropleth map of the given column in the risk data."""
    _fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    if basemap_source is not None:
        risk_data = risk_data.to_crs(epsg=3857)

    cmap = _get_cmap_for_column(column)

    risk_data.plot(
        column=f"{column}_{Scenarios.CURRENT}",
        cmap=cmap,
        linewidth=linewidth,
        ax=ax[0],
        edgecolor=edgecolor,
        legend=True,
        vmin=0,
        vmax=100,
        alpha=0.7 if basemap_source is not None else 1.0,
    )

    risk_data.plot(
        column=f"{column}_{Scenarios.FORECAST}",
        cmap=cmap,
        linewidth=linewidth,
        ax=ax[1],
        edgecolor=edgecolor,
        legend=True,
        vmin=0,
        vmax=100,
        alpha=0.7 if basemap_source is not None else 1.0,
    )

    if basemap_source is not None:
        ctx.add_basemap(ax[0], source=basemap_source)
        ctx.add_basemap(ax[1], source=basemap_source)

    ax[0].set_title(f"{title} - {Scenarios.CURRENT.title()}")
    ax[1].set_title(f"{title} - {Scenarios.FORECAST.title()}")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _validate_index(index: gpd.GeoDataFrame, index_vars: list[str]) -> None:
    """Validate a given index."""
    if index.isna().any().any():
        raise ValueError("Index contains NA values.")

    for scenario in Scenarios.all():
        for var in index_vars:
            col = f"{var}_{scenario}"
            if col not in index.columns:
                raise ValueError(f"Missing column: {col}")
            if not index[col].between(0, 100).all():
                raise ValueError(
                    f"{var.replace('_', ' ').title()} for {scenario} "
                    f"contains values outside 0-100."
                )


def _audit_index(
    index: gpd.GeoDataFrame, index_vars: list[str], out_path: pathlib.Path
) -> None:
    """Audit a given index."""
    out_path.mkdir(parents=True, exist_ok=True)

    for var in index_vars:
        # Plot Choropleth Maps for each variable
        plot_choropleth_current_and_forecast(
            index,
            var,
            f"{var.replace('_', ' ').title()}",
            out_path / f"{var}_choropleth.png",
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
    if config.switches.extreme_weather:
        _extreme_weather_index(config, audit_path)
    if config.switches.flooding:
        _flooding_index(config, boundary, audit_path)
    if config.switches.ground_stability:
        _ground_stability_index(config, audit_path)
    if config.switches.coastal_erosion:
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
        [
            "grid_id",
            "part",
            f"extreme_heat_risk_{Scenarios.CURRENT}",
            f"extreme_heat_risk_{Scenarios.FORECAST}",
        ]
    ].merge(
        extreme_cold[
            [
                "grid_id",
                "part",
                f"extreme_cold_risk_{Scenarios.CURRENT}",
                f"extreme_cold_risk_{Scenarios.FORECAST}",
                "geometry",
            ]
        ],
        on=["grid_id", "part"],
        how="inner",
    )

    extreme_heat_cold = gpd.GeoDataFrame(
        extreme_heat_cold, geometry="geometry", crs=hazard_grid.crs
    )
    extreme_heat_cold = extreme_heat_cold.drop(columns=["grid_id", "part"])

    extreme_weather_risk = _overlay_and_clean(
        extreme_heat_cold,
        drought[
            [
                f"drought_risk_{Scenarios.CURRENT}",
                f"drought_risk_{Scenarios.FORECAST}",
                "geometry",
            ]
        ],
        storm[
            [f"storm_risk_{Scenarios.CURRENT}", f"storm_risk_{Scenarios.FORECAST}", "geometry"]
        ],
        target_crs=data_cleaning.BNG_CRS,
    )

    extreme_weather_risk = _iterative_spatial_infilling(
        extreme_weather_risk,
        [
            f"extreme_heat_risk_{Scenarios.CURRENT}",
            f"extreme_heat_risk_{Scenarios.FORECAST}",
            f"extreme_cold_risk_{Scenarios.CURRENT}",
            f"extreme_cold_risk_{Scenarios.FORECAST}",
            f"drought_risk_{Scenarios.CURRENT}",
            f"drought_risk_{Scenarios.FORECAST}",
            f"storm_risk_{Scenarios.CURRENT}",
            f"storm_risk_{Scenarios.FORECAST}",
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
        [
            (
                f"extreme_weather_risk_{Scenarios.CURRENT}",
                f"extreme_weather_risk_{Scenarios.FORECAST}",
            )
        ],
    )

    extreme_weather_risk = gpd.GeoDataFrame(
        extreme_weather_risk, geometry="geometry", crs=data_cleaning.BNG_CRS
    )

    _validate_index(
        extreme_weather_risk,
        [
            "extreme_heat_risk",
            "extreme_cold_risk",
            "drought_risk",
            "storm_risk",
            "extreme_weather_risk",
        ],
    )

    _audit_index(
        extreme_weather_risk,
        [
            "extreme_heat_risk",
            "extreme_cold_risk",
            "drought_risk",
            "storm_risk",
            "extreme_weather_risk",
        ],
        audit_path / "Extreme Weather" / "Extreme Weather Risk Index",
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
    hsd = pd.read_csv(config.paths.model_input / file_paths.HOT_SUMMER_DAYS_MODEL_INPUT_PATH)
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
            (
                f"max_temp_summer_risk_{Scenarios.CURRENT}",
                f"max_temp_summer_risk_{Scenarios.FORECAST}",
            ),
            (f"hot_summer_days_{Scenarios.CURRENT}", f"hot_summer_days_{Scenarios.FORECAST}"),
            (
                f"extreme_summer_days_{Scenarios.CURRENT}",
                f"extreme_summer_days_{Scenarios.FORECAST}",
            ),
        ],
    )

    extreme_heat = _calculate_composite_score(
        extreme_heat,
        _EXTREME_HEAT_WEIGHTS,
        "extreme_heat_risk",
    )

    extreme_heat = min_max_scaling_pair(
        extreme_heat,
        [
            (
                f"extreme_heat_risk_{Scenarios.CURRENT}",
                f"extreme_heat_risk_{Scenarios.FORECAST}",
            )
        ],
    )

    LOG.info("Extreme heat index calculation complete.")
    extreme_heat = gpd.GeoDataFrame(extreme_heat, geometry="geometry", crs=hazard_grid.crs)

    _validate_index(
        extreme_heat,
        [
            "max_temp_summer_risk",
            "hot_summer_days",
            "extreme_summer_days",
            "extreme_heat_risk",
        ],
    )

    _audit_index(
        extreme_heat,
        [
            "max_temp_summer_risk",
            "hot_summer_days",
            "extreme_summer_days",
            "extreme_heat_risk",
        ],
        audit_path / "Extreme Weather" / "Extreme Heat Risk Index",
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

    extreme_cold = _merge_on_key([temp_min, frost, icing], hazard_grid, "grid_id")

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
            (
                f"min_temp_winter_risk_{Scenarios.CURRENT}",
                f"min_temp_winter_risk_{Scenarios.FORECAST}",
            ),
            (f"frost_days_{Scenarios.CURRENT}", f"frost_days_{Scenarios.FORECAST}"),
            (f"icing_days_{Scenarios.CURRENT}", f"icing_days_{Scenarios.FORECAST}"),
        ],
    )

    extreme_cold = _calculate_composite_score(
        extreme_cold,
        _EXTREME_COLD_WEIGHTS,
        "extreme_cold_risk",
    )

    extreme_cold = min_max_scaling_pair(
        extreme_cold,
        [
            (
                f"extreme_cold_risk_{Scenarios.CURRENT}",
                f"extreme_cold_risk_{Scenarios.FORECAST}",
            )
        ],
    )

    extreme_cold = gpd.GeoDataFrame(extreme_cold, geometry="geometry", crs=hazard_grid.crs)

    _validate_index(
        extreme_cold,
        ["min_temp_winter_risk", "frost_days", "icing_days", "extreme_cold_risk"],
    )

    _audit_index(
        extreme_cold,
        ["min_temp_winter_risk", "frost_days", "icing_days", "extreme_cold_risk"],
        audit_path / "Extreme Weather" / "Extreme Cold Risk Index",
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
        [
            f"precip_summer_{Scenarios.CURRENT}",
            f"precip_summer_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    drought_risk = _overlay_and_clean(
        precip_sum_gdf, drought, target_crs=data_cleaning.BNG_CRS
    )

    drought_risk = _iterative_spatial_infilling(
        drought_risk,
        [
            f"precip_summer_{Scenarios.CURRENT}",
            f"precip_summer_{Scenarios.FORECAST}",
            f"drought_severity_index_{Scenarios.CURRENT}",
            f"drought_severity_index_{Scenarios.FORECAST}",
        ],
        _DROUGHT_NEAREST_JOIN_MAX_DISTANCE,
    )

    drought_risk = drought_risk[
        [
            f"drought_severity_index_{Scenarios.CURRENT}",
            f"drought_severity_index_{Scenarios.FORECAST}",
            f"precip_summer_{Scenarios.CURRENT}",
            f"precip_summer_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    drought_risk = min_max_scaling_pair(
        drought_risk,
        [
            (
                f"drought_severity_index_{Scenarios.CURRENT}",
                f"drought_severity_index_{Scenarios.FORECAST}",
            ),
            (f"precip_summer_{Scenarios.CURRENT}", f"precip_summer_{Scenarios.FORECAST}"),
        ],
    )

    # Reverse the polarity for precipitation
    drought_risk[f"precip_summer_risk_{Scenarios.CURRENT}"] = (
        100 - drought_risk[f"precip_summer_{Scenarios.CURRENT}"]
    )
    drought_risk[f"precip_summer_risk_{Scenarios.FORECAST}"] = (
        100 - drought_risk[f"precip_summer_{Scenarios.FORECAST}"]
    )

    drought_risk = _calculate_composite_score(
        drought_risk,
        _DROUGHT_WEIGHTS,
        "drought_risk",
    )

    drought_risk = min_max_scaling_pair(
        drought_risk,
        [(f"drought_risk_{Scenarios.CURRENT}", f"drought_risk_{Scenarios.FORECAST}")],
    )

    drought_risk = gpd.GeoDataFrame(
        drought_risk, geometry="geometry", crs=data_cleaning.BNG_CRS
    )

    _validate_index(
        drought_risk,
        ["drought_severity_index", "precip_summer_risk", "drought_risk"],
    )

    _audit_index(
        drought_risk,
        ["drought_severity_index", "precip_summer_risk", "drought_risk"],
        audit_path / "Extreme Weather" / "Drought Risk Index",
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
    rain_days = gpd.read_file(config.paths.model_input / file_paths.RAIN_DAYS_MODEL_INPUT_PATH)
    wind_spd = gpd.read_file(config.paths.model_input / file_paths.WIND_SPEED_MODEL_INPUT_PATH)
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
        [
            f"precip_winter_{Scenarios.CURRENT}",
            f"precip_winter_{Scenarios.FORECAST}",
            "geometry",
        ]
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
            f"10mm_rain_days_{Scenarios.CURRENT}",
            f"precip_winter_{Scenarios.CURRENT}",
            f"precip_winter_{Scenarios.FORECAST}",
            f"wind_speed_99th_percentile_{Scenarios.CURRENT}",
            f"wind_speed_99th_percentile_{Scenarios.FORECAST}",
            f"avg_exceedance_days_{Scenarios.CURRENT}",
            f"avg_exceedance_days_{Scenarios.FORECAST}",
            f"wind_driven_rain_index_{Scenarios.CURRENT}",
            f"wind_driven_rain_index_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    storm_risk = _iterative_spatial_infilling(
        storm_risk,
        [
            f"10mm_rain_days_{Scenarios.CURRENT}",
            f"precip_winter_{Scenarios.CURRENT}",
            f"precip_winter_{Scenarios.FORECAST}",
            f"wind_speed_99th_percentile_{Scenarios.CURRENT}",
            f"wind_speed_99th_percentile_{Scenarios.FORECAST}",
            f"avg_exceedance_days_{Scenarios.CURRENT}",
            f"avg_exceedance_days_{Scenarios.FORECAST}",
            f"wind_driven_rain_index_{Scenarios.CURRENT}",
            f"wind_driven_rain_index_{Scenarios.FORECAST}",
        ],
        _STORM_NEAREST_JOIN_MAX_DISTANCE,
    )

    storm_risk[f"wind_speed_risk_{Scenarios.CURRENT}"] = storm_risk[
        f"wind_speed_99th_percentile_{Scenarios.CURRENT}"
    ].apply(_wind_risk_scaled)
    storm_risk[f"wind_speed_risk_{Scenarios.FORECAST}"] = storm_risk[
        f"wind_speed_99th_percentile_{Scenarios.FORECAST}"
    ].apply(_wind_risk_scaled)

    storm_risk = min_max_scaling_pair(
        storm_risk,
        [
            (f"wind_speed_risk_{Scenarios.CURRENT}", f"wind_speed_risk_{Scenarios.FORECAST}"),
            (f"precip_winter_{Scenarios.CURRENT}", f"precip_winter_{Scenarios.FORECAST}"),
            (
                f"avg_exceedance_days_{Scenarios.CURRENT}",
                f"avg_exceedance_days_{Scenarios.FORECAST}",
            ),
            (
                f"wind_driven_rain_index_{Scenarios.CURRENT}",
                f"wind_driven_rain_index_{Scenarios.FORECAST}",
            ),
        ],
    )

    # Scale rain days on its own, then duplicate
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
    storm_risk[f"10mm_rain_days_{Scenarios.CURRENT}"] = scaler.fit_transform(
        storm_risk[[f"10mm_rain_days_{Scenarios.CURRENT}"]]
    )
    storm_risk[f"10mm_rain_days_{Scenarios.FORECAST}"] = storm_risk[
        f"10mm_rain_days_{Scenarios.CURRENT}"
    ]

    storm_risk = _calculate_composite_score(
        storm_risk,
        _STORM_WEIGHTS,
        "storm_risk",
    )

    storm_risk = min_max_scaling_pair(
        storm_risk,
        [(f"storm_risk_{Scenarios.CURRENT}", f"storm_risk_{Scenarios.FORECAST}")],
    )

    storm_risk = gpd.GeoDataFrame(storm_risk, geometry="geometry", crs=data_cleaning.BNG_CRS)

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
        audit_path / "Extreme Weather" / "Storm Risk Index",
    )

    LOG.info("Storm index calculation complete.")
    return storm_risk


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
    config: model_config.Config, boundary: gpd.GeoDataFrame, audit_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Overlay all four flooding datasets using a tiled chunking method."""
    LOG.info("Combining all four flooding datasets...")

    # If the direct tiled overlay hasn't been done yet, do it
    if config.switches.compute_flooding_overlay:
        flooding_paths = []
        for flooding_type in config.hazards.flooding:
            for scenario in Scenarios.all():
                flooding_paths.append(
                    file_paths.FLOODING_MODEL_INPUT_PATH
                    / flooding_type
                    / scenario
                    / f"{flooding_type}_{scenario}.gpkg"
                )

        _tile_polygon_flooding_overlay(
            config,
            boundary,
            flooding_paths,
            crs=data_cleaning.BNG_CRS,
            tile_size_m=_FLOODING_TILE_SIZE_M,
        )

    # Read the direct overlay result, and filter to region
    # Eventually want to rename the layer to 'flooding_overlay'
    flooding_risk = gpd.read_file(
        config.paths.model_interim_output
        / file_paths.FLOODING_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH,
        mask=boundary,
        layer="flood_overlay",
    )
    # Eventually want to rename columns to 'flooding' rather than 'flood'
    flooding_risk = flooding_risk.rename(
        columns={
            f"rivers_sea_flood_risk_{Scenarios.CURRENT}": (
                f"rivers_sea_flooding_risk_{Scenarios.CURRENT}"
            ),
            f"rivers_sea_flood_risk_{Scenarios.FORECAST}": (
                f"rivers_sea_flooding_risk_{Scenarios.FORECAST}"
            ),
            f"surface_water_flood_risk_{Scenarios.CURRENT}": (
                f"surface_water_flooding_risk_{Scenarios.CURRENT}"
            ),
            f"surface_water_flood_risk_{Scenarios.FORECAST}": (
                f"surface_water_flooding_risk_{Scenarios.FORECAST}"
            ),
        }
    )

    # Map original risk categories to numeric scores
    for col in [
        f"rivers_sea_flooding_risk_{Scenarios.CURRENT}",
        f"rivers_sea_flooding_risk_{Scenarios.FORECAST}",
        f"surface_water_flooding_risk_{Scenarios.CURRENT}",
        f"surface_water_flooding_risk_{Scenarios.FORECAST}",
    ]:
        flooding_risk[col] = flooding_risk[col].map(_FLOODING_RISK_SCORE_MAP)

    # Fill NA values with 0 (no risk) since no data means no risk in the underlying data
    flooding_risk = flooding_risk.fillna(0)

    flooding_risk = min_max_scaling_pair(
        flooding_risk,
        [
            (
                f"rivers_sea_flooding_risk_{Scenarios.CURRENT}",
                f"rivers_sea_flooding_risk_{Scenarios.FORECAST}",
            ),
            (
                f"surface_water_flooding_risk_{Scenarios.CURRENT}",
                f"surface_water_flooding_risk_{Scenarios.FORECAST}",
            ),
        ],
    )

    flooding_risk = _calculate_composite_score(
        flooding_risk,
        _FLOODING_WEIGHTS,
        "flooding_risk",
    )

    flooding_risk = min_max_scaling_pair(
        flooding_risk,
        [
            (f"flooding_risk_{Scenarios.CURRENT}", f"flooding_risk_{Scenarios.FORECAST}"),
        ],
    )

    _validate_index(
        flooding_risk,
        ["rivers_sea_flooding_risk", "surface_water_flooding_risk", "flooding_risk"],
    )

    _audit_index(
        flooding_risk,
        ["rivers_sea_flooding_risk", "surface_water_flooding_risk", "flooding_risk"],
        audit_path / "Flooding" / "Flooding Risk Index",
    )

    data_cleaning.write_to_file(
        flooding_risk,
        config.paths.model_interim_output / file_paths.FLOODING_RISK_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Flooding risk index calculation complete.")
    return flooding_risk


def _tile_polygon_flooding_overlay(
    config: model_config.Config,
    boundary: gpd.GeoDataFrame,
    layer_paths: list[pathlib.Path],
    crs: str,
    tile_size_m: int = 5000,
) -> gpd.GeoDataFrame:
    """Chunked polygon-polygon overlay using a tile grid."""
    # Create tiles
    tiles = _create_flooding_tiles(config, boundary, tile_size_m)

    # Prepare output GPKG
    output_path = (
        config.paths.model_interim_output
        / file_paths.FLOODING_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH
    )
    layer_name = "flooding_overlay"
    first_write = True

    # For each tile, do spatial filtering and run overlay and clean
    for tile_idx, tile in tiles.iterrows():
        LOG.info("Tile %s/%s starting overlay", tile_idx + 1, len(tiles))

        tile_overlay = _process_flooding_overlay_tile(
            tile=tile, layer_paths=layer_paths, crs=crs
        )

        # If there are no resulting geometries, skip to the next tile
        if tile_overlay is None:
            continue

        data_cleaning.write_to_file(
            tile_overlay, output_path, mode="w" if first_write else "a", layer=layer_name
        )
        first_write = False

        LOG.info("Tile %s wrote %s geometries.", tile_idx + 1, len(tile_overlay))

    LOG.info("Chunked overlay completed. Output written to %s", output_path)


def _create_flooding_tiles(
    config: model_config.Config, boundary: gpd.GeoDataFrame, tile_size_m: int
) -> gpd.GeoDataFrame:
    """Create flooding tiles to be used in chunked overlay."""
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


def _process_flooding_overlay_tile(
    *, tile: gpd.GeoSeries, layer_paths: list[pathlib.Path], crs: str
) -> gpd.GeoDataFrame | None:
    """Process flooding overlay for a single tile."""
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
        ground_stability[Scenarios.CURRENT],
        ground_stability[Scenarios.FORECAST],
        target_crs=data_cleaning.BNG_CRS,
    )

    risk_cols = [
        f"{hazard}_risk_{suffix}"
        for hazard in GroundStabilityCols.all()
        for suffix in Scenarios.all()
    ]

    for col in risk_cols:
        ground_stability[col] = pd.to_numeric(ground_stability[col], errors="coerce")

    ground_stability = _iterative_spatial_infilling(
        ground_stability, risk_cols, _GROUND_STABILITY_NEAREST_JOIN_MAX_DISTANCE
    )

    gs_pairs = [
        (f"{col}_risk_{Scenarios.CURRENT}", f"{col}_risk_{Scenarios.FORECAST}")
        for col in GroundStabilityCols.all()
    ]

    ground_stability = min_max_scaling_pair(ground_stability, gs_pairs)

    ground_stability = _calculate_composite_score(
        ground_stability,
        _GROUND_STABILITY_WEIGHTS,
        "ground_stability_risk",
    )

    ground_stability = min_max_scaling_pair(
        ground_stability,
        [
            (
                f"ground_stability_risk_{Scenarios.CURRENT}",
                f"ground_stability_risk_{Scenarios.FORECAST}",
            )
        ],
    )

    _validate_index(
        ground_stability,
        [f"{col}_risk" for col in GroundStabilityCols.all()] + ["ground_stability_risk"],
    )

    _audit_index(
        ground_stability,
        [f"{col}_risk" for col in GroundStabilityCols.all()] + ["ground_stability_risk"],
        audit_path / "Ground Stability" / "Ground Stability Risk Index",
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
                "Both NCERM and GIZ layers are empty for scenario %s. "
                "Coastal erosion risk will be 0 everywhere.",
                scenario,
            )
            if year == "2055":
                continue
            data_cleaning.write_to_file(
                gpd.GeoDataFrame(
                    columns=[
                        f"coastal_erosion_risk_{Scenarios.CURRENT}",
                        f"coastal_erosion_risk_{Scenarios.FORECAST}",
                        "geometry",
                    ],  # Empty GeoDataFrame
                    geometry="geometry",
                    crs=data_cleaning.BNG_CRS,
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
            erosion_risk[scenario]["erosion_risk"] * _COASTAL_EROSION_WEIGHTS["erosion_risk"]
            + erosion_risk[scenario]["giz_risk"] * _COASTAL_EROSION_WEIGHTS["giz_risk"]
        )

        erosion_risk[scenario] = erosion_risk[scenario].rename(
            columns={"coastal_erosion_risk": f"coastal_erosion_risk_{scenario}"}
        )

    coastal_erosion_risk = _overlay_and_clean(
        erosion_risk[Scenarios.CURRENT],
        erosion_risk[Scenarios.FORECAST],
        target_crs=data_cleaning.BNG_CRS,
    )

    coastal_erosion_risk = _iterative_spatial_infilling(
        coastal_erosion_risk,
        [
            f"coastal_erosion_risk_{Scenarios.CURRENT}",
            f"coastal_erosion_risk_{Scenarios.FORECAST}",
        ],
        nearest_join_max_distance=_COASTAL_EROSION_NEAREST_JOIN_MAX_DISTANCE,
    )
    coastal_erosion_risk = gpd.GeoDataFrame(coastal_erosion_risk, geometry="geometry")
    coastal_erosion_risk = coastal_erosion_risk[
        [
            f"coastal_erosion_risk_{Scenarios.CURRENT}",
            f"coastal_erosion_risk_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    _validate_index(
        coastal_erosion_risk,
        ["coastal_erosion_risk"],
    )

    _audit_index(
        coastal_erosion_risk,
        ["coastal_erosion_risk"],
        audit_path / "Coastal Erosion" / "Coastal Erosion Risk Index",
    )

    data_cleaning.write_to_file(
        coastal_erosion_risk,
        config.paths.model_interim_output
        / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Coastal erosion risk index calculation complete.")
