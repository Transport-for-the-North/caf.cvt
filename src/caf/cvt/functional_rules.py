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
from caf.cvt.definitions import (
    CoastalErosionRiskCols,
    DroughtCols,
    ExtremeColdCols,
    ExtremeHeatCols,
    ExtremeWeatherRiskCols,
    FloodingRiskCols,
    GroundStabilityRiskCols,
    MainHazardRiskCols,
    RiskColumn,
    Scenarios,
    StormCols,
)

LOG = logging.getLogger(__name__)

plt.switch_backend("Agg")  # Use non-interactive backend for plotting

_EXTREME_HEAT_RISK_THRESHOLD = 30
_EXTREME_HEAT_WEIGHTS: dict[str, float] = {
    ExtremeHeatCols.MAX_TEMP_SUMMER: 0.5,
    ExtremeHeatCols.HOT_SUMMER_DAYS: 0.25,
    ExtremeHeatCols.EXTREME_SUMMER_DAYS: 0.25,
}

_EXTREME_COLD_RISK_THRESHOLD = 0
_EXTREME_COLD_WEIGHTS: dict[str, float] = {
    ExtremeColdCols.MIN_TEMP_WINTER: 0.5,
    ExtremeColdCols.FROST_DAYS: 0.25,
    ExtremeColdCols.ICING_DAYS: 0.25,
}

_WIND_SPEED_RISK_THRESHOLD_LOWER = 13.4  # 30 mph in m/s (should not exceed upper threshold)
_WIND_SPEED_RISK_THRESHOLD_UPPER = 20.1  # 45 mph in m/s (should not exceed 25)
_EXTREME_WIND_MAX = 25

_DROUGHT_NEAREST_JOIN_MAX_DISTANCE = 10000
_DROUGHT_WEIGHTS: dict[str, float] = {
    DroughtCols.DROUGHT_SEVERITY_INDEX: 0.75,
    DroughtCols.PRECIP_SUMMER: 0.25,
}

_STORM_NEAREST_JOIN_MAX_DISTANCE = 5000
_STORM_WEIGHTS: dict[str, float] = {
    StormCols.WIND_SPEED: 0.3,
    StormCols.EXCEEDANCE_DAYS: 0.2,
    StormCols.PRECIP_WINTER: 0.15,
    StormCols.RAIN_DAYS: 0.15,
    StormCols.WIND_DRIVEN_RAIN_INDEX: 0.2,
}

_EXTREME_WEATHER_NEAREST_JOIN_MAX_DISTANCE = 10000
_EXTREME_WEATHER_WEIGHTS: dict[str, float] = {
    ExtremeWeatherRiskCols.EXTREME_HEAT: 0.25,
    ExtremeWeatherRiskCols.EXTREME_COLD: 0.25,
    ExtremeWeatherRiskCols.DROUGHT: 0.25,
    ExtremeWeatherRiskCols.STORM: 0.25,
}

_GROUND_STABILITY_NEAREST_JOIN_MAX_DISTANCE = 1000
_GROUND_STABILITY_RISK_SCORE_MAP: dict[
    str, float
] = {
    "Probable": 1,
    "Possible": 0.66,
    "Improbable": 0.33,
    "Unavailable": 0.5,  # Assign neutral value
}

_GEOCLIMATE_YEAR_SCENARIO_MAP = {"2030": Scenarios.CURRENT, "2070": Scenarios.FORECAST}
_GROUND_STABILITY_WEIGHTS: dict[str, float] = {
    GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE: 0.40,
    GroundStabilityRiskCols.LANDSLIDES: 0.10,
    GroundStabilityRiskCols.SHRINK_SWELL: 0.10,
    GroundStabilityRiskCols.COMPRESSIBLE_GROUND: 0.10,
    GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: 0.10,
    GroundStabilityRiskCols.RUNNING_SAND: 0.10,
    GroundStabilityRiskCols.SOLUBLE_ROCKS: 0.10,
}

_COASTAL_EROSION_NEAREST_JOIN_MAX_DISTANCE = 500
_COASTAL_EROSION_YEAR_SCENARIO_MAP = {"2055": Scenarios.CURRENT, "2105": Scenarios.FORECAST}
_COASTAL_EROSION_WEIGHTS: dict[str, float] = {
    CoastalErosionRiskCols.EROSION: 0.9,
    CoastalErosionRiskCols.GIZ: 0.1,
}

_FLOODING_TILE_SIZE_M = 10000
_FLOODING_RISK_SCORE_MAP = {"Unavailable": 0, "Very low": 0, "Low": 1, "Medium": 2, "High": 3}
_FLOODING_WEIGHTS: dict[str, float] = {
    FloodingRiskCols.RIVERS_SEA: 0.5,
    FloodingRiskCols.SURFACE_WATER: 0.5,
}

_PLOT_ALPHA_BASEMAP = 0.7
_PLOT_ALPHA_NO_BASEMAP = 1.0


### GENERAL FUNCTIONS


def min_max_scaling_pair(
    data: pd.DataFrame,
    pairs: list[tuple[str, str]],
    feature_range: tuple[int, int],
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
             Desired range of the transformed data.

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
    for scenario in Scenarios:
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
    risk_data: pd.DataFrame, weights: dict[str, float], output_col: RiskColumn
) -> pd.DataFrame:
    """Calculate composite score given a dataframe with variables and corresponding weights."""
    for scenario in Scenarios:
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
        ]
        hazard_overlay = data_cleaning.validate_geometries(hazard_overlay)
        hazard_overlay = hazard_overlay.reset_index(drop=True)
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


def plot_choropleth_current_and_forecast(
    risk_data: gpd.GeoDataFrame,
    column: RiskColumn,
    title: str,
    out_path: pathlib.Path,
    *,
    feature_range: tuple[int, int],
    linewidth: float = 0.1,
    edgecolor: str | None = "black",
    basemap_source: xyzservices.TileProvider | None = None,
) -> None:
    """Plot a choropleth map of the given column in the risk data.

    For a given column, this function plots two choropleth maps side by side: one for each
    scenario. The maps are saved to the specified output path.

    Parameters
    ----------
    risk_data : gpd.GeoDataFrame
        GeoDataFrame containing the risk data to plot.
    column : RiskColumn
        The column to plot, which should be a subclass of RiskColumn.
    title : str
        The title for the plots.
    out_path : pathlib.Path
        The path to save the output plot.
    feature_range : tuple[int, int]
        The range of values to use for the color scale.
    linewidth : float, optional
        The width of the lines between polygons, by default 0.1.
    edgecolor : str | None, optional
        The color of the edges of the polygons, by default "black".
    basemap_source : xyzservices.TileProvider | None, optional
        The source for the basemap tiles. If none, no basemap is added. By default None.

    Returns
    -------
    None
    """
    _fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    if basemap_source is not None:
        risk_data = risk_data.to_crs(epsg=3857)

    cmap = column.get_cmap()

    risk_data.plot(
        column=f"{column}_{Scenarios.CURRENT}",
        cmap=cmap,
        linewidth=linewidth,
        ax=ax[0],
        edgecolor=edgecolor,
        legend=True,
        vmin=feature_range[0],
        vmax=feature_range[1],
        alpha=_PLOT_ALPHA_BASEMAP if basemap_source is not None else _PLOT_ALPHA_NO_BASEMAP,
    )

    risk_data.plot(
        column=f"{column}_{Scenarios.FORECAST}",
        cmap=cmap,
        linewidth=linewidth,
        ax=ax[1],
        edgecolor=edgecolor,
        legend=True,
        vmin=feature_range[0],
        vmax=feature_range[1],
        alpha=_PLOT_ALPHA_BASEMAP if basemap_source is not None else _PLOT_ALPHA_NO_BASEMAP,
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


def _validate_index(
    index: gpd.GeoDataFrame, index_vars: list[RiskColumn], feature_range: tuple[int, int]
) -> None:
    """Validate a given index."""
    if index.isna().any().any():
        raise ValueError("Index contains NA values.")

    for scenario in Scenarios:
        for var in index_vars:
            col = f"{var}_{scenario}"
            if col not in index.columns:
                raise ValueError(f"Missing column: {col}")
            if not index[col].between(feature_range[0], feature_range[1]).all():
                raise ValueError(
                    f"{var.replace('_', ' ').title()} for {scenario} "
                    f"contains values outside {feature_range[0]}-{feature_range[1]}."
                )


def _audit_index(
    index: gpd.GeoDataFrame,
    index_vars: list[RiskColumn],
    out_path: pathlib.Path,
    feature_range: tuple[int, int],
) -> None:
    """Audit a given index."""
    out_path.mkdir(parents=True, exist_ok=True)

    for var in index_vars:
        # Plot Choropleth Maps for each variable
        plot_choropleth_current_and_forecast(
            risk_data=index,
            column=var,
            title=f"{var.replace('_', ' ').title()}",
            out_path=out_path / f"{var}_choropleth.png",
            feature_range=feature_range,
            basemap_source=xyzservices.providers.CartoDB.Positron,
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
            f"{ExtremeWeatherRiskCols.EXTREME_HEAT}_{Scenarios.CURRENT}",
            f"{ExtremeWeatherRiskCols.EXTREME_HEAT}_{Scenarios.FORECAST}",
        ]
    ].merge(
        extreme_cold[
            [
                "grid_id",
                "part",
                f"{ExtremeWeatherRiskCols.EXTREME_COLD}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.EXTREME_COLD}_{Scenarios.FORECAST}",
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
                f"{ExtremeWeatherRiskCols.DROUGHT}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.DROUGHT}_{Scenarios.FORECAST}",
                "geometry",
            ]
        ],
        storm[
            [
                f"{ExtremeWeatherRiskCols.STORM}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.STORM}_{Scenarios.FORECAST}",
                "geometry",
            ]
        ],
        target_crs=data_cleaning.BNG_CRS,
    )

    extreme_weather_risk = _iterative_spatial_infilling(
        extreme_weather_risk,
        [
            f"{ExtremeWeatherRiskCols.EXTREME_HEAT}_{Scenarios.CURRENT}",
            f"{ExtremeWeatherRiskCols.EXTREME_HEAT}_{Scenarios.FORECAST}",
            f"{ExtremeWeatherRiskCols.EXTREME_COLD}_{Scenarios.CURRENT}",
            f"{ExtremeWeatherRiskCols.EXTREME_COLD}_{Scenarios.FORECAST}",
            f"{ExtremeWeatherRiskCols.DROUGHT}_{Scenarios.CURRENT}",
            f"{ExtremeWeatherRiskCols.DROUGHT}_{Scenarios.FORECAST}",
            f"{ExtremeWeatherRiskCols.STORM}_{Scenarios.CURRENT}",
            f"{ExtremeWeatherRiskCols.STORM}_{Scenarios.FORECAST}",
        ],
        _EXTREME_WEATHER_NEAREST_JOIN_MAX_DISTANCE,
    )

    extreme_weather_risk = _calculate_composite_score(
        extreme_weather_risk,
        _EXTREME_WEATHER_WEIGHTS,
        MainHazardRiskCols.EXTREME_WEATHER,
    )

    feature_range = (config.constants.score_min, config.constants.score_max)
    extreme_weather_risk = min_max_scaling_pair(
        extreme_weather_risk,
        [
            (
                f"{MainHazardRiskCols.EXTREME_WEATHER}_{Scenarios.CURRENT}",
                f"{MainHazardRiskCols.EXTREME_WEATHER}_{Scenarios.FORECAST}",
            )
        ],
        feature_range,
    )

    extreme_weather_risk = gpd.GeoDataFrame(
        extreme_weather_risk, geometry="geometry", crs=data_cleaning.BNG_CRS
    )

    _validate_index(
        extreme_weather_risk,
        [*ExtremeWeatherRiskCols, MainHazardRiskCols.EXTREME_WEATHER],
        feature_range,
    )

    _audit_index(
        extreme_weather_risk,
        [*ExtremeWeatherRiskCols, MainHazardRiskCols.EXTREME_WEATHER],
        audit_path / "Extreme Weather" / "Extreme Weather Risk Index",
        feature_range,
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
        ExtremeHeatCols.MAX_TEMP_SUMMER,
        ExtremeHeatCols.MAX_TEMP_SUMMER,
        _EXTREME_HEAT_RISK_THRESHOLD,
    )

    feature_range = (config.constants.score_min, config.constants.score_max)
    extreme_heat = min_max_scaling_pair(
        extreme_heat,
        [
            (
                f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.CURRENT}",
                f"{ExtremeHeatCols.MAX_TEMP_SUMMER}_{Scenarios.FORECAST}",
            ),
            (
                f"{ExtremeHeatCols.HOT_SUMMER_DAYS}_{Scenarios.CURRENT}",
                f"{ExtremeHeatCols.HOT_SUMMER_DAYS}_{Scenarios.FORECAST}",
            ),
            (
                f"{ExtremeHeatCols.EXTREME_SUMMER_DAYS}_{Scenarios.CURRENT}",
                f"{ExtremeHeatCols.EXTREME_SUMMER_DAYS}_{Scenarios.FORECAST}",
            ),
        ],
        feature_range,
    )

    extreme_heat = _calculate_composite_score(
        extreme_heat,
        _EXTREME_HEAT_WEIGHTS,
        ExtremeWeatherRiskCols.EXTREME_HEAT,
    )

    extreme_heat = min_max_scaling_pair(
        extreme_heat,
        [
            (
                f"{ExtremeWeatherRiskCols.EXTREME_HEAT}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.EXTREME_HEAT}_{Scenarios.FORECAST}",
            )
        ],
        feature_range,
    )

    LOG.info("Extreme heat index calculation complete.")
    extreme_heat = gpd.GeoDataFrame(extreme_heat, geometry="geometry", crs=hazard_grid.crs)

    _validate_index(
        extreme_heat,
        [*ExtremeHeatCols, ExtremeWeatherRiskCols.EXTREME_HEAT],
        feature_range,
    )

    _audit_index(
        extreme_heat,
        [*ExtremeHeatCols, ExtremeWeatherRiskCols.EXTREME_HEAT],
        audit_path / "Extreme Weather" / "Extreme Heat Risk Index",
        feature_range,
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
        ExtremeColdCols.MIN_TEMP_WINTER,
        ExtremeColdCols.MIN_TEMP_WINTER,
        _EXTREME_COLD_RISK_THRESHOLD,
        invert=True,
    )

    feature_range = (config.constants.score_min, config.constants.score_max)
    extreme_cold = min_max_scaling_pair(
        extreme_cold,
        [
            (
                f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.CURRENT}",
                f"{ExtremeColdCols.MIN_TEMP_WINTER}_{Scenarios.FORECAST}",
            ),
            (
                f"{ExtremeColdCols.FROST_DAYS}_{Scenarios.CURRENT}",
                f"{ExtremeColdCols.FROST_DAYS}_{Scenarios.FORECAST}",
            ),
            (
                f"{ExtremeColdCols.ICING_DAYS}_{Scenarios.CURRENT}",
                f"{ExtremeColdCols.ICING_DAYS}_{Scenarios.FORECAST}",
            ),
        ],
        feature_range,
    )

    extreme_cold = _calculate_composite_score(
        extreme_cold,
        _EXTREME_COLD_WEIGHTS,
        ExtremeWeatherRiskCols.EXTREME_COLD,
    )

    extreme_cold = min_max_scaling_pair(
        extreme_cold,
        [
            (
                f"{ExtremeWeatherRiskCols.EXTREME_COLD}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.EXTREME_COLD}_{Scenarios.FORECAST}",
            )
        ],
        feature_range,
    )

    extreme_cold = gpd.GeoDataFrame(extreme_cold, geometry="geometry", crs=hazard_grid.crs)

    _validate_index(
        extreme_cold,
        [*ExtremeColdCols, ExtremeWeatherRiskCols.EXTREME_COLD],
        feature_range,
    )

    _audit_index(
        extreme_cold,
        [*ExtremeColdCols, ExtremeWeatherRiskCols.EXTREME_COLD],
        audit_path / "Extreme Weather" / "Extreme Cold Risk Index",
        feature_range,
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
            f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}",
            f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    drought_risk = _overlay_and_clean(
        precip_sum_gdf, drought, target_crs=data_cleaning.BNG_CRS
    )

    drought_risk = _iterative_spatial_infilling(
        drought_risk,
        [
            f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}",
            f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}",
            f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.CURRENT}",
            f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.FORECAST}",
        ],
        _DROUGHT_NEAREST_JOIN_MAX_DISTANCE,
    )

    drought_risk = drought_risk[
        [
            f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.CURRENT}",
            f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.FORECAST}",
            f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}",
            f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    feature_range = (config.constants.score_min, config.constants.score_max)
    drought_risk = min_max_scaling_pair(
        drought_risk,
        [
            (
                f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.CURRENT}",
                f"{DroughtCols.DROUGHT_SEVERITY_INDEX}_{Scenarios.FORECAST}",
            ),
            (
                f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}",
                f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}",
            ),
        ],
        feature_range,
    )

    # Reverse the polarity for precipitation
    drought_risk[f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}"] = (
        config.constants.score_max
        - drought_risk[f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.CURRENT}"]
    )
    drought_risk[f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}"] = (
        config.constants.score_max
        - drought_risk[f"{DroughtCols.PRECIP_SUMMER}_{Scenarios.FORECAST}"]
    )

    drought_risk = _calculate_composite_score(
        drought_risk,
        _DROUGHT_WEIGHTS,
        ExtremeWeatherRiskCols.DROUGHT,
    )

    drought_risk = min_max_scaling_pair(
        drought_risk,
        [
            (
                f"{ExtremeWeatherRiskCols.DROUGHT}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.DROUGHT}_{Scenarios.FORECAST}",
            )
        ],
        feature_range,
    )

    drought_risk = gpd.GeoDataFrame(
        drought_risk, geometry="geometry", crs=data_cleaning.BNG_CRS
    )

    feature_range = (config.constants.score_min, config.constants.score_max)
    _validate_index(
        drought_risk, [*DroughtCols, ExtremeWeatherRiskCols.DROUGHT], feature_range
    )

    _audit_index(
        drought_risk,
        [*DroughtCols, ExtremeWeatherRiskCols.DROUGHT],
        audit_path / "Extreme Weather" / "Drought Risk Index",
        feature_range,
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
            f"{StormCols.PRECIP_WINTER}_{Scenarios.CURRENT}",
            f"{StormCols.PRECIP_WINTER}_{Scenarios.FORECAST}",
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
            f"{StormCols.RAIN_DAYS}_{Scenarios.CURRENT}",
            f"{StormCols.PRECIP_WINTER}_{Scenarios.CURRENT}",
            f"{StormCols.PRECIP_WINTER}_{Scenarios.FORECAST}",
            f"{StormCols.WIND_SPEED}_{Scenarios.CURRENT}",
            f"{StormCols.WIND_SPEED}_{Scenarios.FORECAST}",
            f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.CURRENT}",
            f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.FORECAST}",
            f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.CURRENT}",
            f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    storm_risk = _iterative_spatial_infilling(
        storm_risk,
        [
            f"{StormCols.RAIN_DAYS}_{Scenarios.CURRENT}",
            f"{StormCols.PRECIP_WINTER}_{Scenarios.CURRENT}",
            f"{StormCols.PRECIP_WINTER}_{Scenarios.FORECAST}",
            f"{StormCols.WIND_SPEED}_{Scenarios.CURRENT}",
            f"{StormCols.WIND_SPEED}_{Scenarios.FORECAST}",
            f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.CURRENT}",
            f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.FORECAST}",
            f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.CURRENT}",
            f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.FORECAST}",
        ],
        _STORM_NEAREST_JOIN_MAX_DISTANCE,
    )

    storm_risk[f"{StormCols.WIND_SPEED}_{Scenarios.CURRENT}"] = storm_risk[
        f"{StormCols.WIND_SPEED}_{Scenarios.CURRENT}"
    ].apply(_wind_risk_scaled)
    storm_risk[f"{StormCols.WIND_SPEED}_{Scenarios.FORECAST}"] = storm_risk[
        f"{StormCols.WIND_SPEED}_{Scenarios.FORECAST}"
    ].apply(_wind_risk_scaled)

    feature_range = (config.constants.score_min, config.constants.score_max)
    storm_risk = min_max_scaling_pair(
        storm_risk,
        [
            (
                f"{StormCols.WIND_SPEED}_{Scenarios.CURRENT}",
                f"{StormCols.WIND_SPEED}_{Scenarios.FORECAST}",
            ),
            (
                f"{StormCols.PRECIP_WINTER}_{Scenarios.CURRENT}",
                f"{StormCols.PRECIP_WINTER}_{Scenarios.FORECAST}",
            ),
            (
                f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.CURRENT}",
                f"{StormCols.EXCEEDANCE_DAYS}_{Scenarios.FORECAST}",
            ),
            (
                f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.CURRENT}",
                f"{StormCols.WIND_DRIVEN_RAIN_INDEX}_{Scenarios.FORECAST}",
            ),
        ],
        feature_range,
    )

    # Scale rain days on its own, then duplicate
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range)
    storm_risk[f"{StormCols.RAIN_DAYS}_{Scenarios.CURRENT}"] = scaler.fit_transform(
        storm_risk[[f"{StormCols.RAIN_DAYS}_{Scenarios.CURRENT}"]]
    ).clip(*feature_range)
    storm_risk[f"{StormCols.RAIN_DAYS}_{Scenarios.FORECAST}"] = storm_risk[
        f"{StormCols.RAIN_DAYS}_{Scenarios.CURRENT}"
    ].clip(*feature_range)

    storm_risk = _calculate_composite_score(
        storm_risk,
        _STORM_WEIGHTS,
        ExtremeWeatherRiskCols.STORM,
    )

    storm_risk = min_max_scaling_pair(
        storm_risk,
        [
            (
                f"{ExtremeWeatherRiskCols.STORM}_{Scenarios.CURRENT}",
                f"{ExtremeWeatherRiskCols.STORM}_{Scenarios.FORECAST}",
            )
        ],
        feature_range,
    )

    storm_risk = gpd.GeoDataFrame(storm_risk, geometry="geometry", crs=data_cleaning.BNG_CRS)

    _validate_index(storm_risk, [*StormCols, ExtremeWeatherRiskCols.STORM], feature_range)

    _audit_index(
        storm_risk,
        [*StormCols, ExtremeWeatherRiskCols.STORM],
        audit_path / "Extreme Weather" / "Storm Risk Index",
        feature_range,
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
            for scenario in Scenarios:
                flooding_paths.append(
                    config.paths.model_input
                    / file_paths.FLOODING_MODEL_INPUT_PATH
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

    overlay_path = (
        config.paths.model_interim_output
        / file_paths.FLOODING_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH
    )

    # Read the direct overlay result, and filter to region
    # Eventually want to rename the layer to 'flooding_overlay'
    try:
        flooding_risk = gpd.read_file(
            overlay_path,
            mask=boundary,
            layer="flooding_overlay",
        )
    except ValueError:
        LOG.warning(
            "Layer 'flooding_overlay' not found, falling back to 'flood_overlay' layer."
        )
        flooding_risk = gpd.read_file(
            overlay_path,
            mask=boundary,
            layer="flood_overlay",
        )

    # Eventually want to rename columns in input data to 'flooding' rather than 'flood'
    flooding_risk = flooding_risk.rename(
        columns={
            f"rivers_sea_flood_risk_{Scenarios.CURRENT}": (
                f"{FloodingRiskCols.RIVERS_SEA}_{Scenarios.CURRENT}"
            ),
            f"rivers_sea_flood_risk_{Scenarios.FORECAST}": (
                f"{FloodingRiskCols.RIVERS_SEA}_{Scenarios.FORECAST}"
            ),
            f"surface_water_flood_risk_{Scenarios.CURRENT}": (
                f"{FloodingRiskCols.SURFACE_WATER}_{Scenarios.CURRENT}"
            ),
            f"surface_water_flood_risk_{Scenarios.FORECAST}": (
                f"{FloodingRiskCols.SURFACE_WATER}_{Scenarios.FORECAST}"
            ),
        }
    )

    # Map original risk categories to numeric scores
    for col in [
        f"{FloodingRiskCols.RIVERS_SEA}_{Scenarios.CURRENT}",
        f"{FloodingRiskCols.RIVERS_SEA}_{Scenarios.FORECAST}",
        f"{FloodingRiskCols.SURFACE_WATER}_{Scenarios.CURRENT}",
        f"{FloodingRiskCols.SURFACE_WATER}_{Scenarios.FORECAST}",
    ]:
        flooding_risk[col] = flooding_risk[col].map(_FLOODING_RISK_SCORE_MAP)

    # Fill NA values with 0 (no risk) since no data means no risk in the underlying data
    flooding_risk = flooding_risk.fillna(0)

    feature_range = (config.constants.score_min, config.constants.score_max)
    flooding_risk = min_max_scaling_pair(
        flooding_risk,
        [
            (
                f"{FloodingRiskCols.RIVERS_SEA}_{Scenarios.CURRENT}",
                f"{FloodingRiskCols.RIVERS_SEA}_{Scenarios.FORECAST}",
            ),
            (
                f"{FloodingRiskCols.SURFACE_WATER}_{Scenarios.CURRENT}",
                f"{FloodingRiskCols.SURFACE_WATER}_{Scenarios.FORECAST}",
            ),
        ],
        feature_range,
    )

    flooding_risk = _calculate_composite_score(
        flooding_risk,
        _FLOODING_WEIGHTS,
        MainHazardRiskCols.FLOODING,
    )

    flooding_risk = min_max_scaling_pair(
        flooding_risk,
        [
            (
                f"{MainHazardRiskCols.FLOODING}_{Scenarios.CURRENT}",
                f"{MainHazardRiskCols.FLOODING}_{Scenarios.FORECAST}",
            ),
        ],
        feature_range,
    )

    _validate_index(
        flooding_risk, [*FloodingRiskCols, MainHazardRiskCols.FLOODING], feature_range
    )

    feature_range = (config.constants.score_min, config.constants.score_max)
    _audit_index(
        flooding_risk,
        [*FloodingRiskCols, MainHazardRiskCols.FLOODING],
        audit_path / "Flooding" / "Flooding Risk Index",
        feature_range,
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
) -> None:
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
        shrink_swell[year][GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE] = shrink_swell[
            year
        ][GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE].map(
            _GROUND_STABILITY_RISK_SCORE_MAP
        )
        shrink_swell[year] = shrink_swell[year][
            [GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE, "geometry"]
        ]
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
        f"{hazard}_{suffix}" for hazard in GroundStabilityRiskCols for suffix in Scenarios
    ]

    for col in risk_cols:
        ground_stability[col] = pd.to_numeric(ground_stability[col], errors="coerce")

    ground_stability = _iterative_spatial_infilling(
        ground_stability, risk_cols, _GROUND_STABILITY_NEAREST_JOIN_MAX_DISTANCE
    )

    gs_pairs = [
        (f"{col}_{Scenarios.CURRENT}", f"{col}_{Scenarios.FORECAST}")
        for col in GroundStabilityRiskCols
    ]

    feature_range = (config.constants.score_min, config.constants.score_max)
    ground_stability = min_max_scaling_pair(ground_stability, gs_pairs, feature_range)

    ground_stability = _calculate_composite_score(
        ground_stability,
        _GROUND_STABILITY_WEIGHTS,
        MainHazardRiskCols.GROUND_STABILITY,
    )

    ground_stability = min_max_scaling_pair(
        ground_stability,
        [
            (
                f"{MainHazardRiskCols.GROUND_STABILITY}_{Scenarios.CURRENT}",
                f"{MainHazardRiskCols.GROUND_STABILITY}_{Scenarios.FORECAST}",
            )
        ],
        feature_range,
    )

    _validate_index(
        ground_stability,
        [*GroundStabilityRiskCols, MainHazardRiskCols.GROUND_STABILITY],
        feature_range,
    )

    _audit_index(
        ground_stability,
        [*GroundStabilityRiskCols, MainHazardRiskCols.GROUND_STABILITY],
        audit_path / "Ground Stability" / "Ground Stability Risk Index",
        feature_range,
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
    ncerm_giz[CoastalErosionRiskCols.GIZ] = config.constants.score_max

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
                        f"{MainHazardRiskCols.COASTAL_EROSION}_{Scenarios.CURRENT}",
                        f"{MainHazardRiskCols.COASTAL_EROSION}_{Scenarios.FORECAST}",
                        "geometry",
                    ],  # Empty GeoDataFrame
                    geometry="geometry",
                    crs=data_cleaning.BNG_CRS,
                ),
                config.paths.model_interim_output
                / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH,
            )
            return

        ncerm[year][CoastalErosionRiskCols.EROSION] = config.constants.score_max

        erosion_risk[scenario] = _overlay_and_clean(
            ncerm_giz, ncerm[year], target_crs=data_cleaning.BNG_CRS
        )

        # If either of the layers is empty, fill the missing column with 0 (no risk)
        for col in [CoastalErosionRiskCols.EROSION, CoastalErosionRiskCols.GIZ]:
            if col not in erosion_risk[scenario]:
                erosion_risk[scenario][col] = 0

        # Compute composite risk score
        erosion_risk[scenario][f"{MainHazardRiskCols.COASTAL_EROSION}"] = (
            erosion_risk[scenario][CoastalErosionRiskCols.EROSION]
            * _COASTAL_EROSION_WEIGHTS[CoastalErosionRiskCols.EROSION]
            + erosion_risk[scenario][CoastalErosionRiskCols.GIZ]
            * _COASTAL_EROSION_WEIGHTS[CoastalErosionRiskCols.GIZ]
        )

        erosion_risk[scenario] = erosion_risk[scenario].rename(
            columns={
                f"{MainHazardRiskCols.COASTAL_EROSION}": (
                    f"{MainHazardRiskCols.COASTAL_EROSION}_{scenario}"
                )
            }
        )

    coastal_erosion_risk = _overlay_and_clean(
        erosion_risk[Scenarios.CURRENT],
        erosion_risk[Scenarios.FORECAST],
        target_crs=data_cleaning.BNG_CRS,
    )

    coastal_erosion_risk = _iterative_spatial_infilling(
        coastal_erosion_risk,
        [
            f"{MainHazardRiskCols.COASTAL_EROSION}_{Scenarios.CURRENT}",
            f"{MainHazardRiskCols.COASTAL_EROSION}_{Scenarios.FORECAST}",
        ],
        nearest_join_max_distance=_COASTAL_EROSION_NEAREST_JOIN_MAX_DISTANCE,
    )
    coastal_erosion_risk = gpd.GeoDataFrame(coastal_erosion_risk, geometry="geometry")
    coastal_erosion_risk = coastal_erosion_risk[
        [
            f"{MainHazardRiskCols.COASTAL_EROSION}_{Scenarios.CURRENT}",
            f"{MainHazardRiskCols.COASTAL_EROSION}_{Scenarios.FORECAST}",
            "geometry",
        ]
    ]

    feature_range = (config.constants.score_min, config.constants.score_max)
    _validate_index(coastal_erosion_risk, [MainHazardRiskCols.COASTAL_EROSION], feature_range)

    _audit_index(
        coastal_erosion_risk,
        [MainHazardRiskCols.COASTAL_EROSION],
        audit_path / "Coastal Erosion" / "Coastal Erosion Risk Index",
        feature_range,
    )

    data_cleaning.write_to_file(
        coastal_erosion_risk,
        config.paths.model_interim_output
        / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Coastal erosion risk index calculation complete.")
