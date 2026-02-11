"""Apply functional rules to hazard and impact datasets, and normalise."""

### LOAD LIBRARIES
import logging
import pathlib
from functools import reduce

import data_cleaning
import file_paths
import geopandas as gpd
import model_config
import numpy as np
import pandas as pd
import sklearn
from shapely.geometry import box

LOG = logging.getLogger(__name__)

_SCENARIO_SUFFIXES = ["_current", "_forecast"]

_EXTREME_HEAT_RISK_THRESHOLD = 30
_EXTREME_COLD_RISK_THRESHOLD = 0

_WIND_SPEED_RISK_THRESHOLD_LOWER = (
    13.4  # 30 mph in m/s (should not exceed the upper threshold)
)
_WIND_SPEED_RISK_THRESHOLD_UPPER = 20.1  # 45 mph in m/s (should not exceed 25)

_PCT_OF_MEDIAN_FILTER_THRESHOLD_EXTREME_WEATHER = 0.035
_PCT_OF_MEDIAN_FILTER_THRESHOLD_STORM = 0.01

_EXTREME_WEATHER_WEIGHTS = {
    "heat_risk": 0.25,
    "cold_risk": 0.25,
    "drought_risk": 0.25,
    "storm_risk": 0.25,
}
_EXTREME_HEAT_WEIGHTS = {
    "max_temp_summer_risk": 0.5,
    "hot_summer_days": 0.25,
    "extreme_summer_days": 0.25,
}
_EXTREME_COLD_WEIGHTS = {"min_temp_winter_risk": 0.5, "frost_days": 0.25, "icing_days": 0.25}
_DROUGHT_WEIGHTS = {"drought_severity_index": 0.75, "precip_summer": 0.25}
_STORM_WEIGHTS = {
    "wind_speed_risk": 0.3,
    "avg_exceedance_days": 0.2,
    "precip_winter": 0.15,
    "rain_days": 0.15,
    "wind_driven_rain_index": 0.2,
}

_FLOOD_GRID_SIZE_M = 1000
_FLOOD_RISK_SCORE_MAP = {"Unavailable": 0, "Very low": 0, "Low": 1, "Medium": 2, "High": 3}
_FLOOD_WEIGHTS = {"rivers_sea_flood_risk": 0.5, "surface_water_flood_risk": 0.5}

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

_COASTAL_EROSION_YEAR_SCENARIO_MAP = {"2055": "current", "2105": "forecast"}
_COASTAL_EROSION_WEIGHTS = {"coastal_erosion_risk": 0.9, "giz_risk": 0.1}

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


def _iterative_spatial_infilling(
    risk_grid: gpd.GeoDataFrame, variables: list[str], max_iterations: int = 10
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
                "No further improvement  after %s iterations using spatial infilling. "
                "Switching to nearest join to fill remaining %s NA values.",
                i,
                current_na_count,
            )
            break

        prev_na_count = current_na_count

        risk_grid = _spatial_infill_na_grids(risk_grid, variables)

    # Fallback: nearest join for remaining NAs
    remaining_na = risk_grid[risk_grid[variables].isna().any(axis=1)]
    if not remaining_na.empty:
        nearest = gpd.sjoin_nearest(
            remaining_na, risk_grid.drop(remaining_na.index), how="left"
        )

        # Calculate the average value of the neighbouring grids
        nearest_avg = nearest.groupby(nearest.index)[
            [f"{var}_right" for var in variables]
        ].mean()
        for var in variables:
            na_condition = risk_grid[var].isna()
            # Fill with neighbour average
            risk_grid.loc[na_condition, var] = risk_grid.loc[na_condition].index.map(
                nearest_avg[f"{var}_right"]
            )

    return risk_grid


def _create_grid(
    xmin: float, ymin: float, xmax: float, ymax: float, cell_size: int
) -> gpd.GeoDataFrame:
    """Take bounds and a cell size and return a grid of the given size within the bounds."""
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
    return gpd.GeoDataFrame(geometry=grid_cells, crs=data_cleaning._BNG_CRS)


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
    for scenario in ["current", "forecast"]:
        col_name = f"{base_col}_{scenario}"
        out_name = f"{output_col}_{scenario}"

        if invert:
            risk_data[out_name] = np.where(
                risk_data[col_name] > threshold, 0, -risk_data[col_name]
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
    for scenario in ["current", "forecast"]:
        risk_data[f"{output_col}_{scenario}"] = sum(
            risk_data[f"{col}_{scenario}"] * weight for col, weight in weights.items()
        )

    return risk_data


def _filter_out_small_geometries(
    risk_data: gpd.GeoDataFrame, pct_of_median: float
) -> gpd.GeoDataFrame:
    """Filter out geometries with area less than a given percentage of the median area."""
    risk_data["area"] = risk_data.geometry.area
    threshold = risk_data["area"].median() * pct_of_median
    len_before_filter = len(risk_data)
    risk_data = risk_data[risk_data["area"] > threshold]  # Filter out tiny geometries
    filter_removed = len_before_filter - len(risk_data)
    LOG.info(
        "Filtered out %s geometries of %s total geometries below area threshold of %s "
        "(%s of median area).",
        filter_removed,
        len_before_filter,
        threshold,
        pct_of_median * 100,
    )
    return risk_data.drop(columns=["area"])


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
    boundary = gpd.read_file(config.other_input.boundary_path)

    _extreme_weather_index(config)
    _flooding_index(config, boundary)
    _ground_stability_index(config)
    _coastal_erosion_index(config)


## HAZARDS

### EXTREME WEATHER


def _extreme_weather_index(config: model_config.Config) -> None:
    """Combine extreme heat, extreme cold, drought and storm indexes into a single index."""
    LOG.info("Calculating extreme weather risk index...")
    tfn_hazard_grid = gpd.read_file(
        config.paths.model_input / file_paths.HAZARD_GRID_MODEL_INPUT_PATH
    )

    tfn_extreme_heat = _extreme_heat_index(config, tfn_hazard_grid)
    tfn_extreme_cold = _extreme_cold_index(config, tfn_hazard_grid)
    tfn_drought = _drought_index(config, tfn_hazard_grid)
    tfn_storm = _storm_index(config, tfn_hazard_grid)

    LOG.info("Combining extreme heat, extreme cold, drought and storm indexes.")
    tfn_extreme_weather_merge = tfn_extreme_heat[
        ["grid_id", "part", "heat_risk_current", "heat_risk_forecast"]
    ].merge(
        tfn_extreme_cold[
            ["grid_id", "part", "cold_risk_current", "cold_risk_forecast", "geometry"]
        ],
        on=["grid_id", "part"],
        how="inner",
    )

    tfn_extreme_weather_merge = gpd.GeoDataFrame(
        tfn_extreme_weather_merge, geometry="geometry", crs="EPSG:3857"
    )
    tfn_extreme_weather_merge = tfn_extreme_weather_merge.drop(columns=["grid_id", "part"])

    tfn_extreme_weather_merge = tfn_extreme_weather_merge.to_crs(data_cleaning._BNG_CRS)
    tfn_drought = tfn_drought.to_crs(data_cleaning._BNG_CRS)
    tfn_storm = tfn_storm.to_crs(data_cleaning._BNG_CRS)

    tfn_extreme_weather_overlay = gpd.overlay(
        tfn_extreme_weather_merge,
        tfn_drought[["drought_risk_current", "drought_risk_forecast", "geometry"]],
        how="union",
    )

    tfn_extreme_weather_overlay = gpd.overlay(
        tfn_extreme_weather_overlay,
        tfn_storm[["storm_risk_current", "storm_risk_forecast", "geometry"]],
        how="union",
    )

    tfn_extreme_weather_overlay = _filter_out_small_geometries(
        tfn_extreme_weather_overlay, _PCT_OF_MEDIAN_FILTER_THRESHOLD_EXTREME_WEATHER
    )

    tfn_extreme_weather_risk = _iterative_spatial_infilling(
        tfn_extreme_weather_overlay,
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
    )

    tfn_extreme_weather_risk = data_cleaning.explode_to_polygons(tfn_extreme_weather_risk)
    tfn_extreme_weather_risk = tfn_extreme_weather_risk.drop(columns=["part"])

    tfn_extreme_weather_risk = _calculate_composite_score(
        tfn_extreme_weather_risk,
        _EXTREME_WEATHER_WEIGHTS,
        "extreme_weather_risk",
    )

    tfn_extreme_weather_risk = min_max_scaling_pair(
        tfn_extreme_weather_risk,
        [("extreme_weather_risk_current", "extreme_weather_risk_forecast")],
    )

    tfn_extreme_weather_risk = gpd.GeoDataFrame(tfn_extreme_weather_risk, geometry="geometry")

    data_cleaning.write_to_file(
        tfn_extreme_weather_risk,
        config.paths.model_interim_output
        / file_paths.EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Extreme weather risk index calculation complete.")


#### EXTREME HEAT


def _extreme_heat_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Combine several datasets into extreme heat index by merging on their hazard grid."""
    LOG.info("Calculating extreme heat index...")
    tfn_temp_max = pd.read_csv(config.paths.model_input / file_paths.TEMP_MAX_MODEL_INPUT_PATH)
    tfn_hsd = pd.read_csv(
        config.paths.model_input / file_paths.HOT_SUMMER_DAYS_MODEL_INPUT_PATH
    )
    tfn_esd = pd.read_csv(
        config.paths.model_input / file_paths.EXTREME_SUMMER_DAYS_MODEL_INPUT_PATH
    )

    tfn_extreme_heat = _merge_on_key([tfn_temp_max, tfn_hsd, tfn_esd], hazard_grid, "grid_id")

    tfn_extreme_heat = _calculate_risk_threshold(
        tfn_extreme_heat,
        "max_temp_summer",
        "max_temp_summer_risk",
        _EXTREME_HEAT_RISK_THRESHOLD,
    )

    tfn_extreme_heat = min_max_scaling_pair(
        tfn_extreme_heat,
        [
            ("max_temp_summer_risk_current", "max_temp_summer_risk_forecast"),
            ("hot_summer_days_current", "hot_summer_days_forecast"),
            ("extreme_summer_days_current", "extreme_summer_days_forecast"),
        ],
    )

    tfn_extreme_heat = _calculate_composite_score(
        tfn_extreme_heat,
        _EXTREME_HEAT_WEIGHTS,
        "heat_risk",
    )

    tfn_extreme_heat = min_max_scaling_pair(
        tfn_extreme_heat, [("heat_risk_current", "heat_risk_forecast")]
    )

    LOG.info("Extreme heat index calculation complete.")
    return gpd.GeoDataFrame(tfn_extreme_heat, geometry="geometry")


#### EXTREME COLD


def _extreme_cold_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Combine several datasets into extreme cold index by merging on their hazard grid."""
    LOG.info("Calculating extreme cold index...")
    tfn_temp_min = pd.read_csv(config.paths.model_input / file_paths.TEMP_MIN_MODEL_INPUT_PATH)
    tfn_frost = pd.read_csv(config.paths.model_input / file_paths.FROST_DAYS_MODEL_INPUT_PATH)
    tfn_icing = pd.read_csv(config.paths.model_input / file_paths.ICING_DAYS_MODEL_INPUT_PATH)

    tfn_extreme_cold = _merge_on_key(
        [tfn_temp_min, tfn_frost, tfn_icing], hazard_grid, "grid_id"
    )

    tfn_extreme_cold = _calculate_risk_threshold(
        tfn_extreme_cold,
        "min_temp_winter",
        "min_temp_winter_risk",
        _EXTREME_COLD_RISK_THRESHOLD,
        invert=True,
    )

    tfn_extreme_cold = min_max_scaling_pair(
        tfn_extreme_cold,
        [
            ("min_temp_winter_risk_current", "min_temp_winter_risk_forecast"),
            ("frost_days_current", "frost_days_forecast"),
            ("icing_days_current", "icing_days_forecast"),
        ],
    )

    tfn_extreme_cold = _calculate_composite_score(
        tfn_extreme_cold,
        _EXTREME_COLD_WEIGHTS,
        "cold_risk",
    )

    tfn_extreme_cold = min_max_scaling_pair(
        tfn_extreme_cold, [("cold_risk_current", "cold_risk_forecast")]
    )

    LOG.info("Extreme cold index calculation complete.")
    return gpd.GeoDataFrame(tfn_extreme_cold, geometry="geometry")


#### DROUGHT


def _drought_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Combine several datasets into single drought index with spatial overlay."""
    LOG.info("Calculating drought index...")
    tfn_drought = gpd.read_file(
        config.paths.model_input / file_paths.DROUGHT_INDEX_MODEL_INPUT_PATH
    )
    tfn_precip_sum = pd.read_csv(
        config.paths.model_input / file_paths.SUMMER_PRECIP_MODEL_INPUT_PATH
    )

    tfn_precip_sum_grid = tfn_precip_sum.merge(hazard_grid, on="grid_id")
    tfn_precip_sum_gdf = gpd.GeoDataFrame(
        tfn_precip_sum_grid, geometry="geometry", crs=hazard_grid.crs
    )
    tfn_precip_sum_gdf = tfn_precip_sum_gdf[
        ["precip_summer_current", "precip_summer_forecast", "geometry"]
    ]

    tfn_drought = tfn_drought.to_crs(data_cleaning._BNG_CRS)
    tfn_precip_sum_gdf = tfn_precip_sum_gdf.to_crs(data_cleaning._BNG_CRS)

    tfn_drought_overlay = gpd.overlay(tfn_precip_sum_gdf, tfn_drought, how="union")
    tfn_drought_overlay = _iterative_spatial_infilling(
        tfn_drought_overlay,
        [
            "precip_summer_current",
            "precip_summer_forecast",
            "drought_severity_index_current",
            "drought_severity_index_forecast",
        ],
    )

    tfn_drought_overlay = data_cleaning.explode_to_polygons(tfn_drought_overlay)
    tfn_drought_risk = tfn_drought_overlay[
        [
            "drought_severity_index_current",
            "drought_severity_index_forecast",
            "precip_summer_current",
            "precip_summer_forecast",
            "geometry",
        ]
    ]

    tfn_drought_risk = min_max_scaling_pair(
        tfn_drought_risk,
        [
            ("drought_severity_index_current", "drought_severity_index_forecast"),
            ("precip_summer_current", "precip_summer_forecast"),
        ],
    )

    # Reverse the polarity for precipitation
    tfn_drought_risk["precip_summer_current"] = 100 - tfn_drought_risk["precip_summer_current"]
    tfn_drought_risk["precip_summer_forecast"] = (
        100 - tfn_drought_risk["precip_summer_forecast"]
    )

    tfn_drought_risk = _calculate_composite_score(
        tfn_drought_risk,
        _DROUGHT_WEIGHTS,
        "drought_risk",
    )

    tfn_drought_risk = min_max_scaling_pair(
        tfn_drought_risk, [("drought_risk_current", "drought_risk_forecast")]
    )
    LOG.info("Drought index calculation complete.")
    return gpd.GeoDataFrame(tfn_drought_risk, geometry="geometry")


#### STORMS


def _storm_index(
    config: model_config.Config, hazard_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Combine several datasets into a single storm index with a spatial overlay."""
    LOG.info("Calculating storm index...")
    tfn_precip_win = pd.read_csv(
        config.paths.model_input / file_paths.WINTER_PRECIP_MODEL_INPUT_PATH
    )
    tfn_rain_days = gpd.read_file(
        config.paths.model_input / file_paths.RAIN_DAYS_MODEL_INPUT_PATH
    )
    tfn_wind_spd = gpd.read_file(
        config.paths.model_input / file_paths.WIND_SPEED_MODEL_INPUT_PATH
    )
    tfn_wdr = gpd.read_file(
        config.paths.model_input / file_paths.WIND_DRIVEN_RAIN_MODEL_INPUT_PATH
    )

    tfn_precip_win_grid = tfn_precip_win.merge(
        hazard_grid, on="grid_id", how="left", validate="one_to_many"
    )
    tfn_precip_win_gdf = gpd.GeoDataFrame(
        tfn_precip_win_grid, geometry="geometry", crs=hazard_grid.crs
    )
    tfn_precip_win_gdf = tfn_precip_win_gdf[
        ["precip_winter_current", "precip_winter_forecast", "geometry"]
    ]

    tfn_wind_spd = tfn_wind_spd.to_crs(data_cleaning._BNG_CRS)
    tfn_rain_days = tfn_rain_days.to_crs(data_cleaning._BNG_CRS)
    tfn_precip_win_gdf = tfn_precip_win_gdf.to_crs(data_cleaning._BNG_CRS)
    tfn_wdr = tfn_wdr.to_crs(data_cleaning._BNG_CRS)

    tfn_storm_overlay = gpd.overlay(tfn_wind_spd, tfn_rain_days, how="union")
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_precip_win_gdf, how="union")
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_wdr, how="union")

    tfn_storm_overlay = tfn_storm_overlay[
        [
            "rain_days_current",
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

    tfn_storm_overlay = _filter_out_small_geometries(
        tfn_storm_overlay, _PCT_OF_MEDIAN_FILTER_THRESHOLD_STORM
    )

    tfn_storm_overlay = _iterative_spatial_infilling(
        tfn_storm_overlay,
        [
            "rain_days_current",
            "precip_winter_current",
            "precip_winter_forecast",
            "wind_speed_99th_percentile_current",
            "wind_speed_99th_percentile_forecast",
            "avg_exceedance_days_current",
            "avg_exceedance_days_forecast",
            "wind_driven_rain_index_current",
            "wind_driven_rain_index_forecast",
        ],
    )

    tfn_storm_risk = data_cleaning.explode_to_polygons(tfn_storm_overlay)
    tfn_storm_risk = tfn_storm_risk.drop(columns=["part"])

    tfn_storm_risk["wind_speed_risk_current"] = tfn_storm_risk[
        "wind_speed_99th_percentile_current"
    ].apply(_wind_risk_scaled)
    tfn_storm_risk["wind_speed_risk_forecast"] = tfn_storm_risk[
        "wind_speed_99th_percentile_forecast"
    ].apply(_wind_risk_scaled)

    tfn_storm_risk = min_max_scaling_pair(
        tfn_storm_risk,
        [
            ("wind_speed_risk_current", "wind_speed_risk_forecast"),
            ("precip_winter_current", "precip_winter_forecast"),
            ("avg_exceedance_days_current", "avg_exceedance_days_forecast"),
            ("wind_driven_rain_index_current", "wind_driven_rain_index_forecast"),
        ],
    )

    # Scale rain days on its own, then duplicate
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
    tfn_storm_risk["rain_days_current"] = scaler.fit_transform(
        tfn_storm_risk[["rain_days_current"]]
    )
    tfn_storm_risk["rain_days_forecast"] = tfn_storm_risk["rain_days_current"]

    tfn_storm_risk = _calculate_composite_score(
        tfn_storm_risk,
        _STORM_WEIGHTS,
        "storm_risk",
    )

    tfn_storm_risk = min_max_scaling_pair(
        tfn_storm_risk, [("storm_risk_current", "storm_risk_forecast")]
    )
    LOG.info("Storm index calculation complete.")
    return gpd.GeoDataFrame(tfn_storm_risk, geometry="geometry")


def _wind_risk_scaled(speed_metres_per_second: float) -> float:
    """Calculate wind risk value given a wind speed, based on classification rule."""
    if speed_metres_per_second < _WIND_SPEED_RISK_THRESHOLD_LOWER:  # Below 30 mph
        return 0
    if speed_metres_per_second <= _WIND_SPEED_RISK_THRESHOLD_UPPER:  # between 30 and 45 mph
        return (speed_metres_per_second - _WIND_SPEED_RISK_THRESHOLD_LOWER) / (
            _WIND_SPEED_RISK_THRESHOLD_UPPER - _WIND_SPEED_RISK_THRESHOLD_LOWER
        )  # Scale to 0 - 1
    return 1 + (speed_metres_per_second - _WIND_SPEED_RISK_THRESHOLD_UPPER) / (
        25 - _WIND_SPEED_RISK_THRESHOLD_UPPER
    )  # Scale beyond 1


### FLOODING


def _flooding_index(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Combine RoFRS & RoFSW into a single risk score by upscaling them to a common grid."""
    LOG.info("Calculating flood risk index...")
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

    LOG.info("Proccessing current flood risk...")
    tfn_flood_risk_c = _upscale_to_grid(
        config, flood_grid, current_flood_scenario_map, "current"
    )
    LOG.info("Current flood risk processing complete.")

    LOG.info("Proccessing forecast flood risk...")
    tfn_flood_risk_f = _upscale_to_grid(
        config, flood_grid, forecast_flood_scenario_map, "forecast"
    )
    LOG.info("Forecast flood risk processing complete.")

    # Merge on geometry columns (these will be exactly the same)
    tfn_flood_risk = tfn_flood_risk_c.merge(tfn_flood_risk_f, on="geometry", how="left")

    tfn_flood_risk = min_max_scaling_pair(
        tfn_flood_risk,
        [
            ("rivers_sea_flood_risk_current", "rivers_sea_flood_risk_forecast"),
            ("surface_water_flood_risk_current", "surface_water_flood_risk_forecast"),
        ],
    )

    tfn_flood_risk = _calculate_composite_score(
        tfn_flood_risk,
        _FLOOD_WEIGHTS,
        "flood_risk",
    )

    tfn_flood_risk = min_max_scaling_pair(
        tfn_flood_risk, [("flood_risk_current", "flood_risk_forecast")]
    )

    data_cleaning.write_to_file(
        tfn_flood_risk,
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


def _process_flood_layer(
    flood_grid: gpd.GeoDataFrame,
    file_path: pathlib.Path,
    risk_column: str,
) -> gpd.GeoDataFrame:
    """Read a flood layer, assigns risk, and return area-weighted flood risk."""
    layer = gpd.read_file(file_path)
    layer[risk_column] = layer["Risk_band"].map(_FLOOD_RISK_SCORE_MAP)
    return _area_weighted_flood_assignment(flood_grid, layer, risk_column)


def _upscale_to_grid(
    config: model_config.Config,
    flood_grid: gpd.GeoDataFrame,
    scenario_map: list[tuple[pathlib.Path, str]],
    scenario: str,
) -> gpd.GeoDataFrame:
    """Upscales each flood layer to the common grid and writes to file."""
    result = flood_grid.copy()
    for path, risk_col in scenario_map:
        result = _process_flood_layer(
            result,
            config.paths.model_input / path,
            risk_col,
        )

        data_cleaning.write_to_file(
            result,
            config.paths.model_interim_output
            / file_paths.FLOOD_RISK_SCENARIO_MODEL_INTERIM_OUTPUT_PATH
            / f"tfn_flood_risk_{scenario}.gpkg",
        )

    return result


def _area_weighted_flood_assignment(
    grid: gpd.GeoDataFrame, flood_gdf: gpd.GeoDataFrame, risk_column: str
) -> gpd.GeoDataFrame:
    """Assign flood risk to grid squares using an area-weighted average."""
    len_before_upscale = len(flood_gdf)
    # Spatial join to find intersecting polygons
    flood_risk_join = gpd.sjoin(
        grid, flood_gdf[[risk_column, "geometry"]], how="left", predicate="intersects"
    )

    # Retrieve flood polygon geometry using index_right
    flood_risk_join = flood_risk_join.merge(
        flood_gdf[[risk_column, "geometry"]],
        left_on="index_right",
        right_index=True,
        suffixes=("", "_flood"),
    )

    # Get geometry of each intersection between grid and flood polygon
    flood_risk_join["intersection"] = flood_risk_join.apply(
        lambda row: row["geometry"].intersection(row["geometry_flood"]), axis=1
    )

    # Calculate area of each intersection
    flood_risk_join["area"] = flood_risk_join["intersection"].area

    # Compute area weighted average flood risk per grid cell
    weighted_avg_flood = flood_risk_join.groupby(flood_risk_join.index).apply(
        lambda group: (group[risk_column] * group["area"]).sum() / group["area"].sum()
    )

    # Assign weighted average flood risk back to the original grid
    grid[risk_column] = weighted_avg_flood

    # Fill missing values with 0 (no risk)
    num_na_rows = grid[risk_column].isna().sum()
    grid[risk_column] = grid[risk_column].fillna(0)
    LOG.info("Filled %s NA values in  flood data column %s with 0.", num_na_rows, risk_column)

    len_after_upscale = len(grid)
    LOG.info(
        "Upscaled flood layer %s from %s geometries to %s grid cells "
        "using area-weighted average.",
        risk_column,
        len_before_upscale,
        len_after_upscale,
    )

    return grid


### GROUND STABILITY


def _ground_stability_index(config: model_config.Config) -> None:
    """Combine GeoSure & GeoClimate risk into a single index, using a spatial overlay."""
    LOG.info("Calculating ground stability risk index...")
    tfn_geosure = gpd.read_file(config.paths.model_input / file_paths.GEOSURE_MODEL_INPUT_PATH)
    tfn_geosure = tfn_geosure.to_crs(data_cleaning._BNG_CRS)

    tfn_ss = {}
    ground_stability = {}
    for year, scenario in _GEOCLIMATE_YEAR_SCENARIO_MAP.items():
        tfn_ss[year] = gpd.read_file(
            config.paths.model_input
            / file_paths.GEOCLIMATE_MODEL_INPUT_PATH
            / f"tfn_bgs_ss_{year}.gpkg"
        )
        tfn_ss[year]["shrink_swell_geoclimate_risk"] = tfn_ss[year][
            "shrink_swell_geoclimate_risk"
        ].map(_GROUND_STABILITY_RISK_SCORE_MAP)
        tfn_ss[year] = tfn_ss[year][["shrink_swell_geoclimate_risk", "geometry"]]
        tfn_ss[year] = tfn_ss[year].to_crs(data_cleaning._BNG_CRS)
        ground_stability[scenario] = gpd.overlay(tfn_geosure, tfn_ss[year], how="union")
        ground_stability[scenario] = ground_stability[scenario].rename(
            columns={
                col: f"{col}_{scenario}"
                for col in ground_stability[scenario].columns
                if col != "geometry"
            }
        )
        ground_stability[scenario] = ground_stability[scenario].to_crs(data_cleaning._BNG_CRS)

    tfn_ground_stability = gpd.overlay(
        ground_stability["current"], ground_stability["forecast"], how="union"
    )

    tfn_ground_stability = data_cleaning.explode_to_polygons(tfn_ground_stability)

    risk_cols = [
        f"{hazard}_risk{suffix}"
        for hazard in _GEOSURE_HAZARDS
        for suffix in _SCENARIO_SUFFIXES
    ]

    for col in risk_cols:
        tfn_ground_stability[col] = pd.to_numeric(tfn_ground_stability[col], errors="coerce")

    tfn_ground_stability = _iterative_spatial_infilling(tfn_ground_stability, risk_cols)

    gs_pairs = [(f"{col}_risk_current", f"{col}_risk_forecast") for col in _GEOSURE_HAZARDS]

    tfn_ground_stability = min_max_scaling_pair(tfn_ground_stability, gs_pairs)

    tfn_ground_stability = _calculate_composite_score(
        tfn_ground_stability,
        _GROUND_STABILITY_WEIGHTS,
        "ground_stability_risk",
    )

    tfn_ground_stability = min_max_scaling_pair(
        tfn_ground_stability,
        [("ground_stability_risk_current", "ground_stability_risk_forecast")],
    )

    data_cleaning.write_to_file(
        tfn_ground_stability,
        config.paths.model_interim_output
        / file_paths.GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Ground stability risk index calculation complete.")


### COASTAL EROSION


def _coastal_erosion_index(config: model_config.Config) -> None:
    """Combine erosion and ground stability risk into single index using a spatial overlay."""
    LOG.info("Calculating coastal erosion risk index...")
    tfn_ncerm_giz = gpd.read_file(
        config.paths.model_input / file_paths.GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH
    )
    tfn_ncerm_giz["giz_risk"] = 1

    tfn_ncerm = {}
    tfn_erosion_risk = {}
    for year, scenario in _COASTAL_EROSION_YEAR_SCENARIO_MAP.items():
        tfn_ncerm[year] = gpd.read_file(
            config.paths.model_input
            / file_paths.COASTAL_EROSION_MODEL_INPUT_PATH
            / f"tfn_ncerm_smp_{year}_70CC.gpkg"
        )
        tfn_ncerm[year]["coastal_erosion_risk"] = 1

        tfn_ncerm_giz = tfn_ncerm_giz.to_crs(tfn_ncerm[year].crs)
        tfn_erosion_risk[scenario] = gpd.overlay(tfn_ncerm_giz, tfn_ncerm[year], how="union")

        tfn_erosion_risk[scenario] = data_cleaning.explode_to_polygons(
            tfn_erosion_risk[scenario]
        )
        # Normalise risk values
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
        normalised_values = scaler.fit_transform(
            tfn_erosion_risk[scenario][["coastal_erosion_risk", "giz_risk"]]
        )

        # Compute composite risk score
        tfn_erosion_risk[scenario]["coastal_erosion_risk"] = (
            normalised_values[:, 0] * _COASTAL_EROSION_WEIGHTS["coastal_erosion_risk"]
            + normalised_values[:, 1] * _COASTAL_EROSION_WEIGHTS["giz_risk"]
        )

        tfn_erosion_risk[scenario] = tfn_erosion_risk[scenario].rename(
            columns={"coastal_erosion_risk": f"coastal_erosion_risk_{scenario}"}
        )

    tfn_coastal_erosion_risk = gpd.overlay(
        tfn_erosion_risk["current"], tfn_erosion_risk["forecast"], how="union"
    )

    tfn_coastal_erosion_risk = tfn_coastal_erosion_risk.fillna(0)
    tfn_coastal_erosion_risk = gpd.GeoDataFrame(tfn_coastal_erosion_risk, geometry="geometry")
    tfn_coastal_erosion_risk = tfn_coastal_erosion_risk[
        ["coastal_erosion_risk_current", "coastal_erosion_risk_forecast", "geometry"]
    ]

    data_cleaning.write_to_file(
        tfn_coastal_erosion_risk,
        config.paths.model_interim_output
        / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH,
    )

    LOG.info("Coastal erosion risk index calculation complete.")
