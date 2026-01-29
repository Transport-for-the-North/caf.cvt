"""Apply functional rules to hazard and impact datasets, and normalise."""

### LOAD LIBRARIES
import logging
import pathlib
from functools import reduce

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from sklearn.preprocessing import MinMaxScaler

from cvt import config, data_cleaning

LOG = logging.getLogger(__name__)

WIND_RISK_THRESHOLD_LOWER = 13.4  # 30 mph in m/s
WIND_RISK_THRESHOLD_UPPER = 20.1  # 45 mph in m/s

### GENERAL FUNCTIONS


def min_max_scaling_pair(
    df: pd.DataFrame, pairs: list[tuple[str, str]], feature_range: tuple[int, int] = (0, 100)
) -> pd.DataFrame:
    """
    Scale paired columns jointly using Min-Max scaling.

    For each tuple ``(col_c, col_f)`` in ``pairs``, this function computes a
    single minimum and maximum across the combined values of both columns and
    applies a shared ``MinMaxScaler`` (from scikit-learn) with the provided
    ``feature_range``. This ensures the two columns in each pair are scaled
    using the same mapping so they are directly comparable.

    Parameters
    ----------
        df: pd.DataFrame
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
    for col_c, col_f in pairs:
        # Combine both columns into one array for global min/max
        combined_values = df[[col_c, col_f]].to_numpy().flatten().reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(combined_values)

        # Transform each column using the same scaler
        df[col_c] = scaler.transform(df[[col_c]].values)
        df[col_f] = scaler.transform(df[[col_f]].values)

    return df


def _spatial_smooth_zero_grids(
    gdf: gpd.GeoDataFrame, variables: list[str]
) -> gpd.GeoDataFrame:
    """Apply spatial smoothing to GeoDataFrame on given variables."""
    neighbours = gpd.sjoin(
        gdf, gdf, how="left", predicate="touches"
    )  # Find neighbouring grids

    # Calculate the average value of the neighbouring grids
    neighbours_avg = neighbours.groupby(neighbours.index)[
        [f"{var}_right" for var in variables]
    ].mean()

    for var in variables:
        # Set condition as variable is NA
        na_condition = gdf[var].isna()

        # Fill with neighbour average
        gdf.loc[na_condition, var] = gdf.loc[na_condition].index.map(
            neighbours_avg[f"{var}_right"]
        )

    return gdf


def _iterative_spatial_smoothing(
    gdf: gpd.GeoDataFrame, variables: list[str], max_iterations: int = 10
) -> gpd.GeoDataFrame:
    """Apply spatial smoothing iteratively to GeoDataFrame on given variables."""
    prev_na_count = None

    for i in range(max_iterations):
        # Count current NA values
        current_na_count = gdf[variables].isna().sum().sum()

        # Stop if all filled
        if current_na_count == 0:
            LOG.info("All NA values filled after %s iterations.", i)
            return gdf

        # Stop if no improvement
        if prev_na_count is not None and current_na_count == prev_na_count:
            LOG.info("No further improvement. Switching to nearest join.")
            break

        prev_na_count = current_na_count

        gdf = _spatial_smooth_zero_grids(gdf, variables)

    # Fallback: nearest join for remaining NAs
    remaining_na = gdf[gdf[variables].isna().any(axis=1)]
    if not remaining_na.empty:
        nearest = gpd.sjoin_nearest(remaining_na, gdf.drop(remaining_na.index), how="left")

        # Calculate the average value of the neighbouring grids
        nearest_avg = nearest.groupby(nearest.index)[
            [f"{var}_right" for var in variables]
        ].mean()
        for var in variables:
            na_condition = gdf[var].isna()
            # Fill with neighbour average
            gdf.loc[na_condition, var] = gdf.loc[na_condition].index.map(
                nearest_avg[f"{var}_right"]
            )

    return gdf


def _create_grid(bounds: np.ndarray, cell_size: int) -> gpd.GeoDataFrame:
    """Take bounds and a cell size and return a grid of the given size within the bounds."""
    xmin, ymin, xmax, ymax = bounds
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
    return gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:27700")


def _merge_on_key(
    dfs: list[pd.DataFrame], grid: gpd.GeoDataFrame, key: str
) -> gpd.GeoDataFrame:
    """Merge dataframes into single dataframe on common key, then merge onto a common grid."""
    merged = reduce(lambda left, right: left.merge(right, on=key, how="outer"), dfs)
    merged_df = merged.merge(grid, on=key, how="left", validate="one_to_many")
    return gpd.GeoDataFrame(merged_df, geometry="geometry", crs=grid.crs)


def _calculate_risk_threshold(
    df: pd.DataFrame, base_col: str, output_col: str, threshold: int, invert: bool = False
) -> pd.DataFrame:
    """Calculate risk level of a given column based on a threshold."""
    for tp in ["c", "f"]:
        col_name = f"{base_col}_{tp}"
        out_name = f"{output_col}_{tp}"

        if invert:
            df[out_name] = np.where(df[col_name] > threshold, 0, -df[col_name])
        else:
            df[out_name] = np.where(df[col_name] < threshold, 0, df[col_name] - threshold)

    return df


def _calculate_composite_score(
    df: pd.DataFrame, weights: dict[str, float], output_col: str
) -> pd.DataFrame:
    """Calculate composite score given a dataframe with variables and corresponding weights."""
    for tp in ["c", "f"]:
        df[f"{output_col}_{tp}"] = sum(
            df[f"{col}_{tp}"] * weight for col, weight in weights.items()
        )

    return df


def _overlay_normalise(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    risk_cols: list[str],
    combined_risk_name: str,
    weights: dict[str, float],
) -> gpd.GeoDataFrame:
    """Overlay two GeoDataFrames, then normalise and calculate a combined risk score."""
    # Overlay two gdf's
    gdf1 = gdf1.to_crs(gdf2.crs)
    composite_gdf = gpd.overlay(gdf2, gdf1, how="union")

    composite_gdf = data_cleaning.explode_to_polygons(composite_gdf)

    # Fill NA values with 0, indicating no risk
    composite_gdf[risk_cols] = composite_gdf[risk_cols].fillna(0)

    # Normalise risk values
    scaler = MinMaxScaler(feature_range=(0, 100))
    normalised_values = scaler.fit_transform(composite_gdf[risk_cols])

    # Compute composite risk score
    composite_gdf[combined_risk_name] = (
        normalised_values[:, 0] * weights[risk_cols[0]]
        + normalised_values[:, 1] * weights[risk_cols[1]]
    )

    return composite_gdf


def _filter_out_small_geometries(
    gdf: gpd.GeoDataFrame, pct_of_median: float
) -> gpd.GeoDataFrame:
    """Filter out geometries with area less than a given percentage of the median area."""
    gdf["area"] = gdf.geometry.area
    threshold = gdf["area"].median() * pct_of_median  # Threshold: 3.5% of median
    gdf = gdf[gdf["area"] > threshold]  # Filter out tiny geometries
    return gdf.drop(columns=["area"])


# FUNCTIONAL RULES


def apply_functional_rules(cfg: config.Config) -> None:
    """
    Apply functional rules to hazard datasets and spatially combine to generate risk indices.

    Create extreme weather, flooding, ground stability, and coastal erosion risk indices by
    combining spatially, either on a shared grid or via spatial overlay. Uses functional rules
    to normalise and classify risk levels.

    Parameters
    ----------
    cfg : Config
        Main config for the model, containing paths and settings.
    """
    boundary = gpd.read_file(cfg.other_input.boundary_path)

    _extreme_weather_index(cfg)
    _flooding_index(cfg, boundary)
    _ground_stability_index(cfg)
    _coastal_erosion_index(cfg)


## HAZARDS

### EXTREME WEATHER


def _extreme_weather_index(cfg: config.Config) -> None:
    """Combine extreme heat, extreme cold, drought and storm indexes into a single index."""
    tfn_common_grid = gpd.read_file(
        cfg.paths.model_input / "Other" / "TfN Common Grid" / "tfn_common_grid.gpkg"
    )

    tfn_extreme_heat = _extreme_heat_index(cfg, tfn_common_grid)
    tfn_extreme_cold = _extreme_cold_index(cfg, tfn_common_grid)
    tfn_drought = _drought_index(cfg, tfn_common_grid)
    tfn_storm = _storm_index(cfg, tfn_common_grid)

    tfn_extreme_weather_merge = tfn_extreme_heat[
        ["grid_id", "part", "heat_risk_c", "heat_risk_f"]
    ].merge(
        tfn_extreme_cold[["grid_id", "part", "cold_risk_c", "cold_risk_f", "geometry"]],
        on=["grid_id", "part"],
        how="inner",
    )

    tfn_extreme_weather_merge = gpd.GeoDataFrame(
        tfn_extreme_weather_merge, geometry="geometry", crs="EPSG:3857"
    )
    tfn_extreme_weather_merge = tfn_extreme_weather_merge.drop(columns=["grid_id", "part"])

    tfn_extreme_weather_merge = tfn_extreme_weather_merge.to_crs("EPSG:27700")
    tfn_drought = tfn_drought.to_crs("EPSG:27700")
    tfn_storm = tfn_storm.to_crs("EPSG:27700")

    tfn_extreme_weather_overlay = gpd.overlay(
        tfn_extreme_weather_merge,
        tfn_drought[["drought_risk_c", "drought_risk_f", "geometry"]],
        how="union",
    )

    tfn_extreme_weather_overlay = gpd.overlay(
        tfn_extreme_weather_overlay,
        tfn_storm[["storm_risk_c", "storm_risk_f", "geometry"]],
        how="union",
    )

    tfn_extreme_weather_overlay = _filter_out_small_geometries(
        tfn_extreme_weather_overlay, 0.035
    )

    tfn_extreme_weather_risk = _iterative_spatial_smoothing(
        tfn_extreme_weather_overlay,
        [
            "heat_risk_c",
            "heat_risk_f",
            "cold_risk_c",
            "cold_risk_f",
            "drought_risk_c",
            "drought_risk_f",
            "storm_risk_c",
            "storm_risk_f",
        ],
    )

    tfn_extreme_weather_risk = data_cleaning.explode_to_polygons(tfn_extreme_weather_risk)
    tfn_extreme_weather_risk = tfn_extreme_weather_risk.drop(columns=["part"])

    tfn_extreme_weather_risk = _calculate_composite_score(
        tfn_extreme_weather_risk,
        {"heat_risk": 0.25, "cold_risk": 0.25, "drought_risk": 0.25, "storm_risk": 0.25},
        "extreme_weather_risk",
    )

    tfn_extreme_weather_risk = min_max_scaling_pair(
        tfn_extreme_weather_risk, [("extreme_weather_risk_c", "extreme_weather_risk_f")]
    )

    tfn_extreme_weather_risk = gpd.GeoDataFrame(tfn_extreme_weather_risk, geometry="geometry")

    data_cleaning.write_to_file(
        tfn_extreme_weather_risk,
        cfg.paths.model_interim_output
        / "TfN Extreme Weather Risk"
        / "tfn_extreme_weather_risk.gpkg",
    )


#### EXTREME HEAT


def _extreme_heat_index(cfg: config.Config, common_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Combine several datasets into extreme heat index by merging on their common grid."""
    tfn_temp_max = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Summer Max Temperature Change Projections"
        / "tfn_temp_max.csv"
    )
    tfn_hsd = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Hot Summer Days Projections"
        / "tfn_hot_days.csv"
    )
    tfn_esd = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Extreme Summer Days Projections"
        / "tfn_extr_days.csv"
    )

    tfn_extreme_heat = _merge_on_key([tfn_temp_max, tfn_hsd, tfn_esd], common_grid, "grid_id")

    tfn_extreme_heat = _calculate_risk_threshold(
        tfn_extreme_heat, "tasmax_s", "tasmax_risk", 30
    )

    tfn_extreme_heat = min_max_scaling_pair(
        tfn_extreme_heat,
        [("tasmax_risk_c", "tasmax_risk_f"), ("hsd_c", "hsd_f"), ("esd_c", "esd_f")],
    )

    tfn_extreme_heat = _calculate_composite_score(
        tfn_extreme_heat,
        {
            "tasmax_risk": 0.5,
            "hsd": 0.25,
            "esd": 0.25,
        },
        "heat_risk",
    )

    tfn_extreme_heat = min_max_scaling_pair(tfn_extreme_heat, [("heat_risk_c", "heat_risk_f")])

    return gpd.GeoDataFrame(tfn_extreme_heat, geometry="geometry")


#### EXTREME COLD


def _extreme_cold_index(cfg: config.Config, common_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Combine several datasets into extreme cold index by merging on their common grid."""
    tfn_temp_min = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Winter Min Temperature Change Projections"
        / "tfn_temp_min.csv"
    )
    tfn_frost = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Frost Days Projections"
        / "tfn_frost_days.csv"
    )
    tfn_icing = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Icing Days Projections"
        / "tfn_ice_days.csv"
    )

    tfn_extreme_cold = _merge_on_key(
        [tfn_temp_min, tfn_frost, tfn_icing], common_grid, "grid_id"
    )

    tfn_extreme_cold = _calculate_risk_threshold(
        tfn_extreme_cold, "tasmin_w", "tasmin_risk", 0, invert=True
    )

    tfn_extreme_cold = min_max_scaling_pair(
        tfn_extreme_cold,
        [
            ("tasmin_risk_c", "tasmin_risk_f"),
            ("frost_d_c", "frost_d_f"),
            ("ice_d_c", "ice_d_f"),
        ],
    )

    tfn_extreme_cold = _calculate_composite_score(
        tfn_extreme_cold, {"tasmin_risk": 0.5, "frost_d": 0.25, "ice_d": 0.25}, "cold_risk"
    )

    tfn_extreme_cold = min_max_scaling_pair(tfn_extreme_cold, [("cold_risk_c", "cold_risk_f")])

    return gpd.GeoDataFrame(tfn_extreme_cold, geometry="geometry")


#### DROUGHT


def _drought_index(cfg: config.Config, common_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Combine several datasets into single drought index with spatial overlay."""
    tfn_drought = gpd.read_file(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Drought Severity Index"
        / "tfn_drought_index.gpkg"
    )
    tfn_precip_sum = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Summer Precipitation Change Projections"
        / "tfn_precip_sum.csv"
    )

    tfn_precip_sum_grid = tfn_precip_sum.merge(common_grid, on="grid_id")
    tfn_precip_sum_gdf = gpd.GeoDataFrame(
        tfn_precip_sum_grid, geometry="geometry", crs=common_grid.crs
    )
    tfn_precip_sum_gdf = tfn_precip_sum_gdf[["pr_s_c", "pr_s_f", "geometry"]]

    tfn_drought = tfn_drought.to_crs("EPSG:27700")
    tfn_precip_sum_gdf = tfn_precip_sum_gdf.to_crs("EPSG:27700")

    tfn_drought_overlay = gpd.overlay(tfn_precip_sum_gdf, tfn_drought, how="union")
    tfn_drought_overlay = _iterative_spatial_smoothing(
        tfn_drought_overlay, ["pr_s_c", "pr_s_f", "dsi_c", "dsi_f"]
    )

    tfn_drought_overlay = data_cleaning.explode_to_polygons(tfn_drought_overlay)
    tfn_drought_risk = tfn_drought_overlay[["dsi_c", "dsi_f", "pr_s_c", "pr_s_f", "geometry"]]

    tfn_drought_risk = min_max_scaling_pair(
        tfn_drought_risk, [("dsi_c", "dsi_f"), ("pr_s_c", "pr_s_f")]
    )

    # Reverse the polarity for precipitation
    tfn_drought_risk["pr_s_c"] = 100 - tfn_drought_risk["pr_s_c"]
    tfn_drought_risk["pr_s_f"] = 100 - tfn_drought_risk["pr_s_f"]

    tfn_drought_risk = _calculate_composite_score(
        tfn_drought_risk, {"dsi": 0.75, "pr_s": 0.25}, "drought_risk"
    )

    tfn_drought_risk = min_max_scaling_pair(
        tfn_drought_risk, [("drought_risk_c", "drought_risk_f")]
    )

    return gpd.GeoDataFrame(tfn_drought_risk, geometry="geometry")


#### STORMS


def _storm_index(cfg: config.Config, common_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Combine several datasets into a single storm index with a spatial overlay."""
    tfn_precip_win = pd.read_csv(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Winter Precipitation Change Projections"
        / "tfn_precip_win.csv"
    )
    tfn_rain_days = gpd.read_file(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN 10mm Rain Days 1991-2020"
        / "tfn_rain_days.gpkg"
    )
    tfn_wind_spd = gpd.read_file(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Wind Speed Projections"
        / "tfn_windspd.gpkg"
    )
    tfn_wdr = gpd.read_file(
        cfg.paths.model_input
        / "Hazards"
        / "Extreme Weather"
        / "TfN Wind Driven Rain Index"
        / "tfn_wdr.gpkg"
    )

    tfn_precip_win_grid = tfn_precip_win.merge(
        common_grid, on="grid_id", how="left", validate="one_to_many"
    )
    tfn_precip_win_gdf = gpd.GeoDataFrame(
        tfn_precip_win_grid, geometry="geometry", crs=common_grid.crs
    )
    tfn_precip_win_gdf = tfn_precip_win_gdf[["pr_w_c", "pr_w_f", "geometry"]]

    tfn_wind_spd = tfn_wind_spd.to_crs("EPSG:27700")
    tfn_rain_days = tfn_rain_days.to_crs("EPSG:27700")
    tfn_precip_win_gdf = tfn_precip_win_gdf.to_crs("EPSG:27700")
    tfn_wdr = tfn_wdr.to_crs("EPSG:27700")

    tfn_storm_overlay = gpd.overlay(tfn_wind_spd, tfn_rain_days, how="union")
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_precip_win_gdf, how="union")
    tfn_storm_overlay = gpd.overlay(tfn_storm_overlay, tfn_wdr, how="union")

    tfn_storm_overlay = tfn_storm_overlay[
        [
            "rain_d_c",
            "pr_w_c",
            "pr_w_f",
            "p99_c",
            "p99_f",
            "avg_excd_c",
            "avg_excd_f",
            "wdr_c",
            "wdr_f",
            "geometry",
        ]
    ]

    tfn_storm_overlay = _filter_out_small_geometries(tfn_storm_overlay, 0.01)

    tfn_storm_overlay = _iterative_spatial_smoothing(
        tfn_storm_overlay,
        [
            "rain_d_c",
            "pr_w_c",
            "pr_w_f",
            "p99_c",
            "p99_f",
            "avg_excd_c",
            "avg_excd_f",
            "wdr_c",
            "wdr_f",
        ],
    )

    tfn_storm_risk = data_cleaning.explode_to_polygons(tfn_storm_overlay)
    tfn_storm_risk = tfn_storm_risk.drop(columns=["part"])

    tfn_storm_risk["wind_spd_risk_c"] = tfn_storm_risk["p99_c"].apply(_wind_risk_scaled)
    tfn_storm_risk["wind_spd_risk_f"] = tfn_storm_risk["p99_f"].apply(_wind_risk_scaled)

    tfn_storm_risk = min_max_scaling_pair(
        tfn_storm_risk,
        [
            ("wind_spd_risk_c", "wind_spd_risk_f"),
            ("pr_w_c", "pr_w_f"),
            ("avg_excd_c", "avg_excd_f"),
            ("wdr_c", "wdr_f"),
        ],
    )

    # Scale rain days on its own, then duplicate
    scaler = MinMaxScaler(feature_range=(0, 100))
    tfn_storm_risk["rain_d_c"] = scaler.fit_transform(tfn_storm_risk[["rain_d_c"]])
    tfn_storm_risk["rain_d_f"] = tfn_storm_risk["rain_d_c"]

    tfn_storm_risk = _calculate_composite_score(
        tfn_storm_risk,
        {"wind_spd_risk": 0.3, "avg_excd": 0.2, "pr_w": 0.15, "rain_d": 0.15, "wdr": 0.2},
        "storm_risk",
    )

    tfn_storm_risk = min_max_scaling_pair(tfn_storm_risk, [("storm_risk_c", "storm_risk_f")])

    return gpd.GeoDataFrame(tfn_storm_risk, geometry="geometry")


def _wind_risk_scaled(speed_mps: float) -> float:
    """Calculate wind risk value given a wind speed, based on classification rule."""
    if speed_mps < WIND_RISK_THRESHOLD_LOWER:  # Below 30 mph
        return 0
    if speed_mps <= WIND_RISK_THRESHOLD_UPPER:  # between 30 and 45 mph
        return (speed_mps - 13.4) / (20.1 - 13.4)  # Scale to 0 - 1
    return 1 + (speed_mps - 20.1) / (25 - 20.1)  # Scale beyond 1


### FLOODING


def _flooding_index(cfg: config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Combine RoFRS & RoFSW into a single risk score by upscaling them to a common grid."""
    risk_score_map = {"Unavailable": 0, "Very low": 0, "Low": 1, "Medium": 2, "High": 3}

    if cfg.switches.create_flood_grid:
        flood_grid = _create_flood_grid(cfg, 1000, boundary)
    else:
        flood_grid = gpd.read_file(
            cfg.paths.model_interim_output / "Other" / "flood_grid.gpkg"
        )

    tfn_flood_risk_c = _upscale_to_grid(
        cfg,
        risk_score_map,
        flood_grid,
        {
            "current": [
                ("TfN RoFRS", "tfn_rofrs.gpkg", "rivers_sea_flood_risk_c"),
                ("TfN RoFSW", "tfn_rofsw.gpkg", "surface_water_flood_risk_c"),
            ]
        },
    )

    tfn_flood_risk_f = _upscale_to_grid(
        cfg,
        risk_score_map,
        flood_grid,
        {
            "forecast": [
                ("TfN RoFRS CC", "tfn_rofrs_cc.gpkg", "rivers_sea_flood_risk_f"),
                ("TfN RoFSW CC", "tfn_rofsw_cc.gpkg", "surface_water_flood_risk_f"),
            ]
        },
    )

    # Merge on geometry columns (these will be exactly the same)
    tfn_flood_risk = tfn_flood_risk_c.merge(tfn_flood_risk_f, on="geometry", how="left")

    tfn_flood_risk = min_max_scaling_pair(
        tfn_flood_risk,
        [
            ("rivers_sea_flood_risk_c", "rivers_sea_flood_risk_f"),
            ("surface_water_flood_risk_c", "surface_water_flood_risk_f"),
        ],
    )

    tfn_flood_risk = _calculate_composite_score(
        tfn_flood_risk,
        {"rivers_sea_flood_risk": 0.5, "surface_water_flood_risk": 0.5},
        "flood_risk",
    )

    tfn_flood_risk = min_max_scaling_pair(tfn_flood_risk, [("flood_risk_c", "flood_risk_f")])

    data_cleaning.write_to_file(
        tfn_flood_risk,
        cfg.paths.model_interim_output / "TfN Flood Risk" / "tfn_flood_risk.gpkg",
    )


def _create_flood_grid(
    cfg: config.Config, size_m: int, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Create a grid of a given size in metres, within a given boundary."""
    bounds = boundary.total_bounds
    grid = _create_grid(bounds, size_m)
    flood_grid = data_cleaning.clip_to_boundary(grid, boundary)
    data_cleaning.write_to_file(
        flood_grid, cfg.paths.model_interim_output / "Other" / "flood_grid.gpkg"
    )
    return flood_grid


def _process_flood_layer(
    flood_grid: gpd.GeoDataFrame,
    file_path: pathlib.Path,
    risk_column: str,
    risk_score_map: dict[str, int],
) -> gpd.GeoDataFrame:
    """Read a flood layer, assigns risk, and return area-weighted flood risk."""
    layer = gpd.read_file(file_path)
    layer[risk_column] = layer["Risk_band"].map(risk_score_map)
    return _area_weighted_flood_assignment(flood_grid, layer, risk_column)


def _upscale_to_grid(
    cfg: config.Config,
    risk_score_map: dict[str, int],
    flood_grid: gpd.GeoDataFrame,
    scenario_map: dict[str, list[tuple[str, str, str]]],
) -> gpd.GeoDataFrame:
    """Upscales each flood layer to the common grid and writes to file."""
    for scenario, layers in scenario_map.items():
        result = flood_grid.copy()
        for folder, file, risk_col in layers:
            result = _process_flood_layer(
                result,
                cfg.paths.model_input / "Hazards" / "Flooding" / folder / file,
                risk_col,
                risk_score_map,
            )

        data_cleaning.write_to_file(
            result,
            cfg.paths.model_interim_output
            / "TfN Flood Risk"
            / f"tfn_flood_risk_{scenario[0]}.gpkg",
        )

    return result


def _area_weighted_flood_assignment(
    grid: gpd.GeoDataFrame, flood_gdf: gpd.GeoDataFrame, risk_column: str
) -> gpd.GeoDataFrame:
    """Assign flood risk to grid squares using an area-weighted average."""
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
    grid[risk_column] = grid[risk_column].fillna(0)

    return grid


### GROUND STABILITY


def _ground_stability_index(cfg: config.Config) -> None:
    """Combine GeoSure & GeoClimate risk into a single index, using a spatial overlay."""
    risk_scores = {  # Map risk scores to normalised values (0-100)
        "Probable": 100,
        "Possible": 66,
        "Improbable": 33,
        "Unavailable": 50,  # Assign neutral value
    }

    tfn_geosure = gpd.read_file(
        cfg.paths.model_input
        / "Hazards"
        / "Ground Stability"
        / "TfN Geosure"
        / "tfn_geosure.gpkg"
    )
    tfn_geosure = tfn_geosure.to_crs("EPSG:27700")

    tfn_ss = {}
    ground_stability = {}
    for year, tp in {"2030": "c", "2070": "f"}.items():
        tfn_ss[year] = gpd.read_file(
            cfg.paths.model_input
            / "Hazards"
            / "Ground Stability"
            / "BGS Shrink Swell"
            / year
            / f"tfn_bgs_ss_{year}.gpkg"
        )
        tfn_ss[year]["shrink_swell_geoclimate_risk"] = tfn_ss[year][
            "shrink_swell_geoclimate_risk"
        ].map(risk_scores)
        tfn_ss[year] = tfn_ss[year][["shrink_swell_geoclimate_risk", "geometry"]]
        tfn_ss[year] = tfn_ss[year].to_crs("EPSG:27700")
        ground_stability[tp] = gpd.overlay(tfn_geosure, tfn_ss[year], how="union")
        ground_stability[tp] = ground_stability[tp].rename(
            columns={
                col: f"{col}_{tp}" for col in ground_stability[tp].columns if col != "geometry"
            }
        )
        ground_stability[tp] = ground_stability[tp].to_crs("EPSG:27700")

    tfn_ground_stability = gpd.overlay(
        ground_stability["c"], ground_stability["f"], how="union"
    )

    tfn_ground_stability = data_cleaning.explode_to_polygons(tfn_ground_stability)

    hazards = [
        "collapsible_deposits",
        "compressible_ground",
        "landslides",
        "running_sand",
        "shrink_swell",
        "soluble_rocks",
        "shrink_swell_geoclimate",
    ]

    suffixes = ["_c", "_f"]

    risk_cols = [f"{hazard}_risk{suffix}" for hazard in hazards for suffix in suffixes]

    for col in risk_cols:
        tfn_ground_stability[col] = pd.to_numeric(tfn_ground_stability[col], errors="coerce")

    tfn_ground_stability = _iterative_spatial_smoothing(tfn_ground_stability, risk_cols)

    gs_pairs = [(f"{col}_risk_c", f"{col}_risk_f") for col in hazards]

    tfn_ground_stability = min_max_scaling_pair(tfn_ground_stability, gs_pairs)

    tfn_ground_stability = _calculate_composite_score(
        tfn_ground_stability,
        {
            "shrink_swell_geoclimate_risk": 0.40,
            "landslides_risk": 0.10,
            "shrink_swell_risk": 0.10,
            "compressible_ground_risk": 0.10,
            "collapsible_deposits_risk": 0.10,
            "running_sand_risk": 0.10,
            "soluble_rocks_risk": 0.10,
        },
        "ground_stability_risk",
    )

    tfn_ground_stability = min_max_scaling_pair(
        tfn_ground_stability, [("ground_stability_risk_c", "ground_stability_risk_f")]
    )

    data_cleaning.write_to_file(
        tfn_ground_stability,
        cfg.paths.model_interim_output
        / "TfN Ground Stability Risk"
        / "tfn_ground_stability_risk.gpkg",
    )


### COASTAL EROSION


def _coastal_erosion_index(cfg: config.Config) -> None:
    """Combine erosion and ground stability risk into single index using a spatial overlay."""
    tfn_ncerm_giz = gpd.read_file(
        cfg.paths.model_input
        / "Hazards"
        / "Coastal Erosion"
        / "NCERM"
        / "Ground Instability Zones"
        / "tfn_ncerm_giz.gpkg"
    )
    tfn_ncerm_giz["giz_risk"] = 1

    tfn_ncerm = {}
    tfn_erosion_risk = {}
    for year, tp in {"2055": "c", "2105": "f"}.items():
        tfn_ncerm[year] = gpd.read_file(
            cfg.paths.model_input
            / "Hazards"
            / "Coastal Erosion"
            / "NCERM"
            / f"SMP_{year}_70CC"
            / f"tfn_ncerm_smp_{year}_70CC.gpkg"
        )
        tfn_ncerm[year]["coastal_erosion_risk"] = 1
        tfn_erosion_risk[tp] = _overlay_normalise(
            tfn_ncerm_giz,
            tfn_ncerm[year],
            ["coastal_erosion_risk", "giz_risk"],
            "coastal_erosion_risk",
            {"coastal_erosion_risk": 0.9, "giz_risk": 0.1},
        )
        tfn_erosion_risk[tp] = tfn_erosion_risk[tp].rename(
            columns={"coastal_erosion_risk": f"coastal_erosion_risk_{tp}"}
        )

    tfn_coastal_erosion_risk = gpd.overlay(
        tfn_erosion_risk["c"], tfn_erosion_risk["f"], how="union"
    )

    tfn_coastal_erosion_risk = tfn_coastal_erosion_risk.fillna(0)
    tfn_coastal_erosion_risk = gpd.GeoDataFrame(tfn_coastal_erosion_risk, geometry="geometry")
    tfn_coastal_erosion_risk = tfn_coastal_erosion_risk[
        ["coastal_erosion_risk_c", "coastal_erosion_risk_f", "geometry"]
    ]

    data_cleaning.write_to_file(
        tfn_coastal_erosion_risk,
        cfg.paths.model_interim_output
        / "TfN Coastal Erosion Risk"
        / "tfn_coastal_erosion_risk.gpkg",
    )
