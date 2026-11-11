"""Module for cleaning hazard datasets."""

# LOAD LIBRARIES
import gc
import logging
import pathlib
import warnings

import fiona
import geopandas as gpd
import pandas as pd
import xarray as xr

from caf.cvt import data_cleaning, definitions, model_config

# LOGGING

LOG = logging.getLogger(__name__)

# MODULE CONSTANTS

_WIND_SPEED_EXCEEDANCE_THRESHOLD = 20
_WIND_SPEED_PERCENTILE = 0.99

# WARNING FILTERS

# This warning is raised by xarray when reading the .nc wind speed files. It does not change
# the data at all, so we can safely ignore it. It is likely caused by a numpy version mismatch.
# Eventually this warning should be addressed, but for now it is safe to ignore.
warnings.filterwarnings(
    "ignore",
    message="numpy.ndarray size changed, may indicate binary incompatibility.*",
    category=RuntimeWarning,
)

## HAZARDS


def clean_hazards(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """
    Clean hazard data ready for analysis.

    Use config switches to clean the relevant hazard datasets, covering extreme weather,
    flooding, ground stability, and coastal erosion.

    Parameters
    ----------
    config: model_config.Config
        The configuration object containing switches and paths.
    boundary: gpd.GeoDataFrame
        The spatial boundary to which hazard data should be clipped.

    Returns
    -------
    None
    """
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
    hazard_grid = data_cleaning.clip_to_boundary(hazard_grid, boundary)
    filter_removed = len_before_filter - len(hazard_grid)
    LOG.debug(
        "Extreme weather grid filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    hazard_grid = data_cleaning.explode_to_polygons(hazard_grid, track_part=True)
    data_cleaning.write_to_file(
        hazard_grid,
        config.paths.hazard_grid_model_input,
    )
    return hazard_grid


def _clean_temp_max(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean max summer temperature change projections, then write to file."""
    temp_max = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.max_temp_summer,
        columns=["tasmax_summer_01_20_median", "tasmax_summer_change_40_median"],
    )
    temp_max["grid_id"] = range(1, len(temp_max) + 1)
    temp_max = temp_max.drop(columns=[definitions.GEOMETRY_COL])
    temp_max = temp_max.rename(
        columns={
            "tasmax_summer_01_20_median": (
                f"{definitions.ExtremeHeatCols.MAX_TEMP_SUMMER}_{definitions.Scenarios.CURRENT}"
            ),
            "tasmax_summer_change_40_median": (
                f"{definitions.ExtremeHeatCols.MAX_TEMP_SUMMER}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    temp_max[
        f"{definitions.ExtremeHeatCols.MAX_TEMP_SUMMER}_{definitions.Scenarios.FORECAST}"
    ] = (
        temp_max[
            f"{definitions.ExtremeHeatCols.MAX_TEMP_SUMMER}_{definitions.Scenarios.CURRENT}"
        ]
        + temp_max[
            f"{definitions.ExtremeHeatCols.MAX_TEMP_SUMMER}_{definitions.Scenarios.FORECAST}"
        ]
    )
    len_before_filter = len(temp_max)
    temp_max = temp_max[temp_max["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(temp_max)
    LOG.debug(
        "Summer max temperature change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(temp_max, config.paths.summer_max_temperature_model_input)


def _clean_temp_min(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean min winter temperature change projections, then write to file."""
    temp_min = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.min_temp_winter,
        columns=["tasmin_winter_01_20_median", "tasmin_winter_change_40_median"],
    )
    temp_min["grid_id"] = range(1, len(temp_min) + 1)
    temp_min = temp_min.drop(columns=[definitions.GEOMETRY_COL])
    temp_min = temp_min.rename(
        columns={
            "tasmin_winter_01_20_median": (
                f"{definitions.ExtremeColdCols.MIN_TEMP_WINTER}_{definitions.Scenarios.CURRENT}"
            ),
            "tasmin_winter_change_40_median": (
                f"{definitions.ExtremeColdCols.MIN_TEMP_WINTER}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    temp_min[
        f"{definitions.ExtremeColdCols.MIN_TEMP_WINTER}_{definitions.Scenarios.FORECAST}"
    ] = (
        temp_min[
            f"{definitions.ExtremeColdCols.MIN_TEMP_WINTER}_{definitions.Scenarios.CURRENT}"
        ]
        + temp_min[
            f"{definitions.ExtremeColdCols.MIN_TEMP_WINTER}_{definitions.Scenarios.FORECAST}"
        ]
    )
    len_before_filter = len(temp_min)
    temp_min = temp_min[temp_min["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(temp_min)
    LOG.debug(
        "Winter minimum temperature change projections filtered - %s of %s (%.1f percent) "
        "rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        temp_min,
        config.paths.winter_min_temperature_model_input,
    )


def _clean_summer_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean summer precipitation change projections, then write to file."""
    precip_sum = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.precip_summer,
        columns=["pr_summer_01_20_median", "pr_summer_change_40_median"],
    )
    precip_sum["grid_id"] = range(1, len(precip_sum) + 1)
    precip_sum = precip_sum.drop(columns=[definitions.GEOMETRY_COL])
    precip_sum = precip_sum.rename(
        columns={
            "pr_summer_01_20_median": (
                f"{definitions.DroughtCols.PRECIP_SUMMER}_{definitions.Scenarios.CURRENT}"
            ),
            "pr_summer_change_40_median": (
                f"{definitions.DroughtCols.PRECIP_SUMMER}_pct_chg_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    precip_sum[f"{definitions.DroughtCols.PRECIP_SUMMER}_{definitions.Scenarios.FORECAST}"] = (
        precip_sum[f"{definitions.DroughtCols.PRECIP_SUMMER}_{definitions.Scenarios.CURRENT}"]
        * (
            1
            + (
                precip_sum[
                    f"{definitions.DroughtCols.PRECIP_SUMMER}_pct_chg_{definitions.Scenarios.FORECAST}"
                ]
                / 100
            )
        )
    )
    precip_sum = precip_sum.drop(
        columns=[
            f"{definitions.DroughtCols.PRECIP_SUMMER}_pct_chg_{definitions.Scenarios.FORECAST}"
        ]
    )
    len_before_filter = len(precip_sum)
    precip_sum = precip_sum[precip_sum["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(precip_sum)
    LOG.debug(
        "Summer precipitation change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(precip_sum, config.paths.summer_precipitation_model_input)


def _clean_winter_precip(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean winter precipitation change projections, then write to file."""
    precip_win = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.precip_winter,
        columns=["pr_winter_01_20_median", "pr_winter_change_40_median"],
    )
    precip_win["grid_id"] = range(1, len(precip_win) + 1)
    precip_win = precip_win.drop(columns=[definitions.GEOMETRY_COL])
    precip_win = precip_win.rename(
        columns={
            "pr_winter_01_20_median": (
                f"{definitions.StormCols.PRECIP_WINTER}_{definitions.Scenarios.CURRENT}"
            ),
            "pr_winter_change_40_median": (
                f"{definitions.StormCols.PRECIP_WINTER}_pct_chg_{definitions.Scenarios.FORECAST}"
            ),
        },
    )
    precip_win[f"{definitions.StormCols.PRECIP_WINTER}_{definitions.Scenarios.FORECAST}"] = (
        precip_win[f"{definitions.StormCols.PRECIP_WINTER}_{definitions.Scenarios.CURRENT}"]
        * (
            1
            + (
                precip_win[
                    f"{definitions.StormCols.PRECIP_WINTER}_pct_chg_{definitions.Scenarios.FORECAST}"
                ]
                / 100
            )
        )
    )
    precip_win = precip_win.drop(
        columns=[
            f"{definitions.StormCols.PRECIP_WINTER}_pct_chg_{definitions.Scenarios.FORECAST}"
        ]
    )
    len_before_filter = len(precip_win)
    precip_win = precip_win[precip_win["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(precip_win)
    LOG.debug(
        "Winter precipitation change projections filtered - %s of %s (%.1f percent) rows "
        "removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(precip_win, config.paths.winter_precipitation_model_input)


def _clean_rain_days(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean 10mm rain days observations, then write to file."""
    rain_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.rain_days,
        mask=boundary,
    )
    len_before_filter = len(rain_days)
    rain_days = data_cleaning.clip_to_boundary(rain_days, boundary)
    rain_days = rain_days.rename(
        columns={
            "Rain10mmDays": f"{definitions.StormCols.RAIN_DAYS}_{definitions.Scenarios.CURRENT}"
        }
    )
    filter_removed = len_before_filter - len(rain_days)
    LOG.debug(
        "10mm rain days 1991-2020 filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    rain_days = data_cleaning.explode_to_polygons(rain_days)
    data_cleaning.write_to_file(rain_days, config.paths.rain_days_model_input)


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
                f"{definitions.DroughtCols.DROUGHT_SEVERITY_INDEX}_{definitions.Scenarios.CURRENT}"
            ),
            "DSI12_40_median": (
                f"{definitions.DroughtCols.DROUGHT_SEVERITY_INDEX}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    len_before_filter = len(drought_index)
    drought_index = data_cleaning.clip_to_boundary(drought_index, boundary)
    filter_removed = len_before_filter - len(drought_index)
    LOG.debug(
        "Drought severity index filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    drought_index = data_cleaning.explode_to_polygons(drought_index)
    data_cleaning.write_to_file(drought_index, config.paths.drought_severity_index_model_input)


def _clean_hot_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean hot summer days projections, then write to file."""
    hot_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.hot_days,
        columns=["HSD_baseline_01_20_median", "HSD_40_median"],
    )
    hot_days["grid_id"] = range(1, len(hot_days) + 1)
    hot_days = hot_days.drop(columns=[definitions.GEOMETRY_COL])
    hot_days = hot_days.rename(
        columns={
            "HSD_baseline_01_20_median": (
                f"{definitions.ExtremeHeatCols.HOT_SUMMER_DAYS}_{definitions.Scenarios.CURRENT}"
            ),
            "HSD_40_median": (
                f"{definitions.ExtremeHeatCols.HOT_SUMMER_DAYS}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    len_before_filter = len(hot_days)
    hot_days = hot_days[hot_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(hot_days)
    LOG.debug(
        "Hot summer days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(hot_days, config.paths.hot_summer_days_model_input)


def _clean_extreme_summer_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean extreme summer days projections, then write to file."""
    extr_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.extreme_summer_days,
        columns=["ESD_baseline_01_20_median", "ESD_40_median"],
    )
    extr_days["grid_id"] = range(1, len(extr_days) + 1)
    extr_days = extr_days.drop(columns=[definitions.GEOMETRY_COL])
    extr_days = extr_days.rename(
        columns={
            "ESD_baseline_01_20_median": (
                f"{definitions.ExtremeHeatCols.EXTREME_SUMMER_DAYS}_{definitions.Scenarios.CURRENT}"
            ),
            "ESD_40_median": (
                f"{definitions.ExtremeHeatCols.EXTREME_SUMMER_DAYS}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    len_before_filter = len(extr_days)
    extr_days = extr_days[extr_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(extr_days)
    LOG.debug(
        "Extreme summer days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        extr_days,
        config.paths.extreme_summer_days_model_input,
    )


def _clean_frost_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean frost days projections, then write to file."""
    frost_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.frost_days,
        columns=["FrostDays_baseline_01_20_median", "FrostDays_40_median"],
    )
    frost_days["grid_id"] = range(1, len(frost_days) + 1)
    frost_days = frost_days.drop(columns=[definitions.GEOMETRY_COL])
    frost_days = frost_days.rename(
        columns={
            "FrostDays_baseline_01_20_median": (
                f"{definitions.ExtremeColdCols.FROST_DAYS}_{definitions.Scenarios.CURRENT}"
            ),
            "FrostDays_40_median": (
                f"{definitions.ExtremeColdCols.FROST_DAYS}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    len_before_filter = len(frost_days)
    frost_days = frost_days[frost_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(frost_days)
    LOG.debug(
        "Frost days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(frost_days, config.paths.frost_days_model_input)


def _clean_icing_days(config: model_config.Config, grid: gpd.GeoDataFrame) -> None:
    """Read and clean icing days projections, then write to file."""
    ice_days = gpd.read_file(
        config.paths.raw_input / config.hazards.extreme_weather.icing_days,
        columns=["IcingDays_baseline_01_20_median", "IcingDays_40_median"],
    )
    ice_days["grid_id"] = range(1, len(ice_days) + 1)
    ice_days = ice_days.drop(columns=[definitions.GEOMETRY_COL])
    ice_days = ice_days.rename(
        columns={
            "IcingDays_baseline_01_20_median": (
                f"{definitions.ExtremeColdCols.ICING_DAYS}_{definitions.Scenarios.CURRENT}"
            ),
            "IcingDays_40_median": (
                f"{definitions.ExtremeColdCols.ICING_DAYS}_{definitions.Scenarios.FORECAST}"
            ),
        }
    )
    len_before_filter = len(ice_days)
    ice_days = ice_days[ice_days["grid_id"].isin(grid["grid_id"])]
    filter_removed = len_before_filter - len(ice_days)
    LOG.debug(
        "Icing days projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(ice_days, config.paths.icing_days_model_input)


def _clean_wind_speed(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read wind speed projections, calculate metrics, clean, then write to file."""
    windspd_current_agg, len_before_filter_current = _read_wind_speed_reduce(
        config.paths.raw_input / config.hazards.extreme_weather.wind_speed["1990_2000"],
        definitions.Scenarios.CURRENT,
    )
    windspd_forecast_agg, len_before_filter_forecast = _read_wind_speed_reduce(
        config.paths.raw_input / config.hazards.extreme_weather.wind_speed["2070_2080"],
        definitions.Scenarios.FORECAST,
    )

    metric_cols = [
        f"{definitions.StormCols.WIND_SPEED}_{definitions.Scenarios.CURRENT}",
        f"{definitions.StormCols.EXCEEDANCE_DAYS}_{definitions.Scenarios.CURRENT}",
        f"{definitions.StormCols.WIND_SPEED}_{definitions.Scenarios.FORECAST}",
        f"{definitions.StormCols.EXCEEDANCE_DAYS}_{definitions.Scenarios.FORECAST}",
    ]

    len_before_filter = len_before_filter_current + len_before_filter_forecast

    windspd_combined = windspd_current_agg.merge(
        windspd_forecast_agg,
        on=["projection_y_coordinate", "projection_x_coordinate", "latitude", "longitude"],
        how="outer",
    ).fillna({col: 0 for col in metric_cols})

    windspd_combined[definitions.GEOMETRY_COL] = [
        data_cleaning.convert_point_to_grid(x, y, 2500)
        for x, y in zip(
            windspd_combined["projection_x_coordinate"],
            windspd_combined["projection_y_coordinate"],
            strict=False,
        )
    ]
    windspd_combined = gpd.GeoDataFrame(
        windspd_combined, geometry=definitions.GEOMETRY_COL, crs=definitions.BNG_CRS
    )
    windspd_combined = windspd_combined[[*metric_cols, definitions.GEOMETRY_COL]]
    windspd_combined = data_cleaning.clip_to_boundary(windspd_combined, boundary)
    filter_removed = len_before_filter - len(windspd_combined)
    LOG.debug(
        "Wind speed projections filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    windspd_combined = data_cleaning.explode_to_polygons(windspd_combined)
    data_cleaning.write_to_file(windspd_combined, config.paths.wind_speed_model_input)


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
    return avg_exc.to_dataset(name=f"{definitions.StormCols.EXCEEDANCE_DAYS}_{scenario}")


def _calculate_windspd_percentile(
    windspd_data: xr.Dataset, quantile: float, variable: str, scenario: str
) -> xr.Dataset:
    """Calculate the wind speed percentiles per geometry for a given variable."""
    pct = windspd_data[variable].quantile(quantile, dim="time")
    return pct.to_dataset(name=f"{definitions.StormCols.WIND_SPEED}_{scenario}")


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
        .agg(
            {
                "WDR_baseline_Median": "mean",
                "WDR_40_Median": "mean",
                definitions.GEOMETRY_COL: "first",
            }
        )
        .reset_index()
    )

    wind_driven_rain = wind_driven_rain.drop(columns=["x_coord", "y_coord"])
    wind_driven_rain = wind_driven_rain.rename(
        columns={
            "WDR_baseline_Median": f"{definitions.StormCols.WIND_DRIVEN_RAIN_INDEX}_{definitions.Scenarios.CURRENT}",
            "WDR_40_Median": f"{definitions.StormCols.WIND_DRIVEN_RAIN_INDEX}_{definitions.Scenarios.FORECAST}",
        }
    )
    wind_driven_rain = gpd.GeoDataFrame(
        wind_driven_rain, geometry=definitions.GEOMETRY_COL, crs="EPSG:3857"
    )
    wind_driven_rain = data_cleaning.clip_to_boundary(wind_driven_rain, boundary)
    filter_removed = len_before_filter - len(wind_driven_rain)
    LOG.debug(
        "Wind driven rain index filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    wind_driven_rain = data_cleaning.explode_to_polygons(wind_driven_rain)
    data_cleaning.write_to_file(
        wind_driven_rain,
        config.paths.wind_driven_rain_model_input,
    )


### FLOODING


def _clean_flooding(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean flooding data ready for analysis."""
    LOG.info("Cleaning flooding data...")
    _clean_rofrs_rofsw(config, boundary)
    if config.hazards.flooding.groundwater is not None:
        _clean_groundwater(config, boundary)
    LOG.info("Finished cleaning flooding data.")


def _clean_rofrs_rofsw(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean and write RoFRS and RoFSW data ready for analysis.

    Gets BNG codes for the boundary, then read and clean the
    Risk of Flooding from Rivers and Sea (RoFRS) and
    Risk of Flooding from Surface Water (RoFSW) data.
    """
    # TODO (DJ): Remove hardcoding of Scotland and check data is empty
    if config.parameters.stb == definitions.SubnationalTransportBodies.TRANSPORT_SCOTLAND:
        LOG.warning("Flooding data is unavailable for Scotland.")
    bng_codes = data_cleaning.get_bng_codes(boundary)
    for flooding_type in [
        definitions.FloodingTypes.RIVERS_SEA,
        definitions.FloodingTypes.SURFACE_WATER,
    ]:
        for scenario in definitions.Scenarios:
            LOG.info("Cleaning %s flooding data for %s scenario...", flooding_type, scenario)
            _clean_flooding_layer(
                config=config,
                flooding_type=flooding_type,
                boundary=boundary,
                out_path=(
                    config.paths.flooding_type_scenario_model_input(flooding_type, scenario)
                ),
                rename_risk_col=f"{flooding_type}_flooding_risk_{scenario}",
                climate_change=(scenario == definitions.Scenarios.FORECAST),
                bng_codes=bng_codes,
            )


def _clean_groundwater(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean groundwater flooding data ready for analysis."""
    LOG.info("Cleaning groundwater flooding data...")
    groundwater_flooding = gpd.read_file(
        config.hazards.flooding.groundwater, columns=["CLASS"], mask=boundary
    )
    # TODO (DJ): #41 Replace with RiskColumn.GROUNDWATER_FLOODING when available
    groundwater_flooding = groundwater_flooding.rename(
        columns={"CLASS": "groundwater_flooding_risk"}
    )
    groundwater_flooding = data_cleaning.clip_to_boundary(groundwater_flooding, boundary)
    groundwater_flooding = data_cleaning.explode_to_polygons(groundwater_flooding)
    groundwater_flooding = groundwater_flooding.to_crs(definitions.BNG_CRS)
    data_cleaning.write_to_file(
        groundwater_flooding,
        output_path=config.paths.groundwater_flooding_model_input,
    )


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
        LOG.debug("Processing tile: %s...", metadata["tile"])

        flooding_data = _read_flooding_zip(zip_path, boundary)

        if flooding_data.empty:
            LOG.debug("%s layer empty. Continuing.", metadata["tile"])
            continue

        flooding_data = data_cleaning.extract_poly_from_geomcollection(flooding_data)

        flooding_data = flooding_data.rename(columns={"Risk_band": rename_risk_col})

        data_cleaning.write_to_file(
            flooding_data,
            out_path,
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
    flooding_root = getattr(config.hazards.flooding, flooding_type)
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
) -> gpd.GeoDataFrame:
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
    if config.switches.ground_stability_premium:
        _clean_premium_geosure(config, boundary)
    else:
        _clean_open_geosure(config, boundary)
    LOG.info("Finished cleaning ground stability data.")


def _clean_premium_geosure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean premium GeoSure data then write to file."""
    geosure_layers = {
        definitions.GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.collapsible_deposits,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.COMPRESSIBLE_GROUND: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.compressible_ground,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.LANDSLIDES: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.landslides,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.RUNNING_SAND: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.running_sands,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.SHRINK_SWELL: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.shrink_swell,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.SOLUBLE_ROCKS: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.soluble_rocks,
            mask=boundary,
            columns=["CLASS"],
        ),
    }

    for geohazard, geosure_data in geosure_layers.items():
        if geosure_data.empty:
            raise ValueError(
                f"GeoSure {geohazard} layer is empty after reading. "
                f"Check the source file and boundary."
            )
        len_before_filter = len(geosure_data)
        geosure_data_clean = geosure_data.rename(
            columns={"CLASS": f"{geohazard}_{definitions.Scenarios.CURRENT}"}
        )
        geosure_data_clean[f"{geohazard}_{definitions.Scenarios.FORECAST}"] = (
            geosure_data_clean[f"{geohazard}_{definitions.Scenarios.CURRENT}"]
        )
        geosure_layers[geohazard] = data_cleaning.clip_to_boundary(
            geosure_data_clean, boundary
        )
        filter_removed = len_before_filter - len(geosure_layers[geohazard])
        LOG.debug(
            "GeoSure %s filtered - %s of %s (%.1f percent) rows removed",
            geohazard.replace("_", " ").title(),
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        geosure_layers[geohazard] = data_cleaning.explode_to_polygons(
            geosure_layers[geohazard]
        )
        geosure_layers[geohazard] = geosure_layers[geohazard].to_crs(definitions.BNG_CRS)
        data_cleaning.write_to_file(
            geosure_layers[geohazard],
            config.paths.geosure_model_input(geohazard),
        )


def _clean_open_geosure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean GeoSureHexGrids data, merge by nearest centroids, then write to file."""
    geosure_layers = {
        definitions.GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_CollapsibleDeposits_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.COMPRESSIBLE_GROUND: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_CompressibleGround_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.LANDSLIDES: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_Landslides_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.RUNNING_SAND: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_RunningSand_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.SHRINK_SWELL: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_ShrinkSwell_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityRiskCols.SOLUBLE_ROCKS: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
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
        geosure_data_clean[code] = pd.to_numeric(geosure_data_clean[code], errors="coerce")
        geosure_layers[code] = data_cleaning.clip_to_boundary(geosure_data_clean, boundary)
        filter_removed = len_before_filter - len(geosure_layers[code])
        LOG.debug(
            "GeoSure %s filtered - %s of %s (%.1f percent) rows removed",
            code.replace("_", " ").title(),
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        geosure_layers[code] = data_cleaning.explode_to_polygons(geosure_layers[code])

    # Merge layers based on nearest centroids
    base_code = next(iter(geosure_layers.keys()))
    geosure = geosure_layers[base_code][[base_code, definitions.GEOMETRY_COL]].copy()
    for code, layer in geosure_layers.items():
        if code == base_code:
            continue  # skip the base layer
        layer_subset = layer[
            [code, definitions.GEOMETRY_COL]
        ]  # Select only the relevant class and geometry columns
        matched = data_cleaning.nearest_centroids(geosure, layer_subset)
        geosure[code] = matched[code]  # Add the matched CLASS column to the base dataframe

    geosure_risk_cols = [col for col in geosure.columns if "_risk" in col]
    for col in geosure_risk_cols:  # Duplicate risk cols for forecast scenario
        geosure[f"{col}_{definitions.Scenarios.FORECAST}"] = geosure[col]
        geosure = geosure.rename(columns={col: f"{col}_{definitions.Scenarios.CURRENT}"})
    geosure_risk_cols = [col for col in geosure.columns if "_risk" in col]
    geosure = geosure[[*geosure_risk_cols, definitions.GEOMETRY_COL]]
    data_cleaning.write_to_file(
        geosure,
        config.paths.geosure_model_input(None),
    )


### COASTAL EROSION


def _clean_coastal_erosion(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean coastal erosion data ready for analysis."""
    LOG.info("Cleaning coastal erosion data...")
    _clean_national_coastal_erosion_risk_mapping(config, boundary)
    LOG.info("Finished cleaning coastal erosion data.")


def _clean_national_coastal_erosion_risk_mapping(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> None:
    """Clean erosion data from the National Coastal Erosion Risk Mapping (NCERM).

    Reads NCERM coastal erosion data for 2055 and 2105, cleans and writes to file.
    """
    for year in ["2055", "2105"]:
        erosion_data = gpd.read_file(
            f"zip://{config.paths.raw_input / config.hazards.coastal_erosion.zip_path}!"
            f"{config.hazards.coastal_erosion.file_path}",
            layer=f"NCERM_SMP_{year}_70CC",
            mask=boundary,
            columns=["smp_name"],
        )
        if erosion_data.empty:
            LOG.debug("NCERM %s layer empty after filtering. Writing empty file.", year)
            data_cleaning.write_to_file(
                gpd.GeoDataFrame(
                    columns=["smp_name", definitions.GEOMETRY_COL],
                    geometry=definitions.GEOMETRY_COL,
                    crs=definitions.BNG_CRS,
                ),
                config.paths.national_coastal_erosion_risk_mapping_model_input(year),
            )
            if year == "2055":
                continue
            return
        len_before_filter = len(erosion_data)
        erosion_data = data_cleaning.clip_to_boundary(erosion_data, boundary)
        filter_removed = len_before_filter - len(erosion_data)
        LOG.debug(
            "NCERM %s filtered - %s of %s (%.1f percent) rows removed",
            year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        erosion_data = data_cleaning.explode_to_polygons(erosion_data)

        data_cleaning.write_to_file(
            erosion_data,
            config.paths.national_coastal_erosion_risk_mapping_model_input(year),
        )
