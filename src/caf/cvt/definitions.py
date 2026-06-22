"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import enum


class Columns(enum.StrEnum):
    """Column definition base class."""

    @classmethod
    def all(cls) -> list:
        """Return a list of all columns."""
        return list(cls)


class MainHazardCols(Columns):
    """Column definitions for main hazard layers."""

    EXTREME_WEATHER = "extreme_weather"
    FLOODING = "flooding"
    GROUND_STABILITY = "ground_stability"
    COASTAL_EROSION = "coastal_erosion"


    @classmethod
    def get_cmap(cls, col: MainHazardCols) -> str:
        """Return the appropriate colormap for a given hazard column."""
        cmap_mapping = {
            cls.EXTREME_WEATHER: "Reds",
            cls.FLOODING: "Blues",
            cls.GROUND_STABILITY: "Oranges",
            cls.COASTAL_EROSION: "Purples",
        }
        return cmap_mapping[col]


class ExtremeWeatherCols(Columns):
    """Column definitions for extreme weather subhazard layers."""

    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    DROUGHT = "drought"
    STORM = "storm"

    @classmethod
    def get_cmap(cls, col: ExtremeWeatherCols) -> str:
        """Return the appropriate colormap for a given extreme weather subhazard column."""
        cmap_mapping = {
            cls.EXTREME_HEAT: "Reds",
            cls.EXTREME_COLD: "Blues",
            cls.DROUGHT: "Oranges",
            cls.STORM: "Blues",
        }
        return cmap_mapping[col]


class FloodingCols(Columns):
    """Column definitions for flooding subhazard layers."""

    RIVERS_SEA = "rivers_sea_flooding"
    SURFACE_WATER = "surface_water_flooding"

    @classmethod
    def get_cmap(cls, col: FloodingCols) -> str:
        """Return the appropriate colormap for a given flooding subhazard column."""
        cmap_mapping = {
            cls.RIVERS_SEA: "Blues",
            cls.SURFACE_WATER: "Blues",
        }
        return cmap_mapping[col]


class GroundStabilityCols(Columns):
    """Column definitions for ground stability subhazard layers."""

    COLLAPSIBLE_DEPOSITS = "collapsible_deposits"
    COMPRESSIBLE_GROUND = "compressible_ground"
    LANDSLIDES = "landslides"
    RUNNING_SAND = "running_sand"
    SHRINK_SWELL = "shrink_swell"
    SOLUBLE_ROCKS = "soluble_rocks"
    SHRINK_SWELL_GEOCLIMATE = "shrink_swell_geoclimate"

    @classmethod
    def get_cmap(cls, col: GroundStabilityCols) -> str:
        """Return the appropriate colormap for a given ground stability subhazard column."""
        cmap_mapping = {
            cls.COLLAPSIBLE_DEPOSITS: "Oranges",
            cls.COMPRESSIBLE_GROUND: "Oranges",
            cls.LANDSLIDES: "Oranges",
            cls.RUNNING_SAND: "Oranges",
            cls.SHRINK_SWELL: "Oranges",
            cls.SOLUBLE_ROCKS: "Oranges",
            cls.SHRINK_SWELL_GEOCLIMATE: "Oranges",
        }
        return cmap_mapping[col]


class ExtremeHeatCols(Columns):
    """Column definitions for extreme heat subhazard layers."""

    MAX_TEMP_SUMMER = "max_temp_summer"
    HOT_SUMMER_DAYS = "hot_summer_days"
    EXTREME_SUMMER_DAYS = "extreme_summer_days"


class ExtremeColdCols(Columns):
    """Column definitions for extreme cold subhazard layers."""

    MIN_TEMP_WINTER = "min_temp_winter"
    FROST_DAYS = "frost_days"
    ICING_DAYS = "icing_days"


class DroughtCols(Columns):
    """Column definitions for drought subhazard layers."""

    DROUGHT_SEVERITY_INDEX = "drought_severity_index"
    PRECIP_SUMMER_RISK = "precip_summer_risk"


class StormCols(Columns):
    """Column definitions for storm subhazard layers."""

    RAIN_DAYS = "10mm_rain_days"
    PRECIP_WINTER = "precip_winter"
    EXCEEDANCE_DAYS = "avg_exceedance_days"
    WIND_SPEED_RISK = "wind_speed_risk"
    WIND_DRIVEN_RAIN_INDEX = "wind_driven_rain_index"


class NoHAMImpactCols(Columns):
    """Column definitions for NoHAM impact layers."""

    UC1_IMPACT = "uc1_impact"
    UC2_IMPACT = "uc2_impact"
    UC3_IMPACT = "uc3_impact"
    UC4_IMPACT = "uc4_impact"
    UC5_IMPACT = "uc5_impact"
    IMPACT = "impact"


class Scenarios(Columns):
    """Column definitions for scenarios."""

    CURRENT = "current"
    FORECAST = "forecast"

    @classmethod
    def scenario_or_column(cls) -> str:
        """Return the name of the scenario column."""
        return f"{cls.CURRENT}_or_{cls.FORECAST}"


class NoHAM:
    """Definitions for NoHAM data."""

    USER_CLASS_1 = "uc1"
    USER_CLASS_2 = "uc2"
    USER_CLASS_3 = "uc3"
    USER_CLASS_4 = "uc4"
    USER_CLASS_5 = "uc5"

    TIME_PERIOD_1 = "TS1"
    TIME_PERIOD_2 = "TS2"
    TIME_PERIOD_3 = "TS3"

    NOHAM_ROAD_ID_THRESHOLD = 10000

    @classmethod
    def all_user_classes(cls) -> list[str]:
        """Return a list of all NoHAM user classes."""
        return [
            cls.USER_CLASS_1,
            cls.USER_CLASS_2,
            cls.USER_CLASS_3,
            cls.USER_CLASS_4,
            cls.USER_CLASS_5,
        ]

    @classmethod
    def all_time_periods(cls) -> list[str]:
        """Return a list of all NoHAM time periods."""
        return [cls.TIME_PERIOD_1, cls.TIME_PERIOD_2, cls.TIME_PERIOD_3]

    @classmethod
    def get_scenario(cls, year: str) -> str:
        """Return the scenario corresponding to a given year."""
        mapping = {
            2023: Scenarios.CURRENT,
            2048: Scenarios.FORECAST,
        }
        return mapping[int(year)]
