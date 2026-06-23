"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import abc
import enum


class PlottingColumn(enum.StrEnum, abc.ABC):
    """Plotting column definition base class."""

    @abc.abstractmethod
    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given plotting column."""



class MainHazardCols(PlottingColumn):
    """Column definitions for main hazard layers."""

    EXTREME_WEATHER = "extreme_weather"
    FLOODING = "flooding"
    GROUND_STABILITY = "ground_stability"
    COASTAL_EROSION = "coastal_erosion"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given hazard column."""
        cmap_mapping = {
            MainHazardCols.EXTREME_WEATHER: "Reds",
            MainHazardCols.FLOODING: "Blues",
            MainHazardCols.GROUND_STABILITY: "Oranges",
            MainHazardCols.COASTAL_EROSION: "Purples",
        }
        return cmap_mapping[self]


class ExtremeWeatherCols(PlottingColumn):
    """Column definitions for extreme weather subhazard layers."""

    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    DROUGHT = "drought"
    STORM = "storm"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given extreme weather subhazard column."""
        cmap_mapping = {
            ExtremeWeatherCols.EXTREME_HEAT: "Reds",
            ExtremeWeatherCols.EXTREME_COLD: "Blues",
            ExtremeWeatherCols.DROUGHT: "Oranges",
            ExtremeWeatherCols.STORM: "Blues",
        }
        return cmap_mapping[self]


class FloodingCols(PlottingColumn):
    """Column definitions for flooding subhazard layers."""

    RIVERS_SEA = "rivers_sea_flooding"
    SURFACE_WATER = "surface_water_flooding"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given flooding subhazard column."""
        cmap_mapping = {
            FloodingCols.RIVERS_SEA: "Blues",
            FloodingCols.SURFACE_WATER: "Blues",
        }
        return cmap_mapping[self]


class GroundStabilityCols(PlottingColumn):
    """Column definitions for ground stability subhazard layers."""

    COLLAPSIBLE_DEPOSITS = "collapsible_deposits"
    COMPRESSIBLE_GROUND = "compressible_ground"
    LANDSLIDES = "landslides"
    RUNNING_SAND = "running_sand"
    SHRINK_SWELL = "shrink_swell"
    SOLUBLE_ROCKS = "soluble_rocks"
    SHRINK_SWELL_GEOCLIMATE = "shrink_swell_geoclimate"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given ground stability subhazard column."""
        cmap_mapping = {
            GroundStabilityCols.COLLAPSIBLE_DEPOSITS: "Oranges",
            GroundStabilityCols.COMPRESSIBLE_GROUND: "Oranges",
            GroundStabilityCols.LANDSLIDES: "Oranges",
            GroundStabilityCols.RUNNING_SAND: "Oranges",
            GroundStabilityCols.SHRINK_SWELL: "Oranges",
            GroundStabilityCols.SOLUBLE_ROCKS: "Oranges",
            GroundStabilityCols.SHRINK_SWELL_GEOCLIMATE: "Oranges",
        }
        return cmap_mapping[self]


class ExtremeHeatCols(PlottingColumn):
    """Column definitions for extreme heat subhazard layers."""

    MAX_TEMP_SUMMER = "max_temp_summer"
    HOT_SUMMER_DAYS = "hot_summer_days"
    EXTREME_SUMMER_DAYS = "extreme_summer_days"

    def get_cmap(self) -> str:
        """Return extreme heat colourmap."""
        return "Reds"


class ExtremeColdCols(PlottingColumn):
    """Column definitions for extreme cold subhazard layers."""

    MIN_TEMP_WINTER = "min_temp_winter"
    FROST_DAYS = "frost_days"
    ICING_DAYS = "icing_days"

    def get_cmap(self) -> str:
        """Return extreme cold colourmap."""
        return "Blues"


class DroughtCols(PlottingColumn):
    """Column definitions for drought subhazard layers."""

    DROUGHT_SEVERITY_INDEX = "drought_severity_index"
    PRECIP_SUMMER_RISK = "precip_summer_risk"

    def get_cmap(self) -> str:
        """Return drought colourmap."""
        return "Oranges"


class StormCols(PlottingColumn):
    """Column definitions for storm subhazard layers."""

    RAIN_DAYS = "10mm_rain_days"
    PRECIP_WINTER = "precip_winter"
    EXCEEDANCE_DAYS = "avg_exceedance_days"
    WIND_SPEED = "wind_speed"
    WIND_DRIVEN_RAIN_INDEX = "wind_driven_rain_index"

    def get_cmap(self) -> str:
        """Return storm colourmap."""
        return "Blues"


class NoHAMImpactCols(enum.strEnum):
    """Column definitions for NoHAM impact layers."""

    UC1_IMPACT = "uc1_impact"
    UC2_IMPACT = "uc2_impact"
    UC3_IMPACT = "uc3_impact"
    UC4_IMPACT = "uc4_impact"
    UC5_IMPACT = "uc5_impact"
    IMPACT = "impact"


class Scenarios(enum.strEnum):
    """Column definitions for scenarios."""

    CURRENT = "current"
    FORECAST = "forecast"

    @classmethod
    def scenario_or_column(cls) -> str:
        """Return the name of the scenario column."""
        return f"{cls.CURRENT}_or_{cls.FORECAST}"


class NoHAMUserClasses(enum.strEnum):
    """Definitions for NoHAM user classes."""

    USER_CLASS_1 = "uc1"
    USER_CLASS_2 = "uc2"
    USER_CLASS_3 = "uc3"
    USER_CLASS_4 = "uc4"
    USER_CLASS_5 = "uc5"


class NoHAMTimePeriods(enum.strEnum):
    """Definitions for NoHAM time periods."""

    TIME_PERIOD_1 = "TS1"
    TIME_PERIOD_2 = "TS2"
    TIME_PERIOD_3 = "TS3"
