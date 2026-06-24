"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import enum


class PlottingColumn(enum.StrEnum):
    """Plotting column definition base class."""

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given plotting column."""
        raise NotImplementedError("Subclasses must implement the get_cmap method.")

    @property
    def base_name(self) -> str:
        """Return the base name of the plotting column."""
        return self.removesuffix("_risk")



class MainHazardRiskCols(PlottingColumn):
    """Column definitions for main hazard layers."""

    EXTREME_WEATHER = "extreme_weather_risk"
    FLOODING = "flooding_risk"
    GROUND_STABILITY = "ground_stability_risk"
    COASTAL_EROSION = "coastal_erosion_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given hazard column."""
        cmap_mapping = {
            MainHazardRiskCols.EXTREME_WEATHER: "Reds",
            MainHazardRiskCols.FLOODING: "Blues",
            MainHazardRiskCols.GROUND_STABILITY: "Oranges",
            MainHazardRiskCols.COASTAL_EROSION: "Purples",
        }
        return cmap_mapping[self]


class ExtremeWeatherRiskCols(PlottingColumn):
    """Column definitions for extreme weather subhazard layers."""

    EXTREME_HEAT = "extreme_heat_risk"
    EXTREME_COLD = "extreme_cold_risk"
    DROUGHT = "drought_risk"
    STORM = "storm_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given extreme weather subhazard column."""
        cmap_mapping = {
            ExtremeWeatherRiskCols.EXTREME_HEAT: "Reds",
            ExtremeWeatherRiskCols.EXTREME_COLD: "Blues",
            ExtremeWeatherRiskCols.DROUGHT: "Oranges",
            ExtremeWeatherRiskCols.STORM: "Blues",
        }
        return cmap_mapping[self]


class FloodingRiskCols(PlottingColumn):
    """Column definitions for flooding subhazard layers."""

    RIVERS_SEA = "rivers_sea_flooding_risk"
    SURFACE_WATER = "surface_water_flooding_risk"

    @classmethod
    def get_cmap(cls) -> str:
        """Return the appropriate colormap for a given flooding subhazard column."""
        return "Blues"


class GroundStabilityRiskCols(PlottingColumn):
    """Column definitions for ground stability subhazard layers."""

    COLLAPSIBLE_DEPOSITS = "collapsible_deposits_risk"
    COMPRESSIBLE_GROUND = "compressible_ground_risk"
    LANDSLIDES = "landslides_risk"
    RUNNING_SAND = "running_sand_risk"
    SHRINK_SWELL = "shrink_swell_risk"
    SOLUBLE_ROCKS = "soluble_rocks_risk"
    SHRINK_SWELL_GEOCLIMATE = "shrink_swell_geoclimate_risk"

    @classmethod
    def get_cmap(cls) -> str:
        """Return the appropriate colormap for a given ground stability subhazard column."""
        return "Oranges"

class CoastalErosionRiskCols(PlottingColumn):
    """Column definitions for coastal erosion subhazard layers."""

    EROSION = "erosion_risk"
    GIZ = "giz_risk"

    @classmethod
    def get_cmap(cls) -> str:
        """Return the appropriate colormap for a given coastal erosion subhazard column."""
        return "Purples"

class ExtremeHeatCols(PlottingColumn):
    """Column definitions for extreme heat subhazard layers."""

    MAX_TEMP_SUMMER = "max_temp_summer"
    HOT_SUMMER_DAYS = "hot_summer_days"
    EXTREME_SUMMER_DAYS = "extreme_summer_days"

    @classmethod
    def get_cmap(cls) -> str:
        """Return extreme heat colourmap."""
        return "Reds"


class ExtremeColdCols(PlottingColumn):
    """Column definitions for extreme cold subhazard layers."""

    MIN_TEMP_WINTER = "min_temp_winter"
    FROST_DAYS = "frost_days"
    ICING_DAYS = "icing_days"

    @classmethod
    def get_cmap(cls) -> str:
        """Return extreme cold colourmap."""
        return "Blues"


class DroughtCols(PlottingColumn):
    """Column definitions for drought subhazard layers."""

    DROUGHT_SEVERITY_INDEX = "drought_severity_index"
    PRECIP_SUMMER = "precip_summer"

    @classmethod
    def get_cmap(cls) -> str:
        """Return drought colourmap."""
        return "Oranges"


class StormCols(PlottingColumn):
    """Column definitions for storm subhazard layers."""

    RAIN_DAYS = "10mm_rain_days"
    PRECIP_WINTER = "precip_winter"
    EXCEEDANCE_DAYS = "avg_exceedance_days"
    WIND_SPEED = "wind_speed"
    WIND_DRIVEN_RAIN_INDEX = "wind_driven_rain_index"

    @classmethod
    def get_cmap(cls) -> str:
        """Return storm colourmap."""
        return "Blues"


class NoHAMImpactCols(enum.StrEnum):
    """Column definitions for NoHAM impact layers."""

    UC1_IMPACT = "uc1_impact"
    UC2_IMPACT = "uc2_impact"
    UC3_IMPACT = "uc3_impact"
    UC4_IMPACT = "uc4_impact"
    UC5_IMPACT = "uc5_impact"
    IMPACT = "impact"


class Scenarios(enum.StrEnum):
    """Column definitions for scenarios."""

    CURRENT = "current"
    FORECAST = "forecast"

    @classmethod
    def scenario_or_column(cls) -> str:
        """Return the name of the scenario column."""
        return f"{cls.CURRENT}_or_{cls.FORECAST}"


class NoHAMUserClasses(enum.StrEnum):
    """Definitions for NoHAM user classes."""

    USER_CLASS_1 = "uc1"
    USER_CLASS_2 = "uc2"
    USER_CLASS_3 = "uc3"
    USER_CLASS_4 = "uc4"
    USER_CLASS_5 = "uc5"


class NoHAMTimePeriods(enum.StrEnum):
    """Definitions for NoHAM time periods."""

    TIME_PERIOD_1 = "TS1"
    TIME_PERIOD_2 = "TS2"
    TIME_PERIOD_3 = "TS3"
