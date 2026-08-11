"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import enum


class RiskColumn(enum.StrEnum):
    """Plotting column definition base class."""

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given plotting column."""
        raise NotImplementedError("Subclasses must implement the get_cmap method.")

    @property
    def base_name(self) -> str:
        """Return the base name of the plotting column."""
        return self.removesuffix("_risk")


class MainHazardRiskCols(RiskColumn):
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


    def get_weights(self) -> dict[RiskColumn, float]:
        """Return the weights for the sub-hazards of a given main hazard column."""
        weights_mapping: dict[MainHazardRiskCols, dict[RiskColumn, float]] = {
            MainHazardRiskCols.EXTREME_WEATHER: {
                ExtremeWeatherRiskCols.EXTREME_HEAT: 0.25,
                ExtremeWeatherRiskCols.EXTREME_COLD: 0.25,
                ExtremeWeatherRiskCols.DROUGHT: 0.25,
                ExtremeWeatherRiskCols.STORM: 0.25,
            },
            MainHazardRiskCols.FLOODING: {
                FloodingRiskCols.RIVERS_SEA: 0.5,
                FloodingRiskCols.SURFACE_WATER: 0.5,
            },
            MainHazardRiskCols.GROUND_STABILITY: {
                GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE: 0.40,
                GroundStabilityRiskCols.LANDSLIDES: 0.10,
                GroundStabilityRiskCols.SHRINK_SWELL: 0.10,
                GroundStabilityRiskCols.COMPRESSIBLE_GROUND: 0.10,
                GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: 0.10,
                GroundStabilityRiskCols.RUNNING_SAND: 0.10,
                GroundStabilityRiskCols.SOLUBLE_ROCKS: 0.10,
            },
            MainHazardRiskCols.COASTAL_EROSION: {
                CoastalErosionRiskCols.EROSION: 0.9,
                CoastalErosionRiskCols.GIZ: 0.1,
            },
        }

        return weights_mapping[self]


class ExtremeWeatherRiskCols(RiskColumn):
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


class FloodingRiskCols(RiskColumn):
    """Column definitions for flooding subhazard layers."""

    RIVERS_SEA = "rivers_sea_flooding_risk"
    SURFACE_WATER = "surface_water_flooding_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given flooding subhazard column."""
        return "Blues"


class GroundStabilityRiskCols(RiskColumn):
    """Column definitions for ground stability subhazard layers."""

    COLLAPSIBLE_DEPOSITS = "collapsible_deposits_risk"
    COMPRESSIBLE_GROUND = "compressible_ground_risk"
    LANDSLIDES = "landslides_risk"
    RUNNING_SAND = "running_sand_risk"
    SHRINK_SWELL = "shrink_swell_risk"
    SOLUBLE_ROCKS = "soluble_rocks_risk"
    SHRINK_SWELL_GEOCLIMATE = "shrink_swell_geoclimate_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given ground stability subhazard column."""
        return "Oranges"


class CoastalErosionRiskCols(RiskColumn):
    """Column definitions for coastal erosion subhazard layers."""

    EROSION = "erosion_risk"
    GIZ = "giz_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given coastal erosion subhazard column."""
        return "Purples"


class ExtremeHeatCols(RiskColumn):
    """Column definitions for extreme heat subhazard layers."""

    MAX_TEMP_SUMMER = "max_temp_summer"
    HOT_SUMMER_DAYS = "hot_summer_days"
    EXTREME_SUMMER_DAYS = "extreme_summer_days"

    def get_cmap(self) -> str:
        """Return extreme heat colourmap."""
        return "Reds"


class ExtremeColdCols(RiskColumn):
    """Column definitions for extreme cold subhazard layers."""

    MIN_TEMP_WINTER = "min_temp_winter"
    FROST_DAYS = "frost_days"
    ICING_DAYS = "icing_days"

    def get_cmap(self) -> str:
        """Return extreme cold colourmap."""
        return "Blues"


class DroughtCols(RiskColumn):
    """Column definitions for drought subhazard layers."""

    DROUGHT_SEVERITY_INDEX = "drought_severity_index"
    PRECIP_SUMMER = "precip_summer"

    def get_cmap(self) -> str:
        """Return drought colourmap."""
        return "Oranges"


class StormCols(RiskColumn):
    """Column definitions for storm subhazard layers."""

    RAIN_DAYS = "10mm_rain_days"
    PRECIP_WINTER = "precip_winter"
    EXCEEDANCE_DAYS = "avg_exceedance_days"
    WIND_SPEED = "wind_speed"
    WIND_DRIVEN_RAIN_INDEX = "wind_driven_rain_index"

    def get_cmap(self) -> str:
        """Return storm colourmap."""
        return "Blues"


class ImpactCols(RiskColumn):
    """Column definitions for impact layers."""

    UC1_IMPACT = "uc1_impact"
    UC2_IMPACT = "uc2_impact"
    UC3_IMPACT = "uc3_impact"
    UC4_IMPACT = "uc4_impact"
    UC5_IMPACT = "uc5_impact"
    UC6_IMPACT = "uc6_impact"
    UC7_IMPACT = "uc7_impact"
    IMPACT = "impact"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given impact column."""
        return "viridis"


class Scenarios(enum.StrEnum):
    """Column definitions for scenarios."""

    CURRENT = "current"
    FORECAST = "forecast"

    @classmethod
    def scenario_or_column(cls) -> str:
        """Return the name of the scenario column."""
        return f"{cls.CURRENT}_or_{cls.FORECAST}"


class UserClasses(enum.StrEnum):
    """Definitions for user classes."""

    USER_CLASS_1 = "uc1"
    USER_CLASS_2 = "uc2"
    USER_CLASS_3 = "uc3"
    USER_CLASS_4 = "uc4"
    USER_CLASS_5 = "uc5"
    USER_CLASS_6 = "uc6"
    USER_CLASS_7 = "uc7"


class TimePeriods(enum.StrEnum):
    """Definitions for time periods."""

    TIME_PERIOD_1 = "TS1"
    TIME_PERIOD_2 = "TS2"
    TIME_PERIOD_3 = "TS3"


class OSRoadStructure(enum.StrEnum):
    """Definitions for OS road structure types."""

    BRIDGE = "Road On Bridge"
    TUNNEL = "Road In Tunnel"


class OSRailStructure(enum.StrEnum):
    """Definitions for OS rail structure types."""

    CUTTING = "In Cutting"
    EMBANKMENT = "On Embankment"
    UNDER_STRUCTURE = "Under Structure"
    ON_STRUCTURE = "On Structure"
    BRIDGE = "On Bridge"
    TUNNEL = "In Tunnel"
    BUILDING = "In Building"


class VulnerabilityModifier(float, enum.Enum):
    """Definitions for vulnerability modifiers."""

    VERY_LOW = 0.8
    LOW = 0.9
    NEUTRAL = 1.0
    HIGH = 1.1
    VERY_HIGH = 1.2


class OSRoadCols(enum.StrEnum):
    """Definitions for OS road columns."""

    ID = "id"
    ROAD_CLASSIFICATION = "road_classification"
    ROAD_FUNCTION = "road_function"
    FORM_OF_WAY = "form_of_way"
    ROAD_CLASSIFICATION_NUMBER = "road_classification_number"
    NAME_1 = "name_1"
    ROAD_STRUCTURE = "road_structure"
    PRIMARY_ROUTE = "primary_route"
    TRUNK_ROAD = "trunk_road"
