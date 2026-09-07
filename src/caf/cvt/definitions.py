"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import enum

# GLOBAL CONSTANTS

BNG_CRS = "EPSG:27700"  # British National Grid CRS, for use in spatially merging datasets

# CLASSES


class RiskColumn(enum.StrEnum):
    """Plotting column definition base class."""

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given plotting column."""
        raise NotImplementedError("Subclasses must implement the get_cmap method.")

    @property
    def base_name(self) -> str:
        """Return the base name of the plotting column."""
        return self.removesuffix("_risk")


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

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Extreme Weather."""
        return {
            ExtremeWeatherRiskCols.EXTREME_HEAT: 0.25,
            ExtremeWeatherRiskCols.EXTREME_COLD: 0.25,
            ExtremeWeatherRiskCols.DROUGHT: 0.25,
            ExtremeWeatherRiskCols.STORM: 0.25,
        }


class FloodingRiskCols(RiskColumn):
    """Column definitions for flooding subhazard layers."""

    RIVERS_SEA = "rivers_sea_flooding_risk"
    SURFACE_WATER = "surface_water_flooding_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given flooding subhazard column."""
        return "Blues"

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Flooding."""
        return {
            FloodingRiskCols.RIVERS_SEA: 0.5,
            FloodingRiskCols.SURFACE_WATER: 0.5,
        }


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

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Ground Stability."""
        return {
            GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: 0.10,
            GroundStabilityRiskCols.COMPRESSIBLE_GROUND: 0.10,
            GroundStabilityRiskCols.LANDSLIDES: 0.10,
            GroundStabilityRiskCols.RUNNING_SAND: 0.10,
            GroundStabilityRiskCols.SHRINK_SWELL: 0.10,
            GroundStabilityRiskCols.SOLUBLE_ROCKS: 0.10,
            GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE: 0.40,
        }


class CoastalErosionRiskCols(RiskColumn):
    """Column definitions for coastal erosion subhazard layers."""

    EROSION = "erosion_risk"
    GIZ = "giz_risk"

    def get_cmap(self) -> str:
        """Return the appropriate colormap for a given coastal erosion subhazard column."""
        return "Purples"

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Coastal Erosion."""
        return {
            CoastalErosionRiskCols.EROSION: 0.9,
            CoastalErosionRiskCols.GIZ: 0.1,
        }


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
            MainHazardRiskCols.EXTREME_WEATHER: ExtremeWeatherRiskCols.get_weights(),
            MainHazardRiskCols.FLOODING: FloodingRiskCols.get_weights(),
            MainHazardRiskCols.GROUND_STABILITY: GroundStabilityRiskCols.get_weights(),
            MainHazardRiskCols.COASTAL_EROSION: CoastalErosionRiskCols.get_weights(),
        }

        return weights_mapping[self]


class ExtremeHeatCols(RiskColumn):
    """Column definitions for extreme heat subhazard layers."""

    MAX_TEMP_SUMMER = "max_temp_summer"
    HOT_SUMMER_DAYS = "hot_summer_days"
    EXTREME_SUMMER_DAYS = "extreme_summer_days"

    def get_cmap(self) -> str:
        """Return extreme heat colourmap."""
        return "Reds"

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Extreme Heat."""
        return {
            ExtremeHeatCols.MAX_TEMP_SUMMER: 0.5,
            ExtremeHeatCols.HOT_SUMMER_DAYS: 0.25,
            ExtremeHeatCols.EXTREME_SUMMER_DAYS: 0.25,
        }


class ExtremeColdCols(RiskColumn):
    """Column definitions for extreme cold subhazard layers."""

    MIN_TEMP_WINTER = "min_temp_winter"
    FROST_DAYS = "frost_days"
    ICING_DAYS = "icing_days"

    def get_cmap(self) -> str:
        """Return extreme cold colourmap."""
        return "Blues"

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Extreme Cold."""
        return {
            ExtremeColdCols.MIN_TEMP_WINTER: 0.5,
            ExtremeColdCols.FROST_DAYS: 0.25,
            ExtremeColdCols.ICING_DAYS: 0.25,
        }


class DroughtCols(RiskColumn):
    """Column definitions for drought subhazard layers."""

    DROUGHT_SEVERITY_INDEX = "drought_severity_index"
    PRECIP_SUMMER = "precip_summer"

    def get_cmap(self) -> str:
        """Return drought colourmap."""
        return "Oranges"

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Drought."""
        return {
            DroughtCols.DROUGHT_SEVERITY_INDEX: 0.75,
            DroughtCols.PRECIP_SUMMER: 0.25,
        }


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

    @classmethod
    def get_weights(cls) -> dict[RiskColumn, float]:
        """Return appropriate weights for Storm."""
        return {
            StormCols.WIND_SPEED: 0.3,
            StormCols.EXCEEDANCE_DAYS: 0.2,
            StormCols.PRECIP_WINTER: 0.15,
            StormCols.RAIN_DAYS: 0.15,
            StormCols.WIND_DRIVEN_RAIN_INDEX: 0.2,
        }


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

    def get_vulnerability(self) -> dict[RiskColumn, VulnerabilityModifier]:
        """Return the vulnerability modifiers for the road structure type."""
        mapping = {
            OSRoadStructure.BRIDGE: {
                ExtremeWeatherRiskCols.EXTREME_HEAT: VulnerabilityModifier.VERY_HIGH,
                ExtremeWeatherRiskCols.EXTREME_COLD: VulnerabilityModifier.VERY_HIGH,
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.HIGH,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.VERY_HIGH,
            },
            OSRoadStructure.TUNNEL: {
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.LOW,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.HIGH,
                FloodingRiskCols.SURFACE_WATER: VulnerabilityModifier.VERY_HIGH,
            },
        }
        return mapping[self]


class OSRailStructure(enum.StrEnum):
    """Definitions for OS rail structure types."""

    CUTTING = "In Cutting"
    EMBANKMENT = "On Embankment"
    UNDER_STRUCTURE = "Under Structure"
    ON_STRUCTURE = "On Structure"
    BRIDGE = "On Bridge"
    TUNNEL = "In Tunnel"
    BUILDING = "In Building"

    def get_vulnerability(self) -> dict[RiskColumn, VulnerabilityModifier]:
        """Return the vulnerability modifiers for the rail structure type."""
        mapping: dict[OSRailStructure, dict[RiskColumn, VulnerabilityModifier]] = {
            OSRailStructure.CUTTING: {
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.HIGH,
                ExtremeWeatherRiskCols.DROUGHT: VulnerabilityModifier.HIGH,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.HIGH,
                FloodingRiskCols.SURFACE_WATER: VulnerabilityModifier.VERY_HIGH,
                GroundStabilityRiskCols.LANDSLIDES: VulnerabilityModifier.VERY_HIGH,
            },
            OSRailStructure.EMBANKMENT: {
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.HIGH,
                ExtremeWeatherRiskCols.DROUGHT: VulnerabilityModifier.HIGH,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.VERY_HIGH,
                FloodingRiskCols.SURFACE_WATER: VulnerabilityModifier.VERY_HIGH,
                GroundStabilityRiskCols.LANDSLIDES: VulnerabilityModifier.VERY_HIGH,
                GroundStabilityRiskCols.SHRINK_SWELL: VulnerabilityModifier.VERY_HIGH,
                GroundStabilityRiskCols.SHRINK_SWELL_GEOCLIMATE: VulnerabilityModifier.VERY_HIGH,
            },
            OSRailStructure.BRIDGE: {
                ExtremeWeatherRiskCols.EXTREME_HEAT: VulnerabilityModifier.VERY_HIGH,
                ExtremeWeatherRiskCols.EXTREME_COLD: VulnerabilityModifier.VERY_HIGH,
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.HIGH,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.VERY_HIGH,
            },
            OSRailStructure.TUNNEL: {
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.LOW,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.HIGH,
                FloodingRiskCols.SURFACE_WATER: VulnerabilityModifier.VERY_HIGH,
            },
            OSRailStructure.BUILDING: {
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.VERY_LOW,
            },
            OSRailStructure.UNDER_STRUCTURE: {},
            OSRailStructure.ON_STRUCTURE: {
                ExtremeWeatherRiskCols.STORM: VulnerabilityModifier.HIGH,
                ExtremeWeatherRiskCols.EXTREME_HEAT: VulnerabilityModifier.HIGH,
                FloodingRiskCols.RIVERS_SEA: VulnerabilityModifier.HIGH,
            },
        }
        return mapping[self]


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


class OSRailCols(enum.StrEnum):
    """Definitions for OS rail columns."""

    ID = "osid"
    DESCRIPTION = "description"
    STRUCTURE = "structure"
    PHYSICAL_LEVEL = "physicallevel"
    RAILWAY_USE = "railwayuse"
    TRACK_REPRESENTATION = "trackrepresentation"
    OPERATIONAL_STATUS = "operationalstatus"


class AssetTypes(enum.StrEnum):
    """Asset Type definitions."""

    ROAD = "road"
    RAIL = "rail"

    def get_asset_hazard_weights(
        self, main_hazard: MainHazardRiskCols
    ) -> dict[RiskColumn, float]:
        """Return asset-specific hazard weights."""
        default_weights = main_hazard.get_weights()

        if self == AssetTypes.ROAD:
            return default_weights
        if self == AssetTypes.RAIL:
            if main_hazard == MainHazardRiskCols.EXTREME_WEATHER:
                return {
                    ExtremeWeatherRiskCols.EXTREME_HEAT: 0.20,
                    ExtremeWeatherRiskCols.EXTREME_COLD: 0.21,
                    ExtremeWeatherRiskCols.DROUGHT: 0.14,
                    ExtremeWeatherRiskCols.STORM: 0.45,
                }
            return default_weights
        raise ValueError(f"Unknown asset type: {self}")
