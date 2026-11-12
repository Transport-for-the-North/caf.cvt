"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import openpyxl

# GLOBAL CONSTANTS

BNG_CRS = "EPSG:27700"  # British National Grid CRS, for use in spatially merging datasets
ID_COL = "id"
NA_COL = "N/A"
GEOMETRY_COL = "geometry"
FLOODING_TILE_SIZE_M = 10000
RASTER_UNAVAILABLE = -9999

# CLASSES


class RiskColumn(enum.StrEnum):
    """Plotting column definition base class."""

    def get_cmap(self) -> ColourMaps:
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

    def get_cmap(self) -> ColourMaps:
        """Return the appropriate colormap for a given extreme weather subhazard column."""
        cmap_mapping = {
            ExtremeWeatherRiskCols.EXTREME_HEAT: ColourMaps.REDS,
            ExtremeWeatherRiskCols.EXTREME_COLD: ColourMaps.BLUES,
            ExtremeWeatherRiskCols.DROUGHT: ColourMaps.ORANGES,
            ExtremeWeatherRiskCols.STORM: ColourMaps.BLUES,
        }
        return cmap_mapping[self]


class FloodingTypes(enum.StrEnum):
    """Enumeration for different types of flooding."""

    RIVERS_SEA = "rivers_sea"
    SURFACE_WATER = "surface_water"
    GROUNDWATER = "groundwater"


class FloodingRiskCols(RiskColumn):
    """Column definitions for flooding subhazard layers."""

    RIVERS_SEA = "rivers_sea_flooding_risk"
    SURFACE_WATER = "surface_water_flooding_risk"
    # TODO (DJ): #41 Add groundwater flooding column

    def get_cmap(self) -> ColourMaps:
        """Return the appropriate colormap for a given flooding subhazard column."""
        return ColourMaps.BLUES

    @classmethod
    def get_risk_map(cls) -> dict[int | str, float]:
        """Return mapping of flooding risk categories to numerical risk values."""
        return {
            0: 0,  # No risk areas stay 0
            "Unavailable": np.nan,
            "Very low": 0,  # Less than 0.1% chance of flooding
            "Low": 0.1,  # 0.1% to 1% chance of flooding
            "Medium": 1,  # 1% to 3.3% chance of flooding
            "High": 3.3,  # Greater than 3.3% chance of flooding
        }


class GroundStabilityRiskCols(RiskColumn):
    """Column definitions for ground stability subhazard layers."""

    COLLAPSIBLE_DEPOSITS = "collapsible_deposits_risk"
    COMPRESSIBLE_GROUND = "compressible_ground_risk"
    LANDSLIDES = "landslides_risk"
    RUNNING_SAND = "running_sand_risk"
    SHRINK_SWELL = "shrink_swell_risk"
    SOLUBLE_ROCKS = "soluble_rocks_risk"

    def get_cmap(self) -> ColourMaps:
        """Return the appropriate colormap for a given ground stability subhazard column."""
        return ColourMaps.ORANGES


class MainHazardRiskCols(RiskColumn):
    """Column definitions for main hazard layers."""

    EXTREME_WEATHER = "extreme_weather_risk"
    FLOODING = "flooding_risk"
    GROUND_STABILITY = "ground_stability_risk"
    COASTAL_EROSION = "coastal_erosion_risk"

    def get_cmap(self) -> ColourMaps:
        """Return the appropriate colormap for a given hazard column."""
        cmap_mapping = {
            MainHazardRiskCols.EXTREME_WEATHER: ColourMaps.REDS,
            MainHazardRiskCols.FLOODING: ColourMaps.BLUES,
            MainHazardRiskCols.GROUND_STABILITY: ColourMaps.ORANGES,
            MainHazardRiskCols.COASTAL_EROSION: ColourMaps.PURPLES,
        }
        return cmap_mapping[self]

    def get_sub_hazards(self) -> list[RiskColumn]:
        """Return a list of sub-hazard columns for the main hazard."""
        sub_hazard_mapping: dict[MainHazardRiskCols, list[RiskColumn]] = {
            MainHazardRiskCols.EXTREME_WEATHER: list(ExtremeWeatherRiskCols),
            MainHazardRiskCols.FLOODING: list(FloodingRiskCols),
            MainHazardRiskCols.GROUND_STABILITY: list(GroundStabilityRiskCols),
            MainHazardRiskCols.COASTAL_EROSION: [],
        }
        return sub_hazard_mapping[self]


class FloodingBands(enum.Enum):
    """Enumeration for different flood bands."""

    RIVERS_CURRENT = 0
    RIVERS_FORECAST = 1
    SURFACE_CURRENT = 2
    SURFACE_FORECAST = 3
    FLOODING_CURRENT = 4
    FLOODING_FORECAST = 5

    @classmethod
    def get_band_map(cls) -> dict[RiskColumn, tuple[int, int]]:
        """Return mapping of flooding risk columns to their corresponding flooding bands."""
        return {
            FloodingRiskCols.RIVERS_SEA: (
                FloodingBands.RIVERS_CURRENT.value,
                FloodingBands.RIVERS_FORECAST.value,
            ),
            FloodingRiskCols.SURFACE_WATER: (
                FloodingBands.SURFACE_CURRENT.value,
                FloodingBands.SURFACE_FORECAST.value,
            ),
            MainHazardRiskCols.FLOODING: (
                FloodingBands.FLOODING_CURRENT.value,
                FloodingBands.FLOODING_FORECAST.value,
            ),
        }


class ExtremeHeatCols(RiskColumn):
    """Column definitions for extreme heat subhazard layers."""

    MAX_TEMP_SUMMER = "max_temp_summer"
    HOT_SUMMER_DAYS = "hot_summer_days"
    EXTREME_SUMMER_DAYS = "extreme_summer_days"

    def get_cmap(self) -> ColourMaps:
        """Return extreme heat colourmap."""
        return ColourMaps.REDS


class ExtremeColdCols(RiskColumn):
    """Column definitions for extreme cold subhazard layers."""

    MIN_TEMP_WINTER = "min_temp_winter"
    FROST_DAYS = "frost_days"
    ICING_DAYS = "icing_days"

    def get_cmap(self) -> ColourMaps:
        """Return extreme cold colourmap."""
        return ColourMaps.BLUES


class DroughtCols(RiskColumn):
    """Column definitions for drought subhazard layers."""

    DROUGHT_SEVERITY_INDEX = "drought_severity_index"
    PRECIP_SUMMER = "precip_summer"

    def get_cmap(self) -> ColourMaps:
        """Return drought colourmap."""
        return ColourMaps.ORANGES


class StormCols(RiskColumn):
    """Column definitions for storm subhazard layers."""

    RAIN_DAYS = "10mm_rain_days"
    PRECIP_WINTER = "precip_winter"
    EXCEEDANCE_DAYS = "avg_exceedance_days"
    WIND_SPEED = "wind_speed"
    WIND_DRIVEN_RAIN_INDEX = "wind_driven_rain_index"

    def get_cmap(self) -> ColourMaps:
        """Return storm colourmap."""
        return ColourMaps.BLUES


class DemandCols(RiskColumn):
    """Column definitions for demand cold layers."""

    DEMAND = "demand"

    def get_cmap(self) -> ColourMaps:
        """Return demand colourmap."""
        return ColourMaps.VIRIDIS


class Scenarios(enum.StrEnum):
    """Column definitions for scenarios."""

    CURRENT = "current"
    FORECAST = "forecast"

    @classmethod
    def scenario_or_column(cls) -> str:
        """Return the name of the scenario column."""
        return f"{cls.CURRENT}_or_{cls.FORECAST}"


class ImpactCols(RiskColumn):
    """Column definitions for impact layers."""

    IMPACT = "impact"

    def get_cmap(self) -> ColourMaps:
        """Return the appropriate colormap for a given impact column."""
        return ColourMaps.VIRIDIS


class TimePeriods(enum.StrEnum):
    """Definitions for time periods."""

    TIME_PERIOD_1 = "TS1"
    TIME_PERIOD_2 = "TS2"
    TIME_PERIOD_3 = "TS3"


class OSRoadStructure(enum.StrEnum):
    """Definitions for OS road structure types."""

    BRIDGE = "Road On Bridge"
    TUNNEL = "Road In Tunnel"


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

    @classmethod
    def get_layer_name(cls) -> str:
        """Return the name of the layer for OS roads."""
        return "road_link"


class NoHAMRoadCols(enum.StrEnum):
    """Definitions for NoHAM road columns."""

    LINK_ID = "link_id"


class OSRailCols(enum.StrEnum):
    """Definitions for OS rail columns."""

    ID = "osid"
    DESCRIPTION = "description"
    GAUGE = "gauge"
    STRUCTURE = "structure"
    PHYSICAL_LEVEL = "physicallevel"
    RAILWAY_USE = "railwayuse"
    TRACK_REPRESENTATION = "trackrepresentation"
    DIRECTION = "direction"
    OPERATIONAL_STATUS = "operationalstatus"

    @classmethod
    def get_descriptive_cols(cls) -> list[str]:
        """Return a list of the retained descriptive columns for OS rail."""
        return [
            OSRailCols.DESCRIPTION,
            OSRailCols.GAUGE,
            OSRailCols.STRUCTURE,
            OSRailCols.PHYSICAL_LEVEL,
            OSRailCols.RAILWAY_USE,
            OSRailCols.TRACK_REPRESENTATION,
            OSRailCols.DIRECTION,
        ]

    @classmethod
    def rename_map(cls) -> dict[str, str]:
        """Return rename map for OS rail columns."""
        return {
            OSRailCols.PHYSICAL_LEVEL: "physical_level",
            OSRailCols.RAILWAY_USE: "railway_use",
            OSRailCols.TRACK_REPRESENTATION: "track_representation",
        }


class OSRailOperationalStatus(enum.StrEnum):
    """Definitions for values of OS rail operational status."""

    ACTIVE = "Active"
    INACTIVE = "Inactive"


class OSRailDescription(enum.StrEnum):
    """Definitions for values of OS rail description."""

    MAIN_LINE = "Main Line"
    TRAM = "Tram"
    RAPID_TRANSPORT_SYSTEM = "Rapid Transport System"
    MAIN_LINE_AND_TRAM = "Main Line And Tram"
    MAIN_LINE_AND_RAPID_TRANSPORT_SYSTEM = "Main Line And Rapid Transport System"
    PRESERVED = "Preserved"
    FUNICULAR = "Funicular"
    MINERAL = "Mineral"
    STATIC_MUSEUM = "Static Museum"


class OSRailRailwayUse(enum.StrEnum):
    """Definitions for values of OS rail railway use."""

    FREIGHT = "Freight"
    PASSENGER = "Passenger"
    FREIGHT_AND_PASSENGER = "Freight And Passenger"
    MINERAL = "Mineral"
    PRESERVED = "Preserved"
    AMUSEMENT = "Amusement"


class OSRailTrackRepresentation(enum.StrEnum):
    """Definitions for values of OS rail track representation."""

    SIDING = "Siding"
    MULTIPLE_TRACKS = "Multiple Tracks"
    SINGLE_TRACK = "Single Track"


class OSRailGauge(enum.StrEnum):
    """Definitions for values of OS rail gauge."""

    STANDARD = "Standard"
    NARROW = "Narrow"
    BROAD = "Broad"


class OSRailStructure(enum.StrEnum):
    """Definitions for OS rail structure types."""

    CUTTING = "In Cutting"
    EMBANKMENT = "On Embankment"
    UNDER_STRUCTURE = "Under Structure"
    ON_STRUCTURE = "On Structure"
    BRIDGE = "On Bridge"
    TUNNEL = "In Tunnel"
    BUILDING = "In Building"


class OSRailPhysicalLevel(enum.StrEnum):
    """Definitions for OS rail physical levels."""

    SURFACE_LEVEL = "Surface Level"
    UNDERGROUND = "Underground"
    LEVEL_1 = "Level 1"
    LEVEL_2 = "Level 2"


class OSRailDirection(enum.StrEnum):
    """Definitions for OS rail directions."""

    BOTH_DIRECTIONS = "Both Directions"


class NaPTANCols(enum.StrEnum):
    """Column definitions for NaPTAN data."""

    ID = "ATCOCode"
    NAME = "CommonName"
    STREET = "Street"
    LOCALITY = "LocalityName"
    PARENT_LOCALITY = "ParentLocalityName"
    STOP_TYPE = "StopType"
    LONGITUDE = "Longitude"
    LATITUDE = "Latitude"
    STATUS = "Status"

    @classmethod
    def rename_map(cls) -> dict[str, str]:
        """Return mapping of original column names to desired column names for NaPTAN data."""
        return {
            NaPTANCols.ID: ID_COL,
            NaPTANCols.NAME: "name",
            NaPTANCols.STREET: "street",
            NaPTANCols.LOCALITY: "locality",
            NaPTANCols.PARENT_LOCALITY: "parent_locality",
        }


class MasterMapSiteCols(enum.StrEnum):
    """Column definitions for MasterMap sites."""

    ID = "toid"
    FUNCTION_STATUS = "function_status"
    FUNCTION_THEME = "function_theme"
    FUNCTION = "function"
    DISTINCTIVE_NAME_1 = "distinctive_name_1"
    DISTINCTIVE_NAME_2 = "distinctive_name_2"
    DISTINCTIVE_NAME_3 = "distinctive_name_3"
    DISTINCTIVE_NAME_4 = "distinctive_name_4"
    STAKEHOLDER_1 = "stakeholder_1"
    STAKEHOLDER_1_ROLE = "stakeholder_1_role"
    EXTENT_DEFINITION = "extent_definition"
    PERIMETER = "perimeter"
    AREA = "area"

    @classmethod
    def get_layer_name(cls) -> str:
        """Return the name of the layer for MasterMap sites."""
        return "functional_site"

    @classmethod
    def get_descriptive_cols(cls) -> list[str]:
        """Return list of descriptive columns for MasterMap sites."""
        return [
            MasterMapSiteCols.FUNCTION_THEME,
            MasterMapSiteCols.FUNCTION,
            MasterMapSiteCols.DISTINCTIVE_NAME_1,
            MasterMapSiteCols.DISTINCTIVE_NAME_2,
            MasterMapSiteCols.DISTINCTIVE_NAME_3,
            MasterMapSiteCols.DISTINCTIVE_NAME_4,
            MasterMapSiteCols.STAKEHOLDER_1,
            MasterMapSiteCols.STAKEHOLDER_1_ROLE,
            MasterMapSiteCols.EXTENT_DEFINITION,
        ]


class MasterMapFunctionStatus(enum.StrEnum):
    """Function status definitions for MasterMap sites."""

    OPERATIONAL = "Operational"
    OUT_OF_SERVICE = "Out of Service"


class MasterMapFunction(enum.StrEnum):
    """Function definitions for MasterMap sites."""

    RAILWAY_STATION = "Railway Station"
    BUS_STATION = "Bus Station"
    COACH_STATION = "Coach Station"
    TRAM_STATION = "Tram Station"
    FERRY_TERMINAL = "Ferry Terminal"
    AIRPORT = "Airport"


class MasterMapFunctionTheme(enum.StrEnum):
    """Function theme definitions for MasterMap sites."""

    ROAD_TRANSPORT = "Road Transport"
    RAIL_TRANSPORT = "Rail Transport"
    WATER_TRANSPORT = "Water Transport"
    AIR_TRANSPORT = "Air Transport"


class NationalCycleNetworkCols(enum.StrEnum):
    """Column definitions for the National Cycle Network."""

    ID = "SegmentID"
    DESCRIPTION = "Desc_"
    GREENWAY = "Greenway"
    ROUTE_TYPE = "RouteType"
    ROUTE_NUMBER = "RouteNo"
    LINK_NUMBER = "LinkNo"
    SURFACE = "Surface"
    QUALITY = "Quality"
    LIGHTING = "Lighting"
    ROAD_CLASS = "RoadClass"

    @classmethod
    def rename_map(cls) -> dict[str, str]:
        """Return mapping of column names to standardized names."""
        return {
            NationalCycleNetworkCols.ID: "id",
            NationalCycleNetworkCols.DESCRIPTION: "description",
            NationalCycleNetworkCols.GREENWAY: "greenway",
            NationalCycleNetworkCols.ROUTE_TYPE: "route_type",
            NationalCycleNetworkCols.ROUTE_NUMBER: "route_number",
            NationalCycleNetworkCols.LINK_NUMBER: "link_number",
            NationalCycleNetworkCols.SURFACE: "surface",
            NationalCycleNetworkCols.QUALITY: "quality",
            NationalCycleNetworkCols.LIGHTING: "lighting",
            NationalCycleNetworkCols.ROAD_CLASS: "road_class",
        }


class AssetTypes(enum.StrEnum):
    """Asset Type definitions."""

    ROAD = "road"
    RAIL = "rail"


class MasterMapHighwaysCols(enum.StrEnum):
    """Column definitions for MasterMap Highways."""

    ID = "toid"
    ROAD_STRUCTURE = "road_structure"
    START_GRADE_SEPARATION = "start_grade_separation"
    END_GRADE_SEPARATION = "end_grade_separation"

    @classmethod
    def get_layer_name(cls) -> str:
        """Return the name of the layer for OS roads."""
        return "road_link"


class OSRoadLookupCols(enum.StrEnum):
    """Column definitions for OS Road Lookup."""

    MASTERMAP_ID = "Roadlink_Id"
    OS_OPEN_ROADS_ID = "OSOpenRoads_RoadLinkIdentifier"


class ColourMaps(enum.StrEnum):
    """Matplotlib colour map definitions."""

    GREEN_YELLOW_RED = "RdYlGn_r"
    ORANGES = "Oranges"
    ROYAL_BLUE = "royalblue"
    DARK_ORANGE = "darkorange"
    REDS = "Reds"
    BLUES = "Blues"
    PURPLES = "Purples"
    VIRIDIS = "viridis"


class HexColourCodes(enum.StrEnum):
    """Hex colour code definitions."""

    BLACK = "000000"
    WHITE = "FFFFFF"
    DARK_GREY = "a9a9a9"
    GREY = "808080"
    LIGHT_GREEN = "90ee90"
    PALE_VIOLET_RED = "db7093"
    DARK_BLUE = "00008b"
    TEAL = "008080"
    GOLD = "ffd700"
    DARK_ORANGE = "ff8c00"
    RED = "F8696B"
    DARK_RED = "8b0000"
    GREEN = "63BE7B"
    YELLOW = "FFEB84"


class RiskCategories(enum.StrEnum):
    """Risk category definitions."""

    VERY_HIGH = "Very High"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"

    @classmethod
    def get_risk_bands(cls) -> dict[RiskCategories, tuple[int, int]]:
        """Return a dictionary of risk bands for each risk category."""
        return {
            RiskCategories.VERY_LOW: (0, 20),
            RiskCategories.LOW: (20, 40),
            RiskCategories.MEDIUM: (40, 60),
            RiskCategories.HIGH: (60, 80),
            RiskCategories.VERY_HIGH: (80, 100),
        }

    @classmethod
    def get_risk_band_colours(cls) -> dict[RiskCategories, str]:
        """Return a dictionary of colours for each risk category."""
        return {
            RiskCategories.VERY_LOW: HexColourCodes.DARK_BLUE,
            RiskCategories.LOW: HexColourCodes.TEAL,
            RiskCategories.MEDIUM: HexColourCodes.GOLD,
            RiskCategories.HIGH: HexColourCodes.DARK_ORANGE,
            RiskCategories.VERY_HIGH: HexColourCodes.DARK_RED,
        }


@dataclass
class CellStyle:
    """Cell style definitions for risk summary spreadsheet."""

    fill_colour: str
    text_colour: str
    bold: bool
    border: openpyxl.styles.Border | None
