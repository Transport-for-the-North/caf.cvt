"""Module for setting up the model configuration."""

import pathlib
from typing import ClassVar, Self, cast

import caf.toolkit as ctk
import pydantic

from caf.cvt.definitions import (
    AssetTypes,
    DroughtCols,
    ExtremeColdCols,
    ExtremeHeatCols,
    ExtremeWeatherRiskCols,
    FloodingRiskCols,
    FloodingTypes,
    GroundStabilityRiskCols,
    MainHazardRiskCols,
    RiskColumn,
    Scenarios,
    StormCols,
)

### CONFIG SET UP


class ZipFileEntry(ctk.BaseConfig):
    """Configuration for a file within a zip archive.

    Attributes
    ----------
    zip_path : pathlib.Path
        Path to the zip file.
    file_path : pathlib.Path
        Path within the zip file.
    """

    zip_path: pathlib.Path
    file_path: pathlib.Path


class NestedZipFileEntry(ctk.BaseConfig):
    """Configuration for a nested zip file entry.

    Attributes
    ----------
    outer_zip_path : pathlib.Path
        Path to the outer zip file.
    inner_zip_path : pathlib.Path
        Path to the inner zip file within the outer zip.
    file_path : pathlib.Path
        Path to the file within the inner zip file.
    """

    outer_zip_path: pathlib.Path
    inner_zip_path: pathlib.Path
    file_path: pathlib.Path


# -------------------------
# CATEGORY CONFIG CLASSES
# -------------------------


class PathConfig(ctk.BaseConfig):
    """Configuration for model paths.

    Paths are declared in one registry to avoid a very large number of individual
    properties while keeping the existing public attribute names.
    """

    root: pydantic.DirectoryPath
    raw_input: pydantic.DirectoryPath

    _resolved_paths: dict[str, pathlib.Path] = pydantic.PrivateAttr(default_factory=dict)
    # Keys of PATH SPECS are the attribute names used to access the resolved paths.
    # The tuple values in PATH SPECS are of the form (parent_attr, sub_path, is_dir)
    _PATH_SPECS: ClassVar[dict[str, tuple[str, str, bool]]] = {
        "model_input": ("root", "model input", True),
        "model_interim_output": ("root", "model interim outputs", True),
        "model_output": ("root", "model outputs", True),
        "log": ("root", "logging", True),
        "audit": ("root", "audit", True),
        "infrastructure_model_input": ("model_input", "Infrastructure", True),
        "hazards_model_input": ("model_input", "Hazards", True),
        "demand_model_input": ("model_input", "Demand", True),
        "road_model_input": ("infrastructure_model_input", "Road", True),
        "rail_model_input": ("infrastructure_model_input", "Rail", True),
        "other_model_input": ("infrastructure_model_input", "Other", True),
        "bespoke_model_input": ("model_input", "Bespoke", True),
        "os_road_model_input": ("road_model_input", "os_road.gpkg", False),
        "noham_road_model_input": ("road_model_input", "noham_2023.gpkg", False),
        "model_roads_model_input": ("road_model_input", "model_roads.gpkg", False),
        "passenger_rail_model_input": ("rail_model_input", "passenger_rail_links.gpkg", False),
        "freight_rail_model_input": ("rail_model_input", "freight_rail_links.gpkg", False),
        "airports_model_input": ("other_model_input", "airports.gpkg", False),
        "bus_stops_model_input": ("other_model_input", "bus_stops.gpkg", False),
        "train_stations_model_input": ("other_model_input", "train_stations.gpkg", False),
        "tram_stations_model_input": ("other_model_input", "tram_stations.gpkg", False),
        "rapid_transport_stations_model_input": (
            "other_model_input",
            "rapid_transport_stations.gpkg",
            False,
        ),
        "ferry_terminals_model_input": ("other_model_input", "ferry_terminals.gpkg", False),
        "bus_coach_stations_model_input": (
            "other_model_input",
            "bus_coach_stations.gpkg",
            False,
        ),
        "tram_network_model_input": ("other_model_input", "tram_network.gpkg", False),
        "rapid_transport_network_model_input": (
            "other_model_input",
            "rapid_transport_network.gpkg",
            False,
        ),
        "national_cycle_network_model_input": (
            "other_model_input",
            "national_cycle_network.gpkg",
            False,
        ),
        "nexus_metro_stations_model_input": (
            "bespoke_model_input",
            "nexus_metro_stations.gpkg",
            False,
        ),
        "hazard_model_input": ("model_input", "Hazards", True),
        "extreme_weather_model_input": ("hazard_model_input", "Extreme Weather", True),
        "flooding_model_input": ("hazard_model_input", "Flooding", True),
        "ground_stability_model_input": ("hazard_model_input", "Ground Stability", True),
        "coastal_erosion_model_input": ("hazard_model_input", "Coastal Erosion", True),
        "hazard_grid_model_input": ("hazard_model_input", "hazard_grid.gpkg", False),
        "summer_max_temperature_model_input": (
            "extreme_weather_model_input",
            "summer_max_temperature_change.csv",
            False,
        ),
        "winter_min_temperature_model_input": (
            "extreme_weather_model_input",
            "winter_min_temperature_change.csv",
            False,
        ),
        "summer_precipitation_model_input": (
            "extreme_weather_model_input",
            "summer_precipitation_change.csv",
            False,
        ),
        "winter_precipitation_model_input": (
            "extreme_weather_model_input",
            "winter_precipitation_change.csv",
            False,
        ),
        "rain_days_model_input": ("extreme_weather_model_input", "10mm_rain_days.gpkg", False),
        "drought_severity_index_model_input": (
            "extreme_weather_model_input",
            "drought_severity_index.gpkg",
            False,
        ),
        "hot_summer_days_model_input": (
            "extreme_weather_model_input",
            "hot_summer_days.csv",
            False,
        ),
        "extreme_summer_days_model_input": (
            "extreme_weather_model_input",
            "extreme_summer_days.csv",
            False,
        ),
        "frost_days_model_input": ("extreme_weather_model_input", "frost_days.csv", False),
        "icing_days_model_input": ("extreme_weather_model_input", "icing_days.csv", False),
        "wind_speed_model_input": ("extreme_weather_model_input", "wind_speed.gpkg", False),
        "wind_driven_rain_model_input": (
            "extreme_weather_model_input",
            "wind_driven_rain_index.gpkg",
            False,
        ),
        "groundwater_flooding_model_input": (
            "flooding_model_input",
            "groundwater.gpkg",
            False,
        ),
        "freight_demand_model_input": (
            "demand_model_input",
            "freight_network_demand.gpkg",
            False,
        ),
        "noham_demand_model_input": ("demand_model_input", "noham_network_demand.gpkg", False),
        "model_roads_demand_model_input": (
            "demand_model_input",
            "model_roads_demand.gpkg",
            False,
        ),
        "nexus_metro_demand_model_input": (
            "demand_model_input",
            "nexus_metro_demand.gpkg",
            False,
        ),
        "extreme_weather_risk_model_interim_output": (
            "model_interim_output",
            "extreme_weather_risk.gpkg",
            False,
        ),
        "flooding_risk_vector_overlay_model_interim_output": (
            "model_interim_output",
            "flooding_risk_vector_overlay.gpkg",
            False,
        ),
        "flooding_risk_vector_model_interim_output": (
            "model_interim_output",
            "flooding_risk_vector.gpkg",
            False,
        ),
        "flooding_risk_raster_model_interim_output": (
            "model_interim_output",
            "flooding_risk_raster.tif",
            False,
        ),
        "ground_stability_risk_model_interim_output": (
            "model_interim_output",
            "ground_stability_risk.gpkg",
            False,
        ),
        "coastal_erosion_risk_model_interim_output": (
            "model_interim_output",
            "coastal_erosion_risk.gpkg",
            False,
        ),
        "road_model_output": ("model_output", "Road", True),
        "rail_model_output": ("model_output", "Rail", True),
        "other_model_output": ("model_output", "Other", True),
        "os_road_model_output": ("road_model_output", "OS Road", True),
        "noham_model_output": ("road_model_output", "NoHAM", True),
        "model_roads_model_output": ("road_model_output", "Model Roads", True),
        "passenger_rail_model_output": ("rail_model_output", "Passenger Rail", True),
        "freight_rail_model_output": ("rail_model_output", "Freight Rail", True),
        "train_stations_model_output": ("other_model_output", "Train Stations", True),
        "airports_model_output": ("other_model_output", "Airports", True),
        "bus_coach_stations_model_output": (
            "other_model_output",
            "Bus and Coach Stations",
            True,
        ),
        "bus_stops_model_output": ("other_model_output", "Bus Stops", True),
        "tram_stations_model_output": ("other_model_output", "Tram Stations", True),
        "rapid_transport_stations_model_output": (
            "other_model_output",
            "Rapid Transport Stations",
            True,
        ),
        "ferry_terminals_model_output": ("other_model_output", "Ferry Terminals", True),
        "national_cycle_network_model_output": (
            "other_model_output",
            "National Cycle Network",
            True,
        ),
        "tram_network_model_output": ("other_model_output", "Tram Network", True),
        "rapid_transport_network_model_output": (
            "other_model_output",
            "Rapid Transport Network",
            True,
        ),
        "nexus_metro_links_model_output": ("other_model_output", "Nexus Metro Links", True),
        "nexus_metro_stations_model_output": (
            "other_model_output",
            "Nexus Metro Stations",
            True,
        ),
        "functional_rules_audit": ("audit", "Functional Rules", True),
        "layering_audit": ("audit", "Layering", True),
        "summary_audit": ("audit", "Summary", True),
        "extreme_weather_audit": ("functional_rules_audit", "Extreme Weather", True),
        "flooding_audit": ("functional_rules_audit", "Flooding", True),
        "ground_stability_audit": ("functional_rules_audit", "Ground Stability", True),
        "coastal_erosion_audit": ("functional_rules_audit", "Coastal Erosion", True),
        "extreme_heat_audit": ("extreme_weather_audit", "Extreme Heat Risk Index", True),
        "extreme_cold_audit": ("extreme_weather_audit", "Extreme Cold Risk Index", True),
        "storm_audit": ("extreme_weather_audit", "Storm Risk Index", True),
        "drought_audit": ("extreme_weather_audit", "Drought Risk Index", True),
        "extreme_weather_index_audit": (
            "extreme_weather_audit",
            "Extreme Weather Risk Index",
            True,
        ),
        "road_layering_audit": ("layering_audit", "Road", True),
        "rail_layering_audit": ("layering_audit", "Rail", True),
        "other_layering_audit": ("layering_audit", "Other", True),
        "bespoke_layering_audit": ("layering_audit", "Bespoke", True),
        "os_road_layering_audit": ("road_layering_audit", "OS Roads", True),
        "noham_layering_audit": ("road_layering_audit", "NoHAM", True),
        "model_roads_layering_audit": ("road_layering_audit", "Model Roads", True),
        "passenger_rail_layering_audit": ("rail_layering_audit", "Passenger Rail", True),
        "freight_rail_layering_audit": ("rail_layering_audit", "Freight Rail", True),
        "train_stations_layering_audit": ("other_layering_audit", "Train Stations", True),
        "airports_layering_audit": ("other_layering_audit", "Airports", True),
        "bus_coach_stations_layering_audit": (
            "other_layering_audit",
            "Bus and Coach Stations",
            True,
        ),
        "bus_stops_layering_audit": ("other_layering_audit", "Bus Stops", True),
        "tram_stations_layering_audit": ("other_layering_audit", "Tram Stations", True),
        "rapid_transport_stations_layering_audit": (
            "other_layering_audit",
            "Rapid Transport Stations",
            True,
        ),
        "ferry_terminals_layering_audit": ("other_layering_audit", "Ferry Terminals", True),
        "national_cycle_network_layering_audit": (
            "other_layering_audit",
            "National Cycle Network",
            True,
        ),
        "tram_network_layering_audit": ("other_layering_audit", "Tram Network", True),
        "rapid_transport_network_layering_audit": (
            "other_layering_audit",
            "Rapid Transport Network",
            True,
        ),
        "nexus_metro_links_layering_audit": (
            "bespoke_layering_audit",
            "Nexus Metro Links",
            True,
        ),
        "nexus_metro_stations_layering_audit": (
            "bespoke_layering_audit",
            "Nexus Metro Stations",
            True,
        ),
        "road_summary_audit": ("summary_audit", "Road", True),
        "rail_summary_audit": ("summary_audit", "Rail", True),
        "other_summary_audit": ("summary_audit", "Other", True),
        "bespoke_summary_audit": ("summary_audit", "Bespoke", True),
        "os_road_summary_audit": ("road_summary_audit", "OS Roads", True),
        "noham_summary_audit": ("road_summary_audit", "NoHAM", True),
        "model_road_summary_audit": ("road_summary_audit", "Model Roads", True),
        "passenger_rail_summary_audit": ("rail_summary_audit", "Passenger Rail", True),
        "freight_rail_summary_audit": ("rail_summary_audit", "Freight Rail", True),
        "national_cycle_network_summary_audit": (
            "other_summary_audit",
            "National Cycle Network",
            True,
        ),
        "tram_network_summary_audit": ("other_summary_audit", "Tram Network", True),
        "rapid_transport_network_summary_audit": (
            "other_summary_audit",
            "Rapid Transport Network",
            True,
        ),
        "nexus_metro_links_summary_audit": (
            "bespoke_summary_audit",
            "Nexus Metro Links",
            True,
        ),
    }

    def __getattr__(self, name: str) -> pathlib.Path:
        """Resolve configured path attributes lazily."""
        if name not in self._PATH_SPECS:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return self._resolve_path(name)

    def _resolve_path(self, name: str) -> pathlib.Path:
        if name in self._resolved_paths:
            return self._resolved_paths[name]

        parent_name, suffix, is_directory = self._PATH_SPECS[name]
        parent = getattr(self, parent_name)
        path = parent / suffix

        if is_directory:
            path.mkdir(parents=True, exist_ok=True)

        self._resolved_paths[name] = path
        return path

    def __dir__(self) -> list[str]:
        """Include dynamically resolved path names in dir()."""
        return sorted({*super().__dir__(), *self._PATH_SPECS})

    def flooding_type_scenario_model_input(
        self, flooding_type: FloodingTypes, scenario: Scenarios
    ) -> pathlib.Path:
        """Return flooding model input path based on flooding type and scenario."""
        return self.flooding_model_input / f"{flooding_type}_{scenario}.gpkg"

    def flooding_tile_grid_model_interim_output(self, tile_size: int) -> pathlib.Path:
        """Return the path to the flooding tile grid model interim output."""
        return self.model_interim_output / f"flooding_tiles_{tile_size}m.gpkg"

    def geosure_model_input(self, geohazard: GroundStabilityRiskCols | None) -> pathlib.Path:
        """Return GeoSure model input path based on the geohazard."""
        if geohazard is not None:
            return self.ground_stability_model_input / f"{geohazard}.gpkg"
        return self.ground_stability_model_input / "geosure.gpkg"

    def national_coastal_erosion_risk_mapping_model_input(self, year: str) -> pathlib.Path:
        """Return national coastal erosion risk mapping model input path."""
        return (
            self.coastal_erosion_model_input
            / f"national_coastal_erosion_risk_mapping_smp_{year}_70CC.gpkg"
        )


class OtherInput(ctk.BaseConfig):
    """Configuration for other raw input data.

    Attributes
    ----------
    stb_path : pathlib.Path
        Path to the boundary file of all STBs.
    ca_path : pathlib.Path
        Path to the boundary file of all CAs.
    boundary_path : pathlib.Path | None = None
        Path to the specific boundary file of the region which the model is running for.
    """

    stb_path: pathlib.Path
    ca_path: pathlib.Path
    boundary_path: pydantic.FilePath | None = None

    # Empty yaml values are loaded as strings, so need to convert to None
    @pydantic.field_validator("boundary_path", mode="before")
    @classmethod
    def _empty_to_none(
        cls, v: str | pydantic.FilePath | None
    ) -> str | pydantic.FilePath | None:
        if v == "":
            return None
        return v


class ModelRoadsEntry(ctk.BaseConfig):
    """Configuration for the transport model road network data.

    Attributes
    ----------
    nodes: pathlib.Path
        Path to the nodes CSV file.
    links: pathlib.Path
        Path to the links CSV file.
    shaped_links: pathlib.Path
        Path to the shaped links shapefile.
    """

    nodes: pathlib.Path
    links: pathlib.Path
    shaped_links: pathlib.Path


class Road(ctk.BaseConfig):
    """Configuration for road infrastructure data.

    Attributes
    ----------
    os_road : ZipFileEntry
        Configuration for the OS road zip file entry.
    os_road_lookup: NestedZipFileEntry
        Configuration for the nested OS road lookup zip file entry.
    mm_highways: ZipFileEntry
        Configuration for the MasterMap Highways Network zip file entry.
    noham: NoHAMEntry
        Configuration for the NoHAM road network data.
    model_roads: ModelRoadsEntry
        Configuration for the transport model road network data.
    """

    os_road: ZipFileEntry
    os_road_lookup: NestedZipFileEntry
    mm_highways: ZipFileEntry
    noham: pathlib.Path
    model_roads: ModelRoadsEntry


class Rail(ctk.BaseConfig):
    """Configuration for rail infrastructure data.

    Attributes
    ----------
    rail_links : ZipFileEntry
        Configuration for the rail links zip file entry.
    """

    rail_links: ZipFileEntry


class Other(ctk.BaseConfig):
    """Configuration for other infrastructure data.

    Attributes
    ----------
    bus_stops : pathlib.Path
        Path to the bus stops data.
    ncn_sustrans : pathlib.Path
        Path to the NCN Sustrans data.
    mastermap_sites: ZipFileEntry
        Configuration for the MasterMap Sites zip file entry.
    """

    bus_stops: pathlib.Path
    ncn_sustrans: pathlib.Path
    mastermap_sites: ZipFileEntry


class InfrastructureConfig(ctk.BaseConfig):
    """Configuration for infrastructure data.

    Attributes
    ----------
    road : Road
        Configuration for road infrastructure data.
    rail : Rail
        Configuration for rail infrastructure data.
    other : Other
        Configuration for other infrastructure data.
    """

    road: Road
    rail: Rail
    other: Other


class ExtremeWeather(ctk.BaseConfig):
    """Configuration for extreme weather data.

    Attributes
    ----------
    wind_spd_current : pathlib.Path
        Path to the current wind speed data.
    wind_spd_forecast : pathlib.Path
        Path to the forecast wind speed data.
    rain_days : ZipFileEntry
        Configuration for rain days zip file entry.
    extreme_summer_days : ZipFileEntry
        Configuration for extreme summer days zip file entry.
    frost_days : ZipFileEntry
        Configuration for frost days zip file entry.
    hot_days : ZipFileEntry
        Configuration for hot days zip file entry.
    icing_days : ZipFileEntry
        Configuration for icing days zip file entry.
    wdr_index : ZipFileEntry
        Configuration for WDR index zip file entry.
    drought_index : ZipFileEntry
        Configuration for drought index zip file entry.
    max_temp_summer : ZipFileEntry
        Configuration for max temperature summer zip file entry.
    precip_summer : ZipFileEntry
        Configuration for precipitation summer zip file entry.
    min_temp_winter : ZipFileEntry
        Configuration for min temperature winter zip file entry.
    precip_winter : ZipFileEntry
        Configuration for precipitation winter zip file entry.
    """

    wind_speed: dict[str, pathlib.Path]
    rain_days: pathlib.Path
    extreme_summer_days: pathlib.Path
    frost_days: pathlib.Path
    hot_days: pathlib.Path
    icing_days: pathlib.Path
    wdr_index: pathlib.Path
    drought_index: pathlib.Path
    max_temp_summer: pathlib.Path
    precip_summer: pathlib.Path
    min_temp_winter: pathlib.Path
    precip_winter: pathlib.Path


class PremiumGeoSureEntry(ctk.BaseConfig):
    """Configuration for the premium GeoSure data.

    Attributes
    ----------
    collapsible_deposits : pathlib.Path
        Configuration for collapsible deposits shapefile.
    compressible_ground : pathlib.Path
        Configuration for compressible ground shapefile.
    landslides : pathlib.Path
        Configuration for landslides shapefile.
    running_sands : pathlib.Path
        Configuration for running sands shapefile.
    shrink_swell : pathlib.Path
        Configuration for shrink swell shapefile.
    soluble_rocks : pathlib.Path
        Configuration for soluble rocks shapefile.
    """

    collapsible_deposits: pathlib.Path
    compressible_ground: pathlib.Path
    landslides: pathlib.Path
    running_sands: pathlib.Path
    shrink_swell: pathlib.Path
    soluble_rocks: pathlib.Path


class GeoSureEntry(ctk.BaseConfig):
    """Configuration for GeoSure data.

    Attributes
    ----------
    open: ZipFileEntry
       Configuration for the open GeoSure zip file entry.
    premium: PremiumGeoSureEntry
       Configuration for the premium GeoSure entry.
    """

    open: ZipFileEntry
    premium: PremiumGeoSureEntry


class GroundStability(ctk.BaseConfig):
    """Configuration for ground stability data.

    Attributes
    ----------
    geosure : GeoSure
        Configuration for GeoSure data.
    """

    geosure: GeoSureEntry


class Flooding(ctk.BaseConfig):
    """Configuration for flooding data.

    Attributes
    ----------
    rivers_sea : pathlib.Path
        Path to the rivers and sea flooding data.
    surface_water : pathlib.Path
        Path to the surface water flooding data.
    groundwater : pathlib.Path
        Path to the groundwater flooding data, if available.
    """

    rivers_sea: pathlib.Path
    surface_water: pathlib.Path
    groundwater: pathlib.Path | None = None

    # Empty yaml values are loaded as strings, so need to convert to None
    @pydantic.field_validator("groundwater", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str) -> str | None:
        if v == "":
            return None
        return v


class HazardsConfig(ctk.BaseConfig):
    """Configuration for hazard data.

    Attributes
    ----------
    coastal_erosion : ZipFileEntry
        Configuration for coastal erosion zip file entry.
    extreme_weather : ExtremeWeather
        Configuration for extreme weather data.
    flooding : Flooding
        Configuration for flooding data.
    ground_stability : GroundStability
        Configuration for ground stability data.
    """

    coastal_erosion: ZipFileEntry
    extreme_weather: ExtremeWeather
    flooding: Flooding
    ground_stability: GroundStability


class ModelRoadFlowsEntry(ctk.BaseConfig):
    """Configuration for the transport model road flows data.

    Attributes
    ----------
    link_flows: pathlib.Path
        Path to the link flows CSV file.
    ufs: pathlib.Path
        Path to the UFS CSV file.
    annualisation_factors: pathlib.Path
        Path to the annualisation factors CSV file.
    """

    link_flows: pathlib.Path
    ufs: pathlib.Path
    annualisation_factors: pathlib.Path


class DemandConfig(ctk.BaseConfig):
    """Configuration for demand data.

    Attributes
    ----------
    freight_demand : pathlib.Path
        Path to the freight demand data.
    noham_demand : pathlib.Path
        Path to the NoHAM demand data.
    noham_years: dict[str, int]
        Dictionary of years for NoHAM demand scenarios.
    model_road_flows : ModelRoadFlowsEntry
        Configuration for the transport model road flows data.
    nexus: dict[str, pathlib.Path]
        Dictionary of paths for nexus demand data.
    """

    freight_demand: pathlib.Path
    noham_demand: pathlib.Path
    noham_years: dict[str, int]
    model_road_flows: ModelRoadFlowsEntry
    nexus: dict[str, pathlib.Path]


class IncidentWeightConfig(ctk.BaseConfig):
    """Configuration for asset risk weighting incident weights.

    Attributes
    ----------
    incident_share : float
        Weight applied to incident share.
    minutes_share : float
        Weight applied to minutes share.
    cost_share : float
        Weight applied to cost share.
    """

    incident_share: float
    minutes_share: float
    cost_share: float


class AssetRiskWeightingConfig(ctk.BaseConfig):
    """Configuration for asset risk weighting workflow.

    Attributes
    ----------
    nwr_eastern_path : pathlib.Path
        Path to Network Rail Eastern weather data CSV.
    nwr_northwest_path : pathlib.Path
        Path to Network Rail North West weather data CSV.
    output_path : pathlib.Path
        Output directory for asset risk weighting results.
    incident_weights : IncidentWeightConfig
        Weights used to calculate rail hazard impacts.
    """

    nwr_eastern_path: pathlib.Path
    nwr_northwest_path: pathlib.Path
    output_path: pathlib.Path
    incident_weights: IncidentWeightConfig


class SwitchConfig(ctk.BaseConfig):
    """Configuration for model switches.

    Attributes
    ----------
    run_data_cleaning : bool
        Whether to run data cleaning.
    run_functional_rules : bool
        Whether to run functional rules.
    run_layering : bool
        Whether to run layering.
    all_roads : bool
        Whether to include all roads in the analysis.
    noham_roads : bool
        Whether to include NoHAM roads in the analysis.
    model_roads : bool
        Whether to include transport model roads in the analysis.
    passenger_rail : bool
        Whether to include passenger rail in the analysis.
    freight_rail : bool
        Whether to include freight rail in the analysis.
    airports : bool
        Whether to include airports in the analysis.
    bus_stops : bool
        Whether to include bus stops in the analysis.
    national_cycle_network : bool
        Whether to include the national cycle network in the analysis.
    train_stations : bool
        Whether to include train stations in the analysis.
    tram_stations : bool
        Whether to include tram stations in the analysis.
    rapid_transport_stations : bool
        Whether to include rapid transport stations in the analysis.
    ferry_terminals : bool
        Whether to include ferry terminals in the analysis.
    bus_coach_stations : bool
        Whether to include bus coach stations in the analysis.
    tram_network : bool
        Whether to include the tram network in the analysis.
    rapid_transport_network : bool
        Whether to include the rapid transport network in the analysis.
    bespoke: bool
        Whether to include bespoke infrastructure data in the analysis.
    extreme_weather : bool
        Whether to include extreme weather hazards in the analysis.
    flooding : bool
        Whether to include flooding hazards in the analysis.
    ground_stability : bool
        Whether to include ground stability hazards in the analysis.
    coastal_erosion : bool
        Whether to include coastal erosion hazards in the analysis.
    compute_flooding_overlay: bool
        Whether to compute the direct flooding overlay.
    flooding_index_raster: bool
        Whether to use the flooding index raster in the analysis.
    """

    run_data_cleaning: bool
    run_functional_rules: bool
    run_layering: bool

    all_roads: bool
    noham_roads: bool
    model_roads: bool
    passenger_rail: bool
    freight_rail: bool
    airports: bool
    bus_stops: bool
    national_cycle_network: bool
    train_stations: bool
    tram_stations: bool
    rapid_transport_stations: bool
    ferry_terminals: bool
    bus_coach_stations: bool
    tram_network: bool
    rapid_transport_network: bool

    bespoke: bool

    extreme_weather: bool
    flooding: bool
    ground_stability: bool
    coastal_erosion: bool

    compute_flooding_overlay: bool = False
    flooding_index_raster: bool = False
    ground_stability_premium: bool = False

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        if not any([self.run_data_cleaning, self.run_functional_rules, self.run_layering]):
            raise ValueError(
                "At least one of 'run_data_cleaning', 'run_functional_rules' "
                "or 'run_layering' must be True."
            )

        if not any(
            [
                self.all_roads,
                self.noham_roads,
                self.model_roads,
                self.passenger_rail,
                self.freight_rail,
                self.airports,
                self.bus_stops,
                self.national_cycle_network,
                self.train_stations,
                self.tram_stations,
                self.rapid_transport_stations,
                self.ferry_terminals,
                self.bus_coach_stations,
                self.tram_network,
                self.rapid_transport_network,
                self.bespoke,
            ]
        ):
            raise ValueError("At least one infrastructure switch must be True.")

        if not any(
            [self.extreme_weather, self.flooding, self.ground_stability, self.coastal_erosion]
        ):
            raise ValueError("At least one hazard switch must be True.")

        return self


class ParameterConfig(ctk.BaseConfig):
    """Configuration for model parameters.

    Attributes
    ----------
    stb : str | None = None
        Name of Sub-National Transport Body to run the tool for (this must exactly match the
        name in the boundary data). Should be empty if running for a CA rather than STB.
    ca : str | None = None
        Name of Combined Authority to run the tool for (this must exactly match the name in the
        boundary data). Should be empty if running for a STB rather than CA.
    user_classes : RoadUserClassConfig
        User class labels for NoHAM and model roads demand processing.
    """

    class RoadUserClassConfig(ctk.BaseConfig):
        """Configuration for road model user classes."""

        # Using pydantic.Field means every config instance gets it's own default list
        # rather than sharing a mutable default across all instances.
        noham: list[str] = pydantic.Field(
            default_factory=lambda: ["uc1", "uc2", "uc3", "uc4", "uc5"]
        )
        model_roads: list[str] = pydantic.Field(
            default_factory=lambda: ["uc1", "uc2", "uc3", "uc4", "uc5", "uc6", "uc7"]
        )

        @staticmethod
        def _normalise_user_class(user_class: str | int) -> str:
            value = str(user_class).strip().lower()
            if value.startswith("uc"):
                suffix = value[2:]
            else:
                suffix = value

            if not suffix.isdigit():
                raise ValueError(
                    f"Invalid user class '{user_class}'. Use values like 'uc1', 'uc2'."
                )
            return f"uc{int(suffix)}"

        @pydantic.field_validator("noham", "model_roads", mode="before")
        @classmethod
        def _validate_user_classes(cls, value: list[str | int]) -> list[str]:
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError("User class lists must be non-empty lists.")

            normalised = [cls._normalise_user_class(v) for v in value]
            if len(set(normalised)) != len(normalised):
                raise ValueError("User class lists must not contain duplicates.")
            return normalised

    stb: str | None = None
    ca: str | None = None
    user_classes: RoadUserClassConfig = pydantic.Field(default_factory=RoadUserClassConfig)

    # Empty yaml values are loaded as strings, so need to convert to None
    @pydantic.field_validator("stb", "ca", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str) -> str | None:
        if v == "":
            return None
        return v

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.stb is None) ^ (self.ca is None):
            raise ValueError("Exactly one of 'stb' or 'ca' must be provided, but not both.")
        return self


class MainExtremeWeatherWeightsConfig(ctk.BaseConfig):
    """Weights for extreme weather sub-hazards."""

    extreme_heat: float = 0.25
    extreme_cold: float = 0.25
    drought: float = 0.25
    storm: float = 0.25

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            ExtremeWeatherRiskCols.EXTREME_HEAT: self.extreme_heat,
            ExtremeWeatherRiskCols.EXTREME_COLD: self.extreme_cold,
            ExtremeWeatherRiskCols.DROUGHT: self.drought,
            ExtremeWeatherRiskCols.STORM: self.storm,
        }


class FloodingWeightsConfig(ctk.BaseConfig):
    """Weights for flooding sub-hazards."""

    rivers_sea: float = 0.5
    surface_water: float = 0.5

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            FloodingRiskCols.RIVERS_SEA: self.rivers_sea,
            FloodingRiskCols.SURFACE_WATER: self.surface_water,
        }


class GroundStabilityWeightsConfig(ctk.BaseConfig):
    """Weights for ground stability sub-hazards."""

    collapsible_deposits: float = 0.17
    compressible_ground: float = 0.17
    landslides: float = 0.17
    running_sand: float = 0.17
    shrink_swell: float = 0.16
    soluble_rocks: float = 0.16

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            GroundStabilityRiskCols.COLLAPSIBLE_DEPOSITS: self.collapsible_deposits,
            GroundStabilityRiskCols.COMPRESSIBLE_GROUND: self.compressible_ground,
            GroundStabilityRiskCols.LANDSLIDES: self.landslides,
            GroundStabilityRiskCols.RUNNING_SAND: self.running_sand,
            GroundStabilityRiskCols.SHRINK_SWELL: self.shrink_swell,
            GroundStabilityRiskCols.SOLUBLE_ROCKS: self.soluble_rocks,
        }


class MainHazardWeightsConfig(ctk.BaseConfig):
    """Weights for main hazard indices."""

    extreme_weather: MainExtremeWeatherWeightsConfig = pydantic.Field(
        default_factory=MainExtremeWeatherWeightsConfig
    )
    flooding: FloodingWeightsConfig = pydantic.Field(default_factory=FloodingWeightsConfig)
    ground_stability: GroundStabilityWeightsConfig = pydantic.Field(
        default_factory=GroundStabilityWeightsConfig
    )

    def for_main_hazard(self, main_hazard: MainHazardRiskCols) -> dict[RiskColumn, float]:
        """Return sub-hazard weights for a main hazard."""
        extreme_weather = cast("MainExtremeWeatherWeightsConfig", self.extreme_weather)  # pylint: disable=no-member
        flooding = cast("FloodingWeightsConfig", self.flooding)  # pylint: disable=no-member
        ground_stability = cast("GroundStabilityWeightsConfig", self.ground_stability)  # pylint: disable=no-member
        mapping: dict[MainHazardRiskCols, dict[RiskColumn, float]] = {
            MainHazardRiskCols.EXTREME_WEATHER: extreme_weather.as_mapping(),  # pylint: disable=no-member
            MainHazardRiskCols.FLOODING: flooding.as_mapping(),  # pylint: disable=no-member
            MainHazardRiskCols.GROUND_STABILITY: ground_stability.as_mapping(),  # pylint: disable=no-member
            MainHazardRiskCols.COASTAL_EROSION: {},
        }
        return mapping[main_hazard]


class ExtremeHeatWeightsConfig(ctk.BaseConfig):
    """Weights for the extreme heat index."""

    max_temp_summer: float = 0.5
    hot_summer_days: float = 0.25
    extreme_summer_days: float = 0.25

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            ExtremeHeatCols.MAX_TEMP_SUMMER: self.max_temp_summer,
            ExtremeHeatCols.HOT_SUMMER_DAYS: self.hot_summer_days,
            ExtremeHeatCols.EXTREME_SUMMER_DAYS: self.extreme_summer_days,
        }


class ExtremeColdWeightsConfig(ctk.BaseConfig):
    """Weights for the extreme cold index."""

    min_temp_winter: float = 0.5
    frost_days: float = 0.25
    icing_days: float = 0.25

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            ExtremeColdCols.MIN_TEMP_WINTER: self.min_temp_winter,
            ExtremeColdCols.FROST_DAYS: self.frost_days,
            ExtremeColdCols.ICING_DAYS: self.icing_days,
        }


class DroughtWeightsConfig(ctk.BaseConfig):
    """Weights for the drought index."""

    drought_severity_index: float = 0.75
    precip_summer: float = 0.25

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            DroughtCols.DROUGHT_SEVERITY_INDEX: self.drought_severity_index,
            DroughtCols.PRECIP_SUMMER: self.precip_summer,
        }


class StormWeightsConfig(ctk.BaseConfig):
    """Weights for the storm index."""

    wind_speed: float = 0.3
    exceedance_days: float = 0.2
    precip_winter: float = 0.15
    rain_days: float = 0.15
    wind_driven_rain_index: float = 0.2

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return weights keyed by risk columns."""
        return {
            StormCols.WIND_SPEED: self.wind_speed,
            StormCols.EXCEEDANCE_DAYS: self.exceedance_days,
            StormCols.PRECIP_WINTER: self.precip_winter,
            StormCols.RAIN_DAYS: self.rain_days,
            StormCols.WIND_DRIVEN_RAIN_INDEX: self.wind_driven_rain_index,
        }


class ExtremeWeatherSubHazardWeightsConfig(ctk.BaseConfig):
    """Weights for extreme weather sub-hazard indices."""

    extreme_heat: ExtremeHeatWeightsConfig = pydantic.Field(
        default_factory=ExtremeHeatWeightsConfig
    )
    extreme_cold: ExtremeColdWeightsConfig = pydantic.Field(
        default_factory=ExtremeColdWeightsConfig
    )
    drought: DroughtWeightsConfig = pydantic.Field(default_factory=DroughtWeightsConfig)
    storm: StormWeightsConfig = pydantic.Field(default_factory=StormWeightsConfig)

    def for_sub_hazard(self, sub_hazard: ExtremeWeatherRiskCols) -> dict[RiskColumn, float]:
        """Return variable weights for an extreme weather sub-hazard."""
        extreme_heat = cast("ExtremeHeatWeightsConfig", self.extreme_heat)  # pylint: disable=no-member
        extreme_cold = cast("ExtremeColdWeightsConfig", self.extreme_cold)  # pylint: disable=no-member
        drought = cast("DroughtWeightsConfig", self.drought)  # pylint: disable=no-member
        storm = cast("StormWeightsConfig", self.storm)  # pylint: disable=no-member
        mapping: dict[ExtremeWeatherRiskCols, dict[RiskColumn, float]] = {
            ExtremeWeatherRiskCols.EXTREME_HEAT: extreme_heat.as_mapping(),  # pylint: disable=no-member
            ExtremeWeatherRiskCols.EXTREME_COLD: extreme_cold.as_mapping(),  # pylint: disable=no-member
            ExtremeWeatherRiskCols.DROUGHT: drought.as_mapping(),  # pylint: disable=no-member
            ExtremeWeatherRiskCols.STORM: storm.as_mapping(),  # pylint: disable=no-member
        }
        return mapping[sub_hazard]


class RailExtremeWeatherWeightsConfig(ctk.BaseConfig):
    """Rail-specific override for extreme weather sub-hazard weights."""

    extreme_heat: float = 0.2
    extreme_cold: float = 0.21
    drought: float = 0.14
    storm: float = 0.45

    def as_mapping(self) -> dict[RiskColumn, float]:
        """Return rail-specific extreme weather weights keyed by risk columns."""
        return {
            ExtremeWeatherRiskCols.EXTREME_HEAT: self.extreme_heat,
            ExtremeWeatherRiskCols.EXTREME_COLD: self.extreme_cold,
            ExtremeWeatherRiskCols.DROUGHT: self.drought,
            ExtremeWeatherRiskCols.STORM: self.storm,
        }


class AssetHazardWeightOverridesConfig(ctk.BaseConfig):
    """Asset-specific hazard weight overrides."""

    rail_extreme_weather: RailExtremeWeatherWeightsConfig = pydantic.Field(
        default_factory=RailExtremeWeatherWeightsConfig
    )


class WeightConfig(ctk.BaseConfig):
    """Configuration for model hazard and index weights."""

    main_hazard: MainHazardWeightsConfig = pydantic.Field(
        default_factory=MainHazardWeightsConfig
    )
    extreme_weather_sub_hazard: ExtremeWeatherSubHazardWeightsConfig = pydantic.Field(
        default_factory=ExtremeWeatherSubHazardWeightsConfig
    )
    asset_hazard_overrides: AssetHazardWeightOverridesConfig = pydantic.Field(
        default_factory=AssetHazardWeightOverridesConfig
    )

    def main_hazard_weights(self, main_hazard: MainHazardRiskCols) -> dict[RiskColumn, float]:
        """Return sub-hazard weights for a main hazard."""
        configured_main_hazard = cast("MainHazardWeightsConfig", self.main_hazard)  # pylint: disable=no-member
        return configured_main_hazard.for_main_hazard(main_hazard)  # pylint: disable=no-member

    def extreme_weather_sub_hazard_weights(
        self, sub_hazard: ExtremeWeatherRiskCols
    ) -> dict[RiskColumn, float]:
        """Return variable weights for an extreme weather sub-hazard index."""
        configured_sub_hazards = cast(
            "ExtremeWeatherSubHazardWeightsConfig",
            self.extreme_weather_sub_hazard,  # pylint: disable=no-member
        )
        return configured_sub_hazards.for_sub_hazard(sub_hazard)  # pylint: disable=no-member

    def asset_hazard_weights(
        self, asset_type: AssetTypes, main_hazard: MainHazardRiskCols
    ) -> dict[RiskColumn, float]:
        """Return asset-specific hazard weights, falling back to default main hazard."""
        configured_overrides = cast(
            "AssetHazardWeightOverridesConfig",
            self.asset_hazard_overrides,  # pylint: disable=no-member
        )
        if asset_type == AssetTypes.RAIL and main_hazard == MainHazardRiskCols.EXTREME_WEATHER:
            return configured_overrides.rail_extreme_weather.as_mapping()  # pylint: disable=no-member
        return self.main_hazard_weights(main_hazard)


class ConstantConfig(ctk.BaseConfig):
    """Configuration for model constants.

    Attributes
    ----------
    noham_road_id_threshold : int
        Threshold for NoHAM road IDs.
    score_min : float
        Minimum score for risk calculations.
    score_max : float
        Maximum score for risk calculations.
    rapid_transport_stakeholders : list[str]
        List of rapid transport stakeholders defined in the MasterMap Site data.
    """

    noham_road_id_threshold: int
    score_min: int
    score_max: int
    rapid_transport_stakeholders: list[str]


# -------------------------
# MAIN CONFIG
# -------------------------


class Config(ctk.BaseConfig):
    """Main configuration for whole model.

    Attributes
    ----------
    switches : SwitchConfig
        Configuration for model switches.
    parameters: ParameterConfig
        Configuration for model parameters.
    paths : PathConfig
        Configuration for base paths.
    infrastructure : InfrastructureConfig
        Configuration for infrastructure data.
    hazards : HazardsConfig
        Configuration for hazards data.
    demand : DemandConfig
        Configuration for demand data.
    asset_risk_weighting : AssetRiskWeightingConfig | None
        Configuration for asset risk weighting workflow.
    weights: WeightConfig
        Configuration for hazard and index weights.
    constants: ConstantConfig
        Configuration for model constants.
    """

    switches: SwitchConfig
    parameters: ParameterConfig
    paths: PathConfig
    other_input: OtherInput
    infrastructure: InfrastructureConfig
    hazards: HazardsConfig
    demand: DemandConfig
    asset_risk_weighting: AssetRiskWeightingConfig | None = None
    weights: WeightConfig = pydantic.Field(default_factory=WeightConfig)
    constants: ConstantConfig
