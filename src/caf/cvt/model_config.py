"""Module for setting up the model configuration."""

import pathlib
from typing import Annotated, Self

import caf.toolkit as ctk
import pydantic

### FUNCTIONS


def _check_none(value: str) -> str | None:
    if value is None:
        return value
    value = value.strip()
    if value in ("", "null"):
        return None
    return value


### CONFIG SET UP


class ZipFileEntry(ctk.BaseConfig):
    """Configuration for a file within a zip archive."""

    zip_path: pathlib.Path
    """Path to the zip file"""
    file_path: Annotated[str | None, pydantic.BeforeValidator(_check_none)] = None
    """Path within the zip file."""
    output_path: Annotated[pathlib.Path | None, pydantic.BeforeValidator(_check_none)] = None
    """Output path for the extracted file"""


# -------------------------
# CATEGORY CONFIG CLASSES
# -------------------------


class PathConfig(ctk.BaseConfig):
    """Configuration for the base module paths.

    Attributes
    ----------
    root : pathlib.Path
        Root directory for the project.
    raw_input : pathlib.Path
        Directory for raw input data.
    """

    root: pathlib.Path
    raw_input: pathlib.Path

    @property
    def model_input(self) -> pathlib.Path:
        """Create model input directory and return path."""
        model_input = self.root / "model input"
        model_input.mkdir(parents=True, exist_ok=True)
        return model_input

    @property
    def model_interim_output(self) -> pathlib.Path:
        """Create model interim output directory and return path."""
        model_interim_output = self.root / "model interim outputs"
        model_interim_output.mkdir(parents=True, exist_ok=True)
        return model_interim_output

    @property
    def model_output(self) -> pathlib.Path:
        """Create model output directory and return path."""
        model_output = self.root / "model outputs"
        model_output.mkdir(parents=True, exist_ok=True)
        return model_output

    @property
    def log_path(self) -> pathlib.Path:
        """Create logging directory and return path."""
        log_path = self.root / "Logging"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    @property
    def audit_path(self) -> pathlib.Path:
        """Create model audit directory and return path."""
        audit_path = self.root / "audit"
        audit_path.mkdir(parents=True, exist_ok=True)
        return audit_path


class OtherInput(ctk.BaseConfig):
    """Configuration for other raw input data.

    Attributes
    ----------
    boundary_path : pathlib.Path | None = None
        Path to the specific boundary file of the region which the model is running for.
    stb_path : pathlib.Path
        Path to the boundary file of all STBs.
    ca_path : pathlib.Path
        Path to the boundary file of all CAs.
    """

    boundary_path: Annotated[pathlib.Path | None, pydantic.BeforeValidator(_check_none)] = None
    stb_path: pathlib.Path
    ca_path: pathlib.Path


class NoHAMEntry(ctk.BaseConfig):
    """Configuration for NoHAM road network data.

    Attributes
    ----------
    year: int
        Year of the NoHAM network.
    file_path: pathlib.Path
    """

    year: int
    file_path: pathlib.Path


class Road(ctk.BaseConfig):
    """Configuration for road infrastructure data.

    Attributes
    ----------
    os_road : ZipFileEntry
        Configuration for the OS road zip file entry.
    noham: NoHAMEntry
        Configuration for the NoHAM road network data.
    """

    os_road: ZipFileEntry
    noham: NoHAMEntry


class Rail(ctk.BaseConfig):
    """Configuration for rail infrastructure data.

    Attributes
    ----------
    rail_links : pathlib.Path
        Path to the rail links data.
    """

    rail_links: pathlib.Path


class Other(ctk.BaseConfig):
    """Configuration for other infrastructure data.

    Attributes
    ----------
    bus_stops : dict[str, pathlib.Path]
        Mapping of region name to bus stops data path.
    ncn_sustrans : pathlib.Path
        Path to the NCN Sustrans data.
    os_mmrn : pathlib.Path
        Path to the OS MMRN data.
    poi_uk : ZipFileEntry
        Configuration for the POI UK zip file entry.
    zapmap : pathlib.Path
        Path to the ZapMap data.
    airports: pathlib.path
        Path to the airports data.
    """

    bus_stops: dict[str, pathlib.Path]
    ncn_sustrans: pathlib.Path
    os_mmrn: pathlib.Path
    poi_uk: ZipFileEntry
    zapmap: pathlib.Path
    airports: pathlib.Path


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


class Flooding(ctk.BaseConfig):
    """Configuration for flooding data.

    Attributes
    ----------
    flooding_path : pathlib.Path
        Path to the flooding data.
    """

    flooding_path: pathlib.Path


class GroundStability(ctk.BaseConfig):
    """Configuration for ground stability data.

    Attributes
    ----------
    geo_shrink_swell : dict
        Dictionary of GeoShrinkSwell data.
    geosure : GeoSure
        Configuration for GeoSure data.
    """

    geo_shrink_swell: dict[str, pathlib.Path]
    geosure: ZipFileEntry


class HazardsConfig(ctk.BaseConfig):
    """Configuration for hazard data.

    Attributes
    ----------
    coastal_erosion : ZipFileEntry
        Configuration for coastal erosion zip file entry.
    extreme_weather : ExtremeWeather
        Configuration for extreme weather data.
    flooding : dict[str, pathlib.Path]
        Configuration for flooding data.
    ground_stability : GroundStability
        Configuration for ground stability data.
    """

    coastal_erosion: ZipFileEntry
    extreme_weather: ExtremeWeather
    flooding: dict[str, pathlib.Path]
    ground_stability: GroundStability


class ImpactConfig(ctk.BaseConfig):
    """Configuration for impact data.

    Attributes
    ----------
    freight_demand : pathlib.Path
        Path to the freight demand data.
    noham_demand : ZipFileEntry
        Configuration for NoHAM demand zip file entry.
    noham_years: dict[str, int]
        Dictionary of years for NoHAM demand scenarios.
    """

    freight_demand: pathlib.Path
    noham_demand: ZipFileEntry
    noham_years: dict[str, int]


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
    passenger_rail : bool
        Whether to include passenger rail in the analysis.
    freight_rail : bool
        Whether to include freight rail in the analysis.
    airports : bool
        Whether to include airports in the analysis.
    bus_stops : bool
        Whether to include bus stops in the analysis.
    petrol_stations : bool
        Whether to include petrol stations in the analysis.
    charging_sites : bool
        Whether to include EV charging sites in the analysis.
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
    noham_zip_extract : bool
        Whether to extract NoHAM zip files.
    """

    run_data_cleaning: bool
    run_functional_rules: bool
    run_layering: bool

    all_roads: bool
    noham_roads: bool
    passenger_rail: bool
    freight_rail: bool
    airports: bool
    bus_stops: bool
    petrol_stations: bool
    charging_sites: bool
    national_cycle_network: bool
    train_stations: bool
    tram_stations: bool
    rapid_transport_stations: bool
    ferry_terminals: bool
    bus_coach_stations: bool
    tram_network: bool
    rapid_transport_network: bool

    extreme_weather: bool
    flooding: bool
    ground_stability: bool
    coastal_erosion: bool

    compute_flooding_overlay: bool = False

    noham_zip_extract: bool = False

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
                self.passenger_rail,
                self.freight_rail,
                self.airports,
                self.bus_stops,
                self.petrol_stations,
                self.charging_sites,
                self.national_cycle_network,
                self.train_stations,
                self.tram_stations,
                self.rapid_transport_stations,
                self.ferry_terminals,
                self.bus_coach_stations,
                self.tram_network,
                self.rapid_transport_network,
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
    """

    stb: Annotated[str | None, pydantic.BeforeValidator(_check_none)] = None
    ca: Annotated[str | None, pydantic.BeforeValidator(_check_none)] = None

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.stb is None) ^ (self.ca is None):
            raise ValueError("Exactly one of 'stb' or 'ca' must be provided, but not both.")
        return self


class ConstantConfig(ctk.BaseConfig):
    """Configuration for model constants.

    Attributes
    ----------
    noham_road_id_threshold : int
        Threshold for NoHAM road IDs.
    """

    noham_road_id_threshold: int


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
    impact : ImpactConfig
        Configuration for impact data.
    constants: ConstantConfig
        Configuration for model constants.
    """

    switches: SwitchConfig
    parameters: ParameterConfig
    paths: PathConfig
    other_input: OtherInput
    infrastructure: InfrastructureConfig
    hazards: HazardsConfig
    impact: ImpactConfig
    constants: ConstantConfig
