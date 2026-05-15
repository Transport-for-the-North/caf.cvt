"""Module for setting up the model configuration."""

import pathlib
from typing import Annotated, Self

import caf.toolkit as ctk
import pydantic

### FUNCTIONS

def _check_none(value: str) -> str | None:
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
    """Configuration for a NoHAM data entry.

    Attributes
    ----------
    year: str
        The year which the NoHAM entry comes from.
    file_path: pathlib.Path
        Path to the NoHAM data.
    """

    year: str
    file_path: pathlib.Path


class Road(ctk.BaseConfig):
    """Configuration for road infrastructure data.

    Attributes
    ----------
    os_road : ZipFileEntry
        Configuration for the OS road zip file entry.
    noham: dict[str, NoHAMEntry]
        Dictionary containing scenario linked to year and file path in a NoHAM entry class.
    """

    os_road: ZipFileEntry
    noham: dict[str, NoHAMEntry]


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


class CoastalErosion(ctk.BaseConfig):
    """Configuration for coastal erosion data.

    Attributes
    ----------
    zip_path : pathlib.Path
        Path to the zip file.
    giz : str
        Internal path to the GIZ file.
    smp : dict
        Dictionary of SMP files.
    """

    zip_path: pathlib.Path
    giz: str
    smp: dict


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

    wind_spd_current: pathlib.Path
    wind_spd_forecast: pathlib.Path
    rain_days: ZipFileEntry
    extreme_summer_days: ZipFileEntry
    frost_days: ZipFileEntry
    hot_days: ZipFileEntry
    icing_days: ZipFileEntry
    wdr_index: ZipFileEntry
    drought_index: ZipFileEntry
    max_temp_summer: ZipFileEntry
    precip_summer: ZipFileEntry
    min_temp_winter: ZipFileEntry
    precip_winter: ZipFileEntry


class Flooding(ctk.BaseConfig):
    """Configuration for flooding data.

    Attributes
    ----------
    flood_path : pathlib.Path
        Path to the flood data.
    """

    flood_path: pathlib.Path


class GeoSure(ctk.BaseConfig):
    """Configuration for GeoSure data.

    Attributes
    ----------
    zip_path : pathlib.Path
        Path to the GeoSure zip file.
    collapsible_deposits : str
        Internal path to the collapsible deposits file.
    compressible_ground : str
        Internal path to the compressible ground file.
    landslides : str
        Internal path to the landslides file.
    running_sand : str
        Internal path to the running sand file.
    shrink_swell : str
        Internal path to the shrink swell file.
    soluble_rocks : str
        Internal path to the soluble rocks file.
    """

    zip_path: pathlib.Path
    collapsible_deposits: str
    compressible_ground: str
    landslides: str
    running_sand: str
    shrink_swell: str
    soluble_rocks: str


class GroundStability(ctk.BaseConfig):
    """Configuration for ground stability data.

    Attributes
    ----------
    geo_shrink_swell : dict
        Dictionary of GeoShrinkSwell data.
    geosure : GeoSure
        Configuration for GeoSure data.
    """

    geo_shrink_swell: dict
    geosure: GeoSure


class HazardsConfig(ctk.BaseConfig):
    """Configuration for hazard data.

    Attributes
    ----------
    coastal_erosion : CoastalErosion
        Configuration for coastal erosion data.
    extreme_weather : ExtremeWeather
        Configuration for extreme weather data.
    flooding : Flooding
        Configuration for flooding data.
    ground_stability : GroundStability
        Configuration for ground stability data.
    """

    coastal_erosion: CoastalErosion
    extreme_weather: ExtremeWeather
    flooding: Flooding
    ground_stability: GroundStability


class ImpactConfig(ctk.BaseConfig):
    """Configuration for impact data.

    Attributes
    ----------
    freight_demand : pathlib.Path
        Path to the freight demand data.
    noham_demand : ZipFileEntry
        Configuration for NoHAM demand zip file entry.
    """

    freight_demand: pathlib.Path
    noham_demand: ZipFileEntry


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
    flood_zip_extract : bool
        Whether to extract flood zip files.
    create_flood_grid : bool
        Whether to create flood grid.
    compute_flood_overlay: bool
        Whether to compute the direct flood overlay.
    flood_overlay_direct: bool
        Whether to use a direct overlay of flood data.
    create_flood_tiles: bool
        Whether to create flood tiles.
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

    flood_zip_extract: bool = False
    create_flood_grid: bool = False

    compute_flood_overlay: bool = False
    flood_overlay_direct: bool = False
    create_flood_tiles: bool = False

    noham_zip_extract: bool = False

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
        if not ((self.stb is None) ^ (self.ca is None)):
            raise ValueError("Exactly one of 'stb' or 'ca' must be provided, but not both.")
        return self


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
    """

    switches: SwitchConfig
    parameters: ParameterConfig
    paths: PathConfig
    other_input: OtherInput
    infrastructure: InfrastructureConfig
    hazards: HazardsConfig
    impact: ImpactConfig
