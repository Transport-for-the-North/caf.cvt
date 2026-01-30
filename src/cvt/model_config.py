"""Module for setting up the model configuration."""

import pathlib

import caf.toolkit as ctk

### CONFIG SET UP


class ZipFileEntry(ctk.BaseConfig):
    """Configuration for a file within a zip archive."""

    zip_path: pathlib.Path
    """Path to the zip file"""
    file_path_path: str | None = None
    """Path within the zip file."""
    output_path: pathlib.Path | None = None
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
    boundary_path : pathlib.Path
        Path to the STB boundary which the model is running for.
    """

    boundary_path: pathlib.Path


class NoHAMEntry:
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
    os_road : pathlib.Path
        Path to the OS road data.
    noham: dict[str, NoHAMEntry]
        Dictionary containing scenario linked to year and file path in a NoHAM entry class.
    """

    os_road: pathlib.Path
    noham: dict[str, NoHAMEntry]


class Rail(ctk.BaseConfig):
    """Configuration for rail infrastructure data.

    Attributes
    ----------
    tfn_rail_links : pathlib.Path
        Path to the TfN rail links data.
    """

    tfn_rail_links: pathlib.Path


class Other(ctk.BaseConfig):
    """Configuration for other infrastructure data.

    Attributes
    ----------
    bus_stops : dict[str, pathlib.Path]
        Mapping of region name to bus stips data path.
    ncn_sustrans : pathlib.Path
        Path to the NCN Sustrans data.
    os_mmrn : pathlib.Path
        Path to the OS MMRN data.
    poi_uk : ZipFileEntry
        Configuration for the POI UK zip file entry.
    zapmap : pathlib.Path
        Path to the ZapMap data.
    """

    bus_stops: dict[str, pathlib.Path]
    ncn_sustrans: pathlib.Path
    os_mmrn: pathlib.Path
    poi_uk: ZipFileEntry
    zapmap: pathlib.Path


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
    flood_zip_extract : bool
        Whether to extract flood zip files.
    noham_zip_extract : bool
        Whether to extract NoHAM zip files.
    create_flood_grid : bool
        Whether to create flood grid.
    """

    run_data_cleaning: bool = False
    run_functional_rules: bool = False
    run_layering: bool = False
    flood_zip_extract: bool = False
    noham_zip_extract: bool = False
    create_flood_grid: bool = False


# -------------------------
# MAIN CONFIG
# -------------------------


class Config(ctk.BaseConfig):
    """Main configuration for whole model.

    Attributes
    ----------
    switches : SwitchConfig
        Configuration for model switches.
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
    paths: PathConfig
    infrastructure: InfrastructureConfig
    hazards: HazardsConfig
    impact: ImpactConfig
