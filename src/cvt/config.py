"""Module for setting up the model configuration."""

from pathlib import Path

import caf.toolkit as ctk

### CONFIG SET UP


class ZipFileEntry(ctk.BaseConfig):
    """Configuration for a file within a zip archive.

    Attributes
    ----------
    zip_path : Path
        Path to the zip file.
    internal_path : str | None
        Path within the zip file.
    output_path : Path | None
        Output path for the extracted file.
    """

    zip_path: Path
    internal_path: str | None = None
    output_path: Path | None = None


# -------------------------
# CATEGORY CONFIG CLASSES
# -------------------------


class PathConfig(ctk.BaseConfig):
    """Configuration for the base module paths.

    Attributes
    ----------
    root : Path
        Root directory for the project.
    raw_input : Path
        Directory for raw input data.
    model_input : Path
        Directory for model input data.
    model_interim_output : Path
        Directory for interim model output.
    model_output : Path
        Directory for final model output.
    log_path : Path
        Directory for log files.
    boundary_path : Path
        Directory for boundary files.
    """

    root: Path
    raw_input: Path
    model_input: Path
    model_interim_output: Path
    model_output: Path
    log_path: Path
    boundary_path: Path


class Road(ctk.BaseConfig):
    """Configuration for road infrastructure data.

    Attributes
    ----------
    os_road : Path
        Path to the OS road data.
    noham_2023 : Path
        Path to the NoHAM 2023 road data.
    noham_2048 : Path
        Path to the NoHAM 2048 road data.
    """

    os_road: Path
    noham_2023: Path
    noham_2048: Path


class Rail(ctk.BaseConfig):
    """Configuration for rail infrastructure data.

    Attributes
    ----------
    tfn_rail_links : Path
        Path to the TfN rail links data.
    """

    tfn_rail_links: Path


class BusStops(ctk.BaseConfig):
    """Configuration for bus stops data.

    Attributes
    ----------
    ne : Path
        Path to the North East bus stops data.
    nw : Path
        Path to the North West bus stops data.
    ys : Path
        Path to the Yorkshire bus stops data.
    """

    ne: Path
    nw: Path
    ys: Path


class Other(ctk.BaseConfig):
    """Configuration for other infrastructure data.

    Attributes
    ----------
    bus_stops : BusStops
        Configuration for bus stops data.
    ncn_sustrans : Path
        Path to the NCN Sustrans data.
    os_mmrn : Path
        Path to the OS MMRN data.
    poi_uk : ZipFileEntry
        Configuration for the POI UK zip file entry.
    zapmap : Path
        Path to the ZapMap data.
    """

    bus_stops: BusStops
    ncn_sustrans: Path
    os_mmrn: Path
    poi_uk: ZipFileEntry
    zapmap: Path


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
    zip_path : Path
        Path to the zip file.
    giz : str
        Internal path to the GIZ file.
    smp : dict
        Dictionary of SMP files.
    """

    zip_path: Path
    giz: str
    smp: dict


class ExtremeWeather(ctk.BaseConfig):
    """Configuration for extreme weather data.

    Attributes
    ----------
    wind_spd_current : Path
        Path to the current wind speed data.
    wind_spd_forecast : Path
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

    wind_spd_current: Path
    wind_spd_forecast: Path
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
    flood_path : Path
        Path to the flood data.
    """

    flood_path: Path


class GeoSure(ctk.BaseConfig):
    """Configuration for GeoSure data.

    Attributes
    ----------
    zip_path : Path
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

    zip_path: Path
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
    freight_demand : Path
        Path to the freight demand data.
    noham_demand : ZipFileEntry
        Configuration for NoHAM demand zip file entry.
    """

    freight_demand: Path
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
