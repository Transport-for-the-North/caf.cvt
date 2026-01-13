import caf.toolkit as ctk
from pathlib import Path

### CONFIG SET UP


class ZipFileEntry(ctk.BaseConfig):
    zip_path: Path
    internal_path: str | None = None
    output_path: Path | None = None


# -------------------------
# CATEGORY CONFIG CLASSES
# -------------------------


class PathConfig(ctk.BaseConfig):
    root: Path
    raw_input: Path
    model_input: Path
    model_interim_output: Path
    model_output: Path
    log_path: Path
    boundary_path: Path


class Road(ctk.BaseConfig):
    os_road: Path
    noham_2023: Path
    noham_2048: Path


class Rail(ctk.BaseConfig):
    tfn_rail_links: Path


class BusStops(ctk.BaseConfig):
    ne: Path
    nw: Path
    ys: Path


class Other(ctk.BaseConfig):
    bus_stops: BusStops
    ncn_sustrans: Path
    os_mmrn: Path
    poi_uk: ZipFileEntry
    zapmap: Path


class InfrastructureConfig(ctk.BaseConfig):
    road: Road
    rail: Rail
    other: Other


class CoastalErosion(ctk.BaseConfig):
    zip_path: Path
    giz: str
    smp: dict


class ExtremeWeather(ctk.BaseConfig):
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
    flood_path: Path


class GeoSure(ctk.BaseConfig):
    zip_path: Path
    collapsible_deposits: str
    compressible_ground: str
    landslides: str
    running_sand: str
    shrink_swell: str
    soluble_rocks: str


class GroundStability(ctk.BaseConfig):
    geo_shrink_swell: dict
    geosure: GeoSure


class HazardsConfig(ctk.BaseConfig):
    coastal_erosion: CoastalErosion
    extreme_weather: ExtremeWeather
    flooding: Flooding
    ground_stability: GroundStability


class ImpactConfig(ctk.BaseConfig):
    freight_demand: Path
    noham_demand: ZipFileEntry


class SwitchConfig(ctk.BaseConfig):
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
    switches: SwitchConfig
    paths: PathConfig
    infrastructure: InfrastructureConfig
    hazards: HazardsConfig
    impact: ImpactConfig
