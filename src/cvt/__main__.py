"""
Main script, which runs the entire model
"""

from data_cleaning import data_cleaning
from functional_rules import apply_functional_rules
from layering import layering
from file_paths import LOG_PATH

import logging
from pathlib import Path
from caf.toolkit.log_helpers import LogHelper, ToolDetails
import caf.toolkit as ctk

def main():
    """Run Climate Vulnerability Tool"""

    # Run data cleaning
    if cfg.run_data_cleaning:
        data_cleaning()

    # Run functional rules
    if cfg.run_functional_rules:
        apply_functional_rules()

    # Run layering
    if cfg.run_layering:
        layering()

### CONFIG SET UP

# --- simple zip + internal path structure ---
class ZipFileEntry(ctk.BaseConfig):
    zip_path: Path
    internal_path: str

# ---- main config ----
class Config(ctk.BaseConfig):
    # switches
    run_data_cleaning: bool
    run_functional_rules: bool
    run_layering: bool
    flood_extract: bool
    noham_extract: bool

    # basic paths
    root: Path
    raw_input: Path
    model_input: Path
    model_interim_output: Path
    model_output: Path
    log_path: Path
    boundary_path: Path

    # infra
    os_road: Path
    noham_2023: Path
    noham_2048: Path
    tfn_rail_links: Path

    bus_stops_ne: Path
    bus_stops_nw: Path
    bus_stops_ys: Path
    ncn_sustrans: Path
    os_mmrn: Path
    poi_uk: ZipFileEntry
    zapmap: Path

    # coastal erosion
    ce_zip_path: Path
    giz: str
    smp: dict

    # extreme weather
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

    # flooding
    flood_path: Path

    # ground stability
    geo_shrink_swell: dict
    geosure: dict

    # impact
    freight_demand: Path
    noham_demand: dict

config_path = Path("../../config.yml")
cfg = Config.load_yaml(config_path)


# Run model, use logging
if __name__ == "__main__":
    log = logging.getLogger('__main__')
    log.setLevel(logging.DEBUG)
    details = ToolDetails("cvt", "1.0.0", full_version=None)
    with LogHelper(__package__, details, log_file=LOG_PATH):
        main()