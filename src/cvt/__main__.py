"""
Main script, which runs the entire model
"""

from data_cleaning import data_cleaning
from functional_rules import apply_functional_rules
from layering import layering
from file_paths import RAW_INPUT, LOG_PATH

import logging
from pathlib import Path
from caf.toolkit.log_helpers import LogHelper, ToolDetails
import caf.toolkit as ctk

def main(conf):
    """Run Climate Vulnerability Tool"""

    # Run data cleaning
    if conf.run_data_cleaning:
        data_cleaning(conf.boundary_path)

    # Run functional rules
    if conf.run_functional_rules:
        apply_functional_rules(conf.boundary_path)

    # Run layering
    if conf.run_layering:
        layering()

# Set up config
class Config(ctk.BaseConfig):
    run_data_cleaning: bool = True
    run_functional_rules: bool = True
    run_layering: bool = True

    boundary_path: Path

config = Config.load_yaml(Path("config.yml"))


# Run model, use logging
if __name__ == "__main__":
    log = logging.getLogger('__main__')
    log.setLevel(logging.DEBUG)
    details = ToolDetails("cvt", "1.0.0", full_version=None)
    with LogHelper(__package__, details, log_file=LOG_PATH):
        main(config)