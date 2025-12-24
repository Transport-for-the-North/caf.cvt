"""
Main script, which runs the entire model
"""

from data_cleaning import data_cleaning
from functional_rules import apply_functional_rules
from layering import layering
from file_paths import RAW_INPUT, LOG_PATH

import logging
from caf.toolkit.log_helpers import LogHelper, ToolDetails

def main():
    """Run Climate Vulnerability Tool"""

    # MODEL PARAMETERS

    # Each script is self-contained, using files created in the previous stage, so they can be run independently
    run_data_cleaning = True
    run_functional_rules = False
    run_layering = False

    # Path to STB boundary
    boundary_path = RAW_INPUT / "Other" / "TfN Boundary" / "Transport_for_the_north_boundary_2020_generalised.shp"

    # Run data cleaning
    if run_data_cleaning:
        data_cleaning(boundary_path)

    # Run functional rules
    if run_functional_rules:
        apply_functional_rules(boundary_path)

    # Run layering
    if run_layering:
        layering()


if __name__ == "__main__":
    log = logging.getLogger('__main__')
    log.setLevel(logging.DEBUG)
    details = ToolDetails("cvt", "1.0.0", full_version=None)
    with LogHelper(__package__, details, log_file=LOG_PATH):
        main()