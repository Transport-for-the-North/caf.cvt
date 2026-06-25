"""Main script."""

import argparse
import logging
import pathlib
import warnings

import caf.toolkit as ctk

from caf import cvt
from caf.cvt.data_cleaning import data_cleaning
from caf.cvt.functional_rules import apply_functional_rules
from caf.cvt.layering import layering
from caf.cvt.model_config import Config

LOG = logging.getLogger(__name__)

# This warning is raised in this model where geopandas overlays take place with large spatial
# datasets. Since the model is designed to handle large datasets, and overlays are expected to
# be complex and time-consuming, we can safely ignore this warning. If the processing time
# becomes unreasonably long, it may be worth investigating an alternative approach.
warnings.filterwarnings(
    "ignore",
    message=r".*organizePolygons\(\) received a polygon with more than 100 parts.*",
    category=RuntimeWarning,
)

# TODO (DJ): Add a comment to explain why this warning is being ignored.
# warnings.filterwarnings(
#    "ignore",
#    message="GeoSeries.notna",
#    category=UserWarning,
# )

# This warning is raised by xarray when reading the .nc wind speed files. It does not change
# the data at all, so we can safely ignore it. It is likely caused by a numpy version mismatch.
# Eventually this warning should be addressed, but for now it is safe to ignore.
warnings.filterwarnings(
    "ignore",
    message="numpy.ndarray size changed, may indicate binary incompatibility.*",
    category=RuntimeWarning,
)

def _main() -> None:
    """Run Climate Vulnerability Tool."""
    parser = argparse.ArgumentParser(
        __package__, description="CLI for the Climate Vulnerability Tool"
    )
    parser.add_argument(
        "-c", "--config", help="Config file to use", type=pathlib.Path, default="config.yml"
    )

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    details = ctk.log_helpers.ToolDetails(__package__, cvt.__version__, full_version=None)

    with ctk.LogHelper(__package__, details, log_file=config.paths.log_path / "cvt.log"):
        if config.switches.run_data_cleaning:
            LOG.info("Starting data cleaning step...")
            data_cleaning(config)
            LOG.info("Finished data cleaning step.")

        if config.switches.run_functional_rules:
            LOG.info("Starting functional rules step...")
            apply_functional_rules(config)
            LOG.info("Finished functional rules step.")

        if config.switches.run_layering:
            LOG.info("Starting layering step...")
            layering(config)
            LOG.info("Finished layering step.")


if __name__ == "__main__":
    _main()
