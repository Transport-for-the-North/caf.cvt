"""Main script."""

import argparse
import logging
import pathlib

import caf.toolkit as ctk
from data_cleaning import data_cleaning
from functional_rules import apply_functional_rules
from layering import layering
from model_config import Config

LOG = logging.getLogger(__name__)


def _main() -> None:
    """Run Climate Vulnerability Tool."""
    parser = argparse.ArgumentParser(
        __package__, description="CLI for the Climate Vulnerability Tool"
    )
    parser.add_argument("-c", "--config", help="Config file to use", type=pathlib.Path, default="config.yml")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    details = ctk.log_helpers.ToolDetails(__name__, "1.0.0", full_version=None)

    with ctk.LogHelper(__name__, details, log_file=config.paths.log_path / "cvt.log"):
        if config.switches.run_data_cleaning:
            LOG.info("Starting data cleaning step...")
            data_cleaning(config)
            LOG.info("Finished data cleaning step.")

        if config.switches.run_functional_rules:
            LOG.info("Starting functional rules step...")
            apply_functional_rules(config)
            LOG.info("Finished data cleaning step.")

        if config.switches.run_layering:
            LOG.info("Starting layering step...")
            layering(config)
            LOG.info("Finished layering step.")


if __name__ == "__main__":
    _main()
