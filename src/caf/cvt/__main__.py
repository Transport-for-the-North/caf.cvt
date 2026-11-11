"""Main script."""

import argparse
import logging
import pathlib

import caf.toolkit as ctk

from caf import cvt
from caf.cvt.asset_risk_weighting import run_asset_risk_weighting
from caf.cvt.data_cleaning import data_cleaning
from caf.cvt.functional_rules import apply_functional_rules
from caf.cvt.layering import layering
from caf.cvt.model_config import Config

LOG = logging.getLogger(__name__)


def _run_model(args: argparse.Namespace) -> None:
    """Run the main Climate Vulnerability Tool workflow."""
    config = Config.load_yaml(args.config)
    details = ctk.log_helpers.ToolDetails(__package__, cvt.__version__, full_version=None)

    with ctk.LogHelper(__package__, details, log_file=config.paths.log / "cvt.log"):
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


def _run_asset_risk_weighting(args: argparse.Namespace) -> None:
    """Run the asset risk weighting workflow."""
    config = Config.load_yaml(args.config)
    if config.asset_risk_weighting is None:
        raise ValueError(
            "Missing 'asset_risk_weighting' section in config file. "
            "Add paths and incident_weights before running this subcommand."
        )

    run_asset_risk_weighting(
        nwr_eastern_path=config.asset_risk_weighting.nwr_eastern_path,
        nwr_northwest_path=config.asset_risk_weighting.nwr_northwest_path,
        output_path=config.asset_risk_weighting.output_path,
        incident_weights=config.asset_risk_weighting.incident_weights,
    )


def _main() -> None:
    """Run Climate Vulnerability Tool CLI."""
    parser = argparse.ArgumentParser(
        __package__, description="CLI for the Climate Vulnerability Tool"
    )
    parser.add_argument(
        "-c", "--config", help="Config file to use", type=pathlib.Path, default="config.yml"
    )
    parser.set_defaults(func=_run_model)

    subparsers = parser.add_subparsers(dest="command")
    asset_weighting_parser = subparsers.add_parser(
        "asset-risk-weighting",
        help="Derive hazard impact weights for rail assets from incident data",
    )
    asset_weighting_parser.set_defaults(func=_run_asset_risk_weighting)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
