"""Main script."""

import logging
from pathlib import Path

import caf.toolkit as ctk
from config import Config
from data_cleaning import data_cleaning
from functional_rules import apply_functional_rules
from layering import layering

LOG = logging.getLogger(__name__)


def _main() -> None:
    """Run Climate Vulnerability Tool."""
    current_dir = Path(__file__).parent
    config_path = current_dir.parents[1] / "config.yml"
    cfg = Config.load_yaml(config_path)
    details = ctk.log_helpers.ToolDetails(__name__, "1.0.0", full_version=None)

    with ctk.LogHelper(__name__, details, log_file=cfg.paths.log_path):
        # Run data cleaning
        if cfg.switches.run_data_cleaning:
            data_cleaning(cfg)

        # Run functional rules
        if cfg.switches.run_functional_rules:
            apply_functional_rules(cfg)

        # Run layering
        if cfg.switches.run_layering:
            layering(cfg)


# Run model
if __name__ == "__main__":
    _main()
