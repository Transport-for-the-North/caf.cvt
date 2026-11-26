"""Asset-specific hazard weighting workflows."""

import logging
import pathlib

import caf.toolkit as ctk
import pandas as pd

from caf import cvt
from caf.cvt.model_config import IncidentWeightConfig

LOG = logging.getLogger(__name__)


def derive_rail_weights(
    nwr_eastern_path: pathlib.Path,
    nwr_northwest_path: pathlib.Path,
    incident_weights: IncidentWeightConfig,
    output_path: pathlib.Path,
) -> None:
    """Derive asset-specific hazard weights for rail assets."""
    rail_incidents = read_clean_nwr_data(nwr_eastern_path, nwr_northwest_path)

    incident_summary = calculate_incident_summary(rail_incidents, incident_weights)

    LOG.info("Writing incident summary to CSV...")
    incident_summary.to_csv(output_path / "rail_incident_summary.csv", index=False)

    consequence_weights = aggregate_consequence_weights(incident_summary)

    LOG.info("Writing consequence weights to CSV...")
    consequence_weights.to_csv(output_path / "rail_consequence_weights.csv", index=False)


def read_clean_nwr_data(
    nwr_eastern_path: pathlib.Path, nwr_northwest_path: pathlib.Path
) -> pd.DataFrame:
    """Read and clean Network Rail weather data."""
    LOG.info("Reading and cleaning Network Rail weather data...")
    nwr_eastern = pd.read_csv(nwr_eastern_path)
    nwr_northwest = pd.read_csv(nwr_northwest_path)
    nwr_incidents = pd.concat([nwr_eastern, nwr_northwest], ignore_index=True)
    rail_incidents = nwr_incidents[
        nwr_incidents["Weather Category"].isin(["Wind", "Heat", "Snow", "Cold", "Subsidence"])
    ][["Weather Category", "Minutes", "cost"]]
    LOG.info("Finished reading and cleaning Network Rail weather data.")
    return rail_incidents


def calculate_incident_summary(
    rail_incidents: pd.DataFrame, incident_weights: IncidentWeightConfig
) -> pd.DataFrame:
    """Calculate consequence weights for rail data based on incident data."""
    LOG.info("Calculating incident summary for rail data...")
    incident_summary = (
        rail_incidents.groupby("Weather Category")
        .agg(
            incidents=pd.NamedAgg(column="Weather Category", aggfunc="count"),
            total_minutes=pd.NamedAgg(column="Minutes", aggfunc="sum"),
            total_cost=pd.NamedAgg(column="cost", aggfunc="sum"),
        )
        .reset_index()
        .round({"total_minutes": 0, "total_cost": 0})
    )

    incident_summary["incident_share"] = (
        incident_summary["incidents"] / incident_summary["incidents"].sum()
    )
    incident_summary["minutes_share"] = (
        incident_summary["total_minutes"] / incident_summary["total_minutes"].sum()
    )
    incident_summary["cost_share"] = (
        incident_summary["total_cost"] / incident_summary["total_cost"].sum()
    )

    incident_summary["consequence"] = round(
        incident_summary["incident_share"] * incident_weights.incident_share
        + incident_summary["minutes_share"] * incident_weights.minutes_share
        + incident_summary["cost_share"] * incident_weights.cost_share,
        2,
    )

    return incident_summary


def aggregate_consequence_weights(incident_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consequence weights for rail data based on incident summary."""
    LOG.info("Aggregating consequence weights for rail data...")
    consequence_weights = incident_summary[["Weather Category", "consequence"]].rename(
        columns={"Weather Category": "hazard", "consequence": "consequence_weight"}
    )

    consequence_weights["hazard"] = consequence_weights["hazard"].map(
        {
            "Wind": "Storm",
            "Heat": "Extreme Heat",
            "Snow": "Extreme Cold",
            "Cold": "Extreme Cold",
            "Subsidence": "Drought",
        }
    )

    return (
        consequence_weights.groupby("hazard")
        .agg(consequence_weight=pd.NamedAgg(column="consequence_weight", aggfunc="sum"))
        .reset_index()
    )


def run_asset_hazard_weighting(
    nwr_eastern_path: pathlib.Path,
    nwr_northwest_path: pathlib.Path,
    output_path: pathlib.Path,
    incident_weights: IncidentWeightConfig,
) -> None:
    """Run asset hazard weighting workflow."""
    log_file = output_path / "asset_hazard_weighting.log"

    with ctk.LogHelper(
        pathlib.Path(__file__).stem,
        ctk.ToolDetails(
            cvt.__package__,
            cvt.__version__,
        ),
        log_file=log_file,
    ):
        derive_rail_weights(
            nwr_eastern_path,
            nwr_northwest_path,
            incident_weights,
            output_path,
        )
