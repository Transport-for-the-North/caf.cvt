"""Module to derive asset-specific hazard weights."""
import logging
import pathlib

import caf.toolkit as ctk
import pandas as pd

import caf.cvt as cvt

_NAME = pathlib.Path(__file__).stem

# LOGGING

LOG = logging.getLogger(_NAME)

# PATHS

DATA_PATH = pathlib.Path("F:/4. Climate/Phase 2/Localisation/v2/Asset Risk Weighting/Data")
INPUT_PATH = DATA_PATH / "Input"
OUTPUT_PATH = DATA_PATH / "Output"


NWR_EASTERN_PATH = INPUT_PATH / "weather data eastern.csv"
NWR_NORTHWEST_PATH = INPUT_PATH / "weather data nwc.csv"

# CONSTANTS

INCIDENT_WEIGHTS = {
    "incident_share": 0.2,
    "minutes_share": 0.4,
    "cost_share": 0.4
}

# MAIN FUNCTIONS

def derive_rail_weights() -> None:
    """Derive asset-specific hazard weights for rail assets."""
    rail_incidents = read_clean_nwr_data()

    incident_summary = calculate_incident_summary(rail_incidents)

    impact_weights = aggregate_impact_weights(incident_summary)

    LOG.info("Writing impact weights to CSV...")
    impact_weights.to_csv(OUTPUT_PATH / "rail_impact_weights.csv", index=False)


def read_clean_nwr_data() -> pd.DataFrame:
    """Read and clean Network Rail weather data."""
    LOG.info("Reading and cleaning Network Rail weather data...")
    nwr_eastern = pd.read_csv(NWR_EASTERN_PATH)
    nwr_northwest = pd.read_csv(NWR_NORTHWEST_PATH)
    nwr_incidents = pd.concat([nwr_eastern, nwr_northwest], ignore_index=True)
    rail_incidents = nwr_incidents[
        nwr_incidents["Weather Category"].isin(["Wind", "Heat", "Snow", "Cold", "Subsidence"])
    ][["Weather Category", "Minutes", "cost"]]
    LOG.info("Finished reading and cleaning Network Rail weather data.")
    return rail_incidents


def calculate_incident_summary(rail_incidents: pd.DataFrame) -> pd.DataFrame:
    """Calculate impact weights for rail data based on incident data."""
    LOG.info("Calculating incident summary for rail data...")
    incident_summary = (
        rail_incidents.groupby("Weather Category")
        .agg(
            incidents=pd.NamedAgg(column="Weather Category", aggfunc="count"),
            total_minutes=pd.NamedAgg(column="Minutes", aggfunc="sum"),
            total_cost=pd.NamedAgg(column="cost", aggfunc="sum")
        )
        .reset_index()
        .round({
            "total_minutes": 0,
            "total_cost": 0
        })
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

    incident_summary["impact"] = round(
        incident_summary["incident_share"] * INCIDENT_WEIGHTS["incident_share"] +
        incident_summary["minutes_share"] * INCIDENT_WEIGHTS["minutes_share"] +
        incident_summary["cost_share"] * INCIDENT_WEIGHTS["cost_share"],
    2)

    return incident_summary


def aggregate_impact_weights(incident_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate impact weights for rail data based on incident summary."""
    LOG.info("Aggregating impact weights for rail data...")
    impact_weights = incident_summary[["Weather Category", "impact"]].rename(
        columns={"Weather Category": "hazard", "impact": "impact_weight"}
    )

    impact_weights["hazard"] = impact_weights["hazard"].map({
        "Wind": "Storm",
        "Heat": "Extreme Heat",
        "Snow": "Extreme Cold",
        "Cold": "Extreme Cold",
        "Subsidence": "Drought"
    })

    return impact_weights.groupby("hazard").agg(
        impact_weight=pd.NamedAgg(column="impact_weight", aggfunc="sum")
    ).reset_index()


# RUN SCRIPT

if __name__ == "__main__":
    with ctk.LogHelper(
            _NAME,
            ctk.ToolDetails(
                cvt.__package__,
                cvt.__version__,
            ),
            log_file=OUTPUT_PATH / "asset_risk_weighting.log",
        ) as log:

        derive_rail_weights()
