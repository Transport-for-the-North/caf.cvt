"""Module for cleaning demand data."""

# LOAD LIBRARIES

import logging
import pathlib

import geopandas as gpd
import pandas as pd
import py7zr

from caf.cvt import data_cleaning, definitions, model_config

# LOGGING

LOG = logging.getLogger(__name__)

# MODULE CONSTANTS

_FREIGHT_DEMAND_NETWORK_MAP_MAX_DISTANCE = 500

## DEMAND


def clean_demand(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """
    Clean demand datasets ready for analysis.

    Use config switches to clean the relevant demand datasets, including freight rail, NoHAM,
    and model road flows.

    Parameters
    ----------
    config : model_config.Config
        The model configuration object containing switches and paths.
    boundary : gpd.GeoDataFrame
        The geographic boundary to clip demand data to.

    Returns
    -------
    None
    """
    LOG.info("Cleaning demand data...")
    if config.switches.freight_rail:
        _clean_freight_demand(config, boundary)
    if config.switches.noham_roads:
        _clean_noham_flows(config)
    if config.switches.model_roads:
        _clean_model_road_flows(config)
    LOG.info("Finished cleaning demand data.")


### FREIGHT


def _clean_freight_demand(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean freight demand data ready for analysis."""
    LOG.info("Cleaning freight demand data...")
    freight_network_demand = _read_freight_demand(
        config.paths.raw_input / config.impact.freight_demand, boundary
    )

    os_freight_network_demand = _map_freight_networks(
        freight_network_demand,
        config.paths.freight_rail_model_input,
    )

    data_cleaning.write_to_file(
        os_freight_network_demand,
        config.paths.freight_demand_model_input,
    )


def _read_freight_demand(path: pathlib.Path, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Read and clean freight demand data, and return as GeoDataFrame."""
    freight_network_demand = gpd.read_file(
        path, mask=boundary, columns=["dij_id", "2022_23_total", "2050_51 sc2_total"]
    )
    freight_network_demand = freight_network_demand.rename(
        columns={
            "2022_23_total": f"demand_{definitions.Scenarios.CURRENT}",
            "2050_51 sc2_total": f"demand_{definitions.Scenarios.FORECAST}",
        }
    )
    len_before_filter = len(freight_network_demand)
    freight_network_demand = data_cleaning.clip_to_boundary(freight_network_demand, boundary)
    filter_removed = len_before_filter - len(freight_network_demand)
    LOG.debug(
        "Freight demand filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    return freight_network_demand


def _map_freight_networks(
    freight_network_demand: gpd.GeoDataFrame, os_path: pathlib.Path
) -> gpd.GeoDataFrame:
    """Map freight demand data onto OS network, then clean and return."""
    os_freight_rail = gpd.read_file(os_path)
    os_freight_rail = os_freight_rail.to_crs(freight_network_demand.crs)
    len_before_mapping = len(freight_network_demand)
    os_freight_network_demand = gpd.sjoin_nearest(
        os_freight_rail,
        freight_network_demand,
        how="left",
        max_distance=_FREIGHT_DEMAND_NETWORK_MAP_MAX_DISTANCE,
        distance_col="distance",
    )
    len_after_mapping = len(os_freight_network_demand)
    LOG.debug(
        "Freight demand mapped to OS network - %s segments mapped onto %s OS network segments",
        len_before_mapping,
        len_after_mapping,
    )

    os_freight_network_demand[
        [f"demand_{definitions.Scenarios.CURRENT}", f"demand_{definitions.Scenarios.FORECAST}"]
    ] = os_freight_network_demand[
        [f"demand_{definitions.Scenarios.CURRENT}", f"demand_{definitions.Scenarios.FORECAST}"]
    ].fillna(0)
    return os_freight_network_demand.drop(columns=["index_right"])


### NOHAM FLOWS


def _clean_noham_flows(config: model_config.Config) -> None:
    """Clean NoHAM flows data, aggregate link flows, merge with network, then write to file."""
    LOG.info("Cleaning NoHAM demand data...")
    noham_network = gpd.read_file(config.paths.noham_road_model_input)
    network_link_ids = set(noham_network[definitions.ID_COL])

    scenario_flows = {}
    for year_label, year in config.impact.noham_years.items():
        if year_label == "baseline":
            scenario = definitions.Scenarios.CURRENT
        elif year_label == "future":
            scenario = definitions.Scenarios.FORECAST
        else:
            raise ValueError(
                f"Unexpected year label: {year_label}, expects 'baseline' or 'future'."
            )
        flows = _aggregate_link_flows_year(config, year, network_link_ids)

        flows = flows.rename(
            columns={
                col: f"{col}_{scenario}" for col in flows.columns if col != definitions.ID_COL
            }
        )

        scenario_flows[scenario] = flows

    current_ids = set(scenario_flows[definitions.Scenarios.CURRENT][definitions.ID_COL])
    forecast_ids = set(scenario_flows[definitions.Scenarios.FORECAST][definitions.ID_COL])
    common_ids = current_ids & forecast_ids
    current_only_ids = current_ids - forecast_ids
    forecast_only_ids = forecast_ids - current_ids

    noham_flows = scenario_flows[definitions.Scenarios.CURRENT].merge(
        scenario_flows[definitions.Scenarios.FORECAST], on=definitions.ID_COL, how="inner"
    )

    LOG.debug(
        "NoHAM flows merged: \n"
        "Current links: %s, Forecast links: %s \n"
        "Common links: %s, Current only links: %s, Forecast only links: %s",
        len(current_ids),
        len(forecast_ids),
        len(common_ids),
        len(current_only_ids),
        len(forecast_only_ids),
    )

    noham_net_flows = noham_network.merge(noham_flows, on=definitions.ID_COL, how="left")

    noham_net_flows = gpd.GeoDataFrame(
        noham_net_flows, geometry=definitions.GEOMETRY_COL, crs=noham_network.crs
    )

    data_cleaning.write_to_file(
        noham_net_flows,
        config.paths.noham_demand_model_input
    )


def _read_noham_h5(
    *,
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    year: int,
    time_period: str,
    user_class: str,
    noham_path: pathlib.Path,
    output_path: pathlib.Path,
    extract: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and clean NoHAM h5 files and extract the link, routes, and od's DataFrames."""
    if extract:
        with py7zr.SevenZipFile(noham_path, mode="r") as archive:
            archive.extract(
                path=output_path,
                targets=[
                    (
                        f"input h5s/"
                        f"{year}/"
                        f"NoHAM_Decarb_DM_Core_{year}_{time_period}_v107_SatPig_{user_class}.h5"
                    )
                ],
            )

    noham_demand_path = (
        output_path
        / "input h5s"
        / str(year)
        / f"NoHAM_Decarb_DM_Core_{year!s}_{time_period}_v107_SatPig_{user_class}.h5"
    )

    if (str(year), time_period) not in route_links_store:
        noham_routes = pd.read_hdf(noham_demand_path, key="/data/Route")
        noham_routes = noham_routes.reset_index()[["route", "link_id"]]
        noham_links = pd.read_hdf(noham_demand_path, key="/data/link")
        noham_links = noham_links[["a", "b"]]
        route_links_store[(str(year), time_period)] = (noham_routes, noham_links)
    else:
        noham_routes, noham_links = route_links_store[(str(year), time_period)]

    noham_ods = pd.read_hdf(noham_demand_path, key="/data/OD")
    noham_ods = noham_ods.reset_index()[["route", "abs_demand"]]

    return noham_ods, noham_routes, noham_links


def _aggregate_link_flows(
    ods: pd.DataFrame, routes: pd.DataFrame, links: pd.DataFrame
) -> pd.DataFrame:
    """Take NoHAM od's, routes, and link to create aggregated link flows DataFrame."""
    # Merge OD demand onto routes
    od_routes = routes.merge(ods[["route", "abs_demand"]], on="route", how="inner")

    link_demand = od_routes.groupby("link_id")["abs_demand"].sum().reset_index()

    return link_demand.merge(links[["a", "b"]], left_on="link_id", right_index=True)


def _process_single_noham_layer(
    config: model_config.Config,
    *,
    year: int,
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    time_period: str,
    user_class: str,
    link_ids: set[str],
) -> pd.DataFrame:
    LOG.debug("Processing NoHAM demand: %s %s %s", year, time_period, user_class)
    # TODO (DJ): Process NoHAM data without extracting zip files
    noham_ods, noham_routes, noham_links = _read_noham_h5(
        route_links_store=route_links_store,
        year=year,
        time_period=time_period,
        user_class=user_class,
        noham_path=config.paths.raw_input / config.impact.noham_demand,
        output_path=config.paths.raw_input / pathlib.Path("Impact/NoHAM Link Flows/h5 files"),
        extract=config.switches.noham_zip_extract,
    )
    noham_links["noham_link_id"] = (
        noham_links["a"].astype(str) + "_" + noham_links["b"].astype(str)
    )
    noham_links = noham_links[noham_links["noham_link_id"].isin(link_ids)]
    noham_routes = noham_routes[noham_routes["link_id"].isin(noham_links.index)]

    link_demand = _aggregate_link_flows(
        noham_ods,
        noham_routes,
        noham_links,
    )

    link_demand["link_id"] = link_demand["a"].astype(str) + "_" + link_demand["b"].astype(str)

    link_demand = link_demand[["link_id", "abs_demand"]]
    link_demand = link_demand.rename(
        columns={"abs_demand": f"{user_class}_{time_period}"}
    )  # Rename demand column

    LOG.debug(
        "%s ODs, %s Routes and %s Links aggregated to %s link flows",
        len(noham_ods),
        len(noham_routes),
        len(noham_links),
        len(link_demand),
    )

    return link_demand


def _aggregate_link_flows_year(
    config: model_config.Config, year: int, network_link_ids: set[str]
) -> pd.DataFrame:
    """Aggregate link flows for each year, time period, and user class."""
    route_links_store: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    noham_ucs = definitions.UserClasses.get_noham_classes()

    ts_dfs = []
    for time_period in definitions.TimePeriods:
        uc_dfs = []
        for user_class in noham_ucs:
            uc_dfs.append(
                _process_single_noham_layer(
                    config,
                    year=year,
                    route_links_store=route_links_store,
                    time_period=time_period,
                    user_class=user_class,
                    link_ids=network_link_ids,
                )
            )

        # Merge all user class dataframes
        combined_uc_df = uc_dfs[0]
        for df_uc in uc_dfs[1:]:
            combined_uc_df = combined_uc_df.merge(df_uc, on="link_id", how="outer")

        # Compute total demand for all vehicles for each time period
        combined_uc_df[f"all_vehs_{time_period}"] = combined_uc_df[
            [f"{uc}_{time_period}" for uc in noham_ucs]
        ].sum(axis=1)

        # Store result
        ts_dfs.append(combined_uc_df)

    # Merge all time period dataframes
    combined_ts_df = ts_dfs[0]
    for df_ts in ts_dfs[1:]:
        combined_ts_df = combined_ts_df.merge(df_ts, on="link_id", how="outer")

    # Compute totals for each user class across all time periods
    for uc in noham_ucs:
        combined_ts_df[f"{uc}_total"] = combined_ts_df[
            [f"{uc}_{tp}" for tp in definitions.TimePeriods]
        ].sum(axis=1)

    # Compute total of each user class across all time periods
    combined_ts_df["all_vehs_total"] = combined_ts_df[
        [f"all_vehs_{tp}" for tp in definitions.TimePeriods]
    ].sum(axis=1)

    combined_ts_df = combined_ts_df.rename(columns={"link_id": definitions.ID_COL})

    return combined_ts_df[
        [definitions.ID_COL, "all_vehs_total"] + [f"{uc}_total" for uc in noham_ucs]
    ]


### TRANSPORT MODEL FLOWS


def _clean_model_road_flows(config: model_config.Config) -> None:
    """Clean model flows data, aggregate link flows, merge with network, then write to file."""
    LOG.info("Cleaning model road flows data...")
    uc_link_flows = pd.read_csv(config.impact.model_road_flows.link_flows)
    ufs = pd.read_csv(config.impact.model_road_flows.ufs)
    annualisation_factors = pd.read_csv(config.impact.model_road_flows.annualisation_factors)

    model_road_links = gpd.read_file(
        config.paths.model_road_model_input
    )

    uc_link_flows = uc_link_flows.drop(columns=["id"])
    model_road_uc_link_flows = uc_link_flows.loc[
        uc_link_flows["link_id"].isin(model_road_links["link_id"])
    ]

    model_road_uc_link_flows = model_road_uc_link_flows.merge(
        ufs,
        left_on="ufs_id",
        right_on="id",
        how="left",
    )

    annualisation_factors = annualisation_factors.drop(columns=["id"])

    model_road_uc_link_flows = model_road_uc_link_flows.merge(
        annualisation_factors,
        on=["time_period", "userclass"],
        how="left",
    )

    model_road_uc_link_flows["annual_flow"] = (
        model_road_uc_link_flows["actual_flow"] * model_road_uc_link_flows["factor"]
    )

    model_road_uc_link_flows = model_road_uc_link_flows.drop(
        columns=["ufs_id", "id", "factor", "actual_flow"]
    )

    model_road_uc_link_flows = (
        model_road_uc_link_flows.groupby(["link_id", "userclass", "year", "scenario"])
        .agg({"annual_flow": "sum"})
        .reset_index()
    )

    # TODO (DJ): #40 Add forecast year link flow data when available, for now just copy
    years = sorted(model_road_uc_link_flows["year"].unique())
    current_year = years[0]
    forecast_year = years[0]

    current_flows = model_road_uc_link_flows[
        model_road_uc_link_flows["year"] == current_year
    ].copy()
    forecast_flows = model_road_uc_link_flows[
        model_road_uc_link_flows["year"] == forecast_year
    ].copy()

    if current_year == forecast_year:
        forecast_flows = current_flows.copy()

    current_flows = current_flows.pivot_table(
        index="link_id",
        columns="userclass",
        values="annual_flow",
        aggfunc="sum",
    )

    forecast_flows = forecast_flows.pivot_table(
        index="link_id",
        columns="userclass",
        values="annual_flow",
        aggfunc="sum",
    )

    current_flows.columns = [
        f"uc{uc}_demand_{definitions.Scenarios.CURRENT}" for uc in current_flows.columns
    ]

    forecast_flows.columns = [
        f"uc{uc}_demand_{definitions.Scenarios.FORECAST}" for uc in forecast_flows.columns
    ]

    model_road_uc_link_flows = current_flows.join(forecast_flows, how="left")

    model_road_uc_link_flows[f"demand_{definitions.Scenarios.CURRENT}"] = (
        model_road_uc_link_flows[
            [
                c
                for c in model_road_uc_link_flows.columns
                if c.endswith(f"_{definitions.Scenarios.CURRENT}")
            ]
        ].sum(axis=1)
    )

    model_road_uc_link_flows[f"demand_{definitions.Scenarios.FORECAST}"] = (
        model_road_uc_link_flows[
            [
                c
                for c in model_road_uc_link_flows.columns
                if c.endswith(f"_{definitions.Scenarios.FORECAST}")
            ]
        ].sum(axis=1)
    )

    model_road_uc_link_flows = model_road_uc_link_flows.merge(
        model_road_links[["link_id", definitions.GEOMETRY_COL]],
        on="link_id",
        how="left",
    )

    model_road_uc_link_flows = gpd.GeoDataFrame(
        model_road_uc_link_flows, geometry=definitions.GEOMETRY_COL, crs=definitions.BNG_CRS
    )

    data_cleaning.write_to_file(
        model_road_uc_link_flows,
        config.paths.model_roads_demand_model_input,
    )
