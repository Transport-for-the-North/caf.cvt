"""Intersect infrastructure with hazard layers to assign risk scores to infrastructure."""

import logging
import pathlib

import geopandas as gpd
import pandas as pd
import xyzservices

from caf.cvt import data_cleaning, file_paths, functional_rules, model_config
from caf.cvt.definitions import (
    ExtremeWeatherRiskCols,
    FloodingRiskCols,
    GroundStabilityRiskCols,
    ImpactCols,
    MainHazardRiskCols,
    NoHAMUserClasses,
    RiskColumn,
    Scenarios,
)

LOG = logging.getLogger(__name__)

_DEMAND_WEIGHT = 0.5

_TRAIN_STATIONS_BUFFER_SIZE_M = 100
_CHARGING_SITES_BUFFER_SIZE_M = 25
_BUS_COACH_STATIONS_BUFFER_SIZE_M = 50
_TRAM_STATIONS_BUFFER_SIZE_M = 25
_RAPID_TRANSPORT_STATIONS_BUFFER_SIZE_M = 50
_FERRY_TERMINALS_BUFFER_SIZE_M = 50
_PETROL_STATIONS_BUFFER_SIZE_M = 50

# GENERAL FUNCTIONS


def _infrastructure_risk_intersect(
    infrastructure_data: gpd.GeoDataFrame, hazards_dict: dict[str, gpd.GeoDataFrame]
) -> gpd.GeoDataFrame:
    """Intersect infrastructure with hazard risk layers.

    Spatially combine infrastructure with hazard layers using an intersection spatial join,
    then calculate hazard risk score as the max risk value of the intersection. Return merged
    GeoDataFrame with hazard risk columns added.
    """
    infrastructure_with_risk = infrastructure_data.copy()

    for _hazard_name, hazard_data in hazards_dict.items():
        # Spatial join to find intersections with hazards
        hazard_gdf_match = hazard_data.to_crs(infrastructure_with_risk.crs)  # Match CRS
        intersections = gpd.sjoin(
            infrastructure_with_risk, hazard_gdf_match, how="left", predicate="intersects"
        )

        # Identify risk columns
        risk_columns = hazard_gdf_match.columns[
            hazard_gdf_match.columns.str.contains("risk", case=False)
        ]

        # Calculate hazard risk score per infrastructure segment as max value of intersection
        agg = intersections.groupby(intersections.index)[risk_columns].max()

        # Merge back into main DataFrame
        infrastructure_with_risk = infrastructure_with_risk.join(agg, how="left")

    return infrastructure_with_risk.fillna(0)


def _reshape_for_scenarios(
    risk_data: gpd.GeoDataFrame, id_col: str, risk_cols_order: list[RiskColumn]
) -> gpd.GeoDataFrame:
    """Reshape dataframe by adding a current/forecast column to distinguish identical rows."""
    # Identify risk and descriptive columns
    risk_cols = [
        col
        for col in risk_data.columns
        if col.endswith((f"_{Scenarios.CURRENT}", f"_{Scenarios.FORECAST}"))
    ]
    descriptive_cols = [
        col
        for col in risk_data.columns
        if col not in risk_cols and col not in (id_col, "geometry")
    ]

    # Separate geometry for later
    geometry = risk_data[[id_col, "geometry"]].copy()

    # Melt only risk columns
    melted = risk_data.melt(
        id_vars=[id_col, *descriptive_cols],
        value_vars=risk_cols,
        var_name="variable",
        value_name="value",
    )

    # Extract scenario and clean variable names
    scenario_pattern = rf"_({'|'.join(Scenarios)})$"
    scenario_col = Scenarios.scenario_or_column()
    melted[scenario_col] = (
        melted["variable"]
        .str.extract(scenario_pattern)[0]
        .map({s: s.title() for s in Scenarios})
    )
    melted["variable"] = melted["variable"].str.replace(scenario_pattern, "", regex=True)

    # Pivot back so each risk variable becomes a column
    reshaped = melted.pivot_table(
        index=[id_col, scenario_col, *descriptive_cols],
        columns="variable",
        values="value",
    ).reset_index()

    # Reorder risk columns based on original order
    reshaped = reshaped[[id_col, scenario_col, *descriptive_cols, *risk_cols_order]]

    # Merge geometry back
    reshaped_gdf = reshaped.merge(geometry, on=id_col)
    return gpd.GeoDataFrame(reshaped_gdf, geometry="geometry", crs=risk_data.crs)


def _prepare_model_output(
    risk_data: gpd.GeoDataFrame,
    drop_cols: list[str],
    rename_map: dict[str, str],
    risk_cols_order: list[RiskColumn],
) -> gpd.GeoDataFrame:
    """Perform standard cleaning operations on risk data to prepare for model output."""
    risk_data = risk_data.drop(columns=drop_cols)
    risk_data = risk_data.drop_duplicates(subset=["geometry"])
    risk_data = risk_data.rename(columns=rename_map)
    risk_data = risk_data.to_crs(data_cleaning.BNG_CRS)
    risk_data = _reshape_for_scenarios(risk_data, "id", risk_cols_order)
    risk_data[risk_cols_order] = risk_data[risk_cols_order].round(1)
    return risk_data.rename(columns={col: f"{col}_score" for col in risk_cols_order})


def _split_csv_shapefile(
    config: model_config.Config,
    gdf: gpd.GeoDataFrame,
    id_col: str,
    out_path_no_suffix: pathlib.Path,
) -> None:
    """Split GeoDataFrame into a CSV and Shapefile, then write to file.

    Separates GeoDataFrame into a dataframe with an ID and attributes, and a Shapefile with an
    ID and geometry, then writes them to file.
    """
    # Separate spatial and attribute data
    spatial_gdf = gdf[[id_col, "geometry"]].copy()
    attribute_df = gdf.drop(columns=["geometry"])

    # Remove duplicates from spatial data
    spatial_gdf = spatial_gdf.drop_duplicates()

    # Save to file
    data_cleaning.write_to_file(
        spatial_gdf, config.paths.model_output / out_path_no_suffix.with_suffix(".shp")
    )
    data_cleaning.write_to_file(
        attribute_df, config.paths.model_output / out_path_no_suffix.with_suffix(".csv")
    )


def _audit_infrastructure_risk(
    infrastructure_risk: gpd.GeoDataFrame,
    infrastructure_name: str,
    cols: list[RiskColumn],
    audit_path: pathlib.Path,
    *,
    feature_range: tuple[int, int],
    linewidth: float = 0.5,
) -> None:
    """Plot choropleth maps for infrastructure risk."""
    audit_path.mkdir(parents=True, exist_ok=True)

    for risk_col in cols:
        functional_rules.plot_choropleth_current_and_forecast(
            risk_data=infrastructure_risk,
            column=risk_col,
            title=f"{infrastructure_name} {risk_col.replace('_', ' ').title()}",
            out_path=audit_path / f"{risk_col}_choropleth.png",
            linewidth=linewidth,
            edgecolor=None,
            feature_range=feature_range,
            basemap_source=xyzservices.providers.CartoDB.Positron,
        )


def _get_impact_weights(hazards: list[str]) -> dict[str, float]:
    """Generate impact weights."""
    impact_weights = {"demand": _DEMAND_WEIGHT}

    # Divide remaining weight equally amongst hazards
    hazard_weight = (1 - _DEMAND_WEIGHT) / len(hazards)
    for hazard in hazards:
        impact_weights[hazard] = hazard_weight
    return impact_weights


# LAYERING


def layering(config: model_config.Config) -> None:
    """
    Layer infrastructure with hazard risk to assign risk to each piece of infrastructure.

    Read in hazard layers from functional rules output, then spatially intersect with
    infrastructure layers to assign risk to each piece of infrastructure. Calculate impact
    indices for NoHAM and freight rail.

    Parameters
    ----------
    config : Config
        Main config for the model, containing paths and settings.
    """
    hazard_layers = _read_hazard_layers(config)

    risk_cols = []

    if config.switches.extreme_weather:
        risk_cols.extend([MainHazardRiskCols.EXTREME_WEATHER, *ExtremeWeatherRiskCols])

    if config.switches.flooding:
        risk_cols.extend([MainHazardRiskCols.FLOODING, *FloodingRiskCols])

    if config.switches.ground_stability:
        risk_cols.extend([MainHazardRiskCols.GROUND_STABILITY, *GroundStabilityRiskCols])

    if config.switches.coastal_erosion:
        risk_cols.extend([MainHazardRiskCols.COASTAL_EROSION])

    audit_path = config.paths.audit_path / "Layering"

    _infrastructure_layering(config, hazard_layers, risk_cols, audit_path)


## HAZARD LAYERS


def _read_hazard_layers(config: model_config.Config) -> dict[str, gpd.GeoDataFrame]:
    """Read and clean hazard layers, and return them in a dictionary."""
    hazard_layers = {}
    if config.switches.extreme_weather:
        LOG.info("Reading extreme weather layer.")
        hazard_layers["Extreme Weather"] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH
        )
    if config.switches.flooding:
        LOG.info("Reading flooding layer.")
        hazard_layers["Flooding"] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.FLOODING_RISK_MODEL_INTERIM_OUTPUT_PATH
        )
    if config.switches.ground_stability:
        LOG.info("Reading ground stability layer.")
        hazard_layers["Ground Stability"] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH
        )
    if config.switches.coastal_erosion:
        LOG.info("Reading coastal erosion layer.")
        hazard_layers["Coastal Erosion"] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH
        )
    return hazard_layers


## INFRASTRUCTURE-HAZARD LAYERING


def _infrastructure_layering(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer roads, rail, and other infrastructure with hazards."""
    _get_road_risk(config, hazard_layers, risk_cols, audit_path)
    _get_rail_risk(config, hazard_layers, risk_cols, audit_path)
    _get_other_risk(config, hazard_layers, risk_cols, audit_path)


### ROAD


def _get_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer OS Open Roads and NoHAM with hazards to assign risk."""
    LOG.info("Calculating road risk...")
    if config.switches.all_roads:
        _os_open_road_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.noham_roads:
        _noham_road_risk(config, hazard_layers, risk_cols, audit_path)
    LOG.info("Road risk calculation complete.")


#### OS Open Roads


def _os_open_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Intersect OS Road infrastructure with hazards, clean output, and write to file."""
    LOG.info("Layering OS Open Roads with hazard risk...")
    os_road = gpd.read_file(config.paths.model_input / file_paths.OS_ROAD_MODEL_INPUT_PATH)

    if os_road.empty:
        LOG.warning("OS Open Roads layer is empty. Skipping.")
        return

    os_road_risk = _infrastructure_risk_intersect(os_road, hazard_layers)

    _audit_infrastructure_risk(
        os_road_risk,
        "All Roads",
        risk_cols,
        audit_path / "Road" / "OS Roads",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        os_road_risk,
        config.paths.model_output / "Road" / "OS Roads" / "os_road_risk.gpkg",
    )

    os_road_risk = _prepare_model_output(
        risk_data=os_road_risk,
        drop_cols=[],
        rename_map={"identifier": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config, os_road_risk, "id", pathlib.Path("Road") / "OS Roads" / "os_road_risk"
    )
    LOG.info("Finished layering OS Open Roads with hazard risk.")


#### NoHAM


def _noham_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get NoHAM road risk and write to file.

    Intersect NoHAM with hazards, calculate impact index, clean output, and write
    to file.
    """
    LOG.info("Layering NoHAM with hazard risk and calculating impact index...")
    noham_net_flows = gpd.read_file(
        config.paths.model_input / file_paths.NOHAM_FLOWS_MODEL_INPUT_PATH
    )

    if noham_net_flows.empty:
        LOG.warning("NoHAM network flows layer is empty. Skipping.")
        return

    noham_risk = _infrastructure_risk_intersect(noham_net_flows, hazard_layers)

    feature_range = (config.constants.score_min, config.constants.score_max)
    noham_risk = _noham_impact_index(noham_risk, feature_range)

    risk_impact_cols = [*risk_cols, *ImpactCols]

    _audit_infrastructure_risk(
        noham_risk,
        "NoHAM Roads",
        risk_impact_cols,
        audit_path / "Road" / "NoHAM",
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        noham_risk,
        config.paths.model_output / "Road" / "NoHAM" / "noham_risk.gpkg",
    )

    noham_risk = _prepare_model_output(
        risk_data=noham_risk,
        drop_cols=[],
        rename_map={"link_id": "id"},
        risk_cols_order=risk_impact_cols,
    )

    _split_csv_shapefile(
        config, noham_risk, "id", pathlib.Path("Road") / "NoHAM" / "noham_risk"
    )

    LOG.info("Finished layering NoHAM with hazard risk and calculating impact index.")


def _noham_impact_index(
    noham: gpd.GeoDataFrame, feature_range: tuple[int, int]
) -> gpd.GeoDataFrame:
    """Normalise NoHAM demand, then calculate impact index."""
    noham = _normalise_uc_demand(noham, feature_range)
    noham = _normalise_total_demand(noham, feature_range)
    noham = _calculate_noham_impact(noham)
    return _normalise_noham_impact(noham, feature_range)


def _normalise_uc_demand(noham: pd.DataFrame, feature_range: tuple[int, int]) -> pd.DataFrame:
    """Normalise NoHAM demand for each user class individually."""
    pairs = [
        (f"{uc}_total_{Scenarios.CURRENT}", f"{uc}_total_{Scenarios.FORECAST}")
        for uc in NoHAMUserClasses
    ]

    noham = functional_rules.min_max_scaling_pair(noham, pairs, feature_range)

    rename_map = {
        col: col.replace("total", "demand")
        for uc in NoHAMUserClasses
        for col in [f"{uc}_total_{Scenarios.CURRENT}", f"{uc}_total_{Scenarios.FORECAST}"]
    }
    return noham.rename(columns=rename_map)


def _normalise_total_demand(
    noham: pd.DataFrame, feature_range: tuple[int, int]
) -> pd.DataFrame:
    """Normalise NoHAM demand across all user classes combined."""
    pairs = [(f"all_vehs_total_{Scenarios.CURRENT}", f"all_vehs_total_{Scenarios.FORECAST}")]
    noham = functional_rules.min_max_scaling_pair(noham, pairs, feature_range)
    return noham.rename(
        columns={
            f"all_vehs_total_{Scenarios.CURRENT}": f"demand_{Scenarios.CURRENT}",
            f"all_vehs_total_{Scenarios.FORECAST}": f"demand_{Scenarios.FORECAST}",
        }
    )


def _calculate_noham_impact(noham: pd.DataFrame) -> pd.DataFrame:
    """Calculate NoHAM impact score for each user class, and for all vehicles."""
    # Calculate impact metric for each user class
    risk_cols = [
        col for col in MainHazardRiskCols if f"{col}_{Scenarios.CURRENT}" in noham.columns
    ]

    hazards = [col.removesuffix("_risk") for col in risk_cols]
    impact_weights = _get_impact_weights(hazards)

    for scenario in Scenarios:
        hazard_component = sum(
            noham[f"{risk_col}_{scenario}"] * impact_weights[risk_col.removesuffix("_risk")]
            for risk_col in risk_cols
        )
        for uc in NoHAMUserClasses:
            impact_component = noham[f"{uc}_demand_{scenario}"] * impact_weights["demand"]
            noham[f"{uc}_impact_{scenario}"] = impact_component + hazard_component

        impact_component = noham[f"demand_{scenario}"] * impact_weights["demand"]
        noham[f"impact_{scenario}"] = impact_component + hazard_component

    demand_cols = [col for col in noham.columns if "demand" in col]
    return noham.drop(columns=demand_cols)


def _normalise_noham_impact(
    noham: pd.DataFrame, feature_range: tuple[int, int]
) -> pd.DataFrame:
    """Normalise NoHAM impact scores across all user classes combined."""
    pairs = [
        (f"{uc}_impact_{Scenarios.CURRENT}", f"{uc}_impact_{Scenarios.FORECAST}")
        for uc in NoHAMUserClasses
    ] + [(f"impact_{Scenarios.CURRENT}", f"impact_{Scenarios.FORECAST}")]

    return functional_rules.min_max_scaling_pair(noham, pairs, feature_range)


### RAIL


def _get_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer passenger rail and freight rail network with hazard to assign risk."""
    LOG.info("Calculating rail risk...")
    if config.switches.passenger_rail:
        _passenger_rail_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.freight_rail:
        _freight_rail_risk(config, hazard_layers, risk_cols, audit_path)
    LOG.info("Rail risk calculation complete.")


#### Passenger Rail


def _passenger_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Intersect passenger rail network with hazard to assign risk, clean and write to file."""
    LOG.info("Layering passenger rail network with hazard risk...")
    passenger_rail_network = gpd.read_file(
        config.paths.model_input / file_paths.PASSENGER_RAIL_MODEL_INPUT_PATH
    )

    if passenger_rail_network.empty:
        LOG.warning("Passenger rail network layer is empty. Skipping.")
        return

    passenger_rail_network_risk = _infrastructure_risk_intersect(
        passenger_rail_network, hazard_layers
    )

    _audit_infrastructure_risk(
        passenger_rail_network_risk,
        "Passenger Rail",
        risk_cols,
        audit_path / "Rail" / "Passenger Rail",
        linewidth=1.0,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        passenger_rail_network_risk,
        config.paths.model_output
        / "Rail"
        / "Passenger Rail"
        / "passenger_rail_network_risk.gpkg",
    )

    passenger_rail_network_risk = _prepare_model_output(
        risk_data=passenger_rail_network_risk,
        drop_cols=[],
        rename_map={
            "osid": "id",
            "desc": "description",
            "phys_level": "physical_level",
            "rail_use": "railway_use",
            "track_rep": "track_representation",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        passenger_rail_network_risk,
        "id",
        pathlib.Path("Rail") / "Passenger Rail" / "passenger_rail_network_risk",
    )

    LOG.info("Finished layering passenger rail network with hazard risk.")


#### Freight Rail


def _freight_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Calculate freight rail risk and write to file.

    Intersect freight network with hazard risk, calculate impact index, clean and write to
    file.
    """
    LOG.info("Layering freight rail network with hazard risk and calculating impact index...")
    freight_rail_network = gpd.read_file(
        config.paths.model_input / file_paths.FREIGHT_DEMAND_MODEL_INPUT_PATH
    )

    if freight_rail_network.empty:
        LOG.warning("Freight rail network layer is empty. Skipping.")
        return

    freight_rail_network_risk = _infrastructure_risk_intersect(
        freight_rail_network, hazard_layers
    )

    feature_range = (config.constants.score_min, config.constants.score_max)
    freight_rail_network_risk = _freight_impact_index(freight_rail_network_risk, feature_range)

    # Set the correct CRS
    freight_rail_network_risk = freight_rail_network_risk.set_crs(
        data_cleaning.BNG_CRS, allow_override=True
    )

    _audit_infrastructure_risk(
        freight_rail_network_risk,
        "Freight Rail",
        [*risk_cols, ImpactCols.IMPACT],
        audit_path / "Rail" / "Freight Rail",
        linewidth=1.0,
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        freight_rail_network_risk,
        config.paths.model_output / "Rail" / "Freight Rail" / "freight_rail_network_risk.gpkg",
    )

    freight_rail_network_risk = _prepare_model_output(
        risk_data=freight_rail_network_risk,
        drop_cols=[
            "dij_id",
            "distance",
        ],
        rename_map={
            "osid": "id",
            "desc": "description",
            "phys_level": "physical_level",
            "rail_use": "railway_use",
            "track_rep": "track_representation",
        },
        risk_cols_order=[*risk_cols, ImpactCols.IMPACT],
    )

    _split_csv_shapefile(
        config,
        freight_rail_network_risk,
        "id",
        pathlib.Path("Rail") / "Freight Rail" / "freight_rail_network_risk",
    )
    LOG.info(
        "Finished layering freight rail network with hazard risk and calculating impact index."
    )


def _freight_impact_index(
    freight_rail_network_risk: gpd.GeoDataFrame, feature_range: tuple[int, int]
) -> gpd.GeoDataFrame:
    """Calculate impact index using freight demand data and hazard risk."""
    freight_rail_network_risk = functional_rules.min_max_scaling_pair(
        freight_rail_network_risk,
        [(f"demand_{Scenarios.CURRENT}", f"demand_{Scenarios.FORECAST}")],
        feature_range,
    )

    freight_rail_network_risk = _calculate_freight_impact(freight_rail_network_risk)

    freight_rail_network_risk = functional_rules.min_max_scaling_pair(
        freight_rail_network_risk,
        [(f"impact_{Scenarios.CURRENT}", f"impact_{Scenarios.FORECAST}")],
        feature_range,
    )

    return gpd.GeoDataFrame(freight_rail_network_risk, geometry="geometry", crs="EPSG:4326")


def _calculate_freight_impact(freight_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite impact score for current and forecast years."""
    risk_cols = [
        col
        for col in MainHazardRiskCols
        if f"{col}_{Scenarios.CURRENT}" in freight_data.columns
    ]

    hazards = [col.removesuffix("_risk") for col in risk_cols]
    impact_weights = _get_impact_weights(hazards)

    for scenario in Scenarios:
        impact_component = freight_data[f"demand_{scenario}"] * impact_weights["demand"]
        hazard_component = sum(
            freight_data[f"{risk_col}_{scenario}"]
            * impact_weights[risk_col.removesuffix("_risk")]
            for risk_col in risk_cols
        )

        freight_data[f"impact_{scenario}"] = impact_component + hazard_component

    demand_cols = [col for col in freight_data.columns if "demand" in col]
    return freight_data.drop(columns=demand_cols)


### OTHER


def _get_other_risk(  # noqa: C901
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer other infrastructure with hazards to assign risk."""
    LOG.info("Calculating risk for other infrastructure...")
    if config.switches.train_stations:
        _train_stations_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.charging_sites:
        _charging_sites_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.airports:
        _airports_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.bus_coach_stations:
        _bus_coach_stations_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.bus_stops:
        _bus_stops_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.tram_stations:
        _tram_stations_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.rapid_transport_stations:
        _rapid_transport_stations_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.ferry_terminals:
        _ferry_terminals_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.petrol_stations:
        _petrol_stations_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.national_cycle_network:
        _ncn_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.tram_network:
        _tram_network_risk(config, hazard_layers, risk_cols, audit_path)
    if config.switches.rapid_transport_network:
        _rapid_transport_network_risk(config, hazard_layers, risk_cols, audit_path)
    LOG.info("Risk calculation for other infrastructure complete.")


def _buffer_geometry(infrastructure: gpd.GeoDataFrame, buffer_size_m: int) -> gpd.GeoDataFrame:
    """Buffers the geometries of a given GeoDataFrame to a given size in metres."""
    infrastructure = infrastructure.to_crs(data_cleaning.BNG_CRS)
    infrastructure["geometry"] = infrastructure.buffer(buffer_size_m)
    return infrastructure


#### Train Stations


def _train_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get train station risk and write to file.

    Buffer train stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering train stations with hazard risk...")
    train_stations = gpd.read_file(
        config.paths.model_input / file_paths.TRAIN_STATIONS_MODEL_INPUT_PATH
    )

    if train_stations.empty:
        LOG.warning("Train stations layer is empty. Skipping.")
        return

    train_stations = _buffer_geometry(train_stations, _TRAIN_STATIONS_BUFFER_SIZE_M)

    train_stations_risk = _infrastructure_risk_intersect(train_stations, hazard_layers)

    _audit_infrastructure_risk(
        train_stations_risk,
        "Train Stations",
        risk_cols,
        audit_path / "Other" / "Train Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        train_stations_risk,
        config.paths.model_output / "Other" / "Train Stations" / "train_stations_risk.gpkg",
    )

    train_stations_risk = _prepare_model_output(
        risk_data=train_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        train_stations_risk,
        "id",
        pathlib.Path("Other") / "Train Stations" / "train_stations_risk",
    )
    LOG.info("Finished layering train stations with hazard risk.")


#### EV Charging Sites


def _charging_sites_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get EV charging site risk and write to file.

    Buffer charging sites, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering EV charging sites with hazard risk...")
    charging_sites = gpd.read_file(
        config.paths.model_input / file_paths.CHARGING_SITES_MODEL_INPUT_PATH
    )

    if charging_sites.empty:
        LOG.warning("EV charging sites layer is empty. Skipping.")
        return

    charging_sites = _buffer_geometry(charging_sites, _CHARGING_SITES_BUFFER_SIZE_M)

    charging_sites_risk = _infrastructure_risk_intersect(charging_sites, hazard_layers)

    _audit_infrastructure_risk(
        charging_sites_risk,
        "EV Charging Sites",
        risk_cols,
        audit_path / "Other" / "EV Charging Sites",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        charging_sites_risk,
        config.paths.model_output / "Other" / "EV Charging Sites" / "charging_sites_risk.gpkg",
    )

    charging_sites_risk = _prepare_model_output(
        risk_data=charging_sites_risk,
        drop_cols=[],
        rename_map={"devices": "installed_devices"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        charging_sites_risk,
        "id",
        pathlib.Path("Other") / "EV Charging Sites" / "charging_sites_risk",
    )
    LOG.info("Finished layering EV charging sites with hazard risk.")


#### Airports


def _airports_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get airport risk and write to file.

    Intersect airports with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering airports with hazard risk...")
    airports = gpd.read_file(config.paths.model_input / file_paths.AIRPORTS_MODEL_INPUT_PATH)

    if airports.empty:
        LOG.warning("Airports layer is empty. Skipping.")
        return

    airports_risk = _infrastructure_risk_intersect(airports, hazard_layers)

    _audit_infrastructure_risk(
        airports_risk,
        "Airports",
        risk_cols,
        audit_path / "Other" / "Airports",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    airports_risk = data_cleaning.explode_to_polygons(airports_risk)

    data_cleaning.write_to_file(
        airports_risk,
        config.paths.model_output / "Other" / "Airports" / "airports_risk.gpkg",
    )

    airports_risk = _prepare_model_output(
        risk_data=airports_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        airports_risk,
        "id",
        pathlib.Path("Other") / "Airports" / "airports_risk",
    )
    LOG.info("Finished layering airports with hazard risk.")


#### Bus and Coach Stations


def _bus_coach_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get bus and coach station risk and write to file.

    Buffer bus and coach stations, then intersect with hazard risk, clean output, and write to
    file.
    """
    LOG.info("Layering bus and coach stations with hazard risk...")
    bus_coach_stations = gpd.read_file(
        config.paths.model_input / file_paths.BUS_COACH_STATIONS_MODEL_INPUT_PATH
    )

    if bus_coach_stations.empty:
        LOG.warning("Bus and coach stations layer is empty. Skipping.")
        return

    bus_coach_stations = _buffer_geometry(
        bus_coach_stations, _BUS_COACH_STATIONS_BUFFER_SIZE_M
    )

    bus_coach_stations_risk = _infrastructure_risk_intersect(bus_coach_stations, hazard_layers)

    _audit_infrastructure_risk(
        bus_coach_stations_risk,
        "Bus and Coach Stations",
        risk_cols,
        audit_path / "Other" / "Bus and Coach Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        bus_coach_stations_risk,
        config.paths.model_output
        / "Other"
        / "Bus and Coach Stations"
        / "bus_coach_stations_risk.gpkg",
    )

    bus_coach_stations_risk = _prepare_model_output(
        risk_data=bus_coach_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        bus_coach_stations_risk,
        "id",
        pathlib.Path("Other") / "Bus and Coach Stations" / "bus_coach_stations_risk",
    )
    LOG.info("Finished layering bus and coach stations with hazard risk.")


#### Bus Stops


def _bus_stops_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Intersect bus stops with hazard risk, clean output, and write to file."""
    LOG.info("Layering bus stops with hazard risk...")
    bus_stops = gpd.read_file(config.paths.model_input / file_paths.BUS_STOPS_MODEL_INPUT_PATH)

    if bus_stops.empty:
        LOG.warning("Bus stops layer is empty. Skipping.")
        return

    bus_stops_risk = _infrastructure_risk_intersect(bus_stops, hazard_layers)

    _audit_infrastructure_risk(
        bus_stops_risk,
        "Bus Stops",
        risk_cols,
        audit_path / "Other" / "Bus Stops",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        bus_stops_risk,
        config.paths.model_output / "Other" / "Bus Stops" / "bus_stops_risk.gpkg",
    )

    bus_stops_risk = _prepare_model_output(
        risk_data=bus_stops_risk,
        drop_cols=[],
        rename_map={"stop_id": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        bus_stops_risk,
        "id",
        pathlib.Path("Other") / "Bus Stops" / "bus_stops_risk",
    )
    LOG.info("Finished layering bus stops with hazard risk.")


#### Tram Stations


def _tram_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get tram station risk and write to file.

    Buffer tram stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering tram stations with hazard risk...")
    tram_stations = gpd.read_file(
        config.paths.model_input / file_paths.TRAM_STATIONS_MODEL_INPUT_PATH
    )

    if tram_stations.empty:
        LOG.warning("Tram stations layer is empty. Skipping.")
        return

    tram_stations = _buffer_geometry(tram_stations, _TRAM_STATIONS_BUFFER_SIZE_M)

    tram_stations_risk = _infrastructure_risk_intersect(tram_stations, hazard_layers)

    _audit_infrastructure_risk(
        tram_stations_risk,
        "Tram Stations",
        risk_cols,
        audit_path / "Other" / "Tram Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        tram_stations_risk,
        config.paths.model_output / "Other" / "Tram Stations" / "tram_stations_risk.gpkg",
    )

    tram_stations_risk = _prepare_model_output(
        risk_data=tram_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tram_stations_risk,
        "id",
        pathlib.Path("Other") / "Tram Stations" / "tram_stations_risk",
    )
    LOG.info("Finished layering tram stations with hazard risk.")


#### Rapid Transport Stations


def _rapid_transport_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get rapid transport station risk and write to file.

    Buffer rapid transport stations, then intersect with hazard risk, clean output, and write
    to file.
    """
    LOG.info("Layering rapid transport stations with hazard risk...")
    rapid_transport_stations = gpd.read_file(
        config.paths.model_input / file_paths.RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH
    )

    if rapid_transport_stations.empty:
        LOG.warning("Rapid transport stations layer is empty. Skipping.")
        return

    rapid_transport_stations = _buffer_geometry(
        rapid_transport_stations, _RAPID_TRANSPORT_STATIONS_BUFFER_SIZE_M
    )

    rapid_transport_stations_risk = _infrastructure_risk_intersect(
        rapid_transport_stations, hazard_layers
    )

    _audit_infrastructure_risk(
        rapid_transport_stations_risk,
        "Rapid Transport Stations",
        risk_cols,
        audit_path / "Other" / "Rapid Transport Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        rapid_transport_stations_risk,
        config.paths.model_output
        / "Other"
        / "Rapid Transport Stations"
        / "rapid_transport_stations_risk.gpkg",
    )

    rapid_transport_stations_risk = _prepare_model_output(
        risk_data=rapid_transport_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        rapid_transport_stations_risk,
        "id",
        pathlib.Path("Other") / "Rapid Transport Stations" / "rapid_transport_stations_risk",
    )
    LOG.info("Finished layering rapid transport stations with hazard risk.")


#### Ferry Terminals


def _ferry_terminals_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get ferry terminal risk and write to file.

    Buffer ferry terminals, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering ferry terminals with hazard risk...")
    ferry_terminals = gpd.read_file(
        config.paths.model_input / file_paths.FERRY_TERMINALS_MODEL_INPUT_PATH
    )

    if ferry_terminals.empty:
        LOG.warning("Ferry terminals layer is empty. Skipping.")
        return

    ferry_terminals = _buffer_geometry(ferry_terminals, _FERRY_TERMINALS_BUFFER_SIZE_M)

    ferry_terminals_risk = _infrastructure_risk_intersect(ferry_terminals, hazard_layers)

    _audit_infrastructure_risk(
        ferry_terminals_risk,
        "Ferry Terminals",
        risk_cols,
        audit_path / "Other" / "Ferry Terminals",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        ferry_terminals_risk,
        config.paths.model_output / "Other" / "Ferry Terminals" / "ferry_terminals_risk.gpkg",
    )

    ferry_terminals_risk = _prepare_model_output(
        risk_data=ferry_terminals_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        ferry_terminals_risk,
        "id",
        pathlib.Path("Other") / "Ferry Terminals" / "ferry_terminals_risk",
    )
    LOG.info("Finished layering ferry terminals with hazard risk.")


#### Petrol Stations


def _petrol_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get petrol station risk and write to file.

    Buffer petrol stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering petrol stations with hazard risk...")
    petrol_stations = gpd.read_file(
        config.paths.model_input / file_paths.PETROL_STATIONS_MODEL_INPUT_PATH
    )

    if petrol_stations.empty:
        LOG.warning("Petrol stations layer is empty. Skipping.")
        return

    petrol_stations = _buffer_geometry(petrol_stations, _PETROL_STATIONS_BUFFER_SIZE_M)

    petrol_stations_risk = _infrastructure_risk_intersect(petrol_stations, hazard_layers)

    _audit_infrastructure_risk(
        petrol_stations_risk,
        "Petrol Stations",
        risk_cols,
        audit_path / "Other" / "Petrol Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        petrol_stations_risk,
        config.paths.model_output / "Other" / "Petrol Stations" / "petrol_stations_risk.gpkg",
    )

    petrol_stations_risk = _prepare_model_output(
        risk_data=petrol_stations_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        petrol_stations_risk,
        "id",
        pathlib.Path("Other") / "Petrol Stations" / "petrol_stations_risk",
    )
    LOG.info("Finished layering petrol stations with hazard risk.")


#### National Cycle Network


def _ncn_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get NCN risk and write to file.

    Intersect National Cycle Network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering National Cycle Network with hazard risk...")
    ncn = gpd.read_file(
        config.paths.model_input / file_paths.NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH
    )

    if ncn.empty:
        LOG.warning("National Cycle Network layer is empty. Skipping.")
        return

    ncn_risk = _infrastructure_risk_intersect(ncn, hazard_layers)

    _audit_infrastructure_risk(
        ncn_risk,
        "National Cycle Network",
        risk_cols,
        audit_path / "Other" / "National Cycle Network",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        ncn_risk,
        config.paths.model_output / "Other" / "National Cycle Network" / "ncn_risk.gpkg",
    )

    ncn_risk = _prepare_model_output(
        risk_data=ncn_risk,
        drop_cols=[],
        rename_map={
            "Desc_": "description",
            "Greenway": "greenway",
            "RouteType": "route_type",
            "RouteNo": "route_number",
            "LinkNo": "link_number",
            "Surface": "surface",
            "Quality": "quality",
            "Lighting": "lighting",
            "RoadClass": "road_class",
            "SegmentID": "id",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        ncn_risk,
        "id",
        pathlib.Path("Other") / "National Cycle Network" / "ncn_risk",
    )
    LOG.info("Finished layering National Cycle Network with hazard risk.")


#### Tram Network


def _tram_network_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get tram network risk and write to file.

    Intersect tram network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering tram network with hazard risk...")
    tram_network = gpd.read_file(
        config.paths.model_input / file_paths.TRAM_NETWORK_MODEL_INPUT_PATH
    )

    if tram_network.empty:
        LOG.warning("Tram network layer is empty. Skipping.")
        return

    tram_risk = _infrastructure_risk_intersect(tram_network, hazard_layers)

    _audit_infrastructure_risk(
        tram_risk,
        "Tram Network",
        risk_cols,
        audit_path / "Other" / "Tram Network",
        linewidth=1.0,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        tram_risk,
        config.paths.model_output / "Other" / "Tram Network" / "tram_network_risk.gpkg",
    )

    tram_risk = _prepare_model_output(
        risk_data=tram_risk,
        drop_cols=[],
        rename_map={
            "osid": "id",
            "desc": "description",
            "phys_level": "physical_level",
            "rail_use": "railway_use",
            "track_rep": "track_representation",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tram_risk,
        "id",
        pathlib.Path("Other") / "Tram Network" / "tram_network_risk",
    )
    LOG.info("Finished layering tram network with hazard risk.")


#### Rapid Transport Network


def _rapid_transport_network_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get rapid transport network risk and write to file.

    Intersect rapid transport network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering rapid transport network with hazard risk...")
    rapid_transport = gpd.read_file(
        config.paths.model_input / file_paths.RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH
    )

    if rapid_transport.empty:
        LOG.warning("Rapid transport network layer is empty. Skipping.")
        return

    rapid_transport_risk = _infrastructure_risk_intersect(rapid_transport, hazard_layers)

    _audit_infrastructure_risk(
        rapid_transport_risk,
        "Rapid Transport Network",
        risk_cols,
        audit_path / "Other" / "Rapid Transport Network",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        rapid_transport_risk,
        config.paths.model_output
        / "Other"
        / "Rapid Transport Network"
        / "rapid_transport_network_risk.gpkg",
    )

    rapid_transport_risk = _prepare_model_output(
        risk_data=rapid_transport_risk,
        drop_cols=[],
        rename_map={
            "osid": "id",
            "desc": "description",
            "phys_level": "physical_level",
            "rail_use": "railway_use",
            "track_rep": "track_representation",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        rapid_transport_risk,
        "id",
        pathlib.Path("Other") / "Rapid Transport Network" / "rapid_transport_network_risk",
    )
    LOG.info("Finished layering rapid transport network with hazard risk.")
