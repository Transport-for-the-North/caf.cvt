"""Intersect infrastructure with hazard layers to assign risk scores to infrastructure."""

import logging
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from caf.cvt import data_cleaning, file_paths, functional_rules, model_config

LOG = logging.getLogger(__name__)

_HAZARD_RISK_COLS = [
    # Extreme Weather risk columns
    "heat_risk",
    "cold_risk",
    "drought_risk",
    "storm_risk",
    "extreme_weather_risk",
    # Flooding risk columns
    "rivers_sea_flood_risk",
    "surface_water_flood_risk",
    "flood_risk",
    # Ground Stability risk columns
    *data_cleaning.GEOSURE_RISK_COLS,
    "shrink_swell_geoclimate_risk",
    "ground_stability_risk",
    # Coastal Erosion risk columns
    "coastal_erosion_risk",
]

_NOHAM_IMPACT_COLS = [
    "uc1_impact",
    "uc2_impact",
    "uc3_impact",
    "uc4_impact",
    "uc5_impact",
    "impact",
]


_IMPACT_WEIGHTS = {
    "demand": 0.5,  # Weight demand as half of impact score
    "flood": 0.125,  # Weight hazards as 0.125 each to make up half
    "extreme_weather": 0.125,
    "ground_stability": 0.125,
    "coastal_erosion": 0.125,
}

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


def _reshape_for_current_forecast(
    risk_data: gpd.GeoDataFrame, id_col: str, risk_cols_order: list[str]
) -> gpd.GeoDataFrame:
    """Reshape dataframe by adding a current/forecast column to distinguish identical rows."""
    # Identify risk and descriptive columns
    risk_cols = [col for col in risk_data.columns if col.endswith(("_current", "_forecast"))]
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
    melted["current_or_forecast"] = (
        melted["variable"]
        .str.extract(r"_(current|forecast)$")[0]
        .map({"current": "Current", "forecast": "Forecast"})
    )
    melted["variable"] = melted["variable"].str.replace(
        r"_(current|forecast)$", "", regex=True
    )

    # Pivot back so each risk variable becomes a column
    reshaped = melted.pivot_table(
        index=[id_col, "current_or_forecast", *descriptive_cols],
        columns="variable",
        values="value",
    ).reset_index()

    # Reorder risk columns based on original order
    reshaped = reshaped[[id_col, "current_or_forecast", *descriptive_cols, *risk_cols_order]]

    # Merge geometry back
    reshaped_gdf = reshaped.merge(geometry, on=id_col)
    return gpd.GeoDataFrame(reshaped_gdf, geometry="geometry", crs=risk_data.crs)


def _prepare_model_output(
    risk_data: gpd.GeoDataFrame,
    drop_cols: list[str],
    rename_map: dict[str, str],
    risk_cols_order: list[str],
) -> gpd.GeoDataFrame:
    """Perform standard cleaning operations on risk data to prepare for model output."""
    risk_data = risk_data.drop(columns=drop_cols)
    risk_data = risk_data.drop_duplicates(subset=["geometry"])
    risk_data = risk_data.rename(columns=rename_map)
    risk_data = risk_data.to_crs(data_cleaning.BNG_CRS)
    risk_data = _reshape_for_current_forecast(risk_data, "id", risk_cols_order)
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
    _infrastructure_layering(config, hazard_layers)


## HAZARD LAYERS


def _read_hazard_layers(config: model_config.Config) -> dict[str, gpd.GeoDataFrame]:
    """Read and clean hazard layers, and return them in a dictionary."""
    return {
        "Extreme Weather": gpd.read_file(
            config.paths.model_interim_output
            / file_paths.EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH
        ),
        "Flooding": gpd.read_file(
            config.paths.model_interim_output
            / file_paths.FLOOD_RISK_DIRECT_MODEL_INTERIM_OUTPUT_PATH
        ),
        "Ground Stability": gpd.read_file(
            config.paths.model_interim_output
            / file_paths.GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH
        ),
        "Coastal Erosion": gpd.read_file(
            config.paths.model_interim_output
            / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH
        ),
    }


## INFRASTRUCTURE-HAZARD LAYERING


def _infrastructure_layering(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Layer roads, rail, and other infrastructure with hazards."""
    _get_road_risk(config, hazard_layers)
    _get_rail_risk(config, hazard_layers)
    _get_other_risk(config, hazard_layers)


### ROAD


def _get_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Layer OS Open Roads and NoHAM with hazards to assign risk."""
    LOG.info("Calculating road risk...")
    if config.switches.all_roads:
        _os_open_road_risk(config, hazard_layers)
    if config.switches.noham_roads:
        _noham_road_risk(config, hazard_layers)
    LOG.info("Road risk calculation complete.")


#### OS Open Roads


def _os_open_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Intersect OS Road infrastructure with hazards, clean output, and write to file."""
    LOG.info("Layering OS Open Roads with hazard risk...")
    os_road = gpd.read_file(config.paths.model_input / file_paths.OS_ROAD_MODEL_INPUT_PATH)

    os_road_risk = _infrastructure_risk_intersect(os_road, hazard_layers)

    os_road_risk = _prepare_model_output(
        risk_data=os_road_risk,
        drop_cols=[],
        rename_map={"identifier": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
    )

    _split_csv_shapefile(
        config, os_road_risk, "id", pathlib.Path("Road") / "OS Roads" / "os_road_risk"
    )
    LOG.info("Finished layering OS Open Roads with hazard risk.")


#### NoHAM


def _noham_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Get NoHAM road risk and write to file.

    Intersect NoHAM 2023 and 2048 with hazards, calculate impact index, clean output, and write
    to file.
    """
    LOG.info("Layering NoHAM with hazard risk and calculating impact index...")
    noham = {}
    noham_risk_scenario = {}
    for scenario in functional_rules.SCENARIO_NAMES:
        noham[scenario] = gpd.read_file(
            config.paths.model_input
            / file_paths.NOHAM_FLOWS_MODEL_INPUT_PATH
            / f"noham_net_flows_{scenario}.gpkg"
        )
        noham_risk_scenario[scenario] = _infrastructure_risk_intersect(
            noham[scenario], hazard_layers
        )
        other_scenario = "forecast" if scenario == "current" else "current"
        drop_cols = [
            col
            for col in noham_risk_scenario[scenario].columns
            if col.endswith(f"_{other_scenario}")
        ]
        noham_risk_scenario[scenario] = noham_risk_scenario[scenario].drop(
            columns=drop_cols
        )
        noham_risk_scenario[scenario] = noham_risk_scenario[scenario].drop_duplicates(
            subset=["geometry"]
        )
        noham_risk_scenario[scenario].columns = [
            col.removesuffix(f"_{scenario}")
            for col in noham_risk_scenario[scenario].columns
        ]
        if scenario == "current":
            noham_risk_scenario[scenario]["current_or_forecast"] = "Current"
        else:
            noham_risk_scenario[scenario]["current_or_forecast"] = "Forecast"

    noham_risk_scenario["current"], noham_risk_scenario["forecast"] = (
        _noham_impact_index(
            noham_risk_scenario["current"],
            noham_risk_scenario["forecast"],
        )
    )

    risk_impact_cols = [*_HAZARD_RISK_COLS, *_NOHAM_IMPACT_COLS]
    non_risk_impact_cols = ["link_id", "current_or_forecast", "geometry"]

    # Remove suffixes from risk and impact columns
    noham_risk_scenario["current"] = noham_risk_scenario["current"].rename(
        columns=lambda c: (
            c.removesuffix("_current") if c.removesuffix("_current") in risk_impact_cols else c
        )
    )
    noham_risk_scenario["forecast"] = noham_risk_scenario["forecast"].rename(
        columns=lambda c: (
            c.removesuffix("_forecast")
            if c.removesuffix("_forecast") in risk_impact_cols
            else c
        )
    )

    # Concatenate
    noham_risk = pd.concat(
        [noham_risk_scenario["current"], noham_risk_scenario["forecast"]],
        ignore_index=True,
    )

    noham_risk = noham_risk[
        [*non_risk_impact_cols, *_HAZARD_RISK_COLS, *_NOHAM_IMPACT_COLS]
    ]

    noham_risk[risk_impact_cols] = noham_risk[risk_impact_cols].round(1)
    noham_risk = noham_risk.to_crs(data_cleaning.BNG_CRS)
    noham_risk = noham_risk.rename(columns={"link_id": "id"})
    noham_risk = noham_risk.rename(
        columns={col: f"{col}_score" for col in risk_impact_cols}
    )

    _split_csv_shapefile(
        config, noham_risk, "id", pathlib.Path("Road") / "NoHAM" / "noham_risk"
    )

    LOG.info("Finished layering NoHAM with hazard risk and calculating impact index.")


def _noham_impact_index(
    noham_c: gpd.GeoDataFrame,
    noham_f: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Normalise NoHAM demand, then calculate impact index."""
    user_classes = ["uc1", "uc2", "uc3", "uc4", "uc5"]

    # Normalise user class demand together
    noham_c, noham_f = _normalise_uc_demand(noham_c, noham_f, user_classes)

    # Normalise total demand separately
    noham_c, noham_f = _normalise_total_col(
        noham_c, noham_f, "all_vehs_total", "demand"
    )

    # Calculate impact scores
    noham_c, noham_f = _calculate_noham_impact(noham_c, noham_f, user_classes)

    impact_cols_c = [f"{uc}_impact_current" for uc in user_classes] + ["impact_current"]
    impact_cols_f = [f"{uc}_impact_forecast" for uc in user_classes] + ["impact_forecast"]

    noham_c, noham_f = _normalise_total_cols(
        noham_c, noham_f, impact_cols_c, impact_cols_f
    )

    noham_c = noham_c[
        ["link_id", "current_or_forecast", "geometry", *_HAZARD_RISK_COLS, *impact_cols_c]
    ]
    noham_f = noham_f[
        ["link_id", "current_or_forecast", "geometry", *_HAZARD_RISK_COLS, *impact_cols_f]
    ]

    return noham_c, noham_f


def _normalise_uc_demand(
    noham_c: pd.DataFrame, noham_f: pd.DataFrame, user_classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise NoHAM demand for each user class individually."""
    uc_total_cols = [f"{uc}_total" for uc in user_classes]
    combined_values = np.vstack(
        [noham_c[uc_total_cols].to_numpy(), noham_f[uc_total_cols].to_numpy()]
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(combined_values)
    for noham_data, scenario in [(noham_c, "current"), (noham_f, "forecast")]:
        scaled = scaler.transform(noham_data[uc_total_cols].values)
        noham_data[[f"{uc}_demand_{scenario}" for uc in user_classes]] = scaled
    return noham_c, noham_f


def _normalise_total_col(
    noham_c: pd.DataFrame, noham_f: pd.DataFrame, old_column: str, new_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise total demand for one column."""
    # Normalise all vehicles total separately
    combined_values = np.vstack(
        [
            noham_c[old_column].to_numpy().reshape(-1, 1),
            noham_f[old_column].to_numpy().reshape(-1, 1),
        ]
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(combined_values)
    for noham_data, scenario in [(noham_c, "current"), (noham_f, "forecast")]:
        scaled = scaler.transform(noham_data[old_column].to_numpy().reshape(-1, 1))
        noham_data[f"{new_column}_{scenario}"] = scaled
    return noham_c, noham_f


def _normalise_total_cols(
    noham_c: pd.DataFrame, noham_f: pd.DataFrame, cols_c: list[str], cols_f: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise total demand for several columns."""
    combined_values = np.vstack([noham_c[cols_c].to_numpy(), noham_f[cols_f].to_numpy()])
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(combined_values)
    for noham_data, cols in [(noham_c, cols_c), (noham_f, cols_f)]:
        scaled = scaler.transform(noham_data[cols].to_numpy())
        noham_data[cols] = scaled
    return noham_c, noham_f


def _calculate_noham_impact(
    noham_c: pd.DataFrame,
    noham_f: pd.DataFrame,
    user_classes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate NoHAM impact score for each user class, and for all vehicles."""
    # Calculate impact metric for each user class
    for uc in user_classes:
        for noham_data, scenario in [(noham_c, "current"), (noham_f, "forecast")]:
            noham_data[f"{uc}_impact_{scenario}"] = (
                noham_data[f"{uc}_demand_{scenario}"] * _IMPACT_WEIGHTS["demand"]
                + noham_data["flood_risk"] * _IMPACT_WEIGHTS["flood"]
                + noham_data["extreme_weather_risk"] * _IMPACT_WEIGHTS["extreme_weather"]
                + noham_data["ground_stability_risk"] * _IMPACT_WEIGHTS["ground_stability"]
                + noham_data["coastal_erosion_risk"] * _IMPACT_WEIGHTS["coastal_erosion"]
            )

    for noham_data, scenario in [(noham_c, "current"), (noham_f, "forecast")]:
        noham_data[f"impact_{scenario}"] = (
            noham_data[f"demand_{scenario}"] * _IMPACT_WEIGHTS["demand"]
            + noham_data["flood_risk"] * _IMPACT_WEIGHTS["flood"]
            + noham_data["extreme_weather_risk"] * _IMPACT_WEIGHTS["extreme_weather"]
            + noham_data["ground_stability_risk"] * _IMPACT_WEIGHTS["ground_stability"]
            + noham_data["coastal_erosion_risk"] * _IMPACT_WEIGHTS["coastal_erosion"]
        )

    return noham_c, noham_f


### RAIL


def _get_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Layer passenger rail and freight rail network with hazard to assign risk."""
    LOG.info("Calculating rail risk...")
    if config.switches.passenger_rail:
        _passenger_rail_risk(config, hazard_layers)
    if config.switches.freight_rail:
        _freight_rail_risk(config, hazard_layers)
    LOG.info("Rail risk calculation complete.")


#### Passenger Rail


def _passenger_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Intersect passenger rail network with hazard to assign risk, clean and write to file."""
    LOG.info("Layering passenger rail network with hazard risk...")
    passenger_rail_network = gpd.read_file(
        config.paths.model_input / file_paths.PASSENGER_RAIL_MODEL_INPUT_PATH
    )

    passenger_rail_network_risk = _infrastructure_risk_intersect(
        passenger_rail_network, hazard_layers
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
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Calculate freight rail risk and write to file.

    Intersect freight network with hazard risk, calculate impact index, clean and write to
    file.
    """
    LOG.info("Layering freight rail network with hazard risk and calculating impact index...")
    freight_rail_network = gpd.read_file(
        config.paths.model_input / file_paths.FREIGHT_DEMAND_MODEL_INPUT_PATH
    )

    freight_rail_network_risk = _infrastructure_risk_intersect(
        freight_rail_network, hazard_layers
    )

    freight_rail_network_risk = _freight_impact_index(freight_rail_network_risk)

    # Set the correct CRS
    freight_rail_network_risk = freight_rail_network_risk.set_crs(
        data_cleaning.BNG_CRS, allow_override=True
    )

    freight_rail_network_risk = _prepare_model_output(
        risk_data=freight_rail_network_risk,
        drop_cols=["dij_id", "distance", "demand_current", "demand_forecast"],
        rename_map={
            "osid": "id",
            "desc": "description",
            "phys_level": "physical_level",
            "rail_use": "railway_use",
            "track_rep": "track_representation",
        },
        risk_cols_order=[*_HAZARD_RISK_COLS, "impact"],
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


def _freight_impact_index(freight_rail_network_risk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate impact index using freight demand data and hazard risk."""
    freight_rail_network_risk = functional_rules.min_max_scaling_pair(
        freight_rail_network_risk, [("demand_current", "demand_forecast")]
    )

    freight_rail_network_risk = _calculate_freight_impact(freight_rail_network_risk)

    freight_rail_network_risk = functional_rules.min_max_scaling_pair(
        freight_rail_network_risk, [("impact_current", "impact_forecast")]
    )

    return gpd.GeoDataFrame(freight_rail_network_risk, geometry="geometry", crs="EPSG:4326")


def _calculate_freight_impact(freight_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite impact score for current and forecast years."""
    for scenario in functional_rules.SCENARIO_NAMES:
        freight_data[f"impact_{scenario}"] = (
            freight_data[f"demand_{scenario}"] * _IMPACT_WEIGHTS["demand"]
            + freight_data[f"flood_risk_{scenario}"] * _IMPACT_WEIGHTS["flood"]
            + freight_data[f"extreme_weather_risk_{scenario}"]
            * _IMPACT_WEIGHTS["extreme_weather"]
            + freight_data[f"ground_stability_risk_{scenario}"]
            * _IMPACT_WEIGHTS["ground_stability"]
            + freight_data[f"coastal_erosion_risk_{scenario}"]
            * _IMPACT_WEIGHTS["coastal_erosion"]
        )

    return freight_data


### OTHER


def _get_other_risk(  # noqa: C901
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Layer other infrastructure with hazards to assign risk."""
    LOG.info("Calculating risk for other infrastructure...")
    if config.switches.train_stations:
        _train_stations_risk(config, hazard_layers)
    if config.switches.charging_sites:
        _charging_sites_risk(config, hazard_layers)
    if config.switches.airports:
        _airports_risk(config, hazard_layers)
    if config.switches.bus_coach_stations:
        _bus_coach_stations_risk(config, hazard_layers)
    if config.switches.bus_stops:
        _bus_stops_risk(config, hazard_layers)
    if config.switches.tram_stations:
        _tram_stations_risk(config, hazard_layers)
    if config.switches.rapid_transport_stations:
        _rapid_transport_stations_risk(config, hazard_layers)
    if config.switches.ferry_terminals:
        _ferry_terminals_risk(config, hazard_layers)
    if config.switches.petrol_stations:
        _petrol_stations_risk(config, hazard_layers)
    if config.switches.national_cycle_network:
        _ncn_risk(config, hazard_layers)
    if config.switches.tram_network:
        _tram_network_risk(config, hazard_layers)
    if config.switches.rapid_transport_network:
        _rapid_transport_network_risk(config, hazard_layers)
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
) -> None:
    """Get train station risk and write to file.

    Buffer train stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering train stations with hazard risk...")
    train_stations = gpd.read_file(
        config.paths.model_input / file_paths.TRAIN_STATIONS_MODEL_INPUT_PATH
    )

    train_stations = _buffer_geometry(train_stations, _TRAIN_STATIONS_BUFFER_SIZE_M)

    train_stations_risk = _infrastructure_risk_intersect(train_stations, hazard_layers)

    train_stations_risk = _prepare_model_output(
        risk_data=train_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get EV charging site risk and write to file.

    Buffer charging sites, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering EV charging sites with hazard risk...")
    charging_sites = gpd.read_file(
        config.paths.model_input / file_paths.CHARGING_SITES_MODEL_INPUT_PATH
    )

    charging_sites = _buffer_geometry(charging_sites, _CHARGING_SITES_BUFFER_SIZE_M)

    charging_sites_risk = _infrastructure_risk_intersect(charging_sites, hazard_layers)

    charging_sites_risk = _prepare_model_output(
        risk_data=charging_sites_risk,
        drop_cols=[],
        rename_map={"devices": "installed_devices"},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get airport risk and write to file.

    Intersect airports with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering airports with hazard risk...")
    airports = gpd.read_file(
        config.paths.model_input / file_paths.AIRPORTS_MODEL_INPUT_PATH
    )

    airports_risk = _infrastructure_risk_intersect(airports, hazard_layers)

    airports_risk = _prepare_model_output(
        risk_data=airports_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get bus and coach station risk and write to file.

    Buffer bus and coach stations, then intersect with hazard risk, clean output, and write to
    file.
    """
    LOG.info("Layering bus and coach stations with hazard risk...")
    bus_coach_stations = gpd.read_file(
        config.paths.model_input / file_paths.BUS_COACH_STATIONS_MODEL_INPUT_PATH
    )

    bus_coach_stations = _buffer_geometry(
        bus_coach_stations, _BUS_COACH_STATIONS_BUFFER_SIZE_M
    )

    bus_coach_stations_risk = _infrastructure_risk_intersect(
        bus_coach_stations, hazard_layers
    )

    bus_coach_stations_risk = _prepare_model_output(
        risk_data=bus_coach_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Intersect bus stops with hazard risk, clean output, and write to file."""
    LOG.info("Layering bus stops with hazard risk...")
    bus_stops = gpd.read_file(
        config.paths.model_input / file_paths.BUS_STOPS_MODEL_INPUT_PATH
    )

    bus_stops_risk = _infrastructure_risk_intersect(bus_stops, hazard_layers)

    bus_stops_risk = _prepare_model_output(
        risk_data=bus_stops_risk,
        drop_cols=[],
        rename_map={"stop_id": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get tram station risk and write to file.

    Buffer tram stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering tram stations with hazard risk...")
    tram_stations = gpd.read_file(
        config.paths.model_input / file_paths.TRAM_STATIONS_MODEL_INPUT_PATH
    )

    tram_stations = _buffer_geometry(tram_stations, _TRAM_STATIONS_BUFFER_SIZE_M)

    tram_stations_risk = _infrastructure_risk_intersect(tram_stations, hazard_layers)

    tram_stations_risk = _prepare_model_output(
        risk_data=tram_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get rapid transport station risk and write to file.

    Buffer rapid transport stations, then intersect with hazard risk, clean output, and write
    to file.
    """
    LOG.info("Layering rapid transport stations with hazard risk...")
    rapid_transport_stations = gpd.read_file(
        config.paths.model_input / file_paths.RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH
    )

    rapid_transport_stations = _buffer_geometry(
        rapid_transport_stations, _RAPID_TRANSPORT_STATIONS_BUFFER_SIZE_M
    )

    rapid_transport_stations_risk = _infrastructure_risk_intersect(
        rapid_transport_stations, hazard_layers
    )

    rapid_transport_stations_risk = _prepare_model_output(
        risk_data=rapid_transport_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
    )

    _split_csv_shapefile(
        config,
        rapid_transport_stations_risk,
        "id",
        pathlib.Path("Other")
        / "Rapid Transport Stations"
        / "rapid_transport_stations_risk",
    )
    LOG.info("Finished layering rapid transport stations with hazard risk.")


#### Ferry Terminals


def _ferry_terminals_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Get ferry terminal risk and write to file.

    Buffer ferry terminals, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering ferry terminals with hazard risk...")
    ferry_terminals = gpd.read_file(
        config.paths.model_input / file_paths.FERRY_TERMINALS_MODEL_INPUT_PATH
    )

    ferry_terminals = _buffer_geometry(ferry_terminals, _FERRY_TERMINALS_BUFFER_SIZE_M)

    ferry_terminals_risk = _infrastructure_risk_intersect(
        ferry_terminals, hazard_layers
    )

    ferry_terminals_risk = _prepare_model_output(
        risk_data=ferry_terminals_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get petrol station risk and write to file.

    Buffer petrol stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering petrol stations with hazard risk...")
    petrol_stations = gpd.read_file(
        config.paths.model_input / file_paths.PETROL_STATIONS_MODEL_INPUT_PATH
    )

    petrol_stations = _buffer_geometry(petrol_stations, _PETROL_STATIONS_BUFFER_SIZE_M)

    petrol_stations_risk = _infrastructure_risk_intersect(
        petrol_stations, hazard_layers
    )

    petrol_stations_risk = _prepare_model_output(
        risk_data=petrol_stations_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get NCN risk and write to file.

    Intersect National Cycle Network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering National Cycle Network with hazard risk...")
    ncn = gpd.read_file(
        config.paths.model_input / file_paths.NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH
    )

    ncn_risk = _infrastructure_risk_intersect(ncn, hazard_layers)

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
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get tram network risk and write to file.

    Intersect tram network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering tram network with hazard risk...")
    tram_network = gpd.read_file(
        config.paths.model_input / file_paths.TRAM_NETWORK_MODEL_INPUT_PATH
    )

    tram_risk = _infrastructure_risk_intersect(tram_network, hazard_layers)

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
        risk_cols_order=_HAZARD_RISK_COLS,
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
) -> None:
    """Get rapid transport network risk and write to file.

    Intersect rapid transport network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering rapid transport network with hazard risk...")
    rapid_transport = gpd.read_file(
        config.paths.model_input / file_paths.RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH
    )

    rapid_transport_risk = _infrastructure_risk_intersect(
        rapid_transport, hazard_layers
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
        risk_cols_order=_HAZARD_RISK_COLS,
    )

    _split_csv_shapefile(
        config,
        rapid_transport_risk,
        "id",
        pathlib.Path("Other") / "Rapid Transport Network" / "rapid_transport_network_risk",
    )
    LOG.info("Finished layering rapid transport network with hazard risk.")
