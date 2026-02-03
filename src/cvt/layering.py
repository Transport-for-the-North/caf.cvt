"""Intersect infrastructure with hazard layers to assign risk scores to infrastructure."""

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import data_cleaning, functional_rules, model_config
from data_cleaning import _BNG_CRS

# GENERAL FUNCTIONS


def _infrastructure_risk_intersect(
    gdf: gpd.GeoDataFrame, hazards_dict: dict[str, gpd.GeoDataFrame]
) -> gpd.GeoDataFrame:
    """Intersect infrastructure with hazard risk layers.

    Spatailly combine infrastructure with hazard layers using an intersection spatial join,
    then calculate hazard risk score as the max risk value of the intersection. Return merged
    GeoDataFrame with hazard risk columns added.
    """
    gdf_with_risk = gdf.copy()

    for _hazard, hazard_gdf in hazards_dict.items():
        # Spatial join to find intersections with hazards
        hazard_gdf_match = hazard_gdf.to_crs(gdf_with_risk.crs)  # Match CRS
        intersections = gpd.sjoin(
            gdf_with_risk, hazard_gdf_match, how="left", predicate="intersects"
        )

        # Identify risk columns
        risk_columns = hazard_gdf_match.columns[
            hazard_gdf_match.columns.str.contains("risk", case=False)
        ]

        # Calculate hazard risk score per infrastructure segment as max value of intersection
        agg = intersections.groupby(intersections.index)[risk_columns].max()

        # Merge back into main DataFrame
        gdf_with_risk = gdf_with_risk.join(agg, how="left")

    return gdf_with_risk.fillna(0)


def _reshape_for_current_forecast(
    gdf: gpd.GeoDataFrame, id_col: str, risk_cols_order: list[str]
) -> gpd.GeoDataFrame:
    """Reshape dataframe by adding a current/forecast column to distinguish identical rows."""
    # Identify risk and descriptive columns
    risk_cols = [col for col in gdf.columns if col.endswith(("_current", "_forecast"))]
    id_cols = [id_col]
    descriptive_cols = [
        col
        for col in gdf.columns
        if col not in risk_cols and col not in id_cols and col != "geometry"
    ]

    # Separate geometry for later
    geometry = gdf[[id_col, "geometry"]].copy()

    # Melt only risk columns
    melted = gdf.melt(
        id_vars=id_cols + descriptive_cols,
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
        index=[*id_cols, "current_or_forecast", *descriptive_cols],
        columns="variable",
        values="value",
    ).reset_index()

    # Reorder risk columns based on original order
    reshaped = reshaped[[*id_cols, "current_or_forecast", *descriptive_cols, *risk_cols_order]]

    # Merge geometry back
    reshaped_gdf = reshaped.merge(geometry, on=id_col)
    return gpd.GeoDataFrame(reshaped_gdf, geometry="geometry", crs=gdf.crs)


def _prepare_model_output(
    gdf: gpd.GeoDataFrame,
    drop_cols: list[str],
    desc_cols: list[str],
    rename_map: dict[str, str],
    risk_cols_order: list[str],
) -> gpd.GeoDataFrame:
    """Perform standard cleaning operations on GeoDataFrame to prepare for model output."""
    gdf = gdf.drop(columns=drop_cols)
    gdf = gdf.drop_duplicates(subset=["geometry"])
    gdf = gdf.rename(columns=rename_map)
    gdf[desc_cols] = gdf[desc_cols].replace(0, "N/A")
    gdf = gdf.to_crs(epsg=_BNG_CRS)
    gdf = _reshape_for_current_forecast(gdf, "id", risk_cols_order)
    gdf[risk_cols_order] = gdf[risk_cols_order].round(1)
    return gdf.rename(columns={col: f"{col}_score" for col in risk_cols_order})


def _split_csv_shapefile(
    config: model_config.Config,
    gdf: gpd.GeoDataFrame,
    id_col: str,
    inf_type: str,
    folder: str,
    filename: str,
) -> None:
    """Split GeoDataFrame into a CSV and Shapefile, then write to file.

    Separates GeoDataFrame into a dataframe with an ID and attributes, and a Shapefile with an
    ID and geometry, then writes them to file.
    """
    # Separate spatial and attribute data
    spatial_gdf = gdf[[id_col, "geometry"]].copy()
    attribute_df = gdf.drop(columns=["geometry"])

    # Save to file
    data_cleaning.write_to_file(
        spatial_gdf, config.paths.model_output / inf_type / folder / f"{filename}.shp"
    )
    data_cleaning.write_to_file(
        attribute_df, config.paths.model_output / inf_type / folder / f"{filename}.csv"
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

    risk_cols = [
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
        "collapsible_deposits_risk",
        "compressible_ground_risk",
        "landslides_risk",
        "running_sand_risk",
        "shrink_swell_risk",
        "soluble_rocks_risk",
        "shrink_swell_geoclimate_risk",
        "ground_stability_risk",
        # Coastal Erosion risk columns
        "coastal_erosion_risk",
    ]

    impact_weights = {
        "demand": 0.5,  # Weight demand as half of impact score
        "flood": 0.125,  # Weight hazards as 0.125 each to make up half
        "extreme_weather": 0.125,
        "ground_stability": 0.125,
        "coastal_erosion": 0.125,
    }

    _infrastructure_layering(config, hazard_layers, risk_cols, impact_weights)


## HAZARD LAYERS


def _read_hazard_layers(config: model_config.Config) -> dict[str, gpd.GeoDataFrame]:
    """Read and clean hazard layers, and return them in a dictionary."""
    return {
        "Extreme Weather": gpd.read_file(
            config.paths.model_interim_output
            / "TfN Extreme Weather Risk"
            / "tfn_extreme_weather_risk.gpkg"
        ),
        "Flooding": gpd.read_file(
            config.paths.model_interim_output / "TfN Flood Risk" / "tfn_flood_risk.gpkg"
        ),
        "Ground Stability": gpd.read_file(
            config.paths.model_interim_output
            / "TfN Ground Stability Risk"
            / "tfn_ground_stability_risk.gpkg"
        ),
        "Coastal Erosion": gpd.read_file(
            config.paths.model_interim_output
            / "TfN Coastal Erosion Risk"
            / "tfn_coastal_erosion_risk.gpkg"
        ),
    }


## INFRASTRUCTURE-HAZARD LAYERING


def _infrastructure_layering(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
    impact_weights: dict[str, float],
) -> None:
    """Layer roads, rail, and other infrastructure with hazards."""
    _get_road_risk(config, hazard_layers, risk_cols, impact_weights)
    _get_rail_risk(config, hazard_layers, risk_cols, impact_weights)
    _get_other_risk(config, hazard_layers, risk_cols)


### ROAD


def _get_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
    impact_weights: dict[str, float],
) -> None:
    """Layer OS Open Roads and NoHAM with hazards to assign risk."""
    _os_open_road_risk(config, hazard_layers, risk_cols)
    _noham_road_risk(config, hazard_layers, risk_cols, impact_weights)


#### OS Open Roads


def _os_open_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Intersect OS Road infrastructure with hazards, clean output, and write to file."""
    tfn_os_road = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Road"
        / "TfN OS Road"
        / "tfn_os_road.gpkg"
    )

    tfn_os_road_risk = _infrastructure_risk_intersect(tfn_os_road, hazard_layers)

    tfn_os_road_risk = _prepare_model_output(
        gdf=tfn_os_road_risk,
        drop_cols=[],
        desc_cols=["road_number", "name", "function"],
        rename_map={"identifier": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config, tfn_os_road_risk, "id", "Road", "OS Roads", "tfn_os_road_risk"
    )


#### NoHAM


def _noham_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
    impact_weights: dict[str, float],
) -> None:
    """Get NoHAM road risk and write to file.

    Intersect NoHAM 2023 and 2048 with hazards, calculate impact index, clean output, and write
    to file.
    """
    tfn_noham = {}
    tfn_noham_risk_scenario = {}
    for year, scenario in {"2023": "current", "2048": "forecast"}.items():
        tfn_noham[scenario] = gpd.read_file(
            config.paths.model_input
            / "Impact"
            / "TfN NoHAM Flows"
            / year
            / f"tfn_noham_net_flows_{scenario}.gpkg"
        )
        tfn_noham_risk_scenario[scenario] = _infrastructure_risk_intersect(
            tfn_noham[scenario], hazard_layers
        )
        other_scenario = "forecast" if scenario == "current" else "current"
        drop_cols = [
            col
            for col in tfn_noham_risk_scenario[scenario].columns
            if col.endswith(f"_{other_scenario}")
        ]
        tfn_noham_risk_scenario[scenario] = tfn_noham_risk_scenario[scenario].drop(
            columns=drop_cols
        )
        tfn_noham_risk_scenario[scenario] = tfn_noham_risk_scenario[scenario].drop_duplicates(
            subset=["geometry"]
        )
        tfn_noham_risk_scenario[scenario].columns = [
            col.removesuffix(f"_{scenario}")
            for col in tfn_noham_risk_scenario[scenario].columns
        ]
        if scenario == "current":
            tfn_noham_risk_scenario[scenario]["current_or_forecast"] = "Current"
        else:
            tfn_noham_risk_scenario[scenario]["current_or_forecast"] = "Forecast"

    tfn_noham_risk_scenario["current"], tfn_noham_risk_scenario["forecast"] = (
        _noham_impact_index(
            tfn_noham_risk_scenario["current"],
            tfn_noham_risk_scenario["forecast"],
            impact_weights,
            risk_cols,
        )
    )

    # Remove suffixes from risk and impact columns
    tfn_noham_risk_scenario["current"].columns = [
        col.removesuffix("_current") for col in tfn_noham_risk_scenario["current"].columns
    ]
    tfn_noham_risk_scenario["forecast"].columns = [
        col.removesuffix("_forecast") for col in tfn_noham_risk_scenario["forecast"].columns
    ]

    # Concatenate
    tfn_noham_risk = pd.concat(
        [tfn_noham_risk_scenario["current"], tfn_noham_risk_scenario["forecast"]],
        ignore_index=True,
    )

    noham_impact_cols = [
        "uc1_impact",
        "uc2_impact",
        "uc3_impact",
        "uc4_impact",
        "uc5_impact",
        "impact",
    ]
    tfn_noham_risk = tfn_noham_risk[
        ["link_id", "current_or_forecast", "geometry", *risk_cols, *noham_impact_cols]
    ]

    cols_to_round = [
        col for col in tfn_noham_risk.columns if col not in ["link_id", "geometry"]
    ]
    tfn_noham_risk[cols_to_round] = tfn_noham_risk[cols_to_round].round(1)
    tfn_noham_risk = tfn_noham_risk.to_crs(epsg=_BNG_CRS)
    tfn_noham_risk = tfn_noham_risk.rename(columns={"link_id": "id"})
    tfn_noham_risk = tfn_noham_risk.rename(
        columns={col: f"{col}_score" for col in cols_to_round}
    )

    _split_csv_shapefile(config, tfn_noham_risk, "id", "Road", "NoHAM", "tfn_noham_risk")


def _noham_impact_index(
    tfn_noham_c: gpd.GeoDataFrame,
    tfn_noham_f: gpd.GeoDataFrame,
    impact_weights: dict[str, float],
    risk_cols: list[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Normalise NoHAM demand, then calculate impact index."""
    user_classes = ["uc1", "uc2", "uc3", "uc4", "uc5"]

    # Normalise user class demand together
    tfn_noham_c, tfn_noham_f = _normalise_uc_demand(tfn_noham_c, tfn_noham_f, user_classes)

    # Normalise total demand separately
    tfn_noham_c, tfn_noham_f = _normalise_total_col(
        tfn_noham_c, tfn_noham_f, "all_vehs_total", "demand"
    )

    # Calculate impact scores
    tfn_noham_c, tfn_noham_f = _calculate_noham_impact(
        tfn_noham_c, tfn_noham_f, user_classes, impact_weights
    )

    impact_cols_c = [f"{uc}_impact_current" for uc in user_classes] + ["impact_current"]
    impact_cols_f = [f"{uc}_impact_forecast" for uc in user_classes] + ["impact_forecast"]

    tfn_noham_c, tfn_noham_f = _normalise_total_cols(
        tfn_noham_c, tfn_noham_f, impact_cols_c, impact_cols_f
    )

    tfn_noham_c = tfn_noham_c[
        ["link_id", "current_or_forecast", "geometry", *risk_cols, *impact_cols_c]
    ]
    tfn_noham_f = tfn_noham_f[
        ["link_id", "current_or_forecast", "geometry", *risk_cols, *impact_cols_f]
    ]

    return tfn_noham_c, tfn_noham_f


def _normalise_uc_demand(
    df_c: pd.DataFrame, df_f: pd.DataFrame, user_classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise NoHAM demand for each user class individually."""
    uc_total_cols = [f"{uc}_total" for uc in user_classes]
    combined_values = np.vstack(
        [df_c[uc_total_cols].to_numpy(), df_f[uc_total_cols].to_numpy()]
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(combined_values)
    for df, scenario in [(df_c, "current"), (df_f, "forecast")]:
        scaled = scaler.transform(df[uc_total_cols].values)
        df[[f"{uc}_demand_{scenario}" for uc in user_classes]] = scaled
    return df_c, df_f


def _normalise_total_col(
    df_c: pd.DataFrame, df_f: pd.DataFrame, old_column: str, new_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise total demand for one column."""
    # Normalise all vehicles total separately
    combined_values = np.vstack(
        [
            df_c[old_column].to_numpy().reshape(-1, 1),
            df_f[old_column].to_numpy().reshape(-1, 1),
        ]
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(combined_values)
    for df, scenario in [(df_c, "current"), (df_f, "forecast")]:
        scaled = scaler.transform(df[old_column].to_numpy().reshape(-1, 1))
        df[f"{new_column}_{scenario}"] = scaled
    return df_c, df_f


def _normalise_total_cols(
    df_c: pd.DataFrame, df_f: pd.DataFrame, cols_c: list[str], cols_f: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise total demand for several columns."""
    combined_values = np.vstack([df_c[cols_c].to_numpy(), df_f[cols_f].to_numpy()])
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(combined_values)
    for df, cols in [(df_c, cols_c), (df_f, cols_f)]:
        scaled = scaler.transform(df[cols].to_numpy())
        df[cols] = scaled
    return df_c, df_f


def _calculate_noham_impact(
    df_c: pd.DataFrame,
    df_f: pd.DataFrame,
    user_classes: list[str],
    impact_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate NoHAM impact score for each user class, and for all vehicles."""
    # Calculate impact metric for each user class
    for uc in user_classes:
        for df, scenario in [(df_c, "current"), (df_f, "forecast")]:
            df[f"{uc}_impact_{scenario}"] = (
                df[f"{uc}_demand_{scenario}"] * impact_weights["demand"]
                + df["flood_risk"] * impact_weights["flood"]
                + df["extreme_weather_risk"] * impact_weights["extreme_weather"]
                + df["ground_stability_risk"] * impact_weights["ground_stability"]
                + df["coastal_erosion_risk"] * impact_weights["coastal_erosion"]
            )

    for df, scenario in [(df_c, "current"), (df_f, "forecast")]:
        df[f"impact_{scenario}"] = (
            df[f"demand_{scenario}"] * impact_weights["demand"]
            + df["flood_risk"] * impact_weights["flood"]
            + df["extreme_weather_risk"] * impact_weights["extreme_weather"]
            + df["ground_stability_risk"] * impact_weights["ground_stability"]
            + df["coastal_erosion_risk"] * impact_weights["coastal_erosion"]
        )

    return df_c, df_f


### RAIL


def _get_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
    impact_weights: dict[str, float],
) -> None:
    """Layer passenger rail and freight rail network with hazard to assign risk."""
    _passenger_rail_risk(config, hazard_layers, risk_cols)
    _freight_rail_risk(config, hazard_layers, risk_cols, impact_weights)


#### Passenger Rail


def _passenger_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Intersect passenger rail network with hazard to assign risk, clean and write to file."""
    tfn_rail_network = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Rail"
        / "TfN OS Passenger Rail"
        / "tfn_pass_rail_links.gpkg"
    )

    tfn_rail_network_risk = _infrastructure_risk_intersect(tfn_rail_network, hazard_layers)

    tfn_rail_network_risk = _prepare_model_output(
        gdf=tfn_rail_network_risk,
        drop_cols=[],
        desc_cols=[
            "description",
            "structure",
            "physical_level",
            "railway_use",
            "track_representation",
        ],
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
        tfn_rail_network_risk,
        "id",
        "Rail",
        "Passenger Rail",
        "tfn_passenger_rail_network_risk",
    )


#### Freight Rail


def _freight_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
    impact_weights: dict[str, float],
) -> None:
    """Calculate freight rail risk and write to file.

    Intersect freight network with hazard risk, calculate impact index, clean and write to
    file.
    """
    tfn_freight_network = gpd.read_file(
        config.paths.model_input
        / "Impact"
        / "TfN Freight Flows"
        / "tfn_freight_network_demand.gpkg"
    )

    tfn_freight_network_risk = _infrastructure_risk_intersect(
        tfn_freight_network, hazard_layers
    )

    tfn_freight_network_risk = _freight_impact_index(tfn_freight_network_risk, impact_weights)

    # Set the correct CRS
    tfn_freight_network_risk = tfn_freight_network_risk.set_crs(
        epsg=_BNG_CRS, allow_override=True
    )

    tfn_freight_network_risk = _prepare_model_output(
        gdf=tfn_freight_network_risk,
        drop_cols=["dij_id", "distance", "demand_current", "demand_forecast"],
        desc_cols=[
            "description",
            "structure",
            "physical_level",
            "railway_use",
            "track_representation",
        ],
        rename_map={
            "osid": "id",
            "desc": "description",
            "phys_level": "physical_level",
            "rail_use": "railway_use",
            "track_rep": "track_representation",
        },
        risk_cols_order=[*risk_cols, "impact"],
    )

    _split_csv_shapefile(
        config,
        tfn_freight_network_risk,
        "id",
        "Rail",
        "Freight Rail",
        "tfn_freight_rail_network_risk",
    )


def _freight_impact_index(
    tfn_freight_network_risk: gpd.GeoDataFrame, impact_weights: dict[str, float]
) -> gpd.GeoDataFrame:
    """Calculate impact index using freight demand data and hazard risk."""
    tfn_freight_network_risk = functional_rules.min_max_scaling_pair(
        tfn_freight_network_risk, [("2022_23_total", "2050_51 sc2_total")]
    )

    tfn_freight_network_risk = tfn_freight_network_risk.rename(
        columns={"2022_23_total": "demand_current", "2050_51 sc2_total": "demand_forecast"}
    )

    tfn_freight_network_risk = _calculate_freight_impact(
        tfn_freight_network_risk, impact_weights
    )

    tfn_freight_network_risk = functional_rules.min_max_scaling_pair(
        tfn_freight_network_risk, [("impact_current", "impact_forecast")]
    )

    return gpd.GeoDataFrame(tfn_freight_network_risk, geometry="geometry", crs="EPSG:4326")


def _calculate_freight_impact(
    df: pd.DataFrame, impact_weights: dict[str, float]
) -> pd.DataFrame:
    """Calculate composite impact score for current and forecast years."""
    for scenario in ["current", "forecast"]:
        df[f"impact_{scenario}"] = (
            df[f"demand_{scenario}"] * impact_weights["demand"]
            + df[f"flood_risk_{scenario}"] * impact_weights["flood"]
            + df[f"extreme_weather_risk_{scenario}"] * impact_weights["extreme_weather"]
            + df[f"ground_stability_risk_{scenario}"] * impact_weights["ground_stability"]
            + df[f"coastal_erosion_risk_{scenario}"] * impact_weights["coastal_erosion"]
        )

    return df


### OTHER


def _get_other_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Layer other infrastructure with hazards to assign risk."""
    _train_stations_risk(config, hazard_layers, risk_cols)
    _ev_charging_sites_risk(config, hazard_layers, risk_cols)
    _airports_risk(config, hazard_layers, risk_cols)
    _bus_coach_stations_risk(config, hazard_layers, risk_cols)
    _bus_stops_risk(config, hazard_layers, risk_cols)
    _tram_stations_risk(config, hazard_layers, risk_cols)
    _rapid_transport_stations_risk(config, hazard_layers, risk_cols)
    _ferry_terminals_risk(config, hazard_layers, risk_cols)
    _petrol_stations_risk(config, hazard_layers, risk_cols)
    _ncn_risk(config, hazard_layers, risk_cols)
    _tram_network_risk(config, hazard_layers, risk_cols)
    _rapid_transport_network_risk(config, hazard_layers, risk_cols)


def _buffer_geometry(gdf: gpd.GeoDataFrame, buffer_size_m: int) -> gpd.GeoDataFrame:
    """Buffers the geometries of a given GeoDataFrame to a given size in metres."""
    gdf = gdf.to_crs(epsg=_BNG_CRS)
    gdf["geometry"] = gdf.buffer(buffer_size_m)
    return gdf


#### Train Stations


def _train_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get train station risk and write to file.

    Buffer train stations, then intersect with hazard risk, clean output, and write to file.
    """
    tfn_train_stations = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Train Stations"
        / "tfn_train_stations.gpkg"
    )

    tfn_train_stations = _buffer_geometry(tfn_train_stations, 100)

    tfn_train_stations_risk = _infrastructure_risk_intersect(tfn_train_stations, hazard_layers)

    tfn_train_stations_risk = _prepare_model_output(
        gdf=tfn_train_stations_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tfn_train_stations_risk,
        "id",
        "Other",
        "Train Stations",
        "tfn_train_stations_risk",
    )


#### EV Charging Sites


def _ev_charging_sites_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get EV charging site risk and write to file.

    Buffer charging sites, then intersect with hazard risk, clean output, and write to file.
    """
    tfn_chg_sites = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN EV Charging Sites"
        / "tfn_chg_sites.gpkg"
    )

    tfn_chg_sites = _buffer_geometry(tfn_chg_sites, 25)

    tfn_chg_sites_risk = _infrastructure_risk_intersect(tfn_chg_sites, hazard_layers)

    tfn_chg_sites_risk = _prepare_model_output(
        gdf=tfn_chg_sites_risk,
        drop_cols=[],
        desc_cols=["name", "speed"],
        rename_map={"devices": "installed_devices"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config, tfn_chg_sites_risk, "id", "Other", "EV Charging Sites", "tfn_chg_sites_risk"
    )


#### Airports


def _airports_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get airport risk and write to file.

    Intersect airports with hazard risk, clean output, and write to file.
    """
    tfn_airports = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Airports"
        / "tfn_airports.gpkg"
    )

    tfn_airports_risk = _infrastructure_risk_intersect(tfn_airports, hazard_layers)

    tfn_airports_risk = _prepare_model_output(
        gdf=tfn_airports_risk,
        drop_cols=[],
        desc_cols=["name"],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config, tfn_airports_risk, "id", "Other", "Airports", "tfn_airports_risk"
    )


#### Bus and Coach Stations


def _bus_coach_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get bus and coach station risk and write to file.

    Buffer bus and coach stations, then intersect with hazard risk, clean output, and write to
    file.
    """
    tfn_bus_coach_stations = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Bus Coach Stations"
        / "tfn_bus_coach_stations.gpkg"
    )

    tfn_bus_coach_stations = _buffer_geometry(tfn_bus_coach_stations, 50)

    tfn_bus_coach_stations_risk = _infrastructure_risk_intersect(
        tfn_bus_coach_stations, hazard_layers
    )

    tfn_bus_coach_stations_risk = _prepare_model_output(
        gdf=tfn_bus_coach_stations_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tfn_bus_coach_stations_risk,
        "id",
        "Other",
        "Bus and Coach Stations",
        "tfn_bus_coach_stations_risk",
    )


#### Bus Stops


def _bus_stops_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Intersect bus stops with hazard risk, clean output, and write to file."""
    tfn_bus_stops = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Bus Stops"
        / "tfn_bus_stops.gpkg"
    )

    tfn_bus_stops_risk = _infrastructure_risk_intersect(tfn_bus_stops, hazard_layers)

    tfn_bus_stops_risk = _prepare_model_output(
        gdf=tfn_bus_stops_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={"stop_id": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config, tfn_bus_stops_risk, "id", "Other", "Bus Stops", "tfn_bus_stops_risk"
    )


#### Tram Stations


def _tram_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get tram station risk and write to file.

    Buffer tram stations, then intersect with hazard risk, clean output, and write to file.
    """
    tfn_tram_stations = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Tram Stations"
        / "tfn_tram_stations.gpkg"
    )

    tfn_tram_stations = _buffer_geometry(tfn_tram_stations, 25)

    tfn_tram_stations_risk = _infrastructure_risk_intersect(tfn_tram_stations, hazard_layers)

    tfn_tram_stations_risk = _prepare_model_output(
        gdf=tfn_tram_stations_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tfn_tram_stations_risk,
        "id",
        "Other",
        "Tram Stations",
        "tfn_tram_stations_risk",
    )


#### Rapid Transport Stations


def _rapid_transport_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get rapid transport station risk and write to file.

    Buffer rapid transport stations, then interrsect with hazard risk, clean output, and write
    to file.
    """
    tfn_metro_stations = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Rapid Transport Stations"
        / "tfn_rapid_transport_stations.gpkg"
    )

    tfn_metro_stations = _buffer_geometry(tfn_metro_stations, 50)

    tfn_metro_stations_risk = _infrastructure_risk_intersect(tfn_metro_stations, hazard_layers)

    tfn_metro_stations_risk = _prepare_model_output(
        gdf=tfn_metro_stations_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tfn_metro_stations_risk,
        "id",
        "Other",
        "Rapid Transport Stations",
        "tfn_rapid_transport_stations_risk",
    )


#### Ferry Terminals


def _ferry_terminals_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get ferry terminal risk and write to file.

    Buffer ferry terminals, then intersect with hazard risk, clean output, and write to file.
    """
    tfn_ferry_terminals = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Ferry Terminals"
        / "tfn_ferry_terminals.gpkg"
    )

    tfn_ferry_terminals = _buffer_geometry(tfn_ferry_terminals, 50)

    tfn_ferry_terminals_risk = _infrastructure_risk_intersect(
        tfn_ferry_terminals, hazard_layers
    )

    tfn_ferry_terminals_risk = _prepare_model_output(
        gdf=tfn_ferry_terminals_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tfn_ferry_terminals_risk,
        "id",
        "Other",
        "Ferry Terminals",
        "tfn_ferry_terminals_risk",
    )


#### Petrol Stations


def _petrol_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get petrol station risk and write to file.

    Buffer petrol stations, then intersect with hazard risk, clean output, and write to file.
    """
    tfn_petrol_stations = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Petrol Stations"
        / "tfn_petrol_stations.gpkg"
    )

    tfn_petrol_stations = _buffer_geometry(tfn_petrol_stations, 50)

    tfn_petrol_stations_risk = _infrastructure_risk_intersect(
        tfn_petrol_stations, hazard_layers
    )

    tfn_petrol_stations_risk = _prepare_model_output(
        gdf=tfn_petrol_stations_risk,
        drop_cols=[],
        desc_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tfn_petrol_stations_risk,
        "id",
        "Other",
        "Petrol Stations",
        "tfn_petrol_stations_risk",
    )


#### National Cycle Network


def _ncn_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get NCN risk and write to file.

    Intersect National Cycle Network with hazard risk, clean output, then write to file.
    """
    tfn_ncn = gpd.read_file(
        config.paths.model_input / "Infrastructure" / "Other" / "TfN NCN" / "tfn_ncn.gpkg"
    )

    tfn_ncn_risk = _infrastructure_risk_intersect(tfn_ncn, hazard_layers)

    tfn_ncn_risk = _prepare_model_output(
        gdf=tfn_ncn_risk,
        drop_cols=[],
        desc_cols=[
            "description",
            "greenway",
            "route_type",
            "route_number",
            "link_number",
            "surface",
            "quality",
            "lighting",
            "road_class",
        ],
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
        config, tfn_ncn_risk, "id", "Other", "National Cycle Network", "tfn_ncn_risk"
    )


#### Tram Network


def _tram_network_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get tram network risk and write to file.

    Intersect tram network with hazard risk, clean output, then write to file.
    """
    tfn_tram_network = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN OS Tram Network"
        / "tfn_os_tram_links.gpkg"
    )

    tfn_tram_risk = _infrastructure_risk_intersect(tfn_tram_network, hazard_layers)

    tfn_tram_risk = _prepare_model_output(
        gdf=tfn_tram_risk,
        drop_cols=[],
        desc_cols=[
            "description",
            "structure",
            "physical_level",
            "railway_use",
            "track_representation",
        ],
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
        config, tfn_tram_risk, "id", "Other", "Tram Network", "tfn_tram_links_risk"
    )


#### Rapid Transport Network


def _rapid_transport_network_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[str],
) -> None:
    """Get rapid transport network risk and write to file.

    Intersect rapid transport network with hazard risk, clean output, then write to file.
    """
    tfn_rapid_transport = gpd.read_file(
        config.paths.model_input
        / "Infrastructure"
        / "Other"
        / "TfN Rapid Transport Network"
        / "tfn_rapid_transport_links.gpkg"
    )

    tfn_rapid_transport_risk = _infrastructure_risk_intersect(
        tfn_rapid_transport, hazard_layers
    )

    tfn_rapid_transport_risk = _prepare_model_output(
        gdf=tfn_rapid_transport_risk,
        drop_cols=[],
        desc_cols=[
            "description",
            "structure",
            "physical_level",
            "railway_use",
            "track_representation",
        ],
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
        tfn_rapid_transport_risk,
        "id",
        "Other",
        "Rapid Transport Network",
        "tfn_rapid_transport_risk",
    )
