"""Module for cleaning hazard datasets."""

# LOAD LIBRARIES
import base64
import gc
import json
import logging
import pathlib
import sys
import time
import warnings

import fiona
import geopandas as gpd
import pandas as pd
import requests
import xarray as xr

from caf.cvt import cleaning_funcs, definitions, model_config

# LOGGING

LOG = logging.getLogger(__name__)

# MODULE CONSTANTS

_SUCCESS_STATUS_CODE = 200

_CHESS_SCAPE_SCENARIOS = {
    definitions.Scenarios.CURRENT: {
        "rcp": "rcp85",
        "period": range(2001, 2021),
    },
    f"{definitions.Scenarios.FUTURE}_2deg": {
        "rcp": "rcp60",
        "period": range(2061, 2081),
    },
    f"{definitions.Scenarios.FUTURE}_4deg": {
        "rcp": "rcp85",
        "period": range(2061, 2081),
    }
}

_MISSING_YEAR_MONTH = (2080, 12)

# WARNING FILTERS

# This warning is raised by xarray when reading the .nc wind speed files. It does not change
# the data at all, so we can safely ignore it. It is likely caused by a numpy version mismatch.
# Eventually this warning should be addressed, but for now it is safe to ignore.
warnings.filterwarnings(
    "ignore",
    message="numpy.ndarray size changed, may indicate binary incompatibility.*",
    category=RuntimeWarning,
)

## HAZARDS


def clean_hazards(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """
    Clean hazard data ready for analysis.

    Use config switches to clean the relevant hazard datasets, covering extreme weather,
    flooding, ground stability, and coastal erosion.

    Parameters
    ----------
    config: model_config.Config
        The configuration object containing switches and paths.
    boundary: gpd.GeoDataFrame
        The spatial boundary to which hazard data should be clipped.

    Returns
    -------
    None
    """
    LOG.info("Cleaning hazard data...")
    if config.switches.extreme_weather:
        _clean_extreme_weather(config, boundary)
    if config.switches.flooding:
        _clean_flooding(config, boundary)
    if config.switches.ground_stability:
        _clean_ground_stability(config, boundary)
    if config.switches.coastal_erosion:
        _clean_coastal_erosion(config, boundary)

    if (
        not config.switches.extreme_weather
        and not config.switches.flooding
        and not config.switches.ground_stability
        and not config.switches.coastal_erosion
    ):
        LOG.warning("All hazard data is empty. Ending model running.")
        sys.exit()
    LOG.info("Finished cleaning hazard data.")


### EXTREME WEATHER


def _clean_extreme_weather(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all extreme weather datasets ready for analysis."""
    LOG.info("Cleaning extreme weather data...")

    ceda_token = _get_ceda_token(
        url=config.parameters.chess_scape.token_url,
        username=config.parameters.chess_scape.username,
        password=config.parameters.chess_scape.password
    )

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {ceda_token}"})

    _clean_chess_scape(
        config=config,
        boundary=boundary,
        session=session,
    )

    LOG.info("Finished cleaning extreme weather data.")



def _get_ceda_token(url: str, username: str, password: str) -> str:
    """Get CEDA access token for authentication to the API."""
    LOG.info("Requesting CEDA API access token")
    token = base64.b64encode(f"{username}:{password}".encode()).decode('ascii')
    headers = {
        "Authorization": f"Basic {token}"
    }
    try:
        response = requests.request("POST", url, headers=headers, timeout=10)
    except requests.exceptions.Timeout as e:
        raise RuntimeError("Request to CEDA API timed out.") from e
    response_data = json.loads(response.text)
    if response.status_code == _SUCCESS_STATUS_CODE:
        return response_data["access_token"]

    raise RuntimeError(f"Failed to get CEDA token: {response.text}")


def _get_chess_scape_url(
    rcp: str,
    year: int,
    month: int,
    variable: str,
    resolution: str = "1km",
    timescale: str = "daily",
    realisation: str = "01",
    bias_corrected: bool = True
) -> str:
    """Return the specific URL for a climate dataset."""
    return (
        f"dap2://dap.ceda.ac.uk/thredds/dodsC/badc/deposited2021/chess-scape/data/"
        f"{rcp}{'_bias-corrected' if bias_corrected else ''}/{realisation}/{timescale}"
        f"/{variable}/chess-scape_{rcp}{'_bias-corrected' if bias_corrected else ''}_"
        f"{realisation}_{variable}_uk_{resolution}_{timescale}_{year}{month:02d}01-{year}{month:02d}30.nc"
    )


def _read_clean_chess_scape_file(
    url: str,
    max_retries: int,
    variable: str,
    sentinel: float,
    bounds: tuple[float, float, float, float],
    session: requests.Session,
) -> xr.DataArray:
    """Read and clean CHESS-SCAPE climate data for a given RCP, year, month, and variable."""
    for attempt in range(max_retries):
        try:
            with xr.open_dataset(
                url,
                engine="pydap",
                session=session
            ) as ds:
                data = ds[variable]
                min_x, min_y, max_x, max_y = bounds
                x_is_ascending = bool((data["x"][0] <= data["x"][-1]).item())
                y_is_ascending = bool((data["y"][0] <= data["y"][-1]).item())
                data = data.sel(
                    x=slice(min_x, max_x) if x_is_ascending else slice(max_x, min_x),
                    y=slice(min_y, max_y) if y_is_ascending else slice(max_y, min_y),
                )
                cleaned = (
                    data
                    .where(data != sentinel)
                    .drop_vars(["lat", "lon"], errors="ignore")
                )
                return cleaned.load()
        except Exception as e:
            if "429" not in str(e):
                raise
            wait = 2 ** attempt
            LOG.warning(
                "Rate limited by server. Retrying in %s seconds... (attempt %s/%s)",
                wait,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait)

    raise RuntimeError("Failed to open CHESS-SCAPE dataset after multiple retries.")


def _process_variable_for_scenario(
    scenario: str,
    max_retries: int,
    variable: str,
    sentinel: float,
    bounds: tuple[float, float, float, float],
    session: requests.Session,
) -> xr.DataArray:
    """Read and clean CHESS-SCAPE climate data for variable in entire scenario period."""
    if scenario not in _CHESS_SCAPE_SCENARIOS:
        valid = ", ".join(_CHESS_SCAPE_SCENARIOS)
        raise ValueError(f"Unknown CHESS-SCAPE scenario '{scenario}'. Valid values: {valid}.")

    params = _CHESS_SCAPE_SCENARIOS[scenario]
    LOG.info(
        "Reading CHESS-SCAPE data for %s variable, %s scenario (%s %s)",
        variable,
        scenario,
        params["rcp"],
        params["period"]
    )
    monthly_data = []
    for year in params["period"]:
        LOG.info("Processing %s %s data.", variable, year)
        for month in range(1, 13):
            if (year, month) == _MISSING_YEAR_MONTH:
                continue
            url = _get_chess_scape_url(
                params["rcp"],
                year,
                month,
                variable
            )
            monthly_data.append(
                _read_clean_chess_scape_file(
                    url,
                    max_retries,
                    variable,
                    sentinel,
                    bounds,
                    session,
                )
            )
    if not monthly_data:
        raise ValueError(f"No CHESS-SCAPE time slices found for scenario '{scenario}'.")
    return xr.concat(monthly_data, dim="time").sortby("time")


def _clean_chess_scape(
    config: model_config.Config,
    boundary: gpd.GeoDataFrame,
    session: requests.Session,
) -> None:
    """Read, clean, and write CHESS-SCAPE variables for all configured scenarios."""
    bounds = boundary.total_bounds

    for variable in config.parameters.chess_scape.variables:
        LOG.info("Cleaning CHESS-SCAPE %s data.", variable)
        for scenario in _CHESS_SCAPE_SCENARIOS:
            variable_scenario = _process_variable_for_scenario(
                scenario=scenario,
                max_retries=config.parameters.chess_scape.max_access_retries,
                variable=variable,
                sentinel=config.parameters.chess_scape.sentinel,
                bounds=bounds,
                session=session,
            )
            variable_scenario.to_netcdf(
                config.paths.chess_scape_model_input(variable, scenario),
                engine="netcdf4"
            )


### FLOODING


def _clean_flooding(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean flooding data ready for analysis."""
    LOG.info("Cleaning flooding data...")
    is_empty = _clean_rofrs_rofsw(config, boundary)
    if config.hazards.flooding.groundwater is not None:
        is_empty = _clean_groundwater(config, boundary)

    if is_empty:
        LOG.warning("Flooding data unavailable. Switching flooding off.")
        config.switches.flooding = False
    LOG.info("Finished cleaning flooding data.")


def _clean_rofrs_rofsw(config: model_config.Config, boundary: gpd.GeoDataFrame) -> bool:
    """Clean and write RoFRS and RoFSW data ready for analysis.

    Gets BNG codes for the boundary, then read and clean the
    Risk of Flooding from Rivers and Sea (RoFRS) and
    Risk of Flooding from Surface Water (RoFSW) data.
    """
    bng_codes = cleaning_funcs.get_bng_codes(boundary)
    all_empty = True
    for flooding_type in [
        definitions.FloodingTypes.RIVERS_SEA,
        definitions.FloodingTypes.SURFACE_WATER,
    ]:
        for scenario in definitions.Scenarios:
            LOG.info("Cleaning %s flooding data for %s scenario...", flooding_type, scenario)
            all_empty = _clean_flooding_layer(
                config=config,
                flooding_type=flooding_type,
                boundary=boundary,
                out_path=(
                    config.paths.flooding_type_scenario_model_input(flooding_type, scenario)
                ),
                rename_risk_col=f"{flooding_type}_flooding_likelihood_{scenario}",
                climate_change=(scenario == definitions.Scenarios.FUTURE),
                bng_codes=bng_codes,
                all_empty=all_empty,
            )

    if all_empty:
        LOG.warning(
            "Flooding data unavailable for %s",
            config.parameters.stb
            if config.parameters.stb is not None
            else config.parameters.ca
            if config.parameters.ca is not None
            else "model area.",
        )
    return all_empty


def _clean_groundwater(config: model_config.Config, boundary: gpd.GeoDataFrame) -> bool:
    """Clean groundwater flooding data ready for analysis."""
    LOG.info("Cleaning groundwater flooding data...")
    groundwater_flooding = gpd.read_file(
        config.hazards.flooding.groundwater, columns=["CLASS"], mask=boundary
    )
    # TODO (DJ): #41 Replace with RiskColumn.GROUNDWATER_FLOODING when available
    groundwater_flooding = groundwater_flooding.rename(
        columns={"CLASS": "groundwater_flooding_risk"}
    )
    groundwater_flooding = cleaning_funcs.clip_to_boundary(groundwater_flooding, boundary)
    if groundwater_flooding.empty:
        LOG.warning(
            "Groundwater flooding data unavailable for %s.",
            config.parameters.stb
            if config.parameters.stb is not None
            else config.parameters.ca
            if config.parameters.ca is not None
            else "model area.",
        )
        return True
    groundwater_flooding = cleaning_funcs.explode_to_polygons(groundwater_flooding)
    groundwater_flooding = groundwater_flooding.to_crs(definitions.BNG_CRS)
    cleaning_funcs.write_to_file(
        groundwater_flooding,
        output_path=config.paths.groundwater_flooding_model_input,
    )
    return False


def _clean_flooding_layer(
    config: model_config.Config,
    flooding_type: str,
    boundary: gpd.GeoDataFrame,
    *,
    out_path: pathlib.Path,
    rename_risk_col: str,
    climate_change: bool,
    bng_codes: list[str],
    all_empty: bool,
) -> bool:
    """Clean flooding data and write to file."""
    zip_files = _get_flooding_zip_files(config, flooding_type)
    first_write = True

    for zip_path in zip_files:
        metadata = _parse_flooding_metadata(zip_path)

        if metadata["tile"][:2] not in bng_codes:
            continue
        if metadata["climate_change"] != climate_change:
            continue
        LOG.debug("Processing tile: %s...", metadata["tile"])

        flooding_data = _read_flooding_zip(zip_path, boundary)

        if flooding_data.empty:
            LOG.debug("%s layer empty. Continuing.", metadata["tile"])
            continue

        all_empty = False

        flooding_data = cleaning_funcs.extract_poly_from_geomcollection(flooding_data)

        flooding_data = flooding_data.rename(columns={"Risk_band": rename_risk_col})

        cleaning_funcs.write_to_file(
            flooding_data,
            out_path,
            mode="w" if first_write else "a",
        )

        first_write = False

        del flooding_data
        gc.collect()

    return all_empty


def _get_flooding_zip_files(
    config: model_config.Config,
    flooding_type: str,
) -> list[pathlib.Path]:
    """Return all flooding zip files."""
    flooding_root = getattr(config.hazards.flooding, flooding_type)
    return sorted(flooding_root.rglob("*.zip"))


def _parse_flooding_metadata(zip_path: pathlib.Path) -> dict:
    """Extract metadata from flooding filename."""
    name = zip_path.stem
    climate_change = "Climate_Change" in name

    try:
        if climate_change:
            flooding_type, _, _, _, tile, version = name.split("_")
        else:
            flooding_type, tile, version = name.split("_")
    except ValueError as exc:
        raise ValueError(f"Unexpected flooding zip filename format: '{name}'") from exc

    return {
        "flooding_type": flooding_type,
        "tile": tile,
        "version": version,
        "climate_change": climate_change,
    }


def _read_flooding_zip(
    zip_path: pathlib.Path,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Read flooding gdb file from zip."""
    vsi_path = f"zip://{zip_path}!/{zip_path.stem}.gdb"
    layers = fiona.listlayers(vsi_path)

    if not layers:
        raise ValueError(f"No layers found in zip file: {zip_path}")

    return gpd.read_file(
        vsi_path,
        layer=layers[0],
        mask=boundary,
        columns=["Risk_band"],
        engine="pyogrio",
        use_arrow=True,
    )


### GROUND STABILITY


def _clean_ground_stability(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean ground stability data ready for analysis."""
    LOG.info("Cleaning ground stability data...")
    if config.switches.ground_stability_premium:
        is_empty = _clean_premium_geosure(config, boundary)
    else:
        is_empty = _clean_open_geosure(config, boundary)

    if is_empty:
        LOG.warning("Switching ground stability hazard off.")
        config.switches.ground_stability = False
    LOG.info("Finished cleaning ground stability data.")


def _clean_premium_geosure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> bool:
    """Clean premium GeoSure data then write to file."""
    geosure_layers = {
        definitions.GroundStabilityLikelihoodCols.COLLAPSIBLE_DEPOSITS: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.collapsible_deposits,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.COMPRESSIBLE_GROUND: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.compressible_ground,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.LANDSLIDES: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.landslides,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.RUNNING_SAND: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.running_sands,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.SHRINK_SWELL: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.shrink_swell,
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.SOLUBLE_ROCKS: gpd.read_file(
            config.paths.raw_input
            / config.hazards.ground_stability.geosure.premium.soluble_rocks,
            mask=boundary,
            columns=["CLASS"],
        ),
    }

    all_empty = True
    for geohazard, geosure_data in geosure_layers.items():
        if geosure_data.empty:
            LOG.warning(
                "GeoSure %s layer is empty after reading. Check the source file and boundary.",
                geohazard.replace("_", " ").replace("risk", "").title(),
            )
            continue

        all_empty = False
        len_before_filter = len(geosure_data)
        geosure_data_clean = geosure_data.rename(
            columns={"CLASS": f"{geohazard}_{definitions.Scenarios.CURRENT}"}
        )
        geosure_data_clean[f"{geohazard}_{definitions.Scenarios.FUTURE}"] = geosure_data_clean[
            f"{geohazard}_{definitions.Scenarios.CURRENT}"
        ]
        geosure_layers[geohazard] = cleaning_funcs.clip_to_boundary(
            geosure_data_clean, boundary
        )
        filter_removed = len_before_filter - len(geosure_layers[geohazard])
        LOG.debug(
            "GeoSure %s filtered - %s of %s (%.1f percent) rows removed",
            geohazard.replace("_", " ").title(),
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        geosure_layers[geohazard] = cleaning_funcs.explode_to_polygons(
            geosure_layers[geohazard]
        )
        geosure_layers[geohazard] = geosure_layers[geohazard].to_crs(definitions.BNG_CRS)
        cleaning_funcs.write_to_file(
            geosure_layers[geohazard],
            config.paths.geosure_model_input(geohazard),
        )

    if all_empty:
        LOG.warning(
            "Geosure data is unavailable for %s.Switching ground stability hazard off.",
            config.parameters.stb
            if config.parameters.stb is not None
            else config.parameters.ca
            if config.parameters.ca is not None
            else "model area.",
        )
        return True
    return False


def _clean_open_geosure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> bool:
    """Clean GeoSureHexGrids data, merge by nearest centroids, then write to file."""
    geosure_layers = {
        definitions.GroundStabilityLikelihoodCols.COLLAPSIBLE_DEPOSITS: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_CollapsibleDeposits_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.COMPRESSIBLE_GROUND: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_CompressibleGround_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.LANDSLIDES: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_Landslides_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.RUNNING_SAND: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_RunningSand_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.SHRINK_SWELL: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_ShrinkSwell_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
        definitions.GroundStabilityLikelihoodCols.SOLUBLE_ROCKS: gpd.read_file(
            f"zip://"
            f"{config.paths.raw_input / config.hazards.ground_stability.geosure.open.zip_path}"
            f"!{config.hazards.ground_stability.geosure.open.file_path}",
            layer="GB_Hex_5km_GS_SolubleRocks_v8",
            mask=boundary,
            columns=["CLASS"],
        ),
    }

    all_empty = True
    for code, geosure_data in geosure_layers.items():
        if geosure_data.empty:
            LOG.warning(
                "GeoSure %s layer is empty after reading. Check the source file and boundary.",
                code.replace("_", " ").replace("risk", "").title(),
            )
            continue

        all_empty = False
        len_before_filter = len(geosure_data)
        geosure_data_clean = geosure_data.rename(columns={"CLASS": code})
        geosure_data_clean[code] = pd.to_numeric(geosure_data_clean[code], errors="coerce")
        geosure_layers[code] = cleaning_funcs.clip_to_boundary(geosure_data_clean, boundary)
        LOG.debug(
            "GeoSure %s filtered - %s of %s (%.1f percent) rows removed",
            code.replace("_", " ").title(),
            len_before_filter - len(geosure_layers[code]),
            len_before_filter,
            ((len_before_filter - len(geosure_layers[code])) / len_before_filter) * 100,
        )
        geosure_layers[code] = cleaning_funcs.explode_to_polygons(geosure_layers[code])

    if all_empty:
        LOG.warning(
            "Geosure data is unavailable for %s. Switching ground stability hazard off.",
            cleaning_funcs.get_boundary_name(config),
        )
        return True

    # Merge layers based on nearest centroids
    base_code = next(iter(geosure_layers.keys()))
    geosure = geosure_layers[base_code][[base_code, definitions.GEOMETRY_COL]].copy()
    for code, layer in geosure_layers.items():
        if code == base_code:
            continue  # skip the base layer
        layer_subset = layer[
            [code, definitions.GEOMETRY_COL]
        ]  # Select only the relevant class and geometry columns
        matched = cleaning_funcs.nearest_centroids(geosure, layer_subset)
        geosure[code] = matched[code]  # Add the matched CLASS column to the base dataframe

    geosure_likelihood_cols = [col for col in geosure.columns if "_likelihood" in col]
    for col in geosure_likelihood_cols:  # Duplicate likelihood cols for future scenario
        geosure[f"{col}_{definitions.Scenarios.FUTURE}"] = geosure[col]
        geosure = geosure.rename(columns={col: f"{col}_{definitions.Scenarios.CURRENT}"})
    geosure_likelihood_cols = [col for col in geosure.columns if "_likelihood" in col]
    geosure = geosure[[*geosure_likelihood_cols, definitions.GEOMETRY_COL]]
    cleaning_funcs.write_to_file(
        geosure,
        config.paths.geosure_model_input(None),
    )
    return False


### COASTAL EROSION


def _clean_coastal_erosion(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean coastal erosion data ready for analysis."""
    LOG.info("Cleaning coastal erosion data...")
    is_empty = _clean_national_coastal_erosion_risk_mapping(config, boundary)
    if is_empty:
        LOG.warning(
            "Coastal erosion data is unavailable for %s. "
            "Switching coastal erosion hazard off.",
            cleaning_funcs.get_boundary_name(config),
        )
        config.switches.coastal_erosion = False
    LOG.info("Finished cleaning coastal erosion data.")


def _clean_national_coastal_erosion_risk_mapping(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> bool:
    """Clean erosion data from the National Coastal Erosion Risk Mapping (NCERM).

    Reads NCERM coastal erosion data for 2055 and 2105, cleans and writes to file.
    """
    all_empty = True
    for year in ["2055", "2105"]:
        erosion_data = gpd.read_file(
            f"zip://{config.paths.raw_input / config.hazards.coastal_erosion.zip_path}!"
            f"{config.hazards.coastal_erosion.file_path}",
            layer=f"NCERM_SMP_{year}_70CC",
            mask=boundary,
            columns=["smp_name"],
        )
        if erosion_data.empty:
            LOG.debug("NCERM %s layer empty after filtering.", year)
            continue
        all_empty = False
        len_before_filter = len(erosion_data)
        erosion_data = cleaning_funcs.clip_to_boundary(erosion_data, boundary)
        filter_removed = len_before_filter - len(erosion_data)
        LOG.debug(
            "NCERM %s filtered - %s of %s (%.1f percent) rows removed",
            year,
            filter_removed,
            len_before_filter,
            (filter_removed / len_before_filter) * 100,
        )
        erosion_data = cleaning_funcs.explode_to_polygons(erosion_data)

        cleaning_funcs.write_to_file(
            erosion_data,
            config.paths.national_coastal_erosion_risk_mapping_model_input(year),
        )

    return all_empty
