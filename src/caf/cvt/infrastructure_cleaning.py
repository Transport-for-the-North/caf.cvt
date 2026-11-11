"""Module for cleaning transport infrastructure datasets."""

# LOAD LIBRARIES

import logging
import zipfile

import geopandas as gpd
import pandas as pd

from caf.cvt import data_cleaning, definitions, model_config

# LOGGING

LOG = logging.getLogger(__name__)

# MAIN FUNCTIONS


def clean_infrastructure(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all infrastructure datasets ready for analysis.

    Use config switches to clean the relevant infrastructure datasets, covering roads,
    rail, and other transport infrastructure.

    Parameters
    ----------
    config: model_config.Config
        The configuration object containing switches and paths.
    boundary: gpd.GeoDataFrame
        The spatial boundary to which all infrastructure datasets should be clipped.

    Returns
    -------
    None
    """
    LOG.info("Cleaning infrastructure data...")
    _clean_roads(config, boundary)

    any_rail = any(
        [
            config.switches.passenger_rail,
            config.switches.freight_rail,
            config.switches.tram_network,
            config.switches.rapid_transport_network,
        ]
    )

    if any_rail:
        rail_links = _get_rail_links(config, boundary)
    else:
        rail_links = None

    _clean_rail(config, rail_links)

    any_transport_sites = any(
        [
            config.switches.airports,
            config.switches.train_stations,
            config.switches.tram_stations,
            config.switches.rapid_transport_stations,
            config.switches.ferry_terminals,
            config.switches.bus_coach_stations,
        ]
    )

    if any_transport_sites:
        transport_sites = _clean_mastermap_sites(config, boundary)
    else:
        transport_sites = None

    _clean_other(config, boundary, rail_links, transport_sites)

    LOG.info("Finished cleaning infrastructure data.")


### ROAD


def _clean_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Clean all roads datasets ready for analysis."""
    road_cleaning_enabled = any(
        [
            config.switches.all_roads,
            config.switches.noham_roads,
            config.switches.model_roads,
        ]
    )
    if road_cleaning_enabled:
        LOG.info("Cleaning roads data...")
        if config.switches.all_roads:
            LOG.info("Cleaning all roads data...")
            _clean_os_roads(config, boundary)
        if config.switches.noham_roads:
            LOG.info("Cleaning NoHAM roads data...")
            _clean_noham_roads(config, boundary)
        if config.switches.model_roads:
            LOG.info("Cleaning transport model roads data...")
            _clean_model_roads(config, boundary)
        LOG.info("Finished cleaning roads data.")


def _clean_os_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean OS Open Roads dataset, then write to file."""
    os_road = gpd.read_file(
        f"zip://{config.paths.raw_input / config.infrastructure.road.os_road.zip_path}!"
        f"{config.infrastructure.road.os_road.file_path.as_posix()}",
        mask=boundary,
        columns=[col.value for col in definitions.OSRoadCols],
        layer=definitions.OSRoadCols.get_layer_name(),
    )
    os_road = _populate_os_road_structure(config, os_road)
    len_before_filter = len(os_road)
    os_road = os_road.drop_duplicates(
        subset=[definitions.OSRoadCols.ID, definitions.GEOMETRY_COL]
    )
    os_road = os_road.rename(columns={definitions.OSRoadCols.ID: definitions.ID_COL})
    os_road = os_road.replace(0, definitions.NA_COL)
    os_road = data_cleaning.validate_geometries(os_road)
    os_road = data_cleaning.clip_to_boundary(os_road, boundary)
    filter_removed = len_before_filter - len(os_road)
    LOG.debug(
        "OS roads filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        os_road,
        config.paths.os_road_model_input,
    )


def _populate_os_road_structure(
    config: model_config.Config, os_road: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Populate OS road structure column based on MasterMap Highways data."""
    mm_highways = gpd.read_file(
        f"zip://{config.paths.raw_input / config.infrastructure.road.mm_highways.zip_path}!"
        f"{config.infrastructure.road.mm_highways.file_path.as_posix()}",
        columns=[
            definitions.MasterMapHighwaysCols.ID,
            definitions.MasterMapHighwaysCols.ROAD_STRUCTURE,
            definitions.MasterMapHighwaysCols.START_GRADE_SEPARATION,
            definitions.MasterMapHighwaysCols.END_GRADE_SEPARATION,
        ],
        layer=definitions.MasterMapHighwaysCols.get_layer_name(),
        ignore_geometry=True,
    )

    # Filter to only include bridges (grade separation >= 1 and not tunnels)
    mm_highways = mm_highways[
        (
            (mm_highways[definitions.MasterMapHighwaysCols.START_GRADE_SEPARATION] >= 1)
            | (mm_highways[definitions.MasterMapHighwaysCols.END_GRADE_SEPARATION] >= 1)
        )
        & (
            mm_highways[definitions.MasterMapHighwaysCols.ROAD_STRUCTURE]
            != definitions.OSRoadStructure.TUNNEL
        )
    ]
    mm_highways.loc[:, definitions.MasterMapHighwaysCols.ROAD_STRUCTURE] = (
        definitions.OSRoadStructure.BRIDGE
    )

    with (
        zipfile.ZipFile(
            config.paths.raw_input / config.infrastructure.road.os_road_lookup.outer_zip_path
        ) as outer,
        outer.open(
            config.infrastructure.road.os_road_lookup.inner_zip_path.as_posix()
        ) as inner_zip_bytes,
        zipfile.ZipFile(inner_zip_bytes) as inner,
        inner.open(config.infrastructure.road.os_road_lookup.file_path.as_posix()) as csv_file,
    ):
        os_road_lookup = pd.read_csv(
            csv_file,
            usecols=[
                definitions.OSRoadLookupCols.MASTERMAP_ID,
                definitions.OSRoadLookupCols.OS_OPEN_ROADS_ID,
            ],
        )

    bridges_in_mm = len(mm_highways)
    os_road_lookup = os_road_lookup[
        os_road_lookup[definitions.OSRoadLookupCols.MASTERMAP_ID].isin(
            mm_highways[definitions.MasterMapHighwaysCols.ID]
        )
    ]
    LOG.info(
        "Bridges in MasterMap missing from OS Road Lookup: %s",
        bridges_in_mm - os_road_lookup[definitions.OSRoadLookupCols.MASTERMAP_ID].nunique(),
    )
    os_road_lookup = os_road_lookup[
        ~os_road_lookup[definitions.OSRoadLookupCols.OS_OPEN_ROADS_ID].isna()
    ]

    # OS Road Lookup contains duplicates,
    # where one MasterMap ID may map to multiple OS Open Roads IDs.
    mm_highways = (
        mm_highways[
            [
                definitions.MasterMapHighwaysCols.ID,
                definitions.MasterMapHighwaysCols.ROAD_STRUCTURE,
            ]
        ]
        .merge(
            os_road_lookup,
            left_on=definitions.MasterMapHighwaysCols.ID,
            right_on=definitions.OSRoadLookupCols.MASTERMAP_ID,
            how="right",
            validate="one_to_many",
        )
        .drop(
            columns=[
                definitions.OSRoadLookupCols.MASTERMAP_ID,
                definitions.MasterMapHighwaysCols.ID,
            ]
        )
        .drop_duplicates(subset=[definitions.OSRoadLookupCols.OS_OPEN_ROADS_ID])
    )

    return (
        os_road.merge(
            mm_highways,
            left_on=definitions.OSRoadCols.ID,
            right_on=definitions.OSRoadLookupCols.OS_OPEN_ROADS_ID,
            how="left",
            suffixes=("_os", "_mm"),
            validate="one_to_one",
        )
        .assign(
            **{
                definitions.OSRoadCols.ROAD_STRUCTURE: lambda df: df[
                    f"{definitions.OSRoadCols.ROAD_STRUCTURE}_os"
                ].combine_first(df[f"{definitions.OSRoadCols.ROAD_STRUCTURE}_mm"])
            }
        )
        .drop(
            columns=[
                definitions.OSRoadLookupCols.OS_OPEN_ROADS_ID,
                f"{definitions.OSRoadCols.ROAD_STRUCTURE}_os",
                f"{definitions.OSRoadCols.ROAD_STRUCTURE}_mm",
            ]
        )
    )


def _clean_noham_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean NoHAM network dataset, then write to file."""
    noham_network = gpd.read_file(
        config.paths.raw_input / config.infrastructure.road.noham,
        mask=boundary,
        columns=[definitions.NoHAMRoadCols.LINK_ID],
    )
    len_before_filter = len(noham_network)
    noham_network_clean = noham_network.drop_duplicates(
        subset=[definitions.NoHAMRoadCols.LINK_ID, definitions.GEOMETRY_COL]
    )
    noham_network_clean[["a", "b"]] = (
        noham_network_clean[definitions.NoHAMRoadCols.LINK_ID]
        .str.split("_", expand=True)
        .astype(int)
    )
    # Filter out links with a or b less than 10,000 (zone connectors)
    noham_network_clean = noham_network_clean[
        (noham_network_clean["a"] >= config.constants.noham_road_id_threshold)
        & (noham_network_clean["b"] >= config.constants.noham_road_id_threshold)
    ]
    noham_network_clean = noham_network_clean.drop(columns=["a", "b"])
    noham_network_clean = noham_network_clean.rename(
        columns={definitions.NoHAMRoadCols.LINK_ID: definitions.ID_COL}
    )
    noham_network_clean = data_cleaning.validate_geometries(noham_network_clean)
    noham_network_clipped = data_cleaning.clip_to_boundary(noham_network_clean, boundary)
    noham_network_clipped = noham_network_clipped.reset_index(drop=True)
    filter_removed = len_before_filter - len(noham_network_clipped)
    LOG.debug(
        "NoHAM network filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        noham_network_clipped,
        config.paths.noham_road_model_input,
    )


def _clean_model_roads(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read and clean transport model road network dataset, then write to file."""
    nodes = pd.read_csv(config.infrastructure.road.model_roads.nodes)
    zone_nodes = nodes.loc[nodes["is_zone"] == 1, "id"]

    links = pd.read_csv(config.infrastructure.road.model_roads.links)
    links = links.rename(columns={"id": "link_id"})
    links_before_filter = len(links)
    links = links[~links["a_node"].isin(zone_nodes) & ~links["b_node"].isin(zone_nodes)]
    LOG.debug("Removed %s zone connectors.", links_before_filter - len(links))
    links = links.merge(
        nodes.rename(columns={"id": "a_node", "easting": "X1", "northing": "Y1"})[
            ["a_node", "X1", "Y1"]
        ],
        on="a_node",
    )
    links = links.merge(
        nodes.rename(columns={"id": "b_node", "easting": "X2", "northing": "Y2"})[
            ["b_node", "X2", "Y2"]
        ],
        on="b_node",
    )

    shaped_links = gpd.read_file(config.infrastructure.road.model_roads.shaped_links)
    shaped_links = shaped_links.to_crs(definitions.BNG_CRS)
    if len(shaped_links) != len(links):
        LOG.warning(
            "Number of shaped links (%s) does not match number of links (%s).",
            len(shaped_links),
            len(links),
        )

    coord_cols = ["X1", "Y1", "X2", "Y2"]

    for col in coord_cols:
        links[col] = round(links[col].astype(float), 0)
        shaped_links[col] = round(shaped_links[col].astype(float), 0)

    # Join shaped links to links using coordinates
    # This link key assumes (X1, Y1) -> (X2, Y2) is stored the same direction in both datasets
    # If the link key was defined differently,
    # there will be NaNs after the merge for unmatched links.
    model_roads = links[["link_id", *coord_cols]].merge(
        shaped_links[[*coord_cols, definitions.GEOMETRY_COL]],
        on=coord_cols,
        how="left",
        validate="one_to_one",
    )

    if model_roads[definitions.GEOMETRY_COL].isna().sum() != 0:
        LOG.warning(
            "Some model roads do not have a matching shaped link. Missing geometries: %s",
            model_roads[definitions.GEOMETRY_COL].isna().sum(),
        )

    model_roads = gpd.GeoDataFrame(
        model_roads.drop(columns=coord_cols),
        geometry=definitions.GEOMETRY_COL,
        crs=definitions.BNG_CRS,
    )

    len_before_filter = len(model_roads)
    model_roads = data_cleaning.clip_to_boundary(model_roads, boundary)
    LOG.debug(
        "Model roads filtered - %s of %s (%.1f percent) rows removed",
        len_before_filter - len(model_roads),
        len_before_filter,
        ((len_before_filter - len(model_roads)) / len_before_filter) * 100,
    )

    data_cleaning.write_to_file(
        model_roads,
        config.paths.model_roads_model_input,
    )


### RAIL


def _clean_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame | None) -> None:
    """Clean all rail datasets ready for analysis."""
    rail_cleaning_enabled = any(
        [
            config.switches.passenger_rail,
            config.switches.freight_rail,
        ]
    )
    if rail_cleaning_enabled:
        LOG.info("Cleaning rail data...")
        if config.switches.passenger_rail:
            LOG.info("Cleaning passenger rail data...")
            _clean_passenger_rail(config, rail_links)
        if config.switches.freight_rail:
            LOG.info("Cleaning freight rail data...")
            _clean_freight_rail(config, rail_links)
        LOG.info("Finished cleaning rail data.")


def _get_rail_links(
    config: model_config.Config,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Read and clean OS Rail Network data from the Multi-Modal Routing Network."""
    rail_links = gpd.read_file(
        f"zip://{config.paths.raw_input / config.infrastructure.rail.rail_links.zip_path}!"
        f"{config.infrastructure.rail.rail_links.file_path}",
        mask=boundary,
        columns=[col.value for col in definitions.OSRailCols],
    )
    len_before_filter = len(rail_links)
    rail_links = rail_links[
        rail_links[definitions.OSRailCols.OPERATIONAL_STATUS]
        == definitions.OSRailOperationalStatus.ACTIVE
    ]
    rail_links = rail_links.drop(columns=definitions.OSRailCols.OPERATIONAL_STATUS)
    rail_links = rail_links[
        rail_links[definitions.OSRailCols.TRACK_REPRESENTATION]
        != definitions.OSRailTrackRepresentation.SIDING
    ]
    rail_links = rail_links[
        ~rail_links[definitions.OSRailCols.DESCRIPTION].isin(
            [
                definitions.OSRailDescription.PRESERVED,
                definitions.OSRailDescription.FUNICULAR,
                definitions.OSRailDescription.MINERAL,
                definitions.OSRailDescription.STATIC_MUSEUM,
            ]
        )
    ]
    rail_links = rail_links.drop_duplicates(
        subset=[definitions.OSRailCols.ID, definitions.GEOMETRY_COL]
    )
    rail_links[definitions.OSRailCols.get_descriptive_cols()] = rail_links[
        definitions.OSRailCols.get_descriptive_cols()
    ].replace(0, definitions.NA_COL)
    rail_links = rail_links.rename(columns={definitions.OSRailCols.ID: definitions.ID_COL})
    rail_links = data_cleaning.validate_geometries(rail_links)
    rail_links = data_cleaning.clip_to_boundary(rail_links, boundary)
    filter_removed = len_before_filter - len(rail_links)
    LOG.debug(
        "OS Transport Network Rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )

    return rail_links


def _clean_passenger_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to passenger rail network, then write to file."""
    len_before_filter = len(rail_links)
    passenger_rail = rail_links[
        rail_links[definitions.OSRailCols.RAILWAY_USE].isin(
            [
                definitions.OSRailRailwayUse.FREIGHT_AND_PASSENGER,
                definitions.OSRailRailwayUse.PASSENGER,
            ]
        )
    ]
    passenger_rail = passenger_rail[
        passenger_rail[definitions.OSRailCols.DESCRIPTION].isin(
            [
                definitions.OSRailDescription.MAIN_LINE,
                definitions.OSRailDescription.MAIN_LINE_AND_TRAM,
                definitions.OSRailDescription.MAIN_LINE_AND_RAPID_TRANSPORT_SYSTEM,
            ]
        )
    ]
    filter_removed = len_before_filter - len(passenger_rail)
    LOG.debug(
        "Passenger rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        passenger_rail,
        config.paths.passenger_rail_model_input,
    )


def _clean_freight_rail(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail data to freight rail network, then write to file."""
    len_before_filter = len(rail_links)
    freight_rail = rail_links[
        rail_links[definitions.OSRailCols.RAILWAY_USE].isin(
            [
                definitions.OSRailRailwayUse.FREIGHT_AND_PASSENGER,
                definitions.OSRailRailwayUse.FREIGHT,
            ]
        )
    ]
    filter_removed = len_before_filter - len(freight_rail)
    LOG.debug(
        "Freight rail links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        freight_rail,
        config.paths.freight_rail_model_input,
    )


### OTHER


def _clean_other(  # noqa: C901
    config: model_config.Config,
    boundary: gpd.GeoDataFrame,
    rail_links: gpd.GeoDataFrame | None,
    transport_sites: gpd.GeoDataFrame | None,
) -> None:
    """Clean all other datasets ready for analysis."""
    other_cleaning_enabled = any(
        [
            config.switches.airports,
            config.switches.bus_stops,
            config.switches.national_cycle_network,
            config.switches.train_stations,
            config.switches.tram_stations,
            config.switches.rapid_transport_stations,
            config.switches.ferry_terminals,
            config.switches.bus_coach_stations,
            config.switches.tram_network,
            config.switches.rapid_transport_network,
        ]
    )
    if other_cleaning_enabled:
        LOG.info("Cleaning other infrastructure data...")
        if config.switches.airports:
            LOG.info("Cleaning airports data...")
            _clean_airports(config, transport_sites)
        if config.switches.bus_stops:
            LOG.info("Cleaning bus stops data...")
            _clean_bus_stops(config, boundary)
        if config.switches.national_cycle_network:
            LOG.info("Cleaning national cycle network data...")
            _clean_national_cycle_network(config, boundary)
        if config.switches.train_stations:
            LOG.info("Cleaning train stations data...")
            _clean_train_stations(config, transport_sites)
        if config.switches.tram_stations:
            LOG.info("Cleaning tram stations data...")
            _clean_tram_stations(config, transport_sites)
        if config.switches.rapid_transport_stations:
            LOG.info("Cleaning rapid transport stations data...")
            _clean_rapid_transport_stations(config, transport_sites)
        if config.switches.ferry_terminals:
            LOG.info("Cleaning ferry terminals data...")
            _clean_ferry_terminals(config, transport_sites)
        if config.switches.bus_coach_stations:
            LOG.info("Cleaning bus coach stations data...")
            _clean_bus_coach_stations(config, transport_sites)
        if config.switches.tram_network:
            LOG.info("Cleaning tram network data...")
            _clean_tram_network(config, rail_links)
        if config.switches.rapid_transport_network:
            LOG.info("Cleaning rapid transport network data...")
            _clean_rapid_transport_network(config, rail_links)
        LOG.info("Finished cleaning other infrastructure data.")


def _clean_mastermap_sites(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Read, clean and prepare mastermap sites data for further filtering."""
    mm_sites = gpd.read_file(
        f"zip://"
        f"{config.paths.raw_input / config.infrastructure.other.mastermap_sites.zip_path}!"
        f"{config.infrastructure.other.mastermap_sites.file_path.as_posix()}",
        layer=definitions.MasterMapSiteCols.get_layer_name(),
        mask=boundary,
        columns=[col.value for col in definitions.MasterMapSiteCols],
    )
    len_before_filter = len(mm_sites)
    mm_sites = mm_sites[
        mm_sites[definitions.MasterMapSiteCols.FUNCTION_STATUS]
        == definitions.MasterMapFunctionStatus.OPERATIONAL
    ]
    mm_sites = mm_sites.drop(columns=[definitions.MasterMapSiteCols.FUNCTION_STATUS])
    transport_sites = mm_sites[
        mm_sites[definitions.MasterMapSiteCols.FUNCTION_THEME].isin(
            [
                definitions.MasterMapFunctionTheme.RAIL_TRANSPORT,
                definitions.MasterMapFunctionTheme.AIR_TRANSPORT,
                definitions.MasterMapFunctionTheme.ROAD_TRANSPORT,
                definitions.MasterMapFunctionTheme.WATER_TRANSPORT,
            ]
        )
    ]
    transport_sites = transport_sites.drop_duplicates(
        subset=[definitions.MasterMapSiteCols.ID, definitions.GEOMETRY_COL]
    )
    transport_sites = transport_sites.rename(
        columns={definitions.MasterMapSiteCols.ID: definitions.ID_COL}
    )
    transport_sites[definitions.MasterMapSiteCols.get_descriptive_cols()] = transport_sites[
        definitions.MasterMapSiteCols.get_descriptive_cols()
    ].fillna(definitions.NA_COL)
    transport_sites = data_cleaning.validate_geometries(transport_sites)
    filter_removed = len_before_filter - len(transport_sites)
    LOG.debug(
        "OS MasterMap sites filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    return transport_sites


def _clean_airports(config: model_config.Config, transport_sites: gpd.GeoDataFrame) -> None:
    """Filter OS MM Sites for airports and write to file."""
    len_before_filter = len(transport_sites)
    air_transport = transport_sites[
        transport_sites[definitions.MasterMapSiteCols.FUNCTION_THEME]
        == definitions.MasterMapFunctionTheme.AIR_TRANSPORT
    ]
    airports = air_transport[
        air_transport[definitions.MasterMapSiteCols.FUNCTION]
        == definitions.MasterMapFunction.AIRPORT
    ]
    airports = airports.drop(
        columns=[
            definitions.MasterMapSiteCols.FUNCTION_THEME,
            definitions.MasterMapSiteCols.FUNCTION,
            definitions.MasterMapSiteCols.STAKEHOLDER_1,
            definitions.MasterMapSiteCols.STAKEHOLDER_1_ROLE,
            definitions.MasterMapSiteCols.EXTENT_DEFINITION,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_3,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_4,
        ]
    )
    filter_removed = len_before_filter - len(airports)
    LOG.debug(
        "Transport sites filtered to airports - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(airports, config.paths.airports_model_input)


def _clean_bus_stops(config: model_config.Config, boundary: gpd.GeoDataFrame) -> None:
    """Read, combine and clean regional bus stops datasets, then write to file."""
    # TODO (DJ): #33 Some areas of NaPTAN are missing, e.g. Cumbria Combined Authority
    # Investigate alternative dataset with full coverage.
    missing_areas = definitions.NaPTANCols.get_missing_areas()
    if config.parameters.stb and config.parameters.stb in missing_areas:
        LOG.warning("Bus Stops data is partially missing for %s", config.parameters.stb)

    if config.parameters.ca and config.parameters.ca in missing_areas:
        LOG.warning("Bus Stops data is partially missing for %s", config.parameters.ca)

    naptan = pd.read_csv(
        config.paths.raw_input / config.infrastructure.other.bus_stops,
        usecols=[col.value for col in definitions.NaPTANCols],
    )
    len_before_filter = len(naptan)
    bus_stops = naptan[naptan[definitions.NaPTANCols.STOP_TYPE] == "BCT"]
    bus_stops = bus_stops[bus_stops[definitions.NaPTANCols.STATUS] == "active"]
    bus_stops = gpd.GeoDataFrame(
        bus_stops.copy(),
        geometry=gpd.points_from_xy(
            bus_stops[definitions.NaPTANCols.LONGITUDE],
            bus_stops[definitions.NaPTANCols.LATITUDE],
        ),
        crs="EPSG:4326",
    )
    bus_stops = bus_stops.to_crs(definitions.BNG_CRS)
    bus_stops = bus_stops.drop(
        columns=[
            definitions.NaPTANCols.STOP_TYPE,
            definitions.NaPTANCols.LONGITUDE,
            definitions.NaPTANCols.LATITUDE,
            definitions.NaPTANCols.STATUS,
        ]
    )
    bus_stops = bus_stops.rename(columns=definitions.NaPTANCols.rename_map())
    bus_stops = data_cleaning.clip_to_boundary(bus_stops, boundary)
    filter_removed = len_before_filter - len(bus_stops)
    LOG.debug(
        "NaPTAN filtered to bus stops - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(bus_stops, config.paths.bus_stops_model_input)


def _clean_train_stations(
    config: model_config.Config, transport_sites: gpd.GeoDataFrame
) -> None:
    """Filter OS MM Sites for train stations and write to file."""
    len_before_filter = len(transport_sites)
    rail_transport = transport_sites[
        transport_sites[definitions.MasterMapSiteCols.FUNCTION_THEME]
        == definitions.MasterMapFunctionTheme.RAIL_TRANSPORT
    ]
    train_stations = rail_transport[
        rail_transport[definitions.MasterMapSiteCols.FUNCTION].str.contains(
            definitions.MasterMapFunction.RAILWAY_STATION
        )
    ]
    # Stakeholder names use '+' to join multiple stakeholders,
    # e.g. London Underground + Network Rail,
    # So we want to filter for strings containing 'Network Rail'
    train_stations = train_stations[
        (
            train_stations[definitions.MasterMapSiteCols.STAKEHOLDER_1].str.contains(
                definitions.MasterMapStakeholder.NETWORK_RAIL
            )
        )
    ]
    train_stations = train_stations.drop(
        columns=[
            definitions.MasterMapSiteCols.FUNCTION_THEME,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_3,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_4,
            definitions.MasterMapSiteCols.STAKEHOLDER_1_ROLE,
        ]
    )
    filter_removed = len_before_filter - len(train_stations)
    LOG.debug(
        "Train stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        train_stations,
        config.paths.train_stations_model_input,
    )


def _clean_tram_stations(
    config: model_config.Config, transport_sites: gpd.GeoDataFrame
) -> None:
    """Filter OS MM Sites for tram stations and write to file."""
    len_before_filter = len(transport_sites)
    rail_transport = transport_sites[
        transport_sites[definitions.MasterMapSiteCols.FUNCTION_THEME]
        == definitions.MasterMapFunctionTheme.RAIL_TRANSPORT
    ]
    tram_stations = rail_transport[
        rail_transport[definitions.MasterMapSiteCols.FUNCTION].str.contains(
            definitions.MasterMapFunction.TRAM_STATION
        )
    ]
    tram_stations = tram_stations.drop(
        columns=[
            definitions.MasterMapSiteCols.FUNCTION_THEME,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_3,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_4,
            definitions.MasterMapSiteCols.STAKEHOLDER_1_ROLE,
        ]
    )
    filter_removed = len_before_filter - len(tram_stations)
    LOG.debug(
        "Tram stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        tram_stations,
        config.paths.tram_stations_model_input,
    )


def _clean_rapid_transport_stations(
    config: model_config.Config, transport_sites: gpd.GeoDataFrame
) -> None:
    """Filter OS MM Sites for rapid transport stations and write to file."""
    len_before_filter = len(transport_sites)
    rail_transport = transport_sites[
        transport_sites[definitions.MasterMapSiteCols.FUNCTION_THEME]
        == definitions.MasterMapFunctionTheme.RAIL_TRANSPORT
    ]
    # TODO (DJ): Make this an input file so that it can be expanded if necessary, rather than
    # hardcoding, validate input file.
    rapid_transport_stations = rail_transport[
        rail_transport[definitions.MasterMapSiteCols.STAKEHOLDER_1].str.contains(
            "|".join(
                [
                    definitions.MasterMapStakeholder.TYNE_AND_WEAR_METRO,
                    definitions.MasterMapStakeholder.GLASGOW_SUBWAY,
                    definitions.MasterMapStakeholder.LONDON_UNDERGROUND,
                    definitions.MasterMapStakeholder.DOCKLANDS_LIGHT_RAILWAY,
                ]
            )
        )
    ]
    rapid_transport_stations = rapid_transport_stations.drop(
        columns=[
            definitions.MasterMapSiteCols.FUNCTION_THEME,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_3,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_4,
            definitions.MasterMapSiteCols.STAKEHOLDER_1_ROLE,
        ]
    )
    filter_removed = len_before_filter - len(rapid_transport_stations)
    LOG.debug(
        "Rapid transport stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        rapid_transport_stations,
        config.paths.rapid_transport_stations_model_input,
    )


def _clean_ferry_terminals(
    config: model_config.Config, transport_sites: gpd.GeoDataFrame
) -> None:
    """Filter OS MM Sites for ferry terminals and write to file."""
    len_before_filter = len(transport_sites)
    water_transport = transport_sites[
        transport_sites[definitions.MasterMapSiteCols.FUNCTION_THEME]
        == definitions.MasterMapFunctionTheme.WATER_TRANSPORT
    ]
    ferry_terminals = water_transport[
        water_transport[definitions.MasterMapSiteCols.FUNCTION].str.contains(
            definitions.MasterMapFunction.FERRY_TERMINAL
        )
    ]
    ferry_terminals = ferry_terminals.drop(
        columns=[
            definitions.MasterMapSiteCols.FUNCTION_THEME,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_3,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_4,
            definitions.MasterMapSiteCols.STAKEHOLDER_1_ROLE,
        ]
    )
    filter_removed = len_before_filter - len(water_transport)
    LOG.debug(
        "Ferry terminals filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        ferry_terminals,
        config.paths.ferry_terminals_model_input,
    )


def _clean_bus_coach_stations(
    config: model_config.Config, transport_sites: gpd.GeoDataFrame
) -> None:
    """Filter OS MM Sites for bus and coach stations and write to file."""
    len_before_filter = len(transport_sites)
    road_transport = transport_sites[
        transport_sites[definitions.MasterMapSiteCols.FUNCTION_THEME]
        == definitions.MasterMapFunctionTheme.ROAD_TRANSPORT
    ]
    bus_coach_stations = road_transport[
        road_transport[definitions.MasterMapSiteCols.FUNCTION].str.contains(
            "|".join(
                [
                    definitions.MasterMapFunction.BUS_STATION,
                    definitions.MasterMapFunction.COACH_STATION,
                ]
            )
        )
    ]
    bus_coach_stations = bus_coach_stations.drop(
        columns=[
            definitions.MasterMapSiteCols.FUNCTION_THEME,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_3,
            definitions.MasterMapSiteCols.DISTINCTIVE_NAME_4,
            definitions.MasterMapSiteCols.STAKEHOLDER_1_ROLE,
        ]
    )
    filter_removed = len_before_filter - len(bus_coach_stations)
    LOG.debug(
        "Bus and coach stations filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        bus_coach_stations,
        config.paths.bus_coach_stations_model_input,
    )


def _clean_tram_network(config: model_config.Config, rail_links: gpd.GeoDataFrame) -> None:
    """Filter OS rail links for tram network, then write to file."""
    len_before_filter = len(rail_links)
    tram_links = rail_links[
        rail_links[definitions.OSRailCols.RAILWAY_USE].isin(
            [
                definitions.OSRailRailwayUse.FREIGHT_AND_PASSENGER,
                definitions.OSRailRailwayUse.PASSENGER,
            ]
        )
    ]
    tram_links = tram_links[
        tram_links[definitions.OSRailCols.DESCRIPTION].isin(
            [
                definitions.OSRailDescription.TRAM,
                definitions.OSRailDescription.MAIN_LINE_AND_TRAM,
            ]
        )
    ]
    filter_removed = len_before_filter - len(tram_links)
    LOG.debug(
        "Tram network links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(tram_links, config.paths.tram_network_model_input)


def _clean_rapid_transport_network(
    config: model_config.Config, rail_links: gpd.GeoDataFrame
) -> None:
    """Filter OS rail links for rapid transport network, then write to file."""
    len_before_filter = len(rail_links)
    rapid_transport_links = rail_links[
        rail_links[definitions.OSRailCols.RAILWAY_USE].isin(
            [
                definitions.OSRailRailwayUse.FREIGHT_AND_PASSENGER,
                definitions.OSRailRailwayUse.PASSENGER,
            ]
        )
    ]
    rapid_transport_links = rapid_transport_links[
        rapid_transport_links[definitions.OSRailCols.DESCRIPTION].isin(
            [
                definitions.OSRailDescription.RAPID_TRANSPORT_SYSTEM,
                definitions.OSRailDescription.MAIN_LINE_AND_RAPID_TRANSPORT_SYSTEM,
            ]
        )
    ]

    filter_removed = len_before_filter - len(rapid_transport_links)
    LOG.debug(
        "Rapid transport links filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        rapid_transport_links,
        config.paths.rapid_transport_network_model_input,
    )


def _clean_national_cycle_network(
    config: model_config.Config, boundary: gpd.GeoDataFrame
) -> None:
    """Read and clean National Cycle Network data, then write to file."""
    ncn = gpd.read_file(
        config.paths.raw_input / config.infrastructure.other.ncn_sustrans,
        mask=boundary,
        columns=[col.value for col in definitions.NationalCycleNetworkCols],
    )
    len_before_filter = len(ncn)
    ncn = ncn.drop_duplicates(
        subset=[definitions.NationalCycleNetworkCols.ID, definitions.GEOMETRY_COL]
    )
    ncn_cols_replace = [
        col.value
        for col in definitions.NationalCycleNetworkCols
        if col != definitions.NationalCycleNetworkCols.ID
    ]
    ncn[ncn_cols_replace] = ncn[ncn_cols_replace].replace(0, definitions.NA_COL)
    ncn = ncn.rename(columns=definitions.NationalCycleNetworkCols.rename_map())
    ncn = data_cleaning.validate_geometries(ncn)
    ncn = data_cleaning.clip_to_boundary(ncn, boundary)
    filter_removed = len_before_filter - len(ncn)
    LOG.debug(
        "National Cycle Network filtered - %s of %s (%.1f percent) rows removed",
        filter_removed,
        len_before_filter,
        (filter_removed / len_before_filter) * 100,
    )
    data_cleaning.write_to_file(
        ncn,
        config.paths.national_cycle_network_model_input,
    )
