"""Module to clean and prepare Nexus transport data."""

import itertools
import logging
import pathlib

import caf.toolkit as ctk
import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely

import caf.cvt as cvt

_NAME = pathlib.Path(__file__).stem

# LOGGING

LOG = logging.getLogger(_NAME)

# PATHS

DATA_PATH = pathlib.Path("D:/Climate Vulnerability Tool/Localisation/v2/Nexus")

LOG_PATH = DATA_PATH / "Data" / "logging"

METRO_LINES_PATH = DATA_PATH / "Metro_Lines/Metro_Lines.shp"
METRO_EXT_LINES_PATH = DATA_PATH / "M2W_Line_PelawToSouthHylton.gpkg"
METRO_STATIONS_PATH = DATA_PATH / "Metro_Stations/Metro_Stations.shp"
METRO_EXT_STATIONS_PATH = DATA_PATH / "M2W_Stations.gpkg"

MODEL_INPUT_PATH = DATA_PATH / "data" / "model_input"
NEXUS_METRO_LINKS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Bespoke/Nexus Metro/metro_links.gpkg"
)
NEXUS_METRO_STATIONS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Bespoke/Nexus Metro/metro_stations.gpkg"
)
NEXUS_METRO_LINK_FLOWS_MODEL_INPUT_PATH = pathlib.Path(
    "Impact/Nexus Metro Link Flows/nexus_metro_link_flows.gpkg"
)

# MODULE CONSTANTS

GEOMETRY_COL = "geometry"

CURRENT = "current"
FORECAST = "forecast"

MONUMENT_LINKS = [9, 11]

MANUAL_MERGE_LINK_IDS = [
    (34, 35),
    (55, 66),
    (64, 65, 70),
    (9, 71),
    (59, 60),
    (13, 21, 56),
    (14, 13),
]


SOUTH_GOSFORTH_LINK_ID = 34
SOUTH_GOSFORTH_FROM_STATION_ID = 8

PELAW_LINK_ID = 73
PELAW_FROM_STATION_ID = 19

SOUTH_HYLTON_LINK_ID = 76
SOUTH_HYLTON_TO_STATION_ID = 31

# GENERAL FUNCTIONS

def _first_not_null(series: pd.Series) -> int | None:
    """Return the first non-null value in a pandas Series."""
    values = series.dropna()
    return values.iloc[0] if len(values) > 0 else None


# NETWORK AND STATIONS CLEANING FUNCTIONS


def _clean_nexus_metro() -> None:
    """Clean Nexus Metro data ready for analysis."""
    LOG.info("Cleaning nexus metro data...")
    metro_links = _aggregate_metro_links()

    metro_stations = _aggregate_metro_stations()

    snapped_metro_stations = _snap_stations_to_links(metro_stations, metro_links)

    metro_links = _split_metro_links(metro_links, snapped_metro_stations)

    metro_links.to_file(
        MODEL_INPUT_PATH / NEXUS_METRO_LINKS_MODEL_INPUT_PATH,
        driver="GPKG"
    )
    snapped_metro_stations.to_file(
        MODEL_INPUT_PATH / NEXUS_METRO_STATIONS_MODEL_INPUT_PATH,
        driver="GPKG"
    )



def _aggregate_metro_links() -> gpd.GeoDataFrame:
    """Aggregate existing and extension metro links."""
    metro_links = gpd.read_file(
        METRO_LINES_PATH,
        columns=["OBJECTID_1", "Ownership"]
    )
    metro_links["extension"] = False

    metro_ext_lines = gpd.read_file(
        METRO_EXT_LINES_PATH,
        columns=["Section"]
    )
    metro_ext_lines = metro_ext_lines.explode(index_parts=False).reset_index(drop=True)
    metro_ext_lines["extension"] = True
    metro_ext_lines["Ownership"] = None
    metro_ext_lines = metro_ext_lines.drop(columns=["Section"])
    metro_ext_lines["OBJECTID_1"] = range(
        max(metro_links["OBJECTID_1"]) + 1,
        max(metro_links["OBJECTID_1"]) + 1 + len(metro_ext_lines),
    )

    metro_links = pd.concat([metro_links, metro_ext_lines], ignore_index=True)
    return metro_links.rename(columns={"OBJECTID_1": "id"})


def _aggregate_metro_stations() -> gpd.GeoDataFrame:
    """Aggregate existing and extension metro stations."""
    metro_stations = gpd.read_file(
        METRO_STATIONS_PATH,
        columns=["OBJECTID", "Name", "Symbol"],
    )
    metro_stations["extension"] = False

    metro_ext_stations = gpd.read_file(
        METRO_EXT_STATIONS_PATH,
        columns=["StationName"]
    )
    metro_ext_stations = metro_ext_stations.rename(columns={"StationName": "Name"})
    metro_ext_stations["OBJECTID"] = range(
        max(metro_stations["OBJECTID"]) + 1,
        max(metro_stations["OBJECTID"]) + 1 + len(metro_ext_stations),
    )
    metro_ext_stations["extension"] = True

    metro_stations = pd.concat([metro_stations, metro_ext_stations], ignore_index=True)
    return metro_stations.rename(columns={"OBJECTID": "id"})


def _split_metro_links(
    metro_links: gpd.GeoDataFrame,
    metro_stations: gpd.GeoDataFrame,
    tolerance: float = 1,
    min_split_dist: float = 0.01,
) -> gpd.GeoDataFrame:
    """Split metro link geometries at metro station locations."""
    split_rows = []

    for _, row in metro_links.iterrows():
        line = row.geometry

        # Find stations on this line
        stations_on_line = metro_stations[metro_stations.geometry.distance(line) < tolerance]

        # Get stations positions along line
        stations = []
        for _, station in stations_on_line.iterrows():
            stations.append(
                {
                    "position": line.project(station.geometry),
                    "station_id": station["id"],
                    "station_name": station["Name"],
                }
            )

        # Remove duplicates and sort
        stations = sorted(stations, key=lambda x: x["position"])

        if len(stations) == 0:
            new_row = row.copy()

            new_row["from_station_id"] = None
            new_row["to_station_id"] = None
            new_row["from_station_name"] = None
            new_row["to_station_name"] = None
            split_rows.append(new_row)
            continue

        stations.insert(0, {"position": 0, "station_id": None, "station_name": None})
        stations.insert(
            len(stations), {"position": line.length, "station_id": None, "station_name": None}
        )

        # Create line segments
        for start_station, end_station in itertools.pairwise(stations):
            if end_station["position"] - start_station["position"] < min_split_dist:
                continue
            segment = shapely.ops.substring(
                line,
                start_station["position"],
                end_station["position"],
            )

            new_row = row.copy()

            new_row["from_station_id"] = start_station["station_id"]
            new_row["to_station_id"] = end_station["station_id"]
            new_row["from_station_name"] = start_station["station_name"]
            new_row["to_station_name"] = end_station["station_name"]

            new_row.geometry = segment
            split_rows.append(new_row)

    metro_links = gpd.GeoDataFrame(
        split_rows,
        columns=[
            *metro_links.columns,
            "from_station_id",
            "to_station_id",
            "from_station_name",
            "to_station_name",
        ],
        crs=metro_links.crs,
    ).reset_index(drop=True)

    metro_links["metro_link_id"] = range(1, len(metro_links) + 1)

    return _manual_metro_adjustments(metro_links)


def _snap_stations_to_links(
    metro_stations: gpd.GeoDataFrame,
    metro_links: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Snap metro stations to the nearest metro link."""
    snapped_stations = metro_stations.copy()
    for idx, station in snapped_stations.iterrows():
        nearest_line_idx = metro_links.distance(station.geometry).idxmin()

        nearest_line = metro_links.loc[nearest_line_idx, GEOMETRY_COL]

        snapped_stations.loc[idx, GEOMETRY_COL] = nearest_line.interpolate(
            nearest_line.project(station.geometry)
        )

    # Manually snap Monument to intersection of links 9 and 11
    monument_id = snapped_stations.loc[
        snapped_stations["Name"] == "Monument", "id"
    ].to_numpy()[0]
    link_9 = metro_links.loc[metro_links["id"] == MONUMENT_LINKS[0], GEOMETRY_COL].to_numpy()[
        0
    ]
    link_11 = metro_links.loc[metro_links["id"] == MONUMENT_LINKS[1], GEOMETRY_COL].to_numpy()[
        0
    ]
    intersection_point = link_9.intersection(link_11)
    snapped_stations.loc[snapped_stations["id"] == monument_id, GEOMETRY_COL] = (
        intersection_point
    )

    return snapped_stations


def _manual_metro_adjustments(metro_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply manual adjustments to the metro links."""
    # Apply manual adjustments to the metro links
    rows_to_drop = set()
    new_rows = []

    for merge_group in MANUAL_MERGE_LINK_IDS:
        group = metro_links.loc[metro_links["metro_link_id"].isin(merge_group)].copy()

        if group.empty:
            continue

        merged_geometry = shapely.MultiLineString(group[GEOMETRY_COL].to_list())
        new_row = group.iloc[0].copy()
        new_row[GEOMETRY_COL] = merged_geometry

        new_row["from_station_id"] = _first_not_null(group["from_station_id"])
        new_row["to_station_id"] = _first_not_null(group["to_station_id"])
        new_row["from_station_name"] = _first_not_null(group["from_station_name"])
        new_row["to_station_name"] = _first_not_null(group["to_station_name"])
        new_row["metro_link_id"] = merge_group[0]

        new_rows.append(new_row)
        rows_to_drop.update(group.index)

    metro_links = metro_links.drop(index=list(rows_to_drop))
    metro_links = pd.concat(
        [
            metro_links,
            gpd.GeoDataFrame(new_rows, columns=metro_links.columns, crs=metro_links.crs),
        ],
        ignore_index=True,
    )

    # Assign from and to stations
    metro_links.loc[
        metro_links["metro_link_id"] == SOUTH_GOSFORTH_LINK_ID,
        "from_station_id"
    ] = SOUTH_GOSFORTH_FROM_STATION_ID
    metro_links.loc[
        metro_links["metro_link_id"] == SOUTH_GOSFORTH_LINK_ID,
        "from_station_name"
    ] = "South Gosforth"

    metro_links.loc[
        metro_links["metro_link_id"] == PELAW_LINK_ID,
        "from_station_id"
    ] = PELAW_FROM_STATION_ID
    metro_links.loc[
        metro_links["metro_link_id"] == PELAW_LINK_ID,
        "from_station_name"
    ] = "Pelaw"

    metro_links.loc[
        metro_links["metro_link_id"] == SOUTH_HYLTON_LINK_ID,
        "to_station_id"
    ] = SOUTH_HYLTON_TO_STATION_ID
    metro_links.loc[
        metro_links["metro_link_id"] == SOUTH_HYLTON_LINK_ID,
        "to_station_name"
    ] = "South Hylton"

    return metro_links


# METRO DEMAND FUNCTIONS

def _clean_nexus_demand() -> None:
    """Clean nexus infrastructure demand data."""
    LOG.info("Cleaning nexus demand data.")
    baseline = pd.read_csv(
        DATA_PATH / "MDM Outputs/M2W Do Something/Light_Rail_Demand_Revenue_2024.csv",
        usecols=[
            "Prod Station ID",
            "Prod Station Name",
            "Attr Station ID",
            "Attr Station Name",
            "Ticket ID",
            "Time Period ID",
            "Demand",
        ],
    )
    future = pd.read_csv(
        DATA_PATH / "MDM Outputs/M2W Do Something/Light_Rail_Demand_Revenue_2050.csv",
        usecols=[
            "Prod Station ID",
            "Prod Station Name",
            "Attr Station ID",
            "Attr Station Name",
            "Ticket ID",
            "Time Period ID",
            "Demand",
        ],
    )

    demand = baseline.merge(
        future[
            ["Prod Station ID", "Attr Station ID", "Time Period ID", "Ticket ID", "Demand"]
        ],
        on=["Prod Station ID", "Attr Station ID", "Time Period ID", "Ticket ID"],
        how="inner",
        suffixes=(f"_{CURRENT}", f"_{FORECAST}"),
        validate="one_to_one",
    )

    # Remove Murton Gap station (not in scope)
    len_before_filter = len(demand)
    demand = demand[
        (demand["Prod Station Name"] != "Murton Gap")
        | (demand["Attr Station Name"] != "Murton Gap")
    ]
    LOG.debug(
        "Filtered out Murton Gap station: %d rows removed.", len_before_filter - len(demand)
    )

    # Create lookup between demand station IDs and network station IDs and translate
    snapped_metro_stations = gpd.read_file(
        MODEL_INPUT_PATH / NEXUS_METRO_STATIONS_MODEL_INPUT_PATH
    )
    station_id_lookup = (
        snapped_metro_stations[["id", "Name"]].merge(
            baseline[["Prod Station ID", "Prod Station Name"]].drop_duplicates(),
            left_on="Name",
            right_on="Prod Station Name",
            how="left",
        )
    )[["id", "Prod Station ID"]].rename(columns={"Prod Station ID": "Demand ID"})
    demand[["Prod Station ID", "Attr Station ID"]] = demand[
        ["Prod Station ID", "Attr Station ID"]
    ].replace(station_id_lookup.set_index("Demand ID")["id"])

    # Aggregate OD data by summing over all origin-destination pairs
    len_before_agg = len(demand)
    demand = _aggregate_nexus_demand(demand)
    LOG.debug(
        "Aggregated nexus demand: %d rows reduced to %d rows.", len_before_agg, len(demand)
    )

    # Map onto network links between stations
    metro_flows = _map_demand_to_metro_links(demand)

    metro_flows.to_file(
        MODEL_INPUT_PATH / NEXUS_METRO_LINK_FLOWS_MODEL_INPUT_PATH,
        driver="GPKG"
    )

    return metro_flows


def _aggregate_nexus_demand(demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nexus demand data by origin-destination pairs."""
    demand_agg = demand.groupby(["Prod Station ID", "Attr Station ID"], as_index=False)[
        [f"Demand_{CURRENT}", f"Demand_{FORECAST}"]
    ].sum()

    demand_agg["station_a"] = demand_agg[["Prod Station ID", "Attr Station ID"]].min(axis=1)
    demand_agg["station_b"] = demand_agg[["Prod Station ID", "Attr Station ID"]].max(axis=1)

    return demand_agg.groupby(["station_a", "station_b"], as_index=False)[
        [f"Demand_{CURRENT}", f"Demand_{FORECAST}"]
    ].sum()


def _map_demand_to_metro_links(
    demand: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Map aggregated nexus demand onto metro network links."""
    metro_network = gpd.read_file(
        MODEL_INPUT_PATH / NEXUS_METRO_LINKS_MODEL_INPUT_PATH
    )

    routable_network = metro_network[
        metro_network["from_station_id"].notna() & metro_network["to_station_id"].notna()
    ].copy()

    metro_graph = nx.Graph()
    for _, row in routable_network.iterrows():
        metro_graph.add_edge(
            row["from_station_id"],
            row["to_station_id"],
            metro_link_id=row["metro_link_id"],
            demand_current=0,
            demand_forecast=0,
        )

    for _, row in demand.iterrows():
        origin = row["station_a"]
        destination = row["station_b"]

        # Find the shortest path between the origin and destination stations
        try:
            path = nx.shortest_path(metro_graph, source=origin, target=destination)
        except nx.NetworkXNoPath:
            LOG.warning(
                "No path found between origin %s and destination %s.", origin, destination
            )
            continue

        # Add OD demand onto each metro link along the shortest path
        for u, v in itertools.pairwise(path):
            metro_graph[u][v][f"demand_{CURRENT}"] += row[
                f"Demand_{CURRENT}"
            ]
            metro_graph[u][v][f"demand_{FORECAST}"] += row[
                f"Demand_{FORECAST}"
            ]

    # Write the flows back to the network
    demand_lookup = {}
    for _, _, data in metro_graph.edges(data=True):
        demand_lookup[data["metro_link_id"]] = {
            f"demand_{CURRENT}": data[f"demand_{CURRENT}"],
            f"demand_{FORECAST}": data[f"demand_{FORECAST}"],
        }

    metro_network["demand_current"] = metro_network["metro_link_id"].map(
        lambda x: demand_lookup.get(x, {}).get(f"demand_{CURRENT}", 0)
    )
    metro_network["demand_forecast"] = metro_network["metro_link_id"].map(
        lambda x: demand_lookup.get(x, {}).get(f"demand_{FORECAST}", 0)
    )

    return metro_network


# RUN SCRIPT

if __name__ == "__main__":
    with ctk.LogHelper(
        _NAME,
        ctk.ToolDetails(
            cvt.__package__,
            cvt.__version__,
        ),
        log_file=LOG_PATH / "asset_risk_weighting.log",
    ) as log:
        _clean_nexus_metro()
        _clean_nexus_demand()
