"""Module containing file paths for model inputs and interim outputs."""

import pathlib

# Infrastructure Model Inputs Paths

OS_ROAD_MODEL_INPUT_PATH = pathlib.Path("Infrastructure/Road/OS Road/os_road.gpkg")

NOHAM_NETWORK_MODEL_INPUT_PATH = pathlib.Path("Infrastructure/Road/NoHAM")

PASSENGER_RAIL_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Rail/OS Passenger Rail/pass_rail_links.gpkg"
)

FREIGHT_RAIL_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Rail/OS Freight Rail/freight_rail_links.gpkg"
)

AIRPORTS_MODEL_INPUT_PATH = pathlib.Path("Infrastructure/Other/Airports/airports.gpkg")

BUS_STOPS_MODEL_INPUT_PATH = pathlib.Path("Infrastructure/Other/Bus Stops/bus_stops.gpkg")

PETROL_STATIONS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Petrol Stations/petrol_stations.gpkg"
)


TRAIN_STATIONS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Train Stations/train_stations.gpkg"
)
TRAM_STATIONS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Tram Stations/tram_stations.gpkg"
)
RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Rapid Transport Stations/rapid_transport_stations.gpkg"
)
FERRY_TERMINALS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Ferry Terminals/ferry_terminals.gpkg"
)
BUS_COACH_STATIONS_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Bus Coach Stations/bus_coach_stations.gpkg"
)
TRAM_NETWORK_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Tram Network/tram_network.gpkg"
)
RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/Rapid Transport Network/rapid_transport_network.gpkg"
)
CHARGING_SITES_MODEL_INPUT_PATH = pathlib.Path(
    "Infrastructure/Other/EV Charging Sites/chg_sites.gpkg"
)
NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH = pathlib.Path("Infrastructure/Other/NCN/ncn.gpkg")

# Hazards Model Inputs Paths
HAZARD_GRID_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Hazard Grid/hazard_grid.gpkg"
)

TEMP_MAX_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Summer Max Temperature Change Projections/temp_max.csv"
)
TEMP_MIN_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Winter Min Temperature Change Projections/temp_min.csv"
)

SUMMER_PRECIP_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Summer Precipitation Change Projections/precip_sum.csv"
)
WINTER_PRECIP_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Winter Precipitation Change Projections/precip_win.csv"
)

RAIN_DAYS_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/10mm Rain Days 1991-2020/rain_days.gpkg"
)
DROUGHT_INDEX_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Drought Severity Index/drought_index.gpkg"
)
HOT_SUMMER_DAYS_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Hot Summer Days Projections/hot_days.csv"
)
EXTREME_SUMMER_DAYS_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Extreme Summer Days Projections/extr_days.csv"
)
FROST_DAYS_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Frost Days Projections/frost_days.csv"
)
ICING_DAYS_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Icing Days Projections/ice_days.csv"
)
WIND_SPEED_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Wind Speed Projections/windspd.gpkg"
)

WIND_DRIVEN_RAIN_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Extreme Weather/Wind Driven Rain Index/wdr.gpkg"
)

FLOODING_MODEL_INPUT_PATH = pathlib.Path("Hazards/Flooding")


GEOSURE_MODEL_INPUT_PATH = pathlib.Path("Hazards/Ground Stability/GeoSure/geosure.gpkg")
GEOCLIMATE_SHRINK_SWELL_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Ground Stability/BGS Shrink Swell"
)
GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH = pathlib.Path(
    "Hazards/Coastal Erosion/NCERM/ncerm_giz.gpkg"
)
NCERM_MODEL_INPUT_PATH = pathlib.Path("Hazards/Coastal Erosion/NCERM")

# Impact Model Inputs Paths
FREIGHT_DEMAND_MODEL_INPUT_PATH = pathlib.Path(
    "Impact/Freight Flows/freight_network_demand.gpkg"
)
NOHAM_FLOWS_MODEL_INPUT_PATH = pathlib.Path("Impact/NoHAM Flows/noham_net_flows.gpkg")

# Model Interim Output Paths
EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path(
    "Extreme Weather Risk/extreme_weather_risk.gpkg"
)

TILE_GRID_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path("Flood Risk/flood_tiles.gpkg")
FLOODING_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path(
    "Flooding Risk/flooding_risk_tile_overlay.gpkg"
)
FLOODING_RISK_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path("Flooding Risk/flooding_risk.gpkg")

GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path(
    "Ground Stability Risk/ground_stability_risk.gpkg"
)

COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path(
    "Coastal Erosion Risk/coastal_erosion_risk.gpkg"
)
