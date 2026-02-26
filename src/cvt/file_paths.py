"""Module containing file paths for model inputs and interim outputs."""

import pathlib

# Infrasructure Model Inputs Paths
OS_ROAD_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure") / "Road" / "TfN OS Road" / "tfn_os_road.gpkg"
)
NOHAM_NETWORK_MODEL_INPUT_PATH = pathlib.Path("Infrastructure") / "Road" / "TfN NoHAM"
PASSENGER_RAIL_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Rail"
    / "TfN OS Passenger Rail"
    / "tfn_pass_rail_links.gpkg"
)
FREIGHT_RAIL_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Rail"
    / "TfN OS Freight Rail"
    / "tfn_freight_rail_links.gpkg"
)
AIRPORTS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure") / "Other" / "TfN Airports" / "tfn_airports.gpkg"
)
BUS_STOPS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure") / "Other" / "TfN Bus Stops" / "tfn_bus_stops.gpkg"
)
PETROL_STATIONS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN Petrol Stations"
    / "tfn_petrol_stations.gpkg"
)
TRAIN_STATIONS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN OS Train Stations"
    / "tfn_train_stations.gpkg"
)
TRAM_STATIONS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN OS Tram Stations"
    / "tfn_tram_stations.gpkg"
)
RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN OS Rapid Transport Stations"
    / "tfn_rapid_transport_stations.gpkg"
)
FERRY_TERMINALS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN OS Ferry Terminals"
    / "tfn_ferry_terminals.gpkg"
)
BUS_COACH_STATIONS_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN OS Bus Coach Stations"
    / "tfn_bus_coach_stations.gpkg"
)
TRAM_NETWORK_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure") / "Other" / "TfN OS Tram Network" / "tfn_os_tram_links.gpkg"
)
RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure")
    / "Other"
    / "TfN Rapid Transport Network"
    / "tfn_rapid_transport_links.gpkg"
)
CHARGING_SITES_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure") / "Other" / "TfN EV Charging Sites" / "tfn_chg_sites.gpkg"
)
NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH = (
    pathlib.Path("Infrastructure") / "Other" / "TfN NCN" / "tfn_ncn.gpkg"
)

# Hazards Model Inputs Paths
HAZARD_GRID_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Extreme Weather" / "TfN Hazard Grid" / "tfn_hazard_grid.gpkg"
)
TEMP_MAX_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Summer Max Temperature Change Projections"
    / "tfn_temp_max.csv"
)
TEMP_MIN_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Winter Min Temperature Change Projections"
    / "tfn_temp_min.csv"
)
SUMMER_PRECIP_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Summer Precipitation Change Projections"
    / "tfn_precip_sum.csv"
)
WINTER_PRECIP_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Winter Precipitation Change Projections"
    / "tfn_precip_win.csv"
)
RAIN_DAYS_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN 10mm Rain Days 1991-2020"
    / "tfn_rain_days.gpkg"
)
DROUGHT_INDEX_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Drought Severity Index"
    / "tfn_drought_index.gpkg"
)
HOT_SUMMER_DAYS_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Hot Summer Days Projections"
    / "tfn_hot_days.csv"
)
EXTREME_SUMMER_DAYS_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Extreme Summer Days Projections"
    / "tfn_extr_days.csv"
)
FROST_DAYS_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Frost Days Projections"
    / "tfn_frost_days.csv"
)
ICING_DAYS_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Icing Days Projections"
    / "tfn_ice_days.csv"
)
WIND_SPEED_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards")
    / "Extreme Weather"
    / "TfN Wind Speed Projections"
    / "tfn_windspd.gpkg"
)
WIND_DRIVEN_RAIN_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Extreme Weather" / "TfN Wind Driven Rain Index" / "tfn_wdr.gpkg"
)
FLOOD_RIVERS_SEA_CLIMATE_CHANGE_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Flooding" / "TfN RoFRS CC" / "tfn_rofrs_cc.gpkg"
)
FLOOD_RIVERS_SEA_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Flooding" / "TfN RoFRS" / "tfn_rofrs.gpkg"
)
FLOOD_SURFACE_WATER_CLIMATE_CHANGE_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Flooding" / "TfN RoFSW CC" / "tfn_rofsw_cc.gpkg"
)
FLOOD_SURFACE_WATER_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Flooding" / "TfN RoFSW" / "tfn_rofsw.gpkg"
)
GEOSURE_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Ground Stability" / "TfN GeoSure" / "tfn_geosure.gpkg"
)
GEOCLIMATE_SHRINK_SWELL_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Ground Stability" / "BGS Shrink Swell"
)
GROUND_INSTABILITY_ZONES_MODEL_INPUT_PATH = (
    pathlib.Path("Hazards") / "Coastal Erosion" / "NCERM" / "tfn_ncerm_giz.gpkg"
)
NCERM_MODEL_INPUT_PATH = pathlib.Path("Hazards") / "Coastal Erosion" / "NCERM"

# Impact Model Inputs Paths
FREIGHT_DEMAND_MODEL_INPUT_PATH = (
    pathlib.Path("Impact") / "TfN Freight Flows" / "tfn_freight_network_demand.gpkg"
)
NOHAM_FLOWS_MODEL_INPUT_PATH = pathlib.Path("Impact") / "TfN NoHAM Flows"

# Model Interim Output Paths
EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH = (
    pathlib.Path("TfN Extreme Weather Risk") / "tfn_extreme_weather_risk.gpkg"
)
FLOOD_GRID_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path("Other") / "flood_grid.gpkg"
FLOOD_RISK_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path("TfN Flood Risk") / "tfn_flood_risk.gpkg"

TILE_GRID_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path("Other") / "tiles.gpkg"
FLOOD_RISK_TILE_MODEL_INTERIM_OUTPUT_PATH = (
    pathlib.Path("TfN Flood Risk") / "tfn_flood_risk_tile_overlay.gpkg"
)
FLOOD_RISK_DIRECT_MODEL_INTERIM_OUTPUT_PATH = (
    pathlib.Path("TfN Flood Risk") / "tfn_flood_risk_direct.gpkg"
)
FLOOD_RISK_SCENARIO_MODEL_INTERIM_OUTPUT_PATH = pathlib.Path("TfN Flood Risk")

GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH = (
    pathlib.Path("TfN Ground Stability Risk") / "tfn_ground_stability_risk.gpkg"
)

COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH = (
    pathlib.Path("TfN Coastal Erosion Risk") / "tfn_coastal_erosion_risk.gpkg"
)
