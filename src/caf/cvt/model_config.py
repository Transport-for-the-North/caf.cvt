"""Module for setting up the model configuration."""

import pathlib
from typing import Self

import caf.toolkit as ctk
import pydantic

from caf.cvt.definitions import (
    FLOODING_TILE_SIZE_M,
    FloodingTypes,
    GroundStabilityRiskCols,
    Scenarios,
)

### CONFIG SET UP


class ZipFileEntry(ctk.BaseConfig):
    """Configuration for a file within a zip archive.

    Attributes
    ----------
    zip_path : pathlib.Path
        Path to the zip file.
    file_path : pathlib.Path
        Path within the zip file.
    """

    zip_path: pathlib.Path
    file_path: pathlib.Path


class NestedZipFileEntry(ctk.BaseConfig):
    """Configuration for a nested zip file entry.

    Attributes
    ----------
    outer_zip_path : pathlib.Path
        Path to the outer zip file.
    inner_zip_path : pathlib.Path
        Path to the inner zip file within the outer zip.
    file_path : pathlib.Path
        Path to the file within the inner zip file.
    """

    outer_zip_path: pathlib.Path
    inner_zip_path: pathlib.Path
    file_path: pathlib.Path


# -------------------------
# CATEGORY CONFIG CLASSES
# -------------------------


class PathConfig(ctk.BaseConfig):
    """Configuration for the base module paths.

    Attributes
    ----------
    root : pydantic.DirectoryPath
        Root directory for the project.
    raw_input : pydantic.DirectoryPath
        Directory for raw input data.
    """

    root: pydantic.DirectoryPath
    raw_input: pydantic.DirectoryPath

    @property
    def model_input(self) -> pathlib.Path:
        """Create model input directory and return path."""
        model_input = self.root / "model input"
        model_input.mkdir(parents=True, exist_ok=True)
        return model_input

    @property
    def model_interim_output(self) -> pathlib.Path:
        """Create model interim output directory and return path."""
        model_interim_output = self.root / "model interim outputs"
        model_interim_output.mkdir(parents=True, exist_ok=True)
        return model_interim_output

    @property
    def model_output(self) -> pathlib.Path:
        """Create model output directory and return path."""
        model_output = self.root / "model outputs"
        model_output.mkdir(parents=True, exist_ok=True)
        return model_output

    @property
    def log(self) -> pathlib.Path:
        """Create logging directory and return path."""
        log_path = self.root / "Logging"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    @property
    def audit(self) -> pathlib.Path:
        """Create model audit directory and return path."""
        audit_path = self.root / "audit"
        audit_path.mkdir(parents=True, exist_ok=True)
        return audit_path

    @property
    def infrastructure_model_input(self) -> pathlib.Path:
        """Create infrastructure model input directory and return path."""
        infrastructure_model_input = self.model_input / "Infrastructure"
        infrastructure_model_input.mkdir(parents=True, exist_ok=True)
        return infrastructure_model_input

    @property
    def hazards_model_input(self) -> pathlib.Path:
        """Create hazards model input directory and return path."""
        hazards_model_input = self.model_input / "Hazards"
        hazards_model_input.mkdir(parents=True, exist_ok=True)
        return hazards_model_input

    @property
    def demand_model_input(self) -> pathlib.Path:
        """Create demand model input directory and return path."""
        demand_model_input = self.model_input / "Demand"
        demand_model_input.mkdir(parents=True, exist_ok=True)
        return demand_model_input

    @property
    def road_model_input(self) -> pathlib.Path:
        """Create road model input directory and return path."""
        road_model_input = self.infrastructure_model_input / "Road"
        road_model_input.mkdir(parents=True, exist_ok=True)
        return road_model_input

    @property
    def rail_model_input(self) -> pathlib.Path:
        """Create rail model input directory and return path."""
        rail_model_input = self.infrastructure_model_input / "Rail"
        rail_model_input.mkdir(parents=True, exist_ok=True)
        return rail_model_input

    @property
    def other_model_input(self) -> pathlib.Path:
        """Create other model input directory and return path."""
        other_model_input = self.infrastructure_model_input / "Other"
        other_model_input.mkdir(parents=True, exist_ok=True)
        return other_model_input

    @property
    def bespoke_model_input(self) -> pathlib.Path:
        """Create bespoke model input directory and return path."""
        bespoke_model_input = self.model_input / "Bespoke"
        bespoke_model_input.mkdir(parents=True, exist_ok=True)
        return bespoke_model_input

    @property
    def os_road_model_input(self) -> pathlib.Path:
        """Create OS road model input directory and return path."""
        return self.road_model_input / "os_road.gpkg"

    @property
    def noham_road_model_input(self) -> pathlib.Path:
        """Create NoHAM road model input directory and return path."""
        return self.road_model_input / "noham_2023.gpkg"

    @property
    def model_roads_model_input(self) -> pathlib.Path:
        """Create model road input directory and return path."""
        return self.road_model_input / "model_roads.gpkg"

    @property
    def passenger_rail_model_input(self) -> pathlib.Path:
        """Create passenger rail model input directory and return path."""
        return self.rail_model_input / "passenger_rail_links.gpkg"

    @property
    def freight_rail_model_input(self) -> pathlib.Path:
        """Create freight rail model input directory and return path."""
        return self.rail_model_input / "freight_rail_links.gpkg"

    @property
    def airports_model_input(self) -> pathlib.Path:
        """Create airports model input directory and return path."""
        return self.other_model_input / "airports.gpkg"

    @property
    def bus_stops_model_input(self) -> pathlib.Path:
        """Create bus stops model input directory and return path."""
        return self.other_model_input / "bus_stops.gpkg"

    @property
    def train_stations_model_input(self) -> pathlib.Path:
        """Create train stations model input directory and return path."""
        return self.other_model_input / "train_stations.gpkg"

    @property
    def tram_stations_model_input(self) -> pathlib.Path:
        """Create tram stations model input directory and return path."""
        return self.other_model_input / "tram_stations.gpkg"

    @property
    def rapid_transport_stations_model_input(self) -> pathlib.Path:
        """Create rapid transport stations model input directory and return path."""
        return self.other_model_input / "rapid_transport_stations.gpkg"

    @property
    def ferry_terminals_model_input(self) -> pathlib.Path:
        """Create ferry terminals model input directory and return path."""
        return self.other_model_input / "ferry_terminals.gpkg"

    @property
    def bus_coach_stations_model_input(self) -> pathlib.Path:
        """Create bus coach stations model input directory and return path."""
        return self.other_model_input / "bus_coach_stations.gpkg"

    @property
    def tram_network_model_input(self) -> pathlib.Path:
        """Create tram network model input directory and return path."""
        return self.other_model_input / "tram_network.gpkg"

    @property
    def rapid_transport_network_model_input(self) -> pathlib.Path:
        """Create rapid transport network model input directory and return path."""
        return self.other_model_input / "rapid_transport_network.gpkg"

    @property
    def national_cycle_network_model_input(self) -> pathlib.Path:
        """Create national cycle network model input directory and return path."""
        return self.other_model_input / "national_cycle_network.gpkg"

    @property
    def nexus_metro_stations_model_input(self) -> pathlib.Path:
        """Create Nexus Metro stations model input directory and return path."""
        return self.bespoke_model_input / "nexus_metro_stations.gpkg"

    @property
    def hazard_model_input(self) -> pathlib.Path:
        """Create hazard model input directory and return path."""
        hazard_model_input_path = self.model_input / "Hazards"
        hazard_model_input_path.mkdir(parents=True, exist_ok=True)
        return hazard_model_input_path

    @property
    def extreme_weather_model_input(self) -> pathlib.Path:
        """Create extreme weather model input directory and return path."""
        extreme_weather_model_input_path = self.hazard_model_input / "Extreme Weather"
        extreme_weather_model_input_path.mkdir(parents=True, exist_ok=True)
        return extreme_weather_model_input_path

    @property
    def flooding_model_input(self) -> pathlib.Path:
        """Create flooding model input directory and return path."""
        flooding_model_input_path = self.hazard_model_input / "Flooding"
        flooding_model_input_path.mkdir(parents=True, exist_ok=True)
        return flooding_model_input_path

    @property
    def ground_stability_model_input(self) -> pathlib.Path:
        """Create ground stability model input directory and return path."""
        ground_stability_model_input_path = self.hazard_model_input / "Ground Stability"
        ground_stability_model_input_path.mkdir(parents=True, exist_ok=True)
        return ground_stability_model_input_path

    @property
    def coastal_erosion_model_input(self) -> pathlib.Path:
        """Create coastal erosion model input directory and return path."""
        coastal_erosion_model_input_path = self.hazard_model_input / "Coastal Erosion"
        coastal_erosion_model_input_path.mkdir(parents=True, exist_ok=True)
        return coastal_erosion_model_input_path

    @property
    def hazard_grid_model_input(self) -> pathlib.Path:
        """Return hazard grid path."""
        return self.hazard_model_input / "hazard_grid.gpkg"

    @property
    def summer_max_temperature_model_input(self) -> pathlib.Path:
        """Return max temperature model input path."""
        return self.extreme_weather_model_input / "summer_max_temperature_change.csv"

    @property
    def winter_min_temperature_model_input(self) -> pathlib.Path:
        """Return min temperature model input path."""
        return self.extreme_weather_model_input / "winter_min_temperature_change.csv"

    @property
    def summer_precipitation_model_input(self) -> pathlib.Path:
        """Return precipitation model input path."""
        return self.extreme_weather_model_input / "summer_precipitation_change.csv"

    @property
    def winter_precipitation_model_input(self) -> pathlib.Path:
        """Return winter precipitation model input path."""
        return self.extreme_weather_model_input / "winter_precipitation_change.csv"

    @property
    def rain_days_model_input(self) -> pathlib.Path:
        """Return rain days model input path."""
        return self.extreme_weather_model_input / "10mm_rain_days.gpkg"

    @property
    def drought_severity_index_model_input(self) -> pathlib.Path:
        """Return drought severity index model input path."""
        return self.extreme_weather_model_input / "drought_severity_index.gpkg"

    @property
    def hot_summer_days_model_input(self) -> pathlib.Path:
        """Return hot summer days model input path."""
        return self.extreme_weather_model_input / "hot_summer_days.csv"

    @property
    def extreme_summer_days_model_input(self) -> pathlib.Path:
        """Return extreme summer days model input path."""
        return self.extreme_weather_model_input / "extreme_summer_days.csv"

    @property
    def frost_days_model_input(self) -> pathlib.Path:
        """Return frost days model input path."""
        return self.extreme_weather_model_input / "frost_days.csv"

    @property
    def icing_days_model_input(self) -> pathlib.Path:
        """Return icing days model input path."""
        return self.extreme_weather_model_input / "icing_days.csv"

    @property
    def wind_speed_model_input(self) -> pathlib.Path:
        """Return wind speed model input path."""
        return self.extreme_weather_model_input / "wind_speed.gpkg"

    @property
    def wind_driven_rain_model_input(self) -> pathlib.Path:
        """Return wind driven rain model input path."""
        return self.extreme_weather_model_input / "wind_driven_rain_index.gpkg"

    def flooding_type_scenario_model_input(
        self, flooding_type: FloodingTypes, scenario: Scenarios
    ) -> pathlib.Path:
        """Return flooding model input path based on flooding type and scenario."""
        return self.flooding_model_input / f"{flooding_type}_{scenario}.gpkg"

    @property
    def groundwater_flooding_model_input(self) -> pathlib.Path:
        """Return groundwater flooding model input path."""
        return self.flooding_model_input / "groundwater.gpkg"

    def geosure_model_input(self, geohazard: GroundStabilityRiskCols | None) -> pathlib.Path:
        """Return GeoSure model input path based on the geohazard."""
        if geohazard is not None:
            return self.ground_stability_model_input / f"{geohazard}.gpkg"
        return self.ground_stability_model_input / "geosure.gpkg"

    def national_coastal_erosion_risk_mapping_model_input(self, year: str) -> pathlib.Path:
        """Return national coastal erosion risk mapping model input path."""
        return (
            self.coastal_erosion_model_input
            / f"national_coastal_erosion_risk_mapping_smp_{year}_70CC.gpkg"
        )

    @property
    def freight_demand_model_input(self) -> pathlib.Path:
        """Return freight demand model input path."""
        return self.demand_model_input / "freight_network_demand.gpkg"

    @property
    def noham_demand_model_input(self) -> pathlib.Path:
        """Return NoHAM demand model input path."""
        return self.demand_model_input / "noham_network_demand.gpkg"

    @property
    def model_roads_demand_model_input(self) -> pathlib.Path:
        """Return model roads demand model input path."""
        return self.demand_model_input / "model_roads_demand.gpkg"

    @property
    def nexus_metro_demand_model_input(self) -> pathlib.Path:
        """Return Nexus Metro demand model input path."""
        return self.demand_model_input / "nexus_metro_demand.gpkg"

    @property
    def extreme_weather_risk_model_interim_output(self) -> pathlib.Path:
        """Return extreme weather model interim output path."""
        return self.model_interim_output / "extreme_weather_risk.gpkg"

    @property
    def flooding_tile_grid_model_interim_output(self) -> pathlib.Path:
        """Return flooding tile grid model interim output path."""
        return self.model_interim_output / f"flooding_tiles_{FLOODING_TILE_SIZE_M}m.gpkg"

    @property
    def flooding_risk_vector_overlay_model_interim_output(self) -> pathlib.Path:
        """Return flooding risk vector overlay model interim output path."""
        return self.model_interim_output / "flooding_risk_vector_overlay.gpkg"

    @property
    def flooding_risk_vector_model_interim_output(self) -> pathlib.Path:
        """Return flooding risk vector model interim output path."""
        return self.model_interim_output / "flooding_risk_vector.gpkg"

    @property
    def flooding_risk_raster_model_interim_output(self) -> pathlib.Path:
        """Return flooding risk raster model interim output path."""
        return self.model_interim_output / "flooding_risk_raster.tif"

    @property
    def ground_stability_risk_model_interim_output(self) -> pathlib.Path:
        """Return ground stability risk model interim output path."""
        return self.model_interim_output / "ground_stability_risk.gpkg"

    @property
    def coastal_erosion_risk_model_interim_output(self) -> pathlib.Path:
        """Return coastal erosion risk model interim output path."""
        return self.model_interim_output / "coastal_erosion_risk.gpkg"

    @property
    def road_model_output(self) -> pathlib.Path:
        """Create road model output directory and return path."""
        road_model_output_path = self.model_output / "Road"
        road_model_output_path.mkdir(parents=True, exist_ok=True)
        return road_model_output_path

    @property
    def rail_model_output(self) -> pathlib.Path:
        """Create rail model output directory and return path."""
        rail_model_output_path = self.model_output / "Rail"
        rail_model_output_path.mkdir(parents=True, exist_ok=True)
        return rail_model_output_path

    @property
    def other_model_output(self) -> pathlib.Path:
        """Create other infrastructure model output directory and return path."""
        other_model_output_path = self.model_output / "Other"
        other_model_output_path.mkdir(parents=True, exist_ok=True)
        return other_model_output_path

    @property
    def os_road_model_output(self) -> pathlib.Path:
        """Create OS road model output directory and return path."""
        os_road_model_output_path = self.road_model_output / "OS Road"
        os_road_model_output_path.mkdir(parents=True, exist_ok=True)
        return os_road_model_output_path

    @property
    def noham_model_output(self) -> pathlib.Path:
        """Create NoHAM model output directory and return path."""
        noham_model_output_path = self.road_model_output / "NoHAM"
        noham_model_output_path.mkdir(parents=True, exist_ok=True)
        return noham_model_output_path

    @property
    def model_roads_model_output(self) -> pathlib.Path:
        """Create model roads output directory and return path."""
        model_roads_model_output_path = self.road_model_output / "Model Roads"
        model_roads_model_output_path.mkdir(parents=True, exist_ok=True)
        return model_roads_model_output_path

    @property
    def passenger_rail_model_output(self) -> pathlib.Path:
        """Create passenger rail model output directory and return path."""
        passenger_rail_model_output_path = self.rail_model_output / "Passenger Rail"
        passenger_rail_model_output_path.mkdir(parents=True, exist_ok=True)
        return passenger_rail_model_output_path

    @property
    def freight_rail_model_output(self) -> pathlib.Path:
        """Create freight rail model output directory and return path."""
        freight_rail_model_output_path = self.rail_model_output / "Freight Rail"
        freight_rail_model_output_path.mkdir(parents=True, exist_ok=True)
        return freight_rail_model_output_path

    @property
    def train_stations_model_output(self) -> pathlib.Path:
        """Create train stations model output directory and return path."""
        train_stations_model_output_path = self.other_model_output / "Train Stations"
        train_stations_model_output_path.mkdir(parents=True, exist_ok=True)
        return train_stations_model_output_path

    @property
    def airports_model_output(self) -> pathlib.Path:
        """Create airports model output directory and return path."""
        airports_model_output_path = self.other_model_output / "Airports"
        airports_model_output_path.mkdir(parents=True, exist_ok=True)
        return airports_model_output_path

    @property
    def bus_coach_stations_model_output(self) -> pathlib.Path:
        """Create bus and coach stations model output directory and return path."""
        bus_coach_stations_model_output_path = (
            self.other_model_output / "Bus and Coach Stations"
        )
        bus_coach_stations_model_output_path.mkdir(parents=True, exist_ok=True)
        return bus_coach_stations_model_output_path

    @property
    def bus_stops_model_output(self) -> pathlib.Path:
        """Create bus stops model output directory and return path."""
        bus_stops_model_output_path = self.other_model_output / "Bus Stops"
        bus_stops_model_output_path.mkdir(parents=True, exist_ok=True)
        return bus_stops_model_output_path

    @property
    def tram_stations_model_output(self) -> pathlib.Path:
        """Create tram stations model output directory and return path."""
        tram_stations_model_output_path = self.other_model_output / "Tram Stations"
        tram_stations_model_output_path.mkdir(parents=True, exist_ok=True)
        return tram_stations_model_output_path

    @property
    def rapid_transport_stations_model_output(self) -> pathlib.Path:
        """Create rapid transport stations model output directory and return path."""
        rapid_transport_stations_model_output_path = (
            self.other_model_output / "Rapid Transport Stations"
        )
        rapid_transport_stations_model_output_path.mkdir(parents=True, exist_ok=True)
        return rapid_transport_stations_model_output_path

    @property
    def ferry_terminals_model_output(self) -> pathlib.Path:
        """Create ferry terminals model output directory and return path."""
        ferry_terminals_model_output_path = self.other_model_output / "Ferry Terminals"
        ferry_terminals_model_output_path.mkdir(parents=True, exist_ok=True)
        return ferry_terminals_model_output_path

    @property
    def national_cycle_network_model_output(self) -> pathlib.Path:
        """Create national cycle network model output directory and return path."""
        national_cycle_network_model_output_path = (
            self.other_model_output / "National Cycle Network"
        )
        national_cycle_network_model_output_path.mkdir(parents=True, exist_ok=True)
        return national_cycle_network_model_output_path

    @property
    def tram_network_model_output(self) -> pathlib.Path:
        """Create tram network model output directory and return path."""
        tram_network_model_output_path = self.other_model_output / "Tram Network"
        tram_network_model_output_path.mkdir(parents=True, exist_ok=True)
        return tram_network_model_output_path

    @property
    def rapid_transport_network_model_output(self) -> pathlib.Path:
        """Create rapid transport network model output directory and return path."""
        rapid_transport_network_model_output_path = (
            self.other_model_output / "Rapid Transport Network"
        )
        rapid_transport_network_model_output_path.mkdir(parents=True, exist_ok=True)
        return rapid_transport_network_model_output_path

    @property
    def nexus_metro_links_model_output(self) -> pathlib.Path:
        """Create Nexus Metro links model output directory and return path."""
        nexus_metro_links_model_output_path = self.other_model_output / "Nexus Metro Links"
        nexus_metro_links_model_output_path.mkdir(parents=True, exist_ok=True)
        return nexus_metro_links_model_output_path

    @property
    def nexus_metro_stations_model_output(self) -> pathlib.Path:
        """Create Nexus Metro stations model output directory and return path."""
        nexus_metro_stations_model_output_path = (
            self.other_model_output / "Nexus Metro Stations"
        )
        nexus_metro_stations_model_output_path.mkdir(parents=True, exist_ok=True)
        return nexus_metro_stations_model_output_path

    @property
    def functional_rules_audit(self) -> pathlib.Path:
        """Create functional rules audit model output directory and return path."""
        functional_rules_audit_path = self.audit / "Functional Rules"
        functional_rules_audit_path.mkdir(parents=True, exist_ok=True)
        return functional_rules_audit_path

    @property
    def layering_audit(self) -> pathlib.Path:
        """Create layering audit model output directory and return path."""
        layering_audit_path = self.audit / "Layering"
        layering_audit_path.mkdir(parents=True, exist_ok=True)
        return layering_audit_path

    @property
    def summary_audit(self) -> pathlib.Path:
        """Create summary audit model output directory and return path."""
        summary_audit_path = self.audit / "Summary"
        summary_audit_path.mkdir(parents=True, exist_ok=True)
        return summary_audit_path

    @property
    def extreme_weather_audit(self) -> pathlib.Path:
        """Create extreme weather audit model output directory and return path."""
        extreme_weather_audit_path = self.functional_rules_audit / "Extreme Weather"
        extreme_weather_audit_path.mkdir(parents=True, exist_ok=True)
        return extreme_weather_audit_path

    @property
    def flooding_audit(self) -> pathlib.Path:
        """Create flooding audit model output directory and return path."""
        flooding_audit_path = self.functional_rules_audit / "Flooding"
        flooding_audit_path.mkdir(parents=True, exist_ok=True)
        return flooding_audit_path

    @property
    def ground_stability_audit(self) -> pathlib.Path:
        """Create ground stability audit model output directory and return path."""
        ground_stability_audit_path = self.functional_rules_audit / "Ground Stability"
        ground_stability_audit_path.mkdir(parents=True, exist_ok=True)
        return ground_stability_audit_path

    @property
    def coastal_erosion_audit(self) -> pathlib.Path:
        """Create coastal erosion audit model output directory and return path."""
        coastal_erosion_audit_path = self.functional_rules_audit / "Coastal Erosion"
        coastal_erosion_audit_path.mkdir(parents=True, exist_ok=True)
        return coastal_erosion_audit_path

    @property
    def extreme_heat_audit(self) -> pathlib.Path:
        """Create extreme heat audit model output directory and return path."""
        extreme_heat_audit_path = self.extreme_weather_audit / "Extreme Heat Risk Index"
        extreme_heat_audit_path.mkdir(parents=True, exist_ok=True)
        return extreme_heat_audit_path

    @property
    def extreme_cold_audit(self) -> pathlib.Path:
        """Create extreme cold audit model output directory and return path."""
        extreme_cold_audit_path = self.extreme_weather_audit / "Extreme Cold Risk Index"
        extreme_cold_audit_path.mkdir(parents=True, exist_ok=True)
        return extreme_cold_audit_path

    @property
    def storm_audit(self) -> pathlib.Path:
        """Create storm audit model output directory and return path."""
        storm_audit_path = self.extreme_weather_audit / "Storm Risk Index"
        storm_audit_path.mkdir(parents=True, exist_ok=True)
        return storm_audit_path

    @property
    def drought_audit(self) -> pathlib.Path:
        """Create drought audit model output directory and return path."""
        drought_audit_path = self.extreme_weather_audit / "Drought Risk Index"
        drought_audit_path.mkdir(parents=True, exist_ok=True)
        return drought_audit_path

    @property
    def extreme_weather_index_audit(self) -> pathlib.Path:
        """Create extreme weather index audit model output directory and return path."""
        extreme_weather_index_audit_path = (
            self.extreme_weather_audit / "Extreme Weather Risk Index"
        )
        extreme_weather_index_audit_path.mkdir(parents=True, exist_ok=True)
        return extreme_weather_index_audit_path

    @property
    def road_layering_audit(self) -> pathlib.Path:
        """Create road layering audit model output directory and return path."""
        road_layering_audit_path = self.layering_audit / "Road"
        road_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return road_layering_audit_path

    @property
    def rail_layering_audit(self) -> pathlib.Path:
        """Create rail layering audit model output directory and return path."""
        rail_layering_audit_path = self.layering_audit / "Rail"
        rail_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return rail_layering_audit_path

    @property
    def other_layering_audit(self) -> pathlib.Path:
        """Create other layering audit model output directory and return path."""
        other_layering_audit_path = self.layering_audit / "Other"
        other_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return other_layering_audit_path

    @property
    def bespoke_layering_audit(self) -> pathlib.Path:
        """Create bespoke layering audit model output directory and return path."""
        bespoke_layering_audit_path = self.layering_audit / "Bespoke"
        bespoke_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return bespoke_layering_audit_path

    @property
    def os_road_layering_audit(self) -> pathlib.Path:
        """Create OS road layering audit model output directory and return path."""
        os_road_layering_audit_path = self.road_layering_audit / "OS Roads"
        os_road_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return os_road_layering_audit_path

    @property
    def noham_layering_audit(self) -> pathlib.Path:
        """Create NOHAM layering audit model output directory and return path."""
        noham_layering_audit_path = self.road_layering_audit / "NoHAM"
        noham_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return noham_layering_audit_path

    @property
    def model_roads_layering_audit(self) -> pathlib.Path:
        """Create model roads layering audit model output directory and return path."""
        model_roads_layering_audit_path = self.road_layering_audit / "Model Roads"
        model_roads_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return model_roads_layering_audit_path

    @property
    def passenger_rail_layering_audit(self) -> pathlib.Path:
        """Create passenger rail layering audit model output directory and return path."""
        passenger_rail_layering_audit_path = self.rail_layering_audit / "Passenger Rail"
        passenger_rail_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return passenger_rail_layering_audit_path

    @property
    def freight_rail_layering_audit(self) -> pathlib.Path:
        """Create freight rail layering audit model output directory and return path."""
        freight_rail_layering_audit_path = self.rail_layering_audit / "Freight Rail"
        freight_rail_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return freight_rail_layering_audit_path

    @property
    def train_stations_layering_audit(self) -> pathlib.Path:
        """Create train stations layering audit model output directory and return path."""
        train_stations_layering_audit_path = self.other_layering_audit / "Train Stations"
        train_stations_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return train_stations_layering_audit_path

    @property
    def airports_layering_audit(self) -> pathlib.Path:
        """Create airports layering audit model output directory and return path."""
        airports_layering_audit_path = self.other_layering_audit / "Airports"
        airports_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return airports_layering_audit_path

    @property
    def bus_coach_stations_layering_audit(self) -> pathlib.Path:
        """Create bus & coach stations layering audit model output directory and path."""
        bus_coach_stations_layering_audit_path = (
            self.other_layering_audit / "Bus and Coach Stations"
        )
        bus_coach_stations_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return bus_coach_stations_layering_audit_path

    @property
    def bus_stops_layering_audit(self) -> pathlib.Path:
        """Create bus stops layering audit model output directory and return path."""
        bus_stops_layering_audit_path = self.other_layering_audit / "Bus Stops"
        bus_stops_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return bus_stops_layering_audit_path

    @property
    def tram_stations_layering_audit(self) -> pathlib.Path:
        """Create tram stations layering audit model output directory and return path."""
        tram_stations_layering_audit_path = self.other_layering_audit / "Tram Stations"
        tram_stations_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return tram_stations_layering_audit_path

    @property
    def rapid_transport_stations_layering_audit(self) -> pathlib.Path:
        """Create rapid transport stations layering audit model output directory and path."""
        rapid_transport_stations_layering_audit_path = (
            self.other_layering_audit / "Rapid Transport Stations"
        )
        rapid_transport_stations_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return rapid_transport_stations_layering_audit_path

    @property
    def ferry_terminals_layering_audit(self) -> pathlib.Path:
        """Create ferry terminals layering audit model output directory and return path."""
        ferry_terminals_layering_audit_path = self.other_layering_audit / "Ferry Terminals"
        ferry_terminals_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return ferry_terminals_layering_audit_path

    @property
    def national_cycle_network_layering_audit(self) -> pathlib.Path:
        """Create national cycle network layering audit model output directory and path."""
        national_cycle_network_layering_audit_path = (
            self.other_layering_audit / "National Cycle Network"
        )
        national_cycle_network_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return national_cycle_network_layering_audit_path

    @property
    def tram_network_layering_audit(self) -> pathlib.Path:
        """Create tram network layering audit model output directory and return path."""
        tram_network_layering_audit_path = self.other_layering_audit / "Tram Network"
        tram_network_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return tram_network_layering_audit_path

    @property
    def rapid_transport_network_layering_audit(self) -> pathlib.Path:
        """Create rapid transport network layering audit model output directory and path."""
        rapid_transport_network_layering_audit_path = (
            self.other_layering_audit / "Rapid Transport Network"
        )
        rapid_transport_network_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return rapid_transport_network_layering_audit_path

    @property
    def nexus_metro_links_layering_audit(self) -> pathlib.Path:
        """Create Nexus Metro links layering audit model output directory and path."""
        nexus_metro_links_layering_audit_path = (
            self.bespoke_layering_audit / "Nexus Metro Links"
        )
        nexus_metro_links_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return nexus_metro_links_layering_audit_path

    @property
    def nexus_metro_stations_layering_audit(self) -> pathlib.Path:
        """Create Nexus Metro stations layering audit model output directory and path."""
        nexus_metro_stations_layering_audit_path = (
            self.bespoke_layering_audit / "Nexus Metro Stations"
        )
        nexus_metro_stations_layering_audit_path.mkdir(parents=True, exist_ok=True)
        return nexus_metro_stations_layering_audit_path

    @property
    def road_summary_audit(self) -> pathlib.Path:
        """Create road summary audit model output directory and return path."""
        road_summary_audit_path = self.summary_audit / "Road"
        road_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return road_summary_audit_path

    @property
    def rail_summary_audit(self) -> pathlib.Path:
        """Create rail summary audit model output directory and return path."""
        rail_summary_audit_path = self.summary_audit / "Rail"
        rail_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return rail_summary_audit_path

    @property
    def other_summary_audit(self) -> pathlib.Path:
        """Create other summary audit model output directory and return path."""
        other_summary_audit_path = self.summary_audit / "Other"
        other_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return other_summary_audit_path

    @property
    def bespoke_summary_audit(self) -> pathlib.Path:
        """Create bespoke summary audit model output directory and return path."""
        bespoke_summary_audit_path = self.summary_audit / "Bespoke"
        bespoke_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return bespoke_summary_audit_path

    @property
    def os_road_summary_audit(self) -> pathlib.Path:
        """Create OS road summary audit model output directory and return path."""
        os_road_summary_audit_path = self.road_summary_audit / "OS Roads"
        os_road_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return os_road_summary_audit_path

    @property
    def noham_summary_audit(self) -> pathlib.Path:
        """Create NoHAM summary audit model output directory and return path."""
        noham_summary_audit_path = self.road_summary_audit / "NoHAM"
        noham_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return noham_summary_audit_path

    @property
    def model_road_summary_audit(self) -> pathlib.Path:
        """Create model road summary audit model output directory and return path."""
        model_road_summary_audit_path = self.road_summary_audit / "Model Roads"
        model_road_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return model_road_summary_audit_path

    @property
    def passenger_rail_summary_audit(self) -> pathlib.Path:
        """Create passenger rail summary audit model output directory and return path."""
        passenger_rail_summary_audit_path = self.rail_summary_audit / "Passenger Rail"
        passenger_rail_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return passenger_rail_summary_audit_path

    @property
    def freight_rail_summary_audit(self) -> pathlib.Path:
        """Create freight rail summary audit model output directory and return path."""
        freight_rail_summary_audit_path = self.rail_summary_audit / "Freight Rail"
        freight_rail_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return freight_rail_summary_audit_path

    @property
    def national_cycle_network_summary_audit(self) -> pathlib.Path:
        """Create national cycle network summary audit model output directory and path."""
        national_cycle_network_summary_audit_path = (
            self.other_summary_audit / "National Cycle Network"
        )
        national_cycle_network_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return national_cycle_network_summary_audit_path

    @property
    def tram_network_summary_audit(self) -> pathlib.Path:
        """Create tram network summary audit model output directory and return path."""
        tram_network_summary_audit_path = self.other_summary_audit / "Tram Network"
        tram_network_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return tram_network_summary_audit_path

    @property
    def rapid_transport_network_summary_audit(self) -> pathlib.Path:
        """Create rapid transport network summary audit model output directory and path."""
        rapid_transport_network_summary_audit_path = (
            self.other_summary_audit / "Rapid Transport Network"
        )
        rapid_transport_network_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return rapid_transport_network_summary_audit_path

    @property
    def nexus_metro_links_summary_audit(self) -> pathlib.Path:
        """Create Nexus Metro links summary audit model output directory and return path."""
        nexus_metro_links_summary_audit_path = self.bespoke_summary_audit / "Nexus Metro Links"
        nexus_metro_links_summary_audit_path.mkdir(parents=True, exist_ok=True)
        return nexus_metro_links_summary_audit_path


class OtherInput(ctk.BaseConfig):
    """Configuration for other raw input data.

    Attributes
    ----------
    stb_path : pathlib.Path
        Path to the boundary file of all STBs.
    ca_path : pathlib.Path
        Path to the boundary file of all CAs.
    boundary_path : pathlib.Path | None = None
        Path to the specific boundary file of the region which the model is running for.
    """

    stb_path: pathlib.Path
    ca_path: pathlib.Path
    boundary_path: pydantic.FilePath | None = None

    # Empty yaml values are loaded as strings, so need to convert to None
    @pydantic.field_validator("boundary_path", mode="before")
    @classmethod
    def _empty_to_none(
        cls, v: str | pydantic.FilePath | None
    ) -> str | pydantic.FilePath | None:
        if v == "":
            return None
        return v


class ModelRoadsEntry(ctk.BaseConfig):
    """Configuration for the transport model road network data.

    Attributes
    ----------
    nodes: pathlib.Path
        Path to the nodes CSV file.
    links: pathlib.Path
        Path to the links CSV file.
    shaped_links: pathlib.Path
        Path to the shaped links shapefile.
    """

    nodes: pathlib.Path
    links: pathlib.Path
    shaped_links: pathlib.Path


class Road(ctk.BaseConfig):
    """Configuration for road infrastructure data.

    Attributes
    ----------
    os_road : ZipFileEntry
        Configuration for the OS road zip file entry.
    os_road_lookup: NestedZipFileEntry
        Configuration for the nested OS road lookup zip file entry.
    mm_highways: ZipFileEntry
        Configuration for the MasterMap Highways Network zip file entry.
    noham: NoHAMEntry
        Configuration for the NoHAM road network data.
    model_roads: ModelRoadsEntry
        Configuration for the transport model road network data.
    """

    os_road: ZipFileEntry
    os_road_lookup: NestedZipFileEntry
    mm_highways: ZipFileEntry
    noham: pathlib.Path
    model_roads: ModelRoadsEntry


class Rail(ctk.BaseConfig):
    """Configuration for rail infrastructure data.

    Attributes
    ----------
    rail_links : ZipFileEntry
        Configuration for the rail links zip file entry.
    """

    rail_links: ZipFileEntry


class Other(ctk.BaseConfig):
    """Configuration for other infrastructure data.

    Attributes
    ----------
    bus_stops : pathlib.Path
        Path to the bus stops data.
    ncn_sustrans : pathlib.Path
        Path to the NCN Sustrans data.
    mastermap_sites: ZipFileEntry
        Configuration for the MasterMap Sites zip file entry.
    """

    bus_stops: pathlib.Path
    ncn_sustrans: pathlib.Path
    mastermap_sites: ZipFileEntry


class InfrastructureConfig(ctk.BaseConfig):
    """Configuration for infrastructure data.

    Attributes
    ----------
    road : Road
        Configuration for road infrastructure data.
    rail : Rail
        Configuration for rail infrastructure data.
    other : Other
        Configuration for other infrastructure data.
    """

    road: Road
    rail: Rail
    other: Other


class ExtremeWeather(ctk.BaseConfig):
    """Configuration for extreme weather data.

    Attributes
    ----------
    wind_spd_current : pathlib.Path
        Path to the current wind speed data.
    wind_spd_forecast : pathlib.Path
        Path to the forecast wind speed data.
    rain_days : ZipFileEntry
        Configuration for rain days zip file entry.
    extreme_summer_days : ZipFileEntry
        Configuration for extreme summer days zip file entry.
    frost_days : ZipFileEntry
        Configuration for frost days zip file entry.
    hot_days : ZipFileEntry
        Configuration for hot days zip file entry.
    icing_days : ZipFileEntry
        Configuration for icing days zip file entry.
    wdr_index : ZipFileEntry
        Configuration for WDR index zip file entry.
    drought_index : ZipFileEntry
        Configuration for drought index zip file entry.
    max_temp_summer : ZipFileEntry
        Configuration for max temperature summer zip file entry.
    precip_summer : ZipFileEntry
        Configuration for precipitation summer zip file entry.
    min_temp_winter : ZipFileEntry
        Configuration for min temperature winter zip file entry.
    precip_winter : ZipFileEntry
        Configuration for precipitation winter zip file entry.
    """

    wind_speed: dict[str, pathlib.Path]
    rain_days: pathlib.Path
    extreme_summer_days: pathlib.Path
    frost_days: pathlib.Path
    hot_days: pathlib.Path
    icing_days: pathlib.Path
    wdr_index: pathlib.Path
    drought_index: pathlib.Path
    max_temp_summer: pathlib.Path
    precip_summer: pathlib.Path
    min_temp_winter: pathlib.Path
    precip_winter: pathlib.Path


class PremiumGeoSureEntry(ctk.BaseConfig):
    """Configuration for the premium GeoSure data.

    Attributes
    ----------
    collapsible_deposits : pathlib.Path
        Configuration for collapsible deposits shapefile.
    compressible_ground : pathlib.Path
        Configuration for compressible ground shapefile.
    landslides : pathlib.Path
        Configuration for landslides shapefile.
    running_sands : pathlib.Path
        Configuration for running sands shapefile.
    shrink_swell : pathlib.Path
        Configuration for shrink swell shapefile.
    soluble_rocks : pathlib.Path
        Configuration for soluble rocks shapefile.
    """

    collapsible_deposits: pathlib.Path
    compressible_ground: pathlib.Path
    landslides: pathlib.Path
    running_sands: pathlib.Path
    shrink_swell: pathlib.Path
    soluble_rocks: pathlib.Path


class GeoSureEntry(ctk.BaseConfig):
    """Configuration for GeoSure data.

    Attributes
    ----------
    open: ZipFileEntry
       Configuration for the open GeoSure zip file entry.
    premium: PremiumGeoSureEntry
       Configuration for the premium GeoSure entry.
    """

    open: ZipFileEntry
    premium: PremiumGeoSureEntry


class GroundStability(ctk.BaseConfig):
    """Configuration for ground stability data.

    Attributes
    ----------
    geosure : GeoSure
        Configuration for GeoSure data.
    """

    geosure: GeoSureEntry


class Flooding(ctk.BaseConfig):
    """Configuration for flooding data.

    Attributes
    ----------
    rivers_sea : pathlib.Path
        Path to the rivers and sea flooding data.
    surface_water : pathlib.Path
        Path to the surface water flooding data.
    groundwater : pathlib.Path
        Path to the groundwater flooding data, if available.
    """

    rivers_sea: pathlib.Path
    surface_water: pathlib.Path
    groundwater: pathlib.Path | None = None

    # Empty yaml values are loaded as strings, so need to convert to None
    @pydantic.field_validator("groundwater", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str) -> str | None:
        if v == "":
            return None
        return v


class HazardsConfig(ctk.BaseConfig):
    """Configuration for hazard data.

    Attributes
    ----------
    coastal_erosion : ZipFileEntry
        Configuration for coastal erosion zip file entry.
    extreme_weather : ExtremeWeather
        Configuration for extreme weather data.
    flooding : Flooding
        Configuration for flooding data.
    ground_stability : GroundStability
        Configuration for ground stability data.
    """

    coastal_erosion: ZipFileEntry
    extreme_weather: ExtremeWeather
    flooding: Flooding
    ground_stability: GroundStability


class ModelRoadFlowsEntry(ctk.BaseConfig):
    """Configuration for the transport model road flows data.

    Attributes
    ----------
    link_flows: pathlib.Path
        Path to the link flows CSV file.
    ufs: pathlib.Path
        Path to the UFS CSV file.
    annualisation_factors: pathlib.Path
        Path to the annualisation factors CSV file.
    """

    link_flows: pathlib.Path
    ufs: pathlib.Path
    annualisation_factors: pathlib.Path


class DemandConfig(ctk.BaseConfig):
    """Configuration for demand data.

    Attributes
    ----------
    freight_demand : pathlib.Path
        Path to the freight demand data.
    noham_demand : pathlib.Path
        Path to the NoHAM demand data.
    noham_years: dict[str, int]
        Dictionary of years for NoHAM demand scenarios.
    model_road_flows : ModelRoadFlowsEntry
        Configuration for the transport model road flows data.
    nexus: dict[str, pathlib.Path]
        Dictionary of paths for nexus demand data.
    """

    freight_demand: pathlib.Path
    noham_demand: pathlib.Path
    noham_years: dict[str, int]
    model_road_flows: ModelRoadFlowsEntry
    nexus: dict[str, pathlib.Path]


class IncidentWeightConfig(ctk.BaseConfig):
    """Configuration for asset risk weighting incident weights.

    Attributes
    ----------
    incident_share : float
        Weight applied to incident share.
    minutes_share : float
        Weight applied to minutes share.
    cost_share : float
        Weight applied to cost share.
    """

    incident_share: float
    minutes_share: float
    cost_share: float


class AssetRiskWeightingConfig(ctk.BaseConfig):
    """Configuration for asset risk weighting workflow.

    Attributes
    ----------
    nwr_eastern_path : pathlib.Path
        Path to Network Rail Eastern weather data CSV.
    nwr_northwest_path : pathlib.Path
        Path to Network Rail North West weather data CSV.
    output_path : pathlib.Path
        Output directory for asset risk weighting results.
    incident_weights : IncidentWeightConfig
        Weights used to calculate rail hazard impacts.
    """

    nwr_eastern_path: pathlib.Path
    nwr_northwest_path: pathlib.Path
    output_path: pathlib.Path
    incident_weights: IncidentWeightConfig


class SwitchConfig(ctk.BaseConfig):
    """Configuration for model switches.

    Attributes
    ----------
    run_data_cleaning : bool
        Whether to run data cleaning.
    run_functional_rules : bool
        Whether to run functional rules.
    run_layering : bool
        Whether to run layering.
    all_roads : bool
        Whether to include all roads in the analysis.
    noham_roads : bool
        Whether to include NoHAM roads in the analysis.
    model_roads : bool
        Whether to include transport model roads in the analysis.
    passenger_rail : bool
        Whether to include passenger rail in the analysis.
    freight_rail : bool
        Whether to include freight rail in the analysis.
    airports : bool
        Whether to include airports in the analysis.
    bus_stops : bool
        Whether to include bus stops in the analysis.
    national_cycle_network : bool
        Whether to include the national cycle network in the analysis.
    train_stations : bool
        Whether to include train stations in the analysis.
    tram_stations : bool
        Whether to include tram stations in the analysis.
    rapid_transport_stations : bool
        Whether to include rapid transport stations in the analysis.
    ferry_terminals : bool
        Whether to include ferry terminals in the analysis.
    bus_coach_stations : bool
        Whether to include bus coach stations in the analysis.
    tram_network : bool
        Whether to include the tram network in the analysis.
    rapid_transport_network : bool
        Whether to include the rapid transport network in the analysis.
    bespoke: bool
        Whether to include bespoke infrastructure data in the analysis.
    extreme_weather : bool
        Whether to include extreme weather hazards in the analysis.
    flooding : bool
        Whether to include flooding hazards in the analysis.
    ground_stability : bool
        Whether to include ground stability hazards in the analysis.
    coastal_erosion : bool
        Whether to include coastal erosion hazards in the analysis.
    compute_flooding_overlay: bool
        Whether to compute the direct flooding overlay.
    flooding_index_raster: bool
        Whether to use the flooding index raster in the analysis.
    noham_zip_extract : bool
        Whether to extract NoHAM zip files.
    """

    run_data_cleaning: bool
    run_functional_rules: bool
    run_layering: bool

    all_roads: bool
    noham_roads: bool
    model_roads: bool
    passenger_rail: bool
    freight_rail: bool
    airports: bool
    bus_stops: bool
    national_cycle_network: bool
    train_stations: bool
    tram_stations: bool
    rapid_transport_stations: bool
    ferry_terminals: bool
    bus_coach_stations: bool
    tram_network: bool
    rapid_transport_network: bool

    bespoke: bool

    extreme_weather: bool
    flooding: bool
    ground_stability: bool
    coastal_erosion: bool

    compute_flooding_overlay: bool = False
    flooding_index_raster: bool = False
    noham_zip_extract: bool = False
    ground_stability_premium: bool = False

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        if not any([self.run_data_cleaning, self.run_functional_rules, self.run_layering]):
            raise ValueError(
                "At least one of 'run_data_cleaning', 'run_functional_rules' "
                "or 'run_layering' must be True."
            )

        if not any(
            [
                self.all_roads,
                self.noham_roads,
                self.model_roads,
                self.passenger_rail,
                self.freight_rail,
                self.airports,
                self.bus_stops,
                self.national_cycle_network,
                self.train_stations,
                self.tram_stations,
                self.rapid_transport_stations,
                self.ferry_terminals,
                self.bus_coach_stations,
                self.tram_network,
                self.rapid_transport_network,
                self.bespoke,
            ]
        ):
            raise ValueError("At least one infrastructure switch must be True.")

        if not any(
            [self.extreme_weather, self.flooding, self.ground_stability, self.coastal_erosion]
        ):
            raise ValueError("At least one hazard switch must be True.")

        return self


class ParameterConfig(ctk.BaseConfig):
    """Configuration for model parameters.

    Attributes
    ----------
    stb : str | None = None
        Name of Sub-National Transport Body to run the tool for (this must exactly match the
        name in the boundary data). Should be empty if running for a CA rather than STB.
    ca : str | None = None
        Name of Combined Authority to run the tool for (this must exactly match the name in the
        boundary data). Should be empty if running for a STB rather than CA.
    """

    stb: str | None = None
    ca: str | None = None

    # Empty yaml values are loaded as strings, so need to convert to None
    @pydantic.field_validator("stb", "ca", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str) -> str | None:
        if v == "":
            return None
        return v

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.stb is None) ^ (self.ca is None):
            raise ValueError("Exactly one of 'stb' or 'ca' must be provided, but not both.")
        return self


class ConstantConfig(ctk.BaseConfig):
    """Configuration for model constants.

    Attributes
    ----------
    noham_road_id_threshold : int
        Threshold for NoHAM road IDs.
    score_min : float
        Minimum score for risk calculations.
    score_max : float
        Maximum score for risk calculations.
    """

    noham_road_id_threshold: int
    score_min: int
    score_max: int


# -------------------------
# MAIN CONFIG
# -------------------------


class Config(ctk.BaseConfig):
    """Main configuration for whole model.

    Attributes
    ----------
    switches : SwitchConfig
        Configuration for model switches.
    parameters: ParameterConfig
        Configuration for model parameters.
    paths : PathConfig
        Configuration for base paths.
    infrastructure : InfrastructureConfig
        Configuration for infrastructure data.
    hazards : HazardsConfig
        Configuration for hazards data.
    demand : DemandConfig
        Configuration for demand data.
    asset_risk_weighting : AssetRiskWeightingConfig | None
        Configuration for asset risk weighting workflow.
    constants: ConstantConfig
        Configuration for model constants.
    """

    switches: SwitchConfig
    parameters: ParameterConfig
    paths: PathConfig
    other_input: OtherInput
    infrastructure: InfrastructureConfig
    hazards: HazardsConfig
    demand: DemandConfig
    asset_risk_weighting: AssetRiskWeightingConfig | None = None
    constants: ConstantConfig
