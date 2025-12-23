"""
Specify input and output folder paths
"""

from pathlib import Path

# Log Path
LOG_PATH = Path("D:/") / "Climate Vulnerability Tool" / "Logging" / "cvt.log"

# Base Paths
RAW_INPUT = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "raw inputs"
MODEL_INPUT = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model inputs"
MODEL_INTERIM_OUTPUT = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model interim outputs"
MODEL_OUTPUT = Path("D:/") / "Climate Vulnerability Tool" / "Data" / "model outputs"

# Infrastructure
INFRASTRUCTURE_RAW_IN = RAW_INPUT / "Infrastructure"
INFRASTRUCTURE_MODEL_IN = MODEL_INPUT / "Infrastructure"

ROAD_RAW_IN = INFRASTRUCTURE_RAW_IN / "Road"
RAIL_RAW_IN = INFRASTRUCTURE_RAW_IN / "Rail"
OTHER_RAW_IN = INFRASTRUCTURE_RAW_IN / "Other"

ROAD_MODEL_IN = INFRASTRUCTURE_MODEL_IN / "Road"
RAIL_MODEL_IN = INFRASTRUCTURE_MODEL_IN / "Rail"
OTHER_MODEL_IN = INFRASTRUCTURE_MODEL_IN / "Other"

# Hazards
HAZARD_RAW_IN = RAW_INPUT / "Hazard"
HAZARD_MODEL_IN = MODEL_INPUT / "Hazard"
HAZARD_INTERIM_OUT = MODEL_INTERIM_OUTPUT / "Hazard"

EXTREME_WEATHER_RAW_IN = HAZARD_RAW_IN / "Extreme Weather"
FLOODING_RAW_IN = HAZARD_RAW_IN / "Flooding"
GROUND_STABILITY_RAW_IN = HAZARD_RAW_IN / "Ground Stability"
COASTAL_EROSION_RAW_IN = HAZARD_RAW_IN / "Coastal Erosion"

EXTREME_WEATHER_MODEL_IN = HAZARD_MODEL_IN / "Extreme Weather"
FLOODING_MODEL_IN = HAZARD_MODEL_IN / "Flooding"
GROUND_STABILITY_MODEL_IN = HAZARD_MODEL_IN / "Ground Stability"
COASTAL_EROSION_MODEL_IN = HAZARD_MODEL_IN / "Coastal Erosion"

# Impact
IMPACT_RAW_IN = RAW_INPUT / "Impact"
IMPACT_MODEL_IN = MODEL_INPUT / "Impact"
IMPACT_INTERIM_OUT = MODEL_INTERIM_OUTPUT / "Impact"