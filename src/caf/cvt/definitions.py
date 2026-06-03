"""Definitions and constants for caf.cvt model."""

from __future__ import annotations

import enum


class Columns(enum.StrEnum):
    """Column definition base class."""

    @classmethod
    def __contains__(cls, item: object) -> bool:
        """Return True if object is defined within the enum, False otherwise."""
        try:
            cls(str(item))
        except ValueError:
            return False
        return True


class MainHazardCols(Columns):
    """Column definitions for main hazard layers."""

    EXTREME_WEATHER = "extreme_weather"
    FLOODING = "flood"
    GROUND_STABILITY = "ground_stability"
    COASTAL_EROSION = "coastal_erosion"

    @classmethod
    def all(cls) -> list[MainHazardCols]:
        """Return a list of all main hazard columns."""
        return [col for col in cls]

    @classmethod
    def all_risk_cols(cls) -> list[str]:
        """Return a list of all main hazard risk columns."""
        return [f"{col}_risk" for col in cls]

    @classmethod
    def add_risk_suffix(cls, col: str) -> str:
        """Add _risk suffix to column if not already present."""
        return f"{col}_risk"


class ExtremeWeatherCols(Columns):
    """Column definitions for extreme weather subhazard layers."""

    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    DROUGHT = "drought"
    STORM = "storm"


class Scenarios(Columns):
    """Column definitions for scenarios."""

    CURRENT = "current"
    FORECAST = "forecast"

    @classmethod
    def all(cls) -> list[Scenarios]:
        """Return a list of all scenarios."""
        return [col for col in cls]

    @classmethod
    def scenario_or_column(cls) -> str:
        """Return the name of the scenario column."""
        return f"{cls.CURRENT}_or_{cls.FORECAST}"


class NoHAMYears:
    """Mapping between NoHAM years and scenario definitions."""

    YEAR_TO_SCENARIO: dict[int, Scenarios] = {
        2023: Scenarios.CURRENT,
        2048: Scenarios.FORECAST,
    }

    @classmethod
    def get_scenario(cls, year: int) -> str:
        """Return the scenario corresponding to a given year."""
        return cls.YEAR_TO_SCENARIO[int(year)]


class NoHAM:
    """Definitions for NoHAM data."""

    USER_CLASS_1 = "uc1"
    USER_CLASS_2 = "uc2"
    USER_CLASS_3 = "uc3"
    USER_CLASS_4 = "uc4"
    USER_CLASS_5 = "uc5"

    TIME_PERIOD_1 = "TS1"
    TIME_PERIOD_2 = "TS2"
    TIME_PERIOD_3 = "TS3"

    @classmethod
    def all_user_classes(cls) -> list[str]:
        """Return a list of all NoHAM user classes."""
        return [
            cls.USER_CLASS_1,
            cls.USER_CLASS_2,
            cls.USER_CLASS_3,
            cls.USER_CLASS_4,
            cls.USER_CLASS_5,
        ]

    @classmethod
    def all_time_periods(cls) -> list[str]:
        """Return a list of all NoHAM time periods."""
        return [cls.TIME_PERIOD_1, cls.TIME_PERIOD_2, cls.TIME_PERIOD_3]
