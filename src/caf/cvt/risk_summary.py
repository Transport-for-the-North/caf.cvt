"""Generate risk summary for infrastructure data with risk."""

import logging
import pathlib

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows

from caf.cvt import model_config
from caf.cvt.definitions import (
    CellStyle,
    ColourMaps,
    HexColourCodes,
    MainHazardRiskCols,
    RiskCategories,
    RiskColumn,
    Scenarios,
)

# LOGGING

LOG = logging.getLogger(__name__)

# MODULE CONSTANTS

_RISK_CATEGORIES = {
    RiskCategories.VERY_LOW: (0, 20),
    RiskCategories.LOW: (20, 40),
    RiskCategories.MEDIUM: (40, 60),
    RiskCategories.HIGH: (60, 80),
    RiskCategories.VERY_HIGH: (80, 100),
}

_RISK_CATEGORY_COLOURS = {
    RiskCategories.VERY_LOW: HexColourCodes.DARK_BLUE,  # dark blue
    RiskCategories.LOW: HexColourCodes.TEAL,  # teal
    RiskCategories.MEDIUM: HexColourCodes.GOLD,  # gold
    RiskCategories.HIGH: HexColourCodes.DARK_ORANGE,  # dark orange
    RiskCategories.VERY_HIGH: HexColourCodes.DARK_RED,  # dark red
}

_LENGTH_M_COL = "Length (m)"
_RISK_CATEGORY_COL = "Risk Category"

_VERDANA_FONT = "Verdana"
_SOLID_FILL = "solid"
_PERCENT_NUMBER_FORMAT = "0%"
_BAR_CHART_SPACING = 0.2
_THIN_BORDER_STYLE = "thin"
_HEADER_ROW = 1
_COLUMN_WIDTH = 15
_LEFT_HORIZONTAL_ALIGNMENT = "left"

_RISK_DISTRIBUTION_WS_TITLE = "Risk Distribution"
_RISK_DISTRIBUTION_IMAGE_WIDTH = 400
_RISK_DISTRIBUTION_IMAGE_HEIGHT = 400
_RISK_DISTRIBUTION_IMAGE_COLUMN = 2
_RISK_DISTRIBUTION_IMAGE_ROW_SPACING = 3
_RISK_DISTRIBUTION_IMAGE_COLUMN_ADD = 4
_RISK_DISTRIBUTION_IMAGE_ROW_ADD = 21

_RISK_AVERAGES_WS_TITLE = "Risk Averages"
_RISK_AVERAGES_BAR_WIDTH = 0.40
_AVERAGE_CURRENT_RISK_COL = 2
_AVERAGE_FORECAST_RISK_COL = 3
_CHANGE_IN_RISK_COL = 4
_LENGTH_WEIGHTED_PCT_RISK_COL = 5
_RISK_AVERAGES_IMAGE_ROW_SPACING = 3
_RISK_AVERAGES_IMAGE_COLUMN = 2
_RISK_AVERAGES_IMAGE_WIDTH = 500
_RISK_AVERAGES_IMAGE_HEIGHT = 300

_DESCRIPTIVE_RISK_AVERAGES_WS_TITLE = "Descriptive Risk Averages"
_DESCRIPTIVE_RISK_AVERAGES_DESCRIPTION_COL = 1
_DESCRIPTIVE_RISK_AVERAGES_LENGTH_COL = 2
_ROW_LENGTH_ADD = 2


# GENERAL FUNCTIONS


def _style_cell(
    cell: openpyxl.cell.Cell,
    *,
    fill_colour: str,
    text_colour: str = HexColourCodes.BLACK,
    bold: bool = False,
    border: openpyxl.styles.borders.Border | None = None,
    alignment: str = _LEFT_HORIZONTAL_ALIGNMENT,
) -> None:
    """Apply styling to a given cell."""
    cell.fill = openpyxl.styles.PatternFill(
        fill_type=_SOLID_FILL,
        start_color=fill_colour,
        end_color=fill_colour,
    )
    cell.font = openpyxl.styles.Font(
        name=_VERDANA_FONT,
        color=text_colour,
        bold=bold,
    )
    cell.alignment = openpyxl.styles.Alignment(wrap_text=True, horizontal=alignment)
    if border:
        cell.border = border


def _is_column_odd(c_idx: int) -> bool:
    """Check if a column index is odd."""
    return c_idx % 2 != 0


def _is_row_even(r_idx: int) -> bool:
    """Check if a row index is even."""
    return r_idx % 2 == 0


def _is_hazard_score_column(c_idx: int) -> bool:
    return (c_idx - 3) % 3 in [0, 1]


def _is_hazard_change_column(c_idx: int) -> bool:
    return (c_idx != _DESCRIPTIVE_RISK_AVERAGES_LENGTH_COL) and ((c_idx - 3) % 3 not in [0, 1])


# GLOBAL FUNCTIONS


def create_risk_summary(
    config: model_config.Config,
    asset_risk: gpd.GeoDataFrame,
    risk_cols: list[RiskColumn],
    descriptive_cols: list[str],
    out_path: pathlib.Path,
    out_filename: str,
) -> None:
    """Output a summary spreadsheet for climate risk for a given asset."""
    # TODO (DJ): #31 Allow risk summary to work with point data as well as line data.
    # TODO (DJ): #31 Add handling of case where no scenario distinction exists
    # TODO (DJ): #31 Add summary of impact metrics when available
    # TODO (DJ): #31 Rename descriptive columns to more user-friendly names
    asset_risk_summary = asset_risk.copy()
    asset_risk_summary = asset_risk_summary.dropna(
        subset=[f"{col}_{Scenarios.CURRENT}" for col in risk_cols]
        + [f"{col}_{Scenarios.FORECAST}" for col in risk_cols]
    )
    asset_risk_summary[_LENGTH_M_COL] = asset_risk_summary.geometry.length
    total_length = asset_risk_summary[_LENGTH_M_COL].sum()

    risk_distribution = pd.DataFrame(
        [
            {
                "Risk Category": category,
                "Score": f"{lower}-{upper}",
            }
            for category, (lower, upper) in _RISK_CATEGORIES.items()
        ]
    )

    risk_averages = pd.DataFrame(
        [],
        columns=[
            "Hazard",
            f"Average {Scenarios.CURRENT.capitalize()} Risk",
            f"Average {Scenarios.FORECAST.capitalize()} Risk",
            "Change in Risk",
            "Percentage of Length with Increased Risk",
        ],
    )

    descriptive_risk_averages = {
        descriptive_col: pd.DataFrame(
            [
                {descriptive_col: descriptive_feature, _LENGTH_M_COL: 0}
                for descriptive_feature in asset_risk_summary[descriptive_col].unique()
                if descriptive_feature is not None
            ]
        )
        for descriptive_col in descriptive_cols
    }

    for main_hazard in MainHazardRiskCols:
        if main_hazard not in asset_risk_summary.columns.str.replace(
            f"_{Scenarios.CURRENT}", ""
        ):
            continue
        for sub_hazard in MainHazardRiskCols.get_sub_hazards(main_hazard):
            risk_distribution, risk_averages = _fill_risk_summary_column(
                config,
                asset_risk_summary,
                sub_hazard,
                risk_distribution,
                risk_averages,
                total_length,
            )

            descriptive_risk_averages = _fill_descriptive_risk_averages(
                asset_risk_summary,
                sub_hazard,
                descriptive_risk_averages,
                descriptive_cols,
            )

            _hazard_distribution_plot(
                risk_distribution, sub_hazard, out_path / "Risk Distribution"
            )

        risk_distribution, risk_averages = _fill_risk_summary_column(
            config,
            asset_risk_summary,
            main_hazard,
            risk_distribution,
            risk_averages,
            total_length,
        )

        descriptive_risk_averages = _fill_descriptive_risk_averages(
            asset_risk_summary,
            main_hazard,
            descriptive_risk_averages,
            descriptive_cols,
        )

        _hazard_distribution_plot(
            risk_distribution, main_hazard, out_path / "Risk Distribution"
        )

    _risk_averages_plot(risk_averages, out_path / "Risk Averages")

    _write_risk_summary(
        risk_distribution, risk_averages, descriptive_risk_averages, out_path, out_filename
    )


# LOCAL FUNCTIONS

## PRODUCE DATA


def _fill_risk_summary_column(
    config: model_config.Config,
    asset_risk: gpd.GeoDataFrame,
    hazard: MainHazardRiskCols | str,
    risk_distribution: pd.DataFrame,
    risk_averages: pd.DataFrame,
    total_length: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill the risk distribution column for a given main hazard and scenario."""
    length_weighted_avgs = {}
    change_in_risk = {}
    pct_increased_risks = {}

    for scenario in Scenarios:
        risk_column = f"{hazard}_{scenario}"
        out_risk_column = f"{scenario.capitalize()} {hazard.replace('_', ' ').title()}"
        risk_distribution[out_risk_column] = 0.0
        percentages = []
        categories = []
        for category, (lower, upper) in _RISK_CATEGORIES.items():
            if upper == config.constants.score_max:
                mask = (asset_risk[risk_column] >= lower) & (asset_risk[risk_column] <= upper)
            else:
                mask = (asset_risk[risk_column] >= lower) & (asset_risk[risk_column] < upper)
            risk_length = asset_risk.loc[mask, _LENGTH_M_COL].sum()
            pct_risk_in_category = (risk_length / total_length) * 100
            percentages.append(pct_risk_in_category)
            categories.append(category)

        floored_pcts = [int(pct) for pct in percentages]
        remainder = 100 - sum(floored_pcts)
        decimal_parts = [pct - int(pct) for pct in percentages]
        indices = np.argsort(decimal_parts)[::-1]
        for idx in indices[:remainder]:
            floored_pcts[idx] += 1
        for category, pct in zip(categories, floored_pcts, strict=True):
            risk_distribution.loc[
                risk_distribution["Risk Category"] == category, out_risk_column
            ] = pct

        length_weighted_avgs[scenario] = round(
            (asset_risk[risk_column] * asset_risk[_LENGTH_M_COL]).sum() / total_length, 1
        )

    increased_risk_length = asset_risk[
        (
            asset_risk[f"{hazard}_{Scenarios.FORECAST}"]
            > asset_risk[f"{hazard}_{Scenarios.CURRENT}"]
        )
    ][_LENGTH_M_COL].sum()
    pct_increased_risks[hazard] = round((increased_risk_length / total_length) * 100)

    change_in_risk[hazard] = round(
        (length_weighted_avgs[Scenarios.FORECAST] - length_weighted_avgs[Scenarios.CURRENT]), 1
    )

    risk_averages.loc[len(risk_averages)] = [
        hazard.replace("_", " ").title(),
        length_weighted_avgs[Scenarios.CURRENT],
        length_weighted_avgs[Scenarios.FORECAST],
        change_in_risk[hazard],
        pct_increased_risks[hazard],
    ]

    return risk_distribution, risk_averages


def _fill_descriptive_risk_averages(
    asset_risk: gpd.GeoDataFrame,
    hazard: MainHazardRiskCols | str,
    descriptive_risk_averages: dict[str, pd.DataFrame],
    descriptive_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Fill descriptive risk summary for a given hazard and asset risk data."""
    for descriptive_col in descriptive_cols:
        for descriptive_feature in asset_risk[descriptive_col].unique():
            asset_descriptive_risk = asset_risk[
                asset_risk[descriptive_col] == descriptive_feature
            ]
            if asset_descriptive_risk.empty:
                continue
            total_desc_length = asset_descriptive_risk[_LENGTH_M_COL].sum()
            descriptive_risk_averages[descriptive_col].loc[
                descriptive_risk_averages[descriptive_col][descriptive_col]
                == descriptive_feature,
                _LENGTH_M_COL,
            ] = round(total_desc_length)
            for scenario in Scenarios:
                risk_column = f"{hazard}_{scenario}"
                out_risk_column = f"{scenario.capitalize()} {hazard.replace('_', ' ').title()}"
                length_weighted_avg = round(
                    (
                        asset_descriptive_risk[risk_column]
                        * asset_descriptive_risk[_LENGTH_M_COL]
                    ).sum()
                    / total_desc_length
                )
                descriptive_risk_averages[descriptive_col].loc[
                    descriptive_risk_averages[descriptive_col][descriptive_col]
                    == descriptive_feature,
                    out_risk_column,
                ] = length_weighted_avg
        descriptive_risk_averages[descriptive_col][
            f"Change in {hazard.replace('_', ' ').title()}"
        ] = (
            descriptive_risk_averages[descriptive_col][
                f"{Scenarios.FORECAST.capitalize()} {hazard.replace('_', ' ').title()}"
            ]
            - descriptive_risk_averages[descriptive_col][
                f"{Scenarios.CURRENT.capitalize()} {hazard.replace('_', ' ').title()}"
            ]
        )

    return descriptive_risk_averages


## PLOTTING


def _hazard_distribution_plot(
    risk_distribution: pd.DataFrame,
    hazard: RiskColumn,
    out_path: pathlib.Path,
) -> None:
    """Generate distribution plots for the risk distribution summary."""
    out_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    x = [
        f"{Scenarios.CURRENT.capitalize()} {hazard.replace('_', ' ').title()}",
        f"{Scenarios.FORECAST.capitalize()} {hazard.replace('_', ' ').title()}",
    ]

    x_labels = [Scenarios.CURRENT.capitalize(), Scenarios.FORECAST.capitalize()]

    very_high = (
        risk_distribution[risk_distribution[_RISK_CATEGORY_COL] == RiskCategories.VERY_HIGH][x]
        .iloc[0]
        .to_numpy()
    )
    high = (
        risk_distribution[risk_distribution[_RISK_CATEGORY_COL] == RiskCategories.HIGH][x]
        .iloc[0]
        .to_numpy()
    )
    medium = (
        risk_distribution[risk_distribution[_RISK_CATEGORY_COL] == RiskCategories.MEDIUM][x]
        .iloc[0]
        .to_numpy()
    )
    low = (
        risk_distribution[risk_distribution[_RISK_CATEGORY_COL] == RiskCategories.LOW][x]
        .iloc[0]
        .to_numpy()
    )
    very_low = (
        risk_distribution[risk_distribution[_RISK_CATEGORY_COL] == RiskCategories.VERY_LOW][x]
        .iloc[0]
        .to_numpy()
    )

    cmap = hazard.get_cmap()
    cmap = plt.get_cmap(cmap)

    ax.bar(
        x_labels,
        very_low,
        color=cmap(_RISK_CATEGORIES[RiskCategories.VERY_LOW][1]),
        label=RiskCategories.VERY_LOW,
    )
    ax.bar(
        x_labels,
        low,
        bottom=very_low,
        color=cmap(_RISK_CATEGORIES[RiskCategories.LOW][1]),
        label=RiskCategories.LOW,
    )
    ax.bar(
        x_labels,
        medium,
        bottom=very_low + low,
        color=cmap(_RISK_CATEGORIES[RiskCategories.MEDIUM][1]),
        label=RiskCategories.MEDIUM,
    )
    ax.bar(
        x_labels,
        high,
        bottom=very_low + low + medium,
        color=cmap(_RISK_CATEGORIES[RiskCategories.HIGH][1]),
        label=RiskCategories.HIGH,
    )
    ax.bar(
        x_labels,
        very_high,
        bottom=very_low + low + medium + high,
        color=cmap(_RISK_CATEGORIES[RiskCategories.VERY_HIGH][1]),
        label=RiskCategories.VERY_HIGH,
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Percent at Risk")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), reverse=True)
    ax.grid(axis="y", alpha=0.5)
    ax.set_title(
        f"Risk Distribution for {hazard.replace('_', ' ').replace('risk', '').title()}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path / f"risk_distribution_{hazard}.png")
    plt.close(fig)


def _risk_averages_plot(risk_averages: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Generate a plot for risk averages."""
    out_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    current = risk_averages[f"Average {Scenarios.CURRENT.capitalize()} Risk"].to_numpy()
    forecast = risk_averages[f"Average {Scenarios.FORECAST.capitalize()} Risk"].to_numpy()
    hazards = [hazard.replace(" Risk", "") for hazard in risk_averages["Hazard"].to_list()]
    x = np.arange(len(hazards))

    ax.bar(
        x - _BAR_CHART_SPACING,
        current,
        width=_RISK_AVERAGES_BAR_WIDTH,
        color=ColourMaps.ROYAL_BLUE,
        label=Scenarios.CURRENT.capitalize(),
    )
    ax.bar(
        x + _BAR_CHART_SPACING,
        forecast,
        width=_RISK_AVERAGES_BAR_WIDTH,
        color=ColourMaps.DARK_ORANGE,
        label=Scenarios.FORECAST.capitalize(),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(hazards)
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Hazard")
    ax.set_ylabel("Length-Weighted Average Risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "risk_averages.png")
    plt.close(fig)


## WRITING FUNCTIONS


def _write_risk_summary(
    risk_distribution: pd.DataFrame,
    risk_averages: pd.DataFrame,
    descriptive_risk_averages: pd.DataFrame,
    out_path: pathlib.Path,
    out_filename: str,
) -> None:
    """Write a risk summary to an Excel file."""
    wb = openpyxl.Workbook()

    thin_border = openpyxl.styles.Side(
        border_style=_THIN_BORDER_STYLE, color=HexColourCodes.WHITE
    )

    wb = _write_risk_distribution(
        workbook=wb, risk_distribution=risk_distribution, border=thin_border, out_path=out_path
    )

    wb = _write_risk_averages(
        workbook=wb, risk_averages=risk_averages, border=thin_border, out_path=out_path
    )

    wb = _write_descriptive_risk_averages(
        workbook=wb,
        border=thin_border,
        descriptive_risk_averages=descriptive_risk_averages,
    )

    wb.save(out_path / out_filename)
    wb.close()


### RISK DISTRIBUTION SHEET


def _write_risk_distribution(
    workbook: openpyxl.Workbook,
    risk_distribution: pd.DataFrame,
    border: openpyxl.styles.Side,
    out_path: pathlib.Path,
) -> openpyxl.Workbook:
    """Write risk distribution data and plots to an Excel workbook sheet."""
    ws = workbook.active
    ws.title = _RISK_DISTRIBUTION_WS_TITLE

    for r_idx, row in enumerate(
        dataframe_to_rows(risk_distribution, index=False, header=True), 1
    ):
        for c_idx, value in enumerate(row, 1):
            column_letter = openpyxl.utils.get_column_letter(c_idx)
            ws.column_dimensions[column_letter].width = _COLUMN_WIDTH
            horizontal_alignment = _LEFT_HORIZONTAL_ALIGNMENT  # Default alignment

            cell = ws.cell(
                row=r_idx,
                column=c_idx,
                value=value / 100 if isinstance(value, (int, float)) and value != 0 else value,
            )

            style = _get_risk_distribution_style(r_idx=r_idx, c_idx=c_idx, border=border)

            # Format cell
            _style_cell(
                cell,
                fill_colour=style.fill_colour,
                text_colour=style.text_colour,
                bold=style.bold,
                border=style.border,
                alignment=horizontal_alignment,
            )

            if isinstance(value, (int, float)) and value != 0:
                _apply_risk_distribution_data_bar(ws=ws, cell=cell, category=row[0])

    _add_risk_distribution_images(
        ws=ws, risk_distribution=risk_distribution, out_path=out_path
    )

    return workbook


def _add_risk_distribution_images(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    risk_distribution: pd.DataFrame,
    out_path: pathlib.Path,
) -> None:
    hazards = list(
        dict.fromkeys(
            [
                col.replace(f"{Scenarios.CURRENT.capitalize()} ", "").replace(
                    f"{Scenarios.FORECAST.capitalize()} ", ""
                )
                for col in risk_distribution.columns
                if " Risk" in col
            ]
        )
    )
    image_row = len(risk_distribution) + _RISK_DISTRIBUTION_IMAGE_ROW_SPACING
    image_column = _RISK_DISTRIBUTION_IMAGE_COLUMN
    for hazard in hazards:
        risk_distribution_image = openpyxl.drawing.image.Image(
            out_path
            / "Risk Distribution"
            / f"risk_distribution_{hazard.lower().replace(' ', '_')}.png"
        )
        risk_distribution_image.width = _RISK_DISTRIBUTION_IMAGE_WIDTH
        risk_distribution_image.height = _RISK_DISTRIBUTION_IMAGE_HEIGHT

        ws.add_image(
            risk_distribution_image,
            f"{openpyxl.utils.get_column_letter(image_column)}{image_row}",
        )
        image_column += _RISK_DISTRIBUTION_IMAGE_COLUMN_ADD
        if image_column > len(risk_distribution.columns):
            image_column = _RISK_DISTRIBUTION_IMAGE_COLUMN
            image_row += _RISK_DISTRIBUTION_IMAGE_ROW_ADD


def _get_risk_distribution_style(
    r_idx: int,
    c_idx: int,
    border: openpyxl.styles.Side,
) -> CellStyle:
    """Get the style for a cell in the risk distribution sheet."""
    # Determine formatting dynamically
    if r_idx == _HEADER_ROW:  # Apply fill only to header row:
        if _is_column_odd(c_idx):
            cell_border = openpyxl.styles.Border(left=border, top=border, bottom=border)
        else:
            cell_border = openpyxl.styles.Border(right=border, top=border, bottom=border)

        return CellStyle(
            fill_colour=HexColourCodes.BLACK,
            text_colour=HexColourCodes.WHITE,
            bold=True,
            border=cell_border,
        )

    if _is_column_odd(c_idx):
        cell_border = openpyxl.styles.Border(left=border)
    else:
        cell_border = openpyxl.styles.Border(right=border)

    return CellStyle(
        fill_colour=HexColourCodes.GREY,
        text_colour=HexColourCodes.WHITE,
        bold=False,
        border=cell_border,
    )


def _apply_risk_distribution_data_bar(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    cell: openpyxl.cell.Cell,
    category: RiskCategories,
) -> None:
    """Apply data bar formatting to a risk distribution cell."""
    cell.number_format = _PERCENT_NUMBER_FORMAT
    data_bar = openpyxl.formatting.rule.DataBarRule(
        start_type="num",
        start_value=0,
        end_type="num",
        end_value=1,
        color=_RISK_CATEGORY_COLOURS[category],
        showValue=True,
        minLength=None,
        maxLength=None,
    )
    ws.conditional_formatting.add(cell.coordinate, data_bar)


### RISK AVERAGES SHEET


def _write_risk_averages(
    workbook: openpyxl.Workbook,
    risk_averages: pd.DataFrame,
    border: openpyxl.styles.Side,
    out_path: pathlib.Path,
) -> openpyxl.Workbook:
    """Write risk averages data and plots to an Excel workbook sheet."""
    ws = workbook.create_sheet(title=_RISK_AVERAGES_WS_TITLE)

    # Write risk averages data to the sheet
    for r_idx, row in enumerate(dataframe_to_rows(risk_averages, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)

            column_letter = openpyxl.utils.get_column_letter(c_idx)
            ws.column_dimensions[column_letter].width = _COLUMN_WIDTH
            horizontal_alignment = _LEFT_HORIZONTAL_ALIGNMENT

            style = _get_risk_average_style(
                r_idx=r_idx,
                c_idx=c_idx,
                value=value,
                border=border,
            )

            # Format cell
            _style_cell(
                cell=cell,
                fill_colour=style.fill_colour,
                text_colour=style.text_colour,
                bold=style.bold,
                alignment=horizontal_alignment,
                border=style.border,
            )

    _add_risk_averages_image(ws=ws, risk_averages=risk_averages, out_path=out_path)

    return workbook


def _add_risk_averages_image(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    risk_averages: pd.DataFrame,
    out_path: pathlib.Path,
) -> None:
    """Add the risk averages image to the worksheet."""
    image_row = len(risk_averages) + _RISK_AVERAGES_IMAGE_ROW_SPACING
    image_column = _RISK_AVERAGES_IMAGE_COLUMN

    risk_averages_image = openpyxl.drawing.image.Image(
        out_path / "Risk Averages" / "risk_averages.png"
    )
    risk_averages_image.width = _RISK_AVERAGES_IMAGE_WIDTH
    risk_averages_image.height = _RISK_AVERAGES_IMAGE_HEIGHT

    ws.add_image(
        risk_averages_image,
        f"{openpyxl.utils.get_column_letter(image_column)}{image_row}",
    )


def _get_risk_average_style(
    r_idx: int,
    c_idx: int,
    value: str | int | float | None,
    border: openpyxl.styles.Side,
) -> CellStyle:
    """Determine the style for a risk average cell."""
    # Default settings
    text_colour = HexColourCodes.BLACK
    bold = False
    cell_border = None
    # Determine formatting dynamically
    if r_idx == _HEADER_ROW:  # Apply fill only to header row:
        fill_colour = HexColourCodes.BLACK
        bold = True
        text_colour = HexColourCodes.WHITE
    elif c_idx == 1:
        if _is_row_even(r_idx):
            fill_colour = HexColourCodes.DARK_GREY
        else:
            fill_colour = HexColourCodes.GREY
        bold = False
        text_colour = HexColourCodes.BLACK

    if isinstance(value, (int, float)):
        if c_idx in [_AVERAGE_CURRENT_RISK_COL, _AVERAGE_FORECAST_RISK_COL]:
            fill_colour = mpl.colors.to_hex(
                mpl.colormaps[ColourMaps.GREEN_YELLOW_RED](value / 100)
            ).replace("#", "")
        elif c_idx == _CHANGE_IN_RISK_COL:
            if value > 0:
                fill_colour = HexColourCodes.LIGHT_GREEN
            else:
                fill_colour = HexColourCodes.PALE_VIOLET_RED
        elif c_idx == _LENGTH_WEIGHTED_PCT_RISK_COL:
            fill_colour = mpl.colors.to_hex(
                mpl.colormaps[ColourMaps.ORANGES](value / 100)
            ).replace("#", "")
    elif r_idx == _HEADER_ROW:
        cell_border = openpyxl.styles.Border(right=border, top=border, bottom=border)
    else:
        cell_border = openpyxl.styles.Border(right=border)

    return CellStyle(
        fill_colour=fill_colour,
        text_colour=text_colour,
        bold=bold,
        border=cell_border,
    )


### DESCRIPTIVE RISK AVERAGES SHEET


def _write_descriptive_risk_averages(
    workbook: openpyxl.Workbook,
    descriptive_risk_averages: dict[str, pd.DataFrame],
    border: openpyxl.styles.Side,
) -> openpyxl.Workbook:
    """Write descriptive risk averages data to the Excel workbook sheet."""
    ws = workbook.create_sheet(title=_DESCRIPTIVE_RISK_AVERAGES_WS_TITLE)
    total_length = 0

    for _, risk_averages in descriptive_risk_averages.items():
        for r_idx, row in enumerate(
            dataframe_to_rows(risk_averages, index=False, header=True), 1
        ):
            for c_idx, value in enumerate(row, 1):
                column_letter = openpyxl.utils.get_column_letter(c_idx)
                ws.column_dimensions[column_letter].width = _COLUMN_WIDTH
                horizontal_alignment = _LEFT_HORIZONTAL_ALIGNMENT  # Default alignment
                current_row = total_length + r_idx

                cell_value = value

                if _is_hazard_change_column(c_idx) and r_idx != _HEADER_ROW:
                    cell_value = _format_hazard_change(int(value))

                cell = ws.cell(row=current_row, column=c_idx, value=cell_value)

                style = _get_descriptive_risk_average_style(
                    r_idx=r_idx,
                    c_idx=c_idx,
                    value=value,
                    border=border,
                )

                # Format cell
                _style_cell(
                    cell=cell,
                    fill_colour=style.fill_colour,
                    text_colour=style.text_colour,
                    bold=style.bold,
                    border=style.border,
                    alignment=horizontal_alignment,
                )

        total_length += len(risk_averages) + _ROW_LENGTH_ADD

    return workbook


def _get_descriptive_risk_average_style(
    r_idx: int,
    c_idx: int,
    value: int | float | str | None,
    border: openpyxl.styles.Side,
) -> CellStyle:
    """Determine the style for a descriptive risk average cell."""
    if r_idx == _HEADER_ROW:  # Apply fill only to header row:
        cell_border = openpyxl.styles.Border(top=border, bottom=border)
        if c_idx == _DESCRIPTIVE_RISK_AVERAGES_DESCRIPTION_COL:
            cell_border = openpyxl.styles.Border(left=border, top=border, bottom=border)
        elif c_idx == _DESCRIPTIVE_RISK_AVERAGES_LENGTH_COL or _is_hazard_change_column(c_idx):
            cell_border = openpyxl.styles.Border(right=border, top=border, bottom=border)

        return CellStyle(
            fill_colour=HexColourCodes.TEAL,
            text_colour=HexColourCodes.WHITE,
            bold=True,
            border=cell_border,
        )

    if c_idx in [
        _DESCRIPTIVE_RISK_AVERAGES_DESCRIPTION_COL,
        _DESCRIPTIVE_RISK_AVERAGES_LENGTH_COL,
    ]:
        fill_colour = HexColourCodes.DARK_GREY if _is_row_even(r_idx) else HexColourCodes.GREY

        cell_border = None
        if c_idx == _DESCRIPTIVE_RISK_AVERAGES_LENGTH_COL:
            cell_border = openpyxl.styles.Border(right=border)

        return CellStyle(
            fill_colour=fill_colour,
            text_colour=HexColourCodes.BLACK,
            bold=False,
            border=cell_border,
        )

    if _is_hazard_score_column(c_idx):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric value for hazard score, got {value}")
        return CellStyle(
            fill_colour=mpl.colors.to_hex(
                mpl.colormaps[ColourMaps.GREEN_YELLOW_RED](value / 100)
            ).replace("#", ""),
            text_colour=HexColourCodes.BLACK,
            bold=False,
            border=None,
        )

    if _is_hazard_change_column(c_idx):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric value for hazard change, got {value}")
        return CellStyle(
            fill_colour=HexColourCodes.WHITE,
            text_colour=_hazard_change_text_colour(float(value)),
            bold=False,
            border=openpyxl.styles.Border(right=border),
        )

    return CellStyle(
        fill_colour=HexColourCodes.WHITE,
        text_colour=HexColourCodes.BLACK,
        bold=False,
        border=None,
    )


def _format_hazard_change(value: int) -> str:
    """Format hazard change value."""
    if value > 0:
        return f"↑ {value}"

    if value < 0:
        return f"↓ {value}"

    return f"→ {value}"


def _hazard_change_text_colour(value: float) -> str:
    """Get text colour for hazard change."""
    if value > 0:
        return HexColourCodes.RED

    if value < 0:
        return HexColourCodes.GREEN

    return HexColourCodes.YELLOW
