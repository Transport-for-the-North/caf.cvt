"""Intersect infrastructure with hazard layers to assign risk scores to infrastructure."""

import logging
import pathlib

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl import cell
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows

from caf.cvt import data_cleaning, file_paths, functional_rules, model_config
from caf.cvt.definitions import (
    AssetTypes,
    CoastalErosionRiskCols,
    ExtremeWeatherRiskCols,
    FloodingRiskCols,
    GroundStabilityRiskCols,
    ImpactCols,
    MainHazardRiskCols,
    OSRailCols,
    OSRailStructure,
    OSRoadCols,
    OSRoadStructure,
    RiskColumn,
    Scenarios,
    UserClasses,
    VulnerabilityModifier,
)

LOG = logging.getLogger(__name__)

_DEMAND_WEIGHT = 0.5

_TRAIN_STATIONS_BUFFER_SIZE_M = 100
_CHARGING_SITES_BUFFER_SIZE_M = 25
_BUS_COACH_STATIONS_BUFFER_SIZE_M = 50
_TRAM_STATIONS_BUFFER_SIZE_M = 25
_RAPID_TRANSPORT_STATIONS_BUFFER_SIZE_M = 50
_FERRY_TERMINALS_BUFFER_SIZE_M = 50
_PETROL_STATIONS_BUFFER_SIZE_M = 50

_RISK_CATEGORIES = {
    "Very Low": (0, 20),
    "Low": (20, 40),
    "Medium": (40, 60),
    "High": (60, 80),
    "Very High": (80, 100),
}

_RISK_CATEGORY_COLOURS = {
    "Very Low": "00008b",  # dark blue
    "Low": "008080",  # teal
    "Medium": "ffd700",  # gold
    "High": "ff8c00",  # dark orange
    "Very High": "8b0000",  # dark red
}


# GENERAL FUNCTIONS


def _infrastructure_risk_intersect(
    infrastructure_data: gpd.GeoDataFrame, hazards_dict: dict[str, gpd.GeoDataFrame]
) -> gpd.GeoDataFrame:
    """Intersect infrastructure with hazard risk layers.

    Spatially combine infrastructure with hazard layers using an intersection spatial join,
    then calculate hazard risk score as the max risk value of the intersection. Return merged
    GeoDataFrame with hazard risk columns added.
    """
    infrastructure_with_risk = infrastructure_data.copy()

    for _hazard_name, hazard_data in hazards_dict.items():
        # Spatial join to find intersections with hazards
        hazard_gdf_match = hazard_data.to_crs(infrastructure_with_risk.crs)  # Match CRS
        intersections = gpd.sjoin(
            infrastructure_with_risk, hazard_gdf_match, how="left", predicate="intersects"
        )

        # Identify risk columns
        risk_columns = hazard_gdf_match.columns[
            hazard_gdf_match.columns.str.contains("risk", case=False)
        ]

        # Calculate hazard risk score per infrastructure segment as max value of intersection
        agg = intersections.groupby(intersections.index)[risk_columns].max()

        # Merge back into main DataFrame
        infrastructure_with_risk = infrastructure_with_risk.join(agg, how="left")

    return infrastructure_with_risk.fillna(0)


def _duplicate_non_scenario_hazards(risk_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Duplicate non-scenario hazard columns for both scenarios."""
    non_scenario_hazards = risk_data.columns[
        risk_data.columns.str.contains("risk", case=False)
        & ~risk_data.columns.str.endswith((f"_{Scenarios.CURRENT}", f"_{Scenarios.FORECAST}"))
    ].tolist()
    for hazard in non_scenario_hazards:
        LOG.warning("Duplicating non-scenario hazard '%s' for both scenarios.", hazard)
        risk_data[f"{hazard}_{Scenarios.CURRENT}"] = risk_data[hazard]
        risk_data[f"{hazard}_{Scenarios.FORECAST}"] = risk_data[hazard]
    return risk_data


def _reshape_for_scenarios(
    risk_data: gpd.GeoDataFrame, id_col: str, risk_cols_order: list[RiskColumn]
) -> gpd.GeoDataFrame:
    """Reshape dataframe by adding a current/forecast column to distinguish identical rows."""
    # Identify risk and descriptive columns
    risk_cols = [
        col
        for col in risk_data.columns
        if col.endswith((f"_{Scenarios.CURRENT}", f"_{Scenarios.FORECAST}"))
    ]
    descriptive_cols = [
        col
        for col in risk_data.columns
        if col not in risk_cols and col not in (id_col, "geometry")
    ]

    # Separate geometry for later
    geometry = risk_data[[id_col, "geometry"]].copy()

    # Melt only risk columns
    melted = risk_data.melt(
        id_vars=[id_col, *descriptive_cols],
        value_vars=risk_cols,
        var_name="variable",
        value_name="value",
    )

    # Extract scenario and clean variable names
    scenario_pattern = rf"_({'|'.join(Scenarios)})$"
    scenario_col = Scenarios.scenario_or_column()
    melted[scenario_col] = (
        melted["variable"]
        .str.extract(scenario_pattern)[0]
        .map({s: s.title() for s in Scenarios})
    )
    melted["variable"] = melted["variable"].str.replace(scenario_pattern, "", regex=True)

    # Pivot back so each risk variable becomes a column
    reshaped = melted.pivot_table(
        index=[id_col, scenario_col, *descriptive_cols],
        columns="variable",
        values="value",
    ).reset_index()

    # Reorder risk columns based on original order
    reshaped = reshaped[[id_col, scenario_col, *descriptive_cols, *risk_cols_order]]

    # Merge geometry back
    reshaped_gdf = reshaped.merge(geometry, on=id_col)
    return gpd.GeoDataFrame(reshaped_gdf, geometry="geometry", crs=risk_data.crs)


def _prepare_model_output(
    risk_data: gpd.GeoDataFrame,
    drop_cols: list[str],
    rename_map: dict[str, str],
    risk_cols_order: list[RiskColumn],
) -> gpd.GeoDataFrame:
    """Perform standard cleaning operations on risk data to prepare for model output."""
    risk_data = risk_data.drop(columns=drop_cols)
    risk_data = risk_data.drop_duplicates(subset=["geometry"])
    risk_data = risk_data.rename(columns=rename_map)
    risk_data = risk_data.to_crs(data_cleaning.BNG_CRS)
    risk_data = _duplicate_non_scenario_hazards(risk_data)
    risk_data = _reshape_for_scenarios(risk_data, "id", risk_cols_order)
    risk_data[risk_cols_order] = risk_data[risk_cols_order].round(1)
    return risk_data.rename(columns={col: f"{col}_score" for col in risk_cols_order})


def _split_csv_shapefile(
    config: model_config.Config,
    gdf: gpd.GeoDataFrame,
    id_col: str,
    out_path_no_suffix: pathlib.Path,
) -> None:
    """Split GeoDataFrame into a CSV and Shapefile, then write to file.

    Separates GeoDataFrame into a dataframe with an ID and attributes, and a Shapefile with an
    ID and geometry, then writes them to file.
    """
    # Separate spatial and attribute data
    spatial_gdf = gdf[[id_col, "geometry"]].copy()
    attribute_df = gdf.drop(columns=["geometry"])

    # Remove duplicates from spatial data
    spatial_gdf = spatial_gdf.drop_duplicates()

    # Save to file
    data_cleaning.write_to_file(
        spatial_gdf, config.paths.model_output / out_path_no_suffix.with_suffix(".shp")
    )
    data_cleaning.write_to_file(
        attribute_df, config.paths.model_output / out_path_no_suffix.with_suffix(".csv")
    )


def _audit_infrastructure_risk(
    infrastructure_risk: gpd.GeoDataFrame,
    infrastructure_name: str,
    cols: list[RiskColumn],
    audit_path: pathlib.Path,
    *,
    feature_range: tuple[int, int],
    linewidth: float = 0.5,
) -> None:
    """Plot choropleth maps for infrastructure risk."""
    audit_path.mkdir(parents=True, exist_ok=True)

    for risk_col in cols:
        functional_rules.plot_choropleth(
            risk_data=infrastructure_risk,
            column=risk_col,
            title=f"{infrastructure_name} {risk_col.replace('_', ' ').title()}",
            out_path=audit_path / f"{risk_col}_choropleth.png",
            linewidth=linewidth,
            edgecolor=None,
            feature_range=feature_range,
            basemap_source=(
                "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png?key="
                + functional_rules.CARTO_API_KEY
            ),
        )


def _get_impact_weights(hazards: list[str]) -> dict[str, float]:
    """Generate impact weights."""
    impact_weights = {"demand": _DEMAND_WEIGHT}

    # Divide remaining weight equally amongst hazards
    hazard_weight = (1 - _DEMAND_WEIGHT) / len(hazards)
    for hazard in hazards:
        impact_weights[hazard] = hazard_weight
    return impact_weights


def _get_main_hazard_modifiers(
    vulnerabilities: dict[RiskColumn, VulnerabilityModifier],
) -> dict[MainHazardRiskCols, float]:
    """Get main hazard modifiers from sub-hazard modifiers."""
    modifiers = {}
    for main_hazard in MainHazardRiskCols:
        weights = main_hazard.get_weights()
        relevant_weights = {
            hazard: weight for hazard, weight in weights.items() if hazard in vulnerabilities
        }

        if not relevant_weights:
            continue

        modifier = (
            sum(  # Calculate weighted contribution of hazards with explicit modifier
                vulnerabilities[hazard] * weight for hazard, weight in relevant_weights.items()
            )
            +  # Calculate contribution of unspecified hazards
            (1 - sum(relevant_weights.values()))
        )

        modifiers[main_hazard] = modifier

    return modifiers


def _apply_asset_vulnerability(
    risk_data: gpd.GeoDataFrame,
    structure_enum: OSRoadStructure | OSRailStructure,
    structure_col: str,
    feature_range: tuple[int, int],
) -> gpd.GeoDataFrame:
    """Apply asset vulnerability modifiers to risk data."""
    for structure in structure_enum:
        mask = risk_data[structure_col] == structure
        vulnerabilities = structure.get_vulnerability()
        for hazard, modifier in structure.get_vulnerability().items():
            risk_cols = [col for col in risk_data.columns if hazard in col]
            for risk_col in risk_cols:
                risk_data.loc[mask, risk_col] = (
                    risk_data.loc[mask, risk_col] * modifier
                ).clip(lower=feature_range[0], upper=feature_range[1])

        main_hazard_modifiers = _get_main_hazard_modifiers(vulnerabilities)
        for main_hazard, modifier in main_hazard_modifiers.items():
            risk_cols = [col for col in risk_data.columns if main_hazard in col]
            for risk_col in risk_cols:
                risk_data.loc[mask, risk_col] = (
                    risk_data.loc[mask, risk_col] * modifier
                ).clip(lower=feature_range[0], upper=feature_range[1])
    return risk_data


def _apply_asset_hazard_weighting(
    asset_risk: gpd.GeoDataFrame,
    asset_type: AssetTypes,
    hazards: dict[MainHazardRiskCols, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Recalculate main hazard risk scores based on sub-hazard risk scores and weights."""
    for main_hazard in hazards:
        weights = asset_type.get_asset_hazard_weights(main_hazard)
        if main_hazard == MainHazardRiskCols.GROUND_STABILITY:
            asset_risk = functional_rules._calculate_composite_score(
                asset_risk,
                weights,
                main_hazard,
            )
        else:
            asset_risk = functional_rules._calculate_composite_score_scenarios(
                asset_risk,
                weights,
                main_hazard,
            )
    return asset_risk


def _create_risk_summary(
    asset_risk: gpd.GeoDataFrame,
    descriptive_cols: list[str],
    out_path: pathlib.Path,
) -> None:
    """Output a summary spreadsheet for climate risk for a given asset."""
    # TODO (DJ): Allow risk summary to work with point data as well as line data.
    # TODO (DJ): Add handling of case where no scenario distinction exists
    asset_risk["Length (m)"] = asset_risk.geometry.length
    total_length = asset_risk["Length (m)"].sum()

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
                {descriptive_col: descriptive_feature, "Length (m)": 0}
                for descriptive_feature in asset_risk[descriptive_col].unique()
            ]
        )
        for descriptive_col in descriptive_cols
    }

    for main_hazard in MainHazardRiskCols:
        if main_hazard not in asset_risk.columns.str.replace(f"_{Scenarios.CURRENT}", ""):
            continue
        for sub_hazard in MainHazardRiskCols.get_sub_hazards(main_hazard):
            risk_distribution, risk_averages = _fill_risk_summary_column(
                asset_risk,
                sub_hazard,
                risk_distribution,
                risk_averages,
                total_length,
            )

            descriptive_risk_averages = _fill_descriptive_risk_averages(
                asset_risk,
                sub_hazard,
                descriptive_risk_averages,
                descriptive_cols,
            )

            _hazard_distribution_plot(
                risk_distribution, sub_hazard, out_path / "Risk Distribution"
            )

        risk_distribution, risk_averages = _fill_risk_summary_column(
            asset_risk,
            main_hazard,
            risk_distribution,
            risk_averages,
            total_length,
        )

        descriptive_risk_averages = _fill_descriptive_risk_averages(
            asset_risk,
            main_hazard,
            descriptive_risk_averages,
            descriptive_cols,
        )

        _hazard_distribution_plot(
            risk_distribution, main_hazard, out_path / "Risk Distribution"
        )

    _risk_averages_plot(risk_averages, out_path / "Risk Averages")

    _write_risk_summary(risk_distribution, risk_averages, descriptive_risk_averages, out_path)

    return risk_distribution, risk_averages, descriptive_risk_averages


def _fill_risk_summary_column(
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
        for category, (lower, upper) in _RISK_CATEGORIES.items():
            if upper == 100:
                mask = (asset_risk[risk_column] >= lower) & (asset_risk[risk_column] <= upper)
            else:
                mask = (asset_risk[risk_column] >= lower) & (asset_risk[risk_column] < upper)
            risk_length = asset_risk.loc[mask, "Length (m)"].sum()
            pct_risk_in_category = (risk_length / total_length) * 100
            risk_distribution.loc[
                risk_distribution["Risk Category"] == category, out_risk_column
            ] = round(pct_risk_in_category, 1)

        length_weighted_avgs[scenario] = round(
            (asset_risk[risk_column] * asset_risk["Length (m)"]).sum() / total_length, 1
        )

    increased_risk_length = asset_risk[
        (
            asset_risk[f"{hazard}_{Scenarios.FORECAST}"]
            > asset_risk[f"{hazard}_{Scenarios.CURRENT}"]
        )
    ]["Length (m)"].sum()
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
            total_desc_length = asset_descriptive_risk["Length (m)"].sum()
            descriptive_risk_averages[descriptive_col].loc[
                descriptive_risk_averages[descriptive_col][descriptive_col]
                == descriptive_feature,
                "Length (m)",
            ] = round(total_desc_length)
            for scenario in Scenarios:
                risk_column = f"{hazard}_{scenario}"
                out_risk_column = f"{scenario.capitalize()} {hazard.replace('_', ' ').title()}"
                length_weighted_avg = round(
                    (
                        asset_descriptive_risk[risk_column]
                        * asset_descriptive_risk["Length (m)"]
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


def _hazard_distribution_plot(
    risk_distribution: pd.DataFrame,
    hazard: MainHazardRiskCols
    | ExtremeWeatherRiskCols
    | FloodingRiskCols
    | GroundStabilityRiskCols
    | CoastalErosionRiskCols,
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
        risk_distribution[risk_distribution["Risk Category"] == "Very High"][x]
        .iloc[0]
        .to_numpy()
    )
    high = (
        risk_distribution[risk_distribution["Risk Category"] == "High"][x].iloc[0].to_numpy()
    )
    medium = (
        risk_distribution[risk_distribution["Risk Category"] == "Medium"][x].iloc[0].to_numpy()
    )
    low = risk_distribution[risk_distribution["Risk Category"] == "Low"][x].iloc[0].to_numpy()
    very_low = (
        risk_distribution[risk_distribution["Risk Category"] == "Very Low"][x]
        .iloc[0]
        .to_numpy()
    )

    cmap = hazard.get_cmap()
    cmap = plt.get_cmap(cmap)

    ax.bar(x_labels, very_low, color=cmap(0.2), label="Very Low")
    ax.bar(x_labels, low, bottom=very_low, color=cmap(0.4), label="Low")
    ax.bar(x_labels, medium, bottom=very_low + low, color=cmap(0.6), label="Medium")
    ax.bar(x_labels, high, bottom=very_low + low + medium, color=cmap(0.8), label="High")
    ax.bar(
        x_labels,
        very_high,
        bottom=very_low + low + medium + high,
        color=cmap(1.0),
        label="Very High",
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


def _risk_averages_plot(risk_averages: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Generate a plot for risk averages."""
    out_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    current = risk_averages[f"Average {Scenarios.CURRENT.capitalize()} Risk"].to_numpy()
    forecast = risk_averages[f"Average {Scenarios.FORECAST.capitalize()} Risk"].to_numpy()
    hazards = [hazard.replace(" Risk", "") for hazard in risk_averages["Hazard"].to_list()]
    x = np.arange(len(hazards))

    width = 0.40

    ax.bar(
        x - 0.2, current, width=width, color="royalblue", label=Scenarios.CURRENT.capitalize()
    )
    ax.bar(
        x + 0.2,
        forecast,
        width=width,
        color="darkorange",
        label=Scenarios.FORECAST.capitalize(),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(hazards)
    ax.set_xlabel("Hazard")
    ax.set_ylabel("Length-Weighted Average Risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "risk_averages.png")


def _write_risk_summary(
    risk_distribution: pd.DataFrame,
    risk_averages: pd.DataFrame,
    descriptive_risk_averages: pd.DataFrame,
    out_path: pathlib.Path,
) -> None:
    """Write a risk summary to an Excel file."""
    wb = openpyxl.Workbook()

    thin_border = openpyxl.styles.Side(border_style="thin", color="FFFFFF")
    header_left = openpyxl.styles.Border(left=thin_border, top=thin_border, bottom=thin_border)
    header_right = openpyxl.styles.Border(
        right=thin_border, top=thin_border, bottom=thin_border
    )

    cell_left = openpyxl.styles.Border(left=thin_border)
    cell_right = openpyxl.styles.Border(right=thin_border)

    wb = _write_risk_distribution(
        workbook=wb,
        risk_distribution=risk_distribution,
        thin_border=thin_border,
        out_path=out_path
    )

    wb = _write_risk_averages(
        workbook=wb,
        risk_averages=risk_averages,
        thin_border=thin_border,
        out_path=out_path
    )

    wb = _write_descriptive_risk_averages(
        workbook=wb,
        thin_border=thin_border,
        descriptive_risk_averages=descriptive_risk_averages,

    )

    wb.save(out_path / "Risk Summary.xlsx")
    wb.close()


def _write_risk_distribution(
    workbook: openpyxl.Workbook,
    risk_distribution: pd.DataFrame,
    thin_border: openpyxl.styles.Side,
    out_path: pathlib.Path,
) -> openpyxl.Workbook:
    """Write risk distribution data and plots to an Excel workbook sheet."""
    ws = workbook.active
    ws.title = "Risk Distribution"

    for r_idx, row in enumerate(
        dataframe_to_rows(risk_distribution, index=False, header=True), 1
    ):
        for c_idx, value in enumerate(row, 1):
            column_letter = openpyxl.utils.get_column_letter(c_idx)
            ws.column_dimensions[column_letter].width = 15
            horizontal_alignment = "left"  # Default alignment for all cells

            text_colour = "FFFFFF"  # White text

            # Determine formatting dynamically
            if r_idx == 1:  # Apply fill only to header row:
                fill_colour = "000000"  # black
                bold = True
                border = openpyxl.styles.Border(bottom=thin_border)
                if c_idx % 2 != 0:
                    border = openpyxl.styles.Border(left=thin_border, top=thin_border, bottom=thin_border)
                else:
                    border = openpyxl.styles.Border(right=thin_border, top=thin_border, bottom=thin_border)
            else:
                fill_colour = "808080"  # medium grey
                bold = False
                if c_idx % 2 != 0:
                    border = openpyxl.styles.Border(left=thin_border)
                else:
                    border = openpyxl.styles.Border(right=thin_border)

            if isinstance(value, (int, float)) and value != 0:
                cell = ws.cell(row=r_idx, column=c_idx, value=value / 100)

                bar_colour = _RISK_CATEGORY_COLOURS[row[0]]
                cell.number_format = "0%"

                data_bar = openpyxl.formatting.rule.DataBarRule(
                    start_type="num",
                    start_value=0,
                    end_type="num",
                    end_value=1,
                    color=bar_colour,
                    showValue=True,
                    minLength=None,
                    maxLength=None,
                )
                ws.conditional_formatting.add(cell.coordinate, data_bar)

            else:
                cell = ws.cell(row=r_idx, column=c_idx, value=value)

            # Format cell
            _style_cell(
                cell,
                fill_colour=fill_colour,
                text_colour=text_colour,
                bold=bold,
                border=border,
                alignment=horizontal_alignment,
            )

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
    image_row = len(risk_distribution) + 3
    image_column = 2
    for hazard in hazards:
        risk_distribution_image = openpyxl.drawing.image.Image(
            out_path
            / "Risk Distribution"
            / f"risk_distribution_{hazard.lower().replace(' ', '_')}.png"
        )
        risk_distribution_image.width = 400
        risk_distribution_image.height = 400

        ws.add_image(
            risk_distribution_image,
            f"{openpyxl.utils.get_column_letter(image_column)}{image_row}",
        )
        image_column += 4
        if image_column > len(risk_distribution.columns):
            image_column = 2
            image_row += 21

    return workbook


def _write_risk_averages(
    workbook: openpyxl.Workbook,
    risk_averages: pd.DataFrame,
    thin_border: openpyxl.styles.Side,
    out_path: pathlib.Path,
) -> openpyxl.Workbook:
    """Write risk averages data and plots to an Excel workbook sheet."""
    ws = workbook.create_sheet(title="Risk Averages")
    gn_yl_red_cmap = mpl.colormaps["RdYlGn_r"]
    oranges_cmap = mpl.colormaps["Oranges"]

    # Write risk averages data to the sheet
    for r_idx, row in enumerate(dataframe_to_rows(risk_averages, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            column_letter = openpyxl.utils.get_column_letter(c_idx)
            ws.column_dimensions[column_letter].width = 15
            horizontal_alignment = "left"  # Default alignment for all cells

            # Determine formatting dynamically
            if r_idx == 1:  # Apply fill only to header row:
                fill_colour = "000000"  # black
                bold = True
                text_colour = "FFFFFF" # white
            elif c_idx == 1:
                if r_idx % 2 == 0:
                    fill_colour = "a9a9a9"  # dark grey for even rows
                else:
                    fill_colour = "808080"  # grey for odd rows
                bold = False
                text_colour = "000000" # black

            if isinstance(value, (int, float)):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                if c_idx in [2, 3]:
                    fill_colour = mpl.colors.to_hex(gn_yl_red_cmap(value / 100)).replace("#", "")
                elif c_idx == 4:
                    if value > 0:
                        fill_colour = "90ee90" # light green for positive values
                    else:
                        fill_colour = "db7093" # pale violet red for negative values
                elif c_idx == 5:
                    fill_colour = mpl.colors.to_hex(oranges_cmap(value / 100)).replace("#", "")
            else:
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == 1:
                    border = openpyxl.styles.Border(right=thin_border, top=thin_border, bottom=thin_border)
                else:
                    border = openpyxl.styles.Border(right=thin_border)

            # Format cell
            _style_cell(
                cell=cell,
                fill_colour=fill_colour,
                text_colour=text_colour,
                bold=bold,
                alignment=horizontal_alignment,
                border=border,
            )

    image_row = len(risk_averages) + 3
    image_column = 1

    risk_averages_image = openpyxl.drawing.image.Image(
        out_path / "Risk Averages" / "risk_averages.png"
    )
    risk_averages_image.width = 500
    risk_averages_image.height = 300

    ws.add_image(
        risk_averages_image,
        f"{openpyxl.utils.get_column_letter(image_column)}{image_row}",
    )

    return workbook


def _write_descriptive_risk_averages(
    workbook: openpyxl.Workbook,
    descriptive_risk_averages: dict[str, pd.DataFrame],
    thin_border: openpyxl.styles.Side,
) -> openpyxl.Workbook:
    """Write descriptive risk averages data to the Excel workbook sheet."""
    # TODO (DJ): Need to clean up the structure column names before they go into this
    ws = workbook.create_sheet(title="Descriptive Risk Averages")
    gn_yl_red_cmap = mpl.colormaps["RdYlGn_r"]
    total_length = 0

    for _, risk_averages in descriptive_risk_averages.items():
        for r_idx, row in enumerate(dataframe_to_rows(risk_averages, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                column_letter = openpyxl.utils.get_column_letter(c_idx)
                ws.column_dimensions[column_letter].width = 15
                horizontal_alignment = "left"  # Default alignment for all cells
                current_row = total_length + r_idx

                cell = ws.cell(row=current_row, column=c_idx, value=value)

                fill_colour = "FFFFFF"  # default white fill
                text_colour = "000000"  # default black text
                bold = False
                border = None  # default border for all cells

                if r_idx == 1:  # Apply fill only to header row:
                    fill_colour = "008080"  # teal
                    bold = True
                    text_colour = "FFFFFF" # white
                    border = openpyxl.styles.Border(top=thin_border, bottom=thin_border)
                    if c_idx == 1:
                        border = openpyxl.styles.Border(left=thin_border, top=thin_border, bottom=thin_border)
                    elif c_idx == 2 or (c_idx - 3) % 3 not in [0, 1]:
                        border = openpyxl.styles.Border(right=thin_border, top=thin_border, bottom=thin_border)

                elif c_idx in [1, 2]:
                    if r_idx % 2 == 0:
                        fill_colour = "a9a9a9"  # dark grey for even rows
                    else:
                        fill_colour = "808080"  # grey for odd rows
                    bold = False
                    if c_idx == 2:
                        border = openpyxl.styles.Border(right=thin_border)
                elif (c_idx - 3) % 3 in [0, 1]:
                    fill_colour = mpl.colors.to_hex(gn_yl_red_cmap(value / 100)).replace("#", "")
                else:
                    border = openpyxl.styles.Border(right=thin_border)
                    if value > 0:
                        cell.value = f"↑ {value:.1f}"
                        text_colour = "F8696B" # red
                    elif value < 0:
                        cell.value = f"↓ {value:.1f}"
                        text_colour = "63BE7B" # green
                    else:
                        cell.value = f"→ {value:.1f}"
                        text_colour = "FFEB84" # yellow


                # Format cell
                _style_cell(
                    cell=cell,
                    fill_colour=fill_colour,
                    text_colour=text_colour,
                    bold=bold,
                    border=border,
                    alignment=horizontal_alignment,
                )

        total_length += len(risk_averages) + 2


    return workbook


def _style_cell(
        cell: openpyxl.cell.Cell,
        *,
        fill_colour: str,
        text_colour: str = "000000",
        bold: bool = False,
        border: openpyxl.styles.borders.Border | None = None,
        alignment: str = "left"
) -> None:
    """Apply styling to a given cell."""
    cell.fill = openpyxl.styles.PatternFill(
        fill_type="solid",
        start_color=fill_colour,
        end_color=fill_colour,
    )
    cell.font = openpyxl.styles.Font(
        name="Verdana",
        color=text_colour,
        bold=bold,
    )
    cell.alignment = openpyxl.styles.Alignment(
        wrap_text=True,
        horizontal=alignment
    )
    if border:
        cell.border = border


# LAYERING


def layering(config: model_config.Config) -> None:
    """
    Layer infrastructure with hazard risk to assign risk to each piece of infrastructure.

    Read in hazard layers from functional rules output, then spatially intersect with
    infrastructure layers to assign risk to each piece of infrastructure. Calculate impact
    index for relevant data.

    Parameters
    ----------
    config : Config
        Main config for the model, containing paths and settings.
    """
    hazard_layers = _read_hazard_layers(config)

    risk_cols = []

    if config.switches.extreme_weather:
        risk_cols.extend([MainHazardRiskCols.EXTREME_WEATHER, *ExtremeWeatherRiskCols])

    if config.switches.flooding:
        risk_cols.extend([MainHazardRiskCols.FLOODING, *FloodingRiskCols])

    if config.switches.ground_stability:
        risk_cols.extend([MainHazardRiskCols.GROUND_STABILITY, *GroundStabilityRiskCols])

    if config.switches.coastal_erosion:
        risk_cols.extend([MainHazardRiskCols.COASTAL_EROSION])

    audit_path = config.paths.audit_path / "Layering"

    _infrastructure_layering(config, hazard_layers, risk_cols, audit_path)


## HAZARD LAYERS


def _read_hazard_layers(
    config: model_config.Config,
) -> dict[MainHazardRiskCols, gpd.GeoDataFrame]:
    """Read and clean hazard layers, and return them in a dictionary."""
    hazard_layers = {}
    if config.switches.extreme_weather:
        LOG.info("Reading extreme weather layer.")
        hazard_layers[MainHazardRiskCols.EXTREME_WEATHER] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.EXTREME_WEATHER_MODEL_INTERIM_OUTPUT_PATH
        )
    if config.switches.flooding:
        LOG.info("Reading flooding layer.")
        hazard_layers[MainHazardRiskCols.FLOODING] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.FLOODING_RISK_MODEL_INTERIM_OUTPUT_PATH
        )
    if config.switches.ground_stability:
        LOG.info("Reading ground stability layer.")
        hazard_layers[MainHazardRiskCols.GROUND_STABILITY] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.GROUND_STABILITY_MODEL_INTERIM_OUTPUT_PATH
        )
    if config.switches.coastal_erosion:
        LOG.info("Reading coastal erosion layer.")
        hazard_layers[MainHazardRiskCols.COASTAL_EROSION] = gpd.read_file(
            config.paths.model_interim_output
            / file_paths.COASTAL_EROSION_MODEL_INTERIM_OUTPUT_PATH
        )
    return hazard_layers


## INFRASTRUCTURE-HAZARD LAYERING


def _infrastructure_layering(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer roads, rail, and other infrastructure with hazards."""
    _get_road_risk(config, hazard_layers, risk_cols, audit_path)
    _get_rail_risk(config, hazard_layers, risk_cols, audit_path)
    _get_other_risk(config, hazard_layers, risk_cols, audit_path)
    _get_bespoke_risk(config, hazard_layers, risk_cols, audit_path)


### ROAD


def _get_road_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer OS Open Roads, NoHAM, and Transport Model roads with hazards to assign risk."""
    road_risk_enabled = any(
        [config.switches.all_roads, config.switches.noham_roads, config.switches.model_roads]
    )
    if road_risk_enabled:
        LOG.info("Calculating road risk...")
        if config.switches.all_roads:
            _os_open_road_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.noham_roads:
            _noham_road_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.model_roads:
            _model_road_risk(config, hazard_layers, risk_cols, audit_path)
        LOG.info("Road risk calculation complete.")


#### OS Open Roads


def _os_open_road_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Intersect OS Road infrastructure with hazards, clean output, and write to file."""
    LOG.info("Layering OS Open Roads with hazard risk...")
    os_road = gpd.read_file(config.paths.model_input / file_paths.OS_ROAD_MODEL_INPUT_PATH)

    if os_road.empty:
        LOG.warning("OS Open Roads layer is empty. Skipping.")
        return

    os_road_risk = _infrastructure_risk_intersect(os_road, hazard_layers)

    os_road_risk = _apply_asset_hazard_weighting(
        os_road_risk, AssetTypes.ROAD, hazards=hazard_layers
    )

    os_road_risk = _apply_asset_vulnerability(
        os_road_risk,
        structure_enum=OSRoadStructure,
        structure_col=OSRoadCols.ROAD_STRUCTURE,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    _create_risk_summary(
        os_road_risk,
        audit_path / "Summary" / "Road" / "OS Roads",
    )

    _audit_infrastructure_risk(
        os_road_risk,
        "All Roads",
        risk_cols,
        audit_path / "Road" / "OS Roads",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        os_road_risk,
        config.paths.model_output / "Road" / "OS Roads" / "os_road_risk.gpkg",
    )

    os_road_risk = _prepare_model_output(
        risk_data=os_road_risk,
        drop_cols=[],
        rename_map={"identifier": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config, os_road_risk, "id", pathlib.Path("Road") / "OS Roads" / "os_road_risk"
    )
    LOG.info("Finished layering OS Open Roads with hazard risk.")


#### NoHAM Roads


def _noham_road_risk(
    config: model_config.Config,
    hazard_layers: dict[str, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get NoHAM road risk and write to file.

    Intersect NoHAM with hazards, calculate impact index, clean output, and write
    to file.
    """
    LOG.info("Layering NoHAM with hazard risk and calculating impact index...")
    noham_net_flows = gpd.read_file(
        config.paths.model_input / file_paths.NOHAM_FLOWS_MODEL_INPUT_PATH
    )

    if noham_net_flows.empty:
        LOG.warning("NoHAM network flows layer is empty. Skipping.")
        return

    noham_risk = _infrastructure_risk_intersect(noham_net_flows, hazard_layers)

    feature_range = (config.constants.score_min, config.constants.score_max)
    noham_risk = _noham_impact_index(noham_risk, feature_range)

    risk_impact_cols = [*risk_cols, *ImpactCols.get_noham_impact_cols()]

    _audit_infrastructure_risk(
        noham_risk,
        "NoHAM Roads",
        risk_impact_cols,
        audit_path / "Road" / "NoHAM",
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        noham_risk,
        config.paths.model_output / "Road" / "NoHAM" / "noham_risk.gpkg",
    )

    noham_risk = _prepare_model_output(
        risk_data=noham_risk,
        drop_cols=[],
        rename_map={"link_id": "id"},
        risk_cols_order=risk_impact_cols,
    )

    _split_csv_shapefile(
        config, noham_risk, "id", pathlib.Path("Road") / "NoHAM" / "noham_risk"
    )

    LOG.info("Finished layering NoHAM with hazard risk and calculating impact index.")


def _noham_impact_index(
    noham: gpd.GeoDataFrame, feature_range: tuple[int, int]
) -> gpd.GeoDataFrame:
    """Normalise NoHAM demand, then calculate impact index."""
    noham = _normalise_uc_demand(noham, feature_range)
    noham = _normalise_total_demand(noham, feature_range)
    noham = _calculate_noham_impact(noham)
    return _normalise_noham_impact(noham, feature_range)


def _normalise_uc_demand(noham: pd.DataFrame, feature_range: tuple[int, int]) -> pd.DataFrame:
    """Normalise NoHAM demand for each user class individually."""
    noham_ucs = UserClasses.get_noham_classes()
    pairs = [
        (f"{uc}_total_{Scenarios.CURRENT}", f"{uc}_total_{Scenarios.FORECAST}")
        for uc in noham_ucs
    ]

    noham = functional_rules.min_max_scaling_pair(noham, pairs, feature_range)

    rename_map = {
        col: col.replace("total", "demand")
        for uc in noham_ucs
        for col in [f"{uc}_total_{Scenarios.CURRENT}", f"{uc}_total_{Scenarios.FORECAST}"]
    }
    return noham.rename(columns=rename_map)


def _normalise_total_demand(
    noham: pd.DataFrame, feature_range: tuple[int, int]
) -> pd.DataFrame:
    """Normalise NoHAM demand across all user classes combined."""
    pairs = [(f"all_vehs_total_{Scenarios.CURRENT}", f"all_vehs_total_{Scenarios.FORECAST}")]
    noham = functional_rules.min_max_scaling_pair(noham, pairs, feature_range)
    return noham.rename(
        columns={
            f"all_vehs_total_{Scenarios.CURRENT}": f"demand_{Scenarios.CURRENT}",
            f"all_vehs_total_{Scenarios.FORECAST}": f"demand_{Scenarios.FORECAST}",
        }
    )


def _calculate_noham_impact(noham: pd.DataFrame) -> pd.DataFrame:
    """Calculate NoHAM impact score for each user class, and for all vehicles."""
    # Calculate impact metric for each user class
    risk_cols = [
        col for col in MainHazardRiskCols if f"{col}_{Scenarios.CURRENT}" in noham.columns
    ]

    hazards = [col.removesuffix("_risk") for col in risk_cols]
    impact_weights = _get_impact_weights(hazards)

    for scenario in Scenarios:
        hazard_component = sum(
            noham[f"{risk_col}_{scenario}"] * impact_weights[risk_col.removesuffix("_risk")]
            for risk_col in risk_cols
        )
        for uc in UserClasses.get_noham_classes():
            impact_component = noham[f"{uc}_demand_{scenario}"] * impact_weights["demand"]
            noham[f"{uc}_impact_{scenario}"] = impact_component + hazard_component

        impact_component = noham[f"demand_{scenario}"] * impact_weights["demand"]
        noham[f"impact_{scenario}"] = impact_component + hazard_component

    demand_cols = [col for col in noham.columns if "demand" in col]
    return noham.drop(columns=demand_cols)


def _normalise_noham_impact(
    noham: pd.DataFrame, feature_range: tuple[int, int]
) -> pd.DataFrame:
    """Normalise NoHAM impact scores across all user classes combined."""
    pairs = [
        (f"{uc}_impact_{Scenarios.CURRENT}", f"{uc}_impact_{Scenarios.FORECAST}")
        for uc in UserClasses.get_noham_classes()
    ] + [(f"impact_{Scenarios.CURRENT}", f"impact_{Scenarios.FORECAST}")]

    return functional_rules.min_max_scaling_pair(noham, pairs, feature_range)


#### TRANSPORT MODEL ROADS


def _model_road_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get transport model road risk and write to file.

    Intersect transport model roads with hazards, calculate impact index, clean output, and
    write to file.
    """
    LOG.info("Layering transport model roads with hazard risk and calculating impact index...")
    model_net_flows = gpd.read_file(
        config.paths.model_input / file_paths.MODEL_ROAD_FLOWS_MODEL_INPUT_PATH
    )

    if model_net_flows.empty:
        LOG.warning("Transport model network flows layer is empty. Skipping.")
        return

    model_road_risk = _infrastructure_risk_intersect(model_net_flows, hazard_layers)

    model_road_risk = _apply_asset_hazard_weighting(
        model_road_risk, AssetTypes.ROAD, hazards=hazard_layers
    )

    # TODO (DJ): (#26) Apply asset-specific vulnerability modifiers for model roads.

    feature_range = (config.constants.score_min, config.constants.score_max)
    model_road_risk = _model_road_impact_index(model_road_risk, feature_range)

    risk_impact_cols = [*risk_cols, *ImpactCols]

    _audit_infrastructure_risk(
        model_road_risk,
        "Model Roads",
        risk_impact_cols,
        audit_path / "Road" / "Model Roads",
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        model_road_risk,
        config.paths.model_output / "Road" / "Model Roads" / "model_road_risk.gpkg",
    )

    model_road_risk = _prepare_model_output(
        risk_data=model_road_risk,
        drop_cols=[],
        rename_map={"link_id": "id"},
        risk_cols_order=risk_impact_cols,
    )

    _split_csv_shapefile(
        config, model_road_risk, "id", pathlib.Path("Road") / "Model Roads" / "model_road_risk"
    )

    LOG.info(
        "Finished layering transport model roads with hazards and calculating impact index."
    )


def _model_road_impact_index(
    model_road_risk: gpd.GeoDataFrame, feature_range: tuple[int, int]
) -> gpd.GeoDataFrame:
    """Normalise transport model road demand, then calculate impact index."""
    # First, normalise demand for each user class individually and for total demand
    model_road_risk = functional_rules.min_max_scaling_pair(
        data=model_road_risk,
        pairs=[(f"demand_{Scenarios.CURRENT}", f"demand_{Scenarios.FORECAST}")]
        + [
            (f"{uc}_demand_{Scenarios.CURRENT}", f"{uc}_demand_{Scenarios.FORECAST}")
            for uc in UserClasses
        ],
        feature_range=feature_range,
    )

    # Then, calculate impact index for each user class and for total demand
    model_road_risk = _calculate_model_road_impact(model_road_risk)

    # Finally, normalise the impact index for each user class and for total demand
    return functional_rules.min_max_scaling_pair(
        data=model_road_risk,
        pairs=[
            (f"{uc}_impact_{Scenarios.CURRENT}", f"{uc}_impact_{Scenarios.FORECAST}")
            for uc in UserClasses
        ]
        + [(f"impact_{Scenarios.CURRENT}", f"impact_{Scenarios.FORECAST}")],
        feature_range=feature_range,
    )


def _calculate_model_road_impact(model_road_risk: pd.DataFrame) -> pd.DataFrame:
    """Calculate transport model impact score for each user class, and for all vehicles."""
    # Calculate impact metric for each user class
    risk_cols = [
        col
        for col in MainHazardRiskCols
        if f"{col}_{Scenarios.CURRENT}" in model_road_risk.columns
    ]

    hazards = [col.removesuffix("_risk") for col in risk_cols]
    impact_weights = _get_impact_weights(hazards)

    for scenario in Scenarios:
        hazard_component = sum(
            model_road_risk[f"{risk_col}_{scenario}"]
            * impact_weights[risk_col.removesuffix("_risk")]
            for risk_col in risk_cols
        )
        for uc in UserClasses:
            impact_component = (
                model_road_risk[f"{uc}_demand_{scenario}"] * impact_weights["demand"]
            )
            model_road_risk[f"{uc}_impact_{scenario}"] = impact_component + hazard_component

        impact_component = model_road_risk[f"demand_{scenario}"] * impact_weights["demand"]
        model_road_risk[f"impact_{scenario}"] = impact_component + hazard_component

    demand_cols = [col for col in model_road_risk.columns if "demand" in col]
    return model_road_risk.drop(columns=demand_cols)


### RAIL


def _get_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer passenger rail and freight rail network with hazard to assign risk."""
    rail_risk_enabled = any([config.switches.passenger_rail, config.switches.freight_rail])
    if rail_risk_enabled:
        LOG.info("Calculating rail risk...")
        if config.switches.passenger_rail:
            _passenger_rail_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.freight_rail:
            _freight_rail_risk(config, hazard_layers, risk_cols, audit_path)
        LOG.info("Rail risk calculation complete.")


#### Passenger Rail


def _passenger_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Intersect passenger rail network with hazard to assign risk, clean and write to file."""
    LOG.info("Layering passenger rail network with hazard risk...")
    passenger_rail_network = gpd.read_file(
        config.paths.model_input / file_paths.PASSENGER_RAIL_MODEL_INPUT_PATH
    )

    if passenger_rail_network.empty:
        LOG.warning("Passenger rail network layer is empty. Skipping.")
        return

    passenger_rail_network_risk = _infrastructure_risk_intersect(
        passenger_rail_network, hazard_layers
    )

    passenger_rail_network_risk = _apply_asset_hazard_weighting(
        passenger_rail_network_risk, AssetTypes.RAIL, hazards=hazard_layers
    )

    passenger_rail_network_risk = _apply_asset_vulnerability(
        passenger_rail_network_risk,
        structure_enum=OSRailStructure,
        structure_col=OSRailCols.STRUCTURE,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    _create_risk_summary(
        passenger_rail_network_risk,
        [
            OSRailCols.DESCRIPTION,
            OSRailCols.STRUCTURE,
            OSRailCols.PHYSICAL_LEVEL,
            OSRailCols.RAILWAY_USE,
            OSRailCols.TRACK_REPRESENTATION,
        ],
        config.paths.audit_path / "Summary" / "Rail" / "Passenger Rail",
    )

    _audit_infrastructure_risk(
        passenger_rail_network_risk,
        "Passenger Rail",
        risk_cols,
        audit_path / "Rail" / "Passenger Rail",
        linewidth=1.0,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        passenger_rail_network_risk,
        config.paths.model_output
        / "Rail"
        / "Passenger Rail"
        / "passenger_rail_network_risk.gpkg",
    )

    passenger_rail_network_risk = _prepare_model_output(
        risk_data=passenger_rail_network_risk,
        drop_cols=[],
        rename_map={
            OSRailCols.ID: "id",
            OSRailCols.PHYSICAL_LEVEL: "physical_level",
            OSRailCols.RAILWAY_USE: "railway_use",
            OSRailCols.TRACK_REPRESENTATION: "track_representation",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        passenger_rail_network_risk,
        "id",
        pathlib.Path("Rail") / "Passenger Rail" / "passenger_rail_network_risk",
    )

    LOG.info("Finished layering passenger rail network with hazard risk.")


#### Freight Rail


def _freight_rail_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Calculate freight rail risk and write to file.

    Intersect freight network with hazard risk, calculate impact index, clean and write to
    file.
    """
    LOG.info("Layering freight rail network with hazard risk and calculating impact index...")
    freight_rail_network = gpd.read_file(
        config.paths.model_input / file_paths.FREIGHT_DEMAND_MODEL_INPUT_PATH
    )

    if freight_rail_network.empty:
        LOG.warning("Freight rail network layer is empty. Skipping.")
        return

    freight_rail_network_risk = _infrastructure_risk_intersect(
        freight_rail_network, hazard_layers
    )

    freight_rail_network_risk = _apply_asset_hazard_weighting(
        freight_rail_network_risk, AssetTypes.RAIL, hazards=hazard_layers
    )

    feature_range = (config.constants.score_min, config.constants.score_max)

    freight_rail_network_risk = _apply_asset_vulnerability(
        freight_rail_network_risk,
        structure_enum=OSRailStructure,
        structure_col=OSRailCols.STRUCTURE,
        feature_range=feature_range,
    )

    freight_rail_network_risk = _freight_impact_index(freight_rail_network_risk, feature_range)

    # Set the correct CRS
    freight_rail_network_risk = freight_rail_network_risk.set_crs(
        data_cleaning.BNG_CRS, allow_override=True
    )

    _audit_infrastructure_risk(
        freight_rail_network_risk,
        "Freight Rail",
        [*risk_cols, ImpactCols.IMPACT],
        audit_path / "Rail" / "Freight Rail",
        linewidth=1.0,
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        freight_rail_network_risk,
        config.paths.model_output / "Rail" / "Freight Rail" / "freight_rail_network_risk.gpkg",
    )

    freight_rail_network_risk = _prepare_model_output(
        risk_data=freight_rail_network_risk,
        drop_cols=[
            "dij_id",
            "distance",
        ],
        rename_map={
            OSRailCols.ID: "id",
            OSRailCols.PHYSICAL_LEVEL: "physical_level",
            OSRailCols.RAILWAY_USE: "railway_use",
            OSRailCols.TRACK_REPRESENTATION: "track_representation",
        },
        risk_cols_order=[*risk_cols, ImpactCols.IMPACT],
    )

    _split_csv_shapefile(
        config,
        freight_rail_network_risk,
        "id",
        pathlib.Path("Rail") / "Freight Rail" / "freight_rail_network_risk",
    )
    LOG.info(
        "Finished layering freight rail network with hazard risk and calculating impact index."
    )


def _freight_impact_index(
    freight_rail_network_risk: gpd.GeoDataFrame, feature_range: tuple[int, int]
) -> gpd.GeoDataFrame:
    """Calculate impact index using freight demand data and hazard risk."""
    freight_rail_network_risk = functional_rules.min_max_scaling_pair(
        freight_rail_network_risk,
        [(f"demand_{Scenarios.CURRENT}", f"demand_{Scenarios.FORECAST}")],
        feature_range,
    )

    freight_rail_network_risk = _calculate_freight_impact(freight_rail_network_risk)

    freight_rail_network_risk = functional_rules.min_max_scaling_pair(
        freight_rail_network_risk,
        [(f"impact_{Scenarios.CURRENT}", f"impact_{Scenarios.FORECAST}")],
        feature_range,
    )

    return gpd.GeoDataFrame(freight_rail_network_risk, geometry="geometry", crs="EPSG:4326")


def _calculate_freight_impact(freight_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite impact score for current and forecast years."""
    risk_cols = [
        col
        for col in MainHazardRiskCols
        if f"{col}_{Scenarios.CURRENT}" in freight_data.columns
    ]

    hazards = [col.removesuffix("_risk") for col in risk_cols]
    impact_weights = _get_impact_weights(hazards)

    for scenario in Scenarios:
        impact_component = freight_data[f"demand_{scenario}"] * impact_weights["demand"]
        hazard_component = sum(
            freight_data[f"{risk_col}_{scenario}"]
            * impact_weights[risk_col.removesuffix("_risk")]
            for risk_col in risk_cols
        )

        freight_data[f"impact_{scenario}"] = impact_component + hazard_component

    demand_cols = [col for col in freight_data.columns if "demand" in col]
    return freight_data.drop(columns=demand_cols)


### OTHER


def _get_other_risk(  # noqa: C901, PLR0912
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Layer other infrastructure with hazards to assign risk."""
    other_risk_enabled = any(
        [
            config.switches.train_stations,
            config.switches.charging_sites,
            config.switches.airports,
            config.switches.bus_coach_stations,
            config.switches.bus_stops,
            config.switches.tram_stations,
            config.switches.rapid_transport_stations,
            config.switches.ferry_terminals,
            config.switches.petrol_stations,
            config.switches.national_cycle_network,
            config.switches.tram_network,
            config.switches.rapid_transport_network,
        ]
    )
    if other_risk_enabled:
        LOG.info("Calculating risk for other infrastructure...")
        if config.switches.train_stations:
            _train_stations_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.charging_sites:
            _charging_sites_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.airports:
            _airports_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.bus_coach_stations:
            _bus_coach_stations_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.bus_stops:
            _bus_stops_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.tram_stations:
            _tram_stations_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.rapid_transport_stations:
            _rapid_transport_stations_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.ferry_terminals:
            _ferry_terminals_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.petrol_stations:
            _petrol_stations_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.national_cycle_network:
            _ncn_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.tram_network:
            _tram_network_risk(config, hazard_layers, risk_cols, audit_path)
        if config.switches.rapid_transport_network:
            _rapid_transport_network_risk(config, hazard_layers, risk_cols, audit_path)
        LOG.info("Risk calculation for other infrastructure complete.")


def _buffer_geometry(infrastructure: gpd.GeoDataFrame, buffer_size_m: int) -> gpd.GeoDataFrame:
    """Buffers the geometries of a given GeoDataFrame to a given size in metres."""
    infrastructure = infrastructure.to_crs(data_cleaning.BNG_CRS)
    infrastructure["geometry"] = infrastructure.buffer(buffer_size_m)
    return infrastructure


#### Train Stations


def _train_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get train station risk and write to file.

    Buffer train stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering train stations with hazard risk...")
    train_stations = gpd.read_file(
        config.paths.model_input / file_paths.TRAIN_STATIONS_MODEL_INPUT_PATH
    )

    if train_stations.empty:
        LOG.warning("Train stations layer is empty. Skipping.")
        return

    train_stations = _buffer_geometry(train_stations, _TRAIN_STATIONS_BUFFER_SIZE_M)

    train_stations_risk = _infrastructure_risk_intersect(train_stations, hazard_layers)

    _audit_infrastructure_risk(
        train_stations_risk,
        "Train Stations",
        risk_cols,
        audit_path / "Other" / "Train Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        train_stations_risk,
        config.paths.model_output / "Other" / "Train Stations" / "train_stations_risk.gpkg",
    )

    train_stations_risk = _prepare_model_output(
        risk_data=train_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        train_stations_risk,
        "id",
        pathlib.Path("Other") / "Train Stations" / "train_stations_risk",
    )
    LOG.info("Finished layering train stations with hazard risk.")


#### EV Charging Sites


def _charging_sites_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get EV charging site risk and write to file.

    Buffer charging sites, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering EV charging sites with hazard risk...")
    charging_sites = gpd.read_file(
        config.paths.model_input / file_paths.CHARGING_SITES_MODEL_INPUT_PATH
    )

    if charging_sites.empty:
        LOG.warning("EV charging sites layer is empty. Skipping.")
        return

    charging_sites = _buffer_geometry(charging_sites, _CHARGING_SITES_BUFFER_SIZE_M)

    charging_sites_risk = _infrastructure_risk_intersect(charging_sites, hazard_layers)

    _audit_infrastructure_risk(
        charging_sites_risk,
        "EV Charging Sites",
        risk_cols,
        audit_path / "Other" / "EV Charging Sites",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        charging_sites_risk,
        config.paths.model_output / "Other" / "EV Charging Sites" / "charging_sites_risk.gpkg",
    )

    charging_sites_risk = _prepare_model_output(
        risk_data=charging_sites_risk,
        drop_cols=[],
        rename_map={"devices": "installed_devices"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        charging_sites_risk,
        "id",
        pathlib.Path("Other") / "EV Charging Sites" / "charging_sites_risk",
    )
    LOG.info("Finished layering EV charging sites with hazard risk.")


#### Airports


def _airports_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get airport risk and write to file.

    Intersect airports with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering airports with hazard risk...")
    airports = gpd.read_file(config.paths.model_input / file_paths.AIRPORTS_MODEL_INPUT_PATH)

    if airports.empty:
        LOG.warning("Airports layer is empty. Skipping.")
        return

    airports_risk = _infrastructure_risk_intersect(airports, hazard_layers)

    _audit_infrastructure_risk(
        airports_risk,
        "Airports",
        risk_cols,
        audit_path / "Other" / "Airports",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    airports_risk = data_cleaning.explode_to_polygons(airports_risk)

    data_cleaning.write_to_file(
        airports_risk,
        config.paths.model_output / "Other" / "Airports" / "airports_risk.gpkg",
    )

    airports_risk = _prepare_model_output(
        risk_data=airports_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        airports_risk,
        "id",
        pathlib.Path("Other") / "Airports" / "airports_risk",
    )
    LOG.info("Finished layering airports with hazard risk.")


#### Bus and Coach Stations


def _bus_coach_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get bus and coach station risk and write to file.

    Buffer bus and coach stations, then intersect with hazard risk, clean output, and write to
    file.
    """
    LOG.info("Layering bus and coach stations with hazard risk...")
    bus_coach_stations = gpd.read_file(
        config.paths.model_input / file_paths.BUS_COACH_STATIONS_MODEL_INPUT_PATH
    )

    if bus_coach_stations.empty:
        LOG.warning("Bus and coach stations layer is empty. Skipping.")
        return

    bus_coach_stations = _buffer_geometry(
        bus_coach_stations, _BUS_COACH_STATIONS_BUFFER_SIZE_M
    )

    bus_coach_stations_risk = _infrastructure_risk_intersect(bus_coach_stations, hazard_layers)

    _audit_infrastructure_risk(
        bus_coach_stations_risk,
        "Bus and Coach Stations",
        risk_cols,
        audit_path / "Other" / "Bus and Coach Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        bus_coach_stations_risk,
        config.paths.model_output
        / "Other"
        / "Bus and Coach Stations"
        / "bus_coach_stations_risk.gpkg",
    )

    bus_coach_stations_risk = _prepare_model_output(
        risk_data=bus_coach_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        bus_coach_stations_risk,
        "id",
        pathlib.Path("Other") / "Bus and Coach Stations" / "bus_coach_stations_risk",
    )
    LOG.info("Finished layering bus and coach stations with hazard risk.")


#### Bus Stops


def _bus_stops_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Intersect bus stops with hazard risk, clean output, and write to file."""
    LOG.info("Layering bus stops with hazard risk...")
    bus_stops = gpd.read_file(config.paths.model_input / file_paths.BUS_STOPS_MODEL_INPUT_PATH)

    if bus_stops.empty:
        LOG.warning("Bus stops layer is empty. Skipping.")
        return

    bus_stops_risk = _infrastructure_risk_intersect(bus_stops, hazard_layers)

    _audit_infrastructure_risk(
        bus_stops_risk,
        "Bus Stops",
        risk_cols,
        audit_path / "Other" / "Bus Stops",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        bus_stops_risk,
        config.paths.model_output / "Other" / "Bus Stops" / "bus_stops_risk.gpkg",
    )

    bus_stops_risk = _prepare_model_output(
        risk_data=bus_stops_risk,
        drop_cols=[],
        rename_map={"stop_id": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        bus_stops_risk,
        "id",
        pathlib.Path("Other") / "Bus Stops" / "bus_stops_risk",
    )
    LOG.info("Finished layering bus stops with hazard risk.")


#### Tram Stations


def _tram_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get tram station risk and write to file.

    Buffer tram stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering tram stations with hazard risk...")
    tram_stations = gpd.read_file(
        config.paths.model_input / file_paths.TRAM_STATIONS_MODEL_INPUT_PATH
    )

    if tram_stations.empty:
        LOG.warning("Tram stations layer is empty. Skipping.")
        return

    tram_stations = _buffer_geometry(tram_stations, _TRAM_STATIONS_BUFFER_SIZE_M)

    tram_stations_risk = _infrastructure_risk_intersect(tram_stations, hazard_layers)

    _audit_infrastructure_risk(
        tram_stations_risk,
        "Tram Stations",
        risk_cols,
        audit_path / "Other" / "Tram Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        tram_stations_risk,
        config.paths.model_output / "Other" / "Tram Stations" / "tram_stations_risk.gpkg",
    )

    tram_stations_risk = _prepare_model_output(
        risk_data=tram_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tram_stations_risk,
        "id",
        pathlib.Path("Other") / "Tram Stations" / "tram_stations_risk",
    )
    LOG.info("Finished layering tram stations with hazard risk.")


#### Rapid Transport Stations


def _rapid_transport_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get rapid transport station risk and write to file.

    Buffer rapid transport stations, then intersect with hazard risk, clean output, and write
    to file.
    """
    LOG.info("Layering rapid transport stations with hazard risk...")
    rapid_transport_stations = gpd.read_file(
        config.paths.model_input / file_paths.RAPID_TRANSPORT_STATIONS_MODEL_INPUT_PATH
    )

    if rapid_transport_stations.empty:
        LOG.warning("Rapid transport stations layer is empty. Skipping.")
        return

    rapid_transport_stations = _buffer_geometry(
        rapid_transport_stations, _RAPID_TRANSPORT_STATIONS_BUFFER_SIZE_M
    )

    rapid_transport_stations_risk = _infrastructure_risk_intersect(
        rapid_transport_stations, hazard_layers
    )

    _audit_infrastructure_risk(
        rapid_transport_stations_risk,
        "Rapid Transport Stations",
        risk_cols,
        audit_path / "Other" / "Rapid Transport Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        rapid_transport_stations_risk,
        config.paths.model_output
        / "Other"
        / "Rapid Transport Stations"
        / "rapid_transport_stations_risk.gpkg",
    )

    rapid_transport_stations_risk = _prepare_model_output(
        risk_data=rapid_transport_stations_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        rapid_transport_stations_risk,
        "id",
        pathlib.Path("Other") / "Rapid Transport Stations" / "rapid_transport_stations_risk",
    )
    LOG.info("Finished layering rapid transport stations with hazard risk.")


#### Ferry Terminals


def _ferry_terminals_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get ferry terminal risk and write to file.

    Buffer ferry terminals, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering ferry terminals with hazard risk...")
    ferry_terminals = gpd.read_file(
        config.paths.model_input / file_paths.FERRY_TERMINALS_MODEL_INPUT_PATH
    )

    if ferry_terminals.empty:
        LOG.warning("Ferry terminals layer is empty. Skipping.")
        return

    ferry_terminals = _buffer_geometry(ferry_terminals, _FERRY_TERMINALS_BUFFER_SIZE_M)

    ferry_terminals_risk = _infrastructure_risk_intersect(ferry_terminals, hazard_layers)

    _audit_infrastructure_risk(
        ferry_terminals_risk,
        "Ferry Terminals",
        risk_cols,
        audit_path / "Other" / "Ferry Terminals",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        ferry_terminals_risk,
        config.paths.model_output / "Other" / "Ferry Terminals" / "ferry_terminals_risk.gpkg",
    )

    ferry_terminals_risk = _prepare_model_output(
        risk_data=ferry_terminals_risk,
        drop_cols=[],
        rename_map={"nodeid": "id"},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        ferry_terminals_risk,
        "id",
        pathlib.Path("Other") / "Ferry Terminals" / "ferry_terminals_risk",
    )
    LOG.info("Finished layering ferry terminals with hazard risk.")


#### Petrol Stations


def _petrol_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get petrol station risk and write to file.

    Buffer petrol stations, then intersect with hazard risk, clean output, and write to file.
    """
    LOG.info("Layering petrol stations with hazard risk...")
    petrol_stations = gpd.read_file(
        config.paths.model_input / file_paths.PETROL_STATIONS_MODEL_INPUT_PATH
    )

    if petrol_stations.empty:
        LOG.warning("Petrol stations layer is empty. Skipping.")
        return

    petrol_stations = _buffer_geometry(petrol_stations, _PETROL_STATIONS_BUFFER_SIZE_M)

    petrol_stations_risk = _infrastructure_risk_intersect(petrol_stations, hazard_layers)

    _audit_infrastructure_risk(
        petrol_stations_risk,
        "Petrol Stations",
        risk_cols,
        audit_path / "Other" / "Petrol Stations",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        petrol_stations_risk,
        config.paths.model_output / "Other" / "Petrol Stations" / "petrol_stations_risk.gpkg",
    )

    petrol_stations_risk = _prepare_model_output(
        risk_data=petrol_stations_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        petrol_stations_risk,
        "id",
        pathlib.Path("Other") / "Petrol Stations" / "petrol_stations_risk",
    )
    LOG.info("Finished layering petrol stations with hazard risk.")


#### National Cycle Network


def _ncn_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get NCN risk and write to file.

    Intersect National Cycle Network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering National Cycle Network with hazard risk...")
    ncn = gpd.read_file(
        config.paths.model_input / file_paths.NATIONAL_CYCLE_NETWORK_MODEL_INPUT_PATH
    )

    if ncn.empty:
        LOG.warning("National Cycle Network layer is empty. Skipping.")
        return

    ncn_risk = _infrastructure_risk_intersect(ncn, hazard_layers)

    _audit_infrastructure_risk(
        ncn_risk,
        "National Cycle Network",
        risk_cols,
        audit_path / "Other" / "National Cycle Network",
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        ncn_risk,
        config.paths.model_output / "Other" / "National Cycle Network" / "ncn_risk.gpkg",
    )

    ncn_risk = _prepare_model_output(
        risk_data=ncn_risk,
        drop_cols=[],
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
        config,
        ncn_risk,
        "id",
        pathlib.Path("Other") / "National Cycle Network" / "ncn_risk",
    )
    LOG.info("Finished layering National Cycle Network with hazard risk.")


#### Tram Network


def _tram_network_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get tram network risk and write to file.

    Intersect tram network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering tram network with hazard risk...")
    tram_network = gpd.read_file(
        config.paths.model_input / file_paths.TRAM_NETWORK_MODEL_INPUT_PATH
    )

    if tram_network.empty:
        LOG.warning("Tram network layer is empty. Skipping.")
        return

    tram_risk = _infrastructure_risk_intersect(tram_network, hazard_layers)

    tram_risk = _apply_asset_vulnerability(
        tram_risk,
        structure_enum=OSRailStructure,
        structure_col=OSRailCols.STRUCTURE,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    _audit_infrastructure_risk(
        tram_risk,
        "Tram Network",
        risk_cols,
        audit_path / "Other" / "Tram Network",
        linewidth=1.0,
        feature_range=(config.constants.score_min, config.constants.score_max),
    )

    data_cleaning.write_to_file(
        tram_risk,
        config.paths.model_output / "Other" / "Tram Network" / "tram_network_risk.gpkg",
    )

    tram_risk = _prepare_model_output(
        risk_data=tram_risk,
        drop_cols=[],
        rename_map={
            OSRailCols.ID: "id",
            OSRailCols.PHYSICAL_LEVEL: "physical_level",
            OSRailCols.RAILWAY_USE: "railway_use",
            OSRailCols.TRACK_REPRESENTATION: "track_representation",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        tram_risk,
        "id",
        pathlib.Path("Other") / "Tram Network" / "tram_network_risk",
    )
    LOG.info("Finished layering tram network with hazard risk.")


#### Rapid Transport Network


def _rapid_transport_network_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get rapid transport network risk and write to file.

    Intersect rapid transport network with hazard risk, clean output, then write to file.
    """
    LOG.info("Layering rapid transport network with hazard risk...")
    rapid_transport = gpd.read_file(
        config.paths.model_input / file_paths.RAPID_TRANSPORT_NETWORK_MODEL_INPUT_PATH
    )

    if rapid_transport.empty:
        LOG.warning("Rapid transport network layer is empty. Skipping.")
        return

    rapid_transport_risk = _infrastructure_risk_intersect(rapid_transport, hazard_layers)

    feature_range = (config.constants.score_min, config.constants.score_max)

    rapid_transport_risk = _apply_asset_vulnerability(
        rapid_transport_risk,
        structure_enum=OSRailStructure,
        structure_col=OSRailCols.STRUCTURE,
        feature_range=feature_range,
    )

    _audit_infrastructure_risk(
        rapid_transport_risk,
        "Rapid Transport Network",
        risk_cols,
        audit_path / "Other" / "Rapid Transport Network",
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        rapid_transport_risk,
        config.paths.model_output
        / "Other"
        / "Rapid Transport Network"
        / "rapid_transport_network_risk.gpkg",
    )

    rapid_transport_risk = _prepare_model_output(
        risk_data=rapid_transport_risk,
        drop_cols=[],
        rename_map={
            OSRailCols.ID: "id",
            OSRailCols.PHYSICAL_LEVEL: "physical_level",
            OSRailCols.RAILWAY_USE: "railway_use",
            OSRailCols.TRACK_REPRESENTATION: "track_representation",
        },
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        rapid_transport_risk,
        "id",
        pathlib.Path("Other") / "Rapid Transport Network" / "rapid_transport_network_risk",
    )
    LOG.info("Finished layering rapid transport network with hazard risk.")


### BESPOKE


def _get_bespoke_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Get bespoke infrastructure risk and write to file."""
    any_bespoke = any(
        [
            config.switches.bespoke,
        ]
    )
    if any_bespoke:
        LOG.info("Calculating bespoke infrastructure risk...")
        _nexus_metro_links_risk(config, hazard_layers, risk_cols, audit_path)
        _nexus_metro_stations_risk(config, hazard_layers, risk_cols, audit_path)
        LOG.info("Risk calculation for bespoke infrastructure completed.")


def _nexus_metro_links_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Calculate Nexus Metro bespoke infrastructure risk and write to file."""
    LOG.info("Calculating Nexus Metro bespoke infrastructure risk...")
    metro_link_flows = gpd.read_file(
        config.paths.model_input / file_paths.NEXUS_METRO_LINK_FLOWS_MODEL_INPUT_PATH
    )

    if metro_link_flows.empty:
        LOG.warning("Nexus Metro links layer is empty. Skipping.")
        return

    metro_links_risk = _infrastructure_risk_intersect(metro_link_flows, hazard_layers)

    feature_range = (config.constants.score_min, config.constants.score_max)

    metro_links_risk = _metro_impact_index(metro_links_risk, feature_range)

    _audit_infrastructure_risk(
        metro_links_risk,
        "Nexus Metro Links",
        risk_cols,
        audit_path / "Other" / "Nexus Metro Links",
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        metro_links_risk,
        config.paths.model_output
        / "Other"
        / "Nexus Metro Links"
        / "nexus_metro_links_risk.gpkg",
    )

    metro_links_risk = _prepare_model_output(
        risk_data=metro_links_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        metro_links_risk,
        "id",
        pathlib.Path("Other") / "Nexus Metro Links" / "nexus_metro_links_risk",
    )
    LOG.info("Finished calculating Nexus Metro bespoke infrastructure risk.")


def _metro_impact_index(
    metro_risk: gpd.GeoDataFrame, feature_range: tuple[int, int]
) -> gpd.GeoDataFrame:
    """Normalise metro link demand, then calculate impact index."""
    # First, normalise demand
    metro_risk = functional_rules.min_max_scaling_pair(
        data=metro_risk,
        pairs=[(f"demand_{Scenarios.CURRENT}", f"demand_{Scenarios.FORECAST}")],
        feature_range=feature_range,
    )

    # Then, calculate impact index for total demand
    metro_risk = _calculate_metro_impact(metro_risk)

    # Finally, normalise the impact index for total demand
    return functional_rules.min_max_scaling_pair(
        data=metro_risk,
        pairs=[(f"impact_{Scenarios.CURRENT}", f"impact_{Scenarios.FORECAST}")],
        feature_range=feature_range,
    )


def _calculate_metro_impact(metro_risk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate metro impact score for total demand."""
    risk_cols = [
        col for col in MainHazardRiskCols if f"{col}_{Scenarios.CURRENT}" in metro_risk.columns
    ]

    hazards = [col.removesuffix("_risk") for col in risk_cols]
    impact_weights = _get_impact_weights(hazards)

    for scenario in Scenarios:
        hazard_component = sum(
            metro_risk[f"{risk_col}_{scenario}"]
            * impact_weights[risk_col.removesuffix("_risk")]
            for risk_col in risk_cols
        )

        impact_component = metro_risk[f"demand_{scenario}"] * impact_weights["demand"]
        metro_risk[f"impact_{scenario}"] = impact_component + hazard_component

    return metro_risk


def _nexus_metro_stations_risk(
    config: model_config.Config,
    hazard_layers: dict[MainHazardRiskCols, gpd.GeoDataFrame],
    risk_cols: list[RiskColumn],
    audit_path: pathlib.Path,
) -> None:
    """Calculate Nexus Metro stations bespoke infrastructure risk and write to file."""
    LOG.info("Calculating Nexus Metro stations bespoke infrastructure risk...")
    metro_stations = gpd.read_file(
        config.paths.model_input / file_paths.NEXUS_METRO_STATIONS_MODEL_INPUT_PATH
    )

    if metro_stations.empty:
        LOG.warning("Nexus Metro stations layer is empty. Skipping.")
        return

    # TODO (DJ): Consider buffering metro stations to account for surrounding area risk

    metro_stations_risk = _infrastructure_risk_intersect(metro_stations, hazard_layers)

    feature_range = (config.constants.score_min, config.constants.score_max)

    _audit_infrastructure_risk(
        metro_stations_risk,
        "Nexus Metro Stations",
        risk_cols,
        audit_path / "Other" / "Nexus Metro Stations",
        feature_range=feature_range,
    )

    data_cleaning.write_to_file(
        metro_stations_risk,
        config.paths.model_output
        / "Other"
        / "Nexus Metro Stations"
        / "nexus_metro_stations_risk.gpkg",
    )

    metro_stations_risk = _prepare_model_output(
        risk_data=metro_stations_risk,
        drop_cols=[],
        rename_map={},
        risk_cols_order=risk_cols,
    )

    _split_csv_shapefile(
        config,
        metro_stations_risk,
        "id",
        pathlib.Path("Other") / "Nexus Metro Stations" / "nexus_metro_stations_risk",
    )
    LOG.info("Finished calculating Nexus Metro stations bespoke infrastructure risk.")
