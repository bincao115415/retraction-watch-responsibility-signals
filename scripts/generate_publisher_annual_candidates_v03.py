#!/usr/bin/env python3
"""Draft publisher-family annual retraction figures for RW v0.3 final table.

These figures are exploratory publisher-family views, not responsibility shares.
They use RetractionNature == Retraction, years 2000-2026, with 2026 explicitly
marked as partial through the latest RetractionDate in the CSV.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "data" / "final" / "record_level_binary.csv"
OUT_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates"
SOURCE_DIR = OUT_DIR / "source-data"

# Mutually exclusive display families. This preserves Nature Portfolio as a narrower
# label and does not merge Hindawi into Wiley, because the RW Publisher field records
# Hindawi separately and ownership recoding would require a distinct time-aware model.
DISPLAY_ORDER = [
    "Hindawi",
    "IEEE",
    "Elsevier / Cell Press",
    "Springer / BMC / Cureus (non-Nature label)",
    "Wiley (excluding Hindawi)",
    "Nature Publishing Group / Nature Portfolio",
    "SAGE / IOS Press",
    "Taylor & Francis / Dove",
    "Frontiers",
    "MDPI",
    "PLOS",
    "IOP Publishing",
    "Spandidos",
    "EDP Sciences",
    "Oxford University Press",
    "Royal Society of Chemistry",
    "University of El Oued",
    "AIP Publishing",
    "ACM",
    "ACS",
    "ASBMB",
    "Wolters Kluwer",
    "Other publishers",
]

LINE_ORDER = [
    "Hindawi",
    "IEEE",
    "Elsevier / Cell Press",
    "Springer / BMC / Cureus (non-Nature label)",
    "Wiley (excluding Hindawi)",
    "Nature Publishing Group / Nature Portfolio",
    "SAGE / IOS Press",
    "Taylor & Francis / Dove",
    "PLOS",
    "IOP Publishing",
    "Frontiers",
    "MDPI",
    "Spandidos",
    "EDP Sciences",
    "Oxford University Press",
]

COLORS = {
    "Hindawi": "#4E79A7",
    "IEEE": "#F28E2B",
    "Elsevier / Cell Press": "#E15759",
    "Springer / BMC / Cureus (non-Nature label)": "#76B7B2",
    "Wiley (excluding Hindawi)": "#59A14F",
    "Nature Publishing Group / Nature Portfolio": "#EDC948",
    "SAGE / IOS Press": "#B07AA1",
    "Taylor & Francis / Dove": "#FF9DA7",
    "PLOS": "#9C755F",
    "IOP Publishing": "#BAB0AC",
    "Frontiers": "#1F77B4",
    "MDPI": "#2CA02C",
    "Spandidos": "#D62728",
    "EDP Sciences": "#9467BD",
    "Oxford University Press": "#8C564B",
    "Royal Society of Chemistry": "#17BECF",
    "University of El Oued": "#BCBD22",
    "AIP Publishing": "#7F7F7F",
    "ACM": "#AEC7E8",
    "ACS": "#FFBB78",
    "ASBMB": "#98DF8A",
    "Wolters Kluwer": "#C5B0D5",
    "Other publishers": "#D9D9D9",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def publisher_family(publisher: str, journal: str = "") -> str:
    p = str(publisher or "").lower()
    j = str(journal or "").lower()
    if "hindawi" in p:
        return "Hindawi"
    if "ieee" in p or "institute of electrical and electronics engineers" in p:
        return "IEEE"
    if "elsevier" in p or "cell press" in p:
        return "Elsevier / Cell Press"
    if "wiley" in p:
        return "Wiley (excluding Hindawi)"
    if "nature publishing group" in p:
        return "Nature Publishing Group / Nature Portfolio"
    if "springer" in p or "biomed central" in p or re.search(r"\bbmc\b", p) or "cureus" in p:
        return "Springer / BMC / Cureus (non-Nature label)"
    if "taylor and francis" in p or "taylor & francis" in p or "dove press" in p:
        return "Taylor & Francis / Dove"
    if "frontiers" in p:
        return "Frontiers"
    if "mdpi" in p:
        return "MDPI"
    if "sage" in p or "ios press" in p:
        return "SAGE / IOS Press"
    if "plos" in p:
        return "PLOS"
    if "iop publishing" in p:
        return "IOP Publishing"
    if "spandidos" in p:
        return "Spandidos"
    if "edp sciences" in p:
        return "EDP Sciences"
    if "oxford university press" in p:
        return "Oxford University Press"
    if "royal society of chemistry" in p or re.search(r"\brsc\b", p):
        return "Royal Society of Chemistry"
    if "university of el qued" in p or "university of el oued" in p:
        return "University of El Oued"
    if "aip publishing" in p:
        return "AIP Publishing"
    if "association for computing machinery" in p or re.search(r"\bacm\b", p):
        return "ACM"
    if "american chemical society" in p or re.search(r"\bacs\b", p):
        return "ACS"
    if "american society for biochemistry and molecular biology" in p or "asbmb" in p:
        return "ASBMB"
    if "wolters kluwer" in p or "lippincott" in p or "medknow" in p:
        return "Wolters Kluwer"
    return "Other publishers"


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, str]:
    cols = ["record_id", "journal", "publisher", "retraction_nature", "retraction_date", "retraction_year"]
    df = pd.read_csv(RECORD_PATH, usecols=cols, low_memory=False)
    df["retraction_year"] = pd.to_numeric(df["retraction_year"], errors="coerce").astype("Int64")
    df["retraction_dt"] = pd.to_datetime(df["retraction_date"], errors="coerce")
    cutoff = df["retraction_dt"].max().date().isoformat()
    ret = df[(df["retraction_nature"] == "Retraction") & (df["retraction_year"].between(2000, 2026))].copy()
    ret["publisher_family"] = [publisher_family(p, j) for p, j in zip(ret["publisher"], ret["journal"])]
    return ret, cutoff


def write_source_tables(ret: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    scope = f"RetractionNature == Retraction; years 2000-2026; 2026 partial through latest CSV RetractionDate {cutoff}."
    note = "Rule-based grouping from RW Publisher field; mutually exclusive display families; not responsibility shares."

    exact = ret.groupby("publisher", dropna=False).size().rename("n_retractions_2000_2026").reset_index()
    exact = exact.sort_values("n_retractions_2000_2026", ascending=False)
    exact["assigned_publisher_family"] = [publisher_family(p) for p in exact["publisher"]]
    exact["data_cutoff_latest_retraction_date_in_csv"] = cutoff
    exact.to_csv(SOURCE_DIR / f"exact_publisher_labels_2000_2026_asof_{cutoff}.source.csv", index=False)

    annual = ret.groupby(["retraction_year", "publisher_family"], observed=True).size().rename("n_retractions").reset_index()
    annual = annual.rename(columns={"retraction_year": "year"})
    grid = pd.MultiIndex.from_product([range(2000, 2027), DISPLAY_ORDER], names=["year", "publisher_family"]).to_frame(index=False)
    annual = grid.merge(annual, on=["year", "publisher_family"], how="left").fillna({"n_retractions": 0})
    annual["n_retractions"] = annual["n_retractions"].astype(int)
    annual["data_cutoff_latest_retraction_date_in_csv"] = cutoff
    annual["scope"] = scope
    annual["grouping_note"] = note
    annual.to_csv(SOURCE_DIR / f"annual_retractions_by_publisher_family_v0.2_2000_2026_asof_{cutoff}.source.csv", index=False)

    # Clean final statistical table used directly by both the stacked bar and
    # the line/small-multiple figures. This is long/tidy source data: one row
    # per year x publisher-family, with explicit display order and annual totals.
    annual_final = annual.copy()
    annual_totals = annual_final.groupby("year")["n_retractions"].transform("sum")
    family_totals = annual_final.groupby("publisher_family")["n_retractions"].transform("sum")
    annual_final.insert(0, "figure_source_table", "publisher_family_annual_counts_final_long")
    annual_final.insert(2, "publisher_family_display_order", annual_final["publisher_family"].map({name: i + 1 for i, name in enumerate(DISPLAY_ORDER)}))
    annual_final["annual_total_retractions"] = annual_totals.astype(int)
    annual_final["percent_of_annual_retractions"] = np.where(
        annual_totals > 0,
        (annual_final["n_retractions"] / annual_totals * 100).round(4),
        0.0,
    )
    annual_final["publisher_family_total_2000_2026"] = family_totals.astype(int)
    annual_final["included_in_stacked_bar"] = True
    annual_final["included_in_line_small_multiples"] = annual_final["publisher_family"].isin(LINE_ORDER)
    annual_final["is_partial_year"] = annual_final["year"].eq(2026)
    annual_final["grouping_basis"] = "Rule-based grouping from the Retraction Watch Publisher field as recorded; not current-owner recoding."
    annual_final["interpretation_limit"] = "Counts are descriptive record-label counts, not publisher responsibility, fault, causality, or normalized retraction rates."
    final_cols = [
        "figure_source_table",
        "year",
        "is_partial_year",
        "publisher_family_display_order",
        "publisher_family",
        "n_retractions",
        "annual_total_retractions",
        "percent_of_annual_retractions",
        "publisher_family_total_2000_2026",
        "included_in_stacked_bar",
        "included_in_line_small_multiples",
        "data_cutoff_latest_retraction_date_in_csv",
        "scope",
        "grouping_basis",
        "interpretation_limit",
    ]
    annual_final = annual_final[final_cols]
    annual_final.to_csv(SOURCE_DIR / f"publisher_family_annual_counts_final_long_2000_2026_asof_{cutoff}.source.csv", index=False)

    # Wide/matrix version for direct plotting: one row per year and one column
    # per publisher family. This is convenient for stacked bars and line plots.
    annual_wide = annual.pivot(index="year", columns="publisher_family", values="n_retractions").reindex(columns=DISPLAY_ORDER).fillna(0).astype(int).reset_index()
    annual_wide.insert(1, "is_partial_year", annual_wide["year"].eq(2026))
    annual_wide.insert(2, "annual_total_retractions", annual_wide[DISPLAY_ORDER].sum(axis=1).astype(int))
    annual_wide.insert(3, "data_cutoff_latest_retraction_date_in_csv", cutoff)
    annual_wide.insert(4, "scope", scope)
    annual_wide.insert(5, "grouping_basis", "Rule-based grouping from the Retraction Watch Publisher field as recorded; not current-owner recoding.")
    annual_wide.insert(6, "interpretation_limit", "Counts are descriptive record-label counts, not publisher responsibility, fault, causality, or normalized retraction rates.")
    annual_wide.to_csv(SOURCE_DIR / f"publisher_family_annual_counts_final_wide_2000_2026_asof_{cutoff}.source.csv", index=False)

    totals = ret.groupby("publisher_family").size().rename("n_retractions_2000_2026").reindex(DISPLAY_ORDER).fillna(0).astype(int).reset_index()
    y2026 = ret[ret["retraction_year"].eq(2026)].groupby("publisher_family").size().rename("n_retractions_2026_partial")
    totals = totals.merge(y2026, on="publisher_family", how="left").fillna({"n_retractions_2026_partial": 0})
    totals["n_retractions_2026_partial"] = totals["n_retractions_2026_partial"].astype(int)
    totals["data_cutoff_latest_retraction_date_in_csv"] = cutoff
    totals["scope"] = scope
    totals["grouping_note"] = note
    totals.to_csv(SOURCE_DIR / f"publisher_family_totals_v0.2_2000_2026_asof_{cutoff}.source.csv", index=False)

    line_annual = annual[annual["publisher_family"].isin(LINE_ORDER)].copy()
    line_annual.to_csv(SOURCE_DIR / f"annual_retractions_selected_line_families_2000_2026_asof_{cutoff}.source.csv", index=False)
    return annual, totals, line_annual


def plot_stacked(annual: pd.DataFrame, cutoff: str) -> None:
    pivot = annual.pivot(index="year", columns="publisher_family", values="n_retractions").reindex(columns=DISPLAY_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bottom = np.zeros(len(pivot))
    years = pivot.index.to_numpy(dtype=int)
    for family in DISPLAY_ORDER:
        vals = pivot[family].to_numpy(dtype=float)
        bars = ax.bar(years, vals, bottom=bottom, width=0.82, color=COLORS.get(family, "#999999"), label=family, linewidth=0)
        # hatch 2026 partial-year segments
        for bar, year in zip(bars, years):
            if year == 2026 and bar.get_height() > 0:
                bar.set_hatch("///")
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.25)
        bottom += vals
    ax.set_title("Annual retraction records by selected publisher family")
    ax.set_ylabel("Number of retraction records")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(1999.4, 2026.6)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2026])
    ax.grid(axis="y", color="#dddddd")
    ax.text(0.0, -0.18, f"2026 is partial through latest CSV RetractionDate {cutoff}. Publisher-family groups are record-label groupings, not responsibility shares.", transform=ax.transAxes, fontsize=7.5, color="#444444")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    fig.subplots_adjust(right=0.68, bottom=0.25)
    save_all(fig, f"publisher_annual_stacked_family_v0.2_2000_2026_asof_{cutoff}")


def plot_line_small_multiples(line_annual: pd.DataFrame, cutoff: str) -> None:
    n = len(LINE_ORDER)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.8, 8.8), sharex=True)
    axes = axes.ravel()
    for ax, family in zip(axes, LINE_ORDER):
        tmp = line_annual[line_annual["publisher_family"].eq(family)].sort_values("year")
        years = tmp["year"].to_numpy(dtype=int)
        vals = tmp["n_retractions"].to_numpy(dtype=float)
        ax.plot(years[years <= 2025], vals[years <= 2025], color=COLORS.get(family, "#333333"), lw=1.4)
        if np.any(years == 2026):
            y2025 = vals[years == 2025][0] if np.any(years == 2025) else np.nan
            y2026 = vals[years == 2026][0]
            if not np.isnan(y2025):
                ax.plot([2025, 2026], [y2025, y2026], color=COLORS.get(family, "#333333"), lw=1.2, ls="--")
            ax.scatter([2026], [y2026], facecolors="white", edgecolors=COLORS.get(family, "#333333"), s=18, zorder=3)
        ax.set_title(family, fontsize=8.2)
        ax.grid(axis="y", color="#e0e0e0")
        ax.set_xlim(1999.5, 2026.5)
        ax.set_xticks([2000, 2010, 2020, 2026])
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Annual retraction records for selected publisher families", y=0.995, fontsize=11)
    fig.text(0.005, 0.5, "Number of retraction records", rotation=90, va="center", fontsize=8.5)
    fig.text(0.5, 0.02, f"Dashed final segment and open marker indicate partial 2026 data through latest CSV RetractionDate {cutoff}. Publisher-family groups are record-label groupings, not responsibility shares.", ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.03, 0.05, 1, 0.97))
    save_all(fig, f"publisher_annual_line_small_multiples_v0.2_2000_2026_asof_{cutoff}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    ret, cutoff = load_data()
    annual, totals, line_annual = write_source_tables(ret, cutoff)
    plot_stacked(annual, cutoff)
    plot_line_small_multiples(line_annual, cutoff)
    print(f"Retractions 2000-2026: {len(ret):,}")
    print(f"Latest CSV RetractionDate: {cutoff}")
    print(f"Wrote figures: {OUT_DIR}")
    print(f"Wrote source data: {SOURCE_DIR}")
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
