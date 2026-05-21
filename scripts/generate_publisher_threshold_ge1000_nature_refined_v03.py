#!/usr/bin/env python3
"""Generate Nature-style refined publisher-family stacked-bar candidate (>=1,000 rule).

This script refines the threshold_ge1000 stacked-bar candidate after visual QC:
- shorter title and labels;
- typographic >= -> ≥;
- compact display labels in the figure, while preserving full labels in source data;
- explicit 2026* tick and partial-year annotation;
- lighter, print-friendly style and colorblind-aware palette;
- direct descriptive annotations for the two dominant event clusters.

The figure is descriptive Retraction Watch publisher-field grouping only. It is not
publisher responsibility, fault, causality, culpability, or a normalized rate.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "source-data"
OUT_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "simplified-stacked" / "nature-refined-ge1000"
SOURCE_DIR = OUT_DIR / "source-data"
CUTOFF = "2026-05-06"
INFILE = SRC_DIR / f"publisher_family_annual_counts_final_long_2000_2026_asof_{CUTOFF}.source.csv"

FULL_ORDER = [
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
    "Other publishers",
]

DISPLAY_LABEL = {
    "Hindawi": "Hindawi",
    "IEEE": "IEEE",
    "Elsevier / Cell Press": "Elsevier/Cell Press",
    "Springer / BMC / Cureus (non-Nature label)": "Springer/BMC/Cureus",
    "Wiley (excluding Hindawi)": "Wiley excl. Hindawi",
    "Nature Publishing Group / Nature Portfolio": "Nature Portfolio",
    "SAGE / IOS Press": "SAGE/IOS Press",
    "Taylor & Francis / Dove": "T&F/Dove",
    "PLOS": "PLOS",
    "IOP Publishing": "IOP Publishing",
    "Other publishers": "Other",
}

# Colorblind-aware, print-friendly palette. Other is deliberately quiet.
COLORS = {
    "Hindawi": "#0072B2",  # blue
    "IEEE": "#E69F00",  # orange
    "Elsevier / Cell Press": "#D55E00",  # vermilion
    "Springer / BMC / Cureus (non-Nature label)": "#009E73",  # green
    "Wiley (excluding Hindawi)": "#56B4E9",  # sky blue
    "Nature Publishing Group / Nature Portfolio": "#F0E442",  # yellow
    "SAGE / IOS Press": "#CC79A7",  # purple
    "Taylor & Francis / Dove": "#882255",  # wine
    "PLOS": "#44AA99",  # teal
    "IOP Publishing": "#999933",  # olive
    "Other publishers": "#D9D9D9",  # light grey
}

RULE_LABEL = "Named publisher families have ≥1,000 records over 2000–2026; smaller labels are grouped as Other."
INTERPRETATION_LIMIT = (
    "Descriptive Retraction Watch publisher-field groupings; not responsibility shares, "
    "fault allocation, causality, culpability, or normalized retraction rates."
)


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 450,
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "grid.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def build_ge1000_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = (
        df.groupby("publisher_family", as_index=False)["n_retractions"]
        .sum()
        .rename(columns={"n_retractions": "publisher_family_total_2000_2026"})
    )
    named_totals = totals[totals["publisher_family"].ne("Other publishers")].copy()
    keep = named_totals[named_totals["publisher_family_total_2000_2026"].ge(1000)]["publisher_family"].tolist()
    keep = [name for name in FULL_ORDER if name in keep]
    order = keep + ["Other publishers"]
    if order != FULL_ORDER:
        raise RuntimeError(f"Unexpected retained order: {order}")

    candidate = df.copy()
    candidate["publisher_family_simplified"] = np.where(
        candidate["publisher_family"].isin(keep), candidate["publisher_family"], "Other publishers"
    )

    annual = (
        candidate.groupby(["year", "publisher_family_simplified"], as_index=False)["n_retractions"]
        .sum()
        .rename(columns={"publisher_family_simplified": "publisher_family"})
    )
    grid = pd.MultiIndex.from_product([range(2000, 2027), order], names=["year", "publisher_family"]).to_frame(index=False)
    annual = grid.merge(annual, on=["year", "publisher_family"], how="left").fillna({"n_retractions": 0})
    annual["n_retractions"] = annual["n_retractions"].astype(int)
    annual["publisher_display_label"] = annual["publisher_family"].map(DISPLAY_LABEL)
    annual["is_partial_year"] = annual["year"].eq(2026)
    annual["annual_total_retractions"] = annual.groupby("year")["n_retractions"].transform("sum").astype(int)
    annual["percent_of_annual_retractions"] = np.where(
        annual["annual_total_retractions"].gt(0),
        (annual["n_retractions"] / annual["annual_total_retractions"] * 100).round(4),
        0.0,
    )
    annual["publisher_family_total_2000_2026"] = annual.groupby("publisher_family")["n_retractions"].transform("sum").astype(int)
    annual["display_order"] = annual["publisher_family"].map({name: i + 1 for i, name in enumerate(order)})
    annual["candidate_rule"] = RULE_LABEL
    annual["data_cutoff_latest_retraction_date_in_csv"] = CUTOFF
    annual["scope"] = f"RetractionNature == Retraction; years 2000–2026; 2026 partial through latest CSV RetractionDate {CUTOFF}."
    annual["interpretation_limit"] = INTERPRETATION_LIMIT

    map_table = candidate[["publisher_family", "publisher_family_simplified"]].drop_duplicates().copy()
    map_table["publisher_family_simplified_display_label"] = map_table["publisher_family_simplified"].map(DISPLAY_LABEL)
    map_table["candidate_rule"] = RULE_LABEL
    map_table["interpretation_limit"] = INTERPRETATION_LIMIT
    map_table = map_table.sort_values(["publisher_family_simplified", "publisher_family"])
    return annual, map_table


def format_axes(ax: plt.Axes, y_max: int) -> None:
    ax.set_title("Annual retraction records by publisher group", loc="left", pad=6, weight="bold")
    ax.set_ylabel("Retraction records")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(1999.35, 2027.20)
    ax.set_ylim(0, y_max)
    # Omit 2025 as a major tick to prevent right-edge crowding; 2026 is marked directly as partial.
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2026])
    ax.set_xticklabels(["2000", "2005", "2010", "2015", "2020", "2026*"])
    ax.set_xticks(range(2000, 2027), minor=True)
    ax.tick_params(axis="x", which="minor", length=1.8, width=0.35, color="#777777")
    ax.grid(axis="y", color="#E5E5E5")
    ax.set_axisbelow(True)


def add_annotations(ax: plt.Axes, pivot: pd.DataFrame) -> None:
    # Descriptive event labels only; no causal or fault assignment.
    ieee_2010_total = float(pivot.loc[2010].sum())
    hindawi_2023 = float(pivot.loc[2023, "Hindawi"])
    total_2023 = float(pivot.loc[2023].sum())
    total_2026 = float(pivot.loc[2026].sum())

    ax.annotate(
        "IEEE 2010–2011\nconference-proceedings cluster",
        xy=(2010.55, ieee_2010_total * 0.90),
        xytext=(2005.6, 6200),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.55, shrinkA=0, shrinkB=2),
        fontsize=6.5,
        color="#333333",
        ha="left",
        va="center",
    )
    ax.annotate(
        "Hindawi 2023\nmass-retraction cluster",
        xy=(2023, hindawi_2023 * 0.55),
        xytext=(2017.6, 11650),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.55, shrinkA=0, shrinkB=2),
        fontsize=6.5,
        color="#333333",
        ha="left",
        va="center",
    )
    ax.annotate(
        "2026 partial",
        xy=(2026, total_2026),
        xytext=(2021.75, max(total_2026 + 1750, 2400)),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.55, shrinkA=0, shrinkB=2),
        fontsize=6.5,
        color="#333333",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
    )
    # Small total label for the high 2023 bar helps orient the scale without cluttering all bars.
    ax.text(2023, total_2023 + 220, f"{int(total_2023):,}", ha="center", va="bottom", fontsize=6.4, color="#333333")


def draw_bars(ax: plt.Axes, annual: pd.DataFrame) -> pd.DataFrame:
    pivot = annual.pivot(index="year", columns="publisher_family", values="n_retractions").reindex(columns=FULL_ORDER).fillna(0)
    bottom = np.zeros(len(pivot))
    years = pivot.index.to_numpy(dtype=int)
    for family in FULL_ORDER:
        vals = pivot[family].to_numpy(dtype=float)
        bars = ax.bar(
            years,
            vals,
            bottom=bottom,
            width=0.78,
            color=COLORS[family],
            label=DISPLAY_LABEL[family],
            linewidth=0.18,
            edgecolor="white",
        )
        for bar, year in zip(bars, years):
            if year == 2026 and bar.get_height() > 0:
                bar.set_hatch("//")
                bar.set_edgecolor("#555555")
                bar.set_linewidth(0.22)
        bottom += vals
    return pivot


def add_note(fig: plt.Figure, x: float, y: float, width_chars: int, *, include_rw_guardrail: bool = False) -> None:
    note = (
        f"*2026 partial, through RetractionDate {CUTOFF}. {RULE_LABEL} "
        "Hatching indicates partial-year data. "
        "Publisher groups are Retraction Watch publisher-field label groupings, not responsibility shares."
    )
    if include_rw_guardrail:
        note += " Retraction Watch reasons are non-exclusive metadata and do not assign legal, moral, causal, or fault responsibility."
    fig.text(x, y, textwrap.fill(note, width_chars), ha="left", va="bottom", fontsize=6.55, color="#444444")


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


def plot_bottom_legend(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 4.95))  # Nature two-column width candidate
    pivot = draw_bars(ax, annual)
    format_axes(ax, y_max=14000)
    add_annotations(ax, pivot)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.103),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.1,
        handletextpad=0.35,
        borderaxespad=0,
    )
    add_note(fig, 0.075, 0.012, 142)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.325)
    save_figure(fig, f"publisher_annual_stacked_threshold_ge1000_nature_refined_bottomlegend_2000_2026_asof_{CUTOFF}")


def plot_right_legend(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 4.20))
    pivot = draw_bars(ax, annual)
    format_axes(ax, y_max=14000)
    add_annotations(ax, pivot)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.38,
        borderaxespad=0,
    )
    add_note(fig, 0.075, 0.018, 108)
    fig.subplots_adjust(left=0.075, right=0.735, top=0.90, bottom=0.22)
    save_figure(fig, f"publisher_annual_stacked_threshold_ge1000_nature_refined_rightlegend_2000_2026_asof_{CUTOFF}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = pd.read_csv(INFILE)
    annual, mapping = build_ge1000_candidate(df)

    stem = f"publisher_annual_stacked_threshold_ge1000_nature_refined_2000_2026_asof_{CUTOFF}"
    annual.to_csv(SOURCE_DIR / f"{stem}.source.csv", index=False)
    mapping.to_csv(SOURCE_DIR / f"{stem}.mapping.csv", index=False)

    total = int(annual.groupby("year")["annual_total_retractions"].first().sum())
    partial_2026 = int(annual.loc[annual["year"].eq(2026), "annual_total_retractions"].iloc[0])
    other_total = int(annual.loc[annual["publisher_family"].eq("Other publishers"), "n_retractions"].sum())
    summary = pd.DataFrame([
        {
            "candidate": "threshold_ge1000_nature_refined",
            "legend_entries_including_other": len(FULL_ORDER),
            "named_families": len(FULL_ORDER) - 1,
            "other_publishers_2000_2026": other_total,
            "other_percent_2000_2026": round(other_total / total * 100, 4),
            "total_retractions_2000_2026": total,
            "partial_2026_retractions": partial_2026,
            "cutoff_latest_retraction_date_in_csv": CUTOFF,
            "rule": RULE_LABEL,
            "interpretation_limit": INTERPRETATION_LIMIT,
        }
    ])
    summary.to_csv(SOURCE_DIR / f"{stem}.summary.csv", index=False)

    plot_bottom_legend(annual)
    plot_right_legend(annual)

    # Verification printout for terminal logs.
    print(f"wrote {OUT_DIR}")
    print(f"total_2000_2026={total}")
    print(f"partial_2026={partial_2026}")
    print(f"other_publishers={other_total} ({other_total/total*100:.2f}%)")
    print(f"source_rows={len(annual)} expected={27 * len(FULL_ORDER)}")


if __name__ == "__main__":
    main()
