#!/usr/bin/env python3
"""Generate a cleaner Nature/BioRender-style >=1,000 publisher stacked bar, 2005-2026.

User-requested refinements:
- show 2005-2026, but label x-axis only every five years (2005, 2010, 2015, 2020, 2025);
- do not label 2026 on the x-axis;
- remove in-panel event/partial-year annotations;
- use Times New Roman;
- use a more Nature/BioRender-like muted palette.

Interpretation guardrail: this is a descriptive Retraction Watch publisher-field
label grouping, not a responsibility share, fault allocation, causality estimate,
or normalized retraction rate.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "source-data"
OUT_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "simplified-stacked" / "nature-biorender-ge1000-2005"
SOURCE_DIR = OUT_DIR / "source-data"
CUTOFF = "2026-05-06"
YEAR_START = 2005
YEAR_END = 2026
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

# Muted Nature/BioRender-like palette: lower saturation than Tableau, clear on white.
COLORS = {
    "Hindawi": "#2F5F8F",  # muted deep blue
    "IEEE": "#E5A64A",  # warm amber
    "Elsevier / Cell Press": "#CF6A5B",  # muted coral
    "Springer / BMC / Cureus (non-Nature label)": "#4E9D8C",  # soft teal green
    "Wiley (excluding Hindawi)": "#7FB3D5",  # soft sky blue
    "Nature Publishing Group / Nature Portfolio": "#D8C45F",  # muted yellow
    "SAGE / IOS Press": "#A58BC4",  # lavender
    "Taylor & Francis / Dove": "#C97997",  # dusty rose
    "PLOS": "#7BAF7A",  # sage green
    "IOP Publishing": "#8FA2AF",  # blue-grey
    "Other publishers": "#D9DEE7",  # light cool grey
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
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.7,
        "axes.titlesize": 9.8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.minor.width": 0.35,
        "grid.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def build_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Threshold is retained from the user-selected 2000-2026 >=1,000 version.
    totals_2000_2026 = (
        df.groupby("publisher_family", as_index=False)["n_retractions"]
        .sum()
        .rename(columns={"n_retractions": "publisher_family_total_2000_2026"})
    )
    named_totals = totals_2000_2026[totals_2000_2026["publisher_family"].ne("Other publishers")].copy()
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
    grid = pd.MultiIndex.from_product([range(YEAR_START, YEAR_END + 1), order], names=["year", "publisher_family"]).to_frame(index=False)
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
    annual["publisher_family_total_2005_2026"] = annual.groupby("publisher_family")["n_retractions"].transform("sum").astype(int)
    annual = annual.merge(totals_2000_2026, on="publisher_family", how="left")
    # For Other publishers, recompute the retained simplified total over 2000-2026.
    other_total_2000_2026 = int(candidate.loc[~candidate["publisher_family"].isin(keep), "n_retractions"].sum())
    annual.loc[annual["publisher_family"].eq("Other publishers"), "publisher_family_total_2000_2026"] = other_total_2000_2026
    annual["publisher_family_total_2000_2026"] = annual["publisher_family_total_2000_2026"].astype(int)
    annual["display_order"] = annual["publisher_family"].map({name: i + 1 for i, name in enumerate(order)})
    annual["candidate_rule"] = RULE_LABEL
    annual["data_cutoff_latest_retraction_date_in_csv"] = CUTOFF
    annual["scope"] = f"RetractionNature == Retraction; plotted years {YEAR_START}–{YEAR_END}; 2026 partial through latest CSV RetractionDate {CUTOFF}."
    annual["interpretation_limit"] = INTERPRETATION_LIMIT

    map_table = candidate[["publisher_family", "publisher_family_simplified"]].drop_duplicates().copy()
    map_table["publisher_family_simplified_display_label"] = map_table["publisher_family_simplified"].map(DISPLAY_LABEL)
    map_table["candidate_rule"] = RULE_LABEL
    map_table["interpretation_limit"] = INTERPRETATION_LIMIT
    map_table = map_table.sort_values(["publisher_family_simplified", "publisher_family"])
    return annual, map_table


def draw_plot(annual: pd.DataFrame) -> None:
    pivot = annual.pivot(index="year", columns="publisher_family", values="n_retractions").reindex(columns=FULL_ORDER).fillna(0)
    years = pivot.index.to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(7.25, 4.15))
    bottom = np.zeros(len(pivot))
    for family in FULL_ORDER:
        vals = pivot[family].to_numpy(dtype=float)
        bars = ax.bar(
            years,
            vals,
            bottom=bottom,
            width=0.76,
            color=COLORS[family],
            label=DISPLAY_LABEL[family],
            linewidth=0.22,
            edgecolor="white",
        )
        for bar, year in zip(bars, years):
            if year == 2026 and bar.get_height() > 0:
                # Keep partial-year cue without an x-axis 2026 label or in-panel text annotation.
                bar.set_hatch("//")
                bar.set_edgecolor("#7A7A7A")
                bar.set_linewidth(0.24)
        bottom += vals

    ax.set_title("Annual retraction records by publisher group", loc="left", pad=6, weight="bold")
    ax.set_ylabel("Retraction records")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(2004.45, 2026.75)
    ax.set_ylim(0, 14000)
    ax.set_xticks([2005, 2010, 2015, 2020, 2025])
    ax.set_xticklabels(["2005", "2010", "2015", "2020", "2025"])
    ax.set_xticks(range(YEAR_START, YEAR_END + 1), minor=True)
    ax.tick_params(axis="x", which="minor", length=1.7, color="#777777")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#E8E8E8")
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        ncol=1,
        handlelength=1.05,
        handletextpad=0.42,
        borderaxespad=0,
    )

    note = (
        f"2026 is partial through RetractionDate {CUTOFF}; hatching marks partial-year data. "
        f"{RULE_LABEL} Publisher groups are Retraction Watch publisher-field label groupings, not responsibility shares."
    )
    fig.text(0.075, 0.018, textwrap.fill(note, 104), ha="left", va="bottom", fontsize=6.65, color="#404040")
    fig.subplots_adjust(left=0.075, right=0.735, top=0.90, bottom=0.22)

    stem = f"publisher_annual_stacked_threshold_ge1000_nature_biorender_2005_2026_asof_{CUTOFF}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = pd.read_csv(INFILE)
    annual, mapping = build_candidate(df)

    stem = f"publisher_annual_stacked_threshold_ge1000_nature_biorender_2005_2026_asof_{CUTOFF}"
    annual.to_csv(SOURCE_DIR / f"{stem}.source.csv", index=False)
    mapping.to_csv(SOURCE_DIR / f"{stem}.mapping.csv", index=False)

    total = int(annual.groupby("year")["annual_total_retractions"].first().sum())
    partial_2026 = int(annual.loc[annual["year"].eq(2026), "annual_total_retractions"].iloc[0])
    other_total = int(annual.loc[annual["publisher_family"].eq("Other publishers"), "n_retractions"].sum())
    summary = pd.DataFrame([
        {
            "candidate": "threshold_ge1000_nature_biorender_2005_2026",
            "plotted_year_start": YEAR_START,
            "plotted_year_end": YEAR_END,
            "x_axis_major_tick_labels": "2005; 2010; 2015; 2020; 2025",
            "x_axis_2026_label_shown": False,
            "in_panel_event_annotations": False,
            "font_family": "Times New Roman",
            "legend_entries_including_other": len(FULL_ORDER),
            "named_families": len(FULL_ORDER) - 1,
            "other_publishers_2005_2026": other_total,
            "other_percent_2005_2026": round(other_total / total * 100, 4),
            "total_retractions_2005_2026": total,
            "partial_2026_retractions": partial_2026,
            "cutoff_latest_retraction_date_in_csv": CUTOFF,
            "retention_rule": RULE_LABEL,
            "interpretation_limit": INTERPRETATION_LIMIT,
        }
    ])
    summary.to_csv(SOURCE_DIR / f"{stem}.summary.csv", index=False)
    draw_plot(annual)

    print(f"wrote {OUT_DIR}")
    print(f"plotted_years={YEAR_START}-{YEAR_END}")
    print(f"total_2005_2026={total}")
    print(f"partial_2026={partial_2026}")
    print(f"other_publishers_2005_2026={other_total} ({other_total/total*100:.2f}%)")
    print(f"source_rows={len(annual)} expected={(YEAR_END-YEAR_START+1)*len(FULL_ORDER)}")


if __name__ == "__main__":
    main()
