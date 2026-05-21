#!/usr/bin/env python3
"""Generate simplified publisher-family stacked-bar candidates.

Input is the final v0.3 publisher-family annual source table. Smaller named
families are recombined into "Other publishers" under explicit threshold rules.
These are descriptive Retraction Watch publisher-field groupings, not
responsibility shares or normalized retraction rates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "source-data"
OUT_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "simplified-stacked"
SOURCE_DIR = OUT_DIR / "source-data"
CUTOFF = "2026-05-06"
INFILE = SRC_DIR / f"publisher_family_annual_counts_final_long_2000_2026_asof_{CUTOFF}.source.csv"

BASE_ORDER = [
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
    "Wolters Kluwer",
    "Oxford University Press",
    "Royal Society of Chemistry",
    "University of El Oued",
    "AIP Publishing",
    "ACM",
    "ACS",
    "ASBMB",
    "Other publishers",
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
    "Wolters Kluwer": "#C5B0D5",
    "Oxford University Press": "#8C564B",
    "Royal Society of Chemistry": "#17BECF",
    "University of El Oued": "#BCBD22",
    "AIP Publishing": "#7F7F7F",
    "ACM": "#AEC7E8",
    "ACS": "#FFBB78",
    "ASBMB": "#98DF8A",
    "Other publishers": "#D9D9D9",
}

CONFIGS = [
    {
        "slug": "major_ge2000",
        "title": "Annual retraction records by major publisher-family labels",
        "rule_label": "Named families retained when 2000-2026 total >= 2,000; all smaller labels combined into Other publishers.",
        "threshold": 2000,
        "force_keep": [],
    },
    {
        "slug": "threshold_ge1000",
        "title": "Annual retraction records by publisher-family labels (>=1,000 records)",
        "rule_label": "Named families retained when 2000-2026 total >= 1,000; all smaller labels combined into Other publishers.",
        "threshold": 1000,
        "force_keep": [],
    },
    {
        "slug": "threshold_ge1000_plus_frontiers_mdpi",
        "title": "Annual retraction records by selected publisher-family labels",
        "rule_label": "Named families retained when 2000-2026 total >= 1,000, with Frontiers and MDPI retained a priori; all other labels combined into Other publishers.",
        "threshold": 1000,
        "force_keep": ["Frontiers", "MDPI"],
    },
]


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
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def build_candidate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    totals = (
        df.groupby("publisher_family", as_index=False)["n_retractions"]
        .sum()
        .rename(columns={"n_retractions": "publisher_family_total_2000_2026"})
    )
    totals = totals[totals["publisher_family"].ne("Other publishers")].copy()
    force = set(cfg["force_keep"])
    keep = totals[
        (totals["publisher_family_total_2000_2026"].ge(cfg["threshold"]))
        | (totals["publisher_family"].isin(force))
    ]["publisher_family"].tolist()
    keep = [name for name in BASE_ORDER if name in keep]
    order = keep + ["Other publishers"]

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
    annual["is_partial_year"] = annual["year"].eq(2026)
    annual["annual_total_retractions"] = annual.groupby("year")["n_retractions"].transform("sum").astype(int)
    annual["percent_of_annual_retractions"] = np.where(
        annual["annual_total_retractions"].gt(0),
        (annual["n_retractions"] / annual["annual_total_retractions"] * 100).round(4),
        0.0,
    )
    annual["publisher_family_total_2000_2026"] = annual.groupby("publisher_family")["n_retractions"].transform("sum").astype(int)
    annual["display_order"] = annual["publisher_family"].map({name: i + 1 for i, name in enumerate(order)})
    annual["candidate_rule"] = cfg["rule_label"]
    annual["data_cutoff_latest_retraction_date_in_csv"] = CUTOFF
    annual["scope"] = f"RetractionNature == Retraction; years 2000-2026; 2026 partial through latest CSV RetractionDate {CUTOFF}."
    annual["interpretation_limit"] = "Descriptive Retraction Watch publisher-field grouping; not responsibility, fault, causality, or normalized retraction rate."

    map_table = (
        candidate[["publisher_family", "publisher_family_simplified"]]
        .drop_duplicates()
        .sort_values(["publisher_family_simplified", "publisher_family"])
    )
    map_table["candidate_rule"] = cfg["rule_label"]
    return annual, order, map_table


def plot_candidate(annual: pd.DataFrame, order: list[str], cfg: dict) -> None:
    pivot = annual.pivot(index="year", columns="publisher_family", values="n_retractions").reindex(columns=order).fillna(0)
    fig, ax = plt.subplots(figsize=(8.6, 4.55))
    bottom = np.zeros(len(pivot))
    years = pivot.index.to_numpy(dtype=int)
    for family in order:
        vals = pivot[family].to_numpy(dtype=float)
        bars = ax.bar(years, vals, bottom=bottom, width=0.82, color=COLORS.get(family, "#999999"), label=family, linewidth=0)
        for bar, year in zip(bars, years):
            if year == 2026 and bar.get_height() > 0:
                bar.set_hatch("///")
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.25)
        bottom += vals
    ax.set_title(cfg["title"])
    ax.set_ylabel("Number of retraction records")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(1999.4, 2026.6)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2026])
    ax.grid(axis="y", color="#dddddd")
    note = f"2026 is partial through latest CSV RetractionDate {CUTOFF}. {cfg['rule_label']} Groups are not responsibility shares."
    ax.text(0.0, -0.20, note, transform=ax.transAxes, fontsize=7.2, color="#444444", wrap=True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    fig.subplots_adjust(right=0.70, bottom=0.27)
    stem = f"publisher_annual_stacked_{cfg['slug']}_2000_2026_asof_{CUTOFF}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = pd.read_csv(INFILE)
    summary_rows = []
    for cfg in CONFIGS:
        annual, order, map_table = build_candidate(df, cfg)
        stem = f"publisher_annual_stacked_{cfg['slug']}_2000_2026_asof_{CUTOFF}"
        annual.to_csv(SOURCE_DIR / f"{stem}.source.csv", index=False)
        map_table.to_csv(SOURCE_DIR / f"{stem}.mapping.csv", index=False)
        plot_candidate(annual, order, cfg)
        total = int(annual[annual["year"].eq(2000)]["annual_total_retractions"].iloc[0])  # placeholder checked below
        total_all = int(annual.groupby("year")["annual_total_retractions"].first().sum())
        other_total = int(annual[annual["publisher_family"].eq("Other publishers")]["n_retractions"].sum())
        summary_rows.append({
            "candidate": cfg["slug"],
            "legend_entries_including_other": len(order),
            "named_families": len(order) - 1,
            "other_publishers_2000_2026": other_total,
            "other_percent_2000_2026": round(other_total / total_all * 100, 4),
            "total_retractions_2000_2026": total_all,
            "rule": cfg["rule_label"],
        })
        print(cfg["slug"], "legend", len(order), "other", other_total, f"{other_total/total_all*100:.2f}%")
    pd.DataFrame(summary_rows).to_csv(SOURCE_DIR / f"publisher_simplified_stacked_candidate_summary_2000_2026_asof_{CUTOFF}.csv", index=False)
    print(f"Wrote simplified stacked candidates: {OUT_DIR}")


if __name__ == "__main__":
    main()
