#!/usr/bin/env python3
"""Generate combined multi-line publisher-family annual candidates.

These figures use the final v0.3 publisher-family annual source table and
optionally recombine smaller labels into Other publishers. Counts are descriptive
Retraction Watch publisher-field groupings, not responsibility shares and not
normalized retraction rates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "source-data"
OUT_DIR = ROOT / "figures" / "v0.3-publisher-annual-candidates" / "combined-line"
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
    "Other publishers": "#8F8F8F",
}

CONFIGS = [
    {
        "slug": "major_ge2000",
        "title": "Annual retraction records by major publisher-family labels",
        "threshold": 2000,
        "force_keep": [],
        "rule_label": "Named families retained when 2000-2026 total >= 2,000; all smaller labels combined into Other publishers.",
    },
    {
        "slug": "threshold_ge1000_plus_frontiers_mdpi",
        "title": "Annual retraction records by selected publisher-family labels",
        "threshold": 1000,
        "force_keep": ["Frontiers", "MDPI"],
        "rule_label": "Named families retained when 2000-2026 total >= 1,000, with Frontiers and MDPI retained a priori; all other labels combined into Other publishers.",
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
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def simplify(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
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
    order = [x for x in BASE_ORDER if x in keep] + ["Other publishers"]

    tmp = df.copy()
    tmp["publisher_family_simplified"] = np.where(tmp["publisher_family"].isin(keep), tmp["publisher_family"], "Other publishers")
    annual = (
        tmp.groupby(["year", "publisher_family_simplified"], as_index=False)["n_retractions"]
        .sum()
        .rename(columns={"publisher_family_simplified": "publisher_family"})
    )
    grid = pd.MultiIndex.from_product([range(2000, 2027), order], names=["year", "publisher_family"]).to_frame(index=False)
    annual = grid.merge(annual, on=["year", "publisher_family"], how="left").fillna({"n_retractions": 0})
    annual["n_retractions"] = annual["n_retractions"].astype(int)
    annual["is_partial_year"] = annual["year"].eq(2026)
    annual["annual_total_retractions"] = annual.groupby("year")["n_retractions"].transform("sum").astype(int)
    annual["publisher_family_total_2000_2026"] = annual.groupby("publisher_family")["n_retractions"].transform("sum").astype(int)
    annual["candidate_rule"] = cfg["rule_label"]
    annual["data_cutoff_latest_retraction_date_in_csv"] = CUTOFF
    annual["scope"] = f"RetractionNature == Retraction; years 2000-2026; 2026 partial through latest CSV RetractionDate {CUTOFF}."
    annual["interpretation_limit"] = "Descriptive Retraction Watch publisher-field grouping; not responsibility, fault, causality, or normalized retraction rate."

    mapping = (
        tmp[["publisher_family", "publisher_family_simplified"]]
        .drop_duplicates()
        .sort_values(["publisher_family_simplified", "publisher_family"])
    )
    mapping["candidate_rule"] = cfg["rule_label"]
    return annual, order, mapping


def plot_multiline(annual: pd.DataFrame, order: list[str], cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for family in order:
        tmp = annual[annual["publisher_family"].eq(family)].sort_values("year")
        years = tmp["year"].to_numpy(dtype=int)
        vals = tmp["n_retractions"].to_numpy(dtype=float)
        color = COLORS.get(family, "#777777")
        lw = 1.7 if family != "Other publishers" else 2.0
        alpha = 0.92 if family != "Other publishers" else 0.80
        ax.plot(years[years <= 2025], vals[years <= 2025], color=color, lw=lw, alpha=alpha, label=family)
        if np.any(years == 2026):
            y2025 = vals[years == 2025][0] if np.any(years == 2025) else np.nan
            y2026 = vals[years == 2026][0]
            if not np.isnan(y2025):
                ax.plot([2025, 2026], [y2025, y2026], color=color, lw=lw, ls="--", alpha=alpha)
            ax.scatter([2026], [y2026], s=18, facecolors="white", edgecolors=color, linewidths=1.0, zorder=5)
    ax.set_title(cfg["title"])
    ax.set_ylabel("Number of retraction records")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(1999.4, 2026.7)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2026])
    ax.grid(axis="y", color="#dddddd")
    ax.text(
        0.0,
        -0.20,
        f"Dashed final segment and open marker indicate partial 2026 data through latest CSV RetractionDate {CUTOFF}. {cfg['rule_label']} Groups are not responsibility shares.",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#444444",
        wrap=True,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    fig.subplots_adjust(right=0.70, bottom=0.28)
    stem = f"publisher_annual_multiline_{cfg['slug']}_2000_2026_asof_{CUTOFF}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = pd.read_csv(INFILE)
    summary = []
    for cfg in CONFIGS:
        annual, order, mapping = simplify(df, cfg)
        stem = f"publisher_annual_multiline_{cfg['slug']}_2000_2026_asof_{CUTOFF}"
        annual.to_csv(SOURCE_DIR / f"{stem}.source.csv", index=False)
        mapping.to_csv(SOURCE_DIR / f"{stem}.mapping.csv", index=False)
        plot_multiline(annual, order, cfg)
        total_all = int(annual.groupby("year")["annual_total_retractions"].first().sum())
        other_total = int(annual[annual["publisher_family"].eq("Other publishers")]["n_retractions"].sum())
        summary.append({
            "candidate": cfg["slug"],
            "line_count_including_other": len(order),
            "named_families": len(order) - 1,
            "other_publishers_2000_2026": other_total,
            "other_percent_2000_2026": round(other_total / total_all * 100, 4),
            "total_retractions_2000_2026": total_all,
            "rule": cfg["rule_label"],
        })
        print(cfg["slug"], "lines", len(order), "other", other_total, f"{other_total/total_all*100:.2f}%")
    pd.DataFrame(summary).to_csv(SOURCE_DIR / f"publisher_multiline_candidate_summary_2000_2026_asof_{CUTOFF}.csv", index=False)
    print(f"Wrote combined line candidates: {OUT_DIR}")


if __name__ == "__main__":
    main()
