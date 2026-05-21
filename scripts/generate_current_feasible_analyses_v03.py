#!/usr/bin/env python3
"""Generate currently feasible analyses from RW final v0.3 data.

Scope: Retraction Watch records and final v0.3 processed fields only.
This script deliberately avoids responsibility/fault percentages and does not treat
title-pattern sister/mirror candidates as verified relationships.

Outputs:
- journal concentration by retraction-year and original-year cohorts;
- dynamic top-N journal rank context tables and top-10 rank-overlap figure;
- retraction-year x original-year heatmap and lag summaries;
- selected governance-signal annual prevalence trends;
- spike decomposition for prominent publisher-year clusters;
- exploratory, unverified sister/mirror/companion title-pattern leads.
"""
from __future__ import annotations

from pathlib import Path
import re
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = Path(
    "/Users/Shared/Hermes/workspaces/research/projects/publication/retractedpublications/"
    "data/rw-derived/2026-05-15-llm-semantic-v0.3-api/record_level_binary.csv"
)
OUT_DIR = ROOT / "figures" / "v0.3-current-feasible-analyses"
SOURCE_DIR = OUT_DIR / "source-data"
CUTOFF = "2026-05-06"

MAIN_START = 2000
MAIN_END = 2025

MAINCAT_COLS = [
    "maincat_research_content_reliability",
    "maincat_attribution_authorship_disclosure_integrity",
    "maincat_editorial_peer_review_governance",
    "maincat_post_publication_transparency_due_process_oversight",
]

GOV_SIGNAL_COLS = [
    "gov_notice_opacity",
    "gov_journal_publisher_investigation",
    "third_party_investigation",
    "gov_paper_mill",
    "gov_peer_review_concern",
    "gov_peer_review_compromised",
    "gov_ai_computer_content",
    "institutional_investigation",
    "gov_editorial_breach_strict",
]

SIGNAL_LABELS = {
    "gov_notice_opacity": "Notice opacity / limited information",
    "gov_journal_publisher_investigation": "Journal/publisher investigation",
    "third_party_investigation": "Third-party investigation",
    "gov_paper_mill": "Paper-mill signal",
    "gov_peer_review_concern": "Peer-review concern",
    "gov_peer_review_compromised": "Compromised peer review",
    "gov_ai_computer_content": "AI/computer-generated content",
    "institutional_investigation": "Institutional investigation",
    "gov_editorial_breach_strict": "Strict editorial breach",
}

MAINCAT_LABELS = {
    "maincat_research_content_reliability": "Research-content reliability",
    "maincat_attribution_authorship_disclosure_integrity": "Attribution/disclosure/ethics",
    "maincat_editorial_peer_review_governance": "Editorial/peer-review governance",
    "maincat_post_publication_transparency_due_process_oversight": "Post-publication oversight",
}

MAINCAT_ABBREVIATIONS = {
    "maincat_research_content_reliability": "R",
    "maincat_attribution_authorship_disclosure_integrity": "A",
    "maincat_editorial_peer_review_governance": "E",
    "maincat_post_publication_transparency_due_process_oversight": "P",
}

# Muted palette close to the current Nature/BioRender style used for manuscript-discussion figures.
COLORS = {
    "blue": "#2F5F8F",
    "amber": "#E5A64A",
    "coral": "#CF6A5B",
    "teal": "#4E9D8C",
    "sky": "#7FB3D5",
    "yellow": "#D8C45F",
    "lavender": "#A58BC4",
    "rose": "#C97997",
    "green": "#7BAF7A",
    "greyblue": "#8FA2AF",
    "lightgrey": "#D9DEE7",
    "darkgrey": "#4C4C4C",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 450,
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.6,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "grid.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def publisher_family(publisher: str, journal: str = "") -> str:
    p = str(publisher or "").lower()
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
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


def load_data() -> pd.DataFrame:
    cols = [
        "record_id", "title", "journal", "publisher", "article_type", "retraction_nature",
        "retraction_date", "retraction_year", "original_paper_date", "original_year",
        "raw_reason", "n_raw_reasons", "n_main_figure_categories",
        *MAINCAT_COLS, *GOV_SIGNAL_COLS,
    ]
    df = pd.read_csv(RECORD_PATH, usecols=cols, low_memory=False)
    df["retraction_year"] = pd.to_numeric(df["retraction_year"], errors="coerce").astype("Int64")
    df["original_year"] = pd.to_numeric(df["original_year"], errors="coerce").astype("Int64")
    df["journal"] = df["journal"].fillna("[missing journal]").astype(str)
    df["publisher"] = df["publisher"].fillna("[missing publisher]").astype(str)
    df["publisher_family"] = [publisher_family(p, j) for p, j in zip(df["publisher"], df["journal"])]
    df["lag_years"] = df["retraction_year"].astype("float") - df["original_year"].astype("float")
    for col in MAINCAT_COLS + GOV_SIGNAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df[df["retraction_nature"].eq("Retraction")].copy()


def add_rank_columns(counts: pd.DataFrame, year_col: str, count_col: str = "n_retractions") -> pd.DataFrame:
    counts = counts.sort_values([year_col, count_col, "journal"], ascending=[True, False, True]).copy()
    counts["journal_rank"] = counts.groupby(year_col)[count_col].rank(method="first", ascending=False).astype(int)
    counts["n_journals_in_year"] = counts.groupby(year_col)["journal"].transform("nunique").astype(int)
    counts["annual_total_retractions"] = counts.groupby(year_col)[count_col].transform("sum").astype(int)
    counts["share_of_year_percent"] = (counts[count_col] / counts["annual_total_retractions"] * 100).round(4)
    counts["rank_percentile_topness"] = np.where(
        counts["n_journals_in_year"].gt(1),
        (1 - (counts["journal_rank"] - 1) / (counts["n_journals_in_year"] - 1)).round(6),
        1.0,
    )
    for n in (1, 5, 10):
        counts[f"is_top{n}"] = counts["journal_rank"].le(n)
    return counts


def annual_concentration_from_counts(ranked: pd.DataFrame, year_col: str, perspective: str) -> pd.DataFrame:
    rows = []
    for year, g in ranked.groupby(year_col, observed=True):
        total = int(g["n_retractions"].sum())
        shares = (g["n_retractions"] / total).to_numpy() if total else np.array([])
        rows.append({
            "perspective": perspective,
            "year": int(year),
            "n_retractions": total,
            "n_journals": int(g["journal"].nunique()),
            "top1_n": int(g.loc[g["journal_rank"].le(1), "n_retractions"].sum()),
            "top5_n": int(g.loc[g["journal_rank"].le(5), "n_retractions"].sum()),
            "top10_n": int(g.loc[g["journal_rank"].le(10), "n_retractions"].sum()),
            "top1_percent": round(g.loc[g["journal_rank"].le(1), "n_retractions"].sum() / total * 100, 4) if total else 0,
            "top5_percent": round(g.loc[g["journal_rank"].le(5), "n_retractions"].sum() / total * 100, 4) if total else 0,
            "top10_percent": round(g.loc[g["journal_rank"].le(10), "n_retractions"].sum() / total * 100, 4) if total else 0,
            "hhi_journal_share": round(float(np.sum(shares ** 2)), 8) if total else 0,
            "effective_n_journals_inverse_hhi": round(float(1 / np.sum(shares ** 2)), 4) if total and np.sum(shares ** 2) > 0 else np.nan,
            "data_cutoff_latest_retraction_date_in_csv": CUTOFF,
        })
    return pd.DataFrame(rows)


def build_journal_rank_and_concentration(ret: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Main retraction-year perspective excludes 2026 partial from the concentration trend.
    main_ret = ret[ret["retraction_year"].between(MAIN_START, MAIN_END)].copy()
    ret_counts = main_ret.groupby(["retraction_year", "journal"], as_index=False).size().rename(columns={"size": "n_retractions"})
    ret_ranks = add_rank_columns(ret_counts, "retraction_year")
    ret_ranks = ret_ranks.rename(columns={
        "retraction_year": "year",
        "journal_rank": "rank_in_retraction_year",
        "n_journals_in_year": "n_journals_in_retraction_year",
        "rank_percentile_topness": "rank_percentile_topness_in_retraction_year",
        "is_top1": "is_top1_in_retraction_year",
        "is_top5": "is_top5_in_retraction_year",
        "is_top10": "is_top10_in_retraction_year",
    })
    ret_conc = annual_concentration_from_counts(
        ret_ranks.rename(columns={"year": "retraction_year", "rank_in_retraction_year": "journal_rank"}),
        "retraction_year",
        "retraction_year",
    )

    # Original-year perspective: records observed in RW through cutoff, grouped by original publication year.
    orig_base = ret[ret["original_year"].between(MAIN_START, MAIN_END)].copy()
    orig_counts = orig_base.groupby(["original_year", "journal"], as_index=False).size().rename(columns={"size": "n_retractions"})
    orig_ranks = add_rank_columns(orig_counts, "original_year")
    orig_ranks = orig_ranks.rename(columns={
        "original_year": "year",
        "journal_rank": "rank_in_original_year_cohort",
        "n_journals_in_year": "n_journals_in_original_year_cohort",
        "rank_percentile_topness": "rank_percentile_topness_in_original_year_cohort",
        "is_top1": "is_top1_in_original_year_cohort",
        "is_top5": "is_top5_in_original_year_cohort",
        "is_top10": "is_top10_in_original_year_cohort",
    })
    orig_conc = annual_concentration_from_counts(
        orig_ranks.rename(columns={"year": "original_year", "rank_in_original_year_cohort": "journal_rank"}),
        "original_year",
        "original_year_observed_retracted_cohort",
    )

    conc = pd.concat([ret_conc, orig_conc], ignore_index=True)
    conc["interpretation_limit"] = (
        "Top-N journals are recomputed within each year and perspective. Original-year cohorts are observed eventually retracted "
        "records in the current RW data, not all papers published in that year."
    )

    ret_ranks.to_csv(SOURCE_DIR / "journal_ranks_by_retraction_year_2000_2025.source.csv", index=False)
    orig_ranks.to_csv(SOURCE_DIR / "journal_ranks_by_original_year_observed_retracted_2000_2025.source.csv", index=False)
    conc.to_csv(SOURCE_DIR / "journal_topn_concentration_retraction_vs_original_year_2000_2025.source.csv", index=False)

    # Record-level context: attach dynamic retraction-year and original-year ranks to each main record.
    ret_join = ret_ranks[[
        "year", "journal", "rank_in_retraction_year", "n_journals_in_retraction_year",
        "rank_percentile_topness_in_retraction_year", "is_top1_in_retraction_year",
        "is_top5_in_retraction_year", "is_top10_in_retraction_year",
    ]].rename(columns={"year": "retraction_year"})
    orig_join = orig_ranks[[
        "year", "journal", "rank_in_original_year_cohort", "n_journals_in_original_year_cohort",
        "rank_percentile_topness_in_original_year_cohort", "is_top1_in_original_year_cohort",
        "is_top5_in_original_year_cohort", "is_top10_in_original_year_cohort",
    ]].rename(columns={"year": "original_year"})
    record_context = main_ret[["record_id", "journal", "publisher", "publisher_family", "retraction_year", "original_year"]].merge(
        ret_join, on=["retraction_year", "journal"], how="left"
    ).merge(orig_join, on=["original_year", "journal"], how="left")
    record_context["top10_rank_context"] = np.select(
        [
            record_context["is_top10_in_retraction_year"].eq(True) & record_context["is_top10_in_original_year_cohort"].eq(True),
            record_context["is_top10_in_retraction_year"].eq(True) & ~record_context["is_top10_in_original_year_cohort"].eq(True),
            ~record_context["is_top10_in_retraction_year"].eq(True) & record_context["is_top10_in_original_year_cohort"].eq(True),
        ],
        ["Top 10 in both perspectives", "Top 10 in retraction year only", "Top 10 in original-year cohort only"],
        default="Not top 10 in either perspective",
    )
    record_context["interpretation_limit"] = (
        "Rank context is dynamic and year-specific. Original-year rank is within observed eventually retracted records, not all publications."
    )
    record_context.to_csv(SOURCE_DIR / "record_level_dynamic_journal_rank_context_2000_2025.source.csv", index=False)

    context_summary = (
        record_context.groupby(["retraction_year", "top10_rank_context"], as_index=False)
        .size().rename(columns={"size": "n_records"})
    )
    context_summary["annual_total_records"] = context_summary.groupby("retraction_year")["n_records"].transform("sum")
    context_summary["percent_of_retraction_year"] = (context_summary["n_records"] / context_summary["annual_total_records"] * 100).round(4)
    context_summary.to_csv(SOURCE_DIR / "dynamic_top10_rank_context_by_retraction_year_2000_2025.source.csv", index=False)
    return conc, record_context, context_summary


def plot_journal_concentration(conc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), sharey=True)
    labels = [("retraction_year", "A. Ranked by retraction year"), ("original_year_observed_retracted_cohort", "B. Ranked by original publication year")]
    palette = {"top1_percent": COLORS["blue"], "top5_percent": COLORS["amber"], "top10_percent": COLORS["coral"]}
    for ax, (persp, title) in zip(axes, labels):
        sub = conc[conc["perspective"].eq(persp)]
        for col, label in [("top1_percent", "Top 1"), ("top5_percent", "Top 5"), ("top10_percent", "Top 10")]:
            ax.plot(sub["year"], sub[col], lw=1.55, color=palette[col], label=label)
        ax.set_title(title, loc="left", weight="bold", pad=5)
        ax.set_xlabel("Year")
        ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
        ax.grid(axis="y", color="#E8E8E8")
    axes[0].set_ylabel("Share of records in yearly top journals (%)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    note = (
        "Top journals are recalculated within each year and perspective. Original-year cohorts include observed eventually retracted "
        "records, not all publications in that year."
    )
    fig.text(0.07, 0.01, textwrap.fill(note, 130), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.23, top=0.88, wspace=0.12)
    save_all(fig, "journal_topn_concentration_retraction_vs_original_year_2000_2025")


def plot_top10_rank_context(context_summary: pd.DataFrame) -> None:
    order = [
        "Top 10 in both perspectives",
        "Top 10 in retraction year only",
        "Top 10 in original-year cohort only",
        "Not top 10 in either perspective",
    ]
    colors = [COLORS["blue"], COLORS["amber"], COLORS["teal"], COLORS["lightgrey"]]
    pivot = context_summary.pivot(index="retraction_year", columns="top10_rank_context", values="percent_of_retraction_year").reindex(columns=order).fillna(0)
    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    bottom = np.zeros(len(pivot))
    for label, color in zip(order, colors):
        vals = pivot[label].to_numpy()
        ax.bar(pivot.index, vals, bottom=bottom, width=0.82, color=color, edgecolor="white", linewidth=0.18, label=label)
        bottom += vals
    ax.set_title("Dynamic top-10 journal context differs by retraction year and publication year", loc="left", weight="bold", pad=6)
    ax.set_ylabel("Share of retraction-year records (%)")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note = "Each record is classified using its journal's dynamic rank in the retraction year and in its original-publication-year cohort."
    fig.text(0.075, 0.01, textwrap.fill(note, 110), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.70, bottom=0.22, top=0.88)
    save_all(fig, "dynamic_top10_journal_rank_context_by_retraction_year_2000_2025")


def build_and_plot_lag_heatmap(ret: pd.DataFrame) -> None:
    main_ret = ret[ret["retraction_year"].between(MAIN_START, MAIN_END)].copy()
    lag_summary = main_ret.groupby("retraction_year").agg(
        n_records=("record_id", "size"),
        median_lag_years=("lag_years", "median"),
        mean_lag_years=("lag_years", "mean"),
        p25_lag_years=("lag_years", lambda x: np.nanpercentile(x, 25)),
        p75_lag_years=("lag_years", lambda x: np.nanpercentile(x, 75)),
        share_lag_0_2=("lag_years", lambda x: np.mean((x >= 0) & (x <= 2)) * 100),
        share_lag_3_5=("lag_years", lambda x: np.mean((x >= 3) & (x <= 5)) * 100),
        share_lag_ge6=("lag_years", lambda x: np.mean(x >= 6) * 100),
    ).reset_index()
    for c in ["median_lag_years", "mean_lag_years", "p25_lag_years", "p75_lag_years", "share_lag_0_2", "share_lag_3_5", "share_lag_ge6"]:
        lag_summary[c] = lag_summary[c].round(4)
    lag_summary.to_csv(SOURCE_DIR / "retraction_lag_summary_by_retraction_year_2000_2025.source.csv", index=False)

    # Matrix: main view keeps original years from 2000 onward; pre-2000 are summarized separately in source.
    heat_base = main_ret[main_ret["original_year"].between(MAIN_START, MAIN_END)].copy()
    matrix = heat_base.groupby(["retraction_year", "original_year"], as_index=False).size().rename(columns={"size": "n_records"})
    matrix.to_csv(SOURCE_DIR / "retraction_year_x_original_year_matrix_2000_2025.source.csv", index=False)
    wide = matrix.pivot(index="original_year", columns="retraction_year", values="n_records").reindex(index=range(MAIN_START, MAIN_END + 1), columns=range(MAIN_START, MAIN_END + 1)).fillna(0)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    data = wide.to_numpy()
    masked = np.ma.masked_where(data == 0, data)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("rw_heat", ["#F7FAFC", COLORS["sky"], COLORS["blue"], "#17324D"])
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, norm=mpl.colors.LogNorm(vmin=1, vmax=max(2, data.max())))
    ax.set_title("Retraction timing is not the same as publication timing", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Retraction year")
    ax.set_ylabel("Original publication year")
    ticks = [2000, 2005, 2010, 2015, 2020, 2025]
    ax.set_xticks([t - MAIN_START for t in ticks], labels=[str(t) for t in ticks])
    ax.set_yticks([t - MAIN_START for t in ticks], labels=[str(t) for t in ticks])
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Retraction records (log scale)")
    note = "Matrix uses records retracted in 2000–2025 with original publication years 2000–2025; older original years are retained in source summaries but outside this heatmap view."
    fig.text(0.09, 0.01, textwrap.fill(note, 105), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.16, top=0.90)
    save_all(fig, "retraction_year_x_original_year_heatmap_2000_2025")

    # Lag distribution plot.
    fig, ax = plt.subplots(figsize=(7.25, 3.1))
    ax.plot(lag_summary["retraction_year"], lag_summary["median_lag_years"], color=COLORS["blue"], lw=1.6, label="Median lag")
    ax.fill_between(lag_summary["retraction_year"].to_numpy(), lag_summary["p25_lag_years"].to_numpy(), lag_summary["p75_lag_years"].to_numpy(), color=COLORS["sky"], alpha=0.35, label="IQR")
    ax.set_title("Lag from publication to retraction varies over time", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Retraction year")
    ax.set_ylabel("Lag, years")
    ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.88)
    save_all(fig, "retraction_lag_summary_by_retraction_year_2000_2025")


def build_and_plot_maincat_overlap_and_trends(ret: pd.DataFrame) -> dict:
    main_ret = ret[ret["retraction_year"].between(MAIN_START, MAIN_END)].copy()
    main_ret["n_maincats_final"] = main_ret[MAINCAT_COLS].sum(axis=1).astype(int)
    main_ret["maincat_combination_binary"] = main_ret[MAINCAT_COLS].astype(int).astype(str).agg("".join, axis=1)
    main_ret["maincat_combination_label"] = main_ret[MAINCAT_COLS].apply(
        lambda row: "; ".join(MAINCAT_LABELS[col] for col in MAINCAT_COLS if int(row[col]) == 1) or "No mapped final maincat layer",
        axis=1,
    )
    main_ret["maincat_combination_short_label"] = main_ret[MAINCAT_COLS].apply(
        lambda row: "+".join(MAINCAT_ABBREVIATIONS[col] for col in MAINCAT_COLS if int(row[col]) == 1) or "No mapped",
        axis=1,
    )
    total_main = len(main_ret)

    count_dist = (
        main_ret.groupby("n_maincats_final", as_index=False)
        .size()
        .rename(columns={"size": "n_records"})
        .set_index("n_maincats_final")
        .reindex(range(0, len(MAINCAT_COLS) + 1), fill_value=0)
        .reset_index()
    )
    count_dist["main_retraction_year_2000_2025_records"] = total_main
    count_dist["percent_of_main_records"] = (count_dist["n_records"] / total_main * 100).round(4)
    count_dist["interpretation_limit"] = "Final maincat fields are non-exclusive metadata layers among Retraction records; not responsibility/fault shares."
    count_dist.to_csv(SOURCE_DIR / "donut_count_distribution_2000_2025.source.csv", index=False)

    combos = (
        main_ret.groupby(["maincat_combination_binary", "maincat_combination_short_label", "maincat_combination_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_records"})
    )
    combos["n_maincats_final"] = combos["maincat_combination_binary"].str.count("1")
    combos.insert(0, "maincat_combination_code", "b" + combos["maincat_combination_binary"])
    combos["main_retraction_year_2000_2025_records"] = total_main
    combos["percent_of_main_records"] = (combos["n_records"] / total_main * 100).round(4)
    combos["combination_field_order"] = " | ".join(MAINCAT_COLS)
    combos["interpretation_limit"] = "Exact four-field combinations of final non-exclusive maincat_* metadata; not responsibility/fault allocation."
    combos = combos.sort_values(["n_records", "maincat_combination_code"], ascending=[False, True])
    combos.to_csv(SOURCE_DIR / "donut_exact_combinations_2000_2025.source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.65), gridspec_kw={"width_ratios": [0.82, 1.55]})
    dist_plot = count_dist[count_dist["n_maincats_final"].between(0, len(MAINCAT_COLS))]
    axes[0].bar(
        dist_plot["n_maincats_final"].astype(str),
        dist_plot["n_records"],
        color=[COLORS["lightgrey"], COLORS["sky"], COLORS["teal"], COLORS["amber"], COLORS["coral"]],
        edgecolor="white",
        linewidth=0.35,
    )
    axes[0].set_title("A. Number of mapped maincat layers", loc="left", weight="bold", pad=5)
    axes[0].set_xlabel("Mapped final maincat flags per record")
    axes[0].set_ylabel("Retraction records")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[0].grid(axis="y", color="#E8E8E8")

    combo_plot = combos[combos["n_maincats_final"].gt(0)].head(10).iloc[::-1]
    axes[1].barh(combo_plot["maincat_combination_short_label"], combo_plot["n_records"], color=COLORS["blue"], edgecolor="white", linewidth=0.25)
    axes[1].set_title("B. Most common exact combinations", loc="left", weight="bold", pad=5)
    axes[1].set_xlabel("Retraction records")
    axes[1].xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].grid(axis="x", color="#E8E8E8")
    axes[1].tick_params(axis="y", labelsize=6.1)
    note = (
        "Final maincat fields are non-exclusive mapped metadata layers. R=research-content reliability; A=attribution/disclosure/ethics; "
        "E=editorial/peer-review governance; P=post-publication oversight. The zero bar marks records without one of the four mapped final maincat layers."
    )
    fig.text(0.065, 0.012, textwrap.fill(note, 120), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.88, wspace=0.80)
    save_all(fig, "maincat_cooccurrence_combination_distribution_2000_2025")

    annual_rows = []
    for year, g in main_ret.groupby("retraction_year", observed=True):
        denom = len(g)
        for col in MAINCAT_COLS:
            n = int(g[col].sum())
            annual_rows.append({
                "year": int(year),
                "maincat": col,
                "maincat_label": MAINCAT_LABELS[col],
                "n_records_with_maincat": n,
                "annual_retraction_records": denom,
                "record_level_prevalence_percent": round(n / denom * 100, 4) if denom else 0,
                "interpretation_limit": "Non-exclusive final maincat_* metadata prevalence among Retraction records; not responsibility/fault share.",
            })
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(SOURCE_DIR / "annual_domain_prevalence_2000_2025.source.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.25, 3.65))
    colors = [COLORS[k] for k in ["blue", "amber", "teal", "coral"]]
    for col, color in zip(MAINCAT_COLS, colors):
        sub = annual[annual["maincat"].eq(col)]
        ax.plot(sub["year"], sub["record_level_prevalence_percent"], lw=1.55, color=color, label=MAINCAT_LABELS[col])
    ax.set_title("Final maincat metadata prevalence over time", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Retraction year")
    ax.set_ylabel("Record-level prevalence among retractions (%)")
    ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note = "Lines use final non-exclusive maincat_* fields among Retraction records in 2000-2025; 2026 partial records are excluded."
    fig.text(0.075, 0.012, textwrap.fill(note, 112), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.70, bottom=0.20, top=0.90)
    save_all(fig, "annual_domain_prevalence_2000_2025")

    lag_base = main_ret[main_ret["lag_years"].notna() & main_ret["lag_years"].ge(0)].copy()
    lag_base["lag_group"] = pd.cut(
        lag_base["lag_years"],
        bins=[-0.1, 0, 2, 5, np.inf],
        labels=["Same year", "1-2 years", "3-5 years", ">=6 years"],
        right=True,
    )
    lag_rows = []
    for lag_group, g in lag_base.groupby("lag_group", observed=False):
        denom = len(g)
        for col in MAINCAT_COLS:
            n = int(g[col].sum())
            lag_rows.append({
                "lag_group": str(lag_group),
                "maincat": col,
                "maincat_label": MAINCAT_LABELS[col],
                "n_records_with_maincat": n,
                "lag_group_retraction_records": denom,
                "record_level_prevalence_percent": round(n / denom * 100, 4) if denom else 0,
                "excluded_records_missing_or_negative_lag": int(total_main - len(lag_base)),
                "interpretation_limit": "Lag-stratified prevalence of non-exclusive final maincat_* metadata; not responsibility/fault share.",
            })
    lag_prev = pd.DataFrame(lag_rows)
    lag_prev.to_csv(SOURCE_DIR / "maincat_lag_stratified_prevalence_2000_2025.source.csv", index=False)

    pivot = lag_prev.pivot(index="lag_group", columns="maincat_label", values="record_level_prevalence_percent").reindex(
        ["Same year", "1-2 years", "3-5 years", ">=6 years"]
    )
    fig, ax = plt.subplots(figsize=(7.25, 3.65))
    x = np.arange(len(pivot.index))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MAINCAT_COLS))
    for col, label, color, offset in zip(MAINCAT_COLS, [MAINCAT_LABELS[c] for c in MAINCAT_COLS], colors, offsets):
        vals = pivot[label].to_numpy()
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", linewidth=0.25, label=label)
    ax.set_title("Final maincat prevalence by publication-to-retraction lag", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Lag group")
    ax.set_ylabel("Record-level prevalence among lag group (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels=pivot.index)
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note = "Lag groups use Retraction records from 2000-2025 with non-missing, non-negative publication-to-retraction lag."
    fig.text(0.075, 0.012, textwrap.fill(note, 112), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.70, bottom=0.20, top=0.90)
    save_all(fig, "maincat_lag_stratified_prevalence_2000_2025")

    return {
        "main_records": total_main,
        "zero_maincat_records": int(count_dist.loc[count_dist["n_maincats_final"].eq(0), "n_records"].iloc[0]),
        "multi_maincat_records": int(main_ret["n_maincats_final"].ge(2).sum()),
        "lag_stratified_records": int(len(lag_base)),
        "lag_excluded_records": int(total_main - len(lag_base)),
    }


def build_and_plot_governance_trends(ret: pd.DataFrame) -> None:
    main_ret = ret[ret["retraction_year"].between(MAIN_START, MAIN_END)].copy()
    rows = []
    for year, g in main_ret.groupby("retraction_year", observed=True):
        denom = len(g)
        for col in GOV_SIGNAL_COLS:
            n = int(g[col].sum())
            rows.append({
                "year": int(year),
                "signal": col,
                "signal_label": SIGNAL_LABELS[col],
                "n_records_with_signal": n,
                "annual_retraction_records": denom,
                "record_level_prevalence_percent": round(n / denom * 100, 4) if denom else 0,
                "interpretation_limit": "Non-exclusive Retraction Watch metadata signal prevalence among retraction records; not fault or responsibility share.",
            })
    annual = pd.DataFrame(rows)
    annual.to_csv(SOURCE_DIR / "governance_signal_annual_prevalence_2000_2025.source.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    colors = [COLORS[k] for k in ["blue", "amber", "coral", "teal", "sky", "lavender", "rose", "green", "darkgrey"]]
    for col, color in zip(GOV_SIGNAL_COLS, colors):
        sub = annual[annual["signal"].eq(col)]
        ax.plot(sub["year"], sub["record_level_prevalence_percent"], lw=1.35, color=color, label=SIGNAL_LABELS[col])
    ax.set_title("Governance-related metadata signals over time", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Retraction year")
    ax.set_ylabel("Record-level prevalence among retractions (%)")
    ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note = "Signals are non-exclusive metadata categories mapped from Retraction Watch reasons; lines do not assign legal, moral, causal, or fault responsibility."
    fig.text(0.075, 0.012, textwrap.fill(note, 110), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.66, bottom=0.20, top=0.90)
    save_all(fig, "governance_signal_annual_prevalence_2000_2025")

    fig, axes = plt.subplots(3, 3, figsize=(7.25, 6.0), sharex=True, sharey=True)
    ymax = max(5, float(annual["record_level_prevalence_percent"].max()))
    ymax = min(100, np.ceil(ymax / 5) * 5)
    for ax, col, color in zip(axes.flat, GOV_SIGNAL_COLS, colors):
        sub = annual[annual["signal"].eq(col)]
        ax.plot(sub["year"], sub["record_level_prevalence_percent"], lw=1.2, color=color)
        ax.set_title(SIGNAL_LABELS[col], loc="left", pad=3, fontsize=7.1)
        ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
        ax.set_ylim(0, ymax)
        ax.set_xticks([2000, 2010, 2020])
        ax.grid(axis="y", color="#E8E8E8")
    for ax in axes[:, 0]:
        ax.set_ylabel("Prevalence (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Retraction year")
    fig.suptitle("Governance-related metadata signals over time (shared y-axis)", x=0.06, y=0.985, ha="left", weight="bold", fontsize=9.4)
    note = (
        "All panels use the same y-axis scale. Signals are non-exclusive Retraction Watch metadata mappings and do not assign legal, moral, causal, or fault responsibility."
    )
    fig.text(0.06, 0.012, textwrap.fill(note, 126), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.105, top=0.91, hspace=0.46, wspace=0.20)
    save_all(fig, "governance_signal_annual_prevalence_small_multiples_2000_2025")


def decompose_cluster(ret: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    sub = ret[mask].copy()
    cluster_dir = SOURCE_DIR / "spike_decomposition"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    total = len(sub)
    top_journals = sub.groupby("journal", as_index=False).size().rename(columns={"size": "n_records"}).sort_values("n_records", ascending=False)
    top_journals["percent_of_cluster"] = (top_journals["n_records"] / total * 100).round(4) if total else 0
    top_journals.to_csv(cluster_dir / f"{slug}_top_journals.source.csv", index=False)

    original_years = sub.groupby("original_year", as_index=False).size().rename(columns={"size": "n_records"}).sort_values("original_year")
    original_years["percent_of_cluster"] = (original_years["n_records"] / total * 100).round(4) if total else 0
    original_years.to_csv(cluster_dir / f"{slug}_original_year_distribution.source.csv", index=False)

    reasons = sub.assign(raw_reason_split=sub["raw_reason"].fillna("").str.split(";"))
    reason_long = reasons.explode("raw_reason_split")
    reason_long["raw_reason_split"] = reason_long["raw_reason_split"].astype(str).str.strip()
    reason_counts = reason_long[reason_long["raw_reason_split"].ne("")].groupby("raw_reason_split", as_index=False).size().rename(columns={"size": "n_records_with_reason"}).sort_values("n_records_with_reason", ascending=False)
    reason_counts["percent_of_cluster_records_nonexclusive"] = (reason_counts["n_records_with_reason"] / total * 100).round(4) if total else 0
    reason_counts.to_csv(cluster_dir / f"{slug}_raw_reason_prevalence.source.csv", index=False)

    signal_cols = MAINCAT_COLS + GOV_SIGNAL_COLS
    signal_rows = []
    for col in signal_cols:
        n = int(sub[col].sum()) if total else 0
        signal_rows.append({
            "signal": col,
            "signal_label": SIGNAL_LABELS.get(col, col.replace("maincat_", "").replace("_", " ")),
            "n_records_with_signal": n,
            "cluster_records": total,
            "percent_of_cluster_records_nonexclusive": round(n / total * 100, 4) if total else 0,
            "interpretation_limit": "Non-exclusive metadata signal prevalence within descriptive cluster; not fault/responsibility allocation.",
        })
    signal_df = pd.DataFrame(signal_rows).sort_values("percent_of_cluster_records_nonexclusive", ascending=False)
    signal_df.to_csv(cluster_dir / f"{slug}_semantic_signal_prevalence.source.csv", index=False)
    return {"name": name, "slug": slug, "n_records": total, "top_journals": top_journals, "signals": signal_df}


def build_and_plot_spike_decomposition(ret: pd.DataFrame) -> None:
    clusters = [
        ("IEEE 2010–2011", ret["publisher_family"].eq("IEEE") & ret["retraction_year"].isin([2010, 2011])),
        ("Hindawi 2023", ret["publisher_family"].eq("Hindawi") & ret["retraction_year"].eq(2023)),
        ("SAGE/IOS 2024–2025", ret["publisher_family"].eq("SAGE / IOS Press") & ret["retraction_year"].isin([2024, 2025])),
    ]
    outputs = [decompose_cluster(ret, name, mask) for name, mask in clusters]
    summary = pd.DataFrame([{"cluster": x["name"], "slug": x["slug"], "n_records": x["n_records"]} for x in outputs])
    summary.to_csv(SOURCE_DIR / "spike_decomposition_cluster_summary.source.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 4.1), sharex=False)
    for ax, out in zip(axes, outputs):
        top = out["top_journals"].head(8).iloc[::-1]
        ax.barh(top["journal"], top["n_records"], color=COLORS["blue"], edgecolor="white", linewidth=0.2)
        ax.set_title(f"{out['name']}\n(n={out['n_records']:,})", loc="left", weight="bold", pad=4)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="x", color="#E8E8E8")
        ax.tick_params(axis="y", labelsize=5.5)
    axes[0].set_xlabel("Records")
    axes[1].set_xlabel("Records")
    axes[2].set_xlabel("Records")
    note = "Panels show top journals/proceedings within descriptive publisher-year clusters; cluster labels are not responsibility claims."
    fig.text(0.06, 0.012, textwrap.fill(note, 120), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.16, top=0.84, wspace=0.85)
    save_all(fig, "spike_decomposition_top_journals_selected_clusters")


def mirror_title_pattern(journal: str) -> str | None:
    j = str(journal or "").strip()
    jl = j.lower()
    # Specific high-salience exact/family leads first. These are only unverified leads unless official relationship ledger confirms them.
    if jl == "heliyon":
        return "specific lead: Heliyon"
    if jl == "iscience":
        return "specific lead: iScience"
    if jl == "scientific reports":
        return "specific lead: Scientific Reports"
    if jl == "nature communications":
        return "specific lead: Nature Communications"
    if jl.startswith("npj "):
        return "specific lead: npj title"
    if jl.startswith("the lancet regional health"):
        return "specific lead: Lancet Regional Health"
    if jl.startswith("cell reports"):
        return "specific lead: Cell Reports family"
    if jl == "rsc advances":
        return "specific lead: RSC Advances"
    if re.search(r"\bjacs au\b", jl) or re.search(r"\bacs .* au\b", jl):
        return "specific lead: ACS Au family"
    if jl.startswith("results in "):
        return "title pattern: Results in ..."
    if jl.startswith("materials today"):
        return "title pattern: Materials Today family"
    # Generic title-pattern leads.
    if re.search(r"\bcommunications\b", jl):
        return "title pattern: Communications"
    if re.search(r"\breports\b", jl):
        return "title pattern: Reports"
    if re.search(r"\badvances\b", jl):
        return "title pattern: Advances"
    if re.search(r"\bopen\b", jl):
        return "title pattern: Open"
    if re.search(r"\bcase reports\b", jl):
        return "title pattern: Case Reports"
    return None


def build_and_plot_mirror_pattern_exploratory(ret: pd.DataFrame) -> None:
    main_ret = ret[ret["retraction_year"].between(MAIN_START, MAIN_END)].copy()
    main_ret["unverified_title_pattern_lead"] = main_ret["journal"].map(mirror_title_pattern)
    leads = main_ret[main_ret["unverified_title_pattern_lead"].notna()].copy()
    leads["evidence_status"] = "unverified title-pattern lead only; not official sister/mirror/companion relationship evidence"
    cols = ["record_id", "journal", "publisher", "publisher_family", "retraction_year", "original_year", "unverified_title_pattern_lead", "evidence_status"]
    leads[cols].to_csv(SOURCE_DIR / "unverified_sister_mirror_companion_title_pattern_record_leads_2000_2025.source.csv", index=False)

    annual = leads.groupby(["retraction_year", "unverified_title_pattern_lead"], as_index=False).size().rename(columns={"size": "n_records"})
    total_annual = main_ret.groupby("retraction_year").size().rename("annual_retraction_records").reset_index()
    annual = annual.merge(total_annual, on="retraction_year", how="left")
    annual["percent_of_annual_retraction_records"] = (annual["n_records"] / annual["annual_retraction_records"] * 100).round(4)
    annual["interpretation_limit"] = "Exploratory title-pattern leads only; official relationship ledger required before sister/mirror/companion claims."
    annual.to_csv(SOURCE_DIR / "unverified_sister_mirror_companion_title_pattern_annual_counts_2000_2025.source.csv", index=False)

    top_titles = leads.groupby(["journal", "unverified_title_pattern_lead"], as_index=False).size().rename(columns={"size": "n_records"}).sort_values("n_records", ascending=False)
    top_titles["interpretation_limit"] = "Title-pattern lead only; not official relationship evidence."
    top_titles.to_csv(SOURCE_DIR / "unverified_sister_mirror_companion_title_pattern_top_titles_2000_2025.source.csv", index=False)

    # Plot top title-pattern lead categories over time (only top 8 by total count for readability).
    top_patterns = annual.groupby("unverified_title_pattern_lead")["n_records"].sum().sort_values(ascending=False).head(8).index.tolist()
    fig, ax = plt.subplots(figsize=(7.25, 3.8))
    palette = [COLORS[k] for k in ["blue", "amber", "coral", "teal", "sky", "lavender", "rose", "green"]]
    for pat, color in zip(top_patterns, palette):
        sub = annual[annual["unverified_title_pattern_lead"].eq(pat)]
        ax.plot(sub["retraction_year"], sub["n_records"], lw=1.35, color=color, label=pat)
    ax.set_title("Exploratory title-pattern leads for companion/mirror-journal follow-up", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Retraction year")
    ax.set_ylabel("Retraction records")
    ax.set_xlim(MAIN_START - 0.5, MAIN_END + 0.5)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note = "Exploratory title-pattern screen only. These rows are discovery leads and do not establish official sister, mirror, companion, or partner-journal relationships."
    fig.text(0.075, 0.012, textwrap.fill(note, 112), fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.64, bottom=0.20, top=0.90)
    save_all(fig, "unverified_sister_mirror_companion_title_pattern_trends_2000_2025")


def write_readme_like_index() -> None:
    # CSV index, not Markdown, to comply with workspace artifact governance.
    rows = []
    for path in sorted(OUT_DIR.glob("**/*")):
        if path.is_file():
            rows.append({"path": str(path), "file_name": path.name, "suffix": path.suffix, "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "artifact_index.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR / "spike_decomposition").mkdir(parents=True, exist_ok=True)
    setup_style()
    ret = load_data()

    # Scope summary.
    scope = pd.DataFrame([
        {
            "scope": "RetractionNature == Retraction",
            "all_retraction_records": len(ret),
            "main_retraction_year_2000_2025_records": int(ret["retraction_year"].between(MAIN_START, MAIN_END).sum()),
            "partial_2026_retraction_records": int(ret["retraction_year"].eq(2026).sum()),
            "unique_journals_all_retractions": int(ret["journal"].nunique()),
            "unique_publishers_all_retractions": int(ret["publisher"].nunique()),
            "data_cutoff_latest_retraction_date_in_csv": CUTOFF,
            "interpretation_limit": "Retraction Watch metadata; no legal, moral, causal, fault, or responsibility assignment.",
        }
    ])
    scope.to_csv(SOURCE_DIR / "analysis_scope_summary.source.csv", index=False)

    conc, record_context, context_summary = build_journal_rank_and_concentration(ret)
    plot_journal_concentration(conc)
    plot_top10_rank_context(context_summary)
    build_and_plot_lag_heatmap(ret)
    maincat_validation = build_and_plot_maincat_overlap_and_trends(ret)
    build_and_plot_governance_trends(ret)
    build_and_plot_spike_decomposition(ret)
    build_and_plot_mirror_pattern_exploratory(ret)
    write_readme_like_index()

    print(f"wrote {OUT_DIR}")
    print(f"main records 2000-2025: {int(ret['retraction_year'].between(MAIN_START, MAIN_END).sum())}")
    print(f"dynamic rank context rows: {len(record_context)}")
    print(f"maincat zero-flag records: {maincat_validation['zero_maincat_records']}")
    print(f"maincat multi-flag records: {maincat_validation['multi_maincat_records']}")
    print(f"lag-stratified maincat records: {maincat_validation['lag_stratified_records']}")
    print(f"lag-stratified excluded missing/negative lag records: {maincat_validation['lag_excluded_records']}")
    print("figures:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
