#!/usr/bin/env python3
"""Generate Nature-style candidate figures from final v0.3 Retraction Watch maincat data.

This script uses the final post-Codex v0.3 API outputs only:
- data/final/record_level_binary.csv
- data/final/validation_report.json

Main manuscript-category figures use maincat_* flags, not raw cat_* or actor-context fields.
The unit is record-level, non-exclusive prevalence among RetractionNature == Retraction,
RetractionDate year 2000-2025. Percentages may sum above 100%.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FINAL = ROOT / "data" / "final"
RECORD_PATH = DATA_FINAL / "record_level_binary.csv"
VALIDATION_PATH = DATA_FINAL / "validation_report.json"
OUT_DIR = ROOT / "figures" / "v0.3-maincat-candidates"
SOURCE_DIR = OUT_DIR / "source-data"

SCOPE_NOTE = "Retractions only, 2000-2025; 2026 excluded as a partial year."
REQUIRED_NOTE = (
    "Retraction Watch reasons were treated as curated, non-exclusive metadata and mapped "
    "to manuscript-facing semantic categories. The categories do not assign legal, moral, "
    "causal, or fault responsibility."
)

CATEGORY_ORDER = [
    "maincat_research_content_reliability",
    "maincat_attribution_authorship_disclosure_integrity",
    "maincat_post_publication_transparency_due_process_oversight",
    "maincat_editorial_peer_review_governance",
]

# Short labels for plotting; full labels are kept in source-data CSVs.
CATEGORY_SHORT = {
    "maincat_research_content_reliability": "Research-content\nreliability",
    "maincat_attribution_authorship_disclosure_integrity": "Attribution, authorship,\ndisclosure & ethics",
    "maincat_post_publication_transparency_due_process_oversight": "Post-publication\nprocess & oversight",
    "maincat_editorial_peer_review_governance": "Editorial & peer-review\ngovernance",
}

CATEGORY_SHORT_ONE_LINE = {
    "maincat_research_content_reliability": "Research-content reliability",
    "maincat_attribution_authorship_disclosure_integrity": "Attribution, authorship, disclosure & ethics",
    "maincat_post_publication_transparency_due_process_oversight": "Post-publication process & oversight",
    "maincat_editorial_peer_review_governance": "Editorial & peer-review governance",
}

CATEGORY_COLORS = {
    "maincat_research_content_reliability": "#0072B2",  # blue
    "maincat_attribution_authorship_disclosure_integrity": "#009E73",  # green
    "maincat_post_publication_transparency_due_process_oversight": "#D55E00",  # vermillion
    "maincat_editorial_peer_review_governance": "#CC79A7",  # reddish purple
}

FOCUSED_FLAGS = [
    ("gov_journal_publisher_investigation", "Investigation by journal/publisher"),
    ("third_party_investigation", "Investigation by third party"),
    ("institutional_investigation", "Institutional investigation"),
    ("official_misconduct_process", "Official misconduct process"),
    ("gov_notice_opacity", "Notice transparency / limited information"),
    ("gov_correction_update_process", "Correction/update/retraction status"),
    ("gov_removal_availability", "Removal or availability status"),
    ("gov_legal_process", "Legal process or threats"),
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_main_scope() -> tuple[pd.DataFrame, dict]:
    validation = json.loads(VALIDATION_PATH.read_text())
    df = pd.read_csv(RECORD_PATH, low_memory=False)
    df["retraction_year"] = pd.to_numeric(df["retraction_year"], errors="coerce").astype("Int64")
    scoped = df[(df["retraction_nature"] == "Retraction") & (df["retraction_year"].between(2000, 2025))].copy()
    for flag in CATEGORY_ORDER + [f for f, _ in FOCUSED_FLAGS]:
        if flag in scoped.columns:
            scoped[flag] = pd.to_numeric(scoped[flag], errors="coerce").fillna(0).astype(int)
    expected = validation["n_retractions_2000_2025"]
    if len(scoped) != expected:
        raise RuntimeError(f"Scoped row count {len(scoped)} does not match validation report {expected}")
    for flag in CATEGORY_ORDER:
        if flag not in scoped.columns:
            raise RuntimeError(f"Missing required maincat column: {flag}")
    return scoped, validation


PLOT_FOOTER = "Record-level, non-exclusive metadata categories; no legal, moral, causal or fault responsibility is assigned."


def add_note(fig: plt.Figure, y: float = 0.012) -> None:
    # Keep the plot footer short. The exact required caption language is preserved
    # in source-data/candidate_figure_notes.csv for the final researchwriting pass.
    fig.text(
        0.01,
        y,
        PLOT_FOOTER,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#333333",
        wrap=True,
    )


def annual_retractions(df: pd.DataFrame) -> None:
    annual = (
        df.groupby("retraction_year", observed=True)
        .size()
        .reindex(range(2000, 2026), fill_value=0)
        .rename("n_retractions")
        .reset_index()
        .rename(columns={"retraction_year": "year"})
    )
    annual["scope"] = SCOPE_NOTE
    annual.to_csv(SOURCE_DIR / "figure_01_annual_retractions_2000_2025.source.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(annual["year"], annual["n_retractions"], color="#222222", lw=1.5)
    ax.scatter(annual["year"], annual["n_retractions"], color="#222222", s=8, zorder=3)
    ax.fill_between(annual["year"].to_numpy(), annual["n_retractions"].to_numpy(), color="#222222", alpha=0.08)
    ax.set_title("Annual retractions in the Retraction Watch corpus")
    ax.set_ylabel("Number of retraction records")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(1999.6, 2025.4)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#dddddd")
    ax.text(0.0, -0.24, SCOPE_NOTE, transform=ax.transAxes, fontsize=7.5, color="#555555")
    add_note(fig, y=0.004)
    fig.subplots_adjust(bottom=0.31)
    save_all(fig, "figure_01_annual_retractions_2000_2025")


def maincat_prevalence(df: pd.DataFrame, validation: dict) -> None:
    labels_full = validation["category_labels_final"]
    denom = len(df)
    rows = []
    key_by_flag = {"maincat_" + k: k for k in labels_full}
    for flag in CATEGORY_ORDER:
        n = int(df[flag].sum())
        rows.append(
            {
                "flag": flag,
                "category": labels_full[key_by_flag[flag]],
                "category_plot_label": CATEGORY_SHORT_ONE_LINE[flag],
                "n_records": n,
                "denominator": denom,
                "percent": n / denom * 100,
                "scope": SCOPE_NOTE,
                "interpretation_note": REQUIRED_NOTE,
            }
        )
    prev = pd.DataFrame(rows).sort_values("percent", ascending=True)
    prev.to_csv(SOURCE_DIR / "figure_02_maincat_prevalence.source.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    colors = [CATEGORY_COLORS[f] for f in prev["flag"]]
    ax.barh(prev["category_plot_label"], prev["percent"], color=colors, height=0.58)
    for i, row in enumerate(prev.itertuples(index=False)):
        ax.text(row.percent + 1.0, i, f"{row.percent:.1f}%\n({row.n_records:,})", va="center", fontsize=7.5)
    ax.set_xlim(0, 75)
    ax.set_xlabel("Record-level prevalence among retractions (%)")
    ax.set_title("Four manuscript-facing semantic categories")
    ax.grid(axis="x", color="#dddddd")
    ax.text(0.0, -0.27, "Non-exclusive categories; percentages may sum above 100%. " + SCOPE_NOTE, transform=ax.transAxes, fontsize=7.5, color="#555555")
    add_note(fig, y=0.004)
    fig.subplots_adjust(left=0.34, bottom=0.36, right=0.91)
    save_all(fig, "figure_02_maincat_prevalence")


def yearly_maincat_trends(df: pd.DataFrame) -> None:
    annual_n = df.groupby("retraction_year", observed=True).size().reindex(range(2000, 2026), fill_value=0)
    long_rows = []
    for year in range(2000, 2026):
        ydf = df[df["retraction_year"] == year]
        denom = int(annual_n.loc[year])
        for flag in CATEGORY_ORDER:
            n = int(ydf[flag].sum()) if denom else 0
            long_rows.append(
                {
                    "year": year,
                    "flag": flag,
                    "category": CATEGORY_SHORT_ONE_LINE[flag],
                    "n_records": n,
                    "denominator": denom,
                    "percent": n / denom * 100 if denom else np.nan,
                    "scope": SCOPE_NOTE,
                    "interpretation_note": REQUIRED_NOTE,
                }
            )
    trends = pd.DataFrame(long_rows)
    trends.to_csv(SOURCE_DIR / "figure_03_yearly_maincat_trends.source.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.75))
    for flag in CATEGORY_ORDER:
        tmp = trends[trends["flag"] == flag]
        ax.plot(tmp["year"], tmp["percent"], lw=1.5, color=CATEGORY_COLORS[flag], label=CATEGORY_SHORT_ONE_LINE[flag])
    ax.set_title("Yearly prevalence of manuscript-facing categories")
    ax.set_ylabel("Retraction records with category (%)")
    ax.set_xlabel("Retraction year")
    ax.set_xlim(1999.6, 2025.4)
    ax.set_ylim(0, 100)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.grid(axis="y", color="#dddddd")
    ax.legend(loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(1.01, 1.02), borderaxespad=0)
    ax.text(0.0, -0.24, "Annual percentages use each year as denominator. " + SCOPE_NOTE, transform=ax.transAxes, fontsize=7.5, color="#555555")
    add_note(fig, y=0.004)
    fig.subplots_adjust(right=0.62, bottom=0.32)
    save_all(fig, "figure_03_yearly_maincat_trends")


def maincat_overlap_matrix(df: pd.DataFrame) -> None:
    denom = len(df)
    matrix = pd.DataFrame(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, dtype=float)
    count_matrix = pd.DataFrame(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, dtype=int)
    for a in CATEGORY_ORDER:
        for b in CATEGORY_ORDER:
            n = int(((df[a] == 1) & (df[b] == 1)).sum())
            matrix.loc[a, b] = n / denom * 100
            count_matrix.loc[a, b] = n
    source_rows = []
    for a in CATEGORY_ORDER:
        for b in CATEGORY_ORDER:
            source_rows.append(
                {
                    "row_flag": a,
                    "row_category": CATEGORY_SHORT_ONE_LINE[a],
                    "column_flag": b,
                    "column_category": CATEGORY_SHORT_ONE_LINE[b],
                    "n_records_with_both": int(count_matrix.loc[a, b]),
                    "denominator": denom,
                    "percent_of_retractions": float(matrix.loc[a, b]),
                    "scope": SCOPE_NOTE,
                    "interpretation_note": REQUIRED_NOTE,
                }
            )
    pd.DataFrame(source_rows).to_csv(SOURCE_DIR / "figure_04_maincat_overlap_matrix.source.csv", index=False)

    labels = [
        "Research\nreliability",
        "Attribution\nethics",
        "Post-pub.\nprocess",
        "Editorial\npeer review",
    ]
    fig, ax = plt.subplots(figsize=(6.6, 5.25))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=75)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False, pad=6)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = matrix.iloc[i, j]
            n = int(count_matrix.iloc[i, j])
            color = "white" if pct > 45 else "#222222"
            ax.text(j, i, f"{pct:.1f}%\n{n:,}", ha="center", va="center", fontsize=7.2, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Records with both categories (%)")
    ax.set_title("Overlap among the four semantic categories")
    ax.text(0.0, -0.23, "Cells show records carrying both row and column categories; diagonal cells are marginal prevalence. " + SCOPE_NOTE, transform=ax.transAxes, fontsize=7.5, color="#555555")
    add_note(fig, y=0.004)
    fig.subplots_adjust(bottom=0.32, left=0.26, right=0.93)
    save_all(fig, "figure_04_maincat_overlap_matrix")


def focused_post_publication(df: pd.DataFrame) -> None:
    denom = len(df)
    rows = []
    for flag, label in FOCUSED_FLAGS:
        if flag not in df.columns:
            continue
        n = int(df[flag].sum())
        rows.append(
            {
                "flag": flag,
                "signal": label,
                "n_records": n,
                "denominator": denom,
                "percent": n / denom * 100,
                "scope": SCOPE_NOTE,
                "interpretation_note": "Process/notice metadata signals only; named actor or investigation source is not a fault attribution. " + REQUIRED_NOTE,
            }
        )
    focused = pd.DataFrame(rows).sort_values("percent", ascending=True)
    focused.to_csv(SOURCE_DIR / "figure_05_post_publication_process_signals.source.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    ax.barh(focused["signal"], focused["percent"], color="#6A51A3", height=0.58)
    for i, row in enumerate(focused.itertuples(index=False)):
        ax.text(row.percent + 0.8, i, f"{row.percent:.1f}%\n({row.n_records:,})", va="center", fontsize=7.2)
    ax.set_xlim(0, max(65, focused["percent"].max() + 8))
    ax.set_xlabel("Record-level prevalence among retractions (%)")
    ax.set_title("Selected post-publication process and notice signals")
    ax.grid(axis="x", color="#dddddd")
    ax.text(0.0, -0.24, "Selected process/notice flags from final v0.3 table; actor or investigation source is not fault attribution. " + SCOPE_NOTE, transform=ax.transAxes, fontsize=7.5, color="#555555")
    add_note(fig, y=0.004)
    fig.subplots_adjust(left=0.35, bottom=0.33, right=0.89)
    save_all(fig, "figure_05_post_publication_process_signals")


def write_notes(validation: dict) -> None:
    rows = []
    for fig_id, title in [
        ("figure_01", "Annual retractions in the Retraction Watch corpus"),
        ("figure_02", "Four manuscript-facing semantic categories"),
        ("figure_03", "Yearly prevalence of manuscript-facing categories"),
        ("figure_04", "Overlap among the four semantic categories"),
        ("figure_05", "Selected post-publication process and notice signals"),
    ]:
        rows.append(
            {
                "figure_id": fig_id,
                "candidate_title": title,
                "scope": SCOPE_NOTE,
                "caption_required_note": REQUIRED_NOTE,
                "main_figure_rule": validation["main_figure_rule"],
                "writing_status": "candidate figure note only; route final Nature caption/prose to researchwriting profile",
            }
        )
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "candidate_figure_notes.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    df, validation = load_main_scope()
    annual_retractions(df)
    maincat_prevalence(df, validation)
    yearly_maincat_trends(df)
    maincat_overlap_matrix(df)
    focused_post_publication(df)
    write_notes(validation)
    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote source data to {SOURCE_DIR}")
    print(f"Main scope rows: {len(df):,}")


if __name__ == "__main__":
    main()
