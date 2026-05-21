#!/usr/bin/env python3
"""Generate selected Retraction Watch maincat figures with 2026 partial data included.

This focused script updates the manuscript-supporting maincat tables and figures to
include Retraction records from 2000 through the partial 2026 data cutoff
(RetractionDate through 2026-05-06). It deliberately treats the flowchart as a
supplementary/explanatory schematic, not as the main manuscript figure.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "data" / "final" / "record_level_binary.csv"
OUT_DIR = ROOT / "figures" / "v0.3-current-feasible-analyses"
SOURCE_DIR = OUT_DIR / "source-data"
SELECTED_DIR = OUT_DIR / "selected-main-figure-candidates"
SELECTED_SOURCE_DIR = SELECTED_DIR / "source-data"
CUTOFF = "2026-05-06"
YEAR_START = 2000
YEAR_END = 2026
YEAR_LABEL = "2000_2026_asof_2026-05-06"

MAINCAT_COLS = [
    "maincat_research_content_reliability",
    "maincat_attribution_authorship_disclosure_integrity",
    "maincat_editorial_peer_review_governance",
    "maincat_post_publication_transparency_due_process_oversight",
]

MAINCAT_LABELS = {
    "maincat_research_content_reliability": "Research-content reliability",
    "maincat_attribution_authorship_disclosure_integrity": "Authorship/attribution/disclosure",
    "maincat_editorial_peer_review_governance": "Editorial/peer-review process",
    "maincat_post_publication_transparency_due_process_oversight": "Post-publication process/notice transparency",
}

MAINCAT_ABBREVIATIONS = {
    "maincat_research_content_reliability": "R",
    "maincat_attribution_authorship_disclosure_integrity": "A",
    "maincat_editorial_peer_review_governance": "E",
    "maincat_post_publication_transparency_due_process_oversight": "P",
}

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
    "ink": "#2B2B2B",
    "note": "#F6EEDB",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 450,
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.4,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "grid.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_all(fig: plt.Figure, stem: str, out_dir: Path = OUT_DIR, extensions: tuple[str, ...] = ("png",)) -> None:
    """Save final figures as PNG by default; add PDF/SVG only when explicitly needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in extensions:
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.14)
    plt.close(fig)


def load_main_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "record_id", "retraction_nature", "retraction_date", "retraction_year",
        "original_year", "raw_reason", "n_raw_reasons", *MAINCAT_COLS,
    ]
    df = pd.read_csv(RECORD_PATH, usecols=cols, low_memory=False)
    df["retraction_year"] = pd.to_numeric(df["retraction_year"], errors="coerce").astype("Int64")
    df["original_year"] = pd.to_numeric(df["original_year"], errors="coerce").astype("Int64")
    df["retraction_date"] = pd.to_datetime(df["retraction_date"], errors="coerce")
    for col in MAINCAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    ret = df[df["retraction_nature"].eq("Retraction")].copy()
    main = ret[ret["retraction_year"].between(YEAR_START, YEAR_END)].copy()
    main["is_partial_year"] = main["retraction_year"].eq(2026)
    main["n_maincats_final"] = main[MAINCAT_COLS].sum(axis=1).astype(int)
    main["maincat_combination_binary"] = main[MAINCAT_COLS].astype(str).agg("".join, axis=1)
    main["maincat_combination_short_label"] = main[MAINCAT_COLS].apply(
        lambda row: "+".join(MAINCAT_ABBREVIATIONS[c] for c in MAINCAT_COLS if int(row[c]) == 1) or "No assigned primary domain",
        axis=1,
    )
    main["maincat_combination_label"] = main[MAINCAT_COLS].apply(
        lambda row: "; ".join(MAINCAT_LABELS[c] for c in MAINCAT_COLS if int(row[c]) == 1) or "No assigned primary domain",
        axis=1,
    )
    return ret, main


def write_source_tables(ret: pd.DataFrame, main: pd.DataFrame) -> dict[str, int | str]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(main)
    n_2026 = int(main["is_partial_year"].sum())
    latest_date = ret["retraction_date"].max().date().isoformat()

    scope = pd.DataFrame([{
        "scope": "RetractionNature == Retraction; retraction_year 2000-2026; 2026 partial included",
        "all_retraction_records": int(len(ret)),
        "analysis_records_2000_2026_including_partial_2026": total,
        "partial_2026_retraction_records": n_2026,
        "latest_retraction_date_in_csv": latest_date,
        "data_cutoff_used_for_labels": CUTOFF,
        "interpretation_limit": "Retraction Watch metadata; no legal, moral, causal, fault, or responsibility assignment.",
    }])
    scope.to_csv(SOURCE_DIR / f"analysis_scope_summary_{YEAR_LABEL}.source.csv", index=False)

    count_dist = (
        main.groupby("n_maincats_final", as_index=False)
        .size()
        .rename(columns={"size": "n_records"})
        .set_index("n_maincats_final")
        .reindex(range(0, len(MAINCAT_COLS) + 1), fill_value=0)
        .reset_index()
    )
    count_dist["analysis_records_2000_2026_including_partial_2026"] = total
    count_dist["percent_of_analysis_records"] = (count_dist["n_records"] / total * 100).round(4)
    count_dist["partial_2026_included"] = True
    count_dist["latest_retraction_date_in_csv"] = latest_date
    count_dist["interpretation_limit"] = "Final principal reason-domain assignments are non-exclusive Retraction Watch metadata mappings; the distribution by number of assigned primary domains is descriptive only, not severity, causality, or responsibility/fault allocation."
    count_dist.to_csv(SOURCE_DIR / f"donut_count_distribution_{YEAR_LABEL}.source.csv", index=False)

    combos = (
        main.groupby(["maincat_combination_binary", "maincat_combination_short_label", "maincat_combination_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_records"})
    )
    combos["n_maincats_final"] = combos["maincat_combination_binary"].str.count("1")
    combos.insert(0, "maincat_combination_code", "b" + combos["maincat_combination_binary"])
    combos["analysis_records_2000_2026_including_partial_2026"] = total
    combos["percent_of_analysis_records"] = (combos["n_records"] / total * 100).round(4)
    combos["partial_2026_included"] = True
    combos["latest_retraction_date_in_csv"] = latest_date
    combos["combination_field_order"] = " | ".join(MAINCAT_COLS)
    combos["interpretation_limit"] = "Exact combinations of final non-exclusive primary reason-domain assignments; not responsibility/fault allocation."
    combos = combos.sort_values(["n_records", "maincat_combination_code"], ascending=[False, True])
    combos.to_csv(SOURCE_DIR / f"donut_exact_combinations_{YEAR_LABEL}.source.csv", index=False)

    annual_rows = []
    for year, g in main.groupby("retraction_year", observed=True):
        denom = len(g)
        for col in MAINCAT_COLS:
            n = int(g[col].sum())
            annual_rows.append({
                "year": int(year),
                "is_partial_year": bool(year == 2026),
                "maincat": col,
                "maincat_label": MAINCAT_LABELS[col],
                "n_records_with_maincat": n,
                "annual_retraction_records": denom,
                "record_level_prevalence_percent": round(n / denom * 100, 4) if denom else 0,
                "latest_retraction_date_in_csv": latest_date,
                "interpretation_limit": "Non-exclusive final principal reason-domain metadata prevalence among Retraction records; not responsibility/fault share. 2026 is partial.",
            })
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(SOURCE_DIR / f"annual_domain_prevalence_{YEAR_LABEL}.source.csv", index=False)

    return {
        "total": total,
        "n_2026": n_2026,
        "latest_date": latest_date,
        "ge2": int(main["n_maincats_final"].ge(2).sum()),
        "ge1": int(main["n_maincats_final"].ge(1).sum()),
    }


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.15, color=COLORS["greyblue"], shrinkA=2, shrinkB=5, mutation_scale=10),
        zorder=1,
    )


def add_box(ax, x: float, y: float, w: float, h: float, label: str, title: str, body: str, fill: str = "#F7FAFC") -> None:
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=0.90,
        edgecolor=COLORS["blue"],
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + 0.014, y + h - 0.024, f"{label}. {title}", fontsize=7.1, weight="bold", color=COLORS["ink"], ha="left", va="top", zorder=4)
    ax.text(x + 0.014, y + h - 0.058, textwrap.fill(body, 27), fontsize=5.75, color="#333333", ha="left", va="top", linespacing=1.08, zorder=4)


def plot_flowchart(meta: dict[str, int | str], count_dist: pd.DataFrame) -> None:
    total = int(meta["total"])
    ge2 = int(meta["ge2"])
    latest_date = str(meta["latest_date"])

    fig, ax = plt.subplots(figsize=(12.2, 7.15))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.045, 0.970, "PRISMA-style derivation of the analysis sample and metadata-signal domains", fontsize=13.2, weight="bold", ha="left", va="top")
    ax.text(
        0.045, 0.927,
        "Record filtering → observed reason labels → reason-domain mapping → record-level domain assignment",
        fontsize=8.2, color="#444444", ha="left", va="top",
    )

    section_style = dict(fontsize=7.2, weight="bold", color=COLORS["blue"], ha="left", va="bottom")
    ax.text(0.055, 0.865, "Identification / eligibility", **section_style)
    ax.text(0.555, 0.865, "Reason-label classification", **section_style)

    def box(label: str, title: str, body: str, x: float, y: float, w: float, h: float, fill: str = "#F7FAFC") -> tuple[float, float, float, float]:
        add_box(ax, x, y, w, h, label, title, body, fill=fill)
        return (x, y, x + w, y + h)

    boxes = {}
    boxes["A"] = box(
        "A", "Retraction Watch source records",
        "All notice records in the source table: 70,016. A record can list more than one RW reason label.",
        0.055, 0.725, 0.380, 0.125,
    )
    boxes["B"] = box(
        "B", "Analysis subset",
        f"Retraction notices, 2000–2026*: {total:,} records; 109/112 RW reason labels observed.",
        0.055, 0.555, 0.380, 0.135, fill="#FFFFFF",
    )
    boxes["C"] = box(
        "C", "RW reason-label codebook",
        "The 112 unique RW reason labels were classified once by their recorded definitions/meanings, not separately for every paper.",
        0.555, 0.725, 0.390, 0.125,
    )
    boxes["D"] = box(
        "D", "Four metadata-signal domains",
        "R content; A authorship; E editorial; P post-publication. Metadata signals, not fault.",
        0.555, 0.555, 0.390, 0.135, fill="#FFFFFF",
    )
    boxes["E"] = box(
        "E", "Record-level assignment rule",
        "Domain assignment follows any mapped observed reason label; assignments are non-exclusive.",
        0.220, 0.355, 0.560, 0.125,
    )
    boxes["F"] = box(
        "F", "Non-exclusive metadata-signal domains",
        "Each record may be assigned to none, one, or several metadata-signal domains (R, A, E, P).",
        0.055, 0.175, 0.420, 0.125, fill="#FFFFFF",
    )
    boxes["G"] = box(
        "G", "Number of assigned metadata-signal domains",
        "Records are summarized as having no assigned domain, one domain, two domains, three domains, or four domains.",
        0.555, 0.175, 0.390, 0.125,
    )

    def mid_right(b): return (b[2] + 0.020, (b[1] + b[3]) / 2)
    def mid_left(b): return (b[0] - 0.020, (b[1] + b[3]) / 2)
    def mid_bottom(b): return ((b[0] + b[2]) / 2, b[1] - 0.014)
    def mid_top(b): return ((b[0] + b[2]) / 2, b[3] + 0.014)

    add_arrow(ax, mid_bottom(boxes["A"]), mid_top(boxes["B"]))
    add_arrow(ax, mid_bottom(boxes["C"]), mid_top(boxes["D"]))
    add_arrow(ax, mid_bottom(boxes["B"]), (boxes["E"][0] + 0.160, boxes["E"][3] + 0.008))
    add_arrow(ax, mid_bottom(boxes["D"]), (boxes["E"][2] - 0.160, boxes["E"][3] + 0.008))
    add_arrow(ax, (boxes["E"][0] + 0.160, boxes["E"][1] - 0.008), mid_top(boxes["F"]))
    add_arrow(ax, mid_right(boxes["F"]), mid_left(boxes["G"]))

    count_names = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}
    dist_lines = [f"{count_names[int(r['n_maincats_final'])]}: {int(r['n_records']):,} ({r['percent_of_analysis_records']:.1f}%)" for _, r in count_dist.iterrows()]
    result_text = "Assigned metadata-signal domains: " + "   ".join(dist_lines[:3]) + "\n" + "                                   " + "   ".join(dist_lines[3:]) + f"\nTwo or more assigned domains: {ge2:,}/{total:,} = {ge2/total*100:.1f}%"
    ax.text(0.055, 0.105, result_text, fontsize=6.35, color="#333333", ha="left", va="top", linespacing=1.45)
    ax.text(
        0.555, 0.095,
        textwrap.fill("Interpretation limit: the number of assigned metadata-signal domains describes how broadly a record's RW reason labels map across the four domains. 'No assigned primary domain' is not evidence of no problem; it usually reflects contextual, status-only, or non-assignable reason labels. Domain labels do not assign responsibility, intent, or fault.", 82),
        fontsize=6.10, color="#3D3420", ha="left", va="top", linespacing=1.12,
    )
    ax.text(
        0.045, 0.010,
        f"*2026 partial through {latest_date}. Full source: 112 RW reason labels; analysis subset: 109 observed. The distribution uses the four metadata-signal domains above.",
        fontsize=5.95, color="#555555", va="bottom",
    )

    stem = f"maincat_mapped_layers_derivation_flowchart_{YEAR_LABEL}"
    save_all(fig, stem, SELECTED_DIR)


def plot_cooccurrence(count_dist: pd.DataFrame, combos: pd.DataFrame, total: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.75), gridspec_kw={"width_ratios": [0.86, 1.65]})
    dist_plot = count_dist.copy()
    colors = [COLORS["lightgrey"], COLORS["sky"], COLORS["teal"], COLORS["amber"], COLORS["coral"]]
    bars = axes[0].bar(
        dist_plot["n_maincats_final"].astype(str), dist_plot["n_records"],
        color=colors, edgecolor="white", linewidth=0.35,
    )
    axes[0].set_title("A. Number of assigned metadata-signal domains", loc="left", weight="bold", pad=5)
    axes[0].set_xlabel("Assigned metadata-signal domains per record")
    axes[0].set_xticks(range(len(dist_plot)))
    axes[0].set_xticklabels(["None", "One", "Two", "Three", "Four"])
    axes[0].set_ylabel("Retraction records")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[0].grid(axis="y", color="#E8E8E8")
    ypad = dist_plot["n_records"].max() * 0.035
    for bar, (_, row) in zip(bars, dist_plot.iterrows()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ypad,
            f"{int(row['n_records']):,}\n{row['percent_of_analysis_records']:.1f}%",
            ha="center", va="bottom", fontsize=6.1, color="#333333",
        )
    axes[0].set_ylim(0, dist_plot["n_records"].max() * 1.18)

    combo_plot = combos[combos["n_maincats_final"].gt(0)].head(10).iloc[::-1]
    bars2 = axes[1].barh(combo_plot["maincat_combination_short_label"], combo_plot["n_records"], color=COLORS["blue"], edgecolor="white", linewidth=0.25)
    axes[1].set_title("B. Top 10 exact mapped-category combinations", loc="left", weight="bold", pad=5)
    axes[1].set_xlabel("Retraction records")
    axes[1].xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].grid(axis="x", color="#E8E8E8")
    axes[1].tick_params(axis="y", labelsize=6.3)
    xmax = combo_plot["n_records"].max() * 1.20
    axes[1].set_xlim(0, xmax)
    for bar, (_, row) in zip(bars2, combo_plot.iterrows()):
        axes[1].text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['n_records']):,} ({row['percent_of_analysis_records']:.1f}%)",
            va="center", ha="left", fontsize=6.1, color="#333333",
        )

    note = (
        "R=research-content reliability; A=authorship/attribution/disclosure; E=editorial/peer-review process; P=post-publication process/notice transparency. "
        f"Retraction records in 2000-2026 are included (N={total:,}); 2026 is partial through RetractionDate {CUTOFF}. "
        "Domain assignments are non-exclusive descriptive mappings, not severity, causality, or responsibility shares."
    )
    fig.text(0.065, 0.012, textwrap.fill(note, 135), fontsize=6.25, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.22, top=0.88, wspace=0.68)
    save_all(fig, f"maincat_cooccurrence_combination_distribution_{YEAR_LABEL}")


def plot_annual_prevalence(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.65))
    colors = [COLORS[k] for k in ["blue", "amber", "teal", "coral"]]
    for col, color in zip(MAINCAT_COLS, colors):
        sub = annual[annual["maincat"].eq(col)].sort_values("year")
        complete = sub[sub["year"].le(2025)]
        partial = sub[sub["year"].eq(2026)]
        ax.plot(complete["year"], complete["record_level_prevalence_percent"], lw=1.5, color=color, label=MAINCAT_LABELS[col])
        if not partial.empty and not complete.empty:
            y2025 = float(complete.loc[complete["year"].eq(2025), "record_level_prevalence_percent"].iloc[0]) if (complete["year"].eq(2025)).any() else float(complete["record_level_prevalence_percent"].iloc[-1])
            y2026 = float(partial["record_level_prevalence_percent"].iloc[0])
            ax.plot([2025, 2026], [y2025, y2026], lw=1.5, ls="--", color=color)
            ax.scatter([2026], [y2026], s=20, facecolors="white", edgecolors=color, linewidths=1.0, zorder=5)
    ax.set_title("Major retraction metadata categories by year", loc="left", weight="bold", pad=6)
    ax.set_xlabel("Retraction year")
    ax.set_ylabel("Record-level prevalence among retractions (%)")
    ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2026])
    ax.set_xticklabels(["2000", "2005", "2010", "2015", "2020", "2026*"])
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note = (
        "Lines use final non-exclusive principal reason-domain assignments among Retraction records in 2000-2026. Dashed final segments and open markers denote partial 2026 data through retraction date 2026-05-06."
    )
    fig.text(0.075, 0.012, textwrap.fill(note, 112), fontsize=6.35, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.70, bottom=0.22, top=0.90)
    save_all(fig, f"annual_domain_prevalence_{YEAR_LABEL}")


def make_contact_sheet(paths: list[Path]) -> None:
    fig, axes = plt.subplots(len(paths), 1, figsize=(7.5, 10.0))
    if len(paths) == 1:
        axes = [axes]
    for ax, p in zip(axes, paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(p.name, loc="left", fontsize=7.2, pad=2)
    fig.tight_layout(pad=0.5)
    fig.savefig(SELECTED_DIR / f"selected_maincat_figures_review_sheet_{YEAR_LABEL}.png", dpi=250, facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    ret, main_df = load_main_data()
    meta = write_source_tables(ret, main_df)
    total = int(meta["total"])

    count_dist = pd.read_csv(SOURCE_DIR / f"donut_count_distribution_{YEAR_LABEL}.source.csv")
    combos = pd.read_csv(SOURCE_DIR / f"donut_exact_combinations_{YEAR_LABEL}.source.csv")
    annual = pd.read_csv(SOURCE_DIR / f"annual_domain_prevalence_{YEAR_LABEL}.source.csv")

    plot_flowchart(meta, count_dist)
    plot_cooccurrence(count_dist, combos, total)
    plot_annual_prevalence(annual)
    print(f"Wrote selected 2000-2026 maincat figures and source data; N={total:,}, partial 2026={meta['n_2026']:,}, cutoff={meta['latest_date']}.")


if __name__ == "__main__":
    main()
