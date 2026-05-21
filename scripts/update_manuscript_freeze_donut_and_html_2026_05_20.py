#!/usr/bin/env python3
"""Post-freeze figure cleanup: donut Figure 1 + HTML-only derivation workflow.

This script intentionally does not change the taxonomy or record-level assignments.
It only updates manuscript-freeze figure packaging after user instruction:
- use HTML for the workflow/derivation process figure;
- remove generated workflow PNGs;
- replace Figure 1 with a donut-style metadata-domain count distribution with callouts.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

CODE_ROOT = Path("/Users/Shared/Claude Code/retraction-watch-responsibility-signals")
FIG_ROOT = CODE_ROOT / "figures" / "v0.3-current-feasible-analyses"
PKG_ROOT = CODE_ROOT / "figures" / "manuscript-freeze-2026-05-20"
PKG_FIG = PKG_ROOT / "figures"
PKG_SRC = PKG_ROOT / "source-data"
YEAR_LABEL = "2000_2026_asof_2026-05-06"
CUTOFF = "2026-05-06"
N_EXPECTED = 64298

COUNT_CSV = PKG_SRC / f"donut_count_distribution_{YEAR_LABEL}.source.csv"
COMBO_CSV = PKG_SRC / f"donut_exact_combinations_{YEAR_LABEL}.source.csv"
DONUT_SOURCE = PKG_SRC / f"figure_1_donut_domain_count_callouts_{YEAR_LABEL}.source.csv"
FIGURE1 = PKG_FIG / f"figure_1_principal_domain_overlap_{YEAR_LABEL}.png"
FIGURE1_SOURCE_ALIAS = FIG_ROOT / f"figure_1_principal_domain_overlap_{YEAR_LABEL}.png"
HTML_WORKFLOW = PKG_FIG / f"supplementary_figure_1_reason_domain_derivation_{YEAR_LABEL}.html"
HTML_WORKFLOW_SOURCE_ALIAS = FIG_ROOT / f"supplementary_figure_1_reason_domain_derivation_{YEAR_LABEL}.html"
MANIFEST = PKG_ROOT / "manifest.csv"

WORKFLOW_PNGS_TO_DELETE = [
    PKG_FIG / f"supplementary_figure_1_reason_domain_derivation_{YEAR_LABEL}.png",
    FIG_ROOT / f"supplementary_figure_1_reason_domain_derivation_{YEAR_LABEL}.png",
    FIG_ROOT / "selected-main-figure-candidates" / f"maincat_mapped_layers_derivation_flowchart_{YEAR_LABEL}.png",
]

COLORS = {
    "None": "#D9DEE7",
    "One": "#6B9FD5",
    "Two": "#49A078",
    "Three": "#D99A2B",
    "Four": "#C95F5F",
    "ink": "#1f2933",
    "muted": "#5d6673",
    "line": "#8a94a6",
}

GROUP_LABELS = {
    0: "None",
    1: "One domain",
    2: "Two domains",
    3: "Three domains",
    4: "Four domains",
}


def pct(x: float) -> str:
    return f"{x:.1f}%"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(COUNT_CSV)
    combos = pd.read_csv(COMBO_CSV)
    if int(counts["n_records"].sum()) != N_EXPECTED:
        raise RuntimeError(f"Count distribution does not sum to {N_EXPECTED}")
    if int(combos["n_records"].sum()) != N_EXPECTED:
        raise RuntimeError(f"Exact combinations do not sum to {N_EXPECTED}")
    return counts, combos


def build_callout_source(counts: pd.DataFrame, combos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, g in counts.sort_values("n_maincats_final").iterrows():
        k = int(g["n_maincats_final"])
        sub = combos[combos["n_maincats_final"].eq(k)].sort_values("n_records", ascending=False)
        if k == 0:
            composition = "No assigned primary domain"
        else:
            composition = "; ".join(
                f"{r.maincat_combination_short_label} ({float(r.percent_of_analysis_records):.1f}%)"
                for r in sub.itertuples(index=False)
            )
        rows.append({
            "assigned_metadata_signal_domain_count": k,
            "display_group": GROUP_LABELS[k],
            "n_records": int(g["n_records"]),
            "denominator": N_EXPECTED,
            "percent_of_records": float(g["percent_of_analysis_records"]),
            "composition_callout": composition,
            "composition_n_terms": len(sub),
            "partial_2026_included": True,
            "data_cutoff_date": CUTOFF,
            "interpretation_limit": "Counts are non-exclusive Retraction Watch metadata-signal domain assignments, not severity, causality, responsibility, or fault shares.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(DONUT_SOURCE, index=False)
    return out


def callout_lines(combos: pd.DataFrame, k: int) -> list[str]:
    sub = combos[combos["n_maincats_final"].eq(k)].sort_values("n_records", ascending=False)
    return [f"{r.maincat_combination_short_label} ({float(r.percent_of_analysis_records):.1f}%)" for r in sub.itertuples(index=False)]


def make_donut(counts: pd.DataFrame, combos: pd.DataFrame) -> None:
    counts = counts.sort_values("n_maincats_final")
    values = counts["n_records"].astype(int).tolist()
    labels = [GROUP_LABELS[int(k)] for k in counts["n_maincats_final"]]
    percentages = counts["percent_of_analysis_records"].astype(float).tolist()
    colors = [COLORS["None"], COLORS["One"], COLORS["Two"], COLORS["Three"], COLORS["Four"]]

    fig, ax = plt.subplots(figsize=(15.6, 9.2))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    wedges, _ = ax.pie(
        values,
        startangle=110,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 2.2},
    )

    ax.text(0, 0.06, "64,298", ha="center", va="center", fontsize=24, fontweight="bold", color=COLORS["ink"])
    ax.text(0, -0.10, "included\nRetraction records", ha="center", va="center", fontsize=10.5, color=COLORS["muted"], linespacing=1.15)

    # Direct labels around the donut.
    for i, wedge in enumerate(wedges):
        theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        x, y = 1.14 * math.cos(theta), 1.14 * math.sin(theta)
        ha = "left" if x >= 0 else "right"
        ax.text(
            x, y,
            f"{labels[i]}\n{values[i]:,} ({percentages[i]:.1f}%)",
            ha=ha, va="center", fontsize=9.4, color=COLORS["ink"], linespacing=1.15,
        )

    # Callout cards for one to four assigned domains.
    card_specs = {
        1: (1.70, 0.78, COLORS["One"]),
        2: (1.70, 0.30, COLORS["Two"]),
        3: (1.70, -0.25, COLORS["Three"]),
        # The four-domain segment sits on the left/lower side of the donut; placing
        # its callout on the same side avoids a leader line crossing the center label.
        4: (-2.82, -0.72, COLORS["Four"]),
    }
    wedge_by_k = {int(k): wedges[i] for i, k in enumerate(counts["n_maincats_final"].tolist())}
    count_by_k = {int(r.n_maincats_final): (int(r.n_records), float(r.percent_of_analysis_records)) for r in counts.itertuples(index=False)}
    for k, (x, y, color) in card_specs.items():
        n, group_pct = count_by_k[k]
        lines = callout_lines(combos, k)
        body = "\n".join(lines)
        h = 0.17 + 0.052 * len(lines)
        box = FancyBboxPatch(
            (x, y - h / 2), 1.24, h,
            boxstyle="round,pad=0.018,rounding_size=0.030",
            linewidth=1.15, edgecolor=color, facecolor="#FFFFFF", zorder=4,
        )
        ax.add_patch(box)
        ax.text(x + 0.045, y + h / 2 - 0.055, f"{k} assigned domain{'s' if k > 1 else ''}: {n:,} ({group_pct:.1f}%)", fontsize=10.0, fontweight="bold", color=COLORS["ink"], ha="left", va="top", zorder=5)
        ax.text(x + 0.045, y + h / 2 - 0.145, body, fontsize=8.4, color="#333333", ha="left", va="top", linespacing=1.18, zorder=5)
        wedge = wedge_by_k[k]
        theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        target_x = x if x >= 0 else x + 1.24
        con = ConnectionPatch(
            xyA=(1.02 * math.cos(theta), 1.02 * math.sin(theta)), coordsA=ax.transData,
            xyB=(target_x, y), coordsB=ax.transData,
            arrowstyle="-", shrinkA=0, shrinkB=0, linewidth=1.0, color=color, alpha=0.85, zorder=3,
        )
        ax.add_artist(con)

    ax.set_xlim(-3.05, 3.10)
    ax.set_ylim(-1.35, 1.34)
    ax.axis("off")
    fig.suptitle("Assigned metadata-signal domains among Retraction records", x=0.05, y=0.965, ha="left", fontsize=18, fontweight="bold", color=COLORS["ink"])
    fig.text(0.05, 0.915, "Donut segments show the number of assigned R/A/E/P metadata-signal domains per record; callouts decompose one-to-four-domain records into exact domain combinations.", ha="left", fontsize=10.5, color=COLORS["muted"])
    fig.text(0.05, 0.035, "R = research-content reliability; A = authorship/attribution/disclosure; E = editorial/peer-review process; P = post-publication process/notice transparency. Percentages use N = 64,298 included Retraction records; 2026 is partial through 2026-05-06. These are descriptive metadata mappings, not severity, causality, responsibility, or fault shares.", ha="left", va="bottom", fontsize=8.6, color="#444444")
    fig.savefig(FIGURE1, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE1_SOURCE_ALIAS, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_workflow_html() -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Supplementary Figure S1 — Retraction Watch metadata-signal derivation</title>
<style>
  :root {{ --ink:#1f2933; --muted:#5d6673; --line:#93a4b7; --blue:#2f5f8f; --panel:#f8fafc; --warn:#fff8e6; --warnline:#d6a843; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; background:#f3f5f8; color:var(--ink); font-family: \"Times New Roman\", Times, serif; }}
  .figure {{ max-width:1120px; margin:0 auto; background:white; border:1px solid #d8dee7; padding:26px 30px 22px; }}
  .kicker {{ font-size:14px; color:var(--blue); font-weight:700; letter-spacing:.03em; text-transform:uppercase; }}
  h1 {{ margin:4px 0 6px; font-size:25px; line-height:1.12; }}
  .subtitle {{ color:var(--muted); font-size:14px; margin-bottom:18px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px 30px; align-items:stretch; }}
  .panel-title {{ color:var(--blue); font-size:15px; font-weight:700; margin:0 0 10px; }}
  .box {{ border:1px solid #a8b4c2; border-radius:10px; background:#fff; padding:12px 14px; min-height:92px; }}
  .box.blue {{ background:#fbfdff; border-color:var(--blue); }}
  .box.warn {{ background:var(--warn); border-color:var(--warnline); }}
  .label {{ display:inline-block; background:var(--blue); color:white; padding:2px 7px; border-radius:999px; font-size:12px; font-weight:700; margin-right:7px; }}
  h2 {{ display:inline; font-size:16px; margin:0; }}
  p {{ margin:8px 0 0; font-size:14px; line-height:1.28; }}
  .arrow {{ text-align:center; color:var(--line); font-size:28px; line-height:1; margin:5px 0; }}
  .merge {{ display:grid; grid-template-columns:1fr 1fr; gap:30px; align-items:center; margin-top:14px; }}
  .merge .arrowline {{ height:1px; background:var(--line); position:relative; }}
  .merge .arrowline:after {{ content:\"\"; position:absolute; right:-2px; top:-4px; border-left:8px solid var(--line); border-top:4px solid transparent; border-bottom:4px solid transparent; }}
  .center {{ max-width:650px; margin:12px auto; }}
  .domain-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:10px; }}
  .domain {{ border:1px solid #d5dce5; border-radius:8px; padding:8px; font-size:13px; background:#fafafa; }}
  .domain strong {{ display:block; font-size:18px; color:var(--blue); }}
  .bars {{ display:grid; grid-template-columns:95px 1fr 95px; gap:8px; align-items:center; font-size:13px; margin-top:12px; }}
  .track {{ height:14px; background:#e8edf3; border-radius:999px; overflow:hidden; }}
  .fill {{ height:100%; background:#6b9fd5; }}
  .guardrails {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:16px; }}
  .caption {{ margin-top:14px; border-top:1px solid #d8dee7; padding-top:10px; font-size:12.5px; line-height:1.28; color:#394150; }}
</style>
</head>
<body>
<main class=\"figure\" role=\"img\" aria-label=\"PRISMA-style derivation of analysis sample and Retraction Watch metadata-signal domains\">
  <div class=\"kicker\">Supplementary Figure S1</div>
  <h1>Derivation of the analysis sample and metadata-signal domains</h1>
  <div class=\"subtitle\">Record filtering → observed reason labels → label-level codebook → record-level non-exclusive R/A/E/P metadata-signal assignments.</div>

  <div class=\"grid\">
    <section>
      <div class=\"panel-title\">Identification / eligibility</div>
      <div class=\"box blue\"><span class=\"label\">A1</span><h2>Retraction Watch source records</h2><p>All notice records in the source table: <strong>70,016</strong>. A record can list more than one Retraction Watch reason label.</p></div>
      <div class=\"arrow\">↓</div>
      <div class=\"box\"><span class=\"label\">A2</span><h2>Analysis subset</h2><p>Included Retraction records, 2000–2026*: <strong>64,298</strong>. The 2026 segment is partial through <strong>{CUTOFF}</strong>. Observed reason labels: <strong>109/112</strong>.</p></div>
    </section>
    <section>
      <div class=\"panel-title\">Reason-label classification</div>
      <div class=\"box blue\"><span class=\"label\">B1</span><h2>RW reason-label codebook</h2><p>The 112 unique Retraction Watch reason labels were classified once by recorded definition/meaning, not separately for every paper.</p></div>
      <div class=\"arrow\">↓</div>
      <div class=\"box\"><span class=\"label\">B2</span><h2>Four metadata-signal domains</h2><div class=\"domain-grid\"><div class=\"domain\"><strong>R</strong>content reliability</div><div class=\"domain\"><strong>A</strong>authorship / disclosure</div><div class=\"domain\"><strong>E</strong>editorial / peer review</div><div class=\"domain\"><strong>P</strong>post-publication process / notice transparency</div></div><p>These classify recorded reason-label metadata; they do not assign responsibility, intent, or fault.</p></div>
    </section>
  </div>

  <div class=\"center\">
    <div class=\"box blue\"><span class=\"label\">C</span><h2>Record-level assignment rule</h2><p>For each included record, a metadata-signal domain is assigned when at least one observed reason label on that record maps to that domain. Assignments are non-exclusive.</p></div>
  </div>

  <div class=\"grid\">
    <div class=\"box\"><span class=\"label\">D1</span><h2>Non-exclusive assignments</h2><p>Each record may be assigned to none, one, or several metadata-signal domains: <strong>R, A, E, P</strong>.</p></div>
    <div class=\"box\"><span class=\"label\">D2</span><h2>Number of assigned domains</h2><p>Records are summarized as no assigned domain, one domain, two domains, three domains, or four domains.</p></div>
  </div>

  <section class=\"box blue\" style=\"margin-top:16px\">
    <span class=\"label\">E</span><h2>Final count distribution among 64,298 included records</h2>
    <div class=\"bars\">
      <div>None</div><div class=\"track\"><div class=\"fill\" style=\"width:17.2%\"></div></div><div>11,086 (17.2%)</div>
      <div>One</div><div class=\"track\"><div class=\"fill\" style=\"width:16.4%\"></div></div><div>10,550 (16.4%)</div>
      <div>Two</div><div class=\"track\"><div class=\"fill\" style=\"width:28.3%\"></div></div><div>18,221 (28.3%)</div>
      <div>Three</div><div class=\"track\"><div class=\"fill\" style=\"width:10.9%\"></div></div><div>6,997 (10.9%)</div>
      <div>Four</div><div class=\"track\"><div class=\"fill\" style=\"width:27.1%\"></div></div><div>17,444 (27.1%)</div>
    </div>
    <p><strong>Two or more assigned domains:</strong> 42,662 / 64,298 = 66.4%.</p>
  </section>

  <div class=\"guardrails\">
    <div class=\"box warn\"><span class=\"label\">Guardrail</span><h2>Interpretation limit</h2><p>The number of assigned metadata-signal domains describes how broadly a record's Retraction Watch reason labels map across the four domains. It does not measure severity, causality, legal or moral fault, or responsibility shares.</p></div>
    <div class=\"box warn\"><span class=\"label\">Guardrail</span><h2>No assigned primary domain</h2><p>This group is not evidence of no problem and does not mean absence of Retraction Watch metadata. It usually reflects contextual, status-only, or non-assignable reason labels.</p></div>
  </div>

  <div class=\"caption\"><strong>Caption draft.</strong> Derivation of the analysis sample and final metadata-signal domain assignments from the Retraction Watch source table. Retraction notices with retraction dates from 2000 through the partial 2026 cutoff ({CUTOFF}; N = {N_EXPECTED:,}) were drawn from all Retraction Watch notice records (N = 70,016). Of 112 unique Retraction Watch reason labels in the source table, 109 were observed in the analysis subset and classified by recorded reason definition into four non-mutually-exclusive metadata-signal domains.</div>
</main>
</body>
</html>
"""
    HTML_WORKFLOW.write_text(html, encoding="utf-8")
    HTML_WORKFLOW_SOURCE_ALIAS.write_text(html, encoding="utf-8")


def delete_workflow_pngs() -> list[str]:
    deleted = []
    for path in WORKFLOW_PNGS_TO_DELETE:
        if path.exists():
            path.unlink()
            deleted.append(str(path))
    return deleted


def update_manifest() -> None:
    if MANIFEST.exists():
        manifest = pd.read_csv(MANIFEST)
    else:
        manifest = pd.DataFrame(columns=["artifact_type", "role", "package_path", "source_path", "analysis_scope", "data_cutoff_date"])
    # Remove deleted workflow PNG and stale duplicate rows for rewritten artifacts.
    remove_paths = {str(path) for path in WORKFLOW_PNGS_TO_DELETE}
    remove_paths.update({str(HTML_WORKFLOW), str(DONUT_SOURCE), str(FIGURE1)})
    manifest = manifest[~manifest["package_path"].isin(remove_paths)].copy()
    new_rows = [
        {
            "artifact_type": "figure",
            "role": "main quantitative donut figure with callouts",
            "package_path": str(FIGURE1),
            "source_path": str(FIGURE1_SOURCE_ALIAS),
            "analysis_scope": "included Retraction records, 2000-2026 partial",
            "data_cutoff_date": CUTOFF,
        },
        {
            "artifact_type": "workflow-html",
            "role": "supplementary derivation workflow, HTML only",
            "package_path": str(HTML_WORKFLOW),
            "source_path": str(HTML_WORKFLOW_SOURCE_ALIAS),
            "analysis_scope": "included Retraction records, 2000-2026 partial",
            "data_cutoff_date": CUTOFF,
        },
        {
            "artifact_type": "source-data",
            "role": "Figure 1 donut callout source data",
            "package_path": str(DONUT_SOURCE),
            "source_path": str(DONUT_SOURCE),
            "analysis_scope": "included Retraction records, 2000-2026 partial",
            "data_cutoff_date": CUTOFF,
        },
    ]
    manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    manifest = manifest.sort_values(["artifact_type", "role", "package_path"]).reset_index(drop=True)
    manifest.to_csv(MANIFEST, index=False)


def main() -> None:
    counts, combos = load_inputs()
    source = build_callout_source(counts, combos)
    make_donut(counts, combos)
    write_workflow_html()
    deleted = delete_workflow_pngs()
    update_manifest()
    print("DONE donut/html update")
    print("count_sum", int(counts["n_records"].sum()))
    print("percent_sum", round(float(counts["percent_of_analysis_records"].sum()), 6))
    print("donut_source", DONUT_SOURCE)
    print("figure1", FIGURE1)
    print("workflow_html", HTML_WORKFLOW)
    print("deleted_pngs", len(deleted))
    for path in deleted:
        print("deleted", path)
    print(source.to_string(index=False))


if __name__ == "__main__":
    main()
