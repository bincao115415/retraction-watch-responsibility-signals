#!/usr/bin/env python3
"""Generate a textless transparent 600 ppi 2000-2026 donut/pie element for PPT.

This uses the frozen 2000-2026 count-distribution source data and does not
change taxonomy, record-level assignments, or the canonical source tables.
Current presentation styling: no title, no caption, no labels, no center text,
transparent background, and only the donut chart itself.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CODE_ROOT = Path("/Users/Shared/Claude Code/retraction-watch-responsibility-signals")
PKG_ROOT = CODE_ROOT / "figures" / "manuscript-freeze-2026-05-20"
PKG_FIG = PKG_ROOT / "figures"
PKG_SRC = PKG_ROOT / "source-data"
FIG_ROOT = CODE_ROOT / "figures" / "v0.3-current-feasible-analyses"
MANIFEST = PKG_ROOT / "manifest.csv"
YEAR_LABEL = "2000_2026_asof_2026-05-06"
CUTOFF = "2026-05-06"
SCOPE = "included Retraction records, 2000-2026 partial"
N_EXPECTED = 64298

COUNT_CSV = PKG_SRC / f"donut_count_distribution_{YEAR_LABEL}.source.csv"
OUTPUT = PKG_FIG / f"figure_1_principal_domain_overlap_{YEAR_LABEL}_ppt_clean_600ppi.png"
OUTPUT_ALIAS = FIG_ROOT / f"figure_1_principal_domain_overlap_{YEAR_LABEL}_ppt_clean_600ppi.png"

# Five mutually exclusive donut groups. The four user-supplied colors are used
# for One/Two/Three/Four; None remains neutral so the user palette stays reserved
# for records with at least one assigned metadata-signal domain.
COLORS = {
    0: "#E6E6E6",  # neutral None
    1: "#A1A9D0",
    2: "#F0988C",
    3: "#F6CAE5",
    4: "#96CCCB",
}


def load_counts() -> pd.DataFrame:
    counts = pd.read_csv(COUNT_CSV).sort_values("n_maincats_final").reset_index(drop=True)
    if int(counts["n_records"].sum()) != N_EXPECTED:
        raise RuntimeError(f"Count distribution does not sum to {N_EXPECTED}")
    pct_sum = round(float(counts["percent_of_analysis_records"].sum()), 4)
    if pct_sum != 100.0:
        raise RuntimeError(f"Percentages sum to {pct_sum}, not 100.0")
    if counts["latest_retraction_date_in_csv"].astype(str).max() != CUTOFF:
        raise RuntimeError("Unexpected cutoff date in source data")
    return counts


def make_figure(counts: pd.DataFrame) -> None:
    values = counts["n_records"].astype(int).tolist()
    ks = counts["n_maincats_final"].astype(int).tolist()
    colors = [COLORS[k] for k in ks]

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_aspect("equal")
    ax.pie(
        values,
        startangle=105,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.2},
    )
    # Keep a small transparent margin so the annular edge and white separators are
    # not cropped when PowerPoint scales the PNG.
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUTPUT, dpi=600, transparent=True)
    fig.savefig(OUTPUT_ALIAS, dpi=600, transparent=True)
    plt.close(fig)


def update_manifest() -> None:
    row = {
        "artifact_type": "figure",
        "role": "PPT-clean 600 ppi 2000-2026 textless transparent donut figure",
        "package_path": str(OUTPUT),
        "source_path": str(OUTPUT_ALIAS),
        "analysis_scope": SCOPE,
        "data_cutoff_date": CUTOFF,
    }
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["package_path"] != str(OUTPUT)].copy()
    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest = manifest.sort_values(["artifact_type", "role", "package_path"]).reset_index(drop=True)
    manifest.to_csv(MANIFEST, index=False)


def main() -> None:
    counts = load_counts()
    make_figure(counts)
    update_manifest()
    print(f"Generated {OUTPUT}")
    print(f"Generated alias {OUTPUT_ALIAS}")
    print(f"N={N_EXPECTED}; percent_sum={counts['percent_of_analysis_records'].sum():.4f}; dpi=600; transparent=True; text_elements=0")
    print(counts[["n_maincats_final", "n_records", "percent_of_analysis_records"]].to_string(index=False))


if __name__ == "__main__":
    main()
