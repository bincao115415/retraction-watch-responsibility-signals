#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "data" / "final" / "record_level_binary.csv"
TAXONOMY_PATH = ROOT / "data" / "final" / "reason_semantic_taxonomy.json"
OVERRIDES_PATH = ROOT / "data" / "final" / "taxonomy_overrides_applied.json"
AUDIT_PATH = ROOT / "figures" / "v0.3-current-feasible-analyses" / "source-data" / "reason_domain_codebook_audit_109_reasons_2000_2026_asof_2026-05-06.source.csv"
P_SUBTYPE_PATH = ROOT / "figures" / "v0.3-current-feasible-analyses" / "source-data" / "p_domain_subtype_annotations_109_reasons_2000_2026_asof_2026-05-06.source.csv"
OUT_DIR = ROOT / "figures" / "v0.3-current-feasible-analyses" / "source-data"
LONG_OUT = OUT_DIR / "robustness_sensitivity_summary_2000_2026_asof_2026-05-06.source.csv"
WIDE_OUT = OUT_DIR / "robustness_sensitivity_domain_matrix_2000_2026_asof_2026-05-06.source.csv"

YEAR_START = 2000
YEAR_END = 2026
CUTOFF = "2026-05-06"

DOMAIN_ORDER = [
    ("R", "research_content_reliability"),
    ("A", "attribution_authorship_disclosure_integrity"),
    ("E", "editorial_peer_review_governance"),
    ("P", "post_publication_transparency_due_process_oversight"),
]

MAINCAT_COLS = {
    "R": "maincat_research_content_reliability",
    "A": "maincat_attribution_authorship_disclosure_integrity",
    "E": "maincat_editorial_peer_review_governance",
    "P": "maincat_post_publication_transparency_due_process_oversight",
}

CAT_COLS = {
    "R": "cat_research_content_reliability",
    "A": "cat_attribution_authorship_disclosure_integrity",
    "E": "cat_editorial_peer_review_governance",
    "P": "cat_post_publication_transparency_due_process_oversight",
}

MEASURE_ORDER = [
    "R",
    "A",
    "E",
    "P",
    "No assigned primary domain",
    "Two or more primary domains",
]

SCENARIO_SPECS = [
    {
        "scenario": "principal_reason_domain_analysis",
        "scenario_label": "Baseline final principal reason-domain fields (existing maincat_* columns)",
        "note": "Uses existing final maincat_* record-level primary reason-domain assignments among Retraction records, 2000-2026 partial through 2026-05-06.",
    },
    {
        "scenario": "all_mapped_reason_domains",
        "scenario_label": "All mapped reason domains (existing cat_* columns)",
        "note": "Uses existing final cat_* record-level domain assignments, including mapped reason-domain categories beyond the principal main-domain rule.",
    },
    {
        "scenario": "principal_strict_only",
        "scenario_label": "Reconstructed principal domains using strict labels only",
        "note": "Record-level OR assignment from canonical_reasons; includes observed labels marked included_in_principal_reason_domain_analysis and classification_strength strict only; categories use the current final taxonomy.",
    },
    {
        "scenario": "principal_strict_or_broad",
        "scenario_label": "Reconstructed principal domains using strict or broad labels",
        "note": "Record-level OR assignment from canonical_reasons; includes observed labels marked included_in_principal_reason_domain_analysis and classification_strength strict or broad; ambiguous labels excluded; categories use the current final taxonomy.",
    },
    {
        "scenario": "principal_excluding_records_with_ambiguous_labels",
        "scenario_label": "Baseline principal domains after excluding records with any ambiguous observed label",
        "note": "Uses existing final maincat_* fields, but the denominator excludes any record containing at least one observed canonical reason label with classification_strength ambiguous.",
    },
    {
        "scenario": "pre_audit_rule_adjustment",
        "scenario_label": "Pre-audit rule-adjustment sensitivity using override before-state",
        "note": "Record-level OR assignment from canonical_reasons; for the 15 rule-adjusted labels uses the archived pre-audit before-state, for non-adjusted labels uses the final taxonomy; includes labels only when the pre-audit principal-analysis-scope rule would include them. This is a sensitivity scenario, not an objective source-truth benchmark.",
    },
    {
        "scenario": "p_process_oversight_without_notice_status_only",
        "scenario_label": "Final principal domains with P restricted to process-oversight or both subtypes",
        "note": "R/A/E stay at existing final principal maincat_* assignments. P is reconstructed from canonical_reasons using final principal-scope P labels whose p_domain_subtype is process_oversight or both; pure notice_status P labels are excluded from P assignment.",
    },
]


def load_records() -> pd.DataFrame:
    df = pd.read_csv(RECORD_PATH, low_memory=False)
    df["retraction_year"] = pd.to_numeric(df["retraction_year"], errors="coerce").astype("Int64")
    df["retraction_date"] = pd.to_datetime(df["retraction_date"], errors="coerce")
    df = df.loc[df["retraction_nature"].eq("Retraction")].copy()
    df = df.loc[df["retraction_year"].between(YEAR_START, YEAR_END)].copy()
    if len(df) != 64298:
        raise ValueError(f"Expected 64298 analysis records, found {len(df)}")
    if df["retraction_date"].max().date().isoformat() != CUTOFF:
        raise ValueError("RetractionDate cutoff mismatch for analysis subset")
    for col in [*MAINCAT_COLS.values(), *CAT_COLS.values()]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["canonical_reason_list"] = df["canonical_reasons"].fillna("").map(
        lambda s: [part.strip() for part in str(s).split(";") if part.strip()]
    )
    return df.reset_index(drop=True)


def load_audit() -> dict[str, dict[str, str]]:
    with AUDIT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {row["reason_label"]: row for row in csv.DictReader(f)}
    if len(rows) != 109:
        raise ValueError(f"Expected 109 observed reason labels in audit file, found {len(rows)}")
    return rows


def load_p_subtypes() -> dict[str, dict[str, str]]:
    with P_SUBTYPE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {row["reason_label"]: row for row in csv.DictReader(f)}
    return rows


def load_final_taxonomy() -> dict[str, dict[str, object]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)["items"]
    out = {item["canonical_reason"]: item for item in items}
    return out


def load_overrides() -> dict[str, dict[str, object]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        taxonomy_items = json.load(f)["items"]
    raw_to_canonical = {
        item["raw_reason"]: item["canonical_reason"] for item in taxonomy_items
    }
    with OVERRIDES_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    out: dict[str, dict[str, object]] = {}
    for item in items:
        raw_reason = item["raw_reason"]
        canonical_reason = raw_to_canonical.get(raw_reason, raw_reason)
        if canonical_reason in out:
            raise ValueError(f"Duplicate canonical override mapping for {canonical_reason}")
        out[canonical_reason] = item
    if len(out) != 15:
        raise ValueError(f"Expected 15 override labels, found {len(out)}")
    return out


def split_pipe(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part for part in text.split("|") if part]


def mapping_to_assignment_df(df: pd.DataFrame, label_to_categories: dict[str, list[str]]) -> pd.DataFrame:
    out = {abbr: [] for abbr, _ in DOMAIN_ORDER}
    for labels in df["canonical_reason_list"]:
        categories = set()
        for label in labels:
            categories.update(label_to_categories.get(label, []))
        for abbr, category in DOMAIN_ORDER:
            out[abbr].append(1 if category in categories else 0)
    return pd.DataFrame(out, index=df.index)


def summarize_assignment_df(assign_df: pd.DataFrame) -> dict[str, tuple[int, float]]:
    denominator = len(assign_df)
    if denominator == 0:
        raise ValueError("Scenario denominator is zero")
    n_domains = assign_df[[abbr for abbr, _ in DOMAIN_ORDER]].sum(axis=1)
    summary: dict[str, tuple[int, float]] = {}
    for abbr, _ in DOMAIN_ORDER:
        n_records = int(assign_df[abbr].sum())
        summary[abbr] = (n_records, round(n_records / denominator * 100, 4))
    no_assigned = int((n_domains == 0).sum())
    two_or_more = int((n_domains >= 2).sum())
    summary["No assigned primary domain"] = (no_assigned, round(no_assigned / denominator * 100, 4))
    summary["Two or more primary domains"] = (two_or_more, round(two_or_more / denominator * 100, 4))
    return summary


def build_scenarios(df: pd.DataFrame, audit: dict[str, dict[str, str]], p_subtypes: dict[str, dict[str, str]], final_taxonomy: dict[str, dict[str, object]], overrides: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    observed_labels = set(audit)
    record_labels = {label for labels in df["canonical_reason_list"] for label in labels}
    if record_labels != observed_labels:
        missing_from_audit = sorted(record_labels - observed_labels)
        missing_from_records = sorted(observed_labels - record_labels)
        raise ValueError(
            "Observed-label mismatch between records and audit file: "
            f"record_only={missing_from_audit} audit_only={missing_from_records}"
        )

    base_assign = pd.DataFrame({abbr: df[col].astype(int) for abbr, col in MAINCAT_COLS.items()}, index=df.index)
    all_mapped_assign = pd.DataFrame({abbr: df[col].astype(int) for abbr, col in CAT_COLS.items()}, index=df.index)

    strict_map = {
        label: split_pipe(row["manuscript_categories_pipe"])
        for label, row in audit.items()
        if row["principal_reason_domain_scope"] == "included_in_principal_reason_domain_analysis"
        and row["classification_strength"] == "strict"
    }
    strict_or_broad_map = {
        label: split_pipe(row["manuscript_categories_pipe"])
        for label, row in audit.items()
        if row["principal_reason_domain_scope"] == "included_in_principal_reason_domain_analysis"
        and row["classification_strength"] in {"strict", "broad"}
    }
    ambiguous_labels = {
        label for label, row in audit.items() if row["classification_strength"] == "ambiguous"
    }

    pre_audit_map: dict[str, list[str]] = {}
    for label in observed_labels:
        if label in overrides:
            before = overrides[label]["before"]
            if before["main_figure_recommended"]:
                pre_audit_map[label] = list(before["manuscript_categories"])
        else:
            item = final_taxonomy[label]
            if item["main_figure_recommended"]:
                pre_audit_map[label] = list(item["manuscript_categories"])

    p_restricted_labels = {
        label
        for label, row in audit.items()
        if row["principal_reason_domain_scope"] == "included_in_principal_reason_domain_analysis"
        and "Post-publication process, notice transparency, and oversight" in split_pipe(row["manuscript_categories_pipe"])
        and p_subtypes[label]["p_domain_subtype"] in {"process_oversight", "both"}
    }

    strict_assign = mapping_to_assignment_df(df, strict_map)
    strict_or_broad_assign = mapping_to_assignment_df(df, strict_or_broad_map)
    pre_audit_assign = mapping_to_assignment_df(df, pre_audit_map)

    non_ambiguous_mask = ~df["canonical_reason_list"].map(
        lambda labels: any(label in ambiguous_labels for label in labels)
    )
    non_ambiguous_df = df.loc[non_ambiguous_mask].copy().reset_index(drop=True)
    non_ambiguous_assign = pd.DataFrame(
        {abbr: non_ambiguous_df[col].astype(int) for abbr, col in MAINCAT_COLS.items()},
        index=non_ambiguous_df.index,
    )

    p_restricted_assign = base_assign.copy()
    p_restricted_assign["P"] = df["canonical_reason_list"].map(
        lambda labels: int(any(label in p_restricted_labels for label in labels))
    )

    scenarios = {
        "principal_reason_domain_analysis": {
            "denominator_records": len(df),
            "assignment_df": base_assign,
            "note": SCENARIO_SPECS[0]["note"],
        },
        "all_mapped_reason_domains": {
            "denominator_records": len(df),
            "assignment_df": all_mapped_assign,
            "note": SCENARIO_SPECS[1]["note"],
        },
        "principal_strict_only": {
            "denominator_records": len(df),
            "assignment_df": strict_assign,
            "note": SCENARIO_SPECS[2]["note"],
        },
        "principal_strict_or_broad": {
            "denominator_records": len(df),
            "assignment_df": strict_or_broad_assign,
            "note": SCENARIO_SPECS[3]["note"],
        },
        "principal_excluding_records_with_ambiguous_labels": {
            "denominator_records": len(non_ambiguous_df),
            "assignment_df": non_ambiguous_assign,
            "note": SCENARIO_SPECS[4]["note"] + f" Excluded {len(df) - len(non_ambiguous_df)} records with at least one ambiguous observed label.",
        },
        "pre_audit_rule_adjustment": {
            "denominator_records": len(df),
            "assignment_df": pre_audit_assign,
            "note": SCENARIO_SPECS[5]["note"],
        },
        "p_process_oversight_without_notice_status_only": {
            "denominator_records": len(df),
            "assignment_df": p_restricted_assign,
            "note": SCENARIO_SPECS[6]["note"],
        },
    }
    return scenarios


def build_output_tables(scenarios: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec_lookup = {spec["scenario"]: spec for spec in SCENARIO_SPECS}
    baseline_summary = summarize_assignment_df(scenarios["principal_reason_domain_analysis"]["assignment_df"])

    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []

    for spec in SCENARIO_SPECS:
        scenario = spec["scenario"]
        scenario_info = scenarios[scenario]
        assign_df = scenario_info["assignment_df"]
        denominator = int(scenario_info["denominator_records"])
        note = str(scenario_info["note"])
        summary = summarize_assignment_df(assign_df)

        wide_row: dict[str, object] = {
            "scenario": scenario,
            "scenario_label": spec_lookup[scenario]["scenario_label"],
            "denominator_records": denominator,
            "note": note,
        }

        for measure in MEASURE_ORDER:
            n_records, percent = summary[measure]
            baseline_percent = baseline_summary[measure][1]
            delta = round(percent - baseline_percent, 4)
            long_rows.append({
                "scenario": scenario,
                "scenario_label": spec_lookup[scenario]["scenario_label"],
                "denominator_records": denominator,
                "measure": measure,
                "n_records": n_records,
                "percent": percent,
                "delta_vs_principal_pp": delta,
                "note": note,
            })
            if measure in {"R", "A", "E", "P"}:
                prefix = measure
            elif measure == "No assigned primary domain":
                prefix = "no_assigned_primary_domain"
            else:
                prefix = "two_or_more_primary_domains"
            wide_row[f"{prefix}_n"] = n_records
            wide_row[f"{prefix}_percent"] = percent
            wide_row[f"{prefix}_delta_vs_principal_pp"] = delta

        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    return long_df, wide_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_records()
    audit = load_audit()
    p_subtypes = load_p_subtypes()
    final_taxonomy = load_final_taxonomy()
    overrides = load_overrides()
    scenarios = build_scenarios(df, audit, p_subtypes, final_taxonomy, overrides)
    long_df, wide_df = build_output_tables(scenarios)
    long_df.to_csv(LONG_OUT, index=False)
    wide_df.to_csv(WIDE_OUT, index=False)
    print(f"Wrote {LONG_OUT}")
    print(f"Wrote {WIDE_OUT}")
    print(f"Analysis records: {len(df)}")
    print(f"Long rows: {len(long_df)}")
    print(f"Wide rows: {len(wide_df)}")


if __name__ == "__main__":
    main()
