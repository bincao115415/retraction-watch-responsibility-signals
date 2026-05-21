#!/usr/bin/env python3
"""Apply accepted cdeep+Codex panel taxonomy-freeze decisions and regenerate artifacts.

This script keeps the raw API taxonomy immutable. It updates the auditable override
JSON, runs the existing post-audit override builder, copies the regenerated final
outputs into the code/artifact repo, and rebuilds manuscript-ready source-data
checks for the 2000-2026 partial Retraction subset.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

HERMES_ROOT = Path("/Users/Shared/Hermes/workspaces/research/projects/publication/retractedpublications")
CODE_ROOT = Path("/Users/Shared/Claude Code/retraction-watch-responsibility-signals")
BASE = HERMES_ROOT / "data" / "rw-derived" / "2026-05-15-llm-semantic-v0.3-api"
CODE_FINAL = CODE_ROOT / "data" / "final"
SOURCE_DIR = CODE_ROOT / "figures" / "v0.3-current-feasible-analyses" / "source-data"
SCRIPTS_DIR = CODE_ROOT / "scripts"
YEAR_LABEL = "2000_2026_asof_2026-05-06"
CUTOFF = "2026-05-06"
N_EXPECTED = 64298

CATEGORY_IDS = [
    "research_content_reliability",
    "attribution_authorship_disclosure_integrity",
    "editorial_peer_review_governance",
    "post_publication_transparency_due_process_oversight",
]
CATEGORY_DISPLAY = {
    "research_content_reliability": "Research-content reliability",
    "attribution_authorship_disclosure_integrity": "Authorship/attribution/disclosure integrity",
    "editorial_peer_review_governance": "Editorial and peer-review process integrity",
    "post_publication_transparency_due_process_oversight": "Post-publication process, notice transparency, and oversight",
}
ABBR = {
    "research_content_reliability": "R",
    "attribution_authorship_disclosure_integrity": "A",
    "editorial_peer_review_governance": "E",
    "post_publication_transparency_due_process_oversight": "P",
}
MAINCAT_COLS = {
    "R": "maincat_research_content_reliability",
    "A": "maincat_attribution_authorship_disclosure_integrity",
    "E": "maincat_editorial_peer_review_governance",
    "P": "maincat_post_publication_transparency_due_process_oversight",
}
CAT_COLS = {k: v.replace("maincat_", "cat_") for k, v in MAINCAT_COLS.items()}

ACCEPTED_FREEZE_CHANGES: dict[str, dict[str, Any]] = {
    "Computer-Aided Content or Computer-Generated Content": {
        "manuscript_categories": ["research_content_reliability"],
        "main_figure_recommended": True,
        "classification_strength": "broad",
        "caution": "Content-authenticity/reliability signal only for the principal analysis; do not infer authorship or disclosure failure without an explicit co-label.",
    },
    "Salami Slicing": {
        "manuscript_categories": ["research_content_reliability", "attribution_authorship_disclosure_integrity"],
        "main_figure_recommended": True,
        "classification_strength": "broad",
        "caution": "Content partitioning and publication-integrity signal; editorial/peer-review process assignment requires a separate editorial or peer-review reason label.",
    },
    "Duplication of Text": {
        "manuscript_categories": ["attribution_authorship_disclosure_integrity"],
        "main_figure_recommended": True,
        "classification_strength": "broad",
        "caution": "Text-only duplication is treated as attribution/publication-integrity metadata, not research-content reliability; data, image, and article-level duplication follow separate object rules.",
    },
}

STANDARD_CAVEATS: dict[str, str] = {
    "Investigation by Journal/Publisher": "Investigation source/locus signal only; not actor fault, oversight failure, or substantive cause. Included as post-publication process-source metadata.",
    "Investigation by Company/Institution": "Investigation source/locus signal only; not actor fault, oversight failure, or substantive cause. Included as post-publication process-source metadata.",
    "Investigation by ORI": "Investigation source/locus signal only; not actor fault, oversight failure, or substantive cause. Included as post-publication process-source metadata.",
    "Investigation by Third Party": "Investigation source/locus signal only; not actor fault, oversight failure, or substantive cause. Included as post-publication process-source metadata.",
    "Misconduct - Official Investigation(s) and/or Finding(s)": "Official investigation/finding signal; mechanism-specific R/A/E domains require separate data, image, authorship, or peer-review labels.",
    "Misconduct by Author": "Broad misconduct-process metadata; do not infer content, attribution, editorial mechanism, or actor responsibility unless a mechanism-specific co-label is present. Contextual-exclusion sensitivity is required.",
    "Misconduct by Company/Institution": "Broad misconduct-process metadata; do not infer content, attribution, editorial mechanism, or actor responsibility unless a mechanism-specific co-label is present. Contextual-exclusion sensitivity is required.",
    "Misconduct by Third Party": "Broad misconduct-process metadata; do not infer content, attribution, editorial mechanism, or actor responsibility unless a mechanism-specific co-label is present. Contextual-exclusion sensitivity is required.",
    "Miscommunication with/by Author": "Communication-process signal only; not misconduct, fault, or substantive retraction mechanism. Contextual-exclusion sensitivity is required.",
    "Miscommunication with/by Company/Institution": "Communication-process signal only; not misconduct, fault, or substantive retraction mechanism. Contextual-exclusion sensitivity is required.",
    "Miscommunication with/by Journal/Publisher": "Communication-process signal only; not misconduct, fault, or substantive retraction mechanism. Contextual-exclusion sensitivity is required.",
    "Miscommunication with/by Third Party": "Communication-process signal only; not misconduct, fault, or substantive retraction mechanism. Contextual-exclusion sensitivity is required.",
}

OBJECT_RULES = {
    "Plagiarism of Text": ["attribution_authorship_disclosure_integrity"],
    "Plagiarism of/in Article": ["attribution_authorship_disclosure_integrity", "research_content_reliability"],
    "Duplication of Data": ["research_content_reliability"],
    "Duplication of/in Image": ["research_content_reliability"],
    "Plagiarism of Data": ["research_content_reliability", "attribution_authorship_disclosure_integrity"],
    "Plagiarism of Image": ["research_content_reliability", "attribution_authorship_disclosure_integrity"],
}
GENERIC_MISCONDUCT_CANONICAL = {"Misconduct by Author", "Misconduct by Company/Institution", "Misconduct by Third Party"}
MISCOMMUNICATION_CANONICAL_PREFIX = "Miscommunication"
NOTICE_STATUS_RAW = {
    "Author Unresponsive",
    "Date of Article and/or Notice Unknown",
    "Notice - Limited or No Information",
}
NOTICE_PROCESS_SIGNALS = {"notice_opacity", "removal_availability", "correction_update_process"}
PROHIBITED_PATTERNS = [
    ("count bin", re.compile(r"count\s+bin", re.I)),
    ("bin 0", re.compile(r"bin\s*0", re.I)),
    ("0-4", re.compile(r"\b0\s*[-–]\s*4\b", re.I)),
    ("binary flag", re.compile(r"binary\s+flag", re.I)),
    ("one-hot", re.compile(r"one[-\s]?hot", re.I)),
    ("main_figure_recommended", re.compile(r"main_figure_recommended", re.I)),
    ("bin", re.compile(r"\bbin\b", re.I)),
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def pipe(values: list[str] | tuple[str, ...] | None) -> str:
    return "|".join(CATEGORY_DISPLAY.get(v, v) for v in (values or []))


def raw_split(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def sanitize_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    value = value.replace("Main figure recommendation is false", "Principal-analysis inclusion is false")
    value = value.replace("main figure recommendation is false", "principal-analysis inclusion is false")
    value = value.replace("Main figure recommendation", "Principal-analysis inclusion")
    value = value.replace("main figure recommendation", "principal-analysis inclusion")
    return value


def patch_override_json() -> None:
    override_path = BASE / "taxonomy_overrides.json"
    data = json.loads(override_path.read_text(encoding="utf-8"))
    data["override_version"] = "rw_llm_semantic_v0.3-api_post_codex_panel_taxonomy_freeze_2026-05-20"
    data["purpose"] = (
        "Apply targeted, auditable corrections after Codex methods audit and the accepted "
        "2026-05-20 cdeep+Codex panel taxonomy-freeze decisions. Raw API outputs remain preserved separately."
    )
    data["global_rules"] = [
        "Retraction Watch Reason terms remain non-exclusive curated metadata signals, not fault, liability, cause, or responsibility determinations.",
        "The four R/A/E/P domains are analyst-defined metadata signal domains for descriptive stratification.",
        "Investigation-by-X labels identify investigation source/locus, not actor fault, oversight failure, or substantive cause.",
        "Generic Misconduct-by-X and Miscommunication labels remain principal P-only in the main analysis for continuity but require contextual-exclusion sensitivity reporting.",
        "Notice/status labels can retain semantic P mapping but remain contextual/not principal; No assigned primary domain means no principal R/A/E/P assignment, not absence of Retraction Watch metadata.",
        "Object type governs plagiarism/duplication rules: text-only attribution, data/image evidence reliability, article-level R+A when appropriate.",
    ]
    overrides = data.setdefault("overrides", {})
    for raw, change in ACCEPTED_FREEZE_CHANGES.items():
        overrides[raw] = change
    for raw, caveat in STANDARD_CAVEATS.items():
        if raw in overrides:
            overrides[raw]["caution"] = caveat
    override_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    CODE_FINAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(override_path, CODE_FINAL / override_path.name)


def copy_final_outputs() -> None:
    for name in [
        "reason_semantic_taxonomy.json",
        "record_level_binary.csv",
        "prevalence_summary.csv",
        "validation_report.json",
        "taxonomy_overrides_applied.json",
    ]:
        shutil.copy2(BASE / name, CODE_FINAL / name)


def load_final_taxonomy() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    tax = json.loads((CODE_FINAL / "reason_semantic_taxonomy.json").read_text(encoding="utf-8"))
    by_raw = {item["raw_reason"]: item for item in tax["items"]}
    by_canonical = {item["canonical_reason"]: item for item in tax["items"]}
    return tax, by_raw, by_canonical


def load_records() -> pd.DataFrame:
    df = pd.read_csv(CODE_FINAL / "record_level_binary.csv", low_memory=False)
    df["retraction_year"] = pd.to_numeric(df["retraction_year"], errors="coerce").astype("Int64")
    df["retraction_date_dt"] = pd.to_datetime(df["retraction_date"], errors="coerce")
    for col in list(MAINCAT_COLS.values()) + list(CAT_COLS.values()):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["retraction_nature"].eq("Retraction") & df["retraction_year"].between(2000, 2026)].copy()
    if len(out) != N_EXPECTED:
        raise RuntimeError(f"Expected {N_EXPECTED} included Retraction records, found {len(out)}")
    latest = out["retraction_date_dt"].max().date().isoformat()
    if latest != CUTOFF:
        raise RuntimeError(f"Expected cutoff {CUTOFF}, found {latest}")
    return out


def annotate_final_record_table(df: pd.DataFrame) -> None:
    """Add explicit analysis-scope columns to the canonical full record table."""
    out = df.drop(columns=["retraction_date_dt"], errors="ignore").copy()
    mask = out["retraction_nature"].eq("Retraction") & out["retraction_year"].between(2000, 2026)
    out["in_analysis_subset_2000_2026_partial"] = mask
    out["analysis_scope_label"] = mask.map({
        True: "included Retraction record, 2000-2026 partial",
        False: "outside included 2000-2026 partial Retraction subset",
    })
    out["data_cutoff_date"] = CUTOFF
    out.to_csv(CODE_FINAL / "record_level_binary.csv", index=False)
    shutil.copy2(CODE_FINAL / "record_level_binary.csv", BASE / "record_level_binary.csv")


def observed_counts(main: pd.DataFrame) -> tuple[Counter[str], set[str]]:
    counts: Counter[str] = Counter()
    for reasons in main["canonical_reasons"].fillna(""):
        labels = set(raw_split(reasons))
        counts.update(labels)
    return counts, set(counts)


def display_domains_from_row(row: pd.Series) -> list[str]:
    return [abbr for abbr, col in MAINCAT_COLS.items() if int(row[col]) == 1]


def domain_display_from_abbr(abbrs: list[str]) -> str:
    if not abbrs:
        return "No assigned primary domain"
    return "+".join(abbrs)


def load_applied_overrides(by_raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items = json.loads((CODE_FINAL / "taxonomy_overrides_applied.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        raw = item["raw_reason"]
        canonical = by_raw[raw]["canonical_reason"]
        out[canonical] = item
    return out


def principal_scope(item_or_change: dict[str, Any]) -> str:
    return "included_in_principal_reason_domain_analysis" if item_or_change.get("main_figure_recommended") is True else "retained_as_contextual_or_notice_status_metadata"


def adjustment_note(before: dict[str, Any], after: dict[str, Any]) -> str:
    return (
        f"categories: {pipe(before.get('manuscript_categories')) or 'none'} -> {pipe(after.get('manuscript_categories')) or 'none'}; "
        f"principal_scope: {principal_scope(before)} -> {principal_scope(after)}; "
        f"classification_strength: {before.get('classification_strength')} -> {after.get('classification_strength')}"
    )


def build_taxonomy_source_tables(by_canonical: dict[str, dict[str, Any]], main: pd.DataFrame, applied: dict[str, dict[str, Any]]) -> pd.DataFrame:
    counts, observed = observed_counts(main)
    rows = []
    for label in sorted(observed):
        item = by_canonical[label]
        before = applied[label]["before"] if label in applied else item
        after = {
            "manuscript_categories": item.get("manuscript_categories", []),
            "main_figure_recommended": item.get("main_figure_recommended"),
            "classification_strength": item.get("classification_strength"),
        }
        row = {
            "reason_label": label,
            "observed_record_count": counts[label],
            "observed_record_percent_of_64298": round(counts[label] / N_EXPECTED * 100, 6),
            "definition_used": sanitize_text(item.get("definition_used", "")),
            "definition_type": item.get("definition_type", "RW definition / mapped label definition"),
            "classification_strength": item.get("classification_strength", ""),
            "manuscript_categories_pipe": pipe(item.get("manuscript_categories", [])),
            "principal_reason_domain_scope": principal_scope(item),
            "pre_audit_categories_pipe": pipe(before.get("manuscript_categories", [])),
            "pre_audit_principal_reason_domain_scope": principal_scope(before),
            "pre_audit_classification_strength": before.get("classification_strength", ""),
            "pre_audit_caution": sanitize_text(before.get("caution", "")),
            "post_audit_rule_adjusted": "yes" if label in applied else "no",
            "post_audit_rule_adjustment_source": "panel_taxonomy_freeze_2026-05-20" if label in applied and by_canonical[label]["raw_reason"] in ACCEPTED_FREEZE_CHANGES else ("post_codex_or_panel_override" if label in applied else ""),
            "post_audit_rule_adjustment": adjustment_note(before, after) if label in applied else "none",
            "p_domain_subtype": "",
            "process_signal_pipe": "|".join(item.get("process_signal", [])),
            "problem_object_pipe": "|".join(item.get("problem_object", [])),
            "actor_context_pipe": "|".join(item.get("actor_context", [])),
            "semantic_rationale": sanitize_text(item.get("semantic_rationale", "")),
            "caution": sanitize_text(item.get("caution", "")),
        }
        rows.append(row)
    codebook = pd.DataFrame(rows).sort_values("reason_label").reset_index(drop=True)
    if len(codebook) != 109:
        raise RuntimeError(f"Expected 109 observed labels, found {len(codebook)}")
    taxonomy_source = codebook.copy()
    taxonomy_source.to_csv(SOURCE_DIR / f"reason_semantic_taxonomy_final_109_reasons_{YEAR_LABEL}.source.csv", index=False)
    return codebook


def build_p_subtype(codebook: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in codebook.to_dict("records"):
        cats = set(str(row["manuscript_categories_pipe"]).split("|")) if row["manuscript_categories_pipe"] else set()
        process = set(str(row["process_signal_pipe"]).split("|")) if row["process_signal_pipe"] else set()
        has_p = CATEGORY_DISPLAY["post_publication_transparency_due_process_oversight"] in cats
        if not has_p:
            subtype = ""
            note = "Not applicable: reason not mapped to the P domain; p_domain_subtype is defined only for P-domain reasons."
        else:
            notice = bool(process & NOTICE_PROCESS_SIGNALS)
            non_notice = bool(process - NOTICE_PROCESS_SIGNALS)
            if notice and non_notice:
                subtype = "both"
                note = "Methodological annotation from both notice/status and process or oversight signals."
            elif notice:
                subtype = "notice_status"
                note = "Methodological annotation from notice/status/update process_signal only."
            else:
                subtype = "process_oversight"
                note = "Methodological annotation from non-notice process or oversight process_signal only."
        rows.append({
            "reason_label": row["reason_label"],
            "observed_record_count": row["observed_record_count"],
            "p_domain_subtype": subtype,
            "process_signal_pipe": row["process_signal_pipe"],
            "manuscript_categories_pipe": row["manuscript_categories_pipe"],
            "subtype_rule_note": note,
        })
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_DIR / f"p_domain_subtype_annotations_109_reasons_{YEAR_LABEL}.source.csv", index=False)
    codebook["p_domain_subtype"] = out.set_index("reason_label").loc[codebook["reason_label"], "p_domain_subtype"].to_list()
    codebook.to_csv(SOURCE_DIR / f"reason_domain_codebook_audit_109_reasons_{YEAR_LABEL}.source.csv", index=False)
    return out


def build_final_mapping_exports(main: pd.DataFrame, codebook: pd.DataFrame) -> None:
    """Export reader-facing record-level and record-by-reason mapping tables."""
    code_lookup = codebook.set_index("reason_label").to_dict("index")
    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for _, row in main.iterrows():
        labels = raw_split(row.get("canonical_reasons", ""))
        record_domains = display_domains_from_row(row)
        record_domain_label = domain_display_from_abbr(record_domains)
        base = {
            "record_id": row.get("record_id"),
            "title": row.get("title"),
            "journal": row.get("journal"),
            "publisher": row.get("publisher"),
            "country": row.get("country"),
            "article_type": row.get("article_type"),
            "retraction_date": row.get("retraction_date"),
            "retraction_year": row.get("retraction_year"),
            "original_paper_date": row.get("original_paper_date"),
            "original_year": row.get("original_year"),
            "retraction_doi": row.get("retraction_doi"),
            "original_paper_doi": row.get("original_paper_doi"),
            "canonical_reason_labels_pipe": "|".join(labels),
            "reason_label_count": len(labels),
            "principal_domains_pipe": record_domain_label,
            "principal_domain_count": len(record_domains),
            "R_principal_assigned": int("R" in record_domains),
            "A_principal_assigned": int("A" in record_domains),
            "E_principal_assigned": int("E" in record_domains),
            "P_principal_assigned": int("P" in record_domains),
            "analysis_scope": "included Retraction records, 2000-2026 partial",
            "partial_2026_included": True,
            "data_cutoff_date": CUTOFF,
            "interpretation_limit": "R/A/E/P are non-exclusive metadata signal domains, not cause, responsibility, fault, liability, or oversight-failure categories.",
        }
        wide_rows.append(base)
        for i, label in enumerate(labels, start=1):
            c = code_lookup.get(label, {})
            long_rows.append({
                "record_id": row.get("record_id"),
                "reason_order_within_record": i,
                "reason_label": label,
                "reason_principal_domains_pipe": c.get("manuscript_categories_pipe", "") or "No assigned primary domain",
                "reason_principal_scope": c.get("principal_reason_domain_scope", ""),
                "classification_strength": c.get("classification_strength", ""),
                "p_domain_subtype": c.get("p_domain_subtype", ""),
                "post_audit_rule_adjusted": c.get("post_audit_rule_adjusted", ""),
                "process_signal_pipe": c.get("process_signal_pipe", ""),
                "problem_object_pipe": c.get("problem_object_pipe", ""),
                "actor_context_pipe": c.get("actor_context_pipe", ""),
                "record_principal_domains_pipe": record_domain_label,
                "analysis_scope": "included Retraction records, 2000-2026 partial",
                "data_cutoff_date": CUTOFF,
            })
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    if len(wide) != N_EXPECTED:
        raise RuntimeError(f"Expected {N_EXPECTED} rows in record-level mapping export, found {len(wide)}")
    wide.to_csv(SOURCE_DIR / f"record_level_domain_assignments_{YEAR_LABEL}.source.csv", index=False)
    long.to_csv(SOURCE_DIR / f"record_reason_domain_mapping_{YEAR_LABEL}.source.csv", index=False)

    concise_cols = [
        "reason_label", "observed_record_count", "observed_record_percent_of_64298",
        "manuscript_categories_pipe", "principal_reason_domain_scope", "classification_strength",
        "p_domain_subtype", "post_audit_rule_adjusted", "post_audit_rule_adjustment_source",
        "semantic_rationale", "caution",
    ]
    concise = codebook[concise_cols].rename(columns={
        "observed_record_count": "n_records",
        "observed_record_percent_of_64298": "percent_of_64298_records",
        "manuscript_categories_pipe": "principal_domains_pipe",
    }).copy()
    concise["analysis_scope"] = "included Retraction records, 2000-2026 partial"
    concise["data_cutoff_date"] = CUTOFF
    concise.to_csv(SOURCE_DIR / f"final_reason_domain_mapping_109_reasons_{YEAR_LABEL}.source.csv", index=False)


def assignment_from_map(main: pd.DataFrame, label_map: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for labels in main["canonical_reasons"].fillna(""):
        cats = set()
        for label in raw_split(labels):
            cats.update(label_map.get(label, []))
        rows.append({abbr: int(cat in cats) for abbr, cat in zip(["R", "A", "E", "P"], CATEGORY_IDS)})
    return pd.DataFrame(rows, index=main.index)


def summarize(assign: pd.DataFrame) -> dict[str, tuple[int, float]]:
    denom = len(assign)
    n_domains = assign[["R", "A", "E", "P"]].sum(axis=1)
    out = {abbr: (int(assign[abbr].sum()), round(float(assign[abbr].sum()) / denom * 100, 4)) for abbr in ["R", "A", "E", "P"]}
    out["No assigned primary domain"] = (int((n_domains == 0).sum()), round(float((n_domains == 0).sum()) / denom * 100, 4))
    out["Two or more primary domains"] = (int((n_domains >= 2).sum()), round(float((n_domains >= 2).sum()) / denom * 100, 4))
    return out


def build_robustness(main: pd.DataFrame, codebook: pd.DataFrame, by_canonical: dict[str, dict[str, Any]], applied: dict[str, dict[str, Any]], p_subtype: pd.DataFrame) -> pd.DataFrame:
    code = codebook.set_index("reason_label").to_dict("index")
    p_lookup = p_subtype.set_index("reason_label")["p_domain_subtype"].to_dict()
    base = pd.DataFrame({abbr: main[col].astype(int).to_list() for abbr, col in MAINCAT_COLS.items()}, index=main.index)
    all_mapped = pd.DataFrame({abbr: main[col].astype(int).to_list() for abbr, col in CAT_COLS.items()}, index=main.index)
    def cats_for(label: str) -> list[str]:
        return list(by_canonical[label].get("manuscript_categories", [])) if by_canonical[label].get("main_figure_recommended") is True else []
    strict_map = {label: list(by_canonical[label].get("manuscript_categories", [])) for label, row in code.items() if row["principal_reason_domain_scope"] == "included_in_principal_reason_domain_analysis" and row["classification_strength"] == "strict"}
    strict_broad_map = {label: list(by_canonical[label].get("manuscript_categories", [])) for label, row in code.items() if row["principal_reason_domain_scope"] == "included_in_principal_reason_domain_analysis" and row["classification_strength"] in {"strict", "broad"}}
    pre_map = {}
    for label in code:
        if label in applied:
            before = applied[label]["before"]
            if before.get("main_figure_recommended") is True:
                pre_map[label] = list(before.get("manuscript_categories", []))
        else:
            pre_map[label] = cats_for(label)
    excluded = {label for label in code if label in GENERIC_MISCONDUCT_CANONICAL or label.startswith(MISCOMMUNICATION_CANONICAL_PREFIX)}
    contextual_excluded_map = {label: cats_for(label) for label in code if label not in excluded}
    p_restricted_map = {label: cats_for(label) for label in code}
    for label in list(p_restricted_map):
        if CATEGORY_IDS[3] in p_restricted_map[label] and p_lookup.get(label) == "notice_status":
            p_restricted_map[label] = [cat for cat in p_restricted_map[label] if cat != CATEGORY_IDS[3]]
    ambiguous = {label for label, row in code.items() if row["classification_strength"] == "ambiguous"}
    non_ambig_mask = ~main["canonical_reasons"].fillna("").map(lambda s: any(label in ambiguous for label in raw_split(s)))
    scenarios: list[tuple[str, str, pd.DataFrame, str]] = [
        ("principal_reason_domain_analysis", "Baseline final principal reason-domain fields", base, "Uses final principal R/A/E/P record-level assignments among Retraction records, 2000-2026 partial through 2026-05-06."),
        ("all_mapped_reason_domains", "All mapped reason domains", all_mapped, "Diagnostic only: uses all semantic R/A/E/P mappings, including contextual and notice-status metadata. This inflates P and should not be used as a principal domain-prevalence estimate."),
        ("principal_strict_only", "Reconstructed principal domains using strict labels only", assignment_from_map(main, strict_map), "Record-level OR assignment from canonical reasons; includes strict observed labels only."),
        ("principal_strict_or_broad", "Reconstructed principal domains using strict or broad labels", assignment_from_map(main, strict_broad_map), "Record-level OR assignment from canonical reasons; excludes ambiguous observed labels."),
        ("principal_excluding_records_with_ambiguous_labels", "Baseline principal domains after excluding records with any ambiguous observed label", base.loc[non_ambig_mask].reset_index(drop=True), f"Uses final principal assignments after excluding {int((~non_ambig_mask).sum())} records with at least one ambiguous observed label."),
        ("pre_audit_rule_adjustment", "Pre-audit and pre-panel rule-adjustment sensitivity using override before-state", assignment_from_map(main, pre_map), "Uses archived before-state for all override labels and final taxonomy for non-adjusted labels."),
        ("p_process_oversight_without_notice_status_only", "Final principal domains with P restricted to process-oversight or both subtypes", assignment_from_map(main, p_restricted_map), "R/A/E follow final principal assignments; P excludes pure notice-status P labels."),
        ("generic_misconduct_miscommunication_contextual_excluded", "Generic Misconduct-by-X and Miscommunication labels set to contextual only", assignment_from_map(main, contextual_excluded_map), "Sensitivity scenario: generic Misconduct-by-X and all Miscommunication labels are treated as contextual/principal=false; other labels retain final taxonomy assignments."),
    ]
    baseline = summarize(base)
    long_rows = []
    wide_rows = []
    measures = ["R", "A", "E", "P", "No assigned primary domain", "Two or more primary domains"]
    for scenario, label, assign, note in scenarios:
        summary = summarize(assign)
        denom = len(assign)
        wide = {"scenario": scenario, "scenario_label": label, "denominator_records": denom, "note": note}
        for measure in measures:
            n, pct = summary[measure]
            delta = round(pct - baseline[measure][1], 4)
            long_rows.append({"scenario": scenario, "scenario_label": label, "denominator_records": denom, "measure": measure, "n_records": n, "percent": pct, "delta_vs_principal_pp": delta, "note": note})
            key = measure if measure in {"R", "A", "E", "P"} else measure.lower().replace(" ", "_")
            wide[f"{key}_n"] = n
            wide[f"{key}_percent"] = pct
            wide[f"{key}_delta_vs_principal_pp"] = delta
        wide_rows.append(wide)
    long = pd.DataFrame(long_rows)
    wide = pd.DataFrame(wide_rows)
    long.to_csv(SOURCE_DIR / f"robustness_sensitivity_summary_{YEAR_LABEL}.source.csv", index=False)
    wide.to_csv(SOURCE_DIR / f"robustness_sensitivity_domain_matrix_{YEAR_LABEL}.source.csv", index=False)
    return long


def build_rule_adjustment_table(applied: dict[str, dict[str, Any]], by_canonical: dict[str, dict[str, Any]], main_counts: Counter[str]) -> None:
    rows = []
    for label, item in sorted(applied.items()):
        after_item = by_canonical[label]
        before = item["before"]
        after = {
            "manuscript_categories": after_item.get("manuscript_categories", []),
            "classification_strength": after_item.get("classification_strength"),
            "main_figure_recommended": after_item.get("main_figure_recommended"),
            "caution": after_item.get("caution", ""),
        }
        rows.append({
            "reason_label": label,
            "raw_reason": after_item["raw_reason"],
            "observed_record_count": main_counts.get(label, 0),
            "before_categories_pipe": pipe(before.get("manuscript_categories", [])),
            "after_categories_pipe": pipe(after.get("manuscript_categories", [])),
            "before_principal_scope": principal_scope(before),
            "after_principal_scope": principal_scope(after),
            "before_classification_strength": before.get("classification_strength", ""),
            "after_classification_strength": after.get("classification_strength", ""),
            "adjustment_note": adjustment_note(before, after),
            "caution": sanitize_text(after.get("caution", "")),
            "source": "cdeep_codex_panel_taxonomy_freeze_2026-05-20" if after_item["raw_reason"] in ACCEPTED_FREEZE_CHANGES else "post_codex_or_panel_standard_override",
        })
    out_path = SOURCE_DIR / f"rule_adjustment_before_after_18_reasons_{YEAR_LABEL}.source.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    old_path = SOURCE_DIR / f"rule_adjustment_before_after_15_reasons_{YEAR_LABEL}.source.csv"
    if old_path.exists():
        old_path.unlink()


def build_reconciliation_and_unobserved(df: pd.DataFrame, by_canonical: dict[str, dict[str, Any]], observed: set[str]) -> None:
    all_ret = df[df["retraction_nature"].eq("Retraction")]
    main = analysis_subset(df)
    rec = pd.DataFrame([
        {"step": "All source notice records", "n_records": len(df), "included_in_analysis": False, "note": "Full Retraction Watch source table."},
        {"step": "Retraction records, all years", "n_records": len(all_ret), "included_in_analysis": False, "note": "RetractionNature == Retraction."},
        {"step": "Retraction records before 2000 or after partial-2026 cutoff", "n_records": len(all_ret) - len(main), "included_in_analysis": False, "note": "Excluded from 2000-2026 partial analysis window."},
        {"step": "Included Retraction records, 2000-2026 partial", "n_records": len(main), "included_in_analysis": True, "note": f"2026 partial through RetractionDate {CUTOFF}."},
    ])
    rec.to_csv(SOURCE_DIR / f"denominator_reconciliation_{YEAR_LABEL}.source.csv", index=False)
    unobserved = sorted(set(by_canonical) - observed)
    pd.DataFrame([{"reason_label": label, "observed_in_2000_2026_retraction_subset": False} for label in unobserved]).to_csv(SOURCE_DIR / f"unobserved_source_codebook_labels_{YEAR_LABEL}.source.csv", index=False)


def write_prevalence_summary(main: pd.DataFrame) -> None:
    rows = []
    for dataset, subset in [
        ("retractions_2000_2026_including_partial_2026", main),
    ]:
        denom = len(subset)
        for abbr, col in MAINCAT_COLS.items():
            n = int(subset[col].sum())
            rows.append({"dataset": dataset, "measure": abbr, "flag": col, "n": n, "denominator": denom, "percent": round(n / denom * 100, 4), "partial_2026_included": True, "latest_retraction_date": CUTOFF})
        n_domains = subset[list(MAINCAT_COLS.values())].sum(axis=1)
        for measure, mask in [
            ("No assigned primary domain", n_domains.eq(0)),
            ("One assigned primary domain", n_domains.eq(1)),
            ("Two assigned primary domains", n_domains.eq(2)),
            ("Three assigned primary domains", n_domains.eq(3)),
            ("Four assigned primary domains", n_domains.eq(4)),
            ("Two or more primary domains", n_domains.ge(2)),
        ]:
            n = int(mask.sum())
            rows.append({"dataset": dataset, "measure": measure, "flag": "principal_domain_count", "n": n, "denominator": denom, "percent": round(n / denom * 100, 4), "partial_2026_included": True, "latest_retraction_date": CUTOFF})
    out = pd.DataFrame(rows)
    out.to_csv(CODE_FINAL / "prevalence_summary.csv", index=False)
    out.to_csv(SOURCE_DIR / f"prevalence_summary_{YEAR_LABEL}.source.csv", index=False)


def run_existing_generators() -> None:
    run([sys.executable, str(SCRIPTS_DIR / "generate_maincat_selected_figures_2000_2026_v03.py")], cwd=CODE_ROOT)
    run([sys.executable, str(SCRIPTS_DIR / "generate_no_assigned_primary_domain_audits_2000_2026_v03.py")], cwd=CODE_ROOT)
    # Remove stale vector exports for affected figures. The active deliverable policy is PNG plus source CSV.
    for name in [
        f"maincat_cooccurrence_combination_distribution_{YEAR_LABEL}",
        f"annual_domain_prevalence_{YEAR_LABEL}",
    ]:
        for ext in ["pdf", "svg"]:
            p = CODE_ROOT / "figures" / "v0.3-current-feasible-analyses" / f"{name}.{ext}"
            if p.exists():
                p.unlink()


def stage_final_figure_aliases() -> None:
    fig_dir = CODE_ROOT / "figures" / "v0.3-current-feasible-analyses"
    final_figs = {
        f"figure_1_principal_domain_overlap_{YEAR_LABEL}.png": fig_dir / f"maincat_cooccurrence_combination_distribution_{YEAR_LABEL}.png",
        f"extended_data_figure_1_annual_domain_prevalence_{YEAR_LABEL}.png": fig_dir / f"annual_domain_prevalence_{YEAR_LABEL}.png",
        f"supplementary_figure_1_reason_domain_derivation_{YEAR_LABEL}.png": fig_dir / "selected-main-figure-candidates" / f"maincat_mapped_layers_derivation_flowchart_{YEAR_LABEL}.png",
    }
    for alias, src in final_figs.items():
        if not src.exists():
            raise RuntimeError(f"Expected final figure source missing: {src}")
        shutil.copy2(src, fig_dir / alias)


def stage_manuscript_freeze_package() -> None:
    """Copy the final manuscript-facing files into a clean freeze package.

    The broader analysis directory intentionally retains exploratory/provenance files.
    This package prevents coauthors from mistaking older 2000-2025 artifacts for the
    final 2000-2026 partial manuscript set.
    """
    src_fig_dir = CODE_ROOT / "figures" / "v0.3-current-feasible-analyses"
    package = CODE_ROOT / "figures" / "manuscript-freeze-2026-05-20"
    package_fig = package / "figures"
    package_source = package / "source-data"
    package_final = package / "final-data"
    for d in [package_fig, package_source, package_final]:
        d.mkdir(parents=True, exist_ok=True)
    copy_plan: list[tuple[Path, Path, str, str]] = []
    for name, role in [
        (f"figure_1_principal_domain_overlap_{YEAR_LABEL}.png", "main quantitative figure"),
        (f"extended_data_figure_1_annual_domain_prevalence_{YEAR_LABEL}.png", "extended data / optional main figure"),
        (f"supplementary_figure_1_reason_domain_derivation_{YEAR_LABEL}.png", "supplementary methods figure"),
    ]:
        copy_plan.append((src_fig_dir / name, package_fig / name, "figure", role))
    source_names = [
        f"final_reason_domain_mapping_109_reasons_{YEAR_LABEL}.source.csv",
        f"record_level_domain_assignments_{YEAR_LABEL}.source.csv",
        f"record_reason_domain_mapping_{YEAR_LABEL}.source.csv",
        f"prevalence_summary_{YEAR_LABEL}.source.csv",
        f"robustness_sensitivity_summary_{YEAR_LABEL}.source.csv",
        f"robustness_sensitivity_domain_matrix_{YEAR_LABEL}.source.csv",
        f"donut_count_distribution_{YEAR_LABEL}.source.csv",
        f"donut_exact_combinations_{YEAR_LABEL}.source.csv",
        f"annual_domain_prevalence_{YEAR_LABEL}.source.csv",
        f"reason_domain_codebook_audit_109_reasons_{YEAR_LABEL}.source.csv",
        f"p_domain_subtype_annotations_109_reasons_{YEAR_LABEL}.source.csv",
        f"rule_adjustment_before_after_18_reasons_{YEAR_LABEL}.source.csv",
        f"denominator_reconciliation_{YEAR_LABEL}.source.csv",
        f"no_assigned_primary_domain_summary_{YEAR_LABEL}.source.csv",
        f"no_assigned_primary_domain_top_reasons_{YEAR_LABEL}.source.csv",
        f"no_assigned_primary_domain_reason_combinations_{YEAR_LABEL}.source.csv",
        f"crosswalk_consistency_check_{YEAR_LABEL}.source.csv",
        f"terminology_audit_log_{YEAR_LABEL}.source.csv",
        f"analysis_scope_summary_{YEAR_LABEL}.source.csv",
    ]
    for name in source_names:
        copy_plan.append((SOURCE_DIR / name, package_source / name, "source-data", "manuscript-ready support table"))
    final_names = [
        "reason_semantic_taxonomy.json",
        "record_level_binary.csv",
        "prevalence_summary.csv",
        "validation_report.json",
        "taxonomy_overrides_applied.json",
        "taxonomy_overrides.json",
    ]
    for name in final_names:
        copy_plan.append((CODE_FINAL / name, package_final / name, "final-data", "reproducible final data/audit artifact"))
    manifest_rows = []
    for src, dst, artifact_type, role in copy_plan:
        if not src.exists():
            raise RuntimeError(f"Expected package source missing: {src}")
        shutil.copy2(src, dst)
        manifest_rows.append({
            "artifact_type": artifact_type,
            "role": role,
            "package_path": str(dst),
            "source_path": str(src),
            "analysis_scope": "included Retraction records, 2000-2026 partial",
            "data_cutoff_date": CUTOFF,
        })
    pd.DataFrame(manifest_rows).to_csv(package / "manifest.csv", index=False)


def build_terminology_audit() -> pd.DataFrame:
    audit_path = SOURCE_DIR / f"terminology_audit_log_{YEAR_LABEL}.source.csv"
    scan_paths = [
        path for path in SOURCE_DIR.glob(f"*{YEAR_LABEL}*.source.csv")
        if path.name != audit_path.name
    ] + [
        CODE_ROOT / "figures" / "v0.3-current-feasible-analyses" / "selected-main-figure-candidates" / "supplementary_derivation_flowchart_2000_2026_asof_2026-05-06.html"
    ]
    rows = []
    for path in scan_paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PROHIBITED_PATTERNS:
                if pattern.search(line):
                    rows.append({
                        "file_path": str(path),
                        "line_number": i,
                        "matched_class": name,
                        "context": line[:500],
                        "severity": "must_fix",
                        "recommended_action": "Remove prohibited/internal manuscript-facing term from source-data or selected figure artifact.",
                    })
    audit = pd.DataFrame(rows, columns=["file_path", "line_number", "matched_class", "context", "severity", "recommended_action"])
    audit.to_csv(SOURCE_DIR / f"terminology_audit_log_{YEAR_LABEL}.source.csv", index=False)
    if not audit.empty:
        raise RuntimeError(f"Terminology audit found prohibited rows: {len(audit)}")
    return audit


def build_crosswalk_check(df: pd.DataFrame, main: pd.DataFrame, by_raw: dict[str, dict[str, Any]], by_canonical: dict[str, dict[str, Any]], codebook: pd.DataFrame, robust: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(check: str, observed: Any, expected: Any, note: str = "") -> None:
        rows.append({"check": check, "status": "PASS" if observed == expected else "FAIL", "observed": observed, "expected": expected, "note": note})
    files = {
        "codebook_109": SOURCE_DIR / f"reason_domain_codebook_audit_109_reasons_{YEAR_LABEL}.source.csv",
        "taxonomy_source": SOURCE_DIR / f"reason_semantic_taxonomy_final_109_reasons_{YEAR_LABEL}.source.csv",
        "concise_reason_mapping_109": SOURCE_DIR / f"final_reason_domain_mapping_109_reasons_{YEAR_LABEL}.source.csv",
        "record_level_mapping_64298": SOURCE_DIR / f"record_level_domain_assignments_{YEAR_LABEL}.source.csv",
        "record_reason_mapping": SOURCE_DIR / f"record_reason_domain_mapping_{YEAR_LABEL}.source.csv",
        "robust_long": SOURCE_DIR / f"robustness_sensitivity_summary_{YEAR_LABEL}.source.csv",
        "robust_wide": SOURCE_DIR / f"robustness_sensitivity_domain_matrix_{YEAR_LABEL}.source.csv",
        "crosswalk": SOURCE_DIR / f"crosswalk_consistency_check_{YEAR_LABEL}.source.csv",
        "prevalence": SOURCE_DIR / f"prevalence_summary_{YEAR_LABEL}.source.csv",
        "rule_adjustment_18": SOURCE_DIR / f"rule_adjustment_before_after_18_reasons_{YEAR_LABEL}.source.csv",
    }
    for label, path in files.items():
        if label != "crosswalk":
            add(f"file exists: {label}", path.exists(), True, str(path))
    add("included Retraction records 2000-2026 partial", len(main), N_EXPECTED)
    add("record-level mapping rows", len(pd.read_csv(SOURCE_DIR / f"record_level_domain_assignments_{YEAR_LABEL}.source.csv")), N_EXPECTED)
    record_table = pd.read_csv(CODE_FINAL / "record_level_binary.csv", usecols=["in_analysis_subset_2000_2026_partial"])
    add("canonical full record table analysis-scope flag sum", int(record_table["in_analysis_subset_2000_2026_partial"].sum()), N_EXPECTED)
    add("2026 partial record count", int((main["retraction_year"] == 2026).sum()), 675)
    add("latest RetractionDate in included subset", main["retraction_date_dt"].max().date().isoformat(), CUTOFF)
    counts = {abbr: int(main[col].sum()) for abbr, col in MAINCAT_COLS.items()}
    expected_counts = {"R": 41933, "A": 33954, "E": 25513, "P": 36359}
    for abbr in ["R", "A", "E", "P"]:
        add(f"principal {abbr} count", counts[abbr], expected_counts[abbr], "Denominator = 64,298 records.")
    n_domains = main[list(MAINCAT_COLS.values())].sum(axis=1)
    add("No assigned primary domain count", int((n_domains == 0).sum()), 11086, "Means no principal R/A/E/P assignment, not absence of RW metadata.")
    add("Two or more primary domains count", int((n_domains >= 2).sum()), 42662)
    add("observed reason labels in analysis subset", len(codebook), 109)
    strength_counts = codebook["classification_strength"].value_counts().to_dict()
    add("classification strength strict/broad/ambiguous", f"{strength_counts.get('strict',0)}/{strength_counts.get('broad',0)}/{strength_counts.get('ambiguous',0)}", "39/63/7")
    for raw, cats in ACCEPTED_FREEZE_CHANGES.items():
        item = by_raw[raw]
        add(f"panel change categories: {raw}", "+".join(ABBR[c] for c in item["manuscript_categories"]), "+".join(ABBR[c] for c in cats["manuscript_categories"]))
        add(f"panel change strength: {raw}", item["classification_strength"], "broad")
        add(f"panel change principal: {raw}", item["main_figure_recommended"], True)
    for raw, expected in OBJECT_RULES.items():
        item = by_raw[raw]
        add(f"object rule categories: {raw}", "+".join(ABBR[c] for c in item["manuscript_categories"]), "+".join(ABBR[c] for c in expected))
    for raw in ["Investigation by Journal/Publisher", "Investigation by Third Party", "Investigation by Company/Institution", "Investigation by ORI"]:
        item = by_raw[raw]
        add(f"Investigation-by-X P only strict principal: {raw}", (item["manuscript_categories"], item["classification_strength"], item["main_figure_recommended"]), ([CATEGORY_IDS[3]], "strict", True), "Caveat: investigation source/locus, not actor fault, oversight failure, or substantive cause.")
    official = by_raw["Misconduct - Official Investigation(s) and/or Finding(s)"]
    add("Official misconduct investigation/finding P only strict principal", (official["manuscript_categories"], official["classification_strength"], official["main_figure_recommended"]), ([CATEGORY_IDS[3]], "strict", True))
    for raw in ["Misconduct by Author", "Misconduct by Company/Institution", "Misconduct by Third Party", "Miscommunication with/by Author", "Miscommunication with/by Company/Institution", "Miscommunication with/by Journal/Publisher", "Miscommunication with/by Third Party"]:
        item = by_raw[raw]
        add(f"generic/miscommunication main analysis P only broad principal: {raw}", (item["manuscript_categories"], item["classification_strength"], item["main_figure_recommended"]), ([CATEGORY_IDS[3]], "broad", True))
    for raw in NOTICE_STATUS_RAW:
        item = by_raw[raw]
        add(f"notice/status contextual principal=false: {raw}", item["main_figure_recommended"], False)
    sens = robust[(robust["scenario"] == "generic_misconduct_miscommunication_contextual_excluded") & (robust["measure"] == "P")].iloc[0]
    add("sensitivity generic misconduct + miscommunication contextual-excluded P count", int(sens["n_records"]), 36139)
    add("sensitivity generic misconduct + miscommunication contextual-excluded P percent", round(float(sens["percent"]), 2), 56.21)
    add("terminology audit must_fix rows", 0, 0)
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_DIR / f"crosswalk_consistency_check_{YEAR_LABEL}.source.csv", index=False)
    if not out["status"].eq("PASS").all():
        raise RuntimeError("Crosswalk consistency check has FAIL rows:\n" + out[out["status"] != "PASS"].to_string(index=False))
    return out


def update_validation_report(df: pd.DataFrame, main: pd.DataFrame, robust: pd.DataFrame, crosswalk: pd.DataFrame, terminology: pd.DataFrame) -> None:
    baseline = robust[robust["scenario"].eq("principal_reason_domain_analysis")]
    report = {
        "pipeline_version": "rw_llm_semantic_v0.3_api_reproducible_post_codex_panel_taxonomy_freeze_2026-05-20",
        "n_records_all": int(len(df)),
        "n_retractions_all_years": int(df["retraction_nature"].eq("Retraction").sum()),
        "n_retractions_2000_2026_including_partial_2026": int(len(main)),
        "partial_2026_records": int((main["retraction_year"] == 2026).sum()),
        "latest_retraction_date_included": CUTOFF,
        "principal_domain_counts": {row["measure"]: {"n": int(row["n_records"]), "percent": float(row["percent"])} for _, row in baseline.iterrows()},
        "sensitivity_scenarios": sorted(robust["scenario"].unique().tolist()),
        "crosswalk_checks_passed": int(crosswalk["status"].eq("PASS").sum()),
        "crosswalk_checks_total": int(len(crosswalk)),
        "terminology_must_fix_rows": int(len(terminology)),
        "interpretation_guardrail": "R/A/E/P are metadata signal domains, not cause, responsibility, fault, liability, or oversight-failure categories.",
    }
    (CODE_FINAL / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ["prevalence_summary.csv", "validation_report.json"]:
        shutil.copy2(CODE_FINAL / name, BASE / name)


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    patch_override_json()
    run([sys.executable, str(SCRIPTS_DIR / "apply_rw_v03_postaudit_overrides.py")], cwd=CODE_ROOT)
    copy_final_outputs()
    tax, by_raw, by_canonical = load_final_taxonomy()
    df = load_records()
    main_df = analysis_subset(df)
    annotate_final_record_table(df)
    main_counts, observed = observed_counts(main_df)
    applied = load_applied_overrides(by_raw)
    codebook = build_taxonomy_source_tables(by_canonical, main_df, applied)
    p_subtype = build_p_subtype(codebook)
    build_final_mapping_exports(main_df, codebook)
    robust = build_robustness(main_df, codebook, by_canonical, applied, p_subtype)
    build_rule_adjustment_table(applied, by_canonical, main_counts)
    build_reconciliation_and_unobserved(df, by_canonical, observed)
    write_prevalence_summary(main_df)
    run_existing_generators()
    stage_final_figure_aliases()
    stage_manuscript_freeze_package()
    # Existing generator reads codebook before no-assigned outputs; rebuild codebook after source overwrites just in case.
    codebook = pd.read_csv(SOURCE_DIR / f"reason_domain_codebook_audit_109_reasons_{YEAR_LABEL}.source.csv")
    terminology = build_terminology_audit()
    crosswalk = build_crosswalk_check(df, main_df, by_raw, by_canonical, codebook, robust)
    update_validation_report(df, main_df, robust, crosswalk, terminology)
    print("DONE panel taxonomy freeze")
    print("N", len(main_df))
    print(robust[robust["scenario"].isin(["principal_reason_domain_analysis", "generic_misconduct_miscommunication_contextual_excluded"])].to_string(index=False))


if __name__ == "__main__":
    main()
