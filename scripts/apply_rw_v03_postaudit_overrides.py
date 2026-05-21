#!/usr/bin/env python3
"""Apply post-Codex targeted overrides to Retraction Watch DeepSeek API v0.3 taxonomy.

This script preserves the raw API taxonomy and writes a separate final taxonomy plus
record-level tables. It adds two category flag layers:
- cat_*: all semantic manuscript-category assignments for audit/supplement.
- maincat_*: only assignments from reason items with main_figure_recommended=true;
  use these for main figures to avoid status-only/ambiguous labels inflating prevalence.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PROJECT = Path('/Users/Shared/Hermes/workspaces/research/projects/publication/retractedpublications')
BASE = PROJECT / 'data' / 'rw-derived' / '2026-05-15-llm-semantic-v0.3-api'
RAW_TAX = BASE / 'reason_semantic_taxonomy.json'
OVERRIDES = BASE / 'taxonomy_overrides.json'
FINAL_TAX = BASE / 'reason_semantic_taxonomy.json'

CATEGORY_IDS = [
    'research_content_reliability',
    'attribution_authorship_disclosure_integrity',
    'editorial_peer_review_governance',
    'post_publication_transparency_due_process_oversight',
]
CATEGORY_LABELS_FINAL = {
    'research_content_reliability': 'Research-content reliability',
    'attribution_authorship_disclosure_integrity': 'Attribution, authorship, disclosure and ethics integrity',
    'editorial_peer_review_governance': 'Editorial and peer-review governance',
    'post_publication_transparency_due_process_oversight': 'Post-publication process, transparency and oversight',
}
CATEGORY_EXPLANATIONS_FINAL = {
    'research_content_reliability': 'Signals that the reliability, reproducibility, validity, or evidentiary integrity of the research content itself is questioned. Includes data, image, methods/materials, analyses, results/conclusions, non-reproducibility, paper-mill and AI/computer-generated content signals when the official definition concerns article-content reliability.',
    'attribution_authorship_disclosure_integrity': 'Signals concerning authorship, affiliation, consent/approval to publish, plagiarism, duplicate publication, citation/reference/attribution, copyright/ownership, conflicts of interest, or research-ethics disclosure/approval. This is an integrity/relationship-transparency family, not a blame category.',
    'editorial_peer_review_governance': 'Signals concerning peer-review integrity, editorial decision-making, rogue/editorial breach, journal/publisher error, or governance of the pre-publication editorial process. Investigation-by-journal/publisher alone is not treated as editorial fault.',
    'post_publication_transparency_due_process_oversight': 'Signals concerning investigation source, official/institutional/third-party oversight, notice transparency, communication, objection, correction/retraction status, removal/availability, or legal/ORI/misconduct process. This category means post-publication process/oversight metadata, not due-process failure or fault by the named actor.',
}

LOW_LEVEL_PREFIXES = ('po_', 'gov_', 'third_party_investigation', 'institutional_investigation', 'official_misconduct_process', 'act_')

PROBLEM_LABELS = {'data','image','results_conclusions','methods_materials','text_article','references_attribution','authorship_affiliation','ethics_welfare','legal_rights','article_general','availability_notice_status'}
PROCESS_LABELS = {'peer_review_compromised','peer_review_concern','journal_publisher_investigation','third_party_investigation','institutional_investigation','official_misconduct_process','journal_publisher_error','notice_opacity','removal_availability','editorial_breach','paper_mill','ai_computer_content','publication_policy_or_sanction','correction_update_process','legal_process','authorship_publication_process','research_ethics_approval_process','content_reuse_or_duplication_process'}
ACTOR_LABELS = {'author','editor','journal_publisher','company_institution','third_party','ori_government','unclear'}
PROBLEM_TO_FLAG = {k:'po_'+k for k in PROBLEM_LABELS}
PROCESS_TO_FLAG = {
    'peer_review_compromised':'gov_peer_review_compromised',
    'peer_review_concern':'gov_peer_review_concern',
    'journal_publisher_investigation':'gov_journal_publisher_investigation',
    'third_party_investigation':'third_party_investigation',
    'institutional_investigation':'institutional_investigation',
    'official_misconduct_process':'official_misconduct_process',
    'journal_publisher_error':'gov_journal_publisher_error',
    'notice_opacity':'gov_notice_opacity',
    'removal_availability':'gov_removal_availability',
    'editorial_breach':'gov_editorial_breach_strict',
    'paper_mill':'gov_paper_mill',
    'ai_computer_content':'gov_ai_computer_content',
    'publication_policy_or_sanction':'gov_publication_policy_or_sanction',
    'correction_update_process':'gov_correction_update_process',
    'legal_process':'gov_legal_process',
    'authorship_publication_process':'gov_authorship_publication_process',
    'research_ethics_approval_process':'gov_research_ethics_approval_process',
    'content_reuse_or_duplication_process':'gov_content_reuse_or_duplication_process',
}
ACTOR_TO_FLAG = {k:'act_'+k for k in ACTOR_LABELS}
CATEGORY_TO_FLAG = {k:'cat_'+k for k in CATEGORY_IDS}
MAINCATEGORY_TO_FLAG = {k:'maincat_'+k for k in CATEGORY_IDS}
ALL_FLAGS = list(PROBLEM_TO_FLAG.values()) + list(PROCESS_TO_FLAG.values()) + list(ACTOR_TO_FLAG.values()) + list(CATEGORY_TO_FLAG.values()) + list(MAINCATEGORY_TO_FLAG.values())


def parse_year(s):
    m = re.search(r'(\d{4})', s or '')
    return int(m.group(1)) if m else None


def split_reasons(s):
    return [x.strip() for x in (s or '').split(';') if x.strip()]


def apply_overrides():
    tax = json.loads(RAW_TAX.read_text(encoding='utf-8'))
    ov = json.loads(OVERRIDES.read_text(encoding='utf-8'))
    by_raw = {x['raw_reason']: x for x in tax['items']}
    applied = []
    for raw, changes in ov['overrides'].items():
        if raw not in by_raw:
            raise SystemExit(f'override target missing: {raw}')
        item = by_raw[raw]
        before = {k: item.get(k) for k in changes}
        item.update(changes)
        item['post_codex_override_applied'] = True
        item['post_codex_override_version'] = ov['override_version']
        applied.append({'raw_reason': raw, 'before': before, 'after': changes})
    for item in tax['items']:
        item.setdefault('post_codex_override_applied', False)
    tax['pipeline_version'] = tax['pipeline_version'] + '_post_codex_final'
    tax['post_codex_override_version'] = ov['override_version']
    tax['post_codex_global_rules'] = ov['global_rules']
    tax['final_manuscript_category_labels'] = CATEGORY_LABELS_FINAL
    tax['final_manuscript_category_explanations'] = CATEGORY_EXPLANATIONS_FINAL
    tax['main_figure_rule'] = 'Use maincat_* flags generated only from reason items with main_figure_recommended=true. cat_* flags preserve broader semantic/audit category assignments.'
    FINAL_TAX.write_text(json.dumps(tax, ensure_ascii=False, indent=2), encoding='utf-8')
    (BASE / 'taxonomy_overrides_applied.json').write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding='utf-8')
    return tax, applied


def build_records(tax):
    by_raw = {x['raw_reason']: x for x in tax['items']}
    rows = []
    with (PROJECT / 'retraction_watch.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            flags = {k: 0 for k in ALL_FLAGS}
            reasons = split_reasons(row['Reason'])
            cats, maincats, canons = [], [], []
            for raw in reasons:
                item = by_raw[raw]
                canons.append(item['canonical_reason'])
                for lab in item.get('problem_object', []): flags[PROBLEM_TO_FLAG[lab]] = 1
                for lab in item.get('process_signal', []): flags[PROCESS_TO_FLAG[lab]] = 1
                for lab in item.get('actor_context', []): flags[ACTOR_TO_FLAG[lab]] = 1
                for lab in item.get('manuscript_categories', []):
                    flags[CATEGORY_TO_FLAG[lab]] = 1
                    cats.append(lab)
                    if item.get('main_figure_recommended') is True:
                        flags[MAINCATEGORY_TO_FLAG[lab]] = 1
                        maincats.append(lab)
            rows.append({
                'record_id': row['Record ID'], 'title': row['Title'], 'journal': row['Journal'], 'publisher': row['Publisher'], 'country': row['Country'], 'author': row['Author'], 'article_type': row['ArticleType'],
                'retraction_nature': row['RetractionNature'], 'retraction_date': row['RetractionDate'], 'retraction_year': parse_year(row['RetractionDate']), 'original_paper_date': row['OriginalPaperDate'], 'original_year': parse_year(row['OriginalPaperDate']),
                'retraction_doi': row['RetractionDOI'], 'original_paper_doi': row['OriginalPaperDOI'], 'original_paper_pubmed_id': row['OriginalPaperPubMedID'],
                'raw_reason': row['Reason'], 'canonical_reasons': ';'.join(canons), 'manuscript_categories': ';'.join(sorted(set(cats))), 'main_figure_categories': ';'.join(sorted(set(maincats))),
                'n_raw_reasons': len(reasons), 'n_manuscript_categories': len(set(cats)), 'n_main_figure_categories': len(set(maincats)), **flags
            })
    out = BASE / 'record_level_binary.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows, out


def prevalence(rows):
    ret = [r for r in rows if r['retraction_nature'] == 'Retraction']
    main = [r for r in ret if r['retraction_year'] and 2000 <= int(r['retraction_year']) <= 2025]
    subsets = [('all_notices', rows), ('retractions_only_all_years', ret), ('retractions_2000_2025', main)]
    out = []
    for label, subset in subsets:
        denom = len(subset)
        for flag in ALL_FLAGS:
            n = sum(int(r[flag]) for r in subset)
            out.append({'dataset': label, 'flag': flag, 'n': n, 'denominator': denom, 'percent': round(n / denom * 100, 4) if denom else 0})
    path = BASE / 'prevalence_summary.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    return out, path


def main():
    tax, applied = apply_overrides()
    rows, record_path = build_records(tax)
    prev, prev_path = prevalence(rows)
    ret = [r for r in rows if r['retraction_nature'] == 'Retraction']
    main_rows = [r for r in ret if r['retraction_year'] and 2000 <= int(r['retraction_year']) <= 2025]
    report = {
        'pipeline_version': tax['pipeline_version'],
        'raw_api_taxonomy': str(RAW_TAX),
        'final_taxonomy': str(FINAL_TAX),
        'overrides_path': str(OVERRIDES),
        'applied_overrides_count': len(applied),
        'n_reason_items': len(tax['items']),
        'n_records_all': len(rows),
        'n_retractions_all_years': len(ret),
        'n_retractions_2000_2025': len(main_rows),
        'category_labels_final': CATEGORY_LABELS_FINAL,
        'category_explanations_final': CATEGORY_EXPLANATIONS_FINAL,
        'main_figure_rule': tax['main_figure_rule'],
        'guardrails': tax.get('post_codex_global_rules', []),
        'main_category_prevalence_2000_2025': [r for r in prev if r['dataset']=='retractions_2000_2025' and r['flag'].startswith('maincat_')],
        'audit_category_prevalence_2000_2025': [r for r in prev if r['dataset']=='retractions_2000_2025' and r['flag'].startswith('cat_')],
    }
    (BASE / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WROTE', FINAL_TAX)
    print('WROTE', record_path)
    print('WROTE', prev_path)

if __name__ == '__main__':
    main()
