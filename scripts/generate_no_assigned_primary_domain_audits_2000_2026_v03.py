#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / 'data' / 'final' / 'record_level_binary.csv'
CODEBOOK_PATH = ROOT / 'figures' / 'v0.3-current-feasible-analyses' / 'source-data' / 'reason_domain_codebook_audit_109_reasons_2000_2026_asof_2026-05-06.source.csv'
OUT_DIR = ROOT / 'figures' / 'v0.3-current-feasible-analyses' / 'source-data'
CUTOFF = pd.Timestamp('2026-05-06')
CUTOFF_LABEL = '2026-05-06'
START_YEAR = 2000
END_YEAR = 2026

MAINCAT_COLS = [
    'maincat_research_content_reliability',
    'maincat_attribution_authorship_disclosure_integrity',
    'maincat_editorial_peer_review_governance',
    'maincat_post_publication_transparency_due_process_oversight',
]

NOTICE_AND_PROCESS_FLAG_COLS = [
    'gov_notice_opacity',
    'gov_removal_availability',
    'gov_correction_update_process',
    'gov_journal_publisher_investigation',
    'third_party_investigation',
    'institutional_investigation',
    'official_misconduct_process',
]

INVESTIGATION_COLS = [
    'gov_journal_publisher_investigation',
    'third_party_investigation',
    'institutional_investigation',
    'official_misconduct_process',
]

GROUP_DEFINITION = (
    'No assigned primary domain = no positive assignment across the four final primary reason-domain fields '
    '(maincat_research_content_reliability, '
    'maincat_attribution_authorship_disclosure_integrity, '
    'maincat_editorial_peer_review_governance, '
    'maincat_post_publication_transparency_due_process_oversight).'
)

INTERPRETATION_LIMIT = (
    'No assigned primary domain is not evidence of no problem and does not mean absence of '
    'Retraction Watch metadata; records may still carry contextual, notice-status, removal, '
    'correction/update, or investigation-process signals without a manuscript-facing primary '
    'domain assignment.'
)


def load_record_data() -> pd.DataFrame:
    usecols = [
        'record_id',
        'retraction_nature',
        'retraction_date',
        'retraction_year',
        'raw_reason',
        'canonical_reasons',
        'n_raw_reasons',
        *MAINCAT_COLS,
        *NOTICE_AND_PROCESS_FLAG_COLS,
    ]
    df = pd.read_csv(RECORD_PATH, usecols=usecols, low_memory=False)
    df = df[df['retraction_nature'].eq('Retraction')].copy()
    df['retraction_date'] = pd.to_datetime(df['retraction_date'], errors='coerce')
    df['retraction_year'] = pd.to_numeric(df['retraction_year'], errors='coerce').astype('Int64')
    year_mask = df['retraction_year'].between(START_YEAR, END_YEAR, inclusive='both').fillna(False)
    date_mask = df['retraction_date'].le(CUTOFF).fillna(False)
    df = df[np.logical_and(year_mask, date_mask)].copy()

    for col in MAINCAT_COLS + NOTICE_AND_PROCESS_FLAG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).clip(0, 1)

    df['n_raw_reasons'] = pd.to_numeric(df['n_raw_reasons'], errors='coerce')
    df['n_assigned_primary_domains'] = df[MAINCAT_COLS].sum(axis=1)
    return df


def normalize_reason_pairs(raw_reason: str, canonical_reasons: str) -> list[tuple[str, str]]:
    raw_list = [] if pd.isna(raw_reason) else [item.strip() for item in str(raw_reason).split(';') if item.strip()]
    canonical_list = [] if pd.isna(canonical_reasons) else [item.strip() for item in str(canonical_reasons).split(';') if item.strip()]
    if len(raw_list) != len(canonical_list):
        raise ValueError(
            f'raw/canonical reason length mismatch: raw={len(raw_list)} canonical={len(canonical_list)} '
            f'for raw_reason={raw_reason!r} canonical_reasons={canonical_reasons!r}'
        )
    return sorted(zip(raw_list, canonical_list), key=lambda pair: (pair[1], pair[0]))


def explode_reason_pairs(no_assigned_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in no_assigned_df.itertuples(index=False):
        pairs = normalize_reason_pairs(row.raw_reason, row.canonical_reasons)
        for raw_reason, canonical_reason in pairs:
            rows.append({
                'record_id': row.record_id,
                'raw_reason': raw_reason,
                'canonical_reason': canonical_reason,
            })
    exploded = pd.DataFrame(rows)
    if exploded.empty:
        raise ValueError('No reason pairs found in the no-assigned-primary-domain subset.')
    return exploded


def build_top_reasons_table(
    exploded_reasons: pd.DataFrame,
    no_assigned_records: int,
    analysis_records: int,
    codebook: pd.DataFrame,
) -> pd.DataFrame:
    top_reasons = (
        exploded_reasons.groupby('canonical_reason', as_index=False)
        .agg(
            record_count=('record_id', 'nunique'),
            reason_occurrence_count=('record_id', 'size'),
            raw_reason_variant_count=('raw_reason', 'nunique'),
            raw_reason_variants_pipe=('raw_reason', lambda s: '|'.join(sorted(pd.unique(s)))),
        )
        .sort_values(['record_count', 'reason_occurrence_count', 'canonical_reason'], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    top_reasons.insert(0, 'reason_rank_within_no_assigned_group', np.arange(1, len(top_reasons) + 1))
    top_reasons['percent_of_no_assigned_records'] = (top_reasons['record_count'] / no_assigned_records * 100).round(4)
    top_reasons['percent_of_analysis_records'] = (top_reasons['record_count'] / analysis_records * 100).round(4)

    audit_cols = [
        'reason_label',
        'classification_strength',
        'principal_reason_domain_scope',
        'manuscript_categories_pipe',
        'process_signal_pipe',
        'p_domain_subtype',
        'caution',
    ]
    top_reasons = top_reasons.merge(
        codebook[audit_cols].rename(columns={'reason_label': 'canonical_reason'}),
        on='canonical_reason',
        how='left',
    )

    top_reasons['assigned_primary_domain_count'] = 0
    top_reasons['no_assigned_records'] = no_assigned_records
    top_reasons['analysis_records_2000_2026_including_partial_2026'] = analysis_records
    top_reasons['latest_retraction_date_in_csv'] = CUTOFF_LABEL
    top_reasons['data_cutoff_used_for_labels'] = CUTOFF_LABEL
    top_reasons['group_definition'] = GROUP_DEFINITION
    top_reasons['interpretation_limit'] = INTERPRETATION_LIMIT
    return top_reasons


def build_reason_combination_table(no_assigned_df: pd.DataFrame, no_assigned_records: int, analysis_records: int) -> pd.DataFrame:
    combination_rows: list[dict[str, object]] = []
    for row in no_assigned_df.itertuples(index=False):
        pairs = normalize_reason_pairs(row.raw_reason, row.canonical_reasons)
        canonical_combo = ';'.join([pair[1] for pair in pairs])
        raw_combo = ';'.join([pair[0] for pair in pairs])
        combination_rows.append({
            'record_id': row.record_id,
            'canonical_reason_combination': canonical_combo,
            'raw_reason_combination': raw_combo,
            'n_reasons_in_combination': len(pairs),
        })

    combinations = pd.DataFrame(combination_rows)
    grouped = (
        combinations.groupby(['canonical_reason_combination', 'n_reasons_in_combination'], as_index=False)
        .agg(
            record_count=('record_id', 'nunique'),
            raw_reason_combination_variant_count=('raw_reason_combination', 'nunique'),
            raw_reason_combination_variants_pipe=('raw_reason_combination', lambda s: '|'.join(sorted(pd.unique(s)))),
        )
        .sort_values(['record_count', 'n_reasons_in_combination', 'canonical_reason_combination'], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    grouped.insert(0, 'combination_rank_within_no_assigned_group', np.arange(1, len(grouped) + 1))
    grouped['percent_of_no_assigned_records'] = (grouped['record_count'] / no_assigned_records * 100).round(4)
    grouped['percent_of_analysis_records'] = (grouped['record_count'] / analysis_records * 100).round(4)
    grouped['assigned_primary_domain_count'] = 0
    grouped['no_assigned_records'] = no_assigned_records
    grouped['analysis_records_2000_2026_including_partial_2026'] = analysis_records
    grouped['latest_retraction_date_in_csv'] = CUTOFF_LABEL
    grouped['data_cutoff_used_for_labels'] = CUTOFF_LABEL
    grouped['group_definition'] = GROUP_DEFINITION
    grouped['interpretation_limit'] = INTERPRETATION_LIMIT
    return grouped


def build_summary_table(
    no_assigned_df: pd.DataFrame,
    exploded_reasons: pd.DataFrame,
    codebook: pd.DataFrame,
    no_assigned_records: int,
    analysis_records: int,
) -> pd.DataFrame:
    n_raw_reasons = no_assigned_df['n_raw_reasons'].dropna()

    codebook_scopes = codebook[['reason_label', 'principal_reason_domain_scope', 'p_domain_subtype']].rename(
        columns={'reason_label': 'canonical_reason'}
    )
    reason_scopes = exploded_reasons.merge(codebook_scopes, on='canonical_reason', how='left')

    scope_records = (
        reason_scopes[['record_id', 'principal_reason_domain_scope']]
        .drop_duplicates()
        .groupby('principal_reason_domain_scope', as_index=False)
        .agg(record_count=('record_id', 'nunique'))
        .sort_values(['record_count', 'principal_reason_domain_scope'], ascending=[False, True])
        .reset_index(drop=True)
    )

    contextual_scope_label = 'retained_as_contextual_or_notice_status_metadata'
    contextual_subtypes = reason_scopes[reason_scopes['principal_reason_domain_scope'].eq(contextual_scope_label)].copy()
    subtype_records = (
        contextual_subtypes[['record_id', 'p_domain_subtype']]
        .dropna()
        .drop_duplicates()
        .groupby('p_domain_subtype', as_index=False)
        .agg(record_count=('record_id', 'nunique'))
        .sort_values(['record_count', 'p_domain_subtype'], ascending=[False, True])
        .reset_index(drop=True)
    )
    top_subtype = subtype_records.iloc[0] if not subtype_records.empty else None

    summary = pd.DataFrame([
        {
            'analysis_scope': 'RetractionNature == Retraction; retraction_year 2000-2026; 2026 partial included',
            'data_cutoff_used_for_labels': CUTOFF_LABEL,
            'latest_retraction_date_in_csv': CUTOFF_LABEL,
            'analysis_records_2000_2026_including_partial_2026': analysis_records,
            'assigned_primary_domain_count': 0,
            'group_definition': GROUP_DEFINITION,
            'no_assigned_records': no_assigned_records,
            'percent_of_analysis_records_no_assigned': round(no_assigned_records / analysis_records * 100, 4),
            'n_raw_reasons_mean': round(float(n_raw_reasons.mean()), 6),
            'n_raw_reasons_median': round(float(n_raw_reasons.median()), 6),
            'n_raw_reasons_q1': round(float(n_raw_reasons.quantile(0.25)), 6),
            'n_raw_reasons_q3': round(float(n_raw_reasons.quantile(0.75)), 6),
            'n_raw_reasons_min': int(n_raw_reasons.min()),
            'n_raw_reasons_max': int(n_raw_reasons.max()),
            'proportion_with_gov_notice_opacity': round(float(no_assigned_df['gov_notice_opacity'].mean() * 100), 4),
            'proportion_with_gov_removal_availability': round(float(no_assigned_df['gov_removal_availability'].mean() * 100), 4),
            'proportion_with_gov_correction_update_process': round(float(no_assigned_df['gov_correction_update_process'].mean() * 100), 4),
            'proportion_with_gov_journal_publisher_investigation': round(float(no_assigned_df['gov_journal_publisher_investigation'].mean() * 100), 4),
            'proportion_with_third_party_investigation': round(float(no_assigned_df['third_party_investigation'].mean() * 100), 4),
            'proportion_with_institutional_investigation': round(float(no_assigned_df['institutional_investigation'].mean() * 100), 4),
            'proportion_with_official_misconduct_process': round(float(no_assigned_df['official_misconduct_process'].mean() * 100), 4),
            'proportion_with_any_investigation_process': round(float(no_assigned_df[INVESTIGATION_COLS].max(axis=1).mean() * 100), 4),
            'records_with_any_retained_as_contextual_or_notice_status_metadata_scope': int(
                scope_records.loc[
                    scope_records['principal_reason_domain_scope'].eq(contextual_scope_label),
                    'record_count'
                ].sum()
            ),
            'share_with_any_retained_as_contextual_or_notice_status_metadata_scope': round(
                float(
                    scope_records.loc[
                        scope_records['principal_reason_domain_scope'].eq(contextual_scope_label),
                        'record_count'
                    ].sum() / no_assigned_records * 100
                ),
                4,
            ),
            'top_contextual_or_notice_status_scope_basis': 'p_domain_subtype among exploded canonical reasons after record-level deduplication',
            'top_contextual_or_notice_status_scope': None if top_subtype is None else top_subtype['p_domain_subtype'],
            'top_contextual_or_notice_status_scope_record_count': 0 if top_subtype is None else int(top_subtype['record_count']),
            'top_contextual_or_notice_status_scope_share_of_no_assigned_records': round(
                0.0 if top_subtype is None else float(top_subtype['record_count'] / no_assigned_records * 100),
                4,
            ),
            'interpretation_limit': INTERPRETATION_LIMIT,
        }
    ])
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_df = load_record_data()
    analysis_records = len(analysis_df)
    no_assigned_df = analysis_df[analysis_df['n_assigned_primary_domains'].eq(0)].copy()
    no_assigned_records = len(no_assigned_df)

    if analysis_records != 64298:
        raise ValueError(f'Expected 64298 analysis records, found {analysis_records}.')
    if no_assigned_records != 11086:
        raise ValueError(f'Expected 11086 no-assigned-primary-domain records, found {no_assigned_records}.')

    codebook = pd.read_csv(CODEBOOK_PATH)
    exploded_reasons = explode_reason_pairs(no_assigned_df)

    top_reasons = build_top_reasons_table(exploded_reasons, no_assigned_records, analysis_records, codebook)
    combinations = build_reason_combination_table(no_assigned_df, no_assigned_records, analysis_records)
    summary = build_summary_table(no_assigned_df, exploded_reasons, codebook, no_assigned_records, analysis_records)

    top_reason_path = OUT_DIR / 'no_assigned_primary_domain_top_reasons_2000_2026_asof_2026-05-06.source.csv'
    combination_path = OUT_DIR / 'no_assigned_primary_domain_reason_combinations_2000_2026_asof_2026-05-06.source.csv'
    summary_path = OUT_DIR / 'no_assigned_primary_domain_summary_2000_2026_asof_2026-05-06.source.csv'

    top_reasons.to_csv(top_reason_path, index=False)
    combinations.to_csv(combination_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f'analysis_records={analysis_records}')
    print(f'no_assigned_records={no_assigned_records}')
    print(f'wrote={top_reason_path}')
    print(f'wrote={combination_path}')
    print(f'wrote={summary_path}')


if __name__ == '__main__':
    main()
