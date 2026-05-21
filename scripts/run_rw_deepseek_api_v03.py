#!/usr/bin/env python3
"""Run reproducible DeepSeek API semantic coding for Retraction Watch reasons.

No secrets are stored in outputs. The API key is read from DEEPSEEK_API_KEY or,
for Bin's local environment, from ~/.env.secrets if present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/Users/Shared/Hermes/workspaces/research/projects/publication/retractedpublications')
CONFIG_PATH = PROJECT / 'configs' / 'rw_v03_deepseek_api_config.json'
ITEMS_PATH = PROJECT / 'data' / 'rw-derived' / '2026-05-15-llm-semantic-v0.2' / 'unique_reason_items_for_llm.json'
OUT = PROJECT / 'data' / 'rw-derived' / '2026-05-15-llm-semantic-v0.3-api'
RAW_DIR = OUT / 'raw-api-responses'

PROBLEM_LABELS = {'data','image','results_conclusions','methods_materials','text_article','references_attribution','authorship_affiliation','ethics_welfare','legal_rights','article_general','availability_notice_status'}
PROCESS_LABELS = {'peer_review_compromised','peer_review_concern','journal_publisher_investigation','third_party_investigation','institutional_investigation','official_misconduct_process','journal_publisher_error','notice_opacity','removal_availability','editorial_breach','paper_mill','ai_computer_content','publication_policy_or_sanction','correction_update_process','legal_process','authorship_publication_process','research_ethics_approval_process','content_reuse_or_duplication_process'}
ACTOR_LABELS = {'author','editor','journal_publisher','company_institution','third_party','ori_government','unclear'}
CATEGORY_IDS = {'research_content_reliability','attribution_authorship_disclosure_integrity','editorial_peer_review_governance','post_publication_transparency_due_process_oversight'}
STRENGTHS = {'strict','broad','ambiguous'}

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
ALL_FLAGS = list(PROBLEM_TO_FLAG.values()) + list(PROCESS_TO_FLAG.values()) + list(ACTOR_TO_FLAG.values()) + list(CATEGORY_TO_FLAG.values())


def load_dotenv_secret() -> None:
    """Load DEEPSEEK_API_KEY from ~/.env.secrets without printing it."""
    if os.environ.get('DEEPSEEK_API_KEY'):
        return
    candidates = [Path.home() / '.env.secrets', Path('/Users/Bin/.env.secrets')]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k == 'DEEPSEEK_API_KEY' and v:
            os.environ[k] = v
            return


def json_sha256(obj) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def parse_year(s: str):
    m = re.search(r'(\d{4})', s or '')
    return int(m.group(1)) if m else None


def split_reasons(s: str):
    return [x.strip() for x in (s or '').split(';') if x.strip()]


def api_call(payload: dict, api_key: str, timeout: int) -> dict:
    req = urllib.request.Request(
        payload.pop('_url'),
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def extract_content(resp: dict) -> str:
    return resp['choices'][0]['message']['content']


def normalize_list_fields(obj: dict) -> dict:
    """Normalize allowed scalar labels to single-item lists; leave invalid values for validation.

    DeepSeek JSON mode guarantees valid JSON, not adherence to array typing. Scalar-to-list
    normalization is deterministic and recorded implicitly in the parsed output.
    """
    for key in ['problem_object','process_signal','actor_context','manuscript_categories']:
        if key not in obj:
            continue
        v = obj[key]
        if v is None:
            obj[key] = []
        elif isinstance(v, str):
            obj[key] = [] if v in {'', 'none', 'None', 'null'} else [v]
    return obj


def validate_item(obj: dict, expected_raw: str) -> list[str]:
    obj = normalize_list_fields(obj)
    errors = []
    required = ['raw_reason','canonical_reason','definition_used','problem_object','process_signal','actor_context','manuscript_categories','classification_strength','main_figure_recommended','caution','semantic_rationale']
    for k in required:
        if k not in obj:
            errors.append(f'missing {k}')
    if obj.get('raw_reason') != expected_raw:
        errors.append(f'raw_reason mismatch: {obj.get("raw_reason")} != {expected_raw}')
    for key in ['problem_object','process_signal','actor_context','manuscript_categories']:
        if key in obj and not isinstance(obj[key], list):
            errors.append(f'{key} not list')
    for v in obj.get('problem_object', []):
        if v not in PROBLEM_LABELS:
            errors.append(f'bad problem_object {v}')
    for v in obj.get('process_signal', []):
        if v not in PROCESS_LABELS:
            errors.append(f'bad process_signal {v}')
    for v in obj.get('actor_context', []):
        if v not in ACTOR_LABELS:
            errors.append(f'bad actor_context {v}')
    for v in obj.get('manuscript_categories', []):
        if v not in CATEGORY_IDS:
            errors.append(f'bad manuscript_category {v}')
    if obj.get('classification_strength') not in STRENGTHS:
        errors.append(f'bad classification_strength {obj.get("classification_strength")}')
    if not isinstance(obj.get('main_figure_recommended'), bool):
        errors.append('main_figure_recommended not bool')
    return errors


def make_payload(config: dict, item: dict) -> dict:
    api = config['api']
    user_prompt = config['user_prompt_template'].replace('{{ITEM_JSON}}', json.dumps(item, ensure_ascii=False, indent=2))
    return {
        '_url': api['base_url'],
        'model': api['model'],
        'messages': [
            {'role': 'system', 'content': config['system_prompt']},
            {'role': 'user', 'content': user_prompt},
        ],
        'thinking': api['thinking'],
        'temperature': api['temperature'],
        'top_p': api['top_p'],
        'response_format': api['response_format'],
        'stream': api['stream'],
        'max_tokens': api['max_tokens_per_item'],
    }


def run_api(args) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    items = json.loads(ITEMS_PATH.read_text(encoding='utf-8'))
    if args.limit:
        items = items[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'api_run_manifest.json'
    manifest = {
        'pipeline_version': config['pipeline_version'],
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_path': str(CONFIG_PATH),
        'items_path': str(ITEMS_PATH),
        'config_sha256': json_sha256(config),
        'items_sha256': json_sha256(items),
        'api_parameters': {k:v for k,v in config['api'].items() if k != 'base_url'},
        'api_base_url': config['api']['base_url'],
        'n_items_requested': len(items),
        'responses': []
    }
    if args.dry_run:
        payload = make_payload(config, items[0])
        payload_no_secret = {k:v for k,v in payload.items() if k != '_url'}
        (OUT / 'dry_run_first_payload.json').write_text(json.dumps(payload_no_secret, ensure_ascii=False, indent=2), encoding='utf-8')
        print('DRY_RUN wrote', OUT / 'dry_run_first_payload.json')
        return
    load_dotenv_secret()
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise SystemExit('Missing DEEPSEEK_API_KEY. Set it in env or ~/.env.secrets; do not commit it.')
    outputs = []
    for idx, item in enumerate(items, start=1):
        raw = item['raw_reason']
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{idx:03d}_{raw}')[:120]
        out_json = RAW_DIR / f'{safe}.json'
        parsed_json = RAW_DIR / f'{safe}.parsed.json'
        if parsed_json.exists() and not args.force:
            obj = json.loads(parsed_json.read_text(encoding='utf-8'))
            outputs.append(obj)
            manifest['responses'].append({'raw_reason': raw, 'status': 'cached', 'parsed_file': str(parsed_json)})
            continue
        payload = make_payload(config, item)
        last_error = None
        for attempt in range(1, config['api']['retry_attempts'] + 1):
            try:
                call_payload = dict(payload)
                resp = api_call(call_payload, api_key, config['api']['timeout_seconds'])
                out_json.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding='utf-8')
                content = extract_content(resp)
                obj = json.loads(content)
                errors = validate_item(obj, raw)
                if errors:
                    raise ValueError('; '.join(errors))
                parsed_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
                outputs.append(obj)
                usage = resp.get('usage', {})
                manifest['responses'].append({'raw_reason': raw, 'status': 'ok', 'parsed_file': str(parsed_json), 'response_file': str(out_json), 'usage': usage})
                print(f'OK {idx}/{len(items)} {raw}')
                break
            except Exception as e:
                last_error = repr(e)
                if attempt >= config['api']['retry_attempts']:
                    manifest['responses'].append({'raw_reason': raw, 'status': 'failed', 'error': last_error})
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
                    raise
                sleep_s = config['api']['retry_backoff_seconds'] * attempt + random.random()
                print(f'RETRY {idx}/{len(items)} {raw}: {last_error}; sleep {sleep_s:.1f}s', file=sys.stderr)
                time.sleep(sleep_s)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    taxonomy = {
        'pipeline_version': config['pipeline_version'],
        'model': config['api']['model'],
        'api_parameters': {k:v for k,v in config['api'].items() if k != 'base_url'},
        'config_sha256': manifest['config_sha256'],
        'items_sha256': manifest['items_sha256'],
        'classification_method': 'Direct DeepSeek API reason-level semantic classification using official Retraction Watch reason definitions, fixed label set, JSON output, temperature=0, top_p=1, thinking disabled, and validation checks.',
        'manuscript_categories': config['manuscript_categories'],
        'items': outputs,
    }
    (OUT / 'reason_semantic_taxonomy.json').write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'reason_semantic_taxonomy.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['raw_reason','canonical_reason','definition_used','problem_object','process_signal','actor_context','manuscript_categories','classification_strength','main_figure_recommended','caution','semantic_rationale']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in outputs:
            rr = row.copy()
            for key in ['problem_object','process_signal','actor_context','manuscript_categories']:
                rr[key] = ';'.join(rr.get(key, []))
            w.writerow({k: rr.get(k, '') for k in fields})
    print('WROTE', OUT / 'reason_semantic_taxonomy.json')


def build_records() -> None:
    tax_path = OUT / 'reason_semantic_taxonomy.json'
    if not tax_path.exists():
        raise SystemExit(f'Missing taxonomy: {tax_path}')
    taxonomy = json.loads(tax_path.read_text(encoding='utf-8'))
    by_raw = {x['raw_reason']: x for x in taxonomy['items']}
    rows = []
    with (PROJECT / 'retraction_watch.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            flags = {k: 0 for k in ALL_FLAGS}
            reasons = split_reasons(row['Reason'])
            cats, canons = [], []
            for raw in reasons:
                item = by_raw[raw]
                canons.append(item['canonical_reason'])
                for lab in item.get('problem_object', []):
                    flags[PROBLEM_TO_FLAG[lab]] = 1
                for lab in item.get('process_signal', []):
                    flags[PROCESS_TO_FLAG[lab]] = 1
                for lab in item.get('actor_context', []):
                    flags[ACTOR_TO_FLAG[lab]] = 1
                for lab in item.get('manuscript_categories', []):
                    flags[CATEGORY_TO_FLAG[lab]] = 1
                    cats.append(lab)
            rows.append({
                'record_id': row['Record ID'], 'title': row['Title'], 'journal': row['Journal'], 'publisher': row['Publisher'],
                'country': row['Country'], 'author': row['Author'], 'article_type': row['ArticleType'],
                'retraction_nature': row['RetractionNature'], 'retraction_date': row['RetractionDate'], 'retraction_year': parse_year(row['RetractionDate']),
                'original_paper_date': row['OriginalPaperDate'], 'original_year': parse_year(row['OriginalPaperDate']),
                'retraction_doi': row['RetractionDOI'], 'original_paper_doi': row['OriginalPaperDOI'], 'original_paper_pubmed_id': row['OriginalPaperPubMedID'],
                'raw_reason': row['Reason'], 'canonical_reasons': ';'.join(canons), 'manuscript_categories': ';'.join(sorted(set(cats))),
                'n_raw_reasons': len(reasons), 'n_manuscript_categories': len(set(cats)), **flags
            })
    record_path = OUT / 'record_level_binary.csv'
    with record_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    def prevalence(subset, label):
        out = []
        denom = len(subset)
        for flag in ALL_FLAGS:
            n = sum(int(r[flag]) for r in subset)
            out.append({'dataset': label, 'flag': flag, 'n': n, 'denominator': denom, 'percent': round(n/denom*100, 4) if denom else 0})
        return out
    ret = [r for r in rows if r['retraction_nature'] == 'Retraction']
    ret_2000_2025 = [r for r in ret if r['retraction_year'] and 2000 <= int(r['retraction_year']) <= 2025]
    prev = prevalence(rows, 'all_notices') + prevalence(ret, 'retractions_only_all_years') + prevalence(ret_2000_2025, 'retractions_2000_2025')
    with (OUT / 'prevalence_summary.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(prev[0].keys()))
        w.writeheader(); w.writerows(prev)
    report = {
        'pipeline_version': taxonomy['pipeline_version'],
        'n_reason_items': len(taxonomy['items']),
        'n_records_all': len(rows),
        'n_retractions_all_years': len(ret),
        'n_retractions_2000_2025': len(ret_2000_2025),
        'category_definitions': taxonomy['manuscript_categories'],
        'guardrails': [
            'Retraction Watch Reason terms are curated, non-exclusive metadata signals, not adjudicated fault.',
            'The four manuscript categories are secondary semantic aggregations created for figure readability, not original Retraction Watch fields.',
            'A record can map to multiple categories; percentages are record-level prevalence and may sum above 100%.',
            'Investigation-by-X remains a process/source signal, not fault by X.'
        ]
    }
    (OUT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WROTE', record_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='write first request payload without calling API')
    ap.add_argument('--limit', type=int, default=0, help='process first N reason items only')
    ap.add_argument('--force', action='store_true', help='rerun even if parsed outputs exist')
    ap.add_argument('--build-records', action='store_true', help='build record-level tables from existing taxonomy')
    args = ap.parse_args()
    if args.build_records:
        build_records()
    else:
        run_api(args)

if __name__ == '__main__':
    main()
