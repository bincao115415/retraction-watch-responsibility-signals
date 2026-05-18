# Retraction Watch responsibility-signal coding

Reproducible code for API-assisted semantic coding of Retraction Watch reason terms for a manuscript on journal/publisher responsibility, editorial governance, peer review, and post-publication oversight.

## What this repository does

The pipeline classifies unique Retraction Watch `Reason` terms, not individual papers, using official reason definitions and fixed DeepSeek API parameters. The reason-level taxonomy is then mapped back to the original Retraction Watch CSV as record-level, non-exclusive binary signals.

The code deliberately avoids using keyword matching as the primary classification method. The model receives one reason item at a time, a fixed label set, and explicit guardrails against assigning blame.

## Data included in this repository

Per the project owner's request, this repository includes both the source data used in the run and the final processed outputs:

```text
data/raw/retraction_watch.csv
    Original Retraction Watch CSV used for this analysis.

data/input/unique_reason_items_for_llm.json
    The 112 unique Retraction Watch reason items and official definitions supplied to the API coder.

data/provenance/
    Raw DeepSeek API run manifest, raw API taxonomy, preliminary v0.3 outputs, and archived raw API responses.

data/final/
    Post-Codex final taxonomy, final record-level binary table, final prevalence summary, targeted overrides, and validation report.
```

Large CSV files are tracked directly in Git because each file is below GitHub's hard 100 MB file limit. GitHub may still warn about files above 50 MB.

## Data scope

Raw Retraction Watch data are included in this repository for this project snapshot.

Expected input layout inside a local project directory:

```text
retraction_watch.csv
data/rw-derived/2026-05-15-llm-semantic-v0.2/unique_reason_items_for_llm.json
```

`unique_reason_items_for_llm.json` should contain all unique raw Retraction Watch reason terms and their official definitions.

## Model and API parameters

Final API run configuration:

- API base URL: `https://api.deepseek.com/chat/completions`
- Model: `deepseek-v4-pro`
- Thinking mode: disabled
- Temperature: `0`
- Top-p: `1`
- Response format: JSON object
- Stream: false
- Unit of classification: one unique Retraction Watch reason item

Rationale:

- `deepseek-v4-pro` is preferred over flash/deprecated aliases because the task is manuscript-support semantic coding where accuracy and definition-following matter more than speed.
- Thinking mode is disabled because DeepSeek documentation states thinking mode does not support `temperature`/`top_p` controls. Disabling it makes the low-temperature sampling setup active and reportable.
- `temperature=0` is used for controlled classification with a fixed label set.
- `top_p=1` is kept neutral because temperature is the active sampling control.
- JSON output plus validation checks are used for machine-readable, auditable outputs.

Important caveat: this is fixed-parameter API-assisted coding with archived prompts/outputs and validation, not guaranteed perfect determinism. Closed API models may change behind the same model name.

## Four manuscript-facing categories

The four categories are secondary semantic aggregations created for figure readability. They are not original Retraction Watch fields and do not assign legal, moral, causal, or fault responsibility.

1. Research-content reliability

   Signals that the reliability, reproducibility, validity, or evidentiary integrity of the research content itself is questioned. Includes data, image, methods/materials, analyses, results/conclusions, non-reproducibility, and article-content authenticity signals when supported by the official definition.

2. Attribution, authorship, disclosure and ethics integrity

   Signals concerning authorship, affiliation, approval/consent to publish, plagiarism, duplicate publication, citation/reference/attribution, copyright/ownership, conflicts of interest, or research-ethics disclosure/approval. This is an integrity and relationship-transparency family, not a blame category.

3. Editorial and peer-review governance

   Signals concerning peer-review integrity, editorial decision-making, rogue/editorial breach, journal/publisher error, or governance of the pre-publication editorial process. Investigation-by-journal/publisher alone is not treated as editorial fault.

4. Post-publication process, transparency and oversight

   Signals concerning investigation source, official/institutional/third-party oversight, notice transparency, communication, objection, correction/retraction status, removal/availability, or legal/ORI/misconduct process. This category means post-publication process/oversight metadata, not due-process failure or fault by the named actor.

## Main-figure versus audit flags

The final post-audit pipeline generates two category layers:

- `cat_*`: all semantic manuscript-category assignments. Use for audit and supplementary tables.
- `maincat_*`: only assignments from reason items with `main_figure_recommended=true`. Use for main figures to prevent status-only or highly ambiguous labels from inflating prevalence.

Percentages are record-level non-exclusive prevalence and may sum above 100%.

## Scripts

```bash
python3 scripts/run_rw_deepseek_api_v03.py --dry-run
python3 scripts/run_rw_deepseek_api_v03.py
python3 scripts/apply_rw_v03_postaudit_overrides.py
```

`DEEPSEEK_API_KEY` must be available in the environment. Do not commit credentials.

## Outputs

Typical output directory:

```text
data/rw-derived/2026-05-15-llm-semantic-v0.3-api/
```

Important output files:

- `reason_semantic_taxonomy.llm_v0.3-api.json`: raw API taxonomy
- `reason_semantic_taxonomy.llm_v0.3-api.final.json`: post-Codex final taxonomy
- `record_level_binary.llm_v0.3-api.final.csv`: final record-level binary table
- `prevalence_summary.llm_v0.3-api.final.csv`: prevalence summaries
- `v0.3_api_final_validation_report.json`: validation and category definitions

## Guardrails

- Retraction Watch reason terms are curated metadata, not adjudicated causes.
- Investigation-by-X is treated as a process/source signal, not fault by X.
- Actor-context labels mean mentioned/source/procedure context, not responsibility.
- The four categories are manuscript-facing semantic groups, not Retraction Watch native categories.
- Main figures should use `maincat_*`, not actor-context percentages.
