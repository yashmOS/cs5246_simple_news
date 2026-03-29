# Simplify News

Codebase for the CS5246 Simplify News project.

## Dataset bootstrap
The notebook can automatically download **CNN/DailyMail** before running the pipeline.

What it does:
- downloads the dataset through `datasets`
- writes local CSV files under `./data/`
- keeps the rest of the pipeline unchanged

Files created:
- `data/train.csv`
- `data/validation.csv`
- `data/test.csv`

If local CSV files already exist, the bootstrap step will reuse them unless you force a re-download.

## Expected data
Place your dataset under `./data/`, or let the notebook download it automatically.

Supported formats:
- `train.csv`
- `validation.csv`
- `test.csv`
- any single CSV containing article and reference-summary columns

The loader will try to detect columns such as:
- article, text, document, story, source
- highlights, summary, target, reference
- title, headline

## Main interface
Use:
- `notebooks/simplify_news_pipeline.ipynb`

This notebook / pipeline flow will:
1. optionally download the dataset
2. load data
3. run EDA
4. generate outputs for the shared non-LLM systems (**S0-S3**)
5. optionally add the LLM-based systems (**S4-S5**) using the same row format
6. evaluate systems
7. save report-ready CSVs and figures under `./outputs/`

## Core row format
Use the same core columns for every system so outputs can be combined into one dataframe.

Input per example:
- `id`
- `article`
- `reference`

Output per system/article:
- `id`
- `system`
- `article`
- `reference`
- `output`

Extra columns are allowed if needed (for example glossary, replacements, metrics, or protected-item diagnostics), but the core columns above should be preserved.

The repo-native pipeline may also keep `doc_id` internally for compatibility.

## Systems
- **S0:** Lead-3 baseline
- **S1:** TextRank
- **S2:** Coverage-aware TextRank
- **S3:** S2 + conservative lexical simplification + glossary
- **S4:** S2 + constrained LLM rewrite
- **S5:** Full-article LLM baseline

## Interpretation of the systems
- **S2** focuses on better content selection and fact preservation.
- **S3** keeps the same selected content as S2 but applies conservative non-LLM simplification and glossary support.
- **S4** uses an LLM to rewrite the S2 content under faithfulness constraints.
- **S5** is the standalone end-to-end LLM baseline that summarizes directly from the full article.

## Reproducibility
All outputs are written to `./outputs/`.
