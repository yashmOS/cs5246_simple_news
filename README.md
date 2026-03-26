# Simplify News

Codebase for CS5246 Simplify News project.

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

This notebook will:
1. optionally download the dataset
2. load data
3. run EDA
4. generate outputs for S0-S3
5. evaluate systems
6. save report-ready CSVs and figures under `./outputs/`

## Systems
- S0: Lead-3
- S1: TextRank
- S2: Coverage-aware TextRank
- S3: Coverage-aware TextRank + conservative lexical simplification + glossary

## Reproducibility
All outputs are written to `./outputs/`.
