# SimpleNews S0–S6 Refactored CLI Workflow

This refactor keeps the notebook as a **display-only viewer** and moves system execution into Python scripts and `make` targets.

## Design goal
- build systems from the command line
- keep reusable logic in `src/`
- support **S0–S6**
- keep the notebook only for **plots and result evaluation display**
- reuse frozen expensive runs by default for **S4, S5, and S6**

## Layout
- `src/` – shared preprocessing, selection, simplification, pipeline, and evaluation code
- `scripts/` – command-line entry points for dataset loading, preprocessing, running systems, combining outputs, tables, and plots
- `s6_dpo/` – packaged DPO-related code for optional S6 training/inference
- `outputs/system_runs/` – one CSV per system
- `outputs/tables/` – summary tables for the report/notebook
- `outputs/figures/` – rebuilt plots for the report/notebook
- `SimpleNews_refactored_control_panel_with_plots_s0_s6.ipynb` – notebook that only displays results

## Systems
- `S0` Lead-3
- `S1` TextRank
- `S2` Coverage-aware TextRank
- `S3` Coverage-aware TextRank + conservative lexical simplification
- `S4` Coverage-aware scaffold + constrained seq2seq rewrite
- `S5` End-to-end seq2seq baseline
- `S6` DPO-based seq2seq system

## Important principle
Frozen CSV outputs are the source of truth whenever they already exist.

That means:
- `S0`–`S3` are cheap and can be recomputed directly.
- `S4`, `S5`, and `S6` default to cached outputs in `outputs/system_runs/` or the frozen raw CSVs if available.
- You only need checkpoint-based inference if you explicitly want to refresh those systems.

## Quick start

### 1. Bootstrap frozen artifacts
If you already have:
- a frozen combined `S0`–`S5` CSV such as `test_s0_s5_v46_stable_top_to_bottom_all_system_outputs(...).csv`
- an `S6` raw CSV such as `s6_dpo_inference_results.csv`

run:

```bash
make bootstrap
```

This will populate:
- `outputs/system_runs/test_S0.csv`
- ...
- `outputs/system_runs/test_S6.csv`

For `S6`, metrics are computed and appended automatically if the raw file only contains `article/reference/output`.

### 2. Rebuild combined outputs, tables, and plots
```bash
make combine
make tables
make plots
```

### 3. Open the notebook
The notebook does **not** run systems. It only reads `outputs/tables/` and `outputs/figures/`.

```bash
jupyter notebook SimpleNews_refactored_control_panel_with_plots_s0_s6.ipynb
```

## Main make targets

### Data
```bash
make data-loading DATASET=cnn
make preprocess SPLIT=test
```

### Baselines
```bash
make baseline-models SPLIT=test
make s2 SPLIT=test
make s3 SPLIT=test
```

### LLM / seq2seq systems
```bash
make s4 SPLIT=test MODE=auto
make s5 SPLIT=test MODE=auto
make s6 SPLIT=test MODE=auto
```

Modes:
- `MODE=auto` → use cached outputs if present, otherwise use checkpoint inference if a checkpoint exists
- `MODE=cached` → require cached outputs only
- `MODE=inference` → force checkpoint-based inference

### Example with explicit checkpoints
```bash
make s4 MODE=inference CHECKPOINT=outputs/checkpoints/s4_finetuned
make s5 MODE=inference CHECKPOINT=outputs/checkpoints/s5_finetuned MODEL_NAME=facebook/bart-large-cnn
make s6 MODE=inference CHECKPOINT=<PATH_TO_YOUR_DPO_CHECKPOINT> MODEL_NAME=facebook/bart-large-cnn
```

## Evaluate one system
```bash
make evaluate SYSTEM=S4 SPLIT=test
```

This writes a compact one-system summary CSV to `outputs/tables/`.

## Full report-artifact rebuild
```bash
make bootstrap
make combine
make tables
make plots
```

## Optional DPO workflow
The packaged `s6_dpo/` folder preserves your DPO-related code and allows an optional training workflow.

```bash
make dpo-data
make dpo-train
make dpo-infer CHECKPOINT=<PATH_TO_YOUR_DPO_CHECKPOINT> MODEL_NAME=facebook/bart-large-cnn
```

Notes:
- `dpo-data` may require OpenAI API access if you use the judge step in `generate_dpo_data.py`.
- `dpo-train` may require `wandb`, `openai`, and `safetensors`, depending on your environment.
- The main CLI workflow does **not** require rerunning DPO if `s6_dpo_inference_results.csv` already exists.

## Notebook philosophy
`SimpleNews_refactored_control_panel_with_plots_s0_s6.ipynb` is intentionally lightweight.

It only:
- displays summary tables
- shows the generated plots
- previews compact evaluation outputs

It does **not**:
- download data
- train models
- rerun systems
- mutate outputs

## Suggested submission workflow
For reproducibility, the cleanest submission structure is:

1. keep the frozen long notebook as archival evidence
2. use this CLI refactor as the primary runnable workflow
3. include the Makefile and README
4. use the display-only notebook for quick visual inspection

That gives you:
- better source-code organization
- cleaner separation of concerns
- clearer command-line reproducibility
- a lighter notebook for graders
