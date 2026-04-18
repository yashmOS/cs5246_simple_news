PYTHON ?= python
PROJECT_ROOT ?= $(CURDIR)
SPLIT ?= test
SYSTEM ?= S3
DATASET ?= cnn
MODE ?= auto
CHECKPOINT ?=
MODEL_NAME ?=

.PHONY: help bootstrap data-loading preprocess baseline-models s0 s1 s2 s3 s4 s5 s6 dpo dpo-data dpo-train dpo-infer combine tables plots evaluate notebook all clean

help:
	@echo "SimpleNews CLI targets"
	@echo "  make bootstrap                    # split frozen S0-S5 outputs and add evaluated S6"
	@echo "  make data-loading DATASET=cnn     # ensure dataset files exist"
	@echo "  make preprocess SPLIT=test        # save preprocess summary for one split"
	@echo "  make baseline-models SPLIT=test   # run S0 and S1"
	@echo "  make s2 / make s3                 # run individual classical systems"
	@echo "  make s4 MODE=auto CHECKPOINT=...  # cached or checkpoint-based S4"
	@echo "  make s5 MODE=auto CHECKPOINT=...  # cached or checkpoint-based S5"
	@echo "  make s6 MODE=auto CHECKPOINT=...  # cached or checkpoint-based S6 (DPO)"
	@echo "  make dpo                          # alias for make s6"
	@echo "  make combine tables plots         # rebuild evaluation artifacts"
	@echo "  make evaluate SYSTEM=S4           # one-system metric summary"
	@echo "  make notebook                     # execute the display-only notebook"

bootstrap:
	$(PYTHON) scripts/bootstrap_cached_artifacts.py --project-root $(PROJECT_ROOT)

data-loading:
	$(PYTHON) scripts/data_loading.py --project-root $(PROJECT_ROOT) --dataset $(DATASET)

preprocess:
	$(PYTHON) scripts/preprocess_dataset.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

baseline-models: s0 s1

s0:
	$(PYTHON) scripts/run_s0.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

s1:
	$(PYTHON) scripts/run_s1.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

s2:
	$(PYTHON) scripts/run_s2.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

s3:
	$(PYTHON) scripts/run_s3.py --project-root $(PROJECT_ROOT) --split $(SPLIT) --force-recompute

s4:
	$(PYTHON) scripts/run_s4.py --project-root $(PROJECT_ROOT) --split $(SPLIT) --mode $(MODE) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),) $(if $(MODEL_NAME),--model-name $(MODEL_NAME),)

s5:
	$(PYTHON) scripts/run_s5.py --project-root $(PROJECT_ROOT) --split $(SPLIT) --mode $(MODE) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),) $(if $(MODEL_NAME),--model-name $(MODEL_NAME),)

s6:
	$(PYTHON) scripts/run_s6.py --project-root $(PROJECT_ROOT) --split $(SPLIT) --mode $(MODE) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),) $(if $(MODEL_NAME),--model-name $(MODEL_NAME),)

dpo: s6

dpo-data:
	$(PYTHON) -m s6_dpo.generate_dpo_data --generate

dpo-train:
	$(PYTHON) -m s6_dpo.dpo

dpo-infer:
	$(PYTHON) scripts/run_s6.py --project-root $(PROJECT_ROOT) --split $(SPLIT) --mode inference $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),) $(if $(MODEL_NAME),--model-name $(MODEL_NAME),)

combine:
	$(PYTHON) scripts/combine_outputs.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

tables:
	$(PYTHON) scripts/build_report_tables.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

plots:
	$(PYTHON) scripts/build_report_plots.py --project-root $(PROJECT_ROOT) --split $(SPLIT)

evaluate:
	$(PYTHON) scripts/evaluate_system.py --project-root $(PROJECT_ROOT) --split $(SPLIT) --system $(SYSTEM)

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute SimpleNews_refactored_control_panel_with_plots_s0_s6.ipynb --output executed_SimpleNews_refactored_control_panel_with_plots_s0_s6.ipynb --ExecutePreprocessor.timeout=-1

all: bootstrap combine tables plots

clean:
	rm -rf outputs/tables/* outputs/figures/* outputs/system_runs/*
