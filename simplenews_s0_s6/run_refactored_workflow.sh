#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python scripts/bootstrap_cached_artifacts.py --project-root "$ROOT"
python scripts/combine_outputs.py --project-root "$ROOT" --split test
python scripts/build_report_tables.py --project-root "$ROOT" --split test
python scripts/build_report_plots.py --project-root "$ROOT" --split test

echo "Done. Open SimpleNews_refactored_control_panel_with_plots_s0_s6.ipynb to view results."
