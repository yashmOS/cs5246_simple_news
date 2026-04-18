from pathlib import Path
import subprocess, sys

root = Path(__file__).resolve().parents[1]
cmd = [sys.executable, str(root / 'scripts' / 'run_system.py'), '--project-root', str(root), '--system', 'S6'] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd))
