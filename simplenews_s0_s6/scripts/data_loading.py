from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import ensure_local_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or verify local dataset files.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="cnn", help="Dataset key. Default: cnn")
    args = parser.parse_args()

    status = ensure_local_dataset(args.project_root.resolve(), dataset_name=args.dataset)
    print(status)


if __name__ == "__main__":
    main()
