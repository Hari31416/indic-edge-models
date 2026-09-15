#!/usr/bin/env python3
"""Root entrypoint script for Kaggle remote experiment runner."""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_runner.cli import main

if __name__ == "__main__":
    sys.exit(main())
