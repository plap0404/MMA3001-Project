"""Download the pork rasher dataset from Roboflow.

Downloads Version 1 of the forked Roboflow project in YOLO format into
``data/pork_v1``. The Roboflow API key is read from the
``ROBOFLOW_API_KEY`` variable in the local ``.env`` file.

Usage (from the project root, with the virtual environment active)::

    python scripts/download_data.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

WORKSPACE = "plap0002-student-monash-edu"
PROJECT = "pork-rasher-error-packaging-hvffk"
VERSION = 1
EXPORT_FORMAT = "yolov8"

# The project root is one level above this scripts/ folder, so the data
# always lands in the same place, wherever the script is run from.
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / f"pork_v{VERSION}"


def get_api_key() -> str:
    """Return the Roboflow API key stored in the .env file.

    Returns:
        The API key as a string.

    Raises:
        SystemExit: If ``ROBOFLOW_API_KEY`` is missing, with a message
            explaining how to fix it.
    """
    load_dotenv(ROOT / ".env")
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY not found. Add it to .env in the project root.")
    return key


def main() -> None:
    """Download the chosen dataset version into ``OUTPUT_DIR``."""
    rf = Roboflow(api_key=get_api_key())
    project = rf.workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download(EXPORT_FORMAT, location=str(OUTPUT_DIR))
    print(f"Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()