"""Data integrity checks for local runs and CI."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

DATA_PATH = Path("data/raw/HAM10000_metadata.csv")
REQUIRED_COLUMNS = {"image_id", "dx"}
EXPECTED_DX_COUNT = 7


def main() -> None:
    if not DATA_PATH.exists():
        print("Data not found (Running in CI environment). Skipping validation.")
        sys.exit(0)

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Failed to read {DATA_PATH}: {exc}")
        sys.exit(1)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"Missing required columns: {', '.join(sorted(missing))}")
        sys.exit(1)

    unique_dx = df["dx"].dropna().unique()
    if len(unique_dx) != EXPECTED_DX_COUNT:
        print(
            f"Unexpected number of dx values: {len(unique_dx)} (expected {EXPECTED_DX_COUNT})"
        )
        sys.exit(1)

    print("Data Validation Passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
