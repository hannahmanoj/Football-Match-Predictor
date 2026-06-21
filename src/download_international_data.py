from __future__ import annotations

from pathlib import Path
import shutil

import kagglehub


DATASET = "martj42/international-football-results-from-1872-to-2017"
OUTPUT_PATH = Path("data/international_matches.csv")


def main() -> None:
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    source_path = dataset_path / "results.csv"

    if not source_path.exists():
        available_files = ", ".join(sorted(path.name for path in dataset_path.iterdir()))
        raise FileNotFoundError(
            f"Could not find results.csv in {dataset_path}. Available files: {available_files}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, OUTPUT_PATH)

    print(f"Downloaded dataset to: {dataset_path}")
    print(f"Saved match results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
