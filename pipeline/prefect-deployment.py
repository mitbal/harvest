"""Deploy the rollback Prefect flows declared in prefect.yaml."""

import subprocess
from pathlib import Path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(["prefect", "deploy", "--all"], cwd=repository_root, check=True)
