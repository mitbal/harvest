import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_streamlit_apptests_pass():
    """Run apptests/ in its own process, where Streamlit is real rather than the mock installed here."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "apptests", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, env={**os.environ, "HARVEST_APPTEST": "1"},
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
