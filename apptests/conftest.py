"""AppTest suites run real Streamlit, so they can't share a process with tests/,
whose conftest replaces Streamlit with a mock. tests/test_apptests.py runs this
folder in a subprocess with HARVEST_APPTEST=1; a plain ``pytest`` skips it.
"""
import os

if not os.environ.get("HARVEST_APPTEST"):
    collect_ignore_glob = ["test_*.py"]
