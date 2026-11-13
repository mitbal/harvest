"""Logging setup for the pipeline scripts.

Railway marks every stderr line as an error and every stdout line as info, so
records below ERROR go to stdout and ERROR and above go to stderr.
"""
import logging
import os
import sys

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging() -> None:
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda record: record.levelno < logging.ERROR)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.ERROR)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format=LOG_FORMAT,
        handlers=[out, err],
    )
