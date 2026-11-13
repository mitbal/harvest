import logging
import sys
import streamlit as st
from pythonjsonlogger.jsonlogger import JsonFormatter


@st.cache_resource
def setup_logging(name, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # The cache resets on code reload while the logger lives on, so don't stack handlers.
    if logger.handlers:
        return logger

    # Railway marks every stderr line as an error, so log to stdout and name the
    # level key "level", which Railway reads from JSON lines.
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(fmt='{asctime}{name}{levelname}{message}',
                              style='{',
                              rename_fields={'levelname': 'level'})
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
