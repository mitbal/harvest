import logging
import streamlit as st
from pythonjsonlogger.jsonlogger import JsonFormatter


@st.cache_resource
def setup_logging(name, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # The cache resets on code reload while the logger lives on, so don't stack handlers.
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = JsonFormatter(fmt='{asctime}{name}{levelname}{message}',
                              style='{')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
