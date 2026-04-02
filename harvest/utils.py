import logging
import streamlit as st
from pythonjsonlogger.jsonlogger import JsonFormatter


@st.cache_resource
def setup_logging(name, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = JsonFormatter(fmt='{asctime}{name}{levelname}{message}',
                              style='{')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
