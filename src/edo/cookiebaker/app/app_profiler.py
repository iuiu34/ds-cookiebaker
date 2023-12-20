"""Streamlit app."""

from streamlit_profiler import Profiler

from edo.sorting_hat_fare_rules_app.app import app

with Profiler():
    app()
