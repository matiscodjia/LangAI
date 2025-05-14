import streamlit as st
import asyncio

from frontend.interface.logic import render_query_results, run_query_pipeline
from frontend.ui.ui_backend import sidebar_controls

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import os
os.environ["STREAMLIT_WATCHFILE"] = "false"

st.set_page_config(layout="wide")
st.title("Dashboard RAG")
# === UI: Sidebar Configuration ===
config = sidebar_controls()

# === UI: Main Input ===
query = st.text_input("Entrez la question")
run_button = st.button("Run")
# === Trigger Processing ===
if run_button and query:
    result = run_query_pipeline(query, config)
    render_query_results(query, result, config)



