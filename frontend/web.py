import streamlit as st
from modules.ui.ui_backend import *
from modules.ui.ui_backend import *
from modules.retrieving.retrieving import *
from modules.interface.logic import *


import asyncio

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



