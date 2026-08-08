import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from agent import build_agent
from sport_utils import detect_sport, get_sport_questions, get_sport_roles

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Sports Statistics AI Agent",
    layout="wide",
)

def extract_text(content) -> str:
    """
    Extracts text content from Gemini response structure.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()

    return str(content)


def load_data_file(source) -> pd.DataFrame:
    """
    Loads CSV or Excel data.
    """
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    name = getattr(source, "name", str(source))
    if str(name).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    return pd.read_csv(source)


# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = None

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None


# Header
st.markdown("<h1 class='title-header'>Sports Statistics AI Agent</h1>", unsafe_allow_html=True)

# Main File Upload (No Sidebar, No Manual Sport Mode Selectbox)
uploaded_file = st.file_uploader(
    "Upload Sports Dataset (CSV or Excel):",
    type=["csv", "xlsx", "xls"],
)


# Upload dataset logic
if uploaded_file is not None:
    active_source = uploaded_file
    active_name = uploaded_file.name
    if st.session_state.df is None or st.session_state.dataset_name != active_name:
        try:
            df_loaded = load_data_file(active_source)
            st.session_state.df = df_loaded
            st.session_state.dataset_name = active_name
            st.session_state.agent_executor = None
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
else:
    st.session_state.df = None
    st.session_state.dataset_name = None
    st.session_state.agent_executor = None

df = st.session_state.df

if df is not None:
    # Auto Detect Sport Type
    sport_mode = detect_sport(df)
    st.markdown(f"<div class='sport-badge'>Detected Sport: <b>{sport_mode}</b></div>", unsafe_allow_html=True)

    # Column layout for Role filter and Preset Questions
    col_role, col_preset = st.columns([1, 2])

    with col_role:
        available_roles = get_sport_roles(sport_mode)
        selected_role = st.selectbox(
            "Filter / Category (Role/Position):",
            options=available_roles,
            index=0,
        )

    with col_preset:
        preset_questions = get_sport_questions(sport_mode, selected_role)
        selected_preset = st.selectbox(
            "Suggested Questions:",
            options=["Select a suggested question..."] + preset_questions,
            index=0,
        )

    # If a suggested question is selected, pre-fill the query text box
    default_query_text = ""
    if selected_preset != "Select a suggested question...":
        default_query_text = selected_preset

    question = st.text_input(
        "Enter your sports question:",
        value=default_query_text,
        placeholder="e.g. Who are the top 5 batters by total runs? or enter a custom query...",
    )

    if st.button("Ask AI Agent", use_container_width=True):
        if not question.strip():
            st.warning(
                "Please enter a question or choose a suggested question from the dropdown."
            )
        else:
            if st.session_state.agent_executor is None:
                with st.spinner("Building AI Sports Agent..."):
                    try:
                        st.session_state.agent_executor = build_agent(df)
                    except Exception as ex:
                        st.error(f"Failed to build agent: {ex}")

            if st.session_state.agent_executor:
                with st.spinner(
                    f"Analyzing {sport_mode} dataset and generating insights..."
                ):
                    try:
                        response = st.session_state.agent_executor.invoke(
                            {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": question,
                                    }
                                ]
                            }
                        )

                        final_msg = response["messages"][-1]
                        text_res = extract_text(final_msg.content)

                        st.success("### AI Analysis Result")
                        st.markdown(text_res)

                        # Check for generated chart images
                        for msg in response["messages"]:
                            c_str = getattr(msg, "content", None)

                            if isinstance(c_str, str) and "CHART_SAVED:" in c_str:
                                for line in c_str.splitlines():
                                    if "CHART_SAVED:" in line:
                                        c_path = line.split("CHART_SAVED:")[-1].strip()

                                        if Path(c_path).exists():
                                            st.image(
                                                c_path,
                                                caption="AI Generated Sports Chart",
                                                use_column_width=True,
                                            )

                    except Exception as err:
                        st.error(f"Execution Error: {err}")

    st.markdown("---")
    with st.expander("Complete Dataset Preview and Schema Details"):
        st.dataframe(df, use_container_width=True)

else:
    st.info("Please upload a sports dataset (CSV or Excel file) above to begin analysis.")
