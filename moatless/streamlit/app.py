import logging
import os
import sys

import streamlit as st
from dotenv import load_dotenv

from moatless.benchmark.report import generate_report
from moatless.search_tree import SearchTree
from moatless.streamlit.shared import trajectory_table
from moatless.streamlit.tree_vizualization import update_visualization

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def main():
    st.set_page_config(
        layout="wide",
        page_title="Moatless Visualizer",
        initial_sidebar_state="collapsed",
    )
    container = st.container()

    # Get file path from command line args if provided
    if "path" in st.query_params:
        file_path = st.query_params["path"]
    elif len(sys.argv) > 1:
        file_path = sys.argv[1]

        # is directory
        if os.path.isdir(file_path):
            logger.info("Generating report for directory")
            generate_report(file_path)
            file_path = os.path.join(file_path, "report.json")

        st.query_params["path"] = file_path
    else:
        file_path = None

    if not file_path:
        st.sidebar.text("Please provide a file path.")
        file_path = st.text_input("Enter the full path to your JSON file:")
        load_button = st.button("Load File")

        if load_button and file_path:
            st.query_params["path"] = file_path
            st.rerun()
    else:
        st.sidebar.text(f"Loading file: {file_path}")

    if file_path:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path).lower()

            if file_name == "report.json":
                with st.spinner("Loading report..."):
                    trajectory_table(file_path)
                    report_dir = os.path.dirname(file_path)
                    if st.button("Regenerate Report"):
                        with st.spinner("Regenerating report..."):
                            generate_report(report_dir, split="verified")
                            st.rerun()

            else:
                with st.spinner("Loading search tree from trajectory file"):
                    st.session_state.search_tree = SearchTree.from_file(file_path)
                    # directory path
                    update_visualization(
                        container, st.session_state.search_tree, file_path
                    )
        else:
            st.error(
                "The specified file does not exist. Please check the path and try again."
            )

    if not file_path:
        st.info("Please provide a valid file path and click 'Load File' to begin.")


if __name__ == "__main__":
    main()
