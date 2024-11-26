import streamlit as st
import os
import json
import sys
import logging
from typing import List, Dict, Optional, Tuple
from moatless.node import Node
from moatless.search_tree import SearchTree
from moatless.benchmark.utils import get_moatless_instance
from moatless.streamlit.shared import show_completion

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

# Make the layout wide
st.set_page_config(layout="wide")

def get_trajectories(dir: str) -> list[Tuple[SearchTree, str]]:
    """Find all search trees in directory, matching generate_report.py logic."""
    trajectories = []
    for root, _, files in os.walk(dir):
        trajectory_path = os.path.join(root, "trajectory.json")
        if not os.path.exists(trajectory_path):
            logger.warning(f"Trajectory file not found: {trajectory_path}")
            continue

        try:
            rel_path = os.path.relpath(root, dir)
            # Check if file is empty
            if os.stat(trajectory_path).st_size == 0:
                logger.warning(f"Empty trajectory file: {trajectory_path}")
                continue

            trajectory = SearchTree.from_file(trajectory_path)
            trajectories.append((trajectory, rel_path))
        except Exception as e:
            logger.exception(f"Failed to load trajectory from {trajectory_path}: {e}")
    
    return trajectories

def collect_fail_reasons(trajectories: List[Tuple[SearchTree, str]]) -> Dict[str, List[tuple[str, Node, SearchTree]]]:
    """Collect all fail reasons and their corresponding nodes across trajectories."""
    fail_reasons: Dict[str, List[tuple[str, Node, SearchTree]]] = {}
    
    for search_tree, rel_path in trajectories:
        instance_id = search_tree.metadata.get("instance_id")
        if not instance_id:
            continue
            
        for node in search_tree.root.get_all_nodes():
            if node.observation and node.observation.properties:
                if "fail_reason" in node.observation.properties:
                    fail_reason = node.observation.properties["fail_reason"]
                    if fail_reason not in fail_reasons:
                        fail_reasons[fail_reason] = []
                    fail_reasons[fail_reason].append((instance_id, node, search_tree))
    
    return fail_reasons

def collect_flags(trajectories: List[Tuple[SearchTree, str]]) -> Dict[str, List[tuple[str, Node, SearchTree]]]:
    """Collect all flags and their corresponding nodes across trajectories."""
    flags: Dict[str, List[tuple[str, Node, SearchTree]]] = {}
    
    for search_tree, rel_path in trajectories:
        instance_id = search_tree.metadata.get("instance_id")
        if not instance_id:
            continue
            
        for node in search_tree.root.get_all_nodes():
            if node.observation and node.observation.properties:
                if "flags" in node.observation.properties:
                    for flag in node.observation.properties["flags"]:
                        if flag not in flags:
                            flags[flag] = []
                        flags[flag].append((instance_id, node, search_tree))
    
    return flags

def create_table_header():
    """Create the header row for the node table."""
    cols = st.columns([1, 3, 3, 2])
    cols[0].markdown("**#**")
    cols[1].markdown("**Node**")
    cols[2].markdown("**Action**")
    cols[3].markdown("**Result**")
    return cols

def create_node_row(node: Node, search_tree: SearchTree, instance: dict):
    """Create a single row in the node table for the given node."""
    
    cols = st.columns([1, 3, 3, 2])
    
    # Node column
    node_str = f"Node{node.node_id}"
    if node.action:
        node_str += f" ({node.action.name})"
    cols[0].subheader(node_str)
    
    # Add token usage info if available
    if node.completions:
        usage_rows = []
        has_cost = False
        
        for completion_type, completion in node.completions.items():
            if completion and completion.usage:
                usage = completion.usage
                tokens = []
                if usage.prompt_tokens:
                    tokens.append(f"{usage.prompt_tokens}↑")
                if usage.completion_tokens:
                    tokens.append(f"{usage.completion_tokens}↓")
                if usage.cached_tokens:
                    tokens.append(f"{usage.cached_tokens}⚡")
                
                if usage.completion_cost:
                    has_cost = True
                    cost = f"${usage.completion_cost:.4f}"
                    usage_rows.append(f"|{completion_type}|{cost}|{' '.join(tokens)}|")
                elif tokens:
                    usage_rows.append(f"|{completion_type}|{' '.join(tokens)}|")
                    
        if usage_rows:
            if has_cost:
                header = "|Type|Cost|Tokens|"
                separator = "|:--|--:|:--|"
            else:
                header = "|Type|Tokens|"
                separator = "|:--|:--|"
            table = "\n".join([header, separator] + usage_rows)
            cols[0].markdown(table, help="↑:prompt ↓:completion ⚡:cached")
            cols[0].markdown("---")
    
    # Action column with tabs
    if node.action:
        tab_names = ["Action", "Completion"]
        
        action_tabs = cols[1].tabs(tab_names)
        
        # Input tab
        with action_tabs[0]:
            if hasattr(node.action, "thoughts") and node.action.thoughts:
                st.markdown(node.action.thoughts)
        
            if hasattr(node.action, "old_str"):
                st.markdown(f"**File path:** `{node.action.path}`")
                st.markdown("**Old string:**")
                st.code(node.action.old_str)
                st.markdown("**New string:**")
                st.code(node.action.new_str)
            elif hasattr(node.action, "file_text"):
                st.write(f"File path: {node.action.path}")
                st.markdown("**File text:**")
                st.code(node.action.file_text)
            else:
                st.json(node.action.model_dump(exclude={"thoughts"}, exclude_none=True))

        # Build tab
        with action_tabs[1]:
            if node.completions and node.completions.get("build_action"):
                show_completion(node.completions["build_action"])
            else:
                st.info("No build completion available")

    # Observation column with tabs
    if node.observation:
        tabs = cols[2].tabs(["Observation", "Message", "JSON"])
        
        # Properties tab (shown by default)
        with tabs[0]:        
            if node.observation and node.observation.properties:

                if "diff" in node.observation.properties:
                    st.markdown("**Diff:**")
                    st.code(node.observation.properties["diff"])
                
                if "new_span_ids" in node.observation.properties:
                    st.markdown("**New span IDs:**")
                    for span_id in node.observation.properties["new_span_ids"]:
                        st.markdown(f"- `{span_id}`")

                if "fail_reason" in node.observation.properties:
                    st.error(f"🛑 {node.observation.properties['fail_reason']}")
                    st.code(node.observation.message)

                if "flags" in node.observation.properties:
                    st.warning(f"⚠️ {', '.join(node.observation.properties['flags'])}")
                    st.code(node.observation.message)
                
                if "test_results" in node.observation.properties:
                    test_results = node.observation.properties["test_results"]
                    total_tests = len(test_results)
                    failed_test_count = sum(
                        1 for test in test_results if test["status"] in ["FAILED", "ERROR"]
                    )
                    
                    if failed_test_count > 0:
                        st.warning(f"⚠️ {failed_test_count} out of {total_tests} tests failed")
                    else:
                        st.success(f"✅ All {total_tests} tests passed")
                    
            
        # Message tab
        with tabs[1]:
            st.code(node.observation.message)
            
        # JSON tab
        with tabs[2]:
            st.json(node.observation.model_dump(), expanded=False)
    
    # Context column with tabs
    if node.file_context:
        context_col = cols[3]
        
        # Create tabs for context
        tab_names = ["Summary"]
        if node.file_context.has_patch():
            tab_names.append("Patch")

        context_tabs = context_col.tabs(tab_names)
        
        # Summary tab
        with context_tabs[0]:
            # Show context summary
            st.markdown(node.file_context.create_summary())
        
        # Patch tab
        if node.file_context.has_patch():
            with context_tabs[1]:
                st.code(node.file_context.generate_git_patch())
            tab_offset = 2
        else:
            tab_offset = 1
        
    
    # Add a separator between rows
    st.markdown("---")

def main():
    st.title("Trajectory Troubleshooter")
    
    # Get directory from query params or command line
    if "dir" in st.query_params:
        dir_path = st.query_params["dir"]
    elif len(sys.argv) > 1:
        dir_path = sys.argv[1]
        # Update query params
        st.query_params["dir"] = dir_path
    else:
        dir_path = "./runs"
    
    # Input directory path with current value from query params
    new_dir_path = st.text_input("Enter directory path:", value=dir_path)
    
    # Update query params if path changes
    if new_dir_path != dir_path:
        st.query_params["dir"] = new_dir_path
        st.rerun()
    
    if not os.path.exists(new_dir_path):
        st.error(f"Directory {new_dir_path} does not exist")
        return
        
    # Read all search trees using the same logic as generate_report.py
    with st.spinner("Reading trajectories..."):
        trajectories = get_trajectories(new_dir_path)
        
    if not trajectories:
        st.error("No trajectories found in directory")
        return
    
    st.info(f"Found {len(trajectories)} trajectories")
        
    # Collect fail reasons and flags
    fail_reasons = collect_fail_reasons(trajectories)
    flags = collect_flags(trajectories)
    
    # Create tabs for fail reasons and flags
    reason_tab, flag_tab = st.tabs(["Fail Reasons", "Flags"])
    
    with reason_tab:
        if not fail_reasons:
            st.info("No fail reasons found in trajectories")
        else:
            # Create selectbox for fail reasons with count
            fail_reason_options = [
                f"{reason} ({len(nodes)})" 
                for reason, nodes in sorted(fail_reasons.items(), key=lambda x: len(x[1]), reverse=True)
            ]
            selected_fail_reason = st.selectbox(
                "Select fail reason:",
                options=fail_reason_options
            )
            
            if selected_fail_reason:
                # Extract the actual reason without count
                reason = selected_fail_reason.split(" (")[0]
                nodes = fail_reasons[reason]
                
                # Create summary table
                summary_cols = st.columns([1, 1, 1, 1])
                summary_cols[0].markdown("**Instance ID**")
                summary_cols[1].markdown("**Node ID**")
                summary_cols[2].markdown("**Action**")
                summary_cols[3].markdown("**Status**")
                
                # Show summary for first 10 nodes
                for instance_id, node, search_tree in nodes:
                    cols = st.columns([1, 1, 1, 1])
                    cols[0].write(instance_id)
                    cols[1].write(f"Node{node.node_id}")
                    cols[2].write(node.action.name)
                    
                    # Determine status from node observation
                    status = "❌ Failed"  # Default since we're looking at fail reasons
                    if node.observation and node.observation.properties:
                        if "fail_reason" in node.observation.properties:
                            status = f"❌ {node.observation.properties['fail_reason']}"
                    cols[3].write(status)
                    
                st.markdown("---")
                
                # Show detailed information for each node
                st.markdown("### Detailed Information")
                create_table_header()
                for instance_id, node, search_tree in nodes[:10]:
                    st.markdown(f"#### Instance: {instance_id}")
                    instance = get_moatless_instance(instance_id)
                    create_node_row(node, search_tree, instance)

    with flag_tab:
        if not flags:
            st.info("No flags found in trajectories")
        else:
            # Create selectbox for flags with count
            flag_options = [
                f"{flag} ({len(nodes)})" 
                for flag, nodes in sorted(flags.items(), key=lambda x: len(x[1]), reverse=True)
            ]
            selected_flag = st.selectbox(
                "Select flag:",
                options=flag_options,
                key="flag_select"  # Unique key to avoid conflict with fail_reason selectbox
            )
            
            if selected_flag:
                # Extract the actual flag without count
                flag = selected_flag.split(" (")[0]
                nodes = flags[flag]
                
                # Create summary table
                summary_cols = st.columns([1, 1, 1, 1])
                summary_cols[0].markdown("**Instance ID**")
                summary_cols[1].markdown("**Node ID**")
                summary_cols[2].markdown("**Action**")
                summary_cols[3].markdown("**Status**")
                
                # Show summary for first 10 nodes
                for instance_id, node, search_tree in nodes:
                    cols = st.columns([1, 1, 1, 1])
                    cols[0].write(instance_id)
                    cols[1].write(f"Node{node.node_id}")
                    cols[2].write(node.action.name)
                    cols[3].write(f"⚠️ {flag}")
                
                st.markdown("---")
                
                # Show detailed information for each node
                st.markdown("### Detailed Information")
                create_table_header()
                for instance_id, node, search_tree in nodes:
                    st.markdown(f"#### Instance: {instance_id}")
                    instance = get_moatless_instance(instance_id)
                    create_node_row(node, search_tree, instance)

if __name__ == "__main__":
    main() 