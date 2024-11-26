import json
import logging
import os
import time
from io import BytesIO
from typing import Optional, List

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots

from moatless.agent.code_agent import create_edit_code_actions
from moatless.benchmark.report import analyse_file_context
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.utils import get_moatless_instance
from moatless.node import Node
from moatless.runtime.testbed import TestbedEnvironment
from moatless.search_tree import SearchTree
from moatless.streamlit.list_visualization import create_linear_table
from moatless.streamlit.shared import show_completion
from moatless.utils.tokenizer import count_tokens

# Add this near the top of the file, after other imports

logger = logging.getLogger(__name__)


def decide_badge(node_info):
    """
    Decide which badge to show for a node based on its properties and the instance data.
    Returns a tuple of (symbol, color) or None if no badge should be shown.
    """
    if node_info.get("resolved") is not None:
        if node_info.get("resolved", False):
            return ("star", "gold")
        else:
            return ("x", "red")

    if node_info.get("error"):
        return ("circle", "red")

    if node_info.get("warning"):
        return ("circle", "yellow")


    if node_info.get("context_status") in ["found_spans"]:
        if node_info.get("patch_status") == "wrong_files":
            return ("circle", "yellow")

        return ("circle", "green")

    if (
        node_info.get("context_status") in ["found_files"]
        or node_info.get("patch_status") == "right_files"
    ):
        return ("circle", "yellow")

    return None


def build_graph(
    root_node: Node, eval_result: dict | None = None, instance: dict | None = None
):
    G = nx.DiGraph()

    # Add new layout logic for linear trajectory
    is_linear = getattr(root_node, "max_expansions", None) == 1

    def is_resolved(node_id):
        if not eval_result:
            return None

        if str(node_id) not in eval_result.get("node_results"):
            return None

        return eval_result["node_results"][str(node_id)].get("resolved", False)

    def add_node_to_graph(node: Node):
        node_id = f"Node{node.node_id}"

        if node.action:
            if node.action.name == "str_replace_editor":
                action_name = node.action.command
            else:
                action_name = node.action.name
        else:
            action_name = ""

        if instance:
            context_stats = analyse_file_context(instance, node.file_context)
        else:
            context_stats = None

        warning = ""
        error = ""
        if node.observation and node.observation.properties:
            if "test_results" in node.observation.properties:
                test_results = node.observation.properties["test_results"]
                failed_test_count = sum(
                    1 for test in test_results if test["status"] in ["FAILED", "ERROR"]
                )

                if failed_test_count > 0:
                    warning = f"{failed_test_count} failed tests"
            if "fail_reason" in node.observation.properties:
                error = f"Fail: {node.observation.properties['fail_reason']}"

        if node.observation and node.observation.expect_correction:
            warning += f"\nExpected correction"

        resolved = is_resolved(node.node_id)

        G.add_node(
            node_id,
            name=action_name,
            type="node",
            visits=node.visits or 1,
            duplicate=node.is_duplicate,
            avg_reward=node.value / node.visits if node.visits else 0,
            reward=node.reward.value if node.reward else 0,
            warning=warning,
            error=error,
            resolved=resolved,
            context_status=context_stats.status if context_stats else None,
            patch_status=context_stats.patch_status if context_stats else None,
            explanation=node.reward.explanation if node.reward else "",
            is_linear=is_linear,
        )

        for child in node.children:
            child_id = f"Node{child.node_id}"
            add_node_to_graph(child)
            G.add_edge(node_id, child_id)

    add_node_to_graph(root_node)
    G.graph["graph"] = {
        "ranksep": "2.0",
        "nodesep": "1.0",
    }  # Increase spacing between ranks and nodes
    return G


def rerun_node(node_id: int, trajectory_path: str, instance: dict):
    """Handle the rerun tab logic for a selected node."""
    if st.button("Rerun Node"):
        with st.spinner("Rerunning node..."):
            try:
                search_tree = st.session_state.search_tree
                node = search_tree.get_node_by_id(node_id)
                new_node = search_tree._expand(node)
                search_tree._simulate(new_node)
                search_tree._backpropagate(new_node)
                search_tree.maybe_persist()

                st.success("Node rerun successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Error during rerun: {str(e)}")
                import traceback

                st.code(traceback.format_exc())


def create_graph_figure(G_subset, G, pos, is_linear=False):
    """Create and return a plotly figure for the graph visualization."""
    edge_x, edge_y = [], []
    for edge in G_subset.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y = [], []
    node_colors, node_sizes, node_text, node_labels = [], [], [], []
    badge_x, badge_y, badge_symbols, badge_colors = [], [], [], []
    node_line_widths = []
    node_line_colors = []
    node_symbols = []

    for node in G_subset.nodes():
        node_info = G.nodes[node]
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        reward = None

        # Set node shape/symbol
        if node_info.get("name") == "Finish":
            node_symbols.append("square")
            node_sizes.append(60)
            node_line_widths.append(4)
            # Set border color based on resolution status
            if node_info.get("resolved") is True:
                node_line_colors.append("#FFD700")  # Gold for resolved
            elif node_info.get("resolved") is False:
                node_line_colors.append("red")
            else:
                node_line_colors.append("#FFD700")  # Default gold color
        elif node_info.get("name") == "Reject":
            node_symbols.append("square")
            node_sizes.append(60)
            node_line_widths.append(4)
            node_line_colors.append("red")
        else:
            node_symbols.append("circle")
            node_sizes.append(60)
            node_line_widths.append(2)
            node_line_colors.append("rgba(0,0,0,0.5)")

        # Set node color based on reward/status
        if node_info.get("name") == "Reject":
            node_colors.append("red")
        elif G.graph.get("is_linear", False):
            node_colors.append("green")
        elif node_info.get("visits", 0) == 0:
            node_colors.append("gray")
        else:
            reward = node_info["reward"]
            if reward is None:
                reward = node_info.get("avg_reward", 0)
            node_colors.append(reward)

        if node_info.get("type") == "node":
            badge = decide_badge(node_info)
            if badge:
                badge_x.append(x + 0.04)
                badge_y.append(y + 0.04)
                badge_symbols.append(badge[0])
                badge_colors.append(badge[1])

            extra = ""
            if node_info.get("contex_status"):
                extra = "Expected span was identified"
            elif node_info.get("file_identified"):
                extra = "Expected file identified"
            elif node_info.get("alternative_span_identified"):
                extra = "Alternative span identified"

            if node_info.get("state_params"):
                extra += "<br>".join(node_info["state_params"])

            if node_info.get("warning"):
                extra += f"<br>Warning: {node_info['warning']}"

            if node_info.get("error"):
                extra += f"<br>Error: {node_info['error']}"

            node_text.append(
                f"ID: {node}<br>"
                f"State: {node_info['name']}<br>"
                f"Duplicates: {node_info.get('duplicate', False)}<br>"
                f"Context status: {node_info.get('context_status', 'unknown')}<br>"
                f"Patch status: {node_info.get('patch_status', 'unknown')}<br>"
                f"Visits: {node_info.get('visits', 0)}<br>"
                f"Reward: {node_info.get('reward', 0)}<br>"
                f"Avg Reward: {node_info.get('avg_reward', 0)}"
                f"<br>{extra}"
            )

            if node_info["name"] == "Pending":
                node_labels.append(f"Start")
            elif node_info["name"] == "Finished":
                node_labels.append(
                    f"{node_info['name']}{node_info['id']}<br>{node_info.get('reward', '0')}"
                )
            elif node_info["name"]:
                node_labels.append(
                    f"{node}<br>{node_info['name']}<br>{node_info.get('reward', '0')}"
                )
            else:
                node_labels.append(f"{node}<br>{node_info.get('reward', '0')}")
        else:
            extra = ""
            if node_info.get("info_params"):
                extra = "<br>".join(node_info["info_params"])

            if node_info.get("warnings"):
                extra += "<br>Warnings:<br>"
                extra += "<br>".join(node_info["warnings"])

                badge = ("diamond", "red")
                badge_x.append(x + 0.04)
                badge_y.append(y + 0.04)
                badge_symbols.append(badge[0])
                badge_colors.append(badge[1])
            elif node_info.get("expected_span_identified") or node_info.get(
                "alternative_span_identified"
            ):
                badge = ("star", "gold")
                badge_x.append(x + 0.04)
                badge_y.append(y + 0.04)
                badge_symbols.append(badge[0])

            node_text.append(extra)
            node_labels.append(f"{node}<br>{node_info.get('name', 'unknown')}")

    fig = go.Figure(make_subplots())

    # Add edge trace
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
    )

    # Add node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            symbol=node_symbols,
            showscale=True,
            colorscale=[
                [0, "red"],
                [0.25, "red"],
                [0.5, "yellow"],
                [1, "green"],
            ],
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title="Reward",
                xanchor="left",
                titleside="right",
                tickmode="array",
                tickvals=[-100, 0, 50, 75, 100],
                ticktext=["-100", "0", "50", "75", "100"],
            ),
            cmin=-100,
            cmax=100,
            line=dict(width=node_line_widths, color=node_line_colors),
        ),
        text=node_labels,
        hovertext=node_text,
        textposition="middle center",
    )
    fig.add_trace(node_trace)

    # Add badge trace
    if badge_x:
        badge_trace = go.Scatter(
            x=badge_x,
            y=badge_y,
            mode="markers",
            hoverinfo="none",
            marker=dict(
                symbol=badge_symbols,
                size=10,
                color=badge_colors,
                line=dict(width=1, color="rgba(0, 0, 0, 0.5)"),
            ),
            showlegend=False,
        )
        fig.add_trace(badge_trace)

    # Update layout based on linear vs tree visualization
    if is_linear:
        fig.update_layout(
            width=max(1000, len(G_subset.nodes()) * 100),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=False,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1,
            ),
        )
        node_trace.update(
            textposition=[
                "bottom center" if i % 2 == 0 else "top center"
                for i in range(len(node_x))
            ],
            textfont=dict(size=10),
        )
    else:
        fig.update_layout(
            title="Search Tree",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600 * (max(1, len(G_subset.nodes()) / 20)),
        )

    return fig



def update_visualization(container, search_tree: SearchTree, selected_tree_path: str, instance: Optional[dict] = None):
    eval_result = None
    logger.info(f"Selected tree path: {selected_tree_path}")
    directory_path = os.path.dirname(selected_tree_path)
    eval_path = f"{directory_path}/eval_result.json"
    if os.path.exists(eval_path):
        with open(f"{directory_path}/eval_result.json", "r") as f:
            eval_result = json.load(f)

    if not instance and search_tree.metadata.get("instance_id"):
        instance = get_moatless_instance(search_tree.metadata["instance_id"])


    # Initialize session state for step-by-step visualization
    if "total_nodes" not in st.session_state:
        st.session_state.total_nodes = count_total_nodes(search_tree.root)
    if "max_node_id" not in st.session_state:
        st.session_state.max_node_id = st.session_state.total_nodes - 1
    if "selected_node_id" not in st.session_state:
        st.session_state.selected_node_id = 0
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False

    # Function to get nodes up to the current max_node_id
    def get_nodes_up_to_id(root_node, max_id):
        nodes = []

        def dfs(node):
            if node.node_id <= max_id:
                nodes.append(node)
                for child in node.children:
                    dfs(child)

        dfs(root_node)
        return nodes

    container.empty()
    with container:
        nodes_to_show = get_nodes_up_to_id(
            search_tree.root, st.session_state.max_node_id
        )
        
        is_linear = getattr(search_tree.root, "max_expansions", None) == 1
        
        if is_linear:
            # Use the new table visualization for linear trajectories
            nodes = search_tree.root.get_all_nodes()
            create_linear_table(nodes, st.session_state.max_node_id, eval_result, instance)
        else:
            graph_col, info_col = st.columns([6, 3])
            with graph_col:
                # Get nodes up to the current max_node_id
                nodes_to_show = get_nodes_up_to_id(
                    search_tree.root, st.session_state.max_node_id
                )
                
                # Original graph visualization code
                G = build_graph(search_tree.root, eval_result, instance)
                G_subset = G.subgraph([f"Node{node.node_id}" for node in nodes_to_show])

                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

                # TODO: Add support selected trajectory

                # Create a mapping of point indices to node IDs
                point_to_node_id = {i: node for i, node in enumerate(G_subset.nodes())}

                # Normalize positions to fit in [0, 1] range
                x_values, y_values = zip(*pos.values())
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)

                # Calculate the scaling factor based on the number of nodes
                num_nodes = len(G.nodes())
                height_scale = max(1, num_nodes / 20)  # Increase height as nodes increase

                for node in pos:
                    x, y = pos[node]
                    normalized_x = 0.5 if x_max == x_min else (x - x_min) / (x_max - x_min)
                    normalized_y = 0.5 if y_max == y_min else (y - y_min) / (y_max - y_min)
                    pos[node] = (normalized_x, normalized_y * height_scale)

                fig = create_graph_figure(G_subset, G, pos, is_linear=is_linear)

                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.button(
                        "⏪ Reset to 0",
                        on_click=lambda: setattr(st.session_state, "max_node_id", 0)
                        or setattr(st.session_state, "selected_node_id", 0),
                        disabled=(
                            st.session_state.max_node_id == 0 or st.session_state.auto_play
                        ),
                    )
                with col2:
                    st.button(
                        "◀ Step Back",
                        on_click=lambda: setattr(
                            st.session_state,
                            "max_node_id",
                            st.session_state.max_node_id - 1,
                        )
                        or setattr(
                            st.session_state,
                            "selected_node_id",
                            st.session_state.max_node_id,
                        ),
                        disabled=(st.session_state.max_node_id <= 0),
                    )
                with col3:
                    st.button(
                        "▶ Step Forward",
                        on_click=lambda: setattr(
                            st.session_state,
                            "max_node_id",
                            st.session_state.max_node_id + 1,
                        )
                        or setattr(
                            st.session_state,
                            "selected_node_id",
                            st.session_state.max_node_id,
                        ),
                        disabled=(
                            st.session_state.max_node_id >= st.session_state.total_nodes - 1
                        ),
                    )
                with col4:
                    st.button(
                        "⏩ Show Full Tree",
                        on_click=lambda: setattr(
                            st.session_state,
                            "max_node_id",
                            st.session_state.total_nodes - 1,
                        )
                        or setattr(
                            st.session_state,
                            "selected_node_id",
                            st.session_state.max_node_id,
                        ),
                        disabled=(
                            st.session_state.max_node_id == st.session_state.total_nodes - 1
                        ),
                    )
                with col5:
                    if st.session_state.auto_play:
                        if st.button("⏹ Stop Auto-play"):
                            st.session_state.auto_play = False
                    else:
                        if st.button("▶️ Start Auto-play"):
                            st.session_state.auto_play = True
                            st.rerun()

                with col6:
                    st.write(f"Showing nodes up to ID: {st.session_state.max_node_id}")
                    st.write(
                        "Auto-play: " + ("On" if st.session_state.auto_play else "Off")
                    )

                chart_placeholder = st.empty()
                event = chart_placeholder.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    on_select="rerun",
                )

                if event and "selection" in event and "points" in event["selection"]:
                    selected_points = event["selection"]["points"]
                    if selected_points:
                        point_index = selected_points[0]["point_index"]
                        if point_index in point_to_node_id:
                            node_id = point_to_node_id[point_index]
                            st.session_state.selected_node_id = int(
                                node_id.split("Node")[1]
                            )

            with info_col:
                # Update the node selection dropdown
                max_visible_node = min(
                    int(st.session_state.max_node_id), st.session_state.total_nodes - 1
                )
                node_options = [f"Node{i}" for i in range(max_visible_node + 1)]

                selected_node_option = st.selectbox(
                    "Select Node",
                    node_options,
                    index=node_options.index(f"Node{st.session_state.selected_node_id}"),
                    key="selected_node_option",
                )
                if selected_node_option:
                    st.session_state.selected_node_id = int(
                        selected_node_option.split("Node")[1]
                    )

                # Display selected node information
                if st.session_state.selected_node_id is not None:
                    node_id = st.session_state.selected_node_id
                    selected_node = find_node_by_id(search_tree.root, node_id)

                    tabs = ["Summary"]

                    if not selected_node or selected_node.node_id == 0:
                        if eval_result and eval_result.get("error"):
                            tabs.append("Error")

                    if selected_node:
                        if selected_node.file_context:
                            tabs.append("FileContext")

                        if (
                            selected_node.action
                            and selected_node.completions.get("build_action") is not None
                        ):
                            tabs.append("Build")

                        if (
                            selected_node.action
                            and selected_node.completions.get("execute_action") is not None
                        ):
                            tabs.append("Execution")

                    # Check both current node and parent for selector completion
                    has_selector = (
                        "selector" in selected_node.completions or 
                        (selected_node.parent and "selector" in selected_node.parent.completions)
                    )
                    if has_selector:
                        tabs.append("Selector")

                    if eval_result and str(selected_node.node_id) in eval_result.get(
                        "node_results", {}
                    ):
                        tabs.append("Evaluation")
                        if selected_node.reward:
                            tabs.append("Reward")

                        if (
                            eval_result
                            and str(selected_node.node_id)
                            in eval_result.get("node_results", {})
                            is not None
                        ):
                            tabs.append("Evaluation")

                        tabs.append("JSON")

                        if selected_node.action:
                            tabs.append("Rerun")

                        if instance:
                            tabs.append("Instance")

                        if selected_node.action:
                            tabs.append("Rerun")

                        tab_contents = st.tabs(tabs)

                        with tab_contents[tabs.index("Summary")]:
                            troubleshoot_url = f"?path={selected_tree_path}&node_id={selected_node.node_id}"
                            st.markdown(
                                f'<a href="{troubleshoot_url}" target="_blank">'
                                '<button style="width:80%; margin:0 10%; padding:4px; background-color:#4a4a4a; color:white; border:none; border-radius:4px; cursor:pointer; font-size:0.9em;">'
                                '🔍 Troubleshoot Node'
                                '</button></a>', 
                                unsafe_allow_html=True
                            )
    
                            if selected_node.action:
                                if selected_node.message:
                                    st.write(selected_node.message)

                                if (
                                    hasattr(selected_node.action, "thoughts")
                                    and selected_node.action.thoughts
                                ):
                                    st.subheader("Thoughts")
                                    st.write(selected_node.action.thoughts)

                                st.subheader(f"Action: {selected_node.action.name}")
                                st.json(
                                    selected_node.action.model_dump(exclude={"thoughts"})
                                )

                                if selected_node.observation:
                                    st.subheader("Output")
                                    st.code(selected_node.observation.message)

                        # Add feedback display
                        if selected_node.feedback:
                            st.subheader("Feedback")
                            st.write(selected_node.feedback)
                        
                        if selected_node.message:
                            st.write(selected_node.message)
                            st.subheader(f"Action: {selected_node.action.name}")
                            st.json(
                                selected_node.action.model_dump(exclude={"scratch_pad"})
                            )

                        if selected_node.observation:
                            st.subheader("Output")
                            st.code(selected_node.observation.message)

                        if selected_node.parent:
                            updated_context = (
                                selected_node.file_context.get_context_diff(
                                    selected_node.parent.file_context
                                )
                            )
                            if not updated_context.is_empty():
                                st.subheader("Updated Context")
                                st.json(updated_context.model_dump())

                        if (
                            selected_node.action.name == "Reject"
                            and selected_node.observation.properties
                            and selected_node.observation.properties.get(
                                "last_completion"
                            )
                        ):
                            st.subheader("Last completion")
                            st.json(
                                selected_node.observation.properties.get(
                                    "last_completion"
                                )
                            )

                        if "Error" in tabs:
                            with tab_contents[tabs.index("Error")]:
                                st.code(eval_result["error"])

                        if "FileContext" in tabs:
                            with tab_contents[tabs.index("FileContext")]:
                                st.json(selected_node.file_context.model_dump())

                        if "Build" in tabs:
                            with tab_contents[tabs.index("Build")]:
                                completion = selected_node.completions.get("build_action")
                                show_completion(completion)

                        if "Execution" in tabs:
                            with tab_contents[tabs.index("Execution")]:
                                completion = selected_node.completions.get("execute_action")
                                show_completion(completion)

                        if "Reward" in tabs:
                            with tab_contents[tabs.index("Reward")]:
                                st.subheader(f"Reward: {selected_node.reward.value}")
                                st.write(selected_node.reward.explanation)
                                st.subheader("Completion")
                                show_completion(
                                    selected_node.completions.get("value_function")
                                )

                        if "Evaluation" in tabs:
                            with tab_contents[tabs.index("Evaluation")]:
                                node_result = eval_result["node_results"].get(str(node_id))
                                if node_result:
                                    st.json(node_result)

                        with tab_contents[tabs.index("JSON")]:
                            st.json(
                                selected_node.model_dump(exclude={"parent", "children"})
                            )

                        if "Instance" in tabs:
                            with tab_contents[tabs.index("Instance")]:
                                st.json(instance)

                    if "Selector" in tabs:
                        with tab_contents[tabs.index("Selector")]:
                            # Try to get completion from current node or parent
                            selector_completion = (
                                selected_node.completions.get("selector") or 
                                (selected_node.parent.completions.get("selector") if selected_node.parent else None)
                            )

                            if selector_completion:
                                st.subheader("LLM Selection Process")
                                show_completion(selector_completion)

                                # Show the selection result
                                if selected_node.reward and selected_node.reward.feedback:
                                    st.subheader("Selection Result")
                                    st.markdown(f"""
                                    **Selected Node:** Node{selected_node.node_id}  
                                    **Explanation:** {selected_node.reward.feedback}
                                    """)

                                # Show which node made the selection
                                completion_owner = (
                                    "current node" 
                                    if "selector" in selected_node.completions 
                                    else "parent node"
                                )
                                st.info(f"This selection was made by the {completion_owner}.")

                    with tab_contents[tabs.index("JSON")]:
                        st.json(
                            selected_node.model_dump(exclude={"parent", "children"})
                        )
                        if "Rerun" in tabs:
                            with tab_contents[tabs.index("Rerun")]:
                                rerun_node(
                                    selected_node.node_id,
                                    search_tree.persist_path,
                                    instance,
                                )

                else:
                    st.info(
                        "Select a node in the graph or from the dropdown to view details"
                    )

            # Auto-play logic
            if (
                st.session_state.auto_play
                and st.session_state.max_node_id < st.session_state.total_nodes - 1
            ):
                st.session_state.max_node_id += 1
                st.session_state.selected_node_id = st.session_state.max_node_id
                time.sleep(1)  # Delay between steps
                st.rerun()


def find_node_by_id(root_node: Node, node_id: int) -> Optional[Node]:
    if root_node.node_id == node_id:
        return root_node
    for child in root_node.children:
        found = find_node_by_id(child, node_id)
        if found:
            return found
    return None


# Helper function to count total nodes in the tree
def count_total_nodes(root_node):
    count = 1
    for child in root_node.children:
        count += count_total_nodes(child)
    return count
