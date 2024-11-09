import logging
import os

import altair as alt
import pandas as pd
import streamlit as st

from moatless.benchmark.report import read_reports, to_dataframe

logger = logging.getLogger(__name__)


def generate_summary(df: pd.DataFrame):
    total_trajectories = len(df)
    status_counts = df["success_status"].value_counts()

    total_cost = df["total_cost"].sum()
    total_prompt_tokens = df["prompt_tokens"].sum()
    total_completion_tokens = df["completion_tokens"].sum()
    total_cached_tokens = df["cached_tokens"].sum()
    avg_cost = df["total_cost"].mean()

    # Filter out 'running', 'error', and 'rejected' statuses for avg_duration calculation
    filtered_df = df[~df["status"].isin(["running", "error", "rejected"])]
    avg_duration = filtered_df["duration"].mean()

    col_metrics, col_chart = st.columns([3, 2])

    with col_metrics:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Trajectories", total_trajectories)
            st.metric("Total Cost", f"${total_cost:.2f}")
            st.metric("Total Prompt Tokens", total_prompt_tokens)
            st.metric("Total Completion Tokens", total_completion_tokens)
            st.metric("Total Cached Tokens", total_cached_tokens)
        with metric_col2:
            st.metric("Avg Cost per Trajectory", f"${avg_cost:.2f}")
            st.metric(
                "Avg Duration (excl. running/error/rejected)", f"{avg_duration:.2f} s"
            )

    with col_chart:
        status_data = pd.DataFrame(
            {"Status": status_counts.index, "Count": status_counts.values}
        )

        status_data["Status_with_Count"] = status_data.apply(
            lambda row: f"{row['Status']} ({row['Count']})", axis=1
        )

        pie_chart = (
            alt.Chart(status_data)
            .mark_arc()
            .encode(
                theta="Count",
                color=alt.Color(
                    "Status_with_Count",
                    scale=alt.Scale(
                        domain=[
                            f'Resolved ({status_counts.get("Resolved", 0)})',
                            f'Running with Resolved Solutions ({status_counts.get("Running with Resolved Solutions", 0)})',
                            f'Partially Resolved ({status_counts.get("Partially Resolved", 0)})',
                            f'Running ({status_counts.get("Running", 0)})',
                            f'Failed ({status_counts.get("Failed", 0)})',
                            f'Rejected ({status_counts.get("Rejected", 0)})',
                        ],
                        range=[
                            "#4CAF50",
                            "#009688",
                            "#FFC107",
                            "#2196F3",
                            "#F44336",
                            "#A52A2A",
                        ],
                    ),
                ),
                tooltip=["Status", "Count"],
            )
            .properties(title="Trajectory Status Distribution", width=300, height=300)
        )

        # Add text labels to the pie chart
        text = pie_chart.mark_text(radiusOffset=20).encode(
            text=alt.Text("Count:Q", format=".0f"), color=alt.value("white")
        )

        # Combine the pie chart and text labels
        final_chart = (pie_chart + text).interactive()

        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)


def trajectory_table(report_path: str):
    st.header("Trajectory List")

    if not os.path.exists(report_path):
        logger.error(f"Report file not found at {report_path}")
        return

    results = read_reports(report_path)
    logger.info(f"Loaded {len(results)} trajectory reports")
    df = to_dataframe(results)

    # Add a new column for success status
    df["success_status"] = df.apply(
        lambda row: "Resolved"
        if row["status"] == "resolved"
        else "Running with Resolved Solutions"
        if row["status"] == "running" and row["resolved_solutions"] > 0
        else "Partially Resolved"
        if row["resolved_solutions"] > 0
        else "Rejected"
        if row["status"] == "rejected"
        else "Running"
        if row["status"] == "running"
        else "Failed",
        axis=1,
    )

    # Format llmonkeys_rate as a percentage
    df["llmonkeys_rate"] = df["llmonkeys_rate"].apply(lambda x: f"{x * 100:.2f}%")

    # Display summary
    generate_summary(df)

    # Add filtering options
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        status_filter = st.multiselect(
            "Status", df["status"].unique(), key="status_filter"
        )
    with col2:
        has_resolved_solutions = st.multiselect("Has Resolved Solutions", ["Yes", "No"])
    with col3:
        instance_filter = st.multiselect(
            "Instance", df["instance_id"].unique(), key="instance_filter"
        )
    with col4:
        llmonkeys_rate_range = st.slider("LLMonkeys Rate (%)", 0, 100, (0, 100), 1)

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if status_filter:
        mask &= df["status"].isin(status_filter)
    if instance_filter:
        mask &= df["instance_id"].isin(instance_filter)
    if has_resolved_solutions:
        if "Yes" in has_resolved_solutions:
            mask &= df["resolved_solutions"] > 0
        if "No" in has_resolved_solutions:
            mask &= df["resolved_solutions"] == 0

    if int(df["resolved_by"].min()) < int(df["resolved_by"].max()):
        with col5:
            resolved_by_range = st.slider(
                "Resolved By",
                int(df["resolved_by"].min()),
                int(df["resolved_by"].max()),
                (int(df["resolved_by"].min()), int(df["resolved_by"].max())),
            )

        # Apply new filters
        mask &= df["llmonkeys_rate"].apply(
            lambda x: float(x.strip("%")) >= llmonkeys_rate_range[0]
            and float(x.strip("%")) <= llmonkeys_rate_range[1]
        )
        mask &= df["resolved_by"].between(resolved_by_range[0], resolved_by_range[1])

    filtered_df = df[mask]

    directory_path = os.path.dirname(report_path)
    # Create a column with clickable links using trajectory_path
    filtered_df["Select"] = filtered_df.apply(
        lambda row: f'<a href="?path={directory_path}/{row["instance_id"]}/trajectory.json">View</a>',
        axis=1,
    )

    # Remove trajectory_path from the columns to be displayed
    display_columns = [
        "Select",
        "instance_id",
        "resolved_by",
        "llmonkeys_rate",
        "success_status",
        "solutions",
        "resolved_solutions",
        "failed_solutions",
        "resolved_max_reward",
        "failed_max_reward",
        "status",
        "all_transitions",
        "failed_actions",
        "duration",
        "prompt_tokens",
        "completion_tokens",
    ]

    filtered_df = filtered_df.sort_values(by="instance_id")

    # Function to apply color to rows based on success_status
    def color_rows(row):
        if row["success_status"] == "Rejected":
            return ["background-color: rgba(165, 42, 42, 0.3)"] * len(
                row
            )  # Brown for rejected
        elif row["success_status"] == "Resolved":
            return ["background-color: rgba(76, 175, 80, 0.3)"] * len(row)
        elif row["success_status"] == "Running with Resolved Solutions":
            return ["background-color: rgba(0, 150, 136, 0.3)"] * len(row)
        elif row["success_status"] == "Partially Resolved":
            return ["background-color: rgba(255, 235, 59, 0.3)"] * len(row)
        elif row["success_status"] == "Running":
            return ["background-color: rgba(33, 150, 243, 0.3)"] * len(row)
        else:
            return ["background-color: rgba(244, 67, 54, 0.3)"] * len(row)

    # Apply the styling
    styled_df = filtered_df[display_columns].style.apply(color_rows, axis=1)

    # Display the filtered and styled dataframe
    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Determine if dark mode is active
    is_dark_mode = st.get_option("theme.base") == "dark"

    # Add CSS to style the table
    st.markdown(
        f"""
    <style>
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th, td {{
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid {'#444' if is_dark_mode else '#ddd'};
    }}
    tr:hover {{background-color: {'rgba(255,255,255,0.1)' if is_dark_mode else 'rgba(0,0,0,0.05)'}; }}
    a {{
        color: {'#58a6ff' if is_dark_mode else '#0366d6'};
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
