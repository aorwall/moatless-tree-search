import logging
import streamlit as st
import os
from datetime import datetime, timezone
import httpx
import asyncio
from typing import Optional
import pandas as pd
import time
import json

from moatless.benchmark.evaluation_v2 import (
    EvaluationStatus,
    InstanceStatus
)
from moatless_tools.evaluation_worker import EvaluationRequest



# Initialize session state
if "instance_progress" not in st.session_state:
    st.session_state.instance_progress = {}
if "last_event_time" not in st.session_state:
    st.session_state.last_event_time = None

WORKER_URL = "http://localhost:8000"

async def create_evaluation(
    model: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
    max_expansions: int,
    max_cost: float,
    instance_ids: list[str],
    repo_base_dir: str,
    min_resolved: int
):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{WORKER_URL}/evaluations/", json={
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "max_iterations": max_iterations,
            "max_expansions": max_expansions,
            "max_cost": max_cost,
            "instance_ids": instance_ids,
            "repo_base_dir": repo_base_dir,
            "min_resolved": min_resolved
        })
        return response.json()

async def get_evaluation_status(evaluation_name: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{WORKER_URL}/evaluations/{evaluation_name}")
        return response.json()

async def get_evaluation_events(evaluation_name: str, since: Optional[str] = None):
    async with httpx.AsyncClient() as client:
        params = {"since": since} if since else None
        response = await client.get(
            f"{WORKER_URL}/evaluations/{evaluation_name}/events",
            params=params
        )
        return response.json()

async def list_evaluations():
    # First try to get from server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{WORKER_URL}/evaluations")
            return response.json()
    except Exception as e:
        print(f"Server error: {e}")
        # If server is not available, read from local files
        evaluations = []
        evals_dir = "evals"
        
        print(f"Looking for evaluations in {os.path.abspath(evals_dir)}")
        if not os.path.exists(evals_dir):
            print(f"Directory {evals_dir} does not exist!")
            return {"evaluations": []}
            
        eval_dirs = os.listdir(evals_dir)
        print(f"Found directories: {eval_dirs}")
        
        for eval_dir in eval_dirs:
            eval_path = os.path.join(evals_dir, eval_dir, "evaluation.json")
            print(f"Checking {eval_path}")
            if not os.path.exists(eval_path):
                print(f"File does not exist: {eval_path}")
                continue
                
            try:
                print(f"Reading file: {eval_path}")
                with open(eval_path, 'r') as f:
                    file_content = f.read()
                    print(f"File content length: {len(file_content)}")
                    eval_data = json.loads(file_content)
                    print(f"Successfully parsed JSON from {eval_path}")
                    
                    instances = eval_data.get("instances", {})
                    print(f"Found {len(instances)} instances")
                    
                    evaluation = {
                        "name": eval_data["evaluation_name"],
                        "model": eval_data["settings"]["model"]["model"],
                        "status": "completed",  # Since we have the file
                        "total_instances": len(instances),
                        "completed_instances": sum(1 for inst in instances.values() 
                                                if inst["status"] == "completed"),
                        "resolved_instances": sum(1 for inst in instances.values() 
                                               if inst.get("resolved", False)),
                        "started_at": eval_data.get("start_time")
                    }
                    print(f"Created evaluation entry: {evaluation}")
                    evaluations.append(evaluation)
                    print(f"Successfully loaded {eval_path}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {eval_path}: {e}")
                print(f"Content preview: {file_content[:200]}...")
            except Exception as e:
                print(f"Error reading {eval_path}: {type(e).__name__}: {e}")
        
        print(f"Total evaluations found: {len(evaluations)}")
        return {"evaluations": evaluations}

async def start_evaluation(evaluation_name: str, rerun_errors: bool = False):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{WORKER_URL}/evaluations/{evaluation_name}/start", json={
            "rerun_errors": rerun_errors
        })
        return response.json()

def show_evaluations_list():
    st.title("MoatLess Evaluations")
    
    # Add button to create new evaluation
    if st.button("Create New Evaluation", type="primary"):
        st.query_params["page"] = "new"
        st.rerun()
    
    # Get list of evaluations
    evaluations = asyncio.run(list_evaluations())
    
    if not evaluations["evaluations"]:
        st.info("No evaluations found. Create a new one to get started!")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(evaluations["evaluations"])
    df["progress"] = df.apply(lambda x: f"{x['completed_instances']}/{x['total_instances']}", axis=1)
    df["resolved"] = df.apply(lambda x: f"{x['resolved_instances']}/{x['total_instances']}", axis=1)
    
    # Format the DataFrame
    display_df = df[[
        "name", "model", "status", "progress", "resolved", "started_at"
    ]].rename(columns={
        "name": "Evaluation",
        "model": "Model",
        "status": "Status",
        "progress": "Progress",
        "resolved": "Resolved",
        "started_at": "Started At"
    })
    
    # Show as a table with clickable links
    st.dataframe(
        display_df,
        column_config={
            "Evaluation": st.column_config.Column(
                "Evaluation",
                help="Click to view evaluation details"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Add clickable links below the table
    for _, row in display_df.iterrows():
        eval_name = row["Evaluation"]
        st.markdown(f"[{eval_name}](?page=evaluation&name={eval_name})")

def show_setup_form():
    st.title("Create New Evaluation")
    
    # Add back button
    if st.button("← Back to List"):
        st.query_params.clear()
        st.rerun()
    
    with st.form("evaluation_setup"):
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.text_input("Model Name", value="gemini/gemini-2.0-flash-exp")
            api_key = st.text_input("API Key (optional)", value="", type="password")
            base_url = st.text_input("Base URL (optional)", value="")
            
            max_iterations = st.number_input("Max Iterations", value=20, min_value=1)
            max_expansions = st.number_input("Max Expansions", value=1, min_value=1)
            max_cost = st.number_input("Max Cost", value=1.0, min_value=0.1)
            min_resolved = st.number_input("Min Resolved", value=20, min_value=1)
            
        with col2:
            repo_base_dir = st.text_input("Repository Base Directory", value="/tmp/repos")
            
            instance_ids = st.text_area(
                "Instance IDs (one per line, optional)", 
                value="scikit-learn__scikit-learn-14894"
            ).strip()
            instance_ids = instance_ids.split('\n') if instance_ids else []

        submitted = st.form_submit_button("Start Evaluation")
        
        if submitted:
            with st.spinner("Starting evaluation..."):
                result = asyncio.run(create_evaluation(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    max_iterations=max_iterations,
                    max_expansions=max_expansions,
                    max_cost=max_cost,
                    instance_ids=instance_ids,
                    repo_base_dir=repo_base_dir,
                    min_resolved=min_resolved
                ))
            
            # Redirect to evaluation page
            st.query_params["page"] = "evaluation"
            st.query_params["name"] = result["evaluation_name"]
            st.rerun()

def show_evaluation_progress(evaluation_name: str):
    st.title("Evaluation Progress")
    
    # Add back button
    if st.button("← Back to List"):
        st.query_params.clear()
        st.rerun()
    
    # Get current status
    logger.info(f"Fetching status for evaluation: {evaluation_name}")
    status = asyncio.run(get_evaluation_status(evaluation_name))
    if "error" in status:
        logger.error(f"Failed to get evaluation status: {status['error']}")
        st.error("Evaluation not found!")
        return
    
    logger.info(f"Retrieved status with {len(status.get('instances', {}))} instances")
    
    # Show settings in a collapsed section
    with st.expander("Evaluation Settings"):
        settings = status["settings"]
        st.write("Model:", settings["model"])
        st.write("Max Iterations:", settings["max_iterations"])
        st.write("Max Expansions:", settings["max_expansions"])
        st.write("Max Cost:", settings["max_cost"])
    
    # Process any new events
    events = asyncio.run(get_evaluation_events(
        evaluation_name,
        since=st.session_state.get("last_event_time")
    ))
    
    for event in events["events"]:
        if event["type"] == "instance_completed":
            st.toast(f"Instance completed: {event['data'].get('instance_id')}")
        elif event["type"] == "instance_error":
            st.toast(f"Error in instance: {event['data'].get('error')}", icon="🚨")
        elif event["type"] == "tree_progress":
            # Update instance progress in session state
            instance_id = event["data"]["instance_id"]
            if "instance_progress" not in st.session_state:
                st.session_state.instance_progress = {}
            st.session_state.instance_progress[instance_id] = event["data"]
        st.session_state["last_event_time"] = event["timestamp"]
    
    # Show status
    if status["status"] == EvaluationStatus.RUNNING:
        if status.get("is_active", False):
            st.info("Evaluation is running...")
        else:
            st.warning("Evaluation was interrupted")
            if st.button("Resume Evaluation"):
                with st.spinner("Resuming evaluation..."):
                    result = asyncio.run(start_evaluation(evaluation_name))
                    if "error" in result:
                        st.error(result["error"])
                        return
                    st.rerun()
    elif status["status"] == EvaluationStatus.COMPLETED:
        st.success("Evaluation completed!")
    elif status["status"] == EvaluationStatus.ERROR:
        st.error("Evaluation encountered errors")
    else:
        # Check if any instances were started but not completed
        has_started_instances = any(inst.get("started_at") for inst in status["instances"].values())
        
        if has_started_instances:
            st.warning("Evaluation was interrupted")
            button_text = "Resume Evaluation"
        else:
            st.warning("Evaluation not started yet")
            button_text = "Start Evaluation"
        
        # Add checkbox for rerunning error instances
        rerun_errors = st.checkbox("Rerun instances with errors", value=False)
        
        if st.button(button_text):
            with st.spinner(f"{button_text}..."):
                result = asyncio.run(start_evaluation(evaluation_name, rerun_errors=rerun_errors))
                if "error" in result:
                    st.error(result["error"])
                    return
                st.rerun()
    
    # Show instance progress
    instances = status.get("instances", {})
    logger.info(f"Processing {len(instances)} instances for display")
    completed = sum(1 for inst in instances.values() 
                   if inst["status"] == InstanceStatus.COMPLETED)
    total = len(instances)
    
    st.progress(completed / total if total > 0 else 0)
    st.write(f"Progress: {completed}/{total} instances completed")
    
    # Show active status
    if status.get("is_active", False):
        st.info("Server is actively processing this evaluation")
    else:
        if status["status"] == EvaluationStatus.RUNNING:
            st.warning("Server is not actively processing this evaluation - it may need to be resumed")
    
    # Show instances in a table with tree search progress
    instances_data = []
    for instance_id, instance in instances.items():
        logger.debug(f"Processing instance data for {instance_id}")
        progress = st.session_state.instance_progress.get(instance_id, {})
        benchmark_result = instance.get('benchmark_result', {})
        
        # Get token counts from benchmark result
        token_usage = benchmark_result.get('token_usage', {}) if benchmark_result else {}
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        
        instances_data.append({
            "Instance": instance_id,
            "Status": instance["status"],
            "Started": instance["started_at"],
            "Completed": instance["completed_at"],
            "Duration": f"{instance['duration']:.2f}s" if instance["duration"] else None,
            "Resolved": "Yes" if instance["resolved"] else "No",
            "Error": instance["error"] or "",
            "Tree Progress": f"{progress.get('iteration', 0)}/{settings['max_iterations']}",
            "Total Cost": f"${progress.get('total_cost', 0):.4f}" if progress.get('total_cost') else None,
            "Transitions": benchmark_result.get('transitions', 0) if benchmark_result else 0,
            "Prompt/Completion": f"{prompt_tokens}/{completion_tokens}"
        })
        logger.debug(f"Added instance {instance_id} to display data")
    
    logger.info(f"Displaying table with {len(instances_data)} instances")
    df = pd.DataFrame(instances_data)
    st.dataframe(
        df,
        column_config={
            "Instance": st.column_config.Column(
                "Instance",
                help="Instance ID",
                width="medium"
            ),
            "Error": st.column_config.Column(
                "Error",
                help="Error message if any",
                width="large"
            ),
            "Tree Progress": st.column_config.ProgressColumn(
                "Tree Progress",
                help="Search tree iteration progress",
                min_value=0,
                max_value=settings["max_iterations"],
                format="%d iterations"
            ),
            "Best Reward": st.column_config.NumberColumn(
                "Best Reward",
                help="Best reward found so far",
                format="%.2f"
            ),
            "Total Cost": st.column_config.NumberColumn(
                "Total Cost",
                help="Total cost spent on tokens",
                format="$%.4f"
            ),
            "Transitions": st.column_config.NumberColumn(
                "Transitions",
                help="Number of transitions in the search tree",
                format="%d"
            ),
            "Total Tokens": st.column_config.NumberColumn(
                "Total Tokens",
                help="Total number of tokens used (prompt + completion)",
                format="%d"
            ),
            "Prompt/Completion": st.column_config.Column(
                "Prompt/Completion",
                help="Number of prompt tokens / completion tokens"
            )
        },
        hide_index=True
    )

    # Show elapsed time
    if status["started_at"]:
        started = datetime.fromisoformat(status["started_at"])
        elapsed = datetime.now(timezone.utc) - started
        st.write(f"Total elapsed time: {elapsed}")
    
    # Auto-refresh while running
    if status["status"] in [EvaluationStatus.RUNNING, EvaluationStatus.PENDING] and status.get("is_active", False):
        time.sleep(1)  # Small delay to prevent too frequent updates
        st.rerun()

def setup_page():
    st.set_page_config(
        page_title="MoatLess Evaluation",
        page_icon="🌳",
        layout="wide"
    )
    
    # Get current page from query params
    page = st.query_params.get("page", "list")
    
    if page == "list":
        show_evaluations_list()
    elif page == "new":
        show_setup_form()
    elif page == "evaluation":
        evaluation_name = st.query_params.get("name")
        if not evaluation_name:
            st.error("No evaluation name provided!")
            return
        show_evaluation_progress(evaluation_name)
    else:
        st.error("Invalid page!")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    # Silence httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    setup_page()
