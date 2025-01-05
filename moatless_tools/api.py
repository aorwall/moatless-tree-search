import os
from pathlib import Path
import logging
from moatless.benchmark.evaluation_v2 import (
    TreeSearchSettings, Evaluation, create_evaluation_name,
    InstanceStatus, EvaluationStatus, EvaluationEvent, EvaluationRunner
)
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.benchmark.evaluation_factory import create_evaluation
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import CompletionModel
from moatless.completion.completion import LLMResponseFormat
from moatless.schema import MessageHistoryType
from moatless.agent.settings import AgentSettings
from moatless.node import Node as MoatlessNode
import shutil
import concurrent.futures
from tqdm import tqdm
import traceback
from datetime import timezone, datetime
import json
import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from dotenv import load_dotenv
import glob
from fastapi.middleware.cors import CORSMiddleware

from moatless.search_tree import SearchTree
from moatless_tools.schema import (
    EvaluationListItemDTO, EvaluationListResponseDTO, EvaluationResponseDTO, 
    EvaluationSettingsDTO, InstanceItemDTO, InstanceResponseDTO,
    NodeDTO, ActionDTO, ObservationDTO, CompletionDTO, FileContextDTO,
    UsageDTO, ActionStepDTO,
)
from moatless_tools.utils import create_evaluation_response, create_instance_response


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

load_dotenv()

app = FastAPI(title="Moatless Evaluation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variable to store instance splits
instance_splits: Dict[str, List[str]] = {}

# Global cache for evaluation items
cached_evaluations: Dict[str, EvaluationListItemDTO] = {}

def load_dataset_splits():
    """Load all dataset splits and map instance IDs to their splits."""
    global instance_splits
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    
    for dataset_file in glob.glob(os.path.join(datasets_dir, "*_dataset.json")):
        try:
            with open(dataset_file) as f:
                dataset = json.load(f)
                split_name = dataset["name"]
                
                # Add this split to each instance's list of splits
                for instance_id in dataset["instance_ids"]:
                    if instance_id not in instance_splits:
                        instance_splits[instance_id] = []
                    instance_splits[instance_id].append(split_name)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_file}: {e}")

# Load datasets at startup
load_dataset_splits()

def get_evaluations_dir():
    """Get the current evaluations directory from app state."""
    return os.getenv("MOATLESS_DIR", "./evals")

repository: EvaluationFileRepository = None

def get_repository() -> EvaluationFileRepository:
    """Get repository instance with current evaluations directory."""
    global repository
    if repository is None:
        repository = EvaluationFileRepository(get_evaluations_dir())
    return repository

def load_resolution_rates():
    """Load resolution rates from the resolved submissions data."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "resolved_submissions.json")
    try:
        with open(dataset_path) as f:
            resolved_data = json.load(f)
            return {
                instance_id: len(data["resolved_submissions"]) / data["no_of_submissions"] 
                if data["no_of_submissions"] > 0 else 0.0
                for instance_id, data in resolved_data.items()
            }
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load resolution rates data")
        return {}

@app.get("/evaluations", response_model=EvaluationListResponseDTO)
async def list_evaluations():
    """List all evaluations from the repository."""
    global cached_evaluations
    eval_dir = get_evaluations_dir()
    
    # Get all evaluation directories
    eval_dirs = set(d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d)))
    cached_names = set(cached_evaluations.keys())
    
    # Find new evaluations and ones to update
    new_evals = eval_dirs - cached_names
    to_update = {name for name in cached_names if name in eval_dirs and cached_evaluations[name].status == "running"}
    
    # Process new and running evaluations
    for eval_name in new_evals | to_update:
        try:
            response_file = os.path.join(eval_dir, eval_name, "evaluation_response.json")
            if not os.path.exists(response_file):
                continue
                
            with open(response_file, "r") as f:
                eval_response = EvaluationResponseDTO(**json.load(f))
                
            # Calculate derived metrics
            resolution_rate = eval_response.resolvedInstances / eval_response.totalInstances if eval_response.totalInstances > 0 else 0.0
            resolved_by_dollar = eval_response.resolvedInstances / eval_response.totalCost if eval_response.totalCost > 0 else 0.0
                
            # Map to list item
            list_item = EvaluationListItemDTO(
                name=eval_name,
                status=eval_response.status,
                model=eval_response.settings.model,
                maxExpansions=eval_response.settings.maxIterations,
                startedAt=eval_response.startedAt,
                totalInstances=eval_response.totalInstances,
                completedInstances=eval_response.completedInstances,
                errorInstances=eval_response.errorInstances,
                resolvedInstances=eval_response.resolvedInstances,
                isActive=eval_response.isActive,
                date=eval_response.startedAt,
                resolutionRate=resolution_rate,
                totalCost=eval_response.totalCost,
                promptTokens=eval_response.promptTokens,
                completionTokens=eval_response.completionTokens,
                resolvedByDollar=resolved_by_dollar
            )
            cached_evaluations[eval_name] = list_item
            
        except Exception as e:
            logger.exception(f"Failed to load evaluation {eval_name}: {e}")
    
    # Remove cached items that no longer exist
    for eval_name in cached_names - eval_dirs:
        del cached_evaluations[eval_name]
    
    # Sort evaluations by date (desc) and name (asc)
    sorted_evaluations = sorted(
        cached_evaluations.values(),
        key=lambda x: (-(x.date.timestamp() if x.date else 0), x.name)
    )
    
    return EvaluationListResponseDTO(evaluations=sorted_evaluations)


@app.get("/evaluations/{evaluation_name}", response_model=EvaluationResponseDTO)
async def get_evaluation(evaluation_name: str):
    logger.info(f"Getting status for evaluation: {evaluation_name}")
    
    # Construct path to evaluation response file
    eval_dir = get_evaluations_dir()
    response_file = os.path.join(eval_dir, evaluation_name, "evaluation_response.json")
    
    # Check if file exists
    if not os.path.exists(response_file):
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
    
    try:
        # Read the evaluation response file
        with open(response_file, "r") as f:
            evaluation_response = json.load(f)
            return EvaluationResponseDTO(**evaluation_response)
    except Exception as e:
        logger.exception(f"Failed to load evaluation response for {evaluation_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load evaluation response")


@app.get("/evaluations/{evaluation_name}/instances/{instance_id}", response_model=InstanceResponseDTO)
async def get_instance(evaluation_name: str, instance_id: str):
    """Get the tree visualization data for an evaluation."""
    logger.info(f"Getting tree data for evaluation: {evaluation_name}")

    eval_dir = get_evaluations_dir()
    instance_dir = os.path.join(eval_dir, evaluation_name, instance_id)
    
    # Load full instance response
    instance_response_path = os.path.join(instance_dir, "instance_response.json")
    if not os.path.exists(instance_response_path):
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        
    try:
        with open(instance_response_path, "r") as f:
            instance_response = json.load(f)
            return InstanceResponseDTO(**instance_response)
    except Exception as e:
        logger.exception(f"Failed to load instance response: {e}")
        raise HTTPException(status_code=500, detail="Failed to load instance data")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
    ) 