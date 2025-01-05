import os
from pathlib import Path
import logging
from moatless.benchmark.evaluation_v2 import (
    TreeSearchSettings, Evaluation, create_evaluation_name,
    InstanceStatus, EvaluationStatus, EvaluationEvent, EvaluationRunner
)
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.benchmark.evaluation_factory import create_evaluation
from moatless.completion import CompletionModel
from moatless.completion.completion import LLMResponseFormat
from moatless.schema import MessageHistoryType
from moatless.agent.settings import AgentSettings
import shutil
import concurrent.futures
from tqdm import tqdm
import traceback
from datetime import timezone, datetime
import json
import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from dotenv import load_dotenv
import glob


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

load_dotenv()

app = FastAPI(title="Moatless Evaluation API")

class EvaluationListItem(BaseModel):
    """Represents an evaluation item in the list view."""
    name: str
    status: EvaluationStatus
    model: str
    maxExpansions: int
    startedAt: Optional[datetime]
    totalInstances: int
    completedInstances: int
    errorInstances: int
    resolvedInstances: int
    isActive: bool

    @classmethod
    def from_evaluation(cls, name: str, evaluation: Evaluation, repository: EvaluationFileRepository, is_active: bool) -> 'EvaluationListItem':
        """Create an EvaluationListItem from an Evaluation."""
        instances = repository.list_instances(name)
        
        # Determine actual status
        status = evaluation.status
        if status == EvaluationStatus.RUNNING and not is_active:
            status = EvaluationStatus.PENDING
            
        return cls(
            name=name,
            status=status,
            model=evaluation.settings.model.model,
            maxExpansions=evaluation.settings.max_expansions,
            startedAt=evaluation.start_time if hasattr(evaluation, 'start_time') else None,
            totalInstances=len(instances),
            completedInstances=sum(1 for inst in instances if inst.status == InstanceStatus.COMPLETED),
            errorInstances=sum(1 for inst in instances if inst.status == InstanceStatus.ERROR),
            resolvedInstances=sum(1 for inst in instances if inst.resolved),
            isActive=is_active
        )

class EvaluationListResponse(BaseModel):
    """Response model for list evaluations endpoint."""
    evaluations: List[EvaluationListItem]

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

# Store running evaluations
evaluations: Dict[str, Evaluation] = {}
evaluation_events: Dict[str, List[dict]] = {}
# Track actively running evaluations
active_evaluations: Dict[str, bool] = {}

class EvaluationRequest(BaseModel):
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_iterations: int
    max_expansions: int
    max_cost: float
    instance_ids: Optional[List[str]] = None
    repo_base_dir: Optional[str] = None
    min_resolved: int

class StartEvaluationRequest(BaseModel):
    rerun_errors: bool = False

def load_existing_evaluations():
    """Load existing evaluations from the evaluations directory."""
    repository = get_repository()
    evals_dir = get_evaluations_dir()
    logger.info(f"Loading existing evaluations from directory: {evals_dir}")
    if not os.path.exists(evals_dir):
        logger.info("Evaluations directory does not exist, creating it")
        os.makedirs(evals_dir)
        return
    
    eval_names = repository.list_evaluations()
    logger.info(f"Found {len(eval_names)} evaluations to load")
    
    for eval_name in eval_names:
        try:
            logger.info(f"Loading evaluation: {eval_name}")
            evaluation = repository.load_evaluation(eval_name)
            evaluations[eval_name] = evaluation
            logger.info(f"Successfully loaded evaluation: {eval_name} with status {evaluation.status}")
            
        except Exception as e:
            logger.exception(f"Failed to load evaluation {eval_name}: {e}")

def save_event_to_file(evaluation_name: str, event: EvaluationEvent):
    """Save event to a JSON file with indentation."""
    try:
        events_dir = Path(get_evaluations_dir()) / evaluation_name
        events_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving event to directory: {events_dir}")
        
        # Convert event data to serializable format
        event_data = {
            "type": event.event_type,
            "data": event.data if event.data is not None else {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        event_file = events_dir / "events.json"
        logger.debug(f"Event file path: {event_file}")
        
        # Load existing events or create new list
        events = []
        if event_file.exists():
            try:
                with open(event_file, 'r') as f:
                    events = json.load(f)
                    logger.debug(f"Loaded {len(events)} existing events")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode existing events file: {e}")
            except Exception as e:
                logger.warning(f"Error reading events file: {e}")
        
        events.append(event_data)
        logger.debug(f"Appending new event of type: {event.event_type}")
        
        # Save with indentation
        try:
            with open(event_file, 'w') as f:
                json.dump(events, f, indent=2, default=str)
            logger.debug(f"Successfully saved {len(events)} events to {event_file}")
        except Exception as e:
            logger.error(f"Failed to save events to file: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in save_event_to_file: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    global repository
    logger.info("Starting evaluation worker server")
    repository = EvaluationFileRepository(get_evaluations_dir())
    logger.info(f"Initialized repository with evaluations directory: {get_evaluations_dir()}")
    load_existing_evaluations()
    logger.info("Completed startup initialization")

@app.get("/evaluations", response_model=EvaluationListResponse)
async def list_evaluations():
    """List all evaluations from the repository."""
    evaluation_items = []
    repository = get_repository()
    
    # Get all evaluations from repository
    eval_names = repository.list_evaluations()
    logger.debug(f"Found evaluation names: {eval_names}")
    
    for eval_name in eval_names:
        try:
            # Load evaluation from repository
            evaluation = repository.load_evaluation(eval_name)
            is_active = active_evaluations.get(eval_name, False)
            
            # Create list item
            list_item = EvaluationListItem.from_evaluation(
                eval_name, 
                evaluation, 
                repository,
                is_active
            )
            evaluation_items.append(list_item)
            logger.debug(f"Added evaluation item: {list_item.model_dump()}")
            
            # Update cache
            evaluations[eval_name] = evaluation
            
        except Exception as e:
            logger.exception(f"Failed to load evaluation {eval_name}: {e}")
    
    logger.debug(f"Returning {len(evaluation_items)} evaluation items")
    return EvaluationListResponse(evaluations=evaluation_items)

@app.post("/evaluations/")
async def create_evaluation_endpoint(request: EvaluationRequest):
    repository = get_repository()
    # Generate base evaluation name
    base_evaluation_name = create_evaluation_name(
        model=request.model,
        max_expansions=request.max_expansions,
        date=None,
        temp_bias=0.0
    )
    
    # Check for existing evaluation directory and modify name if needed
    evaluation_name = base_evaluation_name
    counter = 1
    while os.path.exists(os.path.join(get_evaluations_dir(), evaluation_name)):
        evaluation_name = f"{base_evaluation_name}_{counter}"
        counter += 1
    
    # Create model settings
    model_settings = CompletionModel(
        model=request.model,
        temperature=0.0,
        max_tokens=3000,
        api_key=request.api_key,
        base_url=request.base_url,
        response_format=LLMResponseFormat.REACT
    )

    # Create agent settings
    agent_settings = AgentSettings(
        completion_model=model_settings,
        message_history_type=MessageHistoryType.REACT,
        system_prompt=None
    )

    # Create tree search settings
    tree_search_settings = TreeSearchSettings(
        max_iterations=request.max_iterations,
        max_expansions=request.max_expansions,
        max_cost=request.max_cost,
        model=model_settings,
        agent_settings=agent_settings
    )

    # Create evaluation using factory
    evaluation = create_evaluation(
        repository=repository,
        evaluation_name=evaluation_name,
        settings=tree_search_settings,
        instance_ids=request.instance_ids,
        min_resolved=request.min_resolved
    )

    evaluations[evaluation_name] = evaluation
    return {"evaluation_name": evaluation_name}

@app.post("/evaluations/{evaluation_name}/start")
async def start_evaluation(evaluation_name: str, request: StartEvaluationRequest, background_tasks: BackgroundTasks):
    repository = get_repository()
    if evaluation_name not in evaluations:
        return {"error": "Evaluation not found"}
    
    evaluation = evaluations[evaluation_name]
    
    # Create evaluation runner with repository
    runner = EvaluationRunner(
        repository=repository,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        repo_base_dir=os.getenv("REPO_DIR", "/tmp/repos"),
        num_workers=1,
        use_testbed=True
    )
    
    def event_handler(event: EvaluationEvent):
        if evaluation_name not in evaluation_events:
            evaluation_events[evaluation_name] = []
        evaluation_events[evaluation_name].append({
            "type": event.event_type,
            "data": event.data,
            "timestamp": datetime.now().isoformat()
        })
        print(f"Event received for {evaluation_name}: {event.event_type}")
        save_event_to_file(evaluation_name, event)

    runner.add_event_handler(event_handler)
    
    # Mark evaluation as active
    active_evaluations[evaluation_name] = True
    
    async def run_evaluation_task():
        try:
            await asyncio.to_thread(runner.run_evaluation, evaluation, rerun_errors=request.rerun_errors)
        finally:
            # Mark evaluation as inactive when done
            active_evaluations[evaluation_name] = False
    
    background_tasks.add_task(run_evaluation_task)
    return {"status": "started"}

@app.get("/evaluations/{evaluation_name}")
async def get_evaluation_status(evaluation_name: str):
    repository = get_repository()
    logger.info(f"Getting status for evaluation: {evaluation_name}")
    
    if evaluation_name not in evaluations:
        logger.error(f"Evaluation not found: {evaluation_name}")
        return {"error": "Evaluation not found"}
    
    evaluation = evaluations[evaluation_name]
    logger.info(f"Found evaluation with status: {evaluation.status}")
    
    instances = repository.list_instances(evaluation_name)
    logger.info(f"Retrieved {len(instances)} instances for evaluation {evaluation_name}")
    
    # Check if evaluation is actually running
    is_active = active_evaluations.get(evaluation_name, False)
    actual_status = evaluation.status
    if actual_status == EvaluationStatus.RUNNING and not is_active:
        actual_status = EvaluationStatus.PENDING  # Mark as pending if not actively running
    
    instance_data = {}
    for instance in instances:
        instance_data[instance.instance_id] = {
            "status": instance.status,
            "started_at": instance.started_at.isoformat() if instance.started_at else None,
            "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
            "duration": instance.duration,
            "resolved": instance.resolved,
            "error": instance.error,
            "benchmark_result": instance.benchmark_result.dict() if hasattr(instance, 'benchmark_result') and instance.benchmark_result else None
        }
        logger.debug(f"Added instance data for {instance.instance_id} with status {instance.status}")
    
    logger.info(f"Returning status with {len(instance_data)} instances")
    return {
        "status": actual_status,
        "is_active": is_active,
        "settings": {
            "model": evaluation.settings.model.model,
            "max_iterations": evaluation.settings.max_iterations,
            "max_expansions": evaluation.settings.max_expansions,
            "max_cost": evaluation.settings.max_cost,
        },
        "started_at": evaluation.start_time.isoformat() if hasattr(evaluation, 'start_time') and evaluation.start_time else None,
        "instances": instance_data
    }

@app.get("/evaluations/{evaluation_name}/events")
async def get_evaluation_events(evaluation_name: str, since: Optional[str] = None):
    if evaluation_name not in evaluation_events:
        return {"events": []}
    
    events = evaluation_events[evaluation_name]
    if since:
        events = [e for e in events if e["timestamp"] > since]
    return {"events": events}

#if __name__ == "__main__":
#    uvicorn.run(
#        "evaluation_worker:app",
#        host="0.0.0.0",
#        port=8000,
#    ) 
    