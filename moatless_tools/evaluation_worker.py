import os
from pathlib import Path
import logging
from moatless.benchmark.evaluation_v2 import (
    TreeSearchSettings, Evaluation, create_evaluation_name,
    InstanceStatus, EvaluationStatus, EvaluationEvent
)
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

# Load environment variables
load_dotenv()

app = FastAPI(title="Moatless Evaluation API")

@app.on_event("startup")
async def startup_event():
    load_existing_evaluations()

def get_evaluations_dir():
    """Get the current evaluations directory from app state."""
    return os.getenv("MOATLESS_DIR", "./evals")

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
    repo_base_dir: Optional[str] = None  # Made optional
    min_resolved: int

class StartEvaluationRequest(BaseModel):
    rerun_errors: bool = False

def load_existing_evaluations():
    """Load existing evaluations from the evaluations directory."""
    evals_dir = get_evaluations_dir()
    if not os.path.exists(evals_dir):
        os.makedirs(evals_dir)
        return
    
    for eval_dir in Path(evals_dir).iterdir():
        if not eval_dir.is_dir():
            continue
        
        try:
            evaluation = Evaluation.load(evals_dir, eval_dir.name)
            evaluations[eval_dir.name] = evaluation
            print(f"Loaded evaluation: {eval_dir.name}")
            
        except Exception as e:
            logger.exception(f"Failed to load evaluation {eval_dir.name}: {e}")

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

def run_evaluation(evaluation_name: str, request: EvaluationRequest):
    try:
        print(f"Starting evaluation {evaluation_name} with settings: {request}")
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

        print(f"Created tree search settings: {tree_search_settings}")

        # Create evaluation
        evaluation = Evaluation.create(
            evaluations_dir=get_evaluations_dir(),  # Use function instead of constant
            evaluation_name=evaluation_name,
            settings=tree_search_settings,
            split="combo",
            instance_ids=request.instance_ids if request.instance_ids else None,
            min_resolved=request.min_resolved,
            num_workers=1,
            use_testbed=True,
            dataset_name="princeton-nlp/SWE-bench_Verified",
            repo_base_dir=os.getenv("REPO_DIR", "/tmp/repos")
        )

        def event_handler(event: EvaluationEvent):
            try:
                if evaluation_name not in evaluation_events:
                    evaluation_events[evaluation_name] = []
                
                # Create event data
                event_data = {
                    "type": event.event_type,
                    "data": event.data if event.data is not None else {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                evaluation_events[evaluation_name].append(event_data)
                logger.info(f"Event received for {evaluation_name}: {event.event_type}")
                
                # Save event to file
                save_event_to_file(evaluation_name, event)
                
                # Save evaluation state
                evaluation._save_evaluation_state()
                
            except Exception as e:
                logger.error(f"Error handling event: {e}")
                logger.error(traceback.format_exc())
                raise

        evaluation.add_event_handler(event_handler)
        evaluations[evaluation_name] = evaluation
        
        print(f"Starting evaluation run for {evaluation_name}")
        # Run evaluation
        evaluation._run_evaluation()
        print(f"Evaluation {evaluation_name} completed")
    except Exception as e:
        print(f"Error running evaluation {evaluation_name}: {e}")
        raise

@app.get("/evaluations")
async def list_evaluations():
    return {
        "evaluations": [
            {
                "name": name,
                "status": eval.status if active_evaluations.get(name, False) else EvaluationStatus.PENDING if eval.status == EvaluationStatus.RUNNING else eval.status,
                "model": eval.settings.model.model,
                "max_expansions": eval.settings.max_expansions,
                "started_at": eval.start_time.isoformat() if hasattr(eval, 'start_time') and eval.start_time else None,
                "total_instances": len(eval.instances),
                "completed_instances": sum(1 for inst in eval.instances.values() 
                                        if inst.status == InstanceStatus.COMPLETED),
                "error_instances": sum(1 for inst in eval.instances.values() 
                                    if inst.status == InstanceStatus.ERROR),
                "resolved_instances": sum(1 for inst in eval.instances.values() 
                                    if inst.resolved),
                "is_active": active_evaluations.get(name, False)
            }
            for name, eval in evaluations.items()
        ]
    }

@app.post("/evaluations/")
async def create_evaluation(request: EvaluationRequest):
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
    while os.path.exists(os.path.join(EVALUATIONS_DIR, evaluation_name)):
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

    # Create evaluation without running it
    evaluation = Evaluation.create(
        evaluations_dir=EVALUATIONS_DIR,
        evaluation_name=evaluation_name,
        settings=tree_search_settings,
        split="combo",
        instance_ids=request.instance_ids if request.instance_ids else None,
        min_resolved=request.min_resolved,
        num_workers=1,
        use_testbed=True,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        repo_base_dir=request.repo_base_dir or DEFAULT_REPO_BASE_DIR  # Use default if not provided
    )

    evaluations[evaluation_name] = evaluation
    return {"evaluation_name": evaluation_name}

@app.post("/evaluations/{evaluation_name}/start")
async def start_evaluation(evaluation_name: str, request: StartEvaluationRequest, background_tasks: BackgroundTasks):
    if evaluation_name not in evaluations:
        return {"error": "Evaluation not found"}
    
    evaluation = evaluations[evaluation_name]
    
    def event_handler(event: EvaluationEvent):
        if evaluation_name not in evaluation_events:
            evaluation_events[evaluation_name] = []
        evaluation_events[evaluation_name].append({
            "type": event.event_type,
            "data": event.data,
            "timestamp": datetime.now().isoformat()
        })
        print(f"Event received for {evaluation_name}: {event.event_type}")

    evaluation.add_event_handler(event_handler)
    
    # Mark evaluation as active
    active_evaluations[evaluation_name] = True
    
    async def run_evaluation_task():
        try:
            await asyncio.to_thread(evaluation._run_evaluation, rerun_errors=request.rerun_errors)
        finally:
            # Mark evaluation as inactive when done
            active_evaluations[evaluation_name] = False
    
    background_tasks.add_task(run_evaluation_task)
    return {"status": "started"}

@app.get("/evaluations/{evaluation_name}")
async def get_evaluation_status(evaluation_name: str):
    if evaluation_name not in evaluations:
        return {"error": "Evaluation not found"}
    
    evaluation = evaluations[evaluation_name]
    
    # Check if evaluation is actually running
    is_active = active_evaluations.get(evaluation_name, False)
    actual_status = evaluation.status
    if actual_status == EvaluationStatus.RUNNING and not is_active:
        actual_status = EvaluationStatus.PENDING  # Mark as pending if not actively running
    
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
        "instances": {
            instance_id: {
                "status": instance.status,
                "started_at": instance.started_at.isoformat() if instance.started_at else None,
                "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
                "duration": instance.duration,
                "resolved": instance.resolved,
                "error": instance.error,
                "benchmark_result": instance.benchmark_result.dict() if hasattr(instance, 'benchmark_result') and instance.benchmark_result else None
            }
            for instance_id, instance in evaluation.instances.items()
        }
    }

@app.get("/evaluations/{evaluation_name}/events")
async def get_evaluation_events(evaluation_name: str, since: Optional[str] = None):
    if evaluation_name not in evaluation_events:
        return {"events": []}
    
    events = evaluation_events[evaluation_name]
    if since:
        events = [e for e in events if e["timestamp"] > since]
    return {"events": events}

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    load_dotenv()
    uvicorn.run(
        "evaluation_worker:app",
        host="0.0.0.0",
        port=8000,
        #reload=True,
        #reload_dirs=["moatless_tools", "moatless"]
    ) 
    