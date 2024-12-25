import asyncio
import json
import logging
import os
import shutil
import pytest
import tempfile
import time
from fastapi.testclient import TestClient
from moatless.benchmark.evaluation_v2 import EvaluationStatus, InstanceStatus
from moatless_tools.evaluation_worker import app, EVALUATIONS_DIR

# Configure logging - set to DEBUG for more detailed information
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure all relevant loggers are at DEBUG level
logging.getLogger('moatless').setLevel(logging.DEBUG)
logging.getLogger('moatless.utils.repo').setLevel(logging.DEBUG)
logging.getLogger('moatless.benchmark').setLevel(logging.DEBUG)

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def test_evaluation_request():
    return {
        "model": "gemini/gemini-2.0-flash-exp",
        "api_key": "",
        "base_url": "",
        "max_iterations": 20,
        "max_expansions": 1,
        "max_cost": 1.0,
        "instance_ids": ["scikit-learn__scikit-learn-14894"],
        "min_resolved": 20
    }

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    # Setup: Create temporary evaluation directory
    original_evals_dir = EVALUATIONS_DIR
    temp_evals_dir = tempfile.mkdtemp(prefix="moatless_test_evals_")
    logger.info(f"Created temporary evaluation directory: {temp_evals_dir}")
    
    # Set the evaluation directory in app state
    app.state.EVALUATIONS_DIR = temp_evals_dir
    os.makedirs(temp_evals_dir, exist_ok=True)
    
    yield
    
    # Cleanup: Remove temporary directories and restore original
    logger.info("Cleaning up temporary directories...")
    try:
        shutil.rmtree(temp_evals_dir)
        logger.info(f"Removed temporary evaluation directory: {temp_evals_dir}")
    except Exception as e:
        logger.warning(f"Failed to remove evaluation directory: {e}")
    
    # Restore original directory in app state
    app.state.EVALUATIONS_DIR = original_evals_dir

async def wait_for_evaluation_completion(client, evaluation_name, timeout=300):  # Increased timeout to 5 minutes
    """Wait for evaluation to complete or timeout"""
    start_time = time.time()
    last_log_time = 0
    log_interval = 10  # Log every 10 seconds
    
    while time.time() - start_time < timeout:
        current_time = time.time()
        
        try:
            response = client.get(f"/evaluations/{evaluation_name}")
            assert response.status_code == 200
            
            data = response.json()
            
            # Log status and progress at intervals
            if current_time - last_log_time >= log_interval:
                logger.info(f"Evaluation status: {data['status']}")
                
                # Log instance details
                for instance_id, instance in data["instances"].items():
                    logger.info(f"Instance {instance_id}:")
                    logger.info(f"  Status: {instance['status']}")
                    logger.info(f"  Started: {instance.get('started_at')}")
                    if instance.get('error'):
                        logger.error(f"  Error: {instance['error']}")
                
                # Check instance progress
                completed = sum(1 for inst in data["instances"].values() 
                            if inst["status"] == InstanceStatus.COMPLETED)
                total = len(data["instances"])
                logger.info(f"Progress: {completed}/{total} instances completed")
                logger.info(f"Time elapsed: {int(current_time - start_time)}s")
                
                last_log_time = current_time
            
            if data["status"] in [EvaluationStatus.COMPLETED, EvaluationStatus.ERROR]:
                return data
            
        except Exception as e:
            logger.error(f"Error checking evaluation status: {e}")
            # Continue waiting despite error
        
        await asyncio.sleep(1)
    
    raise TimeoutError(f"Evaluation did not complete within {timeout} seconds")

@pytest.mark.asyncio
async def test_evaluation_lifecycle(test_client, test_evaluation_request):
    """Test the complete lifecycle of an evaluation: create, run, and monitor until completion"""
    
    try:
        # Step 1: Create evaluation
        logger.info("Creating evaluation...")
        logger.debug(f"Evaluation request: {json.dumps(test_evaluation_request, indent=2)}")
        response = test_client.post("/evaluations/", json=test_evaluation_request)
        assert response.status_code == 200
        evaluation_name = response.json()["evaluation_name"]
        logger.info(f"Created evaluation: {evaluation_name}")
        
        # Step 2: Start evaluation
        logger.info("Starting evaluation...")
        response = test_client.post(f"/evaluations/{evaluation_name}/start", json={"rerun_errors": False})
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        
        # Step 3: Monitor until completion
        logger.info("Monitoring evaluation progress...")
        try:
            final_status = await wait_for_evaluation_completion(test_client, evaluation_name)
            logger.info(f"Final evaluation status: {json.dumps(final_status, indent=2)}")
            
            # Verify final state
            assert final_status["status"] in [EvaluationStatus.COMPLETED, EvaluationStatus.ERROR]
            assert "instances" in final_status
            assert len(final_status["instances"]) == len(test_evaluation_request["instance_ids"])
            
            # Check if events were recorded
            response = test_client.get(f"/evaluations/{evaluation_name}/events")
            assert response.status_code == 200
            events = response.json()["events"]
            logger.info(f"Found {len(events)} events")
            for event in events:
                logger.debug(f"Event: {event['type']}")
            
            assert len(events) > 0
            assert any(event["type"] == "evaluation_started" for event in events)
            assert any(event["type"] == "evaluation_completed" for event in events)
            
            # Verify evaluation files were created
            eval_dir = os.path.join(app.state.EVALUATIONS_DIR, evaluation_name)
            assert os.path.exists(eval_dir), f"Evaluation directory {eval_dir} does not exist"
            
            # Verify events.json exists and has content
            events_file = os.path.join(eval_dir, "events.json")
            assert os.path.exists(events_file), f"Events file {events_file} does not exist"
            with open(events_file) as f:
                saved_events = json.load(f)
                assert len(saved_events) > 0, "Events file is empty"
                assert any(event["type"] == "evaluation_started" for event in saved_events)
                assert any(event["type"] == "evaluation_completed" for event in saved_events)
                logger.info(f"Found {len(saved_events)} events in events.json")
            
            # Verify instance directory and eval_result.json
            instance_id = test_evaluation_request["instance_ids"][0]
            instance_dir = os.path.join(eval_dir, instance_id)
            assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist"
            
            eval_result_file = os.path.join(instance_dir, "eval_result.json")
            assert os.path.exists(eval_result_file), f"Eval result file {eval_result_file} does not exist"
            with open(eval_result_file) as f:
                eval_result = json.load(f)
                assert "status" in eval_result, "Eval result missing status"
                assert eval_result["status"] in ["completed", "error"], f"Unexpected status: {eval_result['status']}"
                assert "duration" in eval_result, "Eval result missing duration"
                assert "benchmark_result" in eval_result, "Eval result missing benchmark_result"
                logger.info(f"Eval result status: {eval_result['status']}")
            
            # Verify trajectory.json exists
            trajectory_file = os.path.join(instance_dir, "trajectory.json")
            assert os.path.exists(trajectory_file), f"Trajectory file {trajectory_file} does not exist"
            
        except TimeoutError as e:
            logger.error(f"Evaluation timed out: {e}")
            # Get final status even after timeout
            try:
                response = test_client.get(f"/evaluations/{evaluation_name}")
                if response.status_code == 200:
                    status = response.json()
                    logger.error(f"Status at timeout: {json.dumps(status, indent=2)}")
            except Exception as e:
                logger.error(f"Failed to get status after timeout: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(f"Traceback:", exc_info=True)
        raise 