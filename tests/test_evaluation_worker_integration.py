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
from moatless_tools.evaluation_worker import app, get_repository

# Configure logging - set to DEBUG for more detailed information
logging.basicConfig(
    level=logging.DEBUG,
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
    temp_evals_dir = tempfile.mkdtemp(prefix="moatless_test_evals_")
    logger.info(f"Created temporary evaluation directory: {temp_evals_dir}")
    
    # Set the evaluation directory
    os.environ["MOATLESS_DIR"] = temp_evals_dir
    os.makedirs(temp_evals_dir, exist_ok=True)
    
    yield temp_evals_dir
    
    # Cleanup: Remove temporary directories
    logger.info("Cleaning up temporary directories...")
    try:
        shutil.rmtree(temp_evals_dir)
        logger.info(f"Removed temporary evaluation directory: {temp_evals_dir}")
    except Exception as e:
        logger.warning(f"Failed to remove evaluation directory: {e}")

@pytest.fixture
def repository():
    """Fixture to provide access to the repository instance."""
    from moatless_tools.evaluation_worker import get_repository
    return get_repository()

def test_create_evaluation(test_client, test_evaluation_request, repository):
    """Test creating a new evaluation."""
    response = test_client.post("/evaluations/", json=test_evaluation_request)
    assert response.status_code == 200
    evaluation_name = response.json()["evaluation_name"]
    
    # Now using the repository fixture
    evaluation = repository.load_evaluation(evaluation_name)
    assert evaluation is not None
    assert evaluation.evaluation_name == evaluation_name
    assert evaluation.status == EvaluationStatus.PENDING
    assert evaluation.settings.max_iterations == test_evaluation_request["max_iterations"]
    assert evaluation.settings.max_expansions == test_evaluation_request["max_expansions"]
    assert evaluation.settings.max_cost == test_evaluation_request["max_cost"]
    assert evaluation.settings.model.model == test_evaluation_request["model"]

def test_list_evaluations(test_client, test_evaluation_request):
    """Test listing evaluations."""
    # Create multiple evaluations
    eval_names = []
    for i in range(3):
        response = test_client.post("/evaluations/", json=test_evaluation_request)
        assert response.status_code == 200
        eval_name = response.json()["evaluation_name"]
        eval_names.append(eval_name)
        logger.debug(f"Created evaluation: {eval_name}")
    
    logger.debug(f"Created evaluations: {eval_names}")
    
    # Get list of evaluations
    response = test_client.get("/evaluations")
    assert response.status_code == 200
    data = response.json()
    logger.debug(f"List response: {json.dumps(data, indent=2)}")
    
    # Verify response format
    assert "evaluations" in data
    evaluations = data["evaluations"]
    assert len(evaluations) == 3, f"Expected 3 evaluations, got {len(evaluations)}: {evaluations}"
    
    # Verify each evaluation in the list
    for eval_item in evaluations:
        assert eval_item["name"] in eval_names
        assert eval_item["status"] == EvaluationStatus.PENDING
        assert eval_item["model"] == test_evaluation_request["model"]
        assert eval_item["max_expansions"] == test_evaluation_request["max_expansions"]
        assert eval_item["total_instances"] == 0
        assert eval_item["completed_instances"] == 0
        assert eval_item["error_instances"] == 0
        assert eval_item["resolved_instances"] == 0
        assert not eval_item["is_active"]

def test_get_evaluation(test_client, test_evaluation_request):
    """Test getting a specific evaluation."""
    # Create an evaluation
    response = test_client.post("/evaluations/", json=test_evaluation_request)
    assert response.status_code == 200
    evaluation_name = response.json()["evaluation_name"]
    
    # Get the evaluation
    response = test_client.get(f"/evaluations/{evaluation_name}")
    assert response.status_code == 200
    data = response.json()
    
    # Verify evaluation details
    assert data["status"] == EvaluationStatus.PENDING
    assert not data["is_active"]
    assert "settings" in data
    assert data["settings"]["model"] == test_evaluation_request["model"]
    assert data["settings"]["max_iterations"] == test_evaluation_request["max_iterations"]
    assert data["settings"]["max_expansions"] == test_evaluation_request["max_expansions"]
    assert data["settings"]["max_cost"] == test_evaluation_request["max_cost"]
    assert "instances" in data
    assert len(data["instances"]) == 0

def test_get_nonexistent_evaluation(test_client):
    """Test getting a nonexistent evaluation."""
    response = test_client.get("/evaluations/nonexistent")
    assert response.status_code == 200  # Returns 200 with error message
    assert "error" in response.json()
    assert response.json()["error"] == "Evaluation not found"
