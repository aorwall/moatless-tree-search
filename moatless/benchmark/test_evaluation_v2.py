import json
import os
import tempfile
from datetime import datetime, timezone
import pytest
import threading

from moatless.benchmark.evaluation_v2 import EvaluationInstance, InstanceStatus, Evaluation, TreeSearchSettings
from moatless.agent.agent import ActionAgent
from moatless.completion.completion import CompletionModel
from moatless.repository import FileRepository
from moatless.index.code_index import CodeIndex
from moatless.runtime.runtime import NoEnvironment
from moatless.actions.string_replace import StringReplace
from moatless.actions.find_class import FindClass
from moatless.actions.run_tests import RunTests
from moatless.benchmark.report import BenchmarkResult


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@pytest.fixture
def test_setup():
    # Create temp directory and dependencies
    temp_dir = tempfile.mkdtemp()
    instance_id = "test_instance_123"
    instance = EvaluationInstance(instance_id=instance_id)
    
    # Create real dependencies like in test_search_tree.py
    repository = FileRepository(repo_path=temp_dir)
    code_index = CodeIndex(repository)
    runtime = NoEnvironment()
    completion_model = CompletionModel(model="gpt-4")
    
    # Create actions
    run_tests = RunTests(
        repository=repository,
        code_index=code_index,
        runtime=runtime
    )
    code_change = StringReplace(
        repository=repository,
        completion_model=completion_model
    )
    find_class = FindClass(
        repository=repository,
        code_index=code_index
    )
    
    # Create a real ActionAgent with system prompt
    base_agent = ActionAgent(
        actions=[code_change, run_tests, find_class],
        completion=completion_model,
        system_prompt="You are a helpful coding assistant.",
    )
    
    # Create a basic TreeSearchSettings with real agent
    settings = TreeSearchSettings(
        base_agent=base_agent,
        max_expansions=1,
        max_iterations=25,
    )
    
    yield {
        'temp_dir': temp_dir,
        'instance_id': instance_id,
        'instance': instance,
        'repository': repository,
        'code_index': code_index,
        'runtime': runtime,
        'completion_model': completion_model,
        'base_agent': base_agent,
        'settings': settings
    }
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def test_instance_serialization(test_setup):
    # Test basic instance creation and serialization
    instance = test_setup['instance']
    instance_id = test_setup['instance_id']
    
    # Test initial state
    assert instance.status == InstanceStatus.PENDING
    assert instance.started_at is None
    assert instance.completed_at is None
    assert instance.benchmark_result is None
    
    # Test state transitions
    instance.start()
    assert instance.status == InstanceStatus.STARTED
    assert instance.started_at is not None
    
    submission = "test patch"
    benchmark_result = BenchmarkResult(
        instance_id=instance_id,
        status="completed",
        resolved=True,
        duration=1.0,
        total_cost=0.1,
    )
    instance.complete(submission=submission, resolved=True, benchmark_result=benchmark_result)
    assert instance.status == InstanceStatus.COMPLETED
    assert instance.submission == submission
    assert instance.resolved is True
    assert instance.completed_at is not None
    assert instance.duration is not None
    assert instance.benchmark_result == benchmark_result
    
    # Test serialization
    instance_dict = instance.model_dump()
    assert instance_dict["instance_id"] == instance_id
    assert instance_dict["status"] == InstanceStatus.COMPLETED
    assert instance_dict["submission"] == submission
    assert instance_dict["benchmark_result"]["instance_id"] == instance_id
    assert instance_dict["benchmark_result"]["status"] == "completed"
    assert instance_dict["benchmark_result"]["resolved"] is True
    
    # Test deserialization
    new_instance = EvaluationInstance.model_validate(instance_dict)
    assert new_instance.instance_id == instance.instance_id
    assert new_instance.status == instance.status
    assert new_instance.submission == instance.submission
    assert new_instance.resolved == instance.resolved
    assert new_instance.benchmark_result.instance_id == instance.benchmark_result.instance_id
    assert new_instance.benchmark_result.status == instance.benchmark_result.status
    assert new_instance.benchmark_result.resolved == instance.benchmark_result.resolved


def test_evaluation_serialization(test_setup):
    # Create an evaluation with some instances
    instances = {
        "instance1": EvaluationInstance(instance_id="instance1"),
        "instance2": EvaluationInstance(instance_id="instance2"),
    }
    
    evaluation = Evaluation(
        evaluations_dir=test_setup['temp_dir'],
        evaluation_name="test_eval",
        settings=test_setup['settings'],
        instances=instances
    )
    
    # Test initial state
    assert len(evaluation.instances) == 2
    assert "instance1" in evaluation.instances
    assert "instance2" in evaluation.instances
    
    # Modify instance states
    benchmark_result1 = BenchmarkResult(
        instance_id="instance1",
        status="completed",
        resolved=True,
        duration=1.0,
        total_cost=0.1,
    )
    evaluation.instances["instance1"].start()
    evaluation.instances["instance1"].complete(submission="patch1", resolved=True, benchmark_result=benchmark_result1)
    evaluation.instances["instance2"].start()
    evaluation.instances["instance2"].fail("test error")
    
    # Test serialization
    eval_dict = evaluation.model_dump()
    assert eval_dict["evaluation_name"] == "test_eval"
    assert len(eval_dict["instances"]) == 2
    
    # Verify instance states were preserved
    assert eval_dict["instances"]["instance1"]["status"] == InstanceStatus.COMPLETED
    assert eval_dict["instances"]["instance1"]["submission"] == "patch1"
    assert eval_dict["instances"]["instance1"]["benchmark_result"]["instance_id"] == "instance1"
    assert eval_dict["instances"]["instance2"]["status"] == InstanceStatus.ERROR
    assert eval_dict["instances"]["instance2"]["error"] == "test error"
    
    # Create new dependencies for deserialization
    new_repository = FileRepository(repo_path=tempfile.mkdtemp())
    new_code_index = CodeIndex(new_repository)
    new_runtime = NoEnvironment()
    
    # Reconstruct agent with dependencies
    agent_dict = eval_dict["settings"]["base_agent"]
    new_agent = ActionAgent.from_dict(
        agent_dict,
        repository=new_repository,
        code_index=new_code_index,
        runtime=new_runtime,
    )
    
    # Update settings with reconstructed agent
    eval_dict["settings"]["base_agent"] = new_agent
    
    # Test deserialization
    new_evaluation = Evaluation.model_validate(eval_dict)
    assert new_evaluation.evaluation_name == evaluation.evaluation_name
    assert len(new_evaluation.instances) == len(evaluation.instances)
    
    # Verify instance states after deserialization
    assert new_evaluation.instances["instance1"].status == evaluation.instances["instance1"].status
    assert new_evaluation.instances["instance1"].submission == evaluation.instances["instance1"].submission
    assert new_evaluation.instances["instance1"].benchmark_result.instance_id == evaluation.instances["instance1"].benchmark_result.instance_id
    assert new_evaluation.instances["instance2"].status == evaluation.instances["instance2"].status
    assert new_evaluation.instances["instance2"].error == evaluation.instances["instance2"].error


def test_file_persistence(test_setup):
    # Create an evaluation
    instances = {
        "instance1": EvaluationInstance(instance_id="instance1"),
        "instance2": EvaluationInstance(instance_id="instance2"),
    }
    
    eval_dir = os.path.join(test_setup['temp_dir'], "test_eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    evaluation = Evaluation(
        evaluations_dir=test_setup['temp_dir'],
        evaluation_name="test_eval",
        settings=test_setup['settings'],
        instances=instances
    )
    
    # Modify instance states
    benchmark_result1 = BenchmarkResult(
        instance_id="instance1",
        status="completed",
        resolved=True,
        duration=1.0,
        total_cost=0.1,
    )
    evaluation.instances["instance1"].start()
    evaluation.instances["instance1"].complete(submission="patch1", resolved=True, benchmark_result=benchmark_result1)
    evaluation.instances["instance2"].start()
    evaluation.instances["instance2"].fail("test error")
    
    # Save to file
    eval_file = os.path.join(eval_dir, "evaluation.json")
    with open(eval_file, "w") as f:
        json.dump(evaluation.model_dump(), f, cls=DateTimeEncoder)
    
    # Read from file and verify
    with open(eval_file) as f:
        loaded_dict = json.load(f)
    
    # Create new dependencies for deserialization
    new_repository = FileRepository(repo_path=tempfile.mkdtemp())
    new_code_index = CodeIndex(new_repository)
    new_runtime = NoEnvironment()
    
    # Reconstruct agent with dependencies
    agent_dict = loaded_dict["settings"]["base_agent"]
    new_agent = ActionAgent.from_dict(
        agent_dict,
        repository=new_repository,
        code_index=new_code_index,
        runtime=new_runtime,
    )
    
    # Update settings with reconstructed agent
    loaded_dict["settings"]["base_agent"] = new_agent
    
    loaded_evaluation = Evaluation.model_validate(loaded_dict)
    
    # Verify all data was preserved
    assert loaded_evaluation.evaluation_name == evaluation.evaluation_name
    assert len(loaded_evaluation.instances) == len(evaluation.instances)
    assert loaded_evaluation.instances["instance1"].status == evaluation.instances["instance1"].status
    assert loaded_evaluation.instances["instance1"].submission == evaluation.instances["instance1"].submission
    assert loaded_evaluation.instances["instance1"].benchmark_result.instance_id == evaluation.instances["instance1"].benchmark_result.instance_id
    assert loaded_evaluation.instances["instance2"].error == evaluation.instances["instance2"].error 


def test_evaluation_persistence(test_setup):
    # Create an evaluation with some instances
    instances = {
        "instance1": EvaluationInstance(instance_id="instance1"),
        "instance2": EvaluationInstance(instance_id="instance2"),
    }
    
    eval_dir = os.path.join(test_setup['temp_dir'], "test_eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    evaluation = Evaluation(
        evaluations_dir=test_setup['temp_dir'],
        evaluation_name="test_eval",
        settings=test_setup['settings'],
        instances=instances
    )
    evaluation.evaluation_dir = eval_dir
    evaluation._file_lock = threading.Lock()
    
    # Verify initial state is saved
    evaluation._save_evaluation_state()
    eval_file = os.path.join(eval_dir, "evaluation.json")
    assert os.path.exists(eval_file)
    
    with open(eval_file) as f:
        initial_state = json.load(f)
    assert len(initial_state["instances"]) == 2
    assert initial_state["instances"]["instance1"]["status"] == "pending"
    assert initial_state["instances"]["instance2"]["status"] == "pending"
    
    # Modify first instance and verify state is updated
    benchmark_result1 = BenchmarkResult(
        instance_id="instance1",
        status="completed",
        resolved=True,
        duration=1.0,
        total_cost=0.1,
    )
    evaluation.instances["instance1"].start()
    evaluation.instances["instance1"].complete(submission="patch1", resolved=True, benchmark_result=benchmark_result1)
    evaluation._save_evaluation_state()
    
    with open(eval_file) as f:
        updated_state = json.load(f)
    assert updated_state["instances"]["instance1"]["status"] == "completed"
    assert updated_state["instances"]["instance1"]["submission"] == "patch1"
    assert updated_state["instances"]["instance1"]["benchmark_result"]["instance_id"] == "instance1"
    assert updated_state["instances"]["instance2"]["status"] == "pending"
    
    # Modify second instance and verify state is updated
    evaluation.instances["instance2"].start()
    evaluation.instances["instance2"].fail("test error")
    evaluation._save_evaluation_state()
    
    with open(eval_file) as f:
        final_state = json.load(f)
    assert final_state["instances"]["instance1"]["status"] == "completed"
    assert final_state["instances"]["instance2"]["status"] == "error"
    assert final_state["instances"]["instance2"]["error"] == "test error"
    
    # Create new dependencies for deserialization
    new_repository = FileRepository(repo_path=tempfile.mkdtemp())
    new_code_index = CodeIndex(new_repository)
    new_runtime = NoEnvironment()
    
    # Reconstruct agent with dependencies
    agent_dict = final_state["settings"]["base_agent"]
    new_agent = ActionAgent.from_dict(
        agent_dict,
        repository=new_repository,
        code_index=new_code_index,
        runtime=new_runtime,
    )
    
    # Update settings with reconstructed agent
    final_state["settings"]["base_agent"] = new_agent
    
    # Verify we can load the saved state
    loaded_evaluation = Evaluation.model_validate(final_state)
    assert loaded_evaluation.evaluation_name == evaluation.evaluation_name
    assert len(loaded_evaluation.instances) == len(evaluation.instances)
    assert loaded_evaluation.instances["instance1"].status == evaluation.instances["instance1"].status
    assert loaded_evaluation.instances["instance1"].benchmark_result.instance_id == evaluation.instances["instance1"].benchmark_result.instance_id
    assert loaded_evaluation.instances["instance2"].status == evaluation.instances["instance2"].status
    assert loaded_evaluation.instances["instance2"].error == evaluation.instances["instance2"].error 