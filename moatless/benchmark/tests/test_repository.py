import os
import shutil
import tempfile
from datetime import datetime, timezone

import pytest

from moatless.benchmark.schema import (
    Evaluation,
    EvaluationInstance,
    TreeSearchSettings,
    EvaluationStatus,
    InstanceStatus,
)
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.agent.settings import AgentSettings
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.schema import MessageHistoryType


@pytest.fixture
def repository():
    # Create a temporary directory for test files
    test_dir = tempfile.mkdtemp()
    repo = EvaluationFileRepository(test_dir)
    yield repo
    # Clean up the temporary directory after test
    shutil.rmtree(test_dir)


@pytest.fixture
def test_completion_model():
    return CompletionModel(
        model="gpt-4",
        temperature=0.0,
        max_tokens=2000,
        response_format=LLMResponseFormat.JSON
    )


def test_save_and_load_evaluation(repository, test_completion_model):
    # Create a test evaluation
    evaluation = Evaluation(
        evaluations_dir=repository.evaluations_dir,
        evaluation_name="test_eval",
        settings=TreeSearchSettings(
            agent_settings=AgentSettings(
                completion_model=test_completion_model,
                message_history_type=MessageHistoryType.MESSAGES
            )
        ),
        status=EvaluationStatus.PENDING,
        start_time=datetime.now(timezone.utc),
    )

    # Save the evaluation
    repository.save_evaluation(evaluation)

    # Load the evaluation
    loaded_evaluation = repository.load_evaluation("test_eval")

    # Verify the loaded evaluation matches the original
    assert loaded_evaluation.evaluation_name == evaluation.evaluation_name
    assert loaded_evaluation.status == evaluation.status
    assert loaded_evaluation.settings.max_expansions == evaluation.settings.max_expansions
    assert loaded_evaluation.settings.agent_settings.message_history_type == evaluation.settings.agent_settings.message_history_type
    assert loaded_evaluation.settings.agent_settings.completion_model.model == evaluation.settings.agent_settings.completion_model.model


def test_save_and_load_instance(repository):
    # Create a test instance
    instance = EvaluationInstance(
        instance_id="test_instance",
        status=InstanceStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )

    # Save the instance
    repository.save_instance("test_eval", instance)

    # Load the instance
    loaded_instance = repository.load_instance("test_eval", "test_instance")

    # Verify the loaded instance matches the original
    assert loaded_instance.instance_id == instance.instance_id
    assert loaded_instance.status == instance.status


def test_delete_instance(repository):
    # Create and save a test instance
    instance = EvaluationInstance(
        instance_id="test_instance",
        status=InstanceStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )
    repository.save_instance("test_eval", instance)

    # Verify instance exists
    instance_dir = repository.get_instance_dir("test_eval", "test_instance")
    assert os.path.exists(instance_dir)

    # Delete the instance
    repository.delete_instance("test_eval", "test_instance")

    # Verify instance directory was deleted
    assert not os.path.exists(instance_dir)


def test_list_instances(repository, test_completion_model):
    # Create a test evaluation
    evaluation = Evaluation(
        evaluations_dir=repository.evaluations_dir,
        evaluation_name="test_eval",
        settings=TreeSearchSettings(
            agent_settings=AgentSettings(
                completion_model=test_completion_model,
                message_history_type=MessageHistoryType.MESSAGES
            )
        ),
        status=EvaluationStatus.RUNNING,
    )

    # Save evaluation
    repository.save_evaluation(evaluation)

    # Create and save test instances
    instance1 = EvaluationInstance(
        instance_id="instance1",
        status=InstanceStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        resolved=True,
    )
    instance2 = EvaluationInstance(
        instance_id="instance2",
        status=InstanceStatus.ERROR,
        created_at=datetime.now(timezone.utc),
        error="Test error",
    )

    repository.save_instance(evaluation.evaluation_name, instance1)
    repository.save_instance(evaluation.evaluation_name, instance2)

    # List instances
    instances = repository.list_instances("test_eval")
    assert len(instances) == 2

    # Verify instance details
    instance_dict = {i.instance_id: i for i in instances}
    assert instance_dict["instance1"].status == InstanceStatus.COMPLETED
    assert instance_dict["instance1"].resolved is True
    assert instance_dict["instance2"].status == InstanceStatus.ERROR
    assert instance_dict["instance2"].error == "Test error"


def test_nonexistent_evaluation(repository):
    # Try to load a non-existent evaluation
    with pytest.raises(FileNotFoundError):
        repository.load_evaluation("nonexistent_eval")


def test_nonexistent_instance(repository):
    # Try to load a non-existent instance
    loaded_instance = repository.load_instance("test_eval", "nonexistent_instance")
    assert loaded_instance is None


def test_list_instances_empty_evaluation(repository, test_completion_model):
    # Create a test evaluation with no instances
    evaluation = Evaluation(
        evaluations_dir=repository.evaluations_dir,
        evaluation_name="test_eval",
        settings=TreeSearchSettings(
            agent_settings=AgentSettings(
                completion_model=test_completion_model,
                message_history_type=MessageHistoryType.MESSAGES
            )
        ),
        status=EvaluationStatus.PENDING,
    )

    repository.save_evaluation(evaluation)

    # List instances should return empty list
    instances = repository.list_instances("test_eval")
    assert isinstance(instances, list)
    assert len(instances) == 0


def test_list_instances_nonexistent_evaluation(repository):
    # List instances for non-existent evaluation should return empty list
    instances = repository.list_instances("nonexistent_eval")
    assert isinstance(instances, list)
    assert len(instances) == 0 