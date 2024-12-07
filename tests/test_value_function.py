from unittest.mock import Mock, patch

import pytest

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.model import RewardScaleEntry
from moatless.completion.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.node import Node
from moatless.value_function.base import ValueFunction


@pytest.fixture
def value_function():
    return ValueFunction(completion=CompletionModel(model="gpt"))


@pytest.fixture
def mock_node():
    node = Mock(spec=Node)
    node.get_root.return_value = Mock(message="Root problem statement")
    node.get_trajectory.return_value = [Mock(action=None, observation=None)]
    node.action = FinishArgs(
        scratch_pad="Task seems complete",
        finish_reason="All requirements met"
    )
    node.observation = Mock(message="Finish observation", terminal=True)
    node.possible_actions = ["Finish"]
    node.file_context = Mock(spec=FileContext)
    node.file_context.is_empty.return_value = False
    node.file_context.create_prompt.return_value = "File context prompt"
    node.file_context.generate_git_patch.return_value = "Git patch"
    return node


def test_create_system_prompt(value_function, mock_node):
    system_prompt = value_function._create_system_prompt(mock_node)
        
    assert isinstance(system_prompt, str)

    for critera in Finish.get_evaluation_criteria(0):
        assert critera in system_prompt

    for scale in Finish.get_reward_scale(0):
        assert scale.description in system_prompt


def test_create_message(value_function, mock_node):
    message = value_function._create_message(mock_node)
    
    assert "<problem_statement>" in message.content
    assert "<reasoning_for_completion>\nAll requirements met" in message.content
    assert "<file_context>" in message.content
    assert "<git_patch>" in message.content


def test_format_evaluation_criteria():
    criteria = ["Criterion 1", "Criterion 2"]
    formatted = ValueFunction._format_evaluation_criteria(criteria)
    assert "# Evaluation Criteria:" in formatted
    assert "* Criterion 1" in formatted
    assert "* Criterion 2" in formatted


def test_format_reward_scale():
    scale = [
        RewardScaleEntry(min_value=0, max_value=50, description="Low"),
        RewardScaleEntry(min_value=51, max_value=100, description="High")
    ]
    formatted = ValueFunction._format_reward_scale(scale, 0, 100)
    assert "# Reward Scale and Guidelines:" in formatted
    assert "* **0 to 50**: Low" in formatted
    assert "* **51 to 100**: High" in formatted


def test_model_dump(value_function):
    dump = value_function.model_dump()
    assert "completion" in dump
    assert isinstance(dump["completion"], dict)


def test_model_validate():
    mock_completion = Mock(spec=CompletionModel)
    mock_completion.model_dump.return_value = {"type": "mock"}
    data = {"completion": {"type": "mock"}}
    with patch.object(CompletionModel, 'model_validate', return_value=mock_completion):
        validated = ValueFunction.model_validate(data)
    assert isinstance(validated, ValueFunction)
    assert validated._completion == mock_completion
