import pytest

from moatless.actions.code_change import RequestCodeChangeArgs
from moatless.actions.run_tests import RunTestsArgs
from moatless.feedback import FeedbackGenerator
from moatless.node import Node, Reward


@pytest.fixture
def feedback_generator():
    return FeedbackGenerator()


def test_generate_feedback_no_children(feedback_generator):
    node = Node(node_id=1)
    feedback = feedback_generator.generate_feedback(node)
    assert feedback is None


def test_generate_feedback_run_tests(feedback_generator):
    parent_node = Node(node_id=1)
    child_node = Node(node_id=2, parent=parent_node)
    child_node.action = RunTestsArgs(test_files=["test_file.py"])
    child_node.reward = Reward(value=75, explanation="Test explanation")

    parent_node.children.append(child_node)

    feedback = feedback_generator.generate_feedback(parent_node)
    assert "Test explanation" in feedback
    assert "<Alternative_Branch_Action>" in feedback


def test_generate_feedback_code_change(feedback_generator):
    parent_node = Node(node_id=1)
    request_code_change = RequestCodeChangeArgs(
        scratch_pad="Change the instantiation of CommandParser to use self.prog_name for the prog argument.",
        file_path="django/core/management/__init__.py",
        instructions="Change the instantiation of CommandParser to use self.prog_name for the prog argument.",
        pseudo_code="parser = CommandParser(prog=self.prog_name, usage='%(prog)s subcommand [options] [args]', add_help=False, allow_abbrev=False)",
        change_type="modification",
        start_line=347,
        end_line=347,
    )
    child_node = Node(node_id=2, parent=parent_node)
    child_node.action = request_code_change
    child_node.reward = Reward(
        value=75,
        explanation="Code change explanation",
        feedback="Feedback from alternative branch",
    )

    parent_node.children.append(child_node)

    feedback = feedback_generator.generate_feedback(parent_node)
    assert "<feedback>\nFeedback from alternative branch\n</feedback>" in feedback
    assert "<Alternative_Branch_Action>" not in feedback
