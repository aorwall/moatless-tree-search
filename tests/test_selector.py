import logging

import pytest

from moatless.actions.action import Action
from moatless.actions.model import Observation, ActionArguments
from moatless.selector import Selector, BestFirstSelector, SoftmaxSelector
from moatless.node import Node, Reward


@pytest.fixture
def selector():
    return Selector()


@pytest.fixture
def best_first_selector():
    return BestFirstSelector()


@pytest.fixture
def softmax_selector():
    return SoftmaxSelector()


def test_uct_score(best_first_selector):
    node = Node(node_id=1)
    node.visits = 10
    node.reward = Reward(value=75, explanation="Test explanation")

    score = best_first_selector.uct_score(node)

    assert isinstance(score.final_score, float)
    assert score.exploitation == 75


def test_best_first_selector(best_first_selector):
    node1 = Node(node_id=1)
    node1.visits = 10
    node1.reward = Reward(value=75, explanation="Explanation 1")

    node2 = Node(node_id=2)
    node2.visits = 5
    node2.reward = Reward(value=90, explanation="Explanation 2")

    expandable_nodes = [node1, node2]

    selected_node = best_first_selector.select(expandable_nodes)
    assert selected_node == node2


def test_softmax_selector(softmax_selector):
    node1 = Node(node_id=1)
    node1.visits = 10
    node1.reward = Reward(value=75, explanation="Explanation 1")

    node2 = Node(node_id=2)
    node2.visits = 5
    node2.reward = Reward(value=90, explanation="Explanation 2")

    expandable_nodes = [node1, node2]

    selected_node = softmax_selector.select(expandable_nodes)
    assert selected_node in expandable_nodes


def create_node(node_id, action_name, reward_value, expansions, visits):
    return Node(
        node_id=node_id,
        action=ActionArguments(name=action_name),
        reward=Reward(value=reward_value),
        expansions=expansions,
        visits=visits,
        max_expansions=3,
    )


def test_expect_correction_bonus3():
    selector = BestFirstSelector()

    node43 = Node(
        node_id=43,
        visits=5,
        reward=Reward(value=75),
        output=Observation(expect_correction=True, message=""),
        children=[],
    )

    uct_score = selector.uct_score(node43)
    assert (
        uct_score.expect_correction_bonus == 100.0
    ), "Initial bonus should be 100 when no children are present"

    # Add children to Node43 one by one and check the expect correction bonus
    for i in range(1, 5):
        child_node = create_node(
            node_id=43 + i,
            action_name="RequestMoreContext",
            reward_value=85,
            expansions=0,
            visits=1,
        )
        node43.children.append(child_node)

        # Recalculate the UCT score
        uct_score = selector.uct_score(node43)
        expected_bonus = 100.0 / (1 + len(node43.children) ** 2)

        print(f"Node43 has {len(node43.children)} children {uct_score}")

        # Check that the expect correction bonus decreases as children are added
        assert (
            pytest.approx(uct_score.expect_correction_bonus, rel=1e-2) == expected_bonus
        ), f"Bonus should decrease with {len(node43.children)} children"
