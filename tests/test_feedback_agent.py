import unittest
from unittest.mock import Mock, patch
import json
import os
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.node import Node
from moatless.actions.action import Action
from moatless.completion.completion import CompletionModel
from moatless.search_tree import SearchTree
from moatless.completion.completion import LLMResponseFormat

class TestFeedbackAgent(unittest.TestCase):
    def setUp(self):
        # Create a proper mock of CompletionModel with all required attributes
        self.mock_completion_model = Mock(spec=CompletionModel)
        
        # Set all required attributes from CompletionModel class
        self.mock_completion_model.model = "test-model"
        self.mock_completion_model.temperature = 0.7
        self.mock_completion_model.max_tokens = 1000
        self.mock_completion_model.model_base_url = None
        self.mock_completion_model.model_api_key = None
        self.mock_completion_model.response_format = LLMResponseFormat.JSON
        self.mock_completion_model.stop_words = None
        self.mock_completion_model.metadata = None
        
        # Mock methods
        self.mock_completion_model.create_completion = Mock(return_value=("", None))
        self.mock_completion_model.validate_response_format = Mock(return_value=True)
        self.mock_completion_model.model_dump = Mock(return_value={
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1000,
            "model_base_url": None,
            "model_api_key": None,
            "response_format": "json",
            "stop_words": None,
            "metadata": None
        })
        
        # Initialize the FeedbackAgent
        self.feedback_agent = FeedbackAgent(
            completion_model=self.mock_completion_model,
            instance_dir="test_instance"
        )
        
        # Load real trajectory from file
        trajectory_path = "/share/edc/home/antonis/_swe-planner/moatless-tree-search/evaluations/debug/coding_value_function/10_feedback_tests_fin_bef/openai/Qwen/Qwen2.5-Coder-32B-Instruct/django__django-10914/trajectory.json"
        
        if os.path.exists(trajectory_path):
            self.search_tree = SearchTree.from_file(trajectory_path)
            self.root = self.search_tree.root
        else:
            # Fallback to mock tree if trajectory file not found
            self.root = self._create_mock_tree()

    def _create_mock_tree(self):
        """Fallback method to create a mock tree if trajectory file isn't available"""
        root = Node(node_id=0)
        node1 = Node(node_id=1, parent=root)
        node1.action = Action(name="SemanticSearch")
        root.children.append(node1)
        
        node2 = Node(node_id=2, parent=node1)
        node2.action = Action(name="ViewCode")
        node1.children.append(node2)
        
        node6 = Node(node_id=6, parent=node1)
        node6.action = Action(name="StringReplace")
        node1.children.append(node6)
        
        node3 = Node(node_id=3, parent=node2)
        node2.children.append(node3)
        
        return root

    def test_create_trajectory_explanation(self):
        """Test that trajectory explanation contains all required components"""
        # Get a node from the middle of the tree for testing
        test_node = None
        for node in self.root.get_all_nodes():
            if node.action and node.parent and node.parent.parent:  # Get a node with some history
                test_node = node
                break
        
        if not test_node:
            self.skipTest("No suitable node found in the trajectory")
        
        explanation = self.feedback_agent._create_trajectory_explanation(test_node)
        
        # Check that the explanation contains key components
        self.assertIn("Current Trajectory (Your History)", explanation)
        self.assertIn("Alternative Trajectories (Parallel Universes)", explanation)
        
        # Verify trajectory information
        trajectory = test_node.get_trajectory()
        for node in trajectory:
            self.assertIn(f"Node {node.node_id}", explanation)
            if node.action:
                self.assertIn(node.action.name, explanation)

        # Verify sibling information
        siblings = test_node.get_sibling_nodes()
        if siblings:
            for sibling in siblings:
                self.assertIn(f"Node {sibling.node_id}", explanation)

    def test_trajectory_explanation_in_messages(self):
        """Test that trajectory explanation is included in analysis messages"""
        test_node = next((node for node in self.root.get_all_nodes() 
                         if node.action and node.parent), None)
        
        if not test_node:
            self.skipTest("No suitable node found in the trajectory")
            
        messages = self.feedback_agent._create_analysis_messages(
            test_node,
            test_node.get_sibling_nodes(),
            include_tree=False
        )
        
        # Find the trajectory explanation message
        trajectory_message = None
        for message in messages:
            if "Understanding Trajectories" in message.content:
                trajectory_message = message
                break
        
        self.assertIsNotNone(trajectory_message)
        self.assertIn("<trajectory_explanation>", trajectory_message.content)
        self.assertIn("</trajectory_explanation>", trajectory_message.content)

    def test_root_node_trajectory(self):
        """Test trajectory explanation for root node"""
        explanation = self.feedback_agent._create_trajectory_explanation(self.root)
        
        # Root node should have a minimal trajectory
        self.assertIn(f"Node {self.root.node_id}", explanation)
        self.assertNotIn("→", explanation)  # No trajectory arrows for single node

    def test_action_names_in_trajectory(self):
        """Test that action names are properly included in trajectory explanation"""
        # Find a node with actions in its trajectory
        test_node = next((node for node in self.root.get_all_nodes() 
                         if node.action and node.parent and node.parent.action), None)
        
        if not test_node:
            self.skipTest("No suitable node found in the trajectory")
        
        explanation = self.feedback_agent._create_trajectory_explanation(test_node)
        
        # Check that action names from the trajectory are in the explanation
        trajectory = test_node.get_trajectory()
        for node in trajectory:
            if node.action:
                self.assertIn(node.action.name, explanation)

if __name__ == '__main__':
    unittest.main() 