from moatless.repository import FileRepository
from moatless.index.code_index import CodeIndex
from moatless.runtime.testbed import TestbedEnvironment
from moatless.search_tree import SearchTree
from moatless.completion.completion import CompletionModel
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.benchmark.utils import get_moatless_instance
from moatless.benchmark.swebench import create_repository, create_index
import json

def verify_generated_action(
    trajectory_path: str,
    instance_id: str,
    node_id: int):
    instance = get_moatless_instance(instance_id)
    repository = create_repository(
        instance
    )
    code_index = create_index(instance, repository=repository)

    runtime = TestbedEnvironment(
        repository=repository,
        instance=instance,
    )

    # First load the raw JSON to modify the data
    with open(trajectory_path, 'r') as f:
        tree_data = json.load(f)

    # Recursively process nodes to add missing timestamp if needed
    def process_node(node_data):
        if 'feedback_data' in node_data and isinstance(node_data['feedback_data'], dict):
            if 'timestamp' not in node_data['feedback_data']:
                node_data['feedback_data']['timestamp'] = ""
        if 'message' not in node_data or node_data['message'] is None:
            node_data['message'] = ""
        if 'children' in node_data:
            for child in node_data['children']:
                process_node(child)

    # Process the root node
    process_node(tree_data['root'])

    # Now create the search tree with the processed data
    search_tree = SearchTree.from_dict(
        tree_data,
        repository=repository,
        runtime=runtime,
        code_index=code_index,
    )

    # Get the specific node
    node = search_tree.get_node_by_id(node_id)
    if not node:
        print(f"Node {node_id} not found in trajectory")
        return

    # Print debug info about the node
    print(f"\nNode {node_id} details:")
    print(f"Message: {node.message}")
    print(f"Has observation: {node.observation is not None}")
    print(f"Has reward: {node.reward is not None}")
    if node.parent:
        print(f"Parent node ID: {node.parent.node_id}")
        print(f"Parent message: {node.parent.message}")
    if node.observation:
        print(f"\nObservation: {node.observation}")
    if node.reward:
        print(f"Reward: {node.reward.value}")
    
    # Create feedback generator if not exists
    feedback_generator = search_tree.create_feedback_generator()
    if not feedback_generator:
        print("Could not create feedback generator")
        return

    # Generate feedback for the specific node
    try:
        if not node.message:
            print("\nWarning: Node has no message. Using observation as message.")
            node.message = str(node.observation) if node.observation else "No message or observation available"
            
        # Also ensure parent nodes have messages
        current = node.parent
        while current:
            if not current.message:
                current.message = str(current.observation) if current.observation else "No message available"
            current = current.parent
            
        feedback = feedback_generator.generate_feedback(node)
        print(f"\nFeedback for Node {node_id}:")
        print(feedback)
    except Exception as e:
        print(f"\nError generating feedback:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        # Print the node's message history if available
        if hasattr(node, 'message_history'):
            print("\nNode message history:")
            print(node.message_history)


verify_generated_action(
    "/share/edc/home/antonis/_swe-planner/moatless-tree-search/evaluations/albert/albert_qwen_coder_32b/wrong_feedback_finished/trajectory (3).json",
    "sympy__sympy-13971",
    8
)