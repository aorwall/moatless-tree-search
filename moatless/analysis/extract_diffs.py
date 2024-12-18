import json
import sys
from typing import List, Dict, Optional
from pathlib import Path
import datasets
from litellm import completion
from datetime import datetime

PATCH_ANALYSIS_PROMPT = """You are an expert code reviewer analyzing patches for a bug fix. Compare the golden (correct) patch with the generated patches and provide a detailed analysis.

Golden Patch:


Generated Patches:
{generated_patches}

Please analyze (state the Node IDs of the diffs you're alluding to):
1. Similarity between the generated patches.
2. How many unique diffs there are in the generated patches (group all the Node IDs according to diffs).
    a) excluding test files
    b) including test files
3. How similar/different are the generated patches to the golden patch?
4. Are the generated patches correct? If not, what's wrong with them?
4. What key changes does the golden patch make that the generated patches might have missed?
5. Would any of the generated patches fix the issue?
6. What feedback or tests could have been used to evaluate the correctness of the generated patches, in order to improve them (without knowing the golden patch)?

Provide your analysis in a clear, structured format."""

def analyze_patches_with_claude(golden_patch: str, diffs: List[Dict], model: str = "claude-3-sonnet-20240229") -> str:
    """
    Use Claude to analyze the differences between golden and generated patches.
    
    Args:
        golden_patch: The correct patch from SWE-bench
        diffs: List of generated patches
        model: The Claude model to use
    
    Returns:
        Claude's analysis of the patches
    """
    # Format generated patches to only show node_ids
    generated_patches_str = ""
    for diff in diffs:
        generated_patches_str += f"\nGenerated Patch (Node ID: {diff.get('node_id', 'unknown')}):\n```diff\n{diff['diff']}\n```\n"
    
    # Construct the prompt
    prompt = PATCH_ANALYSIS_PROMPT.format(
        golden_patch=golden_patch,
        generated_patches=generated_patches_str
    )
    
    try:
        # Call Claude via LiteLLM
        response = completion(
            model=model,
            messages=[{
                "role": "system",
                "content": "You are an expert code reviewer analyzing patches for bug fixes."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error analyzing patches with Claude: {str(e)}"

def load_swebench_data(instance_id: str) -> Optional[Dict]:
    """Load the SWE-bench data for a specific instance."""
    try:
        # Load SWE-bench dataset
        dataset = datasets.load_dataset("princeton-nlp/SWE-bench_Lite")
        
        # Search in both dev and test splits
        for split in ['dev', 'test']:
            split_data = dataset[split]
            # Find the matching instance
            for idx, item in enumerate(split_data):
                if item['instance_id'] == instance_id:
                    return {
                        'split': split,
                        'golden_patch': item['patch'],
                        'test_patch': item['test_patch'],
                        'problem_statement': item['problem_statement'],
                        'instance_id': item['instance_id'],
                        'repo': item['repo'],
                        'base_commit': item['base_commit'],
                        'fail_to_pass': item['FAIL_TO_PASS'],
                        'pass_to_pass': item['PASS_TO_PASS']
                    }
        
        print(f"Warning: Instance {instance_id} not found in SWE-bench dataset", file=sys.stderr)
        return None
        
    except Exception as e:
        print(f"Error loading SWE-bench dataset: {e}", file=sys.stderr)
        return None

def extract_diffs(trajectory_file: str, finished_only: bool = False) -> List[Dict]:
    """
    Extract all git diffs from a trajectory file, grouping diffs by node_id.
    
    Args:
        trajectory_file: Path to the trajectory JSON file
        finished_only: If True, only extract diffs from nodes with "Finish" actions
    
    Returns:
        List of dictionaries containing grouped diffs per node_id.
    """
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    # Extract instance_id from metadata
    instance_id = None
    if 'metadata' in data and 'instance_id' in data['metadata']:
        instance_id = data['metadata']['instance_id']
    
    # Use a dictionary to group diffs by node_id
    diffs_by_node = {}
    
    def is_finished_state(node):
        """Check if a node represents a finished state"""
        if isinstance(node, dict):
            # Check action_steps array
            if 'action_steps' in node and isinstance(node['action_steps'], list):
                for step in node['action_steps']:
                    if isinstance(step, dict) and 'action' in step:
                        action = step['action']
                        if isinstance(action, dict):
                            action_args_class = action.get('action_args_class', '').lower()
                            if 'finish' in action_args_class:
                                return True
            
            # Check direct action field
            if 'action' in node:
                action = node['action']
                if isinstance(action, dict):
                    action_args_class = action.get('action_args_class', '').lower()
                    if 'finish' in action_args_class:
                        return True
                elif isinstance(action, str) and 'finish' in action.lower():
                    return True
                
            # Check observation for terminal state
            if 'observation' in node:
                obs = node['observation']
                if isinstance(obs, dict) and obs.get('terminal', False):
                    return True
        return False

    def search_for_diffs(obj, parent_finished: bool = False, node_id: str = None):
        if not isinstance(obj, (dict, list)):
            return

        current_finished = parent_finished or is_finished_state(obj)
        
        if isinstance(obj, dict):
            current_node_id = obj.get('node_id', node_id)
            
            if not finished_only or current_finished:
                if 'file_path' in obj and 'patch' in obj:
                    if obj['patch']:
                        if current_node_id not in diffs_by_node:
                            diffs_by_node[current_node_id] = []
                        diffs_by_node[current_node_id].append({
                            'file_path': obj['file_path'],
                            'diff': obj['patch']
                        })
                
                if 'properties' in obj and isinstance(obj['properties'], dict):
                    if 'diff' in obj['properties']:
                        diff = obj['properties']['diff']
                        if diff:
                            if current_node_id not in diffs_by_node:
                                diffs_by_node[current_node_id] = []
                            diffs_by_node[current_node_id].append({
                                'file_path': diff.split('\n')[0].replace('--- ', ''),
                                'diff': diff
                            })
            
            for key, value in obj.items():
                search_for_diffs(value, current_finished, current_node_id)
        
        elif isinstance(obj, list):
            for item in obj:
                search_for_diffs(item, current_finished, node_id)

    search_for_diffs(data)
    
    # Convert the grouped diffs into the final format
    diffs = []
    for node_id, node_diffs in diffs_by_node.items():
        # Combine all diffs for this node
        combined_diff = ""
        for diff_info in node_diffs:
            combined_diff += f"=== {diff_info['file_path']} ===\n{diff_info['diff']}\n\n"
        
        diffs.append({
            'node_id': node_id or 'unknown',
            'diff': combined_diff.strip(),
            'file_paths': [d['file_path'] for d in node_diffs]
        })
    
    if finished_only:
        print(f"Found {len(diffs)} nodes with diffs in finished states", file=sys.stderr)
    
    return diffs, instance_id

def save_analysis(trajectory_file: Path, swebench_data: Dict, diffs: List[Dict], claude_analysis: Optional[str] = None):
    """
    Save all analysis data to a solutions_analysis.json file in the same directory as the trajectory file.
    """
    trajectory_path = Path(trajectory_file)
    solutions_path = trajectory_path.parent / "solutions_analysis.json"
    
    analysis_data = {
        "timestamp": datetime.now().isoformat(),
        "trajectory_file": str(trajectory_path),
        "swebench_data": swebench_data,
        "generated_diffs": [{
            "file_paths": diff["file_paths"],
            "diff": diff["diff"],
            "node_id": diff.get("node_id", "unknown")
        } for diff in diffs],
        "claude_analysis": claude_analysis
    }
    
    # Create directory if it doesn't exist
    solutions_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save analysis data
    with open(solutions_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\nAnalysis saved to: {solutions_path}", file=sys.stderr)

def save_text_analysis(trajectory_file: Path, swebench_data: Dict, diffs: List[Dict], claude_analysis: Optional[str] = None):
    """
    Save analysis data to a formatted text file in the same directory as the trajectory file.
    """
    trajectory_path = Path(trajectory_file)
    text_path = trajectory_path.parent / "solutions_analysis.txt"
    
    with open(text_path, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("SWE-bench Solution Analysis\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        # Problem Info
        f.write("Problem Information\n")
        f.write("-" * 80 + "\n")
        f.write(f"Instance ID: {swebench_data.get('instance_id', 'N/A')}\n")
        f.write(f"Repository: {swebench_data.get('repo', 'N/A')}\n")
        f.write(f"Base Commit: {swebench_data.get('base_commit', 'N/A')}\n")
        f.write(f"Split: {swebench_data.get('split', 'N/A')}\n\n")
        
        # Problem Statement
        if "problem_statement" in swebench_data:
            f.write("Problem Statement\n")
            f.write("-" * 80 + "\n")
            f.write(swebench_data["problem_statement"])
            f.write("\n\n")
        
        # Golden Patch
        if "golden_patch" in swebench_data:
            f.write("Golden Patch\n")
            f.write("-" * 80 + "\n")
            f.write(swebench_data["golden_patch"])
            f.write("\n\n")
        
        # Generated Patches
        f.write("Generated Patches\n")
        f.write("-" * 80 + "\n")
        for diff in diffs:
            f.write(f"\nNode ID: {diff.get('node_id', 'unknown')}\n")
            f.write(f"Files: {', '.join(diff['file_paths'])}\n")
            f.write("-" * 40 + "\n")
            f.write(diff["diff"])
            f.write("\n")
        
        # Claude Analysis
        if claude_analysis:
            f.write("\nClaude Analysis\n")
            f.write("-" * 80 + "\n")
            f.write(claude_analysis)
            f.write("\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract git diffs from trajectory file')
    parser.add_argument('trajectory_file', help='Path to trajectory JSON file')
    parser.add_argument('--finished-only', '-f', action='store_true', 
                       help='Only extract diffs from finished states')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Print debug information')
    parser.add_argument('--show-golden', '-g', action='store_true',
                       help='Show golden patch and problem info')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Use Claude to analyze patches')
    parser.add_argument('--model', '-m', default="claude-3-sonnet-20240229",
                       help='Claude model to use for analysis')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save analysis to solutions_analysis.json and .txt')
    
    args = parser.parse_args()
    
    # Extract diffs and instance_id
    diffs, instance_id = extract_diffs(args.trajectory_file, args.finished_only)
    
    swebench_data = None
    if args.show_golden or args.analyze or args.save:
        if instance_id:
            swebench_data = load_swebench_data(instance_id)
            if swebench_data and args.show_golden:
                print("\n=== SWE-bench Metadata ===")
                print(f"Instance ID: {swebench_data['instance_id']}")
                print(f"Repository: {swebench_data['repo']}")
                print(f"Base Commit: {swebench_data['base_commit']}")
                print(f"Split: {swebench_data['split']}")
                print("\n=== Problem Statement ===")
                print(swebench_data['problem_statement'])
                print("\n=== Golden Patch ===")
                print(swebench_data['golden_patch'])
                print("\n=== Test Patch ===")
                print(swebench_data['test_patch'])
                print("\n=== Tests ===")
                print("Fail to Pass:", swebench_data['fail_to_pass'])
                print("Pass to Pass:", swebench_data['pass_to_pass'])
                print("=" * 80)
    
    if args.debug:
        # Print metadata from trajectory file
        with open(args.trajectory_file) as f:
            data = json.load(f)
            if 'metadata' in data:
                print("\n=== Trajectory Metadata ===", file=sys.stderr)
                print(json.dumps(data['metadata'], indent=2), file=sys.stderr)
    
    # Print all found diffs
    print("\n=== Extracted Diffs ===")
    for diff_info in diffs:
        print(f"\nNode ID: {diff_info.get('node_id', 'unknown')}")
        print(f"Files: {', '.join(diff_info['file_paths'])}")
        print("=" * 80)
        print(diff_info['diff'])
        print("=" * 80)
    
    # Analyze patches if requested
    claude_analysis = None
    if args.analyze and swebench_data and diffs:
        print("\n=== Claude Analysis ===")
        claude_analysis = analyze_patches_with_claude(
            swebench_data['golden_patch'],
            diffs,
            model=args.model
        )
        print(claude_analysis)
    
    # Save analysis if requested
    if args.save and swebench_data:
        save_analysis(args.trajectory_file, swebench_data, diffs, claude_analysis)
        save_text_analysis(args.trajectory_file, swebench_data, diffs, claude_analysis)

if __name__ == "__main__":

    # sys.argv = [
    #     "moatless/analysis/extract_diffs.py",
    #     "/share/edc/home/antonis/_swe-planner/moatless-tree-search/evaluations/debug/coding_value_function/13_feedback_tests_fin_bef/claude-3-5-haiku-latest/django__django-11848/trajectory.json",
    #     "-g", "-a", "-s"
    # ]
    main()