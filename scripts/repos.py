import os
import logging
from moatless.benchmark.utils import get_moatless_instances
from moatless.benchmark.swebench.utils import create_repository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(split: str = "lite", repo_base_dir: str = None):
    """
    Main function to fetch moatless instances and create repositories for each instance.

    Args:
        split (str): The dataset split to fetch instances from (default: "lite").
        repo_base_dir (str): Base directory to create repositories. Defaults to the current directory's `repos`.
    """
    # Get the absolute path of the current repository
    current_repo_path = os.path.abspath(os.getcwd())
    logger.info(f"Current repository path: {current_repo_path}")

    # Set the repository base directory
    if not repo_base_dir:
        repo_base_dir = os.path.join(current_repo_path, "repos")
    repo_base_dir = os.path.abspath(repo_base_dir)

    logger.info(f"Repository base directory: {repo_base_dir}")

    # Fetch instances using the provided method
    logger.info(f"Fetching instances for split '{split}'...")
    instances = get_moatless_instances(split)

    if not instances:
        logger.warning(f"No instances found for split '{split}'. Exiting.")
        return

    # Ensure the base directory exists
    os.makedirs(repo_base_dir, exist_ok=True)

    for instance_id, instance in instances.items():
        # Define the repository path for this instance
        repo_dir_name = f"swe-bench_{instance_id}"
        repo_path = os.path.join(repo_base_dir, repo_dir_name)

        if os.path.exists(repo_path):
            logger.info(f"Repository already exists for instance '{instance_id}' at '{repo_path}'. Skipping...")
            continue

        try:
            logger.info(f"Creating repository for instance '{instance_id}'...")
            create_repository(instance=instance, repo_base_dir=repo_base_dir)
            logger.info(f"Repository created for instance '{instance_id}' at '{repo_path}'.")
        except Exception as e:
            logger.error(f"Failed to create repository for instance '{instance_id}': {e}")

if __name__ == "__main__":
    # Specify the split and let the script determine the repo directory
    main(split="lite")
