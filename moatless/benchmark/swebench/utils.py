import fcntl
import logging
import os
import shutil
import contextlib
from typing import Optional

from moatless.benchmark.utils import (
    get_missing_files,
    get_missing_spans,
)
from moatless.index import CodeIndex
from moatless.repository.git import GitRepository
from moatless.repository.repository import Repository
from moatless.utils.repo import (
    setup_github_repo,
    get_repo_dir_name,
    retry_clone,
)

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def repository_lock(lock_path: str):
    """Context manager for handling repository locks safely"""
    lock_file = None
    try:
        # Create directory for lock if it doesn't exist
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        lock_file = open(lock_path, "w")
        logger.debug(f"Acquiring lock: {lock_path}")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    except Exception as e:
        logger.error(f"Error while holding repository lock: {e}")
        raise
    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
                logger.debug(f"Released lock: {lock_path}")
                # Clean up the lock file
                if os.path.exists(lock_path):
                    os.unlink(lock_path)
            except Exception as e:
                logger.warning(f"Error cleaning up lock file: {e}")

def verify_repository(repo_path: str, github_url: str) -> bool:
    """Verify if the directory at repo_path is a valid git repository"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except Exception as e:
        logger.warning(f"Repository verification failed: {e}")
        return False

def create_repository(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
    shallow_clone: bool = False,
) -> GitRepository:
    """
    Create a workspace for the given SWE-bench instance.
    
    Args:
        instance: The SWE-bench instance data
        instance_id: The instance ID to load data for
        repo_base_dir: Base directory for repositories
        shallow_clone: If True, perform a shallow clone directly from GitHub. If False, use local repo if available.
    """
    assert instance or instance_id, "Either instance or instance_id must be provided"
    if not instance:
        instance = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # Ensure the directory exists
    os.makedirs(repo_base_dir, exist_ok=True)

    repo_dir_name = get_repo_dir_name(instance["repo"])
    repo_path = f"{repo_base_dir}/swe-bench_{instance['instance_id']}"
    local_repo_path = f"{repo_base_dir}/swe-bench_{repo_dir_name}"
    lock_file_path = f"{local_repo_path}.lock" if not shallow_clone else f"{repo_path}.lock"
    github_url = f"https://github.com/swe-bench/{repo_dir_name}.git"

    # Check if repo exists and is valid
    if os.path.exists(repo_path):
        try:
            logger.info(f"Attempting to use existing repository at {repo_path}")
            repo = GitRepository(repo_path=repo_path)
            if verify_repository(repo_path, github_url):
                logger.info(f"Successfully initialized existing repository: {repo_path}")
                return repo
            else:
                logger.warning(f"Existing repository at {repo_path} is invalid, will recreate")
                shutil.rmtree(repo_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to use existing repository: {e}")
            shutil.rmtree(repo_path, ignore_errors=True)

    # Clone repository
    with repository_lock(lock_file_path):
        try:
            if shallow_clone:
                # Direct shallow clone to repo_path
                clone_msg = "shallow cloning"
                target_path = repo_path
                source = github_url
            else:
                # Use or create local repo for deep clone
                if not os.path.exists(local_repo_path) or not verify_repository(local_repo_path, github_url):
                    if os.path.exists(local_repo_path):
                        logger.info(f"Removing invalid repository at {local_repo_path}")
                        shutil.rmtree(local_repo_path, ignore_errors=True)
                    clone_msg = "cloning"
                    target_path = local_repo_path
                    source = github_url
                else:
                    # Local repo exists and is valid, clone from it
                    clone_msg = "cloning from local"
                    target_path = repo_path
                    source = f"file://{local_repo_path}"

            logger.info(f"{clone_msg} {source} to {target_path}")
            import subprocess
            import signal
            from datetime import datetime, timedelta

            # Configure git to use longer timeouts and larger buffers
            env = os.environ.copy()
            # Set very high timeouts for large repos
            env['GIT_HTTP_LOW_SPEED_LIMIT'] = '1000'  # 1 KB/s
            env['GIT_HTTP_LOW_SPEED_TIME'] = '3600'   # 60 minutes
            env['GIT_TERMINAL_PROMPT'] = '0'          # Disable prompts
            
            # Clone command with increased buffer size
            clone_cmd = [
                'git',
                '-c', 'http.postBuffer=1048576000',      # 1GB buffer
                '-c', 'http.lowSpeedLimit=1000',         # 1 KB/s
                '-c', 'http.lowSpeedTime=3600',          # 60 minutes
                '-c', 'core.compression=0',              # Disable compression to speed up clone
                '-c', 'http.maxRequests=5',              # Reduce concurrent requests
                'clone',
                '--progress'
            ]

            if shallow_clone:
                clone_cmd.extend(['--depth', '1', '--no-single-branch'])
            
            clone_cmd.extend([source, target_path])
            
            try:
                process = subprocess.Popen(
                    clone_cmd,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=env
                )
                
                # Report progress
                while True:
                    try:
                        line = process.stderr.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            logger.debug(line.strip())
                    except Exception as e:
                        logger.warning(f"Error reading git progress: {e}")
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, clone_cmd)
            
                if not verify_repository(target_path, github_url):
                    raise Exception("Repository verification failed after cloning")
                    
                logger.info(f"Successfully cloned repository to {target_path}")

                # If we just created the local repo, we need to clone it to repo_path
                if not shallow_clone and target_path == local_repo_path:
                    logger.info(f"Cloning from local repository to {repo_path}")
                    clone_cmd = [
                        'git', 'clone', f"file://{local_repo_path}", repo_path
                    ]
                    subprocess.run(clone_cmd, check=True)

            except Exception as e:
                logger.error(f"Failed to clone repository: {e}")
                if os.path.exists(target_path):
                    shutil.rmtree(target_path, ignore_errors=True)
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path, ignore_errors=True)
                raise RuntimeError(f"Failed to clone repository {source}: {str(e)}")
        except TimeoutError as e:
            logger.error(f"Clone operation timed out: {e}")
            if os.path.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
            raise RuntimeError(f"Repository clone timed out: {str(e)}")
        except subprocess.CalledProcessError as e:
            if e.returncode == -2:  # SIGINT
                logger.warning("Clone interrupted by user, cleaning up...")
            logger.error(f"Failed to clone repository: {e}")
            if os.path.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repository {source}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            if os.path.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repository {source}: {str(e)}")

    try:
        repo = GitRepository.from_repo(
            git_repo_url=github_url,
            repo_path=repo_path,
            commit=instance["base_commit"]
        )
        logger.info(f"Successfully created repository at {repo_path}")
        return repo
    except Exception as e:
        logger.error(f"Failed to create repository from {github_url}: {e}")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)
        raise RuntimeError(f"Failed to create repository from {github_url}: {str(e)}")


def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    from datasets import load_dataset

    data = load_dataset(dataset_name, split=split)
    return {d["instance_id"]: d for d in data}


def load_instance(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]


def sorted_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    sort_by: str = "created_at",
):
    from datasets import load_dataset

    data = load_dataset(dataset_name, split=split)
    instances = list(data)
    instances = sorted(instances, key=lambda x: x[sort_by])
    return instances


def get_repo_dir_name(repo: str):
    return repo.replace("/", "__")


def found_in_expected_spans(instance: dict, spans: dict):
    for file_path, span_ids in instance["expected_spans"].items():
        if not span_ids:
            logging.warning(
                f"{instance['instance_id']} Expected spans for {file_path} is empty"
            )

    missing_spans = get_missing_spans(instance["expected_spans"], spans)
    return not missing_spans


def found_in_alternative_spans(instance: dict, spans: dict):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.info(
                    f"{instance['instance_id']} Alternative spans for {file_path} is empty"
                )

        missing_spans = get_missing_spans(alternative_spans["spans"], spans)
        if not missing_spans:
            return True

    return False


def found_in_alternative_files(instance: dict, files: list):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.info(
                    f"{instance['instance_id']} Alternative spans for {file_path} is empty"
                )

        missing_spans = get_missing_files(alternative_spans["spans"], files)
        if not missing_spans:
            return True

    return False


def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )


def create_index(
    instance: dict,
    repository: Repository | None = None,
    index_store_dir: Optional[str] = None,
):
    """
    Create a workspace for the given SWE-bench instance.
    """
    if not index_store_dir:
        index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")

    if not repository:
        repository = create_repository(instance)

    code_index = CodeIndex.from_index_name(
        instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
    )
    return code_index
