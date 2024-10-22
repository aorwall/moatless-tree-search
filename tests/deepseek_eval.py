import logging
import os

from moatless.benchmark.evaluation import Evaluation, TreeSearchSettings
from moatless.benchmark.run_evaluation import evaluate_search_and_code

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

repo_base_dir = "/tmp/repos"
evaluations_dir = os.path.join(
    os.getenv("MOATLESS_DIR", "/home/albert/repos/albert/sw-planner-2/trajs"),
    "evaluations",
)

tree_search_settings = TreeSearchSettings(
    max_expansions=3,
    max_iterations=50,
    min_finished_transitions=3,
    max_finished_transitions=5,
    reward_threshold=90,
    provide_feedback=True,
    debate=False,
)


def test_gpt4o_mini():
    evaluate_search_and_code(
        model="gpt-4o-mini-2024-07-18",
        temperature=1.0,
        instance_ids=["django__django-11039"],  # django_dataset,
        max_transitions=30,
        tree_search_settings=tree_search_settings,
        name="django",
        evaluation_name="20240918_django_search_feedback",
        num_workers=1,
    )
