import os

from moatless.benchmark.evaluation import TreeSearchSettings, ModelSettings
from moatless.benchmark.run_evaluation import evaluate_search_and_code

repo_base_dir = "/tmp/repos"
evaluations_dir = os.path.join(
    os.getenv("MOATLESS_DIR", "/home/albert/repos/albert/sw-planner-2/trajs"),
    "evaluations",
)


def test_gpt4o_mini():
    tree_search_settings = TreeSearchSettings(
        model=ModelSettings(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.7,
        ),
        max_expansions=3,
        max_iterations=50,
        min_finished_transitions=3,
        max_finished_transitions=5,
        provide_feedback=True,
        best_first=True,
        debate=False,
    )

    evaluate_search_and_code(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.7,
        max_transitions=30,
        tree_search_settings=tree_search_settings,
        evaluation_name="verify_gpt4o_mini",
        num_workers=1,
        instance_ids=["django__django-15252"],
    )
