# models
# claude-3-5-sonnet-20241022
# claude-3-5-haiku-20241022
# gpt-4o-2024-08-06
# gpt-4o-mini-2024-07-18
# openai/Qwen/Qwen2.5-72B-Instruct
# openai/Qwen/Qwen2.5-72B-Instruct
# openai/Qwen/Qwen2.5-Coder-32B-Instruct

# paths lm_selector/1_feedback_tests/$MODEL \
# --instance_ids $REPOS \


# django__django-11179 \
# astropy__astropy-14365 \

# --split sampled_50_instances

REPOS="""
django__django-11179
astropy__astropy-14365
django__django-13033
django__django-14155 
scikit-learn__scikit-learn-14983
django__django-14999
"""

# django__django-11964
DJANGO_UNSOLVED_CLAUDE="""
django__django-11848
django__django-14999
"""

MODEL="claude-3-5-haiku-20241022"
CWD=$(pwd)
export PYTHONPATH="${CWD}:${PYTHONPATH}"

python ./moatless/benchmark/run_evaluation.py \
        --model $MODEL \
        --repo_base_dir "$CWD/repos" \
        --eval_dir "$CWD/evaluations" \
        --eval_name debug/coding_value_function/16_feedback_tests_fin_bef/$MODEL \
        --temp 0.7 \
        --num_workers 3 \
        --format tools \
        --max_iterations 200 \
        --max_expansions 10 \
        --reward_threshold 101 \
        --max_finished_nodes 20 \
        --use_edit_actions \
        --feedback \
        --feedback_type agent \
        --use_testbed \
        --instance_ids django__django-11848