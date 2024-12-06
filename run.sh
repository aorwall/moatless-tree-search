# models
# claude-3-5-sonnet-20241022
# gpt-4o-mini-2024-07-18
# gpt-4o-2024-08-06
# openai/Qwen/Qwen2.5-72B-Instruct
# openai/Qwen/Qwen2.5-72B-Instruct
# openai/Qwen/Qwen2.5-Coder-32B-Instruct
# claude-3-5-haiku-20241022

# paths lm_selector/1_feedback_tests/$MODEL \
# --instance_ids $REPOS \


# django__django-11179 \
# astropy__astropy-14365 \

REPOS="""
django__django-11179
astropy__astropy-14365
django__django-13033 \
django__django-14155 \
scikit-learn__scikit-learn-14983 \
"""

MODEL="openai/Qwen/Qwen2.5-Coder-32B-Instruct"
CWD=$(pwd)
export PYTHONPATH="${CWD}:${PYTHONPATH}"

python ./moatless/benchmark/run_evaluation.py \
        --model $MODEL \
        --repo_base_dir "$CWD/repos" \
        --eval_dir "$CWD/evaluations" \
        --eval_name debug/fin_bef_reexpand/1_feedback/$MODEL \
        --temp 0.7 \
        --num_workers 5 \
        --format react \
        --max_iterations 250 \
        --max_expansions 10 \
        --min_resolved 1 \
        --max_resolved 100 \
        --use_edit_actions \
        --feedback \
        --feedback_type agent \
        --use_testbed \
        --overwrite \
        --instance_ids $REPOS
