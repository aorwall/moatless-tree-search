# models
# claude-3-5-sonnet-20241022
# gpt-4o-mini-2024-07-18
# gpt-4o-2024-08-06
# openai/Qwen/Qwen2.5-72B-Instruct
# openai/Qwen/Qwen2.5-72B-Instruct
# openai/Qwen/Qwen2.5-Coder-32B-Instruct
# claude-3-5-haiku-20241022

# paths lm_selector/1_feedback_tests/$MODEL \



REPOS="""
scikit-learn__scikit-learn-14983 \
astropy__astropy-14365 \
django__django-13033 \
django__django-14155 \
django__django-11179 \
"""

MODEL="openai/Qwen/Qwen2.5-Coder-32B-Instruct"
CWD=$(pwd)
export PYTHONPATH="${CWD}:${PYTHONPATH}"

python ./moatless/benchmark/run_evaluation.py \
        --model $MODEL \
        --repo_base_dir "$CWD/repos" \
        --eval_dir "$CWD/evaluations" \
        --eval_name debug/selector/26_feedback_tests/$MODEL \
        --temp 0.7 \
        --num_workers 5 \
        --format react \
        --feedback \
        --max_iterations 250 \
        --min_resolved 1 \
        --max_resolved 100 \
        --instance_ids scikit-learn__scikit-learn-25500
