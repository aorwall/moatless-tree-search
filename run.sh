# models
# claude-3-5-sonnet-20241022
# gpt-4o-mini-2024-07-18
# openai/Qwen/Qwen2.5-03B-Instruct


REPOS="""
astropy__astropy-14365 \
django__django-13033 \
django__django-14155 \
django__django-11179 \
"""

MODEL="gpt-4o-mini-2024-07-18"
CWD=$(pwd)
export PYTHONPATH="${CWD}:${PYTHONPATH}"

python ./moatless/benchmark/run_evaluation.py \
        --model $MODEL \
        --repo_base_dir "$CWD/repos" \
        --eval_dir "$CWD/evaluations" \
        --eval_name debug/selector/6/$MODEL \
        --temp 0.7 \
        --num_workers 5 \
        --feedback \
        --instance_id \
                scikit-learn__scikit-learn-14983 \
        --max_iterations 50 \
        --max_expansions 10 \
        --overwrite