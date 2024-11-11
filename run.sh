# models
# claude-3-5-sonnet-20241022
# gpt-4o-mini-2024-07-18
# openai/Qwen/Qwen2.5-03B-Instruct
# openai/Qwen/Qwen2.5-72B-Instruct

MODEL="gpt-4o-mini-2024-07-18"
CWD=$(pwd)
export PYTHONPATH="${CWD}:${PYTHONPATH}"

python ./moatless/benchmark/run_evaluation.py \
        --model $MODEL \
        --repo_base_dir "$CWD/repos" \
        --eval_dir "$CWD/evaluations" \
        --eval_name debug/edit_actions/$MODEL \
        --temp 0.7 \
        --num_workers 1 \
        --feedback \
        --instance_id django__django-11179 \
        --max_iterations 50 \
        --max_expansions 5