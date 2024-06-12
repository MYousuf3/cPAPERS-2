#!/bin/bash

# Define the Python command and its arguments
python_command="python table_zs.py"
# python_command="python table_zs.py --model meta-llama/Llama-2-7b-chat-hf"
# python_command="python table_zs.py --model meta-llama/Meta-Llama-3-8B-Instruct"
# python_command="python table_zs.py --model meta-llama/Meta-Llama-3-70B-Instruct"
args=(
    "--split dev"
    "--use neighboring --modality table --split dev"
    "--use all --modality table --split dev"
    "--use neighboring --modality context --split dev"
    "--use all --modality context --split dev"
    "--use neighboring --modality reference --split dev"
    "--use all --modality reference --split dev"
    "--split test"
    "--use neighboring --modality table --split test"
    "--use all --modality table --split test"
    "--use neighboring --modality context --split test"
    "--use all --modality context --split test"
    "--use neighboring --modality reference --split test"
    "--use all --modality reference --split test"
)

# Execute each Python command and append output to a single result file
for arg in "${args[@]}"; do
    echo "Executing: $python_command $arg"
    echo "Command: $python_command $arg" >> results/zs_table.log
    $python_command $arg >> results/zs_table.log 2>&1
done
