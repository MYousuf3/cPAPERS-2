#!/bin/bash

# Set cache directory for huggingface/vllm to use local storage instead of nethome
export HF_HOME="/home/heck2/myousuf6/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/heck2/myousuf6/.cache/huggingface/transformers"

# Source the .env file to get the HuggingFace token
if [ -f "../.env" ]; then
    export $(cat ../.env | grep HF_TOKEN | xargs)
else
    echo "No .env file found in parent directory. Please create one with HF_TOKEN=your_token"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "No HF_TOKEN found in .env file. Please add HF_TOKEN=your_token to .env"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Create cache directory if it doesn't exist
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"

# Define the Python command and its arguments
python_command="python equation_zs.py"
# python_command="python equation_zs.py --model meta-llama/Llama-2-7b-chat-hf"
# python_command="python equation_zs.py --model meta-llama/Meta-Llama-3-8B-Instruct"
# python_command="python equation_zs.py --model meta-llama/Meta-Llama-3-70B-Instruct"
args=(
    "--split dev"
    "--use neighboring --modality equation --split dev"
    "--use all --modality equation --split dev"
    "--use neighboring --modality context --split dev"
    "--use all --modality context --split dev"
    "--use neighboring --modality reference --split dev"
    "--use all --modality reference --split dev"
    "--split test"
    "--use neighboring --modality equation --split test"
    "--use all --modality equation --split test"
    "--use neighboring --modality context --split test"
    "--use all --modality context --split test"
    "--use neighboring --modality reference --split test"
    "--use all --modality reference --split test"
)

# Execute each Python command and append output to a single result file
for arg in "${args[@]}"; do
    echo "Executing: $python_command $arg"
    echo "Command: $python_command $arg" >> zs_equation.log
    $python_command $arg >> zs_equation.log 2>&1
done
