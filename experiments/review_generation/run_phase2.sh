#!/bin/bash

# Run Phase 2: Baseline Review Generation
# Generates reviews using Llama 3 8B Instruct (zero-shot)

# Set cache directory for huggingface/vllm
export HF_HOME="/home/heck2/myousuf6/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/heck2/myousuf6/.cache/huggingface/transformers"

# Source the .env file to get the HuggingFace token
if [ -f "../../.env" ]; then
    export $(cat ../../.env | grep HF_TOKEN | xargs)
else
    echo "No .env file found. Please create one with HF_TOKEN=your_token"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "No HF_TOKEN found in .env file. Please add HF_TOKEN=your_token to .env"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "=========================================="
echo "Phase 2: Baseline Review Generation"
echo "=========================================="

cd "$(dirname "$0")"

# Default: Generate for validation split with a small batch for testing
# Uncomment and modify as needed

# Test with small batch first
echo "Generating baseline reviews for validation set (first 5 examples)..."
python3 phase2_baseline_generation.py \
    --split validation \
    --batch_size 5 \
    --temperature 0.7 \
    --max_tokens 1024

# Uncomment to generate for full validation set
# python3 phase2_baseline_generation.py --split validation

# Uncomment to generate for test set
# python3 phase2_baseline_generation.py --split test

echo ""
echo "Phase 2 complete!"
echo "Check the output in ../../data/generated_reviews/baseline/"

