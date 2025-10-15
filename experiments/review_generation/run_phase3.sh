#!/bin/bash

# Run Phase 3: Fine-tune Review Model
# Fine-tunes Llama 3 8B Instruct on paper->review pairs

# Set cache directory for huggingface
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
echo "Phase 3: Fine-tuning Review Model"
echo "=========================================="

cd "$(dirname "$0")"

# Run fine-tuning
# Adjust hyperparameters as needed based on your GPU
python3 phase3_finetune_model.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ../../models/finetuned/llama3-8b-reviewer \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_length 2048 \
    --warmup_steps 100 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10

echo ""
echo "Phase 3 complete!"
echo "Fine-tuned model saved to ../../models/finetuned/llama3-8b-reviewer"
echo ""
echo "To generate reviews with the fine-tuned model, run:"
echo "  python3 phase2_baseline_generation.py --model ../../models/finetuned/llama3-8b-reviewer --split test"

