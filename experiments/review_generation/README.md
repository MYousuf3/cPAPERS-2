# Review Generation Pipeline

A complete pipeline for generating paper reviews using LLMs, fine-tuning on real reviews, and comparing outputs using AlpacaFarm.

## Overview

This pipeline implements a 3-phase workflow:

1. **Phase 1: Data Preparation** - Split dataset and prepare training data
2. **Phase 2: Baseline Generation** - Generate reviews using Llama 3 8B Instruct (zero-shot)
3. **Phase 3: Fine-tuning** - Train a specialized review model on paper→review pairs

After these phases, you can compare the baseline (Model A) with the fine-tuned model (Model B) using AlpacaFarm.

## Directory Structure

```
review_generation/
├── phase1_data_preparation.py    # Data splitting and preparation
├── phase2_baseline_generation.py # Zero-shot review generation
├── phase3_finetune_model.py      # Fine-tune review model
├── paper_loader.py                # Load paper content from LaTeX
├── utils.py                       # Utility functions
├── run_phase1.sh                  # Run Phase 1
├── run_phase2.sh                  # Run Phase 2  
├── run_phase3.sh                  # Run Phase 3
└── README.md                      # This file

Data directories (created during pipeline):
../../data/
├── gpt_cleaned_reviews_top1000.json    # Original dataset
├── processed/                          # Train/val/test splits
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   ├── train_training.json
│   ├── validation_training.json
│   ├── test_training.json
│   └── paper_id_splits.json
└── generated_reviews/
    ├── baseline/                       # Model A outputs
    └── finetuned/                      # Model B outputs

../../models/
└── finetuned/
    └── llama3-8b-reviewer/            # Fine-tuned model checkpoints
```

## Requirements

### Python Packages
```bash
pip install torch transformers datasets accelerate
pip install vllm  # For efficient inference
pip install rouge-score sacrebleu nltk bert-score
```

### Hardware
- **Phase 1**: CPU only
- **Phase 2**: GPU recommended (24GB+ VRAM for Llama 3 8B)
- **Phase 3**: GPU required (40GB+ VRAM for full fine-tuning, or use gradient checkpointing)

### Environment Setup
Create a `.env` file in the project root with your HuggingFace token:
```
HF_TOKEN=your_huggingface_token_here
```

## Quick Start

### 1. Run Phase 1: Data Preparation
```bash
bash run_phase1.sh
```

This will:
- Load the review dataset (1000 papers)
- Split into train/validation/test (700/150/150 papers)
- Create training-ready formats
- Output statistics about the dataset

**Output**: Processed data in `../../data/processed/`

### 2. Run Phase 2: Baseline Generation
```bash
bash run_phase2.sh
```

This will:
- Load Llama 3 8B Instruct model
- Generate reviews for validation set (zero-shot)
- Save generated reviews with ground truth

**Note**: By default, runs on first 5 examples for testing. Edit `run_phase2.sh` to run on full dataset.

**Output**: Generated reviews in `../../data/generated_reviews/baseline/`

### 3. Run Phase 3: Fine-tuning
```bash
bash run_phase3.sh
```

This will:
- Load training data
- Fine-tune Llama 3 8B Instruct on paper→review pairs
- Save checkpoints during training
- Evaluate on validation set

**Training time**: ~3-6 hours on A100 (depends on dataset size and hyperparameters)

**Output**: Fine-tuned model in `../../models/finetuned/llama3-8b-reviewer/`

### 4. Generate Reviews with Fine-tuned Model
```bash
python phase2_baseline_generation.py \
    --model ../../models/finetuned/llama3-8b-reviewer \
    --split test \
    --temperature 0.7
```

**Output**: Fine-tuned model reviews in `../../data/generated_reviews/finetuned/`

## Usage Details

### Phase 1: Data Preparation

```bash
python phase1_data_preparation.py
```

**Options**: Modify `main()` function to change:
- Train/val/test split ratios (default: 70/15/15)
- Random seed for reproducibility (default: 42)

**Output files**:
- `train.json`, `validation.json`, `test.json` - Full review entries
- `train_training.json`, `validation_training.json`, `test_training.json` - Training format
- `paper_id_splits.json` - Paper IDs for each split

### Phase 2: Baseline Generation

```bash
python phase2_baseline_generation.py \
    --split validation \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --temperature 0.7 \
    --max_tokens 1024 \
    --batch_size 5
```

**Arguments**:
- `--split`: Which split to use (train/validation/test)
- `--model`: Model name or path
- `--temperature`: Sampling temperature (0.0-1.0)
- `--max_tokens`: Maximum tokens to generate
- `--top_p`: Top-p sampling parameter
- `--batch_size`: Process only first N examples (for testing)

### Phase 3: Fine-tuning

```bash
python phase3_finetune_model.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ../../models/finetuned/llama3-8b-reviewer \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5
```

**Arguments**:
- `--model`: Base model to fine-tune
- `--output_dir`: Where to save checkpoints
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--max_length`: Maximum sequence length
- `--warmup_steps`: Warmup steps
- `--save_steps`: Save checkpoint every N steps
- `--eval_steps`: Evaluate every N steps

**Memory optimization**:
- Model uses bfloat16 training
- Gradient checkpointing enabled
- Adjust `batch_size` and `gradient_accumulation_steps` based on GPU memory

## Paper Content Loading

The pipeline needs to extract paper content from LaTeX source files. This is handled by `paper_loader.py`.

**Current Status**: Basic implementation with TODOs marked

**TODO Items**:
1. Parse LaTeX files to extract sections (abstract, introduction, methodology, results, conclusion)
2. Clean LaTeX markup (equations, citations, figures)
3. Handle different paper formats and structures

**Testing the paper loader**:
```bash
python paper_loader.py
```

This will test loading a sample paper and display extracted sections.

## Evaluation (Phase 4 - AlpacaFarm)

After generating reviews from both models, use AlpacaFarm for pairwise comparison:

### Setup AlpacaFarm
```bash
pip install alpaca-farm
```

### Compare Reviews
You'll need to:
1. Format outputs for AlpacaFarm (pairs of reviews)
2. Run AlpacaFarm evaluator (uses GPT-4 as judge)
3. Calculate win rates and metrics

**Evaluation Criteria**:
- Correctness: Technical accuracy
- Completeness: Coverage of paper aspects
- Specificity: Concrete vs. vague feedback
- Constructiveness: Actionable suggestions

## Dataset Information

**Source**: `gpt_cleaned_reviews_top1000.json`

**Structure**:
```json
{
  "submission_id": "...",
  "paper_id": "1804.08450v1",
  "review_id": "...",
  "submission_title": "Paper Title",
  "review": "Full review text...",
  "cleaned_review_points": [
    "Review point 1",
    "Review point 2",
    ...
  ]
}
```

**Statistics**:
- Total papers: 1000
- Average reviews per paper: ~3
- Average review points per entry: ~15

## Customization

### Custom Prompts
Edit the prompt templates in:
- `phase2_baseline_generation.py` → `create_review_prompt()`
- `phase3_finetune_model.py` → `format_training_example()`

### Different Models
Both Phase 2 and Phase 3 support any HuggingFace model:
```bash
# Use a different model
python phase2_baseline_generation.py --model meta-llama/Llama-2-7b-chat-hf
python phase3_finetune_model.py --model mistralai/Mistral-7B-Instruct-v0.2
```

### Hyperparameter Tuning
Key hyperparameters to tune in Phase 3:
- `learning_rate`: 1e-5 to 5e-5 (lower for stability)
- `epochs`: 2-5 (monitor validation loss)
- `batch_size` × `gradient_accumulation_steps`: Effective batch size 16-32
- `max_length`: 2048-4096 (longer for detailed reviews)

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size`
- Reduce `max_length`
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing (already enabled)

### Model Loading Issues
- Check HF_TOKEN is set correctly
- Verify you have access to gated models (Llama requires approval)
- Check cache directory has enough space

### Paper Content Not Loading
- Verify `extracted_files` directory exists
- Check paper directories exist for your paper_ids
- Run `python paper_loader.py` to test
- Complete TODOs in `paper_loader.py` for better extraction

## Next Steps

1. **Complete paper loading**: Implement TODOs in `paper_loader.py`
2. **Run full pipeline**: Execute all phases on complete dataset
3. **Evaluate with AlpacaFarm**: Compare Model A vs Model B
4. **Analyze results**: Which model performs better? On what types of papers?
5. **Iterate**: Improve prompts, model selection, hyperparameters

## References

- **Llama 3**: Meta's latest language model
- **vLLM**: High-performance inference engine
- **AlpacaFarm**: LLM evaluation framework
- **HuggingFace Transformers**: Model training and inference

## Contact

For questions or issues, refer to the main project documentation.

