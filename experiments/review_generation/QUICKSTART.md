# Quick Start Guide

## Setup (One Time)

### 1. Install Dependencies
```bash
pip install torch transformers datasets accelerate vllm
pip install rouge-score sacrebleu nltk bert-score
pip install tqdm numpy
```

### 2. Configure HuggingFace Token
Create `.env` file in project root:
```bash
echo "HF_TOKEN=your_huggingface_token_here" > ../../.env
```

### 3. Verify Llama 3 Access
Ensure you have access to `meta-llama/Meta-Llama-3-8B-Instruct` on HuggingFace.

---

## Running the Pipeline

### Phase 1: Data Preparation ✅ COMPLETED
```bash
cd /home/heck2/myousuf6/ava/ai-scientist/cPAPERS/experiments/review_generation
bash run_phase1.sh
```

**Output**: `../../data/processed/` with train/val/test splits

**Status**: ✅ Successfully tested, 612 train / 131 val / 133 test examples

---

### Phase 2: Baseline Generation (Model A)

#### Test Run (5 examples)
```bash
bash run_phase2.sh
```

#### Full Validation Set
```bash
python3 phase2_baseline_generation.py --split validation
```

#### Full Test Set
```bash
python3 phase2_baseline_generation.py --split test --temperature 0.7 --max_tokens 1024
```

**Output**: `../../data/generated_reviews/baseline/`

**Requirements**: GPU with 24GB+ VRAM for Llama 3 8B

---

### Phase 3: Fine-tuning (Model B)

#### Default Configuration
```bash
bash run_phase3.sh
```

#### Custom Hyperparameters
```bash
python3 phase3_finetune_model.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ../../models/finetuned/llama3-8b-reviewer \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5
```

**Output**: `../../models/finetuned/llama3-8b-reviewer/`

**Requirements**: GPU with 40GB VRAM (or adjust batch_size for smaller GPUs)

**Training Time**: ~3-4 hours on A100

---

### Generate Reviews with Fine-tuned Model

```bash
python3 phase2_baseline_generation.py \
    --model ../../models/finetuned/llama3-8b-reviewer \
    --split test \
    --temperature 0.7
```

**Output**: `../../data/generated_reviews/finetuned/`

---

## Monitoring Training

### TensorBoard (Phase 3)
```bash
tensorboard --logdir ../../models/finetuned/llama3-8b-reviewer
```

### Check Logs
Phase 3 outputs detailed logs including:
- Training/validation loss
- Steps per second
- Memory usage
- Checkpoint saves

---

## File Locations

```
data/
├── gpt_cleaned_reviews_top1000.json          # Original data
├── processed/                                 # Phase 1 output
│   ├── train_training.json                   # 612 examples
│   ├── validation_training.json              # 131 examples  
│   └── test_training.json                    # 133 examples
└── generated_reviews/
    ├── baseline/                              # Phase 2 (Model A)
    │   ├── validation_baseline_reviews.json
    │   └── test_baseline_reviews.json
    └── finetuned/                             # Phase 2 (Model B)
        └── test_finetuned_reviews.json

models/
└── finetuned/
    └── llama3-8b-reviewer/                    # Phase 3 output
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── config.json
```

---

## Memory Management

### If Out of Memory in Phase 2:
Use vLLM with smaller batch processing:
```bash
python3 phase2_baseline_generation.py --split validation --batch_size 1
```

### If Out of Memory in Phase 3:
Reduce batch size and increase gradient accumulation:
```bash
python3 phase3_finetune_model.py \
    --batch_size 1 \
    --gradient_accumulation_steps 16
```

Or reduce max sequence length:
```bash
python3 phase3_finetune_model.py \
    --max_length 1024 \
    --batch_size 4
```

---

## Validation

### Check Phase 1 Output
```bash
# Verify splits exist
ls -lh ../../data/processed/

# Check training examples
python3 -c "
import json
with open('../../data/processed/train_training.json') as f:
    data = json.load(f)
print(f'Training examples: {len(data)}')
print(f'Sample paper: {data[0][\"submission_title\"]}')
"
```

### Check Phase 2 Output
```bash
# Count generated reviews
python3 -c "
import json
with open('../../data/generated_reviews/baseline/validation_baseline_reviews.json') as f:
    data = json.load(f)
print(f'Generated reviews: {len(data)}')
print(f'Sample generated review length: {len(data[0][\"generated_review\"])} chars')
"
```

### Check Phase 3 Model
```bash
# List checkpoints
ls -lh ../../models/finetuned/llama3-8b-reviewer/

# Check model config
python3 -c "
import json
with open('../../models/finetuned/llama3-8b-reviewer/config.json') as f:
    config = json.load(f)
print(f'Model type: {config[\"model_type\"]}')
print(f'Hidden size: {config[\"hidden_size\"]}')
"
```

---

## TODO Before Production

### Critical
1. **Implement Paper Loading**: Complete `paper_loader.py` to extract content from LaTeX files in `extracted_files/`
   - Test: `python3 paper_loader.py`
   - Verify: Sections are extracted correctly

### Optional
2. Test on small subset first (done automatically in run_phase2.sh)
3. Monitor GPU memory usage during training
4. Validate generated reviews look reasonable before full runs

---

## Quick Commands Reference

```bash
# Phase 1
bash run_phase1.sh

# Phase 2 (test)
bash run_phase2.sh

# Phase 2 (full)
python3 phase2_baseline_generation.py --split test

# Phase 3
bash run_phase3.sh

# Generate with fine-tuned
python3 phase2_baseline_generation.py \
    --model ../../models/finetuned/llama3-8b-reviewer \
    --split test

# Test paper loader
python3 paper_loader.py

# Check GPU
nvidia-smi
```

---

## Expected Results

### Phase 1
- Train: 612 examples
- Val: 131 examples
- Test: 133 examples
- Total: 876 examples with review points

### Phase 2
- Validation: 131 generated reviews
- Test: 133 generated reviews
- Each review includes generated text + ground truth

### Phase 3
- Model checkpoints saved every 500 steps
- Final model saved at end
- Validation loss should decrease over training
- Best checkpoint kept based on eval_loss

---

## Support

See `README.md` for comprehensive documentation.
See `IMPLEMENTATION_SUMMARY.md` for technical details.

For issues with:
- **Data**: Check Phase 1 output files
- **GPU Memory**: Reduce batch_size or max_length
- **Model Loading**: Verify HF_TOKEN and model access
- **Paper Content**: Complete paper_loader.py TODOs

