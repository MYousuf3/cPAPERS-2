"""
Phase 3: Fine-tune Llama 3 8B Instruct for Review Generation (Model B)
Trains the model on paper -> review pairs
"""

import json
import sys
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

sys.path.append('../')

def load_paper_content(paper_id, extracted_files_dir):
    """
    Load paper content from extracted files
    
    TODO: Implement actual paper loading from extracted_files/{paper_id}/
    For now, returns placeholder text
    
    Args:
        paper_id: ArXiv paper ID (e.g., "1804.08450v1")
        extracted_files_dir: Path to extracted_files directory
    
    Returns:
        Dictionary with paper sections
    """
    # TODO: Implement actual paper extraction
    # - Parse LaTeX source files
    # - Extract abstract, introduction, methodology, results, conclusion
    # - Handle different paper formats
    
    return {
        'abstract': f'[TODO: Extract abstract for paper {paper_id}]',
        'introduction': f'[TODO: Extract introduction for paper {paper_id}]',
        'methodology': f'[TODO: Extract methodology for paper {paper_id}]',
        'results': f'[TODO: Extract results for paper {paper_id}]',
        'conclusion': f'[TODO: Extract conclusion for paper {paper_id}]'
    }

def format_training_example(example, extracted_files_dir):
    """
    Format a training example into instruction-following format
    
    Args:
        example: Training example with paper info and review
        extracted_files_dir: Path to extracted files
    
    Returns:
        Formatted text for training
    """
    paper_id = example['paper_id']
    paper_title = example['input']['title']
    ground_truth_review = example['output']
    
    # Load paper content
    paper_content = load_paper_content(paper_id, extracted_files_dir)
    
    # System message
    system_msg = """You are an expert peer reviewer for a top-tier machine learning conference (NeurIPS). Your task is to provide a thorough, constructive review of the submitted paper. Your review should cover the following aspects:

1. **Originality**: Is the work novel? How does it relate to prior work?
2. **Quality**: Is the work technically sound? Are the experiments well-designed and comprehensive?
3. **Clarity**: Is the paper well-written and organized? Are the methods clearly explained?
4. **Significance**: What is the impact of this work? Is it important for the field?

Provide your review as a series of bullet points, with each point being a specific observation, strength, weakness, or question about the paper."""

    # Paper information
    paper_info = f"""**Paper Title:** {paper_title}

**Abstract:** {paper_content.get('abstract', '[Not available]')}

**Introduction:** {paper_content.get('introduction', '[Not available]')}

**Methodology:** {paper_content.get('methodology', '[Not available]')}

**Results:** {paper_content.get('results', '[Not available]')}

**Conclusion:** {paper_content.get('conclusion', '[Not available]')}"""

    # Format as Llama 3 chat format
    formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

Please review the following paper:

{paper_info}

Provide a detailed review with specific bullet points covering originality, quality, clarity, and significance.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ground_truth_review}<|eot_id|>"""
    
    return formatted_text

def prepare_dataset(split_name, processed_dir, extracted_files_dir):
    """
    Prepare dataset for training
    
    Args:
        split_name: 'train' or 'validation'
        processed_dir: Directory with processed data
        extracted_files_dir: Path to extracted files
    
    Returns:
        List of formatted training examples
    """
    input_file = processed_dir / f"{split_name}_training.json"
    print(f"Loading {split_name} data from: {input_file}")
    
    with open(input_file, 'r') as f:
        examples = json.load(f)
    
    print(f"Formatting {len(examples)} {split_name} examples...")
    
    formatted_examples = []
    for example in examples:
        formatted_text = format_training_example(example, extracted_files_dir)
        formatted_examples.append({'text': formatted_text})
    
    return formatted_examples

def tokenize_function(examples, tokenizer, max_length=2048):
    """
    Tokenize examples for training
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Tokenized examples
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll pad dynamically in the collator
        return_tensors=None
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 8B Instruct for review generation")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/models/finetuned/llama3-8b-reviewer",
                       help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    args = parser.parse_args()
    
    # Set cache directories
    os.environ['HF_HOME'] = "/home/heck2/myousuf6/.cache/huggingface"
    os.environ['TRANSFORMERS_CACHE'] = "/home/heck2/myousuf6/.cache/huggingface/transformers"
    
    # Paths
    data_dir = Path("/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/data")
    processed_dir = data_dir / "processed"
    extracted_files_dir = Path("/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/data collection/neurips/2021/extracted_files")
    
    print("="*60)
    print("PHASE 3: FINE-TUNING LLAMA 3 8B INSTRUCT")
    print("="*60)
    
    # Prepare datasets
    print("\nPreparing training dataset...")
    train_examples = prepare_dataset('train', processed_dir, extracted_files_dir)
    
    print("\nPreparing validation dataset...")
    val_examples = prepare_dataset('validation', processed_dir, extracted_files_dir)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set padding token (Llama models don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing"
    )
    
    # Load model
    print(f"\nLoading model: {args.model}")
    print("This may take a few minutes...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto",  # Automatically distribute across available GPUs
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Prepare model for training
    model.train()
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,  # Use bfloat16 training
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        save_total_limit=3,  # Keep only 3 best checkpoints
    )
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training examples: {len(tokenized_datasets['train'])}")
    print(f"Validation examples: {len(tokenized_datasets['validation'])}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Warmup steps: {args.warmup_steps}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
    )
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Train the model
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE!")
    print("="*60)
    print(f"\nFine-tuned model saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Generate reviews using the fine-tuned model")
    print("  2. Compare with baseline reviews using AlpacaFarm")
    print("\nTo use the fine-tuned model for generation, run:")
    print(f"  python phase2_baseline_generation.py --model {args.output_dir} --split test")

if __name__ == "__main__":
    main()

