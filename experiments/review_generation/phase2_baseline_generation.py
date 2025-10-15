"""
Phase 2: Baseline Review Generation (Model A)
Uses Llama 3 8B Instruct in zero-shot mode to generate reviews
"""

import json
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

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

def create_review_prompt(paper_title, paper_content):
    """
    Create a prompt for generating a paper review
    
    Args:
        paper_title: Title of the paper
        paper_content: Dictionary with paper sections
    
    Returns:
        Formatted prompt string
    """
    # System message explaining the task
    system_msg = """You are an expert peer reviewer for a top-tier machine learning conference (NeurIPS). Your task is to provide a thorough, constructive review of the submitted paper. Your review should cover the following aspects:

1. **Originality**: Is the work novel? How does it relate to prior work?
2. **Quality**: Is the work technically sound? Are the experiments well-designed and comprehensive?
3. **Clarity**: Is the paper well-written and organized? Are the methods clearly explained?
4. **Significance**: What is the impact of this work? Is it important for the field?

Provide your review as a series of bullet points, with each point being a specific observation, strength, weakness, or question about the paper."""

    # Construct paper information
    paper_info = f"""**Paper Title:** {paper_title}

**Abstract:** {paper_content.get('abstract', '[Not available]')}

**Introduction:** {paper_content.get('introduction', '[Not available]')}

**Methodology:** {paper_content.get('methodology', '[Not available]')}

**Results:** {paper_content.get('results', '[Not available]')}

**Conclusion:** {paper_content.get('conclusion', '[Not available]')}"""

    # Create the full prompt in Llama 3 format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

Please review the following paper:

{paper_info}

Provide a detailed review with specific bullet points covering originality, quality, clarity, and significance.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    return prompt

def generate_reviews_batch(examples, model, sampling_params, extracted_files_dir):
    """
    Generate reviews for a batch of papers
    
    Args:
        examples: List of paper examples from training format
        model: vLLM model instance
        sampling_params: Sampling parameters for generation
        extracted_files_dir: Path to extracted files
    
    Returns:
        List of generated reviews
    """
    # Create prompts for all examples
    prompts = []
    for example in tqdm(examples, desc="Creating prompts"):
        paper_id = example['paper_id']
        paper_title = example['input']['title']
        
        # Load paper content
        paper_content = load_paper_content(paper_id, extracted_files_dir)
        
        # Create prompt
        prompt = create_review_prompt(paper_title, paper_content)
        prompts.append(prompt)
    
    print(f"\nGenerating reviews for {len(prompts)} papers...")
    
    # Generate reviews using vLLM
    outputs = model.generate(prompts, sampling_params)
    
    # Extract generated text
    generated_reviews = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_reviews.append(generated_text)
    
    return generated_reviews

def save_generated_reviews(examples, generated_reviews, output_file):
    """
    Save generated reviews alongside input data
    
    Args:
        examples: Original examples
        generated_reviews: Generated review texts
        output_file: Path to output file
    """
    results = []
    for example, generated_review in zip(examples, generated_reviews):
        result = {
            'paper_id': example['paper_id'],
            'submission_id': example['submission_id'],
            'submission_title': example['input']['title'],
            'review_id': example['review_id'],
            'generated_review': generated_review,
            'ground_truth_review': example['output'],
            'ground_truth_points': example['review_points_list']
        }
        results.append(result)
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved generated reviews to: {output_file}")
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline reviews using Llama 3 8B Instruct")
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], 
                       default='validation', help="Which split to generate reviews for")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model to use for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Process only first N examples (for testing)")
    args = parser.parse_args()
    
    # Paths
    data_dir = Path("/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/data")
    processed_dir = data_dir / "processed"
    output_dir = data_dir / "generated_reviews" / "baseline"
    extracted_files_dir = Path("/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/data collection/neurips/2021/extracted_files")
    
    # Load training format data
    input_file = processed_dir / f"{args.split}_training.json"
    print(f"Loading data from: {input_file}")
    
    with open(input_file, 'r') as f:
        examples = json.load(f)
    
    # Limit batch size if specified (for testing)
    if args.batch_size:
        examples = examples[:args.batch_size]
        print(f"Limited to first {args.batch_size} examples for testing")
    
    print(f"Loaded {len(examples)} examples from {args.split} split")
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    print("This may take a few minutes...")
    
    # Set cache directories (same as in run_zs_equation.sh)
    import os
    os.environ['HF_HOME'] = "/home/heck2/myousuf6/.cache/huggingface"
    os.environ['TRANSFORMERS_CACHE'] = "/home/heck2/myousuf6/.cache/huggingface/transformers"
    
    model = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096  # Llama 3 supports up to 8K, but 4K is usually enough
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>"]  # Stop token for Llama 3
    )
    
    print(f"\nSampling parameters:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max tokens: {args.max_tokens}")
    
    # Generate reviews
    generated_reviews = generate_reviews_batch(
        examples, 
        model, 
        sampling_params, 
        extracted_files_dir
    )
    
    # Save results
    output_file = output_dir / f"{args.split}_baseline_reviews.json"
    results = save_generated_reviews(examples, generated_reviews, output_file)
    
    # Print sample output
    print("\n" + "="*60)
    print("SAMPLE GENERATED REVIEW")
    print("="*60)
    sample = results[0]
    print(f"\nPaper: {sample['submission_title']}")
    print(f"\nGenerated Review:\n{sample['generated_review'][:500]}...")
    print(f"\nGround Truth (first 3 points):")
    for i, point in enumerate(sample['ground_truth_points'][:3]):
        print(f"  {i+1}. {point}")
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("="*60)
    print(f"\nGenerated {len(results)} baseline reviews")
    print(f"Saved to: {output_file}")
    print("\nNext step: Run Phase 3 to fine-tune a model on this data")

if __name__ == "__main__":
    main()

