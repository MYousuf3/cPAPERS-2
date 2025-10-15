"""
Phase 1: Data Preparation & Analysis
Splits the review dataset into train/validation/test sets
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append('../')

def load_reviews(file_path):
    """Load reviews from JSON file"""
    print(f"Loading reviews from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} review entries")
    return data

def analyze_dataset(data):
    """Analyze the dataset structure and statistics"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Count unique papers
    unique_papers = set()
    unique_submissions = set()
    review_counts = defaultdict(int)
    
    for entry in data:
        paper_id = entry.get('paper_id')
        submission_id = entry.get('submission_id')
        unique_papers.add(paper_id)
        unique_submissions.add(submission_id)
        review_counts[paper_id] += 1
    
    print(f"\nTotal review entries: {len(data)}")
    print(f"Unique papers: {len(unique_papers)}")
    print(f"Unique submissions: {len(unique_submissions)}")
    print(f"Average reviews per paper: {len(data) / len(unique_papers):.2f}")
    
    # Review point statistics
    total_points = 0
    entries_with_points = 0
    for entry in data:
        if entry.get('cleaned_review_points'):
            points = entry['cleaned_review_points']
            if isinstance(points, list) and len(points) > 0:
                total_points += len(points)
                entries_with_points += 1
    
    if entries_with_points > 0:
        print(f"\nEntries with review points: {entries_with_points}")
        print(f"Average review points per entry: {total_points / entries_with_points:.2f}")
    
    # Sample entry structure
    print("\n" + "-"*60)
    print("SAMPLE ENTRY STRUCTURE:")
    print("-"*60)
    sample = data[0]
    for key in sample.keys():
        value = sample[key]
        if isinstance(value, str):
            preview = value[:100] + "..." if len(value) > 100 else value
        elif isinstance(value, list):
            preview = f"List with {len(value)} items"
        else:
            preview = str(value)
        print(f"  {key}: {preview}")
    
    return unique_papers, review_counts

def group_by_paper(data):
    """Group reviews by paper_id"""
    paper_reviews = defaultdict(list)
    
    for entry in data:
        paper_id = entry.get('paper_id')
        if paper_id:
            paper_reviews[paper_id].append(entry)
    
    return paper_reviews

def create_splits(paper_reviews, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split papers into train/validation/test sets
    
    Args:
        paper_reviews: Dictionary mapping paper_id to list of review entries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    
    # Get all paper IDs
    paper_ids = list(paper_reviews.keys())
    random.shuffle(paper_ids)
    
    # Calculate split sizes
    n_papers = len(paper_ids)
    n_train = int(n_papers * train_ratio)
    n_val = int(n_papers * val_ratio)
    
    # Split paper IDs
    train_papers = paper_ids[:n_train]
    val_papers = paper_ids[n_train:n_train + n_val]
    test_papers = paper_ids[n_train + n_val:]
    
    print("\n" + "="*60)
    print("DATASET SPLITS")
    print("="*60)
    print(f"Train papers: {len(train_papers)}")
    print(f"Validation papers: {len(val_papers)}")
    print(f"Test papers: {len(test_papers)}")
    
    # Create splits with all reviews for each paper
    splits = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for paper_id in train_papers:
        splits['train'].extend(paper_reviews[paper_id])
    
    for paper_id in val_papers:
        splits['validation'].extend(paper_reviews[paper_id])
    
    for paper_id in test_papers:
        splits['test'].extend(paper_reviews[paper_id])
    
    print(f"\nTrain review entries: {len(splits['train'])}")
    print(f"Validation review entries: {len(splits['validation'])}")
    print(f"Test review entries: {len(splits['test'])}")
    
    return splits, {'train': train_papers, 'val': val_papers, 'test': test_papers}

def save_splits(splits, output_dir):
    """Save splits to JSON files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING SPLITS")
    print("="*60)
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name}: {output_file} ({len(split_data)} entries)")

def save_paper_ids(paper_id_splits, output_dir):
    """Save paper IDs for each split"""
    output_dir = Path(output_dir)
    output_file = output_dir / "paper_id_splits.json"
    
    with open(output_file, 'w') as f:
        json.dump(paper_id_splits, f, indent=2)
    print(f"Saved paper ID splits: {output_file}")

def create_training_format(splits, output_dir):
    """
    Create training-ready format for fine-tuning
    Format: List of examples with 'input' (paper info) and 'output' (review points)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING TRAINING FORMAT")
    print("="*60)
    
    for split_name, split_data in splits.items():
        training_examples = []
        skipped = 0
        
        for entry in split_data:
            # Skip entries without cleaned review points
            review_points = entry.get('cleaned_review_points')
            if not review_points or not isinstance(review_points, list) or len(review_points) == 0:
                skipped += 1
                continue
            
            # Combine review points into a single text
            review_text = "\n".join([f"- {point}" for point in review_points])
            
            # Create training example
            # TODO: Add actual paper content from extracted_files
            example = {
                'paper_id': entry.get('paper_id'),
                'submission_id': entry.get('submission_id'),
                'submission_title': entry.get('submission_title'),
                'review_id': entry.get('review_id'),
                # Input will include paper content
                'input': {
                    'title': entry.get('submission_title'),
                    # TODO: Load and add paper abstract, introduction, methodology, etc.
                    # from neurips/2021/extracted_files/{paper_id}/
                    'paper_content_placeholder': 'TODO: Load paper content from extracted_files'
                },
                # Output is the review points
                'output': review_text,
                'review_points_list': review_points  # Keep as list for reference
            }
            
            training_examples.append(example)
        
        # Save training format
        output_file = output_dir / f"{split_name}_training.json"
        with open(output_file, 'w') as f:
            json.dump(training_examples, f, indent=2)
        
        print(f"\n{split_name.upper()}:")
        print(f"  Training examples: {len(training_examples)}")
        print(f"  Skipped (no review points): {skipped}")
        print(f"  Saved to: {output_file}")

def main():
    # Paths
    data_dir = Path("/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/data")
    input_file = data_dir / "gpt_cleaned_reviews_top1000.json"
    output_dir = data_dir / "processed"
    
    # Load data
    data = load_reviews(input_file)
    
    # Analyze dataset
    unique_papers, review_counts = analyze_dataset(data)
    
    # Group by paper
    paper_reviews = group_by_paper(data)
    
    # Create splits
    splits, paper_id_splits = create_splits(paper_reviews, seed=42)
    
    # Save splits
    save_splits(splits, output_dir)
    save_paper_ids(paper_id_splits, output_dir)
    
    # Create training format
    create_training_format(splits, output_dir)
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE!")
    print("="*60)
    print(f"\nProcessed data saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Implement paper loading from extracted_files/")
    print("  2. Run Phase 2: Baseline review generation")
    print("  3. Run Phase 3: Fine-tune review model")

if __name__ == "__main__":
    main()

