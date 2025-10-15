"""
Utility functions for review generation pipeline
"""

import json
from pathlib import Path

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_jsonl(file_path):
    """Load JSONL file (one JSON per line)"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def parse_review_points(review_text):
    """
    Parse a review text into individual bullet points
    
    Args:
        review_text: Generated review text
    
    Returns:
        List of review points
    """
    lines = review_text.split('\n')
    points = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Check if line starts with bullet point markers
        if line.startswith('-') or line.startswith('*') or line.startswith('•'):
            points.append(line[1:].strip())
        elif line[0].isdigit() and ('. ' in line[:5] or ') ' in line[:5]):
            # Handle numbered lists like "1. " or "1) "
            points.append(line.split('. ', 1)[1] if '. ' in line else line.split(') ', 1)[1])
        else:
            # If not a bullet point but substantial text, include it
            if len(line) > 20:
                points.append(line)
    
    return points

def compute_coverage(generated_points, ground_truth_points, threshold=0.5):
    """
    Compute coverage of ground truth points by generated points
    Uses simple string matching (can be improved with semantic similarity)
    
    Args:
        generated_points: List of generated review points
        ground_truth_points: List of ground truth review points
        threshold: Similarity threshold for matching
    
    Returns:
        Coverage score (fraction of ground truth points covered)
    """
    from difflib import SequenceMatcher
    
    if not ground_truth_points:
        return 0.0
    
    covered = 0
    for gt_point in ground_truth_points:
        # Check if any generated point is similar to this ground truth point
        for gen_point in generated_points:
            similarity = SequenceMatcher(None, gt_point.lower(), gen_point.lower()).ratio()
            if similarity >= threshold:
                covered += 1
                break
    
    return covered / len(ground_truth_points)

def extract_aspects(review_points):
    """
    Classify review points into aspects (originality, quality, clarity, significance)
    Simple keyword-based classification
    
    Args:
        review_points: List of review points
    
    Returns:
        Dictionary mapping aspects to points
    """
    aspects = {
        'originality': [],
        'quality': [],
        'clarity': [],
        'significance': [],
        'other': []
    }
    
    # Keywords for each aspect
    keywords = {
        'originality': ['novel', 'originality', 'prior work', 'existing', 'previous', 'contribution'],
        'quality': ['quality', 'sound', 'technically', 'experiment', 'evaluation', 'result', 'performance'],
        'clarity': ['clarity', 'clear', 'writing', 'presentation', 'organized', 'explained'],
        'significance': ['significance', 'significant', 'impact', 'important', 'contribution', 'practical']
    }
    
    for point in review_points:
        point_lower = point.lower()
        assigned = False
        
        for aspect, aspect_keywords in keywords.items():
            if any(keyword in point_lower for keyword in aspect_keywords):
                aspects[aspect].append(point)
                assigned = True
                break
        
        if not assigned:
            aspects['other'].append(point)
    
    return aspects

def print_review_comparison(paper_title, generated_review, ground_truth_points):
    """
    Print a formatted comparison of generated and ground truth reviews
    
    Args:
        paper_title: Title of the paper
        generated_review: Generated review text
        ground_truth_points: List of ground truth review points
    """
    print("="*80)
    print(f"PAPER: {paper_title}")
    print("="*80)
    
    print("\nGENERATED REVIEW:")
    print("-"*80)
    print(generated_review)
    
    print("\nGROUND TRUTH POINTS:")
    print("-"*80)
    for i, point in enumerate(ground_truth_points, 1):
        print(f"{i}. {point}")
    
    print("="*80 + "\n")

