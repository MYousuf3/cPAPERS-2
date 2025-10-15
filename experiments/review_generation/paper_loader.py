"""
Paper Content Loader
Utilities to load and parse paper content from extracted LaTeX files

TODO: Complete implementation to parse LaTeX source files
"""

import os
from pathlib import Path
import re

def find_paper_directory(paper_id, extracted_files_dir):
    """
    Find the directory containing extracted files for a paper
    
    Args:
        paper_id: ArXiv paper ID (e.g., "1804.08450v1")
        extracted_files_dir: Base directory with extracted files
    
    Returns:
        Path to paper directory, or None if not found
    """
    # Based on the extraction workflow, directories are named {paper_id}.tar
    # This is because the tar files are extracted with their names preserved
    
    # Try with .tar extension (standard extraction format)
    paper_dir = Path(extracted_files_dir) / f"{paper_id}.tar"
    if paper_dir.exists():
        return paper_dir
    
    # Try without .tar extension (alternative format)
    paper_dir = Path(extracted_files_dir) / paper_id
    if paper_dir.exists():
        return paper_dir
    
    # Try without version suffix (e.g., "1804.08450" instead of "1804.08450v1")
    if 'v' in paper_id:
        paper_id_no_version = paper_id.rsplit('v', 1)[0]
        
        # Try with .tar
        paper_dir_no_version = Path(extracted_files_dir) / f"{paper_id_no_version}.tar"
        if paper_dir_no_version.exists():
            return paper_dir_no_version
        
        # Try without .tar
        paper_dir_no_version = Path(extracted_files_dir) / paper_id_no_version
        if paper_dir_no_version.exists():
            return paper_dir_no_version
    
    return None

def find_main_tex_file(paper_dir):
    """
    Find the main .tex file in the paper directory
    
    Args:
        paper_dir: Directory containing paper files
    
    Returns:
        Path to main .tex file, or None if not found
    """
    paper_dir = Path(paper_dir)
    
    # Strategy 1: Look for common main file names
    common_names = ['main.tex', 'paper.tex', 'manuscript.tex', 'arxiv.tex', 
                   'neurips_2021.tex', 'iclr2021_conference.tex']
    for name in common_names:
        main_file = paper_dir / name
        if main_file.exists():
            return main_file
    
    # Strategy 2: Look for any .tex file with \documentclass
    # This indicates the main document (as opposed to included files)
    tex_files_with_documentclass = []
    for tex_file in paper_dir.glob('*.tex'):
        try:
            # Use latin-1 encoding as in get_equations.py and get_figures.py
            with open(tex_file, 'r', encoding='latin-1', errors='ignore') as f:
                content = f.read()
                if '\\documentclass' in content:
                    tex_files_with_documentclass.append(tex_file)
        except Exception:
            continue
    
    # If we found files with documentclass, prefer the one with \begin{document}
    if tex_files_with_documentclass:
        for tex_file in tex_files_with_documentclass:
            try:
                with open(tex_file, 'r', encoding='latin-1', errors='ignore') as f:
                    if '\\begin{document}' in f.read():
                        return tex_file
            except Exception:
                continue
        # If none have \begin{document}, just return the first one
        return tex_files_with_documentclass[0]
    
    # Strategy 3: If still not found, just return the first .tex file
    tex_files = list(paper_dir.glob('*.tex'))
    if tex_files:
        return tex_files[0]
    
    return None

def extract_section(tex_content, section_name):
    """
    Extract a specific section from LaTeX content
    
    Args:
        tex_content: Full LaTeX content
        section_name: Name of section to extract (e.g., 'abstract', 'introduction')
    
    Returns:
        Extracted section text, or empty string if not found
    """
    section_patterns = {
        'abstract': [
            r'\\begin{abstract}(.*?)\\end{abstract}',
            r'\\abstract{(.*?)}',
        ],
        'introduction': [
            r'\\section\*?\{Introduction\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{INTRODUCTION\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
        ],
        'methodology': [
            r'\\section\*?\{Methodology\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Method\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Methods\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Approach\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
        ],
        'results': [
            r'\\section\*?\{Results\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Experiments\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Experimental Results\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
        ],
        'conclusion': [
            r'\\section\*?\{Conclusion\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Conclusions\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
            r'\\section\*?\{Discussion\}(.*?)(?:\\section|\\appendix|\\bibliography|\Z)',
        ]
    }
    
    patterns = section_patterns.get(section_name.lower(), [])
    
    for pattern in patterns:
        match = re.search(pattern, tex_content, re.DOTALL | re.IGNORECASE)
        if match:
            section_text = match.group(1)
            # Clean up LaTeX commands
            section_text = clean_latex_text(section_text)
            return section_text.strip()
    
    return ""

def clean_latex_text(text):
    """
    Clean LaTeX markup from text to make it readable
    
    Args:
        text: Text with LaTeX markup
    
    Returns:
        Cleaned, human-readable text
    """
    if not text:
        return ""
    
    # Remove LaTeX comments (lines starting with %)
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    
    # Remove figure environments (they distract from main text)
    text = re.sub(r'\\begin{figure}.*?\\end{figure}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin{figure\*}.*?\\end{figure\*}', '', text, flags=re.DOTALL)
    
    # Remove table environments
    text = re.sub(r'\\begin{table}.*?\\end{table}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin{table\*}.*?\\end{table\*}', '', text, flags=re.DOTALL)
    
    # Remove algorithm environments
    text = re.sub(r'\\begin{algorithm}.*?\\end{algorithm}', '', text, flags=re.DOTALL)
    
    # Preserve content from formatting commands
    text = re.sub(r'\\textbf{(.*?)}', r'\1', text)
    text = re.sub(r'\\textit{(.*?)}', r'\1', text)
    text = re.sub(r'\\emph{(.*?)}', r'\1', text)
    text = re.sub(r'\\text{(.*?)}', r'\1', text)
    
    # Replace citations with placeholder
    text = re.sub(r'\\cite\w*{[^}]+}', '[citation]', text)
    text = re.sub(r'\\citep{[^}]+}', '[citation]', text)
    text = re.sub(r'\\citet{[^}]+}', '[citation]', text)
    
    # Replace references with placeholder
    text = re.sub(r'\\ref{[^}]+}', '[ref]', text)
    text = re.sub(r'\\eqref{[^}]+}', '[eqref]', text)
    
    # Replace inline math with placeholder
    text = re.sub(r'\$([^\$]+)\$', r'[\1]', text)
    
    # Replace display math environments with placeholder
    text = re.sub(r'\$\$.*?\$\$', '[equation]', text, flags=re.DOTALL)
    text = re.sub(r'\\begin{equation}.*?\\end{equation}', '[equation]', text, flags=re.DOTALL)
    text = re.sub(r'\\begin{equation\*}.*?\\end{equation\*}', '[equation]', text, flags=re.DOTALL)
    text = re.sub(r'\\begin{align}.*?\\end{align}', '[equation]', text, flags=re.DOTALL)
    text = re.sub(r'\\begin{align\*}.*?\\end{align\*}', '[equation]', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '[equation]', text, flags=re.DOTALL)
    
    # Remove subsection commands but keep their titles
    text = re.sub(r'\\subsection\*?{(.*?)}', r'\n\1: ', text)
    text = re.sub(r'\\subsubsection\*?{(.*?)}', r'\n\1: ', text)
    text = re.sub(r'\\paragraph{(.*?)}', r'\n\1: ', text)
    
    # Remove common LaTeX commands (keep trying to preserve content)
    text = re.sub(r'\\label{[^}]+}', '', text)
    text = re.sub(r'\\caption{(.*?)}', '', text)  # Captions already removed with figures
    
    # Remove itemize/enumerate environments but keep content
    text = re.sub(r'\\begin{itemize}', '', text)
    text = re.sub(r'\\end{itemize}', '', text)
    text = re.sub(r'\\begin{enumerate}', '', text)
    text = re.sub(r'\\end{enumerate}', '', text)
    text = re.sub(r'\\item\s*', '- ', text)
    
    # Remove other common environments
    text = re.sub(r'\\begin{[^}]+}', '', text)
    text = re.sub(r'\\end{[^}]+}', '', text)
    
    # Remove remaining LaTeX commands (backslash followed by letters)
    text = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^}]*)\}', r'\2', text)  # Keep content in braces
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)  # Remove command itself
    
    # Clean up special characters
    text = re.sub(r'\\&', '&', text)
    text = re.sub(r'\\_', '_', text)
    text = re.sub(r'\\%', '%', text)
    text = re.sub(r'\\#', '#', text)
    text = re.sub(r'\\\$', '$', text)
    text = re.sub(r'~', ' ', text)
    
    # Remove excess braces
    text = re.sub(r'[{}]', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n ', '\n', text)  # Remove space at start of line
    text = re.sub(r' \n', '\n', text)  # Remove space at end of line
    
    return text.strip()

def load_paper_content(paper_id, extracted_files_dir):
    """
    Load and extract paper content
    
    Args:
        paper_id: ArXiv paper ID (e.g., "1804.08450v1")
        extracted_files_dir: Base directory with extracted files
    
    Returns:
        Dictionary with paper sections
    """
    # Find paper directory
    paper_dir = find_paper_directory(paper_id, extracted_files_dir)
    if not paper_dir:
        # Return placeholders if paper not found
        return {
            'abstract': f'[Paper {paper_id} not found in extracted_files]',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': ''
        }
    
    # Find main tex file
    main_tex = find_main_tex_file(paper_dir)
    if not main_tex:
        return {
            'abstract': f'[Main tex file not found for paper {paper_id}]',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': ''
        }
    
    # Read tex content using latin-1 encoding (same as in get_equations.py and get_figures.py)
    try:
        with open(main_tex, 'r', encoding='latin-1', errors='ignore') as f:
            tex_content = f.read()
    except Exception as e:
        return {
            'abstract': f'[Error reading paper {paper_id}: {e}]',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': ''
        }
    
    # Extract sections
    return {
        'abstract': extract_section(tex_content, 'abstract'),
        'introduction': extract_section(tex_content, 'introduction'),
        'methodology': extract_section(tex_content, 'methodology'),
        'results': extract_section(tex_content, 'results'),
        'conclusion': extract_section(tex_content, 'conclusion')
    }

def test_paper_loader():
    """Test the paper loader with a sample paper"""
    extracted_files_dir = Path("/home/heck2/myousuf6/ava/ai-scientist/cPAPERS/data collection/neurips/2021/extracted_files")
    
    # Test with a paper that actually exists in the dataset
    # First, let's check what papers are available
    print("Available papers (first 10):")
    print("="*60)
    available_papers = sorted(os.listdir(extracted_files_dir))[:10]
    for i, paper_dir in enumerate(available_papers, 1):
        # Remove .tar extension if present
        paper_id = paper_dir.replace('.tar', '')
        print(f"{i}. {paper_id}")
    
    # Use the first available paper
    if available_papers:
        sample_paper_id = available_papers[0].replace('.tar', '')
        
        print(f"\n{'='*60}")
        print(f"Testing paper loader with paper: {sample_paper_id}")
        print("="*60)
        
        # Test directory finding
        paper_dir = find_paper_directory(sample_paper_id, extracted_files_dir)
        if paper_dir:
            print(f"✓ Found paper directory: {paper_dir}")
            
            # Test finding main tex file
            main_tex = find_main_tex_file(paper_dir)
            if main_tex:
                print(f"✓ Found main tex file: {main_tex.name}")
            else:
                print("✗ Could not find main tex file")
        else:
            print(f"✗ Could not find directory for paper {sample_paper_id}")
        
        print("\n" + "="*60)
        print("Loading paper content...")
        print("="*60)
        
        content = load_paper_content(sample_paper_id, extracted_files_dir)
        
        for section, text in content.items():
            print(f"\n{section.upper()}:")
            print("-"*60)
            if text and not text.startswith('['):
                # Show first 300 chars if content is good
                preview = text[:300] + "..." if len(text) > 300 else text
                print(preview)
                print(f"\n[Total length: {len(text)} characters]")
            elif text:
                # Show error/placeholder messages
                print(text)
            else:
                print("[Empty - section not found]")
        
        print("\n" + "="*60)
        print("Test complete!")
    else:
        print("\n✗ No papers found in extracted_files directory")
        print(f"Directory checked: {extracted_files_dir}")

if __name__ == "__main__":
    test_paper_loader()

