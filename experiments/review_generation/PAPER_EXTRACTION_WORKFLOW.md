# Paper Extraction Workflow: From Paper ID to Source Code

This document explains how the data collection pipeline goes from a paper ID to extracting the actual LaTeX source code and content.

---

## 📋 Complete Workflow Overview

```
Step 1: OpenReview → Paper Metadata
Step 2: ArXiv Search → Paper ID
Step 3: ArXiv Download → .tar.gz files
Step 4: Extract → LaTeX source files
Step 5: Parse → Extract figures, equations, tables, etc.
```

---

## 🔍 Step-by-Step Process

### **Step 1: Retrieve Submissions from OpenReview**

**File**: `collect_data.py` (lines 9-11, 182-184)

```python
def retrieve_submissions(client, invitation):
    submissions = client.get_all_notes(invitation=invitation, details='replies')
    return submissions

# Example usage:
conference = 'NeurIPS.cc/2021/Conference/-/Blind_Submission'
submissions = retrieve_submissions(guest_client, conference)
```

**What it does**:
- Connects to OpenReview API
- Fetches all submissions for a specific conference (e.g., NeurIPS 2021)
- Gets submission metadata: title, reviews, comments
- Returns list of submission objects with details

**Output**: Submission objects with:
- `submission.id` - OpenReview submission ID
- `submission.content['title']` - Paper title
- `submission.details['replies']` - Reviews and comments

---

### **Step 2: Match Paper Title to ArXiv ID**

**File**: `collect_data.py` (lines 21-36, 56-68)

```python
def get_arxiv_id(client, title):
    """Search ArXiv by paper title to get ArXiv ID"""
    try:
        search = arxiv.Search(
            query = f"ti:{title}",  # Search by title
            max_results = 1)
        
        if next(client.results(search)) is not None:
            url = str(next(client.results(search)))
            id = url.split('/')[-1]  # Extract ID from URL
            return id
        
        return None
    except Exception as e:
        print(f"Could not find paper_id for {title}")
        return None
```

**Process**:
1. Takes paper title from OpenReview submission
2. Searches ArXiv using title query: `ti:{title}`
3. Gets first match result
4. Extracts ArXiv ID from URL (e.g., "1804.08450v1")
5. Returns paper_id

**Example**:
- Title: "Batch Active Learning at Scale"
- ArXiv Search: `ti:Batch Active Learning at Scale`
- Result URL: `http://arxiv.org/abs/1804.08450v1`
- Extracted ID: `1804.08450v1`

---

### **Step 3: Download Paper Source from ArXiv**

**File**: `collect_data.py` (lines 38-40, 74-76)

```python
def download_paper(client, paper_id, path):
    """Download LaTeX source files from ArXiv"""
    paper = next(client.results(arxiv.Search(id_list=[paper_id])))
    paper.download_source(filename=f"{path}tar_files/{paper_id}.tar.gz")
```

**Process**:
1. Uses ArXiv API to fetch paper by ID
2. Downloads source files (not PDF)
3. Saves as compressed archive: `{paper_id}.tar.gz`
4. Stores in `neurips/2021/tar_files/`

**Example**:
- Input: `paper_id = "1804.08450v1"`
- Download: LaTeX source from ArXiv
- Save: `neurips/2021/tar_files/1804.08450v1.tar.gz`

**What's inside the .tar.gz**:
- `.tex` files - LaTeX source
- `.bbl` files - Bibliography
- `.eps`, `.png`, `.pdf` - Figures
- `Makefile` - Build instructions
- Other supporting files

---

### **Step 4: Extract .tar.gz Archives**

**File**: `collect_data.py` (lines 117-137)

```python
def unpack_tar_files(path):
    """Extract all .tar.gz files to individual directories"""
    for file in tqdm(os.listdir(os.path.join(path, 'tar_files'))):
        file_path = os.path.join(path, 'tar_files', file)
        extract_path = os.path.join(path + 'extracted_files', 
                                    os.path.splitext(file)[0])
        os.makedirs(extract_path, exist_ok=True)
        
        if file.endswith('.tar.gz') or file.endswith('.tgz'):
            command = ['tar', '-xzf', file_path, '-C', extract_path]
        elif file.endswith('.tar'):
            command = ['tar', '-xf', file_path, '-C', extract_path]
        
        subprocess.run(command, capture_output=True)
```

**Process**:
1. Iterates through all `.tar.gz` files in `tar_files/`
2. Creates directory for each paper: `extracted_files/{paper_id}.tar/`
3. Extracts archive contents using `tar` command
4. Each paper gets its own directory with source files

**Directory Structure**:
```
neurips/2021/
├── tar_files/
│   ├── 1804.08450v1.tar.gz          [Downloaded archive]
│   ├── 2107.12685v1.tar.gz
│   └── ...
└── extracted_files/
    ├── 1804.08450v1.tar/            [Extracted directory]
    │   ├── main.tex                 [Main LaTeX file]
    │   ├── appendix.tex             [Appendix]
    │   ├── references.bbl           [Bibliography]
    │   ├── figure1.eps              [Figures]
    │   └── ...
    ├── 2107.12685v1.tar/
    │   ├── paper.tex
    │   └── ...
    └── ...
```

---

### **Step 5: Parse LaTeX Files to Extract Content**

Once extracted, various scripts parse the LaTeX source files:

#### **A. Extract Equations**

**File**: `get_equations.py`

```python
def process_tex_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        tex_content = file.read()
    
    equation_data = []
    
    # Find all equation environments
    for match in re.finditer(r'\\begin{equation}(.*?)\\end{equation}', 
                             tex_content, re.DOTALL):
        equation = match.group(1)
        
        # Extract label
        label_match = re.search(r'\\label{([^}]+)}', equation)
        equation_label = label_match.group(1) if label_match else None
        
        # Find context (paragraphs before and after)
        # Find references (where equation is cited)
        
        equation_data.append({
            "equation": equation.strip(),
            "label": equation_label,
            "references": references,
            "context": context
        })
```

**What it extracts**:
- Equation LaTeX code
- Equation label (for referencing)
- Context paragraphs (before and after)
- References to equation in text

#### **B. Extract Figures**

**File**: `get_figures.py`

```python
def process_figures(file_path, folder_path, folder_name):
    with open(file_path, 'r', encoding='latin-1') as file:
        tex_content = file.read()
    
    figures_data = []
    
    # Find all figure environments
    figures = re.finditer(r'\\begin{figure}(.*?)\\end{figure}', 
                         tex_content, re.DOTALL)
    
    for figure_match in figures:
        figure_content = figure_match.group(1)
        
        # Extract caption
        caption_match = re.search(r'\\caption{(.*?)}', 
                                 figure_content, re.DOTALL)
        caption = caption_match.group(1) if caption_match else "No caption"
        
        # Extract label
        label_match = re.search(r'\\label{(.*?)}', figure_content)
        figure_label = label_match.group(1) if label_match else None
        
        # Extract figure file path
        includegraphics_match = re.search(
            r'\\includegraphics(?:\[.*?\])?{(.*?)}', 
            figure_content)
        figure_path = includegraphics_match.group(1)
        
        # Copy figure file to centralized location
        # Rename as {paper_id}.{original_filename}
        
        figures_data.append({
            "caption": caption.strip(),
            "label": figure_label,
            "figure_path": new_figure_name,
            "references": references,
            "context": context
        })
```

**What it extracts**:
- Figure caption
- Figure label
- Figure file path
- Context paragraphs
- References to figure in text
- Copies actual figure files to `figure_files/`

#### **C. Extract Tables**

**File**: `get_tables.py` (similar to figures)

Extracts:
- Table caption
- Table label
- Table content (LaTeX tabular environment)
- Context and references

---

## 📊 Data Flow Diagram

```
┌─────────────────────┐
│   OpenReview API    │
│  (NeurIPS 2021)     │
└──────────┬──────────┘
           │ Submission metadata
           │ (title, reviews, comments)
           v
┌─────────────────────┐
│   ArXiv Search      │
│  Search by title    │
└──────────┬──────────┘
           │ Paper ID (e.g., "1804.08450v1")
           v
┌─────────────────────┐
│  ArXiv Download     │
│  download_source()  │
└──────────┬──────────┘
           │ .tar.gz archive
           v
┌─────────────────────┐
│   tar_files/        │
│  1804.08450v1.tar.gz│
└──────────┬──────────┘
           │ Extract with tar -xzf
           v
┌─────────────────────┐
│ extracted_files/    │
│ 1804.08450v1.tar/   │
│   ├── main.tex      │
│   ├── figure1.eps   │
│   └── ...           │
└──────────┬──────────┘
           │ Parse LaTeX files
           v
┌─────────────────────────────────────┐
│     Extracted Content               │
│  ┌───────────┬──────────┬─────────┐ │
│  │ Equations │ Figures  │ Tables  │ │
│  │  + label  │ + caption│ + label │ │
│  │  + context│ + context│ +context│ │
│  └───────────┴──────────┴─────────┘ │
└─────────────────────────────────────┘
```

---

## 🗂️ File Organization

```
neurips/2021/
├── raw_qas.json                         # Step 1-2 output
│   └── [{ submission_id, paper_id, review, title, ... }]
│
├── tar_files/                           # Step 3 output
│   ├── 1804.08450v1.tar.gz
│   ├── 2107.12685v1.tar.gz
│   └── ...
│
├── extracted_files/                     # Step 4 output
│   ├── 1804.08450v1.tar/
│   │   ├── main.tex
│   │   ├── appendix.tex
│   │   ├── figure1.eps
│   │   └── ...
│   ├── 2107.12685v1.tar/
│   │   └── ...
│   └── ...
│
├── figure_files/                        # Centralized figures
│   ├── 1804.08450v1.figure1.eps
│   ├── 1804.08450v1.figure2.png
│   └── ...
│
├── equation_data.json                   # Step 5A output
│   └── [{ folder, filename, equation, label, context, references }]
│
├── figures_data.json                    # Step 5B output
│   └── [{ folder, filename, caption, label, figure_path, context, references }]
│
└── table_data.json                      # Step 5C output
    └── [{ folder, filename, caption, label, content, context, references }]
```

---

## 🎯 Key Insights for Paper Loading

### **For Your Review Generation Pipeline**

Based on this workflow, here's what you need to implement in `paper_loader.py`:

#### **1. Directory Structure**
```python
extracted_files_dir = "neurips/2021/extracted_files/"
paper_dir = f"{extracted_files_dir}/{paper_id}.tar/"
# Note: directories have .tar extension after paper_id
```

#### **2. Finding Main .tex File**
```python
# Common patterns for main files:
common_names = ['main.tex', 'paper.tex', 'manuscript.tex', 'arxiv.tex']

# Or search for file with \documentclass
for tex_file in glob.glob(f"{paper_dir}/*.tex"):
    with open(tex_file) as f:
        if '\\documentclass' in f.read():
            main_file = tex_file
            break
```

#### **3. Extracting Sections**
Use regex patterns similar to `get_equations.py` and `get_figures.py`:

```python
# Abstract
abstract_pattern = r'\\begin{abstract}(.*?)\\end{abstract}'

# Sections
section_pattern = r'\\section\*?{Introduction}(.*?)(?:\\section|\Z)'

# Clean LaTeX commands
text = re.sub(r'\\cite{.*?}', '[citation]', text)
text = re.sub(r'\$.*?\$', '[equation]', text)
```

#### **4. Handling Multiple .tex Files**
Some papers split content across files:
```
main.tex          # Main document
intro.tex         # \input{intro}
methods.tex       # \input{methods}
appendix.tex      # \input{appendix}
```

You may need to:
- Parse `\input{}` and `\include{}` commands
- Recursively load referenced files
- Combine content in proper order

---

## 💡 Implementation Strategy

### **Option 1: Simple (Use Main File Only)**
```python
def load_paper_content(paper_id, extracted_files_dir):
    paper_dir = f"{extracted_files_dir}/{paper_id}.tar/"
    main_tex = find_main_tex_file(paper_dir)
    
    with open(main_tex, 'r', encoding='latin-1') as f:
        content = f.read()
    
    return {
        'abstract': extract_abstract(content),
        'introduction': extract_section(content, 'Introduction'),
        'methodology': extract_section(content, 'Method'),
        'results': extract_section(content, 'Results'),
        'conclusion': extract_section(content, 'Conclusion')
    }
```

### **Option 2: Advanced (Handle Includes)**
```python
def load_paper_content(paper_id, extracted_files_dir):
    paper_dir = f"{extracted_files_dir}/{paper_id}.tar/"
    main_tex = find_main_tex_file(paper_dir)
    
    # Recursively expand \input and \include commands
    full_content = expand_includes(main_tex, paper_dir)
    
    # Extract sections from combined content
    return extract_all_sections(full_content)

def expand_includes(tex_file, base_dir):
    with open(tex_file, 'r', encoding='latin-1') as f:
        content = f.read()
    
    # Find all \input{file} commands
    for match in re.finditer(r'\\input{(.*?)}', content):
        input_file = match.group(1)
        if not input_file.endswith('.tex'):
            input_file += '.tex'
        
        input_path = os.path.join(base_dir, input_file)
        if os.path.exists(input_path):
            # Recursively expand includes
            input_content = expand_includes(input_path, base_dir)
            content = content.replace(match.group(0), input_content)
    
    return content
```

---

## 🔧 Testing Your Implementation

### **Test on Sample Paper**
```python
# Test with first paper from dataset
paper_id = "0704.3259v1"  # Or any paper_id from your data

extracted_files_dir = "neurips/2021/extracted_files/"
paper_dir = f"{extracted_files_dir}/{paper_id}.tar/"

# List files in directory
import os
print(os.listdir(paper_dir))

# Try to load and parse
content = load_paper_content(paper_id, extracted_files_dir)
print("Abstract:", content['abstract'][:200])
print("Introduction:", content['introduction'][:200])
```

### **Validation Checklist**
- [ ] Correctly identifies paper directory
- [ ] Finds main .tex file
- [ ] Extracts abstract successfully
- [ ] Extracts introduction successfully
- [ ] Extracts methodology section
- [ ] Extracts results section
- [ ] Extracts conclusion
- [ ] Handles LaTeX commands (citations, equations)
- [ ] Handles special characters
- [ ] Returns clean, readable text

---

## 📝 Summary

**The complete pipeline**:

1. **OpenReview** → Get submission metadata (title, reviews)
2. **ArXiv Search** → Match title to paper ID
3. **ArXiv API** → Download `.tar.gz` source files
4. **Extract** → Unpack to `extracted_files/{paper_id}.tar/`
5. **Parse** → Extract LaTeX content (equations, figures, tables)

**For your implementation**:
- Paper sources are in `extracted_files/{paper_id}.tar/`
- Each directory contains `.tex` files with LaTeX source
- Use regex patterns to extract sections
- Clean LaTeX markup for readable text
- Handle edge cases (missing sections, multiple files, encoding)

This gives you everything you need to complete the `paper_loader.py` implementation! 🎉


