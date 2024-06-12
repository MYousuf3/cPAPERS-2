import os
import json
import re
from tqdm import tqdm

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def process_tex_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        tex_content = file.read()

    # Split the document into paragraphs to help find context around equations
    paragraphs = re.split(r'\n\s*\n', tex_content)
    
    equation_data = []

    # Use re.finditer to get the match object for more detailed information (like start and end positions)
    for match in re.finditer(r'\\begin{equation}(.*?)\\end{equation}', tex_content, re.DOTALL):
        equation = match.group(1)
        eq_start, eq_end = match.span()

        # Check if the equations has //nonumber
        nonumber_match = re.search(r'\\nonumber', equation)
        if nonumber_match:
            continue

        # Check if the equation has a label
        label_match = re.search(r'\\label{([^}]+)}', equation)
        equation_label = label_match.group(1) if label_match else None

        references = []  # Store references to the equation
        prev_context = None  # Store the paragraph before the equation
        next_context = None  # Store the paragraph after the equation

        # Loop through paragraphs to find those preceding and following the equation
        for i, paragraph in enumerate(paragraphs):
            par_start = tex_content.find(paragraph)
            par_end = par_start + len(paragraph)

            if par_end < eq_start:
                prev_context = paragraph.strip()  # Last paragraph before the equation
                
            if par_start > eq_end and next_context is None:
                next_context = paragraph.strip()  # First paragraph after the equation
                break  # Stop searching once the next context is found

        # Construct the context for the equation based on the preceding and following paragraphs
        context = []
        if prev_context:
            context.append(prev_context)
        if next_context:
            context.append(next_context)

        # If the equation has a label, find references
        if equation_label:
            ref_pattern = re.compile(r'\\eqref\{' + re.escape(equation_label) + r'\}|\\ref\{' + re.escape(equation_label) + r'\}')
            for paragraph in paragraphs:
                if ref_pattern.search(paragraph):
                    references.append(paragraph.strip())

        equation_data.append({
            "equation": equation.strip(),
            "label": equation_label,
            "references": references,
            "context": context
        })

    return equation_data

def is_folder_empty(folder_path):
    return not os.listdir(folder_path)

def main():
    equation_data_all_files = []
    base_path = './neurips/2022/extracted_files/'
    dest_path = './neurips/2022/equation_data.json'

    for folder in tqdm(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        print(f"Processing {folder_path}")

        if is_folder_empty(folder_path):
            print(f"Folder {folder_path} is empty")
            continue

        for tex_file in tqdm(os.listdir(folder_path)):
            if tex_file.endswith(".tex"):
                file_path = os.path.join(folder_path, tex_file)
                print(f"Processing {file_path}")
                equation_data = process_tex_file(file_path)
                equation_data_all_files.extend([{"folder": folder, "filename": tex_file, **equation} for equation in equation_data])
                print('---------------------')

        save_json(equation_data_all_files, dest_path)

if __name__ == "__main__":
    main()


