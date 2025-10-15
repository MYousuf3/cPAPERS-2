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

    # Split the document into paragraphs to help find context around tables
    paragraphs = re.split(r'\n\s*\n', tex_content)
    
    table_data = []

    # Use re.finditer to get the match object for more detailed information (like start and end positions)
    for match in re.finditer(r'\\begin{table}(.*?)\\end{table}', tex_content, re.DOTALL):
        table = match.group(1)
        table_start, table_end = match.span()

        # Extract caption
        caption_match = re.search(r'\\caption{(.*?)}', table, re.DOTALL)
        caption = caption_match.group(1) if caption_match else "No caption"

        # Find actual table block using \tabular tag
        tabular_match = re.search(r'\\begin{tabular}(.*?)\\end{tabular}', table, re.DOTALL)
        tabular = tabular_match.group(1) if tabular_match else "No table"

        # Check if the tables has //nonumber
        nonumber_match = re.search(r'\\nonumber', table)
        if nonumber_match:
            continue

        # Check if the table has a label
        label_match = re.search(r'\\label{([^}]+)}', table)
        table_label = label_match.group(1) if label_match else None

        references = []  # Store references to the table
        prev_context = None  # Store the paragraph before the table
        next_context = None  # Store the paragraph after the table

        # Loop through paragraphs to find those preceding and following the table
        for i, paragraph in enumerate(paragraphs):
            par_start = tex_content.find(paragraph)
            par_end = par_start + len(paragraph)

            if par_end < table_start:
                prev_context = paragraph.strip()  # Last paragraph before the table
                
            if par_start > table_end and next_context is None:
                next_context = paragraph.strip()  # First paragraph after the table
                break  # Stop searching once the next context is found

        # Construct the context for the table based on the preceding and following paragraphs
        context = []
        if prev_context:
            context.append(prev_context)
        if next_context:
            context.append(next_context)

        # If the table has a label, find references
        if table_label:
            ref_pattern = re.compile(r'\\ref\{' + re.escape(table_label) + r'\}')
            for paragraph in paragraphs:
                if ref_pattern.search(paragraph):
                    references.append(paragraph.strip())

        table_data.append({
            "caption": caption.strip(),
            "table": tabular.strip(),
            "label": table_label,
            "references": references,
            "context": context
        })

    return table_data

def is_folder_empty(folder_path):
    return not os.listdir(folder_path)

def main():
    table_data_all_files = []
    base_path = './neurips/2021/extracted_files/'
    dest_path = './neurips/2021/table_data.json'

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
                table_data = process_tex_file(file_path)
                table_data_all_files.extend([{"folder": folder, "filename": tex_file, **table} for table in table_data])
                print('---------------------')
                
        save_json(table_data_all_files, dest_path)

if __name__ == "__main__":
    main()

    