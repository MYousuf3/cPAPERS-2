import os
import json
import re
from tqdm import tqdm
import shutil

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        
def process_figures(file_path, folder_path, folder_name):
    with open(file_path, 'r', encoding='latin-1') as file:
        tex_content = file.read()

    figures_data = []

    # Regular expression to find figure environments, their captions, and labels
    figures = re.finditer(r'\\begin{figure}(.*?)\\end{figure}', tex_content, re.DOTALL)
    paragraphs = re.split(r'\n\s*\n', tex_content)  # Split the document into paragraphs for context

    for figure_match in figures:
        figure_content = figure_match.group(1)
        fig_start, fig_end = figure_match.span()

        # Extract caption
        caption_match = re.search(r'\\caption{(.*?)}', figure_content, re.DOTALL)
        caption = caption_match.group(1) if caption_match else "No caption"

        # Extract label
        label_match = re.search(r'\\label{(.*?)}', figure_content)
        figure_label = label_match.group(1) if label_match else None

        # Extract path to figure file
        includegraphics_match = re.search(r'\\includegraphics(?:\[.*?\])?{(.*?)}', figure_content)
        figure_path = includegraphics_match.group(1) if includegraphics_match else None

        prev_context = None
        next_context = None

        # Loop through paragraphs to find those preceding and following the figure
        for i, paragraph in enumerate(paragraphs):
            par_start = tex_content.find(paragraph)
            par_end = par_start + len(paragraph)

            if par_end < fig_start:
                prev_context = paragraph.strip()
                
            if par_start > fig_end and next_context is None:
                next_context = paragraph.strip()
                break

        # Construct the context for the figure based on the preceding and following paragraphs
        context = []
        if prev_context:
            context.append(prev_context)
        if next_context:
            context.append(next_context)

        references = []
        # Find references to the figure
        if figure_label:
            ref_pattern = re.compile(r'\\ref\{' + re.escape(figure_label) + r'\}')
            for paragraph in paragraphs:
                if ref_pattern.search(paragraph):
                    references.append(paragraph.strip())
        
        # Rename the figure name as paperID.figure_path, and then copy it to ../figure_files
        if figure_path:
            source_file = os.path.join(folder_path, figure_path)

            base_name, _ = os.path.splitext(folder_name)
            new_figure_name = base_name + '.' + os.path.basename(figure_path) # new name
            figure_folder =  os.path.join(os.path.dirname(os.path.dirname(folder_path)), 'figure_files/')
            destination_directory = os.path.join(figure_folder, new_figure_name)

            # copy this file to the figure_files folder
            if os.path.exists(source_file):
                # Check if the destination directory exists, if not, create it
                if not os.path.exists(figure_folder):
                    os.makedirs(figure_folder)
                # Copy the file to the destination directory
                try: # Attempt to copy the file
                    shutil.copy(source_file, destination_directory)
                    # print("File copied successfully.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            else:
                print("Source file does not exist.")
  
        figures_data.append({
            "caption": caption.strip(),
            "label": figure_label,
            "figure_path": new_figure_name if figure_path else "No figure path", # renamed figure name 
            # "raw_figure_path": figure_path if figure_path else "No figure path",
            "references": references,
            "context": context
        })

    return figures_data

def is_folder_empty(folder_path):
    return not os.listdir(folder_path)

def main():
    figures_data_all_files = []
    base_path = './neurips/2021/extracted_files/'
    dest_path = './neurips/2021/figures_data.json'
    
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
                figures_data = process_figures(file_path, folder_path, folder)
                figures_data_all_files.extend([{"folder": folder, "filename": tex_file, **figure} for figure in figures_data])
                print('---------------------')
        
        save_json(figures_data_all_files, dest_path)

if __name__ == "__main__":
    main()
