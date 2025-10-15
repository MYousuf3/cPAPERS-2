import os
import subprocess
import json

base_path = './neurips/2021/final_figure.json'
figure_path = './figure_files/'

# Function to check if a file ends with .pdf or .eps
def is_figure_file(filename):
    return filename.lower().endswith(('.pdf', '.eps'))

# Function to convert a figure file to PNG
def convert_to_png(figure_file):
    png_file = os.path.splitext(figure_file)[0] + '.png'
    subprocess.run(['convert', os.path.join(figure_path, figure_file), os.path.join(figure_path, png_file)])
    return png_file

# Function to process the JSON data
def process_json_data(data):
    for item in data:
        figure_file = item.get('figure')
        if figure_file and is_figure_file(figure_file):
            if os.path.exists(os.path.join(figure_path, figure_file)):
                png_file = convert_to_png(figure_file)
                item['figure'] = png_file
                print(f"Converted '{figure_file}' to '{png_file}' and updated the JSON.")
                # os.remove(os.path.join(figure_path, figure_file))  # Remove the old figure file
                # print(f"Removed old file '{figure_file}'.")
            else:
                print(f"Figure file '{figure_file}' not found in '{figure_path}'.")

    # Write the updated JSON data back to the file
    with open(base_path, 'w') as file:
        json.dump(data, file, indent=4)
    print("Updated JSON file written successfully.")
            

# Open and process the JSON file
with open(base_path, 'r') as file:
    data = json.load(file)

print(f'Total number of items: {len(data)}')
process_json_data(data)