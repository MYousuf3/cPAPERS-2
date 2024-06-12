import os
import shutil 
import re
from pathlib import Path
import json
from tqdm import tqdm 

# Extracts all .tex files and creates a new directory called tex_files with them

# re to match arxiv ID and version, ef: 1305.1206v1
def extract_identifier(input_string):
    # The regular expression pattern to match the identifier
    # \d{4} matches exactly four digits
    # \.\d+ matches a dot followed by one or more digits
    # v\d+ matches 'v' followed by one or more digits
    pattern = r'\d{4}\.\d+v\d+'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # If a match is found, return the matched string
    if match:
        return match.group()
    else:
        return "No match found"


def extract_tex_files(directory):
    tex_files = []
    if not(os.path.exists("tex_files")):
        os.mkdir("tex_files")

    for root, dirs, files in os.walk(directory):
        print(root)
        for file in files:
            if file.endswith(".tex"):
                arxiv_id = extract_identifier(root)
                shutil.copy(os.path.join(root, file), f"./tex_files/{arxiv_id}_{file}")





def create_path_and_get_next_idx(outfile, overwrite=False):
    p = Path(outfile)
    # check
    if p.is_dir():
        raise ValueError(f"Output file {outfile} cannot be a directory.")

    # create output dir
    p.parent.mkdir(parents=True, exist_ok=True)

    # overwrite file if asked
    if overwrite:
        with open(outfile, "w") as f:
            return 0

    # get num lines
    if p.exists():
        with open(p.as_posix(), "rb") as f:
            return sum(1 for _ in f)
    else:
        with open(p.as_posix(), "w") as f:
            return 0
        


def write_record_to_jsonl(outfile, item):
    with open(outfile, "a") as f:
            f.write(json.dumps(item) + "\n")


def load_jsonl(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return [json.loads(record) for record in data]


def extract_text_between_tags(text, start_tag="[ANS]", end_tag="[/ANS]"):

    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # Extract the text between these indices, stripping any leading/trailing whitespace.
    extracted_text = text[start_index:end_index].strip() if start_index > len(start_tag)-1 and end_index != -1 else ""

    return extracted_text


def write_jsonl(data, outfile):
    if os.path.dirname(outfile) != '':
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        for record in data:
            json_record = json.dumps(record)
            f.write(json_record + '\n')



def write_jsonl_multiple(data,keys, outfile):
    # Here data is a list of lists
    # Keys is a list of strings indicating the keys for the dictionaries

    if os.path.dirname(outfile) != '':
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    with open(outfile, 'w') as f:
        for i in range(len(data[0])):
            record = {}
            for j in range(len(keys)):
                record[keys[j]] = data[j][i]
            
            json_record = json.dumps(record)
            f.write(json_record + '\n')



def unflatten(flattened_list, lengths):
    """
    Reconstructs the original list of lists from a flattened list and the lengths of the original sublists.

    Parameters:
    - flattened_list: A list containing all elements from the original list of lists.
    - lengths: A list of integers where each integer represents the length of a sublist in the original list.

    Returns:
    A list of lists reconstructed based on the provided lengths.
    """
    unflattened = []
    start = 0
    for length in lengths:
        # Extract the sublist using the current start index and the length
        end = start + length
        sublist = flattened_list[start:end]
        unflattened.append(sublist)
        # Update the start index for the next iteration
        start = end
    return unflattened


def convert_json_to_jsonl(infile, outfile):
    data = json.load(open(infile, 'r'))
    write_jsonl(data, outfile)



def flatten_dialogue(infile , outfile):

    flat_dialogue = []

    # with open(infile, 'r') as f:
        # data = json.load(f)
    
    data = load_jsonl(infile)

    id = 0

    for datum in tqdm(data):

        if 'dialogue' not in datum:
            continue

        conversation = datum['dialogue']
        q1 = conversation['q1']
        a1 = conversation['a1']
        
        q2 = conversation['q2']
        a2 = conversation['a2']
        
        q3 = conversation['q3']
        a3 = conversation['a3']

        turn1 = {'question': q1, 'answer': a1, 'context':''.join(datum['context']), 'references':datum['references'],'equation':datum['equation'],'id':id}
        id += 1

        turn2 = {'question': q1+a1+q2, 'answer': a2, 'context':''.join(datum['context']), 'references':datum['references'],'equation':datum['equation'],'id':id}
        id += 1
        
        turn3 = {'question': q1+a1+q2+a2+q3, 'answer': a3, 'context':''.join(datum['context']), 'references':datum['references'],'equation':datum['equation'],'id':id}
        id += 1

        with open(outfile, 'a+') as f:
            f.write(json.dumps(turn1) + '\n')
            f.write(json.dumps(turn2) + '\n')
            f.write(json.dumps(turn3) + '\n')



def get_surrounding_elements(lst, index):
    # Create an empty list to store the valid elements
    result = []

    if len(lst) == 1:
        result.append(0)
        return result
    
    # Check if the index-1 is within the valid range and add to the result if true
    if index - 1 >= 0:
        result.append(lst[index - 1])
    
    # Check if the index is within the valid range and add to the result if true
    if 0 <= index < len(lst):
        result.append(lst[index])
    
    # Check if the index+1 is within the valid range and add to the result if true
    if index + 1 < len(lst):
        result.append(lst[index + 1])
    
    return result