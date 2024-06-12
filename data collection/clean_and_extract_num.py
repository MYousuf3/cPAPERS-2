import os
import json
import re
import time

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    

def extract_numbers(input_string):
    try: 
        # Define a regular expression pattern to match numbers
        pattern = r'\d+'
        
        # Use re.findall() to find all occurrences of the pattern in the input string
        numbers = re.findall(pattern, input_string)
        
        # Convert the list of strings to a list of integers
        # numbers = [int(num) for num in numbers]
        
        return numbers[0]
    except Exception as e:
        return 'n/a'
    
def main(base_path):
    qa = open_json(base_path+'/cleaned_qa_pair.json')

    bad1 = "the performance metric"
    bad2 = "Jensen's inequality"
    bad3 = "performance of IncFinetune"
    # bad = "In eq. (2), the paper introduced the temperature parameter to control"
    # bad2 = "The temperature parameter"

    bad_count = 0
    null_count = 0
    question_count = 0
    figure_count = 0
    equation_count = 0
    table_count = 0

    figures = []
    equations = []
    tables = []
    all = []

    for item in qa:
        cleaned = item['cleaned_qa']
        paper_id = item['paper_id']
        if cleaned is None:
            null_count += 1
            continue

        for qa_pair in cleaned:
            if bad1 in qa_pair['question'] or bad1 in qa_pair['answer'] or bad2 in qa_pair['question'] or bad2 in qa_pair['answer'] or bad3 in qa_pair['question'] or bad3 in qa_pair['answer']:
                bad_count += 1
            else:
                question_count += 1
                qa_pair['paper_id'] = paper_id
                if "Figure" in qa_pair.keys():
                    figure_count += 1
                    qa_pair['Figure_number'] = extract_numbers(qa_pair['Figure_with_num'])
                    figures.append(qa_pair)
    
                if "Equation" in qa_pair.keys():
                    equation_count += 1
                    qa_pair['Equation_number'] = extract_numbers(qa_pair['Equation_with_num'])
                    equations.append(qa_pair)
                    
                if "Table" in qa_pair.keys():
                    table_count += 1
                    qa_pair['Table_number'] = extract_numbers(qa_pair['Table_with_num'])
                    tables.append(qa_pair)

                all.append(qa_pair)
    print(f"Total number of QA pairs: {question_count}")
    print(f"Total number of figures: {figure_count}")
    print(f"Total number of equations: {equation_count}")
    print(f"Total number of tables: {table_count}")
    print(f"Total number of bad: {bad_count}")
    print(f"Total number of null: {null_count}")

    save_json(figures, base_path + '/figures.json')
    save_json(equations, base_path + '/equations.json')
    save_json(tables, base_path + '/tables.json')
    save_json(all, base_path + '/full_cleaned_qa_with_nums.json')

if __name__ == "__main__":
    base_path = './neurips/2022'
    main(base_path)
    