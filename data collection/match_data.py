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

def reformat_equations(data):
    result = {}
    for item in data:
        folder = item['folder']
        if folder not in result:
            result[folder] = []
        # Add the desired information to the list
        result[folder].append({
            "filename": item['filename'],
            "equation": item['equation'],
            "label": item['label'],
            "references": item['references'],
            "context": item['context'],
        })
    return result

def reformat_figures(data):
    result = {}
    for item in data:
        folder = item['folder']
        if folder not in result:
            result[folder] = []
        entry = {
            "filename": item.get('filename'),
            "caption": item.get('caption'),
            "label": item.get('label'),
            "figure_path": item.get('figure_path'),
            "references": item.get('references', []),
            "context": item.get('context', [])
        }
        result[folder].append(entry)
    return result

def reformat_table(data):
    result = {}
    for item in data:
        folder = item['folder']
        if folder not in result:
            result[folder] = []
        entry = {
            "filename": item.get('filename'),
            "caption": item.get('caption'),
            "label": item.get('label'),
            "table": item.get('table'),
            "references": item.get('references', []),
            "context": item.get('context', [])
        }
        result[folder].append(entry)
    return result

def match_equations_with_qa_by_paper_id(qas, formatted_figures, formatted_equations, formatted_table):
    matched_data = []

    for qa in qas:
        paper_id = qa['paper_id']
        key = paper_id + '.tar'

        if 'Figure' in qa and qa['Figure_number'] != 'n/a':
            figure_data = formatted_figures.get(key, [])
            number = int(qa['Figure_number'])
            if figure_data and number <= len(figure_data):
                qa['figure_data'] = figure_data[number - 1]
            else:
                qa['figure_data'] = []
        
        if 'Equation' in qa:
            equation_data =  formatted_equations.get(key, [])
            # dump all equation data 
            if equation_data:
                qa['equation_data'] = equation_data
            else:
                qa['equation_data'] = []

        if 'Table' in qa and qa['Table_number'] != 'n/a' :
            table_data = formatted_table.get(key, [])
            # number = int(qa['Table_number'])
            if table_data:
                qa['table_data'] = table_data
            else:
                qa['table_data'] = []

            # if table_data and number <= len(table_data):
            #     qa['table_data'] = table_data[number - 1]
            # else:
            #     qa['table_data'] = []

        matched_data.append(qa)

    return matched_data

def clean_matched_data(matched_data):
    # formatted_data = []
    figure_data = []
    equation_data = []
    table_data = []

    for item in matched_data:
        try:
            if "equation_data" in item.keys():
                if item["equation_data"] == []:
                    continue

                extracted_info = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "equation": item["equation_data"], # getting all equations
                    "equation_number": item["Equation_number"],
                    # "equation": item["equation_data"]["equation"],
                    # "context": item["equation_data"]["context"],
                    # "references": item["equation_data"]["references"],
                    "paper_id": item['paper_id']
                }
                equation_data.append(extracted_info)
                
            elif "figure_data" in item.keys():
                if item["figure_data"] == []:
                    continue
                
                # directory = item["paper_id"]
                # base_path = f'./extracted_files/{directory}.tar/'

                # figure = item["figure_data"].get("figure_path", "No figure path available")
                # if figure != "No figure path available":
                #     figure_path = base_path + figure
                figure_path = item["figure_data"].get("figure_path", "No figure path available")

                extracted_info = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "figure": figure_path,
                    "figure_number": item["Figure_number"],
                    "caption": item["figure_data"]["caption"],
                    "context": item["figure_data"]["context"],
                    "references": item["figure_data"]["references"],
                    "paper_id": item['paper_id']
                }
                figure_data.append(extracted_info)

            elif "table_data" in item.keys():
                if item["table_data"] == []:
                    continue

                extracted_info = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "table_number": item["Table_number"],
                    "table": item["table_data"],
                    # "caption": item["table_data"]["caption"],
                    # "context": item["table_data"]["context"],
                    # "references": item["table_data"]["references"],
                    "paper_id": item['paper_id']
                }
                table_data.append(extracted_info)
            else:
                continue 

            # formatted_data.append(extracted_info)
        except:
            continue
    
    # return formatted_data
    return figure_data, equation_data, table_data

        

def main():
    base_path = './neurips/2022/'
    dataset_path = './neurips/2022/'

    equations = open_json(base_path+'equation_data.json')
    formatted_equations = reformat_equations(equations)
    # print(f"total equations: {len(equations)}")
    print(f"total formated equations: {len(formatted_equations)}")
    
    figures = open_json(base_path+'figures_data.json')
    formatted_figures = reformat_figures(figures)
    save_json(formatted_figures, 'formated_figures.json')
    print(f"total formated figures: {len(formatted_figures)}")

    table = open_json(base_path+'table_data.json')
    formatted_table = reformat_table(table)
    print(f"total formated tables: {len(formatted_table)}")

    qas = open_json(base_path+'/full_cleaned_qa_with_nums.json')
    print(f"total qa pairs: {len(qas)}")

    matched_results = match_equations_with_qa_by_paper_id(qas, formatted_figures, formatted_equations, formatted_table)
    print(f"total matched: {len(matched_results)}")
    save_json(matched_results,  base_path+'matched_data.json')

    matched_data = open_json( base_path+'matched_data.json')
    cleaned_figure, cleaned_equation, cleaned_table = clean_matched_data(matched_data)
    print(f"after cleaning:", len(cleaned_figure) + len(cleaned_equation) + len(cleaned_table))
    save_json(cleaned_figure, dataset_path+'final_figure.json')
    save_json(cleaned_equation, dataset_path+'final_equation.json')
    save_json(cleaned_table, dataset_path+'final_table.json')
   
    
    # Counting
    print(f"Total Count:{len(cleaned_figure) + len(cleaned_equation) + len(cleaned_table)}")
    print(f"Figure Count: {len(cleaned_figure)}")
    print(f"Equation Count: {len(cleaned_equation)}")
    print(f"Table Count: {len(cleaned_table)}")

if __name__ == '__main__':
    main()