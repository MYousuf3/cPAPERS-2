# Conversational Papers (cPAPERS): A Dataset of Situated and Multimodal Interactive Conversations in Scientific Papers

This repository contains the code and dataset for the paper:  
**Conversational Papers (cPAPERS): A Dataset of Situated and Multimodal Interactive Conversations in Scientific Papers**  
[ArXiv link](https://arxiv.org/abs/2406.08398)

The cPAPERS dataset is available on [Hugging Face](https://huggingface.co/datasets/avalab/cPAPERS).

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

## Introduction
Conversational Papers (cPAPERS) is a dataset of conversations in English situated in scientific texts. cPAPERS consists of question-answer pairs pertaining to figures (cPAPERS-FIGS), equations (cPAPERS-EQNS), or tabular information (cPAPERS-TBLS) from scientific papers. 

cPAPERS is designed to facilitate research on interactive conversations within the context of scientific papers. It includes question-answer pairs, extracted figures, tables, equations in LaTeX format, and their surrounding context and references in the papers.

## Usage
To collect question-answer pairs and other relevant data, follow these steps:

1. **Navigate to the data collection folder:**
    ```bash
    cd data_collection
    ```

2. **Download papers, comments, and reviews from OpenReview:**
    ```bash
    python collect_data.py
    ```

3. **Extract QA pairs using LLaMA+GPT:**
    ```bash
    python extract_qas.py
    ```

4. **Clean the QA pairs:**
    ```bash
    python clean_and_extract_num.py
    ```

5. **Retrieve figures, tables, and equations and their surrounding context from the downloaded `.tex` files:**
    ```bash
    python get_figures.py
    python get_equations.py
    python get_tables.py
    ```

6. **Match QA pairs with the extracted figures, tables, and equations data using `paper_id`:**
    ```bash
    python match_data.py
    ```

7. **Convert `.pdf` and `.eps` files to `.png`:**
    ```bash
    python convert_figures.py
    ```

## Experiments
To reproduce the experiments discussed in the paper, use the following commands:

### Zero-Shot Experiments
- Run zero-shot experiments for equations:
    ```bash
    ./run_zs_equation.sh
    ```
- Run zero-shot experiments for figures:
    ```bash
    ./run_zs_figure.sh
    ```
- Run zero-shot experiments for tables:
    ```bash
    ./run_zs_table.sh
    ```

### Fine-Tuning Experiments
- Run fine-tuning experiments for equations:
    ```bash
    ./run_ft_equation.sh
    ```
- Run fine-tuning experiments for figures:
    ```bash
    ./run_ft_figure.sh
    ```
- Run fine-tuning experiments for tables:
    ```bash
    ./run_ft_table.sh
    ```

### Ablation Experiments on Temperature
- Run ablation experiment on the temperature for equations:
    ```bash
    ./run_zs_equation_temperature.sh
    ```
- Run ablation experiment on the temperature for figures:
    ```bash
    ./run_zs_figure_temperature.sh
    ```
- Run ablation experiment on the temperature for tables:
    ```bash
    ./run_zs_table_temperature.sh
    ```

## Citation
If you use this dataset in your research, please cite our paper:

```bibtex
@article{sundar2024cpapers,
  title={cPAPERS: A Dataset of Situated and Multimodal Interactive Conversations in Scientific Papers},
  author={Anirudh Sundar, Jin Xu, William Gay, Christopher Richardson, Larry Heck},
  journal={arXiv preprint arXiv:2406.08398},
  year={2024}
}