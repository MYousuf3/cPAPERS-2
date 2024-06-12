# Fine tune llama for table qa

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer, pipeline 
# import bpdb
from tqdm import tqdm 
from argparse import ArgumentParser
import sys
import os
import utils
# import eval_utils as e_utils
import numpy as np
import language_modeling as lm
from evaluation import evaluator

import torch 

def prep_data(example):
    example["prompt"] = example["input_table"] + " " + example["utterance"]
    return example


def preprocess_function_tables(example):
    example['text'] = f'<s>[INST] {example["neighboring_tables"]} {example["question"]} [/INST] [ANS] {example["answer"]} [/ANS] </s>'
    return example

def preprocess_function_inference_tables(example):
    example['text'] = f'<s>[INST] {example["neighboring_tables"]} {example["question"]} [/INST]'
    return example


def preprocess_function_contexts(example):
    example['text'] = f'<s>[INST] {example["neighboring_contexts"]} {example["question"]} [/INST] [ANS] {example["answer"]} [/ANS] </s>'
    return example


def preprocess_function_inference_contexts(example):
    example['text'] = f'<s>[INST] {example["neighboring_contexts"]} {example["question"]} [/INST]'
    return example


def preprocess_function_references(example):
    example['text'] = f'<s>[INST] {example["neighboring_references"]} {example["question"]} [/INST] [ANS] {example["answer"]} [/ANS] </s>'
    return example


def preprocess_function_inference_references(example):
    example['text'] = f'<s>[INST] {example["neighboring_references"]} {example["question"]} [/INST]'
    return example


def preprocess_function_default(example):
    example['text'] = f'<s>[INST] {example["question"]} [/INST] [ANS] {example["answer"]} [/ANS] </s>'
    return example


def preprocess_function_inference_default(example):
    example['text'] = f'<s>[INST] {example["question"]} [/INST]'
    return example



def lora_setup():
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    
    peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    )

    return peft_config


def train(args):
    dataset = load_dataset("json", data_files="./cPAPERS/dataset/train/table_final.jsonl", split="train")
    
    if args.modality == 'table':
        preprocess_function = preprocess_function_tables
    elif args.modality == 'context':
        preprocess_function = preprocess_function_contexts
    elif args.modality == 'reference':
        preprocess_function = preprocess_function_references
    else:
        preprocess_function = preprocess_function_default

    dataset = dataset.map(preprocess_function)

    peft_config = lora_setup()

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "nf4"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = getattr(torch, "float16")

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False


    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )


    training_args = TrainingArguments(
                output_dir=f'{args.model}_{args.expt_name}_{args.modality}/',
                num_train_epochs=5,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                optim="paged_adamw_32bit",
                per_device_eval_batch_size=4,
                fp16=False, # Overflows with fp16
                bf16=False,
                learning_rate=2e-4,
                warmup_ratio=0.03,
                group_by_length=True,
                max_steps=-1,
                lr_scheduler_type="constant",
                # logging & evaluation strategies
                logging_dir=f'./logs/',
                logging_strategy="epoch",
                evaluation_strategy="no",
                # eval_steps=10,
                # eval_delay=3,
                do_eval=False, 
                save_strategy="epoch",
                save_total_limit=1,
                # load_best_model_at_end=True,
                report_to="wandb",
                push_to_hub=False,    
            )



    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map='auto')
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto',quantization_config=quant_config)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=dataset.select([0, 10, 20, 30, 40, 50]),
        packing=False,
        max_seq_length=2048,
        # max_seq_length=4096,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        # compute_metrics=compute_metrics_fn,
        # data_collator=data_collator,
        # formatting_func=formatting_prompts_func,
    )

    # bpdb.set_trace()

    trainer.train()

    # merge peft weights with base model
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     args.output_dir,
    #     low_cpu_mem_usage=True,
    #  )

def eval(args):
    # Evaluation:

    split = args.split

    eval_dataset = load_dataset("json", data_files=f"./cPAPERS/dataset/{split}/table_final.jsonl", split="train")
    
    if args.modality == 'table':
        preprocess_function_inference = preprocess_function_inference_tables
    elif args.modality == 'context':
        preprocess_function_inference = preprocess_function_inference_contexts
    elif args.modality == 'reference':
        preprocess_function_inference = preprocess_function_inference_references
    else:
        preprocess_function_inference = preprocess_function_inference_default
    
    eval_dataset = eval_dataset.map(preprocess_function_inference)

    if args.vllm:
        from vllm import LLM, SamplingParams

        from vllm.lora.request import LoRARequest

        answers = lm.generate_responses_ft_vllm(eval_dataset['text'], args)
        ground_truth = eval_dataset['answer']

        # bpdb.set_trace()

        Evaluator = evaluator(answers, ground_truth)

        Evaluator.print_results()
        # Evaluator.print_results2('table')
        
        # eval(answers, ground_truth)

        if not(os.path.exists('./outputs')):
            os.makedirs('./outputs')

        output_data = [{'response':answers[i], 'ground_truth':ground_truth[i]} for i in range(len(answers))]

        utils.write_jsonl(output_data, f"./outputs/{args.expt_name}_responses_{args.modality}.jsonl")

        # bpdb.set_trace()
    



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--vllm", action="store_true", default=True)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--modality", help='modality to use -> table, context, or reference', choices=['table', 'context', 'reference'])
    parser.add_argument("--split", required='eval' in sys.argv, default='dev', choices=["dev", "test"], help="evaluate on dev/test")
    parser.add_argument("--outfile", required='eval' in sys.argv, help="output file to store predictions", default="predictions/llama_7b_ft.jsonl")
    parser.add_argument("--expt_name", default="table_ft")

    parser.add_argument('--temperature', type=float, help="temperature", choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9],  default=0)
    parser.add_argument('--seed', type=int, default=5731)
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    if args.eval:
        eval(args)

