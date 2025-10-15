# Zero shot qa for equations
from vllm import LLM, SamplingParams
import json
import utils
# import bpdb
from argparse import ArgumentParser
import evaluation
import language_modeling as lm
import sys, os
from evaluation import evaluator
import re



def eval(responses, ground_truth):
    # Compute the scores
    rouge = evaluation.rouge_scores(responses, ground_truth)
    print(f"Rouge scores: {rouge}")

    bleu = evaluation.bleu_scores(responses, ground_truth)
    print(f"Bleu scores: {bleu}")

    meteor = evaluation.meteor_scores(responses, ground_truth)
    print(f"Meteor scores: {meteor}")

    bert_score = evaluation.bert_score(responses, ground_truth)
    print(f"Bert score: {bert_score}")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, help="Response file to evaluate", required=True)
    
    # Original arguments (commented out)
    # parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    # parser.add_argument("--modality", help='modality to use -> equation, context, or reference', choices=['equation', 'context', 'reference'])
    # parser.add_argument('--use', help="use all equations or neighboring equations", choices=['all', 'neighboring'])
    # parser.add_argument('--split', help="evaluate on dev/test", choices=["dev", "test"], default="dev")
    # parser.add_argument('--temperature', type=float, help="temperature", choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9],  default=0)
    # parser.add_argument('--seed', type=int, default=5731)

    args = parser.parse_args()

    # Load the responses from the specified file
    output_data = utils.load_jsonl(f"./outputs/{args.file}")
    responses = [item['response'] for item in output_data]
    ground_truth = [item['ground_truth'] for item in output_data]
    
    # Original code (commented out)
    # model = args.model
    # data = utils.load_jsonl(f'../data collection/neurips/2021/final_equation.json')

    # Original code for generating responses (commented out)
    """
    prompts = []
    ground_truth = []
    for idx in range(len(data)):
        datum = data[idx]

        # Extract equations, contexts and references from the equation array
        equations = []
        contexts = []
        references = []
        
        # In the 'all' case, we use all equations from the array
        # In the 'neighboring' case, we only use the first equation (closest match)
        eq_list = datum['equation']
        if args.use == 'neighboring' and len(eq_list) > 0:
            eq_list = [eq_list[0]]
            
        for eq in eq_list:
            equations.append(eq['equation'])
            contexts.extend(eq.get('context', []))
            references.extend(eq.get('references', []))

        user_message = datum['question']

        if args.modality=='equation':
            system_prompt = f"Equation: {equations}"
            modality = "equation"
        elif args.modality=='context':
            system_prompt = f"Context: {contexts}"
            modality = "context"
        elif args.modality=='reference':
            system_prompt = f"Reference: {references}"
            modality = "reference"
        else:
            system_prompt = ""
            modality = ""

        prompt = f"<s>[INST] <<SYS>> Answer the question in the user's message. Additional information is provided as context before the user's question. <</SYS>> Context: {system_prompt} User message: {user_message}.[/INST]"
        prompts.append(prompt)
        ground_truth.append(datum['answer'])

    responses = lm.generate_responses_zs_vllm(prompts, args, max_tokens=512)
    output_data = [{'response':responses[i], 'ground_truth':ground_truth[i]} for i in range(len(responses))]

    if not(os.path.exists('./outputs')):
        os.makedirs('./outputs')

    utils.write_jsonl(output_data, f"./outputs/equation_zs_responses_{modality}_{args.use}.jsonl")
    """
    
    print(f"\nEvaluating {len(responses)} responses from {args.file}")
    
    # Initialize evaluator and print results
    Evaluator = evaluator(responses, ground_truth)
    Evaluator.print_results()
