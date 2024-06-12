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
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    
    parser.add_argument("--modality", help='modality to use -> equation, context, or reference', choices=['equation', 'context', 'reference'])
    
    parser.add_argument('--use', help="use all equations or neighboring equations", choices=['all', 'neighboring'])

    parser.add_argument('--split', help="evaluate on dev/test", choices=["dev", "test"], default="dev")

    parser.add_argument('--temperature', type=float, help="temperature", choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9],  default=0)

    parser.add_argument('--seed', type=int, default=5731)

    args = parser.parse_args()

    model = args.model

    data = utils.load_jsonl(f'./cPAPERS/dataset/{args.split}/equation_final.jsonl')

    prompts = []
    ground_truth = []
    for idx in range(len(data)):
        datum = data[idx]

        # Use neighboring equations/contexts/references or all of them
        # Using all of them causes context to be too long 
        if args.use=='neighboring':
            equations = datum['neighboring_equations']
            contexts = datum['neighboring_contexts']
            references = datum['neighboring_references']
        elif args.use=='all':
            equations = datum['equations']
            contexts = datum['contexts']
            references = datum['references']

        user_message = datum['question']

        # Consider the case where only the question refers to the equation
        # pattern_equation = re.compile(r'\b(?:Equation|Eq\.?)\s*\(?\s*\d+\)?\b', re.IGNORECASE)
        # if re.search(pattern_equation, user_message)==None:
            # continue

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

        # prompt = f"<s>[INST] <<SYS>> Answer the question in the user's message. Additional information is provided as context after the user's question. <</SYS>> User message: {user_message}. Context: {system_prompt} [/INST]"
        prompt = f"<s>[INST] <<SYS>> Answer the question in the user's message. Additional information is provided as context before the user's question. <</SYS>> Context: {system_prompt} User message: {user_message}.[/INST]"
        prompts.append(prompt)
        ground_truth.append(datum['answer'])

    responses = lm.generate_responses_zs_vllm(prompts, args, max_tokens=512)

    # import bpdb; bpdb.set_trace()

    output_data = [{'response':responses[i], 'ground_truth':ground_truth[i]} for i in range(len(responses))]

    if not(os.path.exists('./outputs')):
        os.makedirs('./outputs')

    utils.write_jsonl(output_data, f"./outputs/equation_zs_responses_{modality}_{args.use}.jsonl")
    
    from evaluation import evaluator

    Evaluator = evaluator(responses, ground_truth)
    
    Evaluator.print_results()
    
    # eval(responses, ground_truth)

    

    

    




