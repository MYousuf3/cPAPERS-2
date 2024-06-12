# Zero shot qa for figures
from vllm import LLM, SamplingParams
import json
import utils
# import bpdb
from argparse import ArgumentParser
import evaluation
import language_modeling as lm
import sys, os
from evaluation import evaluator




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
    
    parser.add_argument("--modality", help='modality to use -> figure caption, context, or reference', choices=['caption', 'context', 'reference'])

    parser.add_argument('--split', help="evaluate on dev/test", choices=["dev", "test"], default="dev")
    
    parser.add_argument('--temperature', type=float, help="temperature", choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9],  default=0)

    parser.add_argument('--seed', type=int, default=5731)

    args = parser.parse_args()

    model = args.model

    data = utils.load_jsonl(f'./cPAPERS/dataset/{args.split}/figure_final.jsonl')

    prompts = []
    ground_truth = []
    for idx in range(len(data)):
        datum = data[idx]

        user_message = datum['question']

        caption = datum['caption']
        context = datum['context']
        references = datum['references']

        if args.modality=='caption':
            system_prompt = f"Caption: {caption}"
            modality = args.modality
        elif args.modality=='context':
            system_prompt = f"Context: {context}"
            modality = args.modality
        elif args.modality=='reference':
            system_prompt = f"Reference: {references}"
            modality = args.modality
        else:
            system_prompt = ""
            modality = ""


        prompt = f"<s>[INST] <<SYS>> Answer the question in the user's message. Additional information is provided as context after the user's question. <</SYS>> User message: {user_message}. Context: {system_prompt} [/INST]"
        prompts.append(prompt)
        ground_truth.append(datum['answer'])

    responses = lm.generate_responses_zs_vllm(prompts, args, max_tokens=512)

    # import bpdb; bpdb.set_trace()

    output_data = [{'response':responses[i], 'ground_truth':ground_truth[i]} for i in range(len(responses))]

    if not(os.path.exists('./outputs')):
        os.makedirs('./outputs')

    utils.write_jsonl(output_data, f"./outputs/figure_zs_responses_{modality}_.jsonl")

    
    from evaluation import evaluator

    Evaluator = evaluator(responses, ground_truth)

    Evaluator.print_results()
    
    
    
    # eval(responses, ground_truth)

    

    

    




