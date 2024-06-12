import evaluate
import numpy as np
import json

def rouge_scores(hypotheses, references):
    rouge = evaluate.load('rouge')
    
    results = rouge.compute(predictions=hypotheses, references=references)
    return results


def bleu_scores(hypotheses, references):
    bleu = evaluate.load('bleu')
    
    results = bleu.compute(predictions=hypotheses, references=references)
    return results['bleu']


def meteor_scores(hypotheses, references):
    meteor = evaluate.load('meteor')
    
    results = meteor.compute(predictions=hypotheses, references=references)
    return results['meteor']


def bert_score(hypotheses, references):
    bert = evaluate.load('bertscore')
    
    results = bert.compute(predictions=hypotheses, references=references, lang='en', verbose=False)
    return np.mean(results['f1'])




class evaluator():
    def __init__(self, hypotheses, references):
        self.hypotheses = hypotheses
        self.references = references

    def rouge_scores(self):
        rouge = evaluate.load('rouge')
        
        results = rouge.compute(predictions=self.hypotheses, references=self.references)
        return results
    

    def bleu_scores(self):
        bleu = evaluate.load('bleu')
        
        results = bleu.compute(predictions=self.hypotheses, references=self.references)
        return results['bleu']
    

    def meteor_scores(self):
        meteor = evaluate.load('meteor')
        
        results = meteor.compute(predictions=self.hypotheses, references=self.references)
        return results['meteor']
    

    def bert_score(self):
        bert = evaluate.load('bertscore')
        
        results = bert.compute(predictions=self.hypotheses, references=self.references, lang='en', verbose=False)
        return np.mean(results['f1'])
    

    def print_results(self):
        rouge = self.rouge_scores()
        bleu = self.bleu_scores()
        meteor = self.meteor_scores()
        bert = self.bert_score()

        print(f"Rouge scores: {rouge}")
        print(f"Bleu scores: {bleu}")
        print(f"Meteor scores: {meteor}")
        print(f"Bert score: {bert}")


    def print_results2(self, type):
        rouge = self.rouge_scores()
        bleu = self.bleu_scores()
        meteor = self.meteor_scores()
        bert = self.bert_score()

        print(f"Rouge scores: {rouge}")
        print(f"Bleu scores: {bleu}")
        print(f"Meteor scores: {meteor}")
        print(f"Bert score: {bert}")


        # Save results as JSON
        filename = f"./results/ft_{type}.json"
        # Extract required scores
        rouge1 = rouge['rouge1']
        rouge2 = rouge['rouge2']
        rougeL = rouge['rougeL']

        # Combine all scores into a dictionary
        results = {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "bleu": bleu,
            "meteor": meteor,
            "bert": bert
        }

        # Write the results to the JSON file in append mode
        with open(filename, "a") as file:
            file.write(json.dumps(results) + "\n")