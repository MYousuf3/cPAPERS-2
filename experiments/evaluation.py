import numpy as np
import json
import nltk
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
from nltk.translate import meteor_score

def rouge_scores(hypotheses, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for hyp, ref in zip(hypotheses, references):
        score = scorer.score(ref, hyp)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure
    
    # Average the scores
    n = len(hypotheses)
    scores = {k: v/n for k, v in scores.items()}
    return scores


def bleu_scores(hypotheses, references):
    import sacrebleu
    
    # sacrebleu expects a list of references for each hypothesis
    references = [[ref] for ref in references]
    score = sacrebleu.corpus_bleu(hypotheses, references)
    return score.score / 100.0  # Convert to 0-1 scale to match original scale


def meteor_scores(hypotheses, references):
    scores = []
    for hyp, ref in zip(hypotheses, references):
        # Tokenize the hypothesis and reference
        hyp_tokens = nltk.word_tokenize(hyp)
        ref_tokens = nltk.word_tokenize(ref)
        score = meteor_score.meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)
    return np.mean(scores)


def bert_score(hypotheses, references):
    import bert_score
    
    P, R, F1 = bert_score.score(hypotheses, references, lang='en', verbose=False)
    return F1.mean().item()




class evaluator():
    def __init__(self, hypotheses, references):
        self.hypotheses = hypotheses
        self.references = references

    def rouge_scores(self):
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        for hyp, ref in zip(self.hypotheses, self.references):
            score = scorer.score(ref, hyp)
            scores['rouge1'] += score['rouge1'].fmeasure
            scores['rouge2'] += score['rouge2'].fmeasure
            scores['rougeL'] += score['rougeL'].fmeasure
        
        # Average the scores
        n = len(self.hypotheses)
        scores = {k: v/n for k, v in scores.items()}
        return scores
    

    def bleu_scores(self):
        import sacrebleu
        
        # sacrebleu expects a list of references for each hypothesis
        references = [[ref] for ref in self.references]
        score = sacrebleu.corpus_bleu(self.hypotheses, references)
        return score.score / 100.0  # Convert to 0-1 scale to match original scale
    

    def meteor_scores(self):
        scores = []
        for hyp, ref in zip(self.hypotheses, self.references):
            # Tokenize the hypothesis and reference
            hyp_tokens = nltk.word_tokenize(hyp)
            ref_tokens = nltk.word_tokenize(ref)
            score = meteor_score.meteor_score([ref_tokens], hyp_tokens)
            scores.append(score)
        return np.mean(scores)
    

    def bert_score(self):
        import bert_score
        
        P, R, F1 = bert_score.score(self.hypotheses, self.references, lang='en', verbose=False)
        return F1.mean().item()
    

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