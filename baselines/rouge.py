from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import numpy as np

from baselines.baseline import Baseline


class RougeOracle(Baseline):
    def __init__(self, name, rougeType = "rouge2", rougeMethod = "recall"):
        super().__init__(name)
        self.scorer = rouge_scorer.RougeScorer(rougeType)
        self.rougeType = rougeType
        self.rougeMethod = rougeMethod

    def calculate_rouge(self,sentence1,sentence2):
        scores = self.scorer.score(sentence1, sentence2)
        method = self.rougeMethod
        ind = 2
        if method == 'recall':
            ind = 0
        elif method == 'precision':
            ind = 1
        elif method == 'f1': 
            ind =2
        rouge_score = sum([scores[score][ind] for score in self.rougeType])/len(self.rougeType)
        return rouge_score


    def rank_sentences(self, dataset, document_column_name, num_sentences, **kwargs):
        def split_sentences(document):
            texts = document.split("|||")
            senteces = []
            for text in texts:
                senteces += sent_tokenize(text)
            return senteces

        def run_extractive(example):
            sentences = split_sentences(example[document_column_name])
            reference = example[kwargs['summary_colunm_name']]
            scores = [self.calculate_rouge(sent,reference) for sent in sentences]
            # Add to new column
            example[self.name] = {
                "sentences": sentences,
                "scores": scores,
            }
            return example

        dataset = dataset.map(run_extractive)
        return dataset