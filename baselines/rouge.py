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
            sentences_index = list(range(len(sentences)))
            reference = example[kwargs['summary_colunm_name']]
            scores = [self.calculate_rouge(sent,reference) for sent in sentences]

            # Order sentences
            summary_sentences = []
            summary_sentences_indexes = []

            while len(summary_sentences) < min(len(sentences),num_sentences):
                idx = np.argmax(scores)
                test_sentence = sentences[idx]
                summary_sentences.append(test_sentence)
                summary_sentences_indexes.append(sentences_index[idx])
                scores.pop(idx)
                sentences.pop(idx)
                sentences_index.pop(idx)

            # Add to new column
            summary_sentences_scores = [max(summary_sentences_indexes)-x for x in summary_sentences_indexes]
            #summary_sentences_scores = list(range(1, len(summary_sentences) + 1))[::-1]
            example[self.name] = {
                "sentences": summary_sentences,
                "scores": summary_sentences_scores,
            }
            return example

        dataset = dataset.map(run_extractive)
        return dataset