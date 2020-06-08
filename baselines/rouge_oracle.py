from nltk.tokenize import sent_tokenize
from nlp import load_metric
import numpy as np
from baselines.baseline import Baseline


class RougeOracle(Baseline):
    def __init__(self, name, rouge_type="rouge2", rouge_method="precision"):
        super().__init__(name)
        self.rouge_metric = load_metric("rouge")
        self.rouge_type = rouge_type
        if rouge_method == "precision":
            self.rouge_method = 0
        elif rouge_method == "recall":
            self.rouge_method = 1
        elif rouge_method == "fmeasure":
            self.rouge_method = 2
        else:
            raise ValueError('rouge_method must be "precision", "recall" or "fmeasure"')

    def _calculate_rouge(self, prediction, reference):
        score = self.rouge_metric.compute(
            [prediction],
            [reference],
            rouge_types=[self.rouge_type],
            use_agregator=False,
        )
        return score[self.rouge_type][0][self.rouge_method]

    def rank_sentences(
        self, dataset, document_column_name, summary_colunm_name, **kwargs
    ):
        all_sentences = list(map(sent_tokenize, dataset[document_column_name]))
        summaries = dataset[summary_colunm_name]
        all_scores = [
            self._calculate_rouge(sentence, summary)
            for sentences, summary in zip(all_sentences, summaries)
            for sentence in sentences
        ]

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, all_scores)
        ]
        return Baseline.append_column(dataset, data, self.name)
