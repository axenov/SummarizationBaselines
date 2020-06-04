from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

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

    def rank_sentences(self, dataset, document_column_name, **kwargs):
        def split_sentences(document):
            texts = document.split("|||")
            senteces = []
            for text in texts:
                senteces += sent_tokenize(text)
            return senteces
        all_sentences = list(map(split_sentences, dataset[document_column_name]))
        all_summaries = dataset[kwargs['summary_colunm_name']]
        scores =  [[self.calculate_rouge(sent,summary) for sent in sentences] for sentences,summary in zip(all_sentences,all_summaries)]
        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, scores)
        ]
        return Baseline.append_column(dataset, data, self.name)
