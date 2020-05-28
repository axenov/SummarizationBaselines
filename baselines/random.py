from nltk.tokenize import sent_tokenize
from random import random

from baselines.baseline import Baseline


class Random(Baseline):

    """ Description 
    Give a random score to all sentences
    """

    def run(self, documents):
        all_sentences = list(map(sent_tokenize, documents))
        scores = [[random() for sentence in sentences] for sentences in all_sentences]
        return all_sentences, scores
