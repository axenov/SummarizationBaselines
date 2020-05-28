from nltk.tokenize import sent_tokenize

from baselines.baseline import Baseline

class Lead(Baseline):

    """ Description from https://arxiv.org/pdf/1905.13164.pdf
    Lead is a simple baseline that concatenates the title and ranked paragraphs,  
    and extracts the first k tokens;  
    We set k to the length of the ground-truth target.
    """

    def run(self, documents):
        all_sentences = list(map(sent_tokenize, documents))
        scores = [[1 for sentence in sentences] for sentences in all_sentences]
        return all_sentences, scores
