from nltk.tokenize import sent_tokenize

from baselines.baseline import Baseline


class Lead(Baseline):

    """ Description from https://arxiv.org/pdf/1905.13164.pdf
    Lead is a simple baseline that concatenates the title and ranked paragraphs,  
    and extracts the first k tokens;  
    We set k to the length of the ground-truth target.
    """

    def run(self, dataset, document_column_name):
        all_sentences = list(map(sent_tokenize, dataset[document_column_name]))
        scores = [[1 for sentence in sentences] for sentences in all_sentences]

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, scores)
        ]
        return Baseline.append_column(dataset, data, self.name)
