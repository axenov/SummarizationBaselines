""" Baseline base class."""

import os
import numpy as np
from nltk.tokenize import sent_tokenize

class Baseline(object):
    def __init__(self):
        """ 
        A Baseline is the base class for all baselines.
        """
        pass

    def run(self, documents, **kwargs):
        """
        Run the baseline for all documents.
        Args:
            documents (list(str)): documents to be process
        Return:
            sentences (list(list(str))): sentences for each document
            scores (list(list(float))): score for each sentence
        """

        raise NotImplementedError()

    def get_extractive_summaries(self, documents, num_sentences, **kwargs):
        """
        Get the extractive summary of each documents
        Args:
            documents (list(str)): documents to be process
            num_sentences (int or list(int)): number of sentences in the summary. If one value, all summaries will have the same number of sentences. If list of values, the len(list) must be equal to len(documents).
            **kwargs: arguments to pass to the run function
        Return:
            summaries (list(str)): summaries of each document 
        """

        all_sentences, all_scores = self.run(documents, **kwargs)

        if isinstance(num_sentences, int):
            num_sentences = [num_sentences for i in range(len(documents))]
        if len(num_sentences) != len(documents):
            raise ValueError("documents and num_sentences must have the same length")

        summaries = []
        for i in range(len(documents)):
            scores = np.array(all_scores[i])
            sorted_ix = np.argsort(scores)[::-1]
            summary = ' '.join([all_sentences[i][j] for j in sorted_ix[: num_sentences[i]]])
            summaries.append(summary)

        return summaries

    def generate_hypotheses(self, documents, references, num_sentences, **kwargs):
        """
        Get the extractive summary of each documents based on the reference summary length
        Args:
            documents (list(str)): documents to be process
            references (list(str)): references summaries
            num_sentences (int): number of sentences in the summary. -1 for adaptative
            **kwargs: arguments to pass to the run function
        Return:
            summaries (list(str)): summaries of each document 
        """
        if num_sentences == -1:
            num_sentences = [len(sent_tokenize(ref)) for ref in references]
        return self.get_extractive_summaries(documents, num_sentences, **kwargs)

    def write_hypotheses(self, documents, references, num_sentences, filename, write_ref=False, ref_filename=None, **kwargs):
        """
        Write the extractive summary of each documents based on the reference summary length in a file
        Args:
            documents (list(str)): documents to be process
            references (list(str)): references summaries
            filename (str): path to the output file where write hypotheses
            num_sentences (int): number of sentences in the summary. -1 for adaptative
            write_ref (bool): True if save references in ref_filename
            ref_filename (str): path to the output file where write references
            **kwargs: arguments to pass to the run function
        """

        hypotheses = self.generate_hypotheses(documents, references, num_sentences, **kwargs)
        with open(filename, 'w') as f:
            for hyp in hypotheses:
                f.write(hyp.replace('\n', '') + '\n')
        if write_ref:
            if ref_filename == None:
                raise ValueError(f"if write_ref set to True ref_filename ({ref_filename}) must be a correct path")
            with open(ref_filename, 'w') as f:
                for ref in references:
                    f.write(ref.replace('\n', '') + '\n')