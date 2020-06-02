""" Baseline base class."""

import os
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nlp import load_metric
import pyarrow as pa


class Baseline(object):
    def __init__(self, name):
        """ 
        A Baseline is the base class for all baselines.
        """
        self.name = name.replace(" ", "-").lower()

    def rank_sentences(self, dataset, document_column_name, **kwargs):
        """
        Run the extractive baseline for all documents by associating a score to each sentences.
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
        Return:
            dataset (nlp.Dataset): dataset with a new column containing sentences and scores.
        """

        raise NotImplementedError()

    def get_summaries(self, dataset, document_column_name, **kwargs):
        """
        Get the summary of each documents.
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            **kwargs: arguments to pass to the run function
        Return:
            dataset (nlp.Dataset): dataset with a new column for hypothesis
        """
        dataset = self.rank_sentences(dataset, document_column_name, **kwargs)
        num_sentences = kwargs["num_sentences"]

        if isinstance(num_sentences, int) or num_sentences == None:
            num_sentences = [num_sentences for i in range(len(dataset))]
        if len(num_sentences) != len(dataset):
            raise ValueError("documents and num_sentences must have the same length")

        dataset = Baseline.append_column(dataset, num_sentences, "num_sentences")

        def get_extractive_summary(example):
            scores = np.array(example[self.name]["scores"])
            sentences = example[self.name]["sentences"]
            sorted_ix = np.argsort(scores)[::-1]
            hyp = " ".join(
                [sentences[j] for j in sorted_ix[: example["num_sentences"]]]
            )
            example[f"{self.name}_hypothesis"] = hyp
            return example

        dataset = dataset.map(get_extractive_summary)
        dataset.drop("num_sentences")
        return dataset

    def compute_rouge(
        self,
        dataset,
        document_column_name,
        summary_colunm_name,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        **kwargs,
    ):
        """
        Generate hypotheses and compute ROUGE score between summaries and hypotheses
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            summary_colunm_name (str): name of the column of the dataset containing summaries
            rouge_types (lst(str)): list of ROUGE types you want to compute
            **kwargs: arguments to pass to the run function
        Return:
            score (dict(Score)): dict of ROUGE types with the score (see nlp metrics for details)
        """

        dataset = self.get_summaries(dataset, document_column_name, **kwargs)

        rouge_metric = load_metric("rouge")

        def compute_rouge_batch(example):
            predictions = example[f"{self.name}_hypothesis"]
            references = example[summary_colunm_name]
            predictions = list(map(lambda s: " ".join(word_tokenize(s)), predictions))
            references = list(map(lambda s: " ".join(word_tokenize(s)), references))
            rouge_metric.add_batch(predictions, references)

        dataset.map(compute_rouge_batch, batched=True)
        return dataset, rouge_metric.compute(rouge_types=rouge_types)

    @staticmethod
    def append_column(dataset, data, column_name):
        data = pa.array(data)
        dataset._data = dataset.data.append_column(column_name, data)
        return dataset
