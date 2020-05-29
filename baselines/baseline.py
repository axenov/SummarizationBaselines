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

    def run(self, dataset, document_column_name, **kwargs):
        """
        Run the baseline for all documents.
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
        Return:
            dataset (nlp.Dataset): dataset with a new column containing sentences and scores.
        """

        raise NotImplementedError()

    def get_summaries(self, dataset, document_column_name, **kwargs):
        """
        Get the extractive summary of each documents
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            **kwargs: arguments to pass to the run function or get_summaries
        Return:
            dataset (nlp.Dataset): dataset with a new column for hypothesis
        """
        dataset = self.run(dataset, document_column_name, **kwargs)
        num_sentences = kwargs["num_sentences"]

        if isinstance(num_sentences, int):
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

    def generate_hypotheses(
        self, dataset, document_column_name, summary_colunm_name, **kwargs,
    ):
        """
        Get the extractive summary of each documents based on the reference summary length
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            summary_colunm_name (str): name of the column of the dataset containing summaries
            **kwargs: arguments to pass to the run function
        Return:
            summaries (list(str)): summaries of each document 
        """
        return self.get_summaries(dataset, document_column_name, **kwargs)

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

        dataset = self.generate_hypotheses(
            dataset, document_column_name, summary_colunm_name, **kwargs
        )

        rouge_metric = load_metric("rouge")

        def compute_rouge_batch(example):
            predictions = example[f"{self.name}_hypothesis"]
            references = example["summary"]
            predictions = list(map(lambda s: " ".join(word_tokenize(s)), predictions))
            references = list(map(lambda s: " ".join(word_tokenize(s)), references))
            rouge_metric.add(predictions, references)

        dataset.map(compute_rouge_batch, batched=True)
        return dataset, rouge_metric.compute(rouge_types=rouge_types)

    def write_hypotheses(
        self,
        dataset,
        document_column_name,
        summary_colunm_name,
        filename,
        write_ref=False,
        ref_filename=None,
        **kwargs,
    ):
        """
        Write the extractive summary of each documents based on the reference summary length in a file
        Args:
            dataset (nlp.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            summary_colunm_name (str): name of the column of the dataset containing summaries
            filename (str): path to the output file where write hypotheses
            write_ref (bool): True if save references in ref_filename
            ref_filename (str): path to the output file where write references
            **kwargs: arguments to pass to the run function
        """

        dataset = self.generate_hypotheses(
            dataset, document_column_name, summary_colunm_name, **kwargs
        )
        with open(filename, "w") as f:
            for hyp in dataset[f"{self.name}_hypothesis"]:
                f.write(hyp.replace("\n", "") + "\n")
        if write_ref:
            if ref_filename == None:
                raise ValueError(
                    f"if write_ref set to True ref_filename ({ref_filename}) must be a correct path"
                )
            with open(ref_filename, "w") as f:
                for ref in dataset[summary_colunm_name]:
                    f.write(ref.replace("\n", "") + "\n")

    @staticmethod
    def append_column(dataset, data, column_name):
        data = pa.array(data)
        dataset._data = dataset.data.append_column(column_name, data)
        return dataset
