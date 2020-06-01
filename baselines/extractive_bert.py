from nltk.tokenize import sent_tokenize

from baselines.baseline import Baseline

import math
import logging
import sys
import os
import re
import errno

import torch
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from baselines.extractive_bert_ressources.model import Summarizer


def rename_file(path):
    if os.path.exists(path):
        output_file = path.split(".")
        output_file[-2] = f"{output_file[-2]}*"
        output_file = ".".join(output_file)
        return rename_file(output_file)
    else:
        return path


class ExtractiveBert(Baseline):

    """ Description
    Extractive summarization model from Dmitry. 
    """

    def __init__(self, name, model_folder, reg_file, bert_file, alpha=0.7):
        super().__init__(name)
        logger = logging.getLogger(__name__)
        self.model = Summarizer.from_pretrained(
            model_folder, reg_file, bert_file=bert_file, logger=logger
        )
        self.model.test()
        if torch.cuda.is_available():
            self.model.cuda()
        self.alpha = alpha

    def rank_sentences(self, dataset, document_column_name, num_sentences, **kwargs):
        def run_extractive(example):
            print("map function")

            # Data process
            document = (
                example[document_column_name]
                .replace("\n", " ")
                .replace("	", " ")
                .replace("  ", " ")
                .replace("   ", " ")
                .split("|||")
            )

            # Compute importance, sentences and mask
            importance, sentences, mask = self.model(document)
            importance = torch.sigmoid(importance).view(importance.shape[0])
            importance = [element.item() for element in importance.flatten()]

            # Order sentences
            summary_sentences = []
            while len(summary_sentences) < num_sentences:
                relevance = self.model.sentence_relevance(sentences, summary_sentences)
                final_score = [
                    self.alpha * imp - (1 - self.alpha) * rel
                    for imp, rel in zip(importance, relevance)
                ]
                idx = np.argmax(final_score)
                importance[idx] = -1
                summary_sentences.append(sentences[idx])

            # Add to new column
            example[self.name] = {
                "sentences": summary_sentences,
                "scores": list(range(1, len(summary_sentences) + 1))[::-1],
            }
            return example

        dataset = dataset.map(run_extractive)
        return dataset
