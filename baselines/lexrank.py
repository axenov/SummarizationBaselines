import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from baselines.baseline import Baseline


class LexRank(Baseline):

    """ Description from https://arxiv.org/pdf/1905.13164.pdf
    LexRank (Erkan and Radev, 2004) is a widely-used graph-based extractive summarizer; 
    we build a graph with paragraphs as nodes andedges weighted by tf-idf cosine similarity; 
    we run a PageRank-like algorithm on this graph to rank and select paragraphs until 
    the length of the ground-truth summary is reached.
    """

    """ Implementation
    Most of the code is from https://github.com/crabcamp/lexrank
    """

    def run(self, dataset, document_column_name, threshold=0.03, increase_power=True):
        all_sentences = []
        all_scores = []
        for document in dataset[document_column_name]:
            sentences, scores = self.run_single(document, threshold, increase_power)
            all_sentences.append(sentences)
            all_scores.append(scores)

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, all_scores)
        ]
        return Baseline.append_column(dataset, data, self.name)

    def run_single(self, document, threshold=0.03, increase_power=True):

        sentences = sent_tokenize(document)

        # Run tf-idf cosine similarity
        vectorizer = TfidfVectorizer()
        documents_vector = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(documents_vector)

        # Run PageRank
        markov_matrix = self.create_markov_matrix_discrete(similarity_matrix, threshold)
        scores = self.stationary_distribution(
            markov_matrix, increase_power=increase_power,
        )

        return sentences, list(scores)

    @staticmethod
    def create_markov_matrix_discrete(weights_matrix, threshold):
        discrete_weights_matrix = np.zeros(weights_matrix.shape)
        ixs = np.where(weights_matrix >= threshold)
        discrete_weights_matrix[ixs] = 1
        n_1, n_2 = discrete_weights_matrix.shape
        if n_1 != n_2:
            raise ValueError("'weights_matrix' should be square")
        row_sum = discrete_weights_matrix.sum(axis=1, keepdims=True)
        return discrete_weights_matrix / row_sum

    @staticmethod
    def stationary_distribution(transition_matrix, increase_power=True):
        n_1, n_2 = transition_matrix.shape
        if n_1 != n_2:
            raise ValueError("'transition_matrix' should be square")
        distribution = np.zeros(n_1)
        grouped_indices = LexRank.connected_nodes(transition_matrix)
        for group in grouped_indices:
            t_matrix = transition_matrix[np.ix_(group, group)]
            eigenvector = LexRank._power_method(t_matrix, increase_power=increase_power)
            distribution[group] = eigenvector
        return distribution

    @staticmethod
    def _power_method(transition_matrix, increase_power=True):
        eigenvector = np.ones(len(transition_matrix))
        if len(eigenvector) == 1:
            return eigenvector
        transition = transition_matrix.transpose()
        while True:
            eigenvector_next = np.dot(transition, eigenvector)
            if np.allclose(eigenvector_next, eigenvector):
                return eigenvector_next
            eigenvector = eigenvector_next
            if increase_power:
                transition = np.dot(transition, transition)

    @staticmethod
    def connected_nodes(matrix):
        _, labels = connected_components(matrix)
        groups = []
        for tag in np.unique(labels):
            group = np.where(labels == tag)[0]
            groups.append(group)
        return groups
