from nltk.tokenize import sent_tokenize

from baselines.baseline import Baseline

try:
    from baselines.baselines import use
except ImportError:
    from baselines import use


class Combine(Baseline):

    """ Description 
    Class which can combine an extractive and abstractive baselines by performing the abstractive baseline on sentences ordered by the extractive one.
    """

    def __init__(
        self,
        name,
        extractive_class,
        abstractive_class,
        extractive_args,
        abstractive_args,
    ):
        super().__init__(name)
        self.extractive = use(extractive_class, **extractive_args)
        self.abstractive = use(abstractive_class, **abstractive_args)

    def get_summaries(
        self, dataset, document_column_name, extractive_args, abstractive_args
    ):
        # Extractive step
        dataset = self.extractive.get_summaries(
            dataset, document_column_name, **extractive_args
        )

        # Abstractive step
        self.abstractive.name = self.name
        dataset = self.abstractive.get_summaries(
            dataset, f"{self.extractive.name}_hypothesis", **abstractive_args
        )

        return dataset
