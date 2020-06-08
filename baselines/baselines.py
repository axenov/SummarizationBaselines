from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank
from baselines.bart import Bart
from baselines.t5 import T5
from baselines.rouge import RougeOracle
from baselines.rouge_greedy import RougeOracleGreedy
from baselines.extractive_bert import ExtractiveBert
from baselines.textrank import TextRank
from baselines.tfidf import TFIDF


def use(baseline_class, **init_kwargs):
    if baseline_class == "Random":
        return Random(**init_kwargs)
    if baseline_class == "Lead":
        return Lead(**init_kwargs)
    if baseline_class == "RougeOracle":
        return RougeOracle(**init_kwargs)
    if baseline_class == "RougeOracleGreedy":
        return RougeOracleGreedy(**init_kwargs)
    if baseline_class == "LexRank":
        return LexRank(**init_kwargs)
    if baseline_class == "TextRank":
        return TextRank(**init_kwargs)
    if baseline_class == "TFIDF":
        return TFIDF(**init_kwargs)
    if baseline_class == "Bart":
        return Bart(**init_kwargs)
    if baseline_class == "T5":
        return T5(**init_kwargs)
    if baseline_class == "Extractive Bert":
        return ExtractiveBert(**init_kwargs)
    raise ValueError("Baseline baseline_class not correct")
