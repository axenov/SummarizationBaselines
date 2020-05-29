from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank
from baselines.bart import Bart

# from baselines.mmr import mmr
# from baselines.tfidf import TFIDF


def use(name, **init_kwargs):
    if name == "Random":
        return Random("random", **init_kwargs)
    if name == "Lead":
        return Lead("lead", **init_kwargs)
    if name == "LexRank":
        return LexRank("lexrank", **init_kwargs)
    if name == "Bart":
        return Bart("bart", **init_kwargs)
    raise ValueError("Baseline name not correct")
