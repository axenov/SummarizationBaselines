from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank
from baselines.bart import Bart
from baselines.t5 import T5

# from baselines.mmr import mmr
# from baselines.tfidf import TFIDF


def use(name, **init_kwargs):
    if name == "Random":
        return Random(**init_kwargs)
    if name == "Lead":
        return Lead(**init_kwargs)
    if name == "LexRank":
        return LexRank(**init_kwargs)
    if name == "Bart":
        return Bart(**init_kwargs)
    if name == "T5":
        return T5(**init_kwargs)
    raise ValueError("Baseline name not correct")
