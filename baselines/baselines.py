from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank

# from baselines.mmr import mmr
# from baselines.tfidf import TFIDF


def use(name):
    if name == "Random":
        return Random()
    if name == "Lead":
        return Lead()
    if name == "LexRank":
        return LexRank()
