from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank

# from baselines.mmr import mmr
# from baselines.tfidf import TFIDF


def use(name):
    if name == "Random":
        return Random("random")
    if name == "Lead":
        return Lead("lead")
    if name == "LexRank":
        return LexRank("lexrank")
