from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank
from baselines.bart import Bart
from baselines.t5 import T5
from baselines.t5_with_title import T5WithTitle
from baselines.combine import Combine


def use(baseline_class, **init_kwargs):
    if baseline_class == "Random":
        return Random(**init_kwargs)
    if baseline_class == "Lead":
        return Lead(**init_kwargs)
    if baseline_class == "LexRank":
        return LexRank(**init_kwargs)
    if baseline_class == "Bart":
        return Bart(**init_kwargs)
    if baseline_class == "T5":
        return T5(**init_kwargs)
    if baseline_class == "T5 with title":
        return T5WithTitle(**init_kwargs)
    if baseline_class == "Combine":
        return Combine(**init_kwargs)
    raise ValueError("Baseline baseline_class not correct")
