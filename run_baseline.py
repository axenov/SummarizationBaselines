from baselines.baselines import use
from nlp import load_dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--baseline_name", type=str, help="Name of the baseline", default="Random")
parser.add_argument("--filename", type=str, help="Path to the file where save hypotheses", default="hyp.txt")
parser.add_argument("--num_to_process", type=int, help="Number of examples to process. -1 for all", default=-1)
parser.add_argument("--write_ref", type=bool, help="True if save references in ref_filename", default=False)
parser.add_argument("--ref_filename", type=str, help="Path to the file where save references", default="ref.txt")
parser.add_argument("--num_sentences", type=int, help="Number of sentences in the summary. -1 for adaptative", default=-1)


args = parser.parse_args()
num_to_process = args.num_to_process
if num_to_process == -1:
    num_to_process = None

# Init Baseline
bl = use(args.baseline_name)

# Load dataset
test_dataset = load_dataset('en_wiki_multi_news_cleaned.py', split='test', cache_dir='.en-wiki-multi-news-cache')

bl.write_hypotheses(test_dataset['document'][:num_to_process], test_dataset['summary'][:num_to_process], args.num_sentences, args.filename, args.write_ref, args.ref_filename)
