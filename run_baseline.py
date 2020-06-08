from baselines.baselines import use
import utils
from nlp import load_dataset
from pathlib import Path

from parser import parse_json_file

args = parse_json_file("run_args_extractives.json")

# Load dataset
dataset = load_dataset(
    args.dataset.name, split=args.dataset.split, cache_dir=args.dataset.cache_dir
)

# Compute baselines
scores = {}
for baseline in args.baselines:
    print(f"Compute {baseline.baseline_class}...")
    dataset, score = use(baseline.baseline_class, **baseline.init_kwargs).compute_rouge(
        dataset,
        args.dataset.document_column_name,
        args.dataset.summary_colunm_name,
        list(args.run.rouge_types.keys()),
        **baseline.run_kwargs,
    )
    scores[baseline.init_kwargs["name"]] = score

# Save results
Path(args.run.hypotheses_folder).mkdir(parents=True, exist_ok=True)
utils.write_references(
    dataset, args.run.hypotheses_folder, args.dataset.summary_colunm_name
)
utils.write_hypotheses(dataset, args.run.hypotheses_folder)
if args.run.csv_file != None:
    utils.write_csv(scores, args.run.csv_file, args.run.rouge_types)
if args.run.md_file != None:
    utils.write_md(scores, args.run.md_file, args.run.rouge_types)
