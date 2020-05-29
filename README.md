# ExtractiveSumBaselines

Implementation of extractive summarization baselines. For the moment there are:

- Random: Select n sentences randomly.
- Lead: Select the n first sentences.
- LexRank: Compute similarity between sentences and select the n first sentences ranked using PageRank style algorithm

> More baselines to come

## Installation

```
git clone https://github.com/airKlizz/ExtractiveSumBaslines
cd ExtractiveSumBaslines
```

There is no ``requirements.txt`` file yet so install depedencies on demand :hugs:.

## Usage

To run baseline you first have to configure the ``run_args.json`` file with your parameters.

This repository is based on [``nlp`` library](https://github.com/huggingface/nlp) for load data and to compute ROUGE metric. 

The idea is that you have a summarization dataset (``nlp.Dataset`` class) with at least a column with texts to summarize (``document_column_name``) and one column with reference summaries (``summary_colunm_name``). Then you want to run multiple baselines on it and compare ROUGE results of these differents methods of summarization.

You can add your summarization model (extractive or abstractive) as a new baseline to compare its performance with other baselines. Go [here](#add-baseline) for more details to add a baseline.

Once you have all baselines you need and your dataset you can configure the ``run_args.json`` file.

This file is composed as follow:

```
{
    "baselines": [
        # List of baselines you want to run
        # For each baseline you need to pass the name et the number of sentences the summary must be.
        # You can also add args which are passed to the run function of the baseline
        # Example:
        {"name": "Random", "num_sentences": 10},
        {"name": "LexRank", "num_sentences": 10, "threshold": 0.03, "increase_power": true}
        {"name": "Lead", "num_sentences": 10},
    ],

    "dataset": {
        # Dataset parameters with:
        # the name of the nlp dataset or the path to the python script
        # the split of the dataset to test on
        # the cache_dir for the dataset load function
        # the name of the column containing texts to summarize
        # the name of the column containing reference summaries
        # Example:
        "name": "en_wiki_multi_news_cleaned.py",
        "split": "test",
        "cache_dir": ".en-wiki-multi-news-cache",
        "document_column_name": "document",
        "summary_colunm_name": "summary"
    },

    "run": {
        # Arguments of what you want to save of the run
        # rouge_types is a dict of rouge_type and measure for the rouge_type. See nlp metrics for more details.
        # Example:
        "hypotheses_folder": "hypotheses/",
        "csv_file": "results.csv",
        "md_file": "results.md",
        "rouge_types": {
            "rouge1": ["mid.fmeasure"],
            "rouge2": ["mid.fmeasure"],
            "rougeL": ["mid.fmeasure"]
        }
    }
}
```

Once the file is configured you can run the computation by running:

```
python run_baseline.py
```

Results are stored to the files/folder you put in the ``run_args.json`` file.

## Add baseline

If you want to add your baseline you have to create a script similar to ``baselines/lead.py`` which contain a subclass of ``Baseline`` and define the function ``run(self, dataset, document_column_name, **kwargs)``. The ``run`` function must score every sentences of each document. Then just add you baseline on the ``baselines/baselines.py`` file by adding a condition and you can use your baseline.

## Results for ``en-wiki-multi-news``

With 10 sentences in hypotheses:

|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| Random | 37.80% | 11.91% | 17.31% |
| Lead | 37.47% | 12.34% | 17.73% |
| LexRank | 40.06% | 13.86% | 18.33% |
