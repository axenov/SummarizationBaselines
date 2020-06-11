# Summarization Baselines

Implementation of extractive summarization baselines. For the moment there are:

- Random: Select n sentences randomly.
- Lead: Select the n first sentences.
- LexRank: Compute similarity between sentences and select the n first sentences ranked using PageRank style algorithm
- Bart: Transformer model. Implementation thanks [huggingface](https://huggingface.co/).
- T5: Transformer model. Implementation thanks [huggingface](https://huggingface.co/).
- ExtractiveBert: Using Bert to rank sentences.

> More baselines to come

## Installation

```
git clone -b extractive-bert https://github.com/airKlizz/SummarizationBaselines
cd SummarizationBaselines
```

There is no ``requirements.txt`` file yet so install depedencies on demand :hugs:.

## Usage

To run baseline you first have to configure the ``run_args.json`` file with your parameters.

This repository is based on [``nlp`` library](https://github.com/huggingface/nlp) for load data and to compute ROUGE metric. 

The idea is that you have a summarization dataset (``nlp.Dataset`` class) with at least a column with texts to summarize (``document_column_name``) and one column with reference summaries (``summary_colunm_name``). Then you want to run multiple baselines on it and compare ROUGE results of these differents methods of summarization.

You can add your summarization model (extractive or abstractive) as a new baseline to compare its performance with other baselines. Go [here](#add-baseline) for more details to add a baseline.

Once you have all baselines you need and your dataset you can configure the ``run_args.json`` file.

This is an example of a ``run_args.json`` file:

```json
{
    "baselines": [
        {"baseline_class": "Random", "init_kwargs": {"name": "Random"}, "run_kwargs": {"num_sentences": 12, "non_redundant": true, "ordering": true}},
        {"baseline_class": "Lead", "init_kwargs": {"name": "Lead"}, "run_kwargs": {"num_sentences": 12, "non_redundant": true, "ordering": true}},
        {"baseline_class": "LexRank", "init_kwargs": {"name": "LexRank"}, "run_kwargs": {"num_sentences": 12, "non_redundant": true, "ordering": true}},
        {"baseline_class": "TFIDF", "init_kwargs": {"name": "TF-IDF"}, "run_kwargs": {"num_sentences": 12, "title_column_name": "title", "non_redundant": true, "ordering": true}},
        {
            "baseline_class": "Extractive Bert", 
            "init_kwargs": {
                "name": "Extractive Bert", 
                "model_folder": "/content/drive/My Drive/Github/SummarizationBaselines/models/test-rouge2-wiki-bert/", 
                "reg_file": "reg_test-rouge2-wiki-bert_400.pt", 
                "bert_file": null, 
                "alpha": 0.7
            }, 
            "run_kwargs": {
                "num_sentences": 10,
                "non_redundant": true,
                "ordering": true
            }
        }       
    ],

    "dataset": {
        "name": "en_wiki_multi_news_cleaned.py",
        "split": "test",
        "cache_dir": ".en-wiki-multi-news-cache",
        "document_column_name": "document",
        "summary_colunm_name": "summary"
    },

    "run": {
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

The file is composed of 3 arguments:

- ``baselines``: it defines all baselines you want to compare with for each the associate ``class``, ``init_kwargs`` which are arguments pass to the ``init`` function of the ``class`` and ``run_kwargs`` which are arguments pass to the run function,
- ``dataset``: it defines dataset's arguments with the ``name`` which is the name of the ``nlp`` dataset or the path to the dataset python script, the ``split`` and the ``cache_dir`` of the dataset (see [nlp](https://github.com/huggingface/nlp) ``load_dataset`` function), ``document_column_name`` which is the name of the column in the dataset containing the texts to summarize and ``summary_column_name`` which is the name of the column in the dataset containing the summaries,
- ``run``: it defines the ROUGE run arguments with the ``folder`` to save hypotheses, optionnal ``csv_file`` and ``md_file`` to save results to the corresponding format and ``rouge_types`` which are the type of ROUGE scores to compute (see [nlp](https://github.com/huggingface/nlp) ``rouge`` metric). 

Once the file is configured you can run the computation by running:

```
python run_baseline.py
```

Results are stored to the files/folder you put in the ``run_args.json`` file.

## Add baseline

If you want to add your baseline you have to create a script similar to ``baselines/lead.py`` for extractive baseline or ``baselines/bart.py`` for abstractive baseline which contain a subclass of ``Baseline`` and define the function ``def rank_sentences(self, dataset, document_column_name, **kwargs)`` or ``def get_summaries(self, dataset, document_column_name, **kwargs)``. 

For extractive baseline, the function ``rank_sentences`` ranks all sentences of each document and add scores and sentences in a new column of the dataset. It returns the dataset.

For abstractive baseline, the function ``get_summaries`` summaries each document and add summaries (also called hypotheses) in a new column of the dataset. It returns the dataset.

Then just add you baseline on the ``baselines/baselines.py`` file by adding a ``if`` and you can use your baseline.

## Results for ``en-wiki-multi-news``

### Comparison of different extractive baselines
|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| Lead | 41.91% | 15.98% | 21.66% |
| Random | 39.08% | 12.80% | 17.51% |
| Random - sorted| 39.08% | 12.83% | 18.45% |
| LexRank | 39.47% | 14.18% | 18.13% |
| LexRank - sorted| 39.47% | 14.21% | 19.05% |
| LexRank - sorted non-redundant| 40.43% | 14.72% | 19.27% |
| TextRank | 39.34% | 14.37% | 18.54% |
| TextRank - sorted| 39.31% | 14.37% | 19.36% |
| TextRank - sorted non-redundant| 40.50% | 14.91% | 19.62% |
| TF-IDF | 40.28% | 14.19% | 18.74% |
| TF-IDF - sorted | 40.28% | 14.20% | 19.37% |
| TF-IDF - sorted non-redundant| 41.36% | 14.69% | 19.64% |


### Oracle ordered baseline
|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| RougeOracle-rouge1-recall | 44.75% | 19.72% | 23.05% |
| RougeOracle-rouge1-precision | 50.66% | 26.54% | 28.95% |
| RougeOracle-rouge1-f1 | 45.60% | 20.32% | 23.56% |
| RougeOracle-rouge2-recall | 48.99% | 25.70% | 27.17% |
| RougeOracle-rouge2-precision | 49.71% | 27.83% | 28.99% |
| RougeOracle-rouge2-f1 | 49.73% | 26.34% | 27.76% |
| RougeOracle-rougeL-recall | 46.04% | 21.82% | 24.66% |
| RougeOracle-rougeL-precision | 48.79% | 26.20% | 28.91% |
| RougeOracle-rougeL-f1 | 47.06% | 22.64% | 25.43% |
| RougeOracleGreedy-rouge2-recall | 49.62% | 26.73% | 28.20% |
| RougeOracleGreedy-rouge2-recall (unsorted) | 49.62% | 26.63% | 23.00% |
| Extractive Bert (old) | 38.33% | 12.13% | 16.95% |
| Extractive Bert (old sorted) | 38.42% | 12.12% | 17.76% |


# Final results
## Results for ``en-wiki-multi-news`` 12 sentences
|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| Random | 39.08% | 12.80% | 17.51% |
| Lead | 41.91% | 15.98% | 21.66% |
| LexRank | 40.43% | 14.72% | 19.27% |
| TextRank | 40.50% | 14.91% | 19.62% |
| TF-IDF | 41.36% | 14.69% | 19.64% |
| RougeOracle-rouge2-precision | 49.67% | 27.20% | 28.32% |
## Results for ``de-wiki-multi-news`` 12 sentences
|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| Random | 31.38% | 7.95% | 13.18% |
| Lead | 34.46% | 10.37% | 16.13% |
| LexRank | 35.31% | 10.77% | 15.79% |
| TextRank | 33.47% | 9.84% | 15.25% |
| TF-IDF | 34.54% | 10.26% | 15.69% |
| RougeOracle-rouge2-precision | 41.87% | 19.17% | 21.86% |
## Results for ``fr-wiki-multi-news`` 8 sentences
|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| Random | 28.58% | 9.78% | 14.28% |
| Lead | 30.52% | 11.98% | 16.79% |
| LexRank | 29.80% | 10.69% | 15.22% |
| TextRank | 29.97% | 11.16% | 15.93% |
| TF-IDF | 30.80% | 11.00% | 15.93% |
| RougeOracle-rouge2-precision | 34.45% | 16.77% | 20.07% |
> Table obtained by running the example ``run_args.json`` file and the table is automatically generated in ``results.md``.
