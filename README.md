# Summarization Baselines

Implementation of extractive summarization baselines. For the moment there are:

- Random: Select n sentences randomly.
- Lead: Select the n first sentences.
- LexRank: Compute similarity between sentences and select the n first sentences ranked using PageRank style algorithm
- Bart: Transformer model. Implementation thanks [huggingface](https://huggingface.co/).
- T5: Transformer model. Implementation thanks [huggingface](https://huggingface.co/).

> More baselines to come

## Installation

```
git clone https://github.com/airKlizz/SummarizationBaselines
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
        {"baseline_class": "Random", "init_kwargs": {"name": "Random"}, "run_kwargs": {"num_sentences": 10}},
        {"baseline_class": "Lead", "init_kwargs": {"name": "Lead"}, "run_kwargs": {"num_sentences": 10}},
        {"baseline_class": "LexRank", "init_kwargs": {"name": "LexRank"}, "run_kwargs": {"num_sentences": 10, "threshold": 0.03, "increase_power": true}},
        {
            "baseline_class": "Bart", 
            "init_kwargs": {
                "name": "Bart CNN",
                "model_name": "bart-large-cnn",
                "input_max_length": 512,
                "device": "cuda",
                "batch_size": 8
            }, 
            "run_kwargs": {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 400,
                "min_length": 200,
                "no_repeat_ngram_size": 3,
                "early_stopping": true
            }
        },
        {
            "baseline_class": "T5", 
            "init_kwargs": {
                "name": "T5 base",
                "model_name": "t5-base",
                "input_max_length": 512,
                "device": "cuda",
                "batch_size": 8
            }, 
            "run_kwargs": {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 400,
                "min_length": 200,
                "no_repeat_ngram_size": 3,
                "early_stopping": true
            }
        },
        {
            "baseline_class": "T5", 
            "init_kwargs": {
                "name": "T5 fine tuned",
                "model_name": ["t5-base", "/content/drive/My Drive/Colab Notebooks/Multi-wiki-news/English/t5-wild-glitter-2"],
                "input_max_length": 512,
                "device": "cuda",
                "batch_size": 8
            }, 
            "run_kwargs": {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 400,
                "min_length": 200,
                "no_repeat_ngram_size": 3,
                "early_stopping": true
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

If you want to add your baseline you have to create a script similar to ``baselines/lead.py`` for extractive baseline or ``baselines/bart.py`` for abstractive baseline which contain a subclass of ``Baseline`` and define the function ``def run(self, dataset, document_column_name, **kwargs)`` or ``def get_summaries(self, dataset, document_column_name, **kwargs)``. 

For extractive baseline, the function ``run`` ranks all sentences of each document and add scores and sentences in a new column of the dataset. It returns the dataset.

For abstractive baseline, the function ``get_summaries`` summaries each document and add summaries (also called hypotheses) in a new column of the dataset. It returns the dataset.

Then just add you baseline on the ``baselines/baselines.py`` file by adding a ``if`` and you can use your baseline.

## Results for ``en-wiki-multi-news``

|     | rouge1.mid.fmeasure | rouge2.mid.fmeasure | rougeL.mid.fmeasure |
| --- | --- | --- | --- |
| Random | 37.84% | 11.95% | 17.30% |
| Lead | 37.48% | 12.34% | 17.75% |
| LexRank | 40.08% | 13.82% | 18.33% |
| Bart CNN | 37.44% | 12.00% | 18.55% |
| T5 base | 22.90% | 6.82% | 12.92% |
| T5 fine tuned | 41.38% | 16.14% | 22.58% |

> Table obtained by running the example ``run_args.json`` file and the table is automatically generated in ``results.md``.
