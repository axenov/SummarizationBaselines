from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

from baselines.baseline import Baseline


class Bart(Baseline):

    """ Description 
    Bart model from HuggingFace fine-tuned on cnn
    """

    def __init__(self, name, model_name, input_max_length, device, batch_size):
        super().__init__(name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.input_max_length = input_max_length
        self.device = device
        self.batch_size = batch_size

    def get_summaries(
        self, dataset, document_column_name, **kwargs,
    ):
        dataset = self.prepare_dataset(dataset, document_column_name)

        def add_abstractive_summary(example_batch):
            hypotheses_toks = model.generate(
                input_ids=example_batch["input_ids"].to(device),
                attention_mask=example_batch["attention_mask"].to(device),
                decoder_start_token_id=model.config.eos_token_id,
                **kwargs,
            )
            hypotheses = [
                self.tokenizer.decode(
                    toks, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for toks in hypotheses_toks
            ]
            return {f"{self.name}_hypothesis": hypotheses}

        dataset.map(add_abstractive_summary, batched=True, batch_size=self.batch_size)
        dataset.reset_format()
        return dataset

    def prepare_dataset(self, dataset, document_column_name):
        def convert_to_features(
            example_batch,
            input_max_length=self.input_max_length,
            document_column_name=document_column_name,
        ):
            input_encodings = self.tokenizer.batch_encode_plus(
                example_batch[document_column_name],
                pad_to_max_length=True,
                max_length=input_max_length,
            )
            encodings = {
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
            }
            return encodings

        dataset = dataset.map(convert_to_features, batched=True)
        columns = ["input_ids", "attention_mask"]
        dataset.set_format(type="torch", columns=columns)
        return dataset
