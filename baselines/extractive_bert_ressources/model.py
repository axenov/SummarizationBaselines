# -*- coding: utf-8 -*-
import os
import re
import json
import operator
import sys
import nltk
import math
import torch
import numpy as np
import torch.nn as nn
import statistics
import logging

from nltk.stem import WordNetLemmatizer

# from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import word_tokenize
import nltk.tokenize as nt

from nltk.corpus import stopwords
from allennlp.predictors.predictor import Predictor as oie_predictor
import spacy
import neuralcoref
from pytorch_transformers import *

# from pythonrouge.pythonrouge import Pythonrouge
from rouge_score import rouge_scorer


class Predictor(torch.nn.Module):
    def __init__(self, inputSize=768, hidden_size=1024):
        super(Predictor, self).__init__()

        self.hidden_size = hidden_size
        self.gru_sent = torch.nn.GRU(inputSize, hidden_size)
        self.gru_doc = torch.nn.GRU(hidden_size, hidden_size)

        self.linear_sent = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_doc = torch.nn.Linear(hidden_size, hidden_size)
        # self.linear_meta = torch.nn.Linear(4, hidden_size)
        self.linear_final = torch.nn.Linear(hidden_size, 1)
        # self.linear_meta_final = torch.nn.Linear(5, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, sents_emb, mask):  # , meta_inf):
        sents_vectors = []
        for sent_emb in sents_emb:
            _, sent_vector = self.gru_sent(sent_emb)
            sents_vectors.append(sent_vector)
        sents_matrix = torch.cat(sents_vectors, dim=0)
        sents_matrix = sents_matrix.view(len(sents_emb), self.hidden_size)

        docs_matrix = []
        for i in range(mask[-1] + 1):
            doc_sents_vectors = [x for j, x in enumerate(sents_vectors) if mask[j] == i]
            doc_sents_matrix = torch.cat(doc_sents_vectors, dim=0)
            _, doc_vector = self.gru_doc(doc_sents_matrix)
            doc_vector = doc_vector.view(1, self.hidden_size)
            docs_matrix.append(doc_vector)
            # docs_matrix.append(doc_vector.repeat(len(doc_sents_vectors),1))
        # docs_matrix = torch.cat(docs_matrix, dim = 0)
        docs_matrix = torch.mean(torch.cat(docs_matrix), 0)
        docs_matrix = docs_matrix.repeat(len(sents_vectors), 1)

        sents_matrix = self.linear_sent(sents_matrix)
        docs_matrix = self.linear_doc(docs_matrix)
        # meta_matrix = self.linear_meta(torch.cat(meta_inf, dim = 0))
        out = self.linear_final(
            self.sigmoid(sents_matrix + docs_matrix)
        )  # +meta_matrix))
        # out = self.linear_meta_final(torch.cat([out, torch.cat(meta_inf, dim = 0)], dim = 1))
        return out


class Tripple:
    def __init__(self, arg0, verb, arg1):
        self.subject = arg0
        self.verb = verb
        self.object = arg1

    def __str__(self):
        return f"[CLS] {self.subject} [SEP] {self.verb} [SEP] {self.object}"

    def __repr__(self):
        return f"Tripple({self.subject!r},{self.verb!r},{self.object!r})"


class InformationExtractor:
    def __init__(self):
        self.predictor = oie_predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz"
        )
        if torch.cuda.is_available():
            # self.predictor.cuda()
            self.predictor._model = self.predictor._model.cuda(0)

    def Arguments(self):
        _dict = dict({})

        def dict_instance(string):
            values = string.split(": ")
            if len(values) > 1:
                _dict[values[0]] = values[1]
            return _dict

        return dict_instance

    def find_tripples(self, string):
        tripples = []
        # print(string)
        extraction = self.predictor.predict(sentence=string)
        # print(extraction)
        for phrase in extraction["verbs"]:
            arg_builder = self.Arguments()
            args = dict({})
            matches = re.findall(r"\[(.+?)\]", phrase["description"])
            for x in matches:
                args = arg_builder(x)
            if {"V", "ARG0", "ARG1"}.issubset(set(args.keys())):
                tripples.append(Tripple(args["ARG0"], args["V"], args["ARG1"]))
                if "ARG2" in args.keys():
                    tripples.append(Tripple(args["ARG0"], args["V"], args["ARG2"]))
        # print(tripples)
        return tripples


class Summarizer(nn.Module):
    def __init__(
        self,
        logger=None,
        coreference=False,
        extraction=False,
        no_bert=False,
        finetune=False,
        tfidf=False,
        length=False,
        relpos=False,
        pos=False,
    ):
        super(Summarizer, self).__init__()

        self.spacy_pipeline = spacy.load("en")

        self.document = None
        self.coreference = coreference
        if coreference:
            coref = neuralcoref.NeuralCoref(self.spacy_pipeline.vocab)
            self.spacy_pipeline.add_pipe(coref, name="neuralcoref")
        self.extraction = extraction
        if extraction:
            self.extractor = InformationExtractor()

        self.Stopwords = set(stopwords.words("english"))
        self.wordlemmatizer = WordNetLemmatizer()

        self.tfidf = tfidf
        self.length = length
        self.pos = pos
        self.relpos = relpos

        self.finetune = finetune
        self.no_bert = no_bert

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.regression = Predictor()

        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        if logger == None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger
        logging.getLogger().handlers = []

    def train(self):
        if self.finetune:
            self.bert.train()
            self.unfreeze_bert_encoder()
        else:
            self.bert.eval()
            self.freeze_bert_encoder()

    def test(self):
        self.bert.eval()
        self.freeze_bert_encoder()

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def safe_model(self, folder, filename):
        if not os.path.exists(f"{folder}config.json"):
            with open(f"{folder}config.json", "w") as fp:

                args = {
                    "No-BERT": self.no_bert,
                    "Fine-tune": self.finetune,
                    "Coreference": self.coreference,
                    "Extraction": self.extraction,
                    "Position": self.pos,
                    "Relative position": self.relpos,
                    "Length": self.length,
                    "TF_IDF": self.tfidf,
                }
                json.dump(args, fp, indent=4)

        torch.save(self.regression.state_dict(), f"{folder}reg_{filename}.pt")
        self.logger.info(f"Model {folder}reg_{filename}.pt saved.")
        if self.finetune:
            torch.save(self.bert.state_dict(), f"{folder}bert_{filename}.pt")
            self.logger.info(f"Model {folder}bert_{filename}.pt saved.")

    def load_model(self, reg_path, bert_path):
        try:
            if torch.cuda.is_available():
                self.regression.load_state_dict(torch.load(reg_path))
                self.regression.cuda()
            else:
                self.regression.load_state_dict(
                    torch.load(reg_path, map_location="cpu")
                )
        except FileNotFoundError:
            print(f"file {reg_path} not found")

        if bert_path is not None:
            try:
                if torch.cuda.is_available():
                    self.bert.load_state_dict(torch.load(bert_path))
                    self.bert.cuda()
                else:
                    self.bert.load_state_dict(torch.load(bert_path, map_location="cpu"))
            except FileNotFoundError:
                print(f"file {bert_path} not found")

    @classmethod
    def from_pretrained(cls, folder, reg_file, bert_file=None, logger=None):
        try:
            with open(f"{folder}config.json", "r") as config:
                args = json.load(config)
                model = cls(
                    logger=logger,
                    coreference=args["Coreference"],
                    extraction=args["Extraction"],
                    no_bert=args["No-BERT"],
                    tfidf=args["TF_IDF"],
                    length=args["Length"],
                    relpos=args["Relative position"],
                    pos=args["Position"],
                )
                if bert_file is not None:
                    model.load_model(f"{folder}{reg_file}", f"{folder}{bert_file}")
                else:
                    model.load_model(f"{folder}{reg_file}", None)
        except FileNotFoundError:
            print(f"file {folder}config.json not found")
        return model

    def sent_tokenize(self, text, coref=False):
        if coref:
            if isinstance(text, list):
                spacy_text = " ".join(text)
            else:
                spacy_text = text
            # if self.document == None:
            self.document = self.spacy_pipeline(spacy_text)

            if self.document._.has_coref:
                clusters = self.document._.coref_clusters
                output = self.get_resolved(self.document, text, clusters)
            else:
                # output = [str(sent) for sent in self.document.sents]
                output = nt.sent_tokenize(text)
        else:
            if isinstance(text, list):
                output = text
            else:
                # self.document = self.spacy_pipeline(text)
                # output = [str(sent) for sent in self.document.sents]
                output = nt.sent_tokenize(text)
        return output

    def get_resolved(self, doc, text, clusters):
        def sentence_generator(text, doc):
            if isinstance(text, list):
                sentences = text
            else:
                sentences = nt.sent_tokenize(text)
                # sentences = [str(sent) for sent in doc.sents]
            tokenizer = spacy.load("en")
            for sent in sentences:
                yield tokenizer(sent)

        def get_2d_element(arrays, index):
            j = index
            lens = [len(sent) for sent in arrays]
            for i, length in enumerate(lens):
                j = j - length
                if j < 0:
                    return i, length + j

        resolved_list = []
        for sent in sentence_generator(text, doc):
            resolved_list.append(list(tok.text_with_ws for tok in sent))

        for cluster in clusters:
            for coref in cluster:
                if coref != cluster.main:
                    ind1, ind2 = get_2d_element(resolved_list, coref.start)
                    resolved_list[ind1][ind2] = (
                        cluster.main.text + doc[coref.end - 1].whitespace_
                    )
                    for i in range(coref.start + 1, coref.end):
                        ind3, ind4 = get_2d_element(resolved_list, i)
                        resolved_list[ind3][ind4] = ""
        output = ["".join(sublist) for sublist in resolved_list]
        return output

    def calculate_embeddings(self, sentence):
        lines = []
        embeddings = []
        if self.extraction:
            tripples = self.extractor.find_tripples(sentence)
            lines = [str(trpl) for trpl in tripples]
        if len(lines) == 0:
            lines = ["[CLS] " + sentence]
        for line in lines:
            if len(line) > 512:
                line = line[:512]
            assert len(line) <= 512
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(line)
            )
            tokens_tensor = torch.tensor([indexed_tokens])
            if torch.cuda.is_available():
                tokens_tensor = tokens_tensor.cuda()
            if not self.finetune:
                with torch.no_grad():
                    predictions = self.bert(tokens_tensor)
                    # predictions = self.bert.embeddings(tokens_tensor)

            else:
                predictions = self.bert(tokens_tensor)
                # predictions = self.bert.embeddings(tokens_tensor)

            # embeddings.append(predictions[0][0][:1])
            # embeddings.append(torch.mean(predictions[0],1))

            embeddings.append(predictions[0].view(len(indexed_tokens), 1, 768))
            # For only embedding layer
            # embeddings.append(predictions.view(len(indexed_tokens),1,768))
        return embeddings[0]  # torch.mean(torch.stack(embeddings),0)

    def calculate_embeddings_(self, sentence):
        tokens_tensor = torch.zeros([1, 768], dtype=torch.float)
        if torch.cuda.is_available():
            tokens_tensor = tokens_tensor.cuda()
        return tokens_tensor

    # def embeding_features(self,text):
    def build_features(self, text):
        tokenized_sentences = self.sent_tokenize(text, coref=self.coreference)
        if self.no_bert:
            embeddings = [
                self.calculate_embeddings_(sent) for sent in tokenized_sentences
            ]
        else:
            embeddings = [
                self.calculate_embeddings(sent) for sent in tokenized_sentences
            ]
        return embeddings, tokenized_sentences

    def forward(self, input_cluster):
        total_embeddings = []
        total_sentences = []
        # meta_features = []
        mask = []
        doc = input_cluster
        if not isinstance(doc, list):
            doc = [doc]
        for i, text in enumerate(doc):
            # meta_features.append(self.sentence_features(text))
            sentence_embeddings, sentences = self.build_features(text)
            total_embeddings += sentence_embeddings
            total_sentences += sentences
            mask += [i] * len(sentence_embeddings)
        # print(meta_features)
        prediction = self.regression(total_embeddings, mask)  # , meta_features)
        return prediction, total_sentences, mask

    def calculate_rouge(self, sentence1, sentence2, method="f1"):
        scores = self.scorer.score(sentence1, sentence2)
        ind = 2
        if method == "recall":
            ind = 0
        elif method == "precision":
            ind = 1
        elif method == "f1":
            ind = 2
        rouge_score = (scores["rouge1"][ind] + scores["rouge2"][ind]) / 2
        # rouge_score = scores['rougeL'][ind]

        return rouge_score

    def build_labels(self, texts, summary):
        rouge_scores = []
        docs = texts
        if not isinstance(docs, list):
            docs = [docs]
        for doc in docs:
            sentences = self.sent_tokenize(doc)
            rouge_scores += [
                self.calculate_rouge(sent, summary, method="recall")
                for sent in sentences
            ]
        if torch.cuda.is_available():
            output = torch.tensor(rouge_scores).unsqueeze(1).cuda()
        else:
            output = torch.tensor(rouge_scores).unsqueeze(1)

        # output = self.label_softmax(1000* output)
        return output

    def sentence_relevance(self, sentences, summary_sentences):
        return [
            self.calculate_rouge(sent, " ".join(summary_sentences), method="recall")
            for sent in sentences
        ]

    def sentence_features(self, text):
        tokenized_sentences = self.sent_tokenize(text)
        num_sent = len(tokenized_sentences)

        lengths = []
        tokenized_words = []
        for i, sent in enumerate(tokenized_sentences):
            sent_edit = re.sub(r"\d+", "", self.remove_special_characters(sent))
            tokenized_words_with_stopwords_sent = word_tokenize(sent_edit)
            tokenized_words_sent = [
                word
                for word in tokenized_words_with_stopwords_sent
                if word not in self.Stopwords
            ]
            tokenized_words_sent = [
                word.lower() for word in tokenized_words_sent if len(word) > 1
            ]
            tokenized_words += tokenized_words_sent
            lengths.append(float(len(tokenized_words_sent)))
        tokenized_words = self.lemmatize_words(tokenized_words)
        word_freq = self.freq(tokenized_words)

        if self.pos:
            position = [float(i + 1) for i, sent in enumerate(tokenized_sentences)]
            position = torch.tensor(position).unsqueeze(0).transpose(0, 1)
        else:
            position = torch.zeros((num_sent, 1), dtype=torch.float)

        if self.relpos:
            rel_position = [
                (num_sent - i) / num_sent for i, sent in enumerate(tokenized_sentences)
            ]
            rel_position = torch.tensor(rel_position).unsqueeze(0).transpose(0, 1)
        else:
            rel_position = torch.zeros((num_sent, 1), dtype=torch.float)

        if self.length:
            lengths = torch.tensor(lengths).unsqueeze(0).transpose(0, 1)
        else:
            lengths = torch.zeros((num_sent, 1), dtype=torch.float)

        if self.tfidf:
            tfidf_scores = [
                self.calculate_tfids(sent, word_freq, tokenized_sentences)
                for i, sent in enumerate(tokenized_sentences)
            ]
            tfidf_scores = torch.tensor(tfidf_scores).unsqueeze(0).transpose(0, 1)
        else:
            tfidf_scores = torch.zeros((num_sent, 1), dtype=torch.float)

        if torch.cuda.is_available():
            position = position.cuda()
            rel_position = rel_position.cuda()
            lengths = lengths.cuda()
            tfidf_scores = tfidf_scores.cuda()

        features = torch.cat([position, rel_position, lengths, tfidf_scores], 1)
        return features

    def lemmatize_words(self, words):
        lemmatized_words = []
        for word in words:
            lemmatized_words.append(self.wordlemmatizer.lemmatize(word))
        return lemmatized_words

    def stem_words(self, words):
        stemmed_words = []
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        return stemmed_words

    def remove_special_characters(self, text):
        regex = r"[^a-zA-Z0-9\s]"
        text = re.sub(regex, "", text)
        return text

    def freq(self, words):
        words = [word.lower() for word in words]
        dict_freq = {}
        words_unique = []
        for word in words:
            if word not in words_unique:
                words_unique.append(word)
        for word in words_unique:
            dict_freq[word] = words.count(word)
        return dict_freq

    def pos_tagging(self, text):
        pos_tag = nltk.pos_tag(text.split())
        pos_tagged_noun_verb = []
        for word, tag in pos_tag:
            if (
                tag == "NN"
                or tag == "NNP"
                or tag == "NNS"
                or tag == "VB"
                or tag == "VBD"
                or tag == "VBG"
                or tag == "VBN"
                or tag == "VBP"
                or tag == "VBZ"
            ):
                pos_tagged_noun_verb.append(word)
        return pos_tagged_noun_verb

    def tf_score(self, word, sentence):
        freq_sum = 0
        word_frequency_in_sentence = 0
        len_sentence = len(sentence)
        for word_in_sentence in sentence.split():
            if word == word_in_sentence:
                word_frequency_in_sentence = word_frequency_in_sentence + 1
        tf = word_frequency_in_sentence / len_sentence
        return tf

    def idf_score(self, no_of_sentences, word, sentences):
        no_of_sentence_containing_word = 0
        for sentence in sentences:
            sentence = self.remove_special_characters(str(sentence))
            sentence = re.sub(r"\d+", "", sentence)
            sentence = sentence.split()
            sentence = [
                word
                for word in sentence
                if word.lower() not in self.Stopwords and len(word) > 1
            ]
            sentence = [word.lower() for word in sentence]
            sentence = [self.wordlemmatizer.lemmatize(word) for word in sentence]
            if word in sentence:
                no_of_sentence_containing_word = no_of_sentence_containing_word + 1
        idf = math.log10(no_of_sentences / no_of_sentence_containing_word)
        return idf

    def tf_idf_score(self, tf, idf):
        return tf * idf

    def word_tfidf(self, dict_freq, word, sentences, sentence):
        word_tfidf = []
        tf = self.tf_score(word, sentence)
        idf = self.idf_score(len(sentences), word, sentences)
        tf_idf = self.tf_idf_score(tf, idf)
        return tf_idf

    def calculate_tfids(self, sentence, dict_freq, sentences):
        sentence_score = 0
        sentence = self.remove_special_characters(str(sentence))
        sentence = re.sub(r"\d+", "", sentence)
        pos_tagged_sentence = []
        no_of_sentences = len(sentences)
        pos_tagged_sentence = self.pos_tagging(sentence)
        for word in pos_tagged_sentence:
            if (
                word.lower() not in self.Stopwords
                and word not in self.Stopwords
                and len(word) > 1
            ):
                word = word.lower()
                word = self.wordlemmatizer.lemmatize(word)
                sentence_score = sentence_score + self.word_tfidf(
                    dict_freq, word, sentences, sentence
                )
        return sentence_score
