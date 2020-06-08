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
			#docs_matrix.append(doc_vector)
			docs_matrix.append(doc_vector.repeat(len(doc_sents_vectors),1))
		docs_matrix = torch.cat(docs_matrix, dim = 0)
		#docs_matrix = torch.mean(torch.cat(docs_matrix), 0)
		#docs_matrix = docs_matrix.repeat(len(sents_vectors), 1)

		sents_matrix = self.linear_sent(sents_matrix)
		docs_matrix = self.linear_doc(docs_matrix)
		# meta_matrix = self.linear_meta(torch.cat(meta_inf, dim = 0))
		out = self.linear_final(
			self.sigmoid(sents_matrix + docs_matrix)
		)  # +meta_matrix))
		# out = self.linear_meta_final(torch.cat([out, torch.cat(meta_inf, dim = 0)], dim = 1))
		return out



class Summarizer(nn.Module):
	def __init__(
		self,
		logger=None,
		finetune=False):
		super(Summarizer, self).__init__()

		self.spacy_pipeline = spacy.load("en")

		self.finetune = finetune

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
		model = cls(logger=logger)
		if bert_file is not None:
			model.load_model(f"{folder}{reg_file}", f"{folder}{bert_file}")
		else:
			model.load_model(f"{folder}{reg_file}", None)
		return model

	def sent_tokenize(self, text):
		if isinstance(text, list):
			output = text
		else:
			output = nt.sent_tokenize(text)
		return output


	def calculate_embeddings(self, sentence):
		#line = sentence
		line = "[CLS] " + sentence
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

		embeddings = predictions[0].view(len(indexed_tokens), 1, 768)
		# For only embedding layer
		# embeddings.append(predictions.view(len(indexed_tokens),1,768))
		return embeddings


	def build_features(self, text):
		tokenized_sentences = self.sent_tokenize(text)
		embeddings = [self.calculate_embeddings(sent) for sent in tokenized_sentences]
		return embeddings, tokenized_sentences

	def forward(self, input_cluster):
		total_embeddings = []
		total_sentences = []
		mask = []
		doc = input_cluster
		if not isinstance(doc, list):
			doc = [doc]
		for i, text in enumerate(doc):
			sentence_embeddings, sentences = self.build_features(text)
			total_embeddings += sentence_embeddings
			total_sentences += sentences
			mask += [i] * len(sentence_embeddings)
		prediction = self.regression(total_embeddings, mask)
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
