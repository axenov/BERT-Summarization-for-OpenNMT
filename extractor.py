#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDF():

	def __init__(self):
		self.vectorizer = TfidfVectorizer()

	def _get_len(self, sent):
		return len(word_tokenize(sent))

	def summarize(self, texts):
		summaries = []
		for text in texts:
			if self._get_len(text) < 512:
				summaries.append(text)
				continue
			else:
				sentences = sent_tokenize(text)
				documents_vector = self.vectorizer.fit_transform(sentences)
				scores = cosine_similarity(documents_vector[:1], documents_vector[1:])[0]
				sorted_ix = np.argsort(scores)[::-1]

				summary_sentences = []
				summary_ids = []
				summary_length = 0

				for sent_id in sorted_ix:
					summary_sentences.append(sentences[sent_id])
					summary_ids.append(sent_id)
					summary_length += self._get_len(sentences[sent_id])

					if summary_length >= 512:
						summary_sentences = summary_sentences[:-1]
						summary_ids = summary_ids[:-1]
						break

				summary_sentences = [summary_sentences[i] for i in np.argsort(summary_ids)]
				summaries.append(" ".join(summary_sentences))

		return summaries
