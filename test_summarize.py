#!/usr/bin/env python
# -*- coding: utf-8 -*-
from summarizer import AbstractiveSummarizer

if __name__ == "__main__":

	FILE_NAME = "sample_de.txt"
	texts = []
	with open("data/"+FILE_NAME) as f:
		for text in f:
			texts.append(text)

	model = AbstractiveSummarizer(language = 'de', method = 'bert', extract = True)
	for summ in model.summarize(texts):
		print(summ)
