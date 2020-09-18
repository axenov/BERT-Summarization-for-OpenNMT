#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
from extractor import TFIDF
from transformers import BertTokenizer
import unicodedata
import os
import sys
import torch
import fileinput
from nif.annotation import *



def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFKC', s))


class AbstractiveSummarizer():

	def __init__(self, language = 'en', method = 'bert', extract = False):
		'''
		Arguments:
			language: "en" or "de"
			method: "bert" or "conv"
			extract: True or False
		'''
		self.method = method


		parser = self._get_parser(language, method)
		self.opt = parser.parse_args()

		if torch.cuda.is_available():
			self.opt.gpu = 0

		ArgumentParser.validate_translate_opts(self.opt)
		self.translator = build_translator(self.opt, report_score=True)

		self.language = language
		if language == 'en':
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		elif language == 'de':
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
			

		
		self.extract = extract
		if extract:
			self.extractor = TFIDF()


	def _get_parser(self, language, method):
		parser = ArgumentParser(description='summarize.py')
		if not os.path.exists(os.path.join(os.getcwd(), "config/en_bert_transformer.yml")):
			sys.stderr.write("ERROR: Config.ini not found.")

		if language == 'en' and method == 'bert':
			config_file = 'en_bert_transformer.yml'

		elif language == 'en' and method == 'conv':
			config_file = 'en_conv_transformer.yml'

		elif language == 'de' and method == 'bert':
			config_file = 'de_bert_transformer.yml'	

		elif language == 'de' and method == 'conv':
			config_file = 'de_conv_transformer.yml'	


		prec_argv = sys.argv
		sys.argv.extend(['-config', 'config/' + config_file]) 
		sys.argv = prec_argv

		opts.config_opts(parser)
		opts.translate_opts(parser)

		return parser

	def summarize(self, texts):
		'''
		Arguments:
			texts: list(str)
		'''
		if self.extract:
			summaries = self.extractor.summarize(texts)
		else:
			summaries = texts

		with open ('src.txt', "w", encoding='utf-8') as src_f:
			for summary in summaries:
				src_f.write('[CLS] '+ ' '.join(self.tokenizer.tokenize(strip_accents(summary))) + '\n')


		src_shards = split_corpus(self.opt.src, self.opt.shard_size)
		for i, src_shard in enumerate(src_shards):
			self.translator.translate(
			src = src_shard,
			batch_size = self.opt.batch_size,
			attn_debug = self.opt.attn_debug
			)

		output = []
		for line in fileinput.FileInput("output.out",inplace=1):
			line=line.replace(" ##","").replace(" .",".").replace(" ,",",").replace(" !","!").replace(" ?","?").replace("\n","")
			print(line)
			output.append(line)

		os.remove('src.txt') 
		os.remove("output.out") 

		return output

	def analyzeNIF(self, nifDocument):
		d = nifDocument
		text = d.context.nif__is_string
		text = text.replace("\n", "")
		summary = self.summarize([text])

		kwargs = {"nif__summary" : summary}
		nc = NIFContext(text,d.context.uri_prefix,**kwargs)
		d2 = NIFDocument(nc, structures=d.structures)
		return d2