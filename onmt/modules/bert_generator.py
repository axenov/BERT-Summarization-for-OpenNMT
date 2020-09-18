import torch
import torch.nn as nn
from onmt.modules import Embeddings, CopyGenerator
import onmt.modules
from pytorch_transformers import *
from onmt.modules.util_class import Cast
from onmt.global_model import GlobalModel
import numpy as np
import time



class BertGenerator(nn.Module):

    def __init__(self, generator_function, dec_rnn_size,base_field):
        super(BertGenerator, self).__init__()

        self.generator = nn.Sequential(nn.Linear(dec_rnn_size, len(base_field.vocab)), Cast(torch.float32),generator_function)

    def forward(self, hidden, tgts = None, lang_scores = None):

        log_probs = self.generator(hidden)

        if tgts == None: return log_probs

        increment = 1.1
        language_scores = lang_scores[:,[v for v in GlobalModel.converter.values()]]

        percentile = language_scores.topk(language_scores.shape[1]*(100-80)//100)[0][:,-1].repeat(language_scores.shape[1],1).transpose(0,1)
        mean = log_probs.mean(dim=1).repeat(log_probs.shape[1],1).transpose(0,1)


        log_probs = log_probs + language_scores



        return log_probs