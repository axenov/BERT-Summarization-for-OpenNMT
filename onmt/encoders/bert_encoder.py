"""
Implementation of Bert encoder
"""
from torch import int64, LongTensor, tensor, stack, cat, device
import torch
import torch.nn as nn
from torch import no_grad
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
#from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_transformers import *

import time
import math
from onmt.global_model import GlobalModel



class BertEncoder(EncoderBase):
    """
    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, embeddings, hidden_size, multiling):
        super(BertEncoder, self).__init__()

        self.embeddings = embeddings
        self.BertEmbed_size = 768*4
        self.pre_out = nn.Linear(self.BertEmbed_size, hidden_size)
        self.relu = nn.ReLU()

        #self.temp = torch.tensor([])
        #self.iu = 0
        #self.embed_lin = nn.Linear(768,  hidden_size)


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(embeddings,opt.enc_rnn_size,opt.bert_multilingual)


    def get_hidden_states_window(self,input_text,window_length,stride):
        #Calculate number of windows
        if len(input_text)>window_length:
            num_batch = math.ceil(len(input_text)/(window_length - stride))
        else: num_batch = 1
        #Positions of windows
        start_tockens = [(window_length - stride)*i  for i in range(num_batch)]
        #Initialize output
        output = torch.zeros(1,len(input_text),self.BertEmbed_size)
        output = output.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #output_embed = torch.zeros(1,len(input_text),768)
        #output_embed = output_embed.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        for i,pos in enumerate(start_tockens):
            text_batch = input_text[pos:pos+min(window_length,len(input_text))]
            #Convert words to indeces
            indexed_tokens = GlobalModel.tokenizer.convert_tokens_to_ids(text_batch)
            tokens_tensor = LongTensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            #Get Bert hidden states
            with no_grad():
                _,_,hidden_states = GlobalModel.bert_embeddings(tokens_tensor)
                #bert_embeddings = self.bertEmbedding.embeddings(tokens_tensor)
            hidden_state = torch.cat((hidden_states[9],hidden_states[10],hidden_states[11],hidden_states[12]), 2)
            #output[:, pos:pos + min(window_length,len(input_text))] += hidden_state
            output_embed[:, pos:pos + min(window_length,len(input_text))] += bert_embeddings
            if i!=0:
            	output[:, pos:pos + stride] = output[:, pos:pos + stride]/2
            	#output_embed[:, pos:pos + stride] = output_embed[:, pos:pos + stride]/2


        return output

    def get_hidden_states(self,input_text):
        #Initialize output
        output = torch.zeros(1,len(input_text),768*4)
        output = output.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #output_embed = torch.zeros(1,len(input_text),768)
        #output_embed = output_embed.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        indexed_tokens = GlobalModel.tokenizer.convert_tokens_to_ids(input_text)

        tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
        tokens_tensor = tokens_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #Get Bert hidden states
        with no_grad():
            _,_,hidden_states = GlobalModel.bert_embeddings(tokens_tensor)
            #output_embed = self.bertEmbedding.embeddings(tokens_tensor)
        output = torch.cat((hidden_states[9],hidden_states[10],hidden_states[11],hidden_states[12]), 2)

        return output


    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        #start_time = time.time()
        self._check_args(src, lengths)

        #emb = self.embeddings(src)

        src_transp =src.transpose(0, 1).contiguous()
        batch_num = src_transp.shape[0]
        seq_length= src_transp.shape[1]

        src_bert_indeces = src_transp.view(src_transp.numel())
        src_bert_indeces = [GlobalModel.vocab[src_bert_indeces[i]] for i in range(src_transp.numel())]
        #src_bert_indeces[src_bert_indeces == '<unk>'] = '[UNK]'
        src_bert_indeces = ['[UNK]' if wd == '<unk>' else wd for wd in src_bert_indeces]


        bert_tensors=[]
        #bert_embeddings=[]
        for i in range(batch_num):
            input_text = src_bert_indeces[i*seq_length:(i+1)*seq_length]
            encoder_embedded = self.get_hidden_states(input_text)
            #encoder_embedded = self.get_hidden_states_window(input_text,512,256)
            bert_tensors.append(encoder_embedded)
            #bert_embeddings.append(bert_embedding)

        bert_tensors = cat(bert_tensors,0)
        #bert_embeddings = cat(bert_embeddings,0)

        bert_tensors = bert_tensors.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #Linear layer to map Bert hidden state to the required size
        #out = self.relu(self.pre_out(bert_tensors))
        out = self.pre_out(bert_tensors)

        #print(self.pre_out.weight.data)
        #if self.iu == 0:
        #    self.temp = self.pre_out.weight.data
        #    self.iu = 1 
        #print(self.temp)
        #if torch.all(self.pre_out.weight.data.eq(self.temp)):
        #    print("yes")
        #else: 
        #    print("No")

        #self.temp = self.pre_out.weight.data.clone()
        #emb = self.embed_lin(bert_embeddings).transpose(0, 1).contiguous()

        #print("--- %s seconds ---" % (time.time() - start_time))
        #.transpose(0, 1).contiguous()
        #print(out)
        return out.transpose(0, 1).contiguous(), out.transpose(0, 1).contiguous(), lengths
