"""
Implementation of Bert encoder
"""
from torch import int64, LongTensor, tensor, stack, cat, device
import torch
import torch.nn as nn
from torch import no_grad
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.encoders.transformer import TransformerEncoderLayer
from onmt.modules.position_ffn import PositionwiseFeedForward
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import time
import math



class BertTransformerEncoder(EncoderBase):
    """
    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, embeddings, hidden_size,vocab, multiling, num_layers, heads, d_ff, dropout,
                 max_relative_positions):
        super(BertTransformerEncoder, self).__init__()

        self.embeddings = embeddings

        if multiling:
            self.bertEmbedding =  BertModel.from_pretrained('bert-base-multilingual-cased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        else:
            self.bertEmbedding =  BertModel.from_pretrained('bert-base-cased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
       
        self.bertEmbedding.eval()
        self.bertEmbedding.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.vocab=vocab

        self.BertEmbed_size = 768*4
        self.hidden_size = self.BertEmbed_size + hidden_size
        self.pre_out = nn.Linear(self.hidden_size, hidden_size)

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                hidden_size, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)]) 
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            embeddings,
            opt.enc_rnn_size,
            opt.vocab, 
            opt.bert_multilingual,
            opt.enc_layers,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            opt.max_relative_positions)


    def get_hidden_states(self,input_text,window_length,stride):
        #Calculate number of windows
        if len(input_text)>window_length:
            num_batch = math.ceil(len(input_text)/(window_length - stride))
        else: num_batch = 1
        #Positions of windows
        start_tockens = [(window_length - stride)*i  for i in range(num_batch)]
        #Initialize output
        output = torch.zeros(1,len(input_text),self.BertEmbed_size)
        output = output.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for i,pos in enumerate(start_tockens):
            text_batch = input_text[pos:pos+min(window_length,len(input_text))]
            #Convert words to indeces
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(text_batch)
            tokens_tensor = LongTensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            #Get Bert hidden states
            with no_grad():
                hidden_states, _ = self.bertEmbedding(tokens_tensor)
            #hidden_state = hidden_states[8]+hidden_states[9]+hidden_states[10]+hidden_states[11]
            hidden_state = torch.cat((hidden_states[8],hidden_states[9],hidden_states[10],hidden_states[11]), 2)
            output[:, pos:pos + min(window_length,len(input_text))] += hidden_state
            if i!=0:
               output[:, pos:pos + stride] = output[:, pos:pos + stride]/2
        return output


    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        emb = self.embeddings(src)
        src_transp =src.transpose(0, 1).contiguous()
        #bert_tensors =[[] for _ in src[0]]
        bert_tensors=[]
        for i,sequence in enumerate(src_transp):
            #Get an array of words 
            input_text=[]
            for j,word_id in enumerate(sequence):
                #print(self.vocab.itos[word_id])
                word = self.vocab.itos[word_id]
                if word == '<unk>':
                    word = '[UNK]'
                input_text.append(word)
            with no_grad():
                #Get Bert hidden states
                encoder_embedded = self.get_hidden_states(input_text,512,16)
                bert_tensors.append(encoder_embedded)

        bert_tensors = cat(bert_tensors,0)
        bert_tensors = bert_tensors.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #Linear layer to map Bert hidden state to the required size
        print(bert_tensors)
        out = emb.transpose(0, 1).contiguous()
        print(out)
        out = torch.cat((out,bert_tensors),2)
        out = self.pre_out(out)
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous(), lengths
