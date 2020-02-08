"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch import no_grad

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from onmt.global_model import GlobalModel



class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn


class BertTransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, embeddings,
                 max_relative_positions, multiling):
        super(BertTransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.BertEmbed_size = 768
        self.pre_out = nn.Linear(self.BertEmbed_size, d_model)


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.bert_multilingual)

    def get_bert_embeddings(self,input_text):
        #Initialize output

        output_embed = torch.zeros(1,len(input_text),self.BertEmbed_size)
        output_embed = output_embed.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        indexed_tokens = GlobalModel.tokenizer.convert_tokens_to_ids(input_text)
        tokens_tensor = torch.LongTensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #Get Bert hidden states
        with no_grad():
            #_,_,hidden_states = GlobalModel.bert_embeddings(tokens_tensor)
            output_embed = GlobalModel.bert_embeddings.embeddings(tokens_tensor)
        #output_embed = torch.cat((hidden_states[9],hidden_states[10],hidden_states[11],hidden_states[12]), 2)
        return  output_embed

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        src = self.state["src"]
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        tgt_transp =tgt.transpose(0, 1).contiguous()
        
        tgt_bert_indeces = tgt_transp.view(tgt_transp.numel())
        tgt_bert_indeces = [GlobalModel.vocab[tgt_bert_indeces[i]] for i in range(tgt_transp.numel())]
        tgt_bert_indeces = ['[UNK]' if wd == '<unk>' else wd for wd in tgt_bert_indeces]

        bert_embeddings=[]
        for i in range(tgt_batch):
            input_text = tgt_bert_indeces[i*tgt_len:(i+1)*tgt_len]
            bert_embedding = self.get_bert_embeddings(input_text)
            bert_embeddings.append(bert_embedding)
        bert_embeddings = torch.cat(bert_embeddings,0)

        bert_embeddings = bert_embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #Linear layer to map Bert hidden state to the required size
        emb = self.pre_out(bert_embeddings).transpose(0, 1).contiguous()

        #tgt_start = tgt[0,:,:].unsqueeze(0)

        #emb0 = self.embeddings(tgt_start, step=step)
        #print(emb0.shape)
        #last_token = False
        #tgt_transp =tgt.transpose(0, 1).contiguous()

        #bert_embeddings=[]
        #for i,sequence in enumerate(tgt_transp):
        #    #Get an array of words 
        #    input_text=[]
        #    for j,word_id in enumerate(sequence):
        #        word = self.vocab.itos[word_id]
        #        if word == '<unk>':
        #            word = '[UNK]'
        #        print(word)
        #        if word == '</s>' or word == '[PAD]':
        #          last_token=True
        #          #tgt_end = tgt[-1,:,:].unsqueeze(0)
        #          #print(word_id.shape)
        #          emb1 = self.embeddings(word_id.unsqueeze(0).unsqueeze(1), step=step)#
        #        else: input_text.append(word)

        #    with no_grad():
        #        #Get Bert hidden states
        #        bert_embedding = self.get_bert_embeddings(input_text,512,16)
        #        bert_embedding = self.pre_out(bert_embedding)
        #        if last_token:
        #          bert_embedding = torch.cat([bert_embedding,emb1],1)
        #        print(bert_embedding.shape)
        #        bert_embeddings.append(bert_embedding)


        #bert_embeddings = torch.cat(bert_embeddings,0)
        #bert_embeddings = bert_embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ##Linear layer to map Bert embeddings to the required size
        #emb = bert_embeddings.transpose(0, 1).contiguous()
        #print(emb.shape)
        #if last_token: emb = torch.cat([emb0,emb,emb1],0)
        #else: 
        #emb = torch.cat([emb0,emb],0)


        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth))
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache
