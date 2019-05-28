"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch as torch
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.rnn_factory import rnn_factory

"""
#Multi-Head Attention module 
import math
import torch
import torch.nn as nn

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


class MultiHeadedRNN(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.1,rnn_type, bidirectional):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        num_directions = 2 if bidirectional else 1
        assert model_dim % num_directions == 0
        self.model_dim = model_dim // num_directions
        self.head_count = head_count
        super(MultiHeadedRNN, self).__init__()

        hidden_size = hidden_size // num_directions

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=self.dim_per_head,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)


    def forward(self, input_seq, type=None):

        batch_size = input_seq.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        input_len = input_seq.size(1)
        device = input.device
        linear_rnns = zeros(batch_size,input_len, head_count * self.dim_per_head)
        for i in range(head_count):
            linear_rnns[:,:,i*self.dim_per_head:(i+1)*self.dim_per_head] rnn_states, rnn_final = self.rnn(input_seq)

        # Apply attention dropout and compute context vectors.
        drop_states = self.dropout(states)

        context = unshape(drop_states)

        output = self.final_linear(context)
        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn

"""
class RNNEncoderLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout,
                 rnn_type, bidirectional):
        super(RNNEncoderLayer, self).__init__()

        #self.mh_rnn = MultiHeadedRNN(
        #    heads, d_model, dropout=dropout, rnn_type, bidirectional)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=d_model,
                        hidden_size=d_model,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context,_ = self.rnn(input_norm)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

        


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class HybridTransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions,rnn_type, bidirectional):
        super(HybridTransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.rnn = RNNEncoderLayer(
                d_model, heads, d_ff, dropout,
                rnn_type, bidirectional)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.rnn_type,
            opt.brnn)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        out_rnn = out
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out).transpose(0, 1).contiguous()
        #print("out: {}".format(out.shape))

        out_rnn = self.rnn(out_rnn, mask)
        out_rnn = self.layer_norm(out)
        #print("out_rnn: {}".format(out_rnn.shape))
        
        out_comb = torch.stack([out,out_rnn])
        #print("out_comb: {}".format(out_comb.shape))


        return emb, out_comb, lengths
