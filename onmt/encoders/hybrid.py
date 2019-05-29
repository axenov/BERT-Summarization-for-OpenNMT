"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch as torch
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules import ConvMultiHeadedAttention

class RNNEncoderLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout,
                 rnn_type, bidirectional,
                 max_relative_positions=0):
        super(RNNEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
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

        #attn, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask, type="self")
        #attn_drop = self.dropout(attn) + inputs
        #attn_drop_norm = self.layer_norm_2(attn_drop)
        #context,_ = self.rnn(attn_drop_norm)
        #out = self.dropout(context) + attn_drop     

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

    def __init__(self, d_model, heads, d_ff, dropout, layer_index, num_layers,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()


        if layer_index<num_layers//2:
            self.self_attn = ConvMultiHeadedAttention(
                heads, d_model,5,3, dropout=dropout,
                max_relative_positions=max_relative_positions)

        else:
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
                d_model, heads, d_ff, dropout,i,num_layers,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.rnn = RNNEncoderLayer(
                d_model, heads, d_ff, dropout,
                rnn_type, bidirectional,
                max_relative_positions=max_relative_positions)

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
