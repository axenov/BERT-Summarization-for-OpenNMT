"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.bert_encoder import BertEncoder
from onmt.encoders.bert_transformer_encoder import BertTransformerEncoder
from onmt.encoders.conv_transformer import ConvTransformerEncoder
from onmt.encoders.hybrid import HybridTransformerEncoder




str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder, "bert": BertEncoder,"bert-transformer":BertTransformerEncoder,"conv-transformer":ConvTransformerEncoder,"hybrid":HybridTransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder","HybridTransformerEncoder",
           "MeanEncoder", "str2enc", "BertEncoder","BertTransformerEncoder","ConvTransformerEncoder"]
