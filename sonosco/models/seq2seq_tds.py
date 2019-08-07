import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as tfn

from collections import OrderedDict
from typing import List, Tuple, Dict, Any
from dataclasses import field

from sonosco.model.serialization import serializable
from .modules import SubsampleBlock, TDSBlock, Linear, BatchRNN, InferenceBatchSoftmax, supported_rnns
from .attention import DotAttention


LOGGER = logging.getLogger(__name__)


@serializable
class TDSEncoder(nn.Module):
    """TDS (time-depth separable convolutional) encoder.
    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channels in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        strides (list): strides in TDS layers
        poolings (list) size of poolings in TDS layers
        dropout (float) probability to drop nodes in hidden-hidden connection
        batch_norm (bool): if True, apply batch normalization
        bottleneck_dim (int): dimension of the bottleneck layer after the last layer
    """
    input_dim: int
    in_channel: int
    dropout: float
    bottleneck_dim: int
    channels: list = field(default_factory=list)
    kernel_sizes: list = field(default_factory=list)

    def __post_init__(self):
        assert self.input_dim % self.in_channel == 0
        assert len(self.channels) > 0
        assert len(self.channels) == len(self.kernel_sizes)

        super().__init__()

        self.input_freq = self.input_dim // self.in_channel
        self.bridge = None

        layers = OrderedDict()
        in_ch = self.in_channel
        in_freq = self.input_freq
        subsample_factor = 1

        for layer, (channel, kernel_size) in enumerate(zip(self.channels, self.kernel_sizes)):
            # subsample
            if in_ch != channel:
                layers['subsample%d' % layer] = SubsampleBlock(in_channel=in_ch,
                                                               out_channel=channel,
                                                               in_freq=in_freq,
                                                               dropout=self.dropout)

                subsample_factor *= 2

            # Conv
            layers['tds%d_block%d' % (channel, layer)] = TDSBlock(channel=channel,
                                                                  kernel_size=kernel_size,
                                                                  in_freq=in_freq,
                                                                  dropout=self.dropout)

            in_ch = channel

        self._output_dim = int(in_ch * in_freq)

        if self.bottleneck_dim > 0:
            self.bridge = Linear(self._output_dim, self.bottleneck_dim)
            self._output_dim = self.bottleneck_dim

        self.layers = nn.Sequential(layers)

        # Initialize parameters
        self.subsample_factor = subsample_factor
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        LOGGER.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                LOGGER.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() == 2:
                fan_in = p.size(1)
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))  # linear weight
                LOGGER.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            elif p.dim() == 4:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))  # conv weight
                LOGGER.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            else:
                raise ValueError

    def forward(self, xs, xlens):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', out_ch * feat_dim]`
            xlens (list): A list of length `[B]`
        """
        bs, time, input_dim = xs.size()
        xs = xs.contiguous().view(bs, time, self.in_channel, input_dim // self.in_channel).transpose(2, 1)
        # `[B, in_ch, T, input_dim // in_ch]`

        xs = self.layers(xs)  # `[B, out_ch, T, feat_dim]`
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        # Update xlens
        xlens /= self.subsample_factor

        return xs, xlens


@serializable
class TDSDecoder(nn.Module):
    input_dim: int = 1024
    embedding_dim: int = 512
    vocab_dim: int = 1000
    key_dim: int = 512
    value_dim: int = 512
    rnn_hidden_dim: int = 512
    rnn_type_str: str = "gru"
    attention_type: str = "dot"

    def __post_init__(self):
        assert self.input_dim == self.key_dim + self.value_dim
        assert self.rnn_hidden_dim == self.key_dim

        super().__init__()

        self.rnn_type = supported_rnns[self.rnn_type_str]

        self.word_piece_embedding = nn.Embedding(self.vocab_dim, self.embedding_dim)

        self.rnn = BatchRNN(input_size=self.embedding_dim, hidden_size=self.rnn_hidden_dim,
                            rnn_type=self.rnn_type, batch_norm=False)

        self.attention = DotAttention(self.key_dim)

        self.output_mlp = Linear(in_size=self.value_dim + self.rnn_hidden_dim, out_size=self.vocab_dim)

        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, encoding, encoding_lens, y_labels, y_lens):
        # split into keys and values
        # keys [B,T,K], values [B,T,V]
        keys, values = torch.split(encoding, [self.key_dim, self.value_dim], dim=-1)

        # embed value that we get from teacher-forcing
        y_embed = self.word_piece_embedding(y_labels)

        y_embed = y_embed.transpose(0, 1).contiguous()  # TxBxD
        queries = self.rnn(y_embed, y_lens)
        queries = queries.transpose(0, 1)

        # summaries [B,T_dec,V], scores [B,T_dec,T_enc]
        # TODO: add encoding_lens for attention calculation
        summaries, scores = self.attention(queries, keys, values)

        outputs = self.output_mlp(torch.cat([summaries, queries], dim=-1))

        probs = self.inference_softmax(outputs)

        return probs


@serializable
class TDSSeq2Seq(nn.Module):
    encoder_args: dict = field(default_factory=dict)
    decoder_args: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__init__()
        self.encoder = TDSEncoder(**self.encoder_args)
        self.decoder = TDSDecoder(**self.decoder_args)

    def forward(self, xs, xlens, y_labels=None, y_lens=None):
        encoding, encoding_lens = self.encoder(xs, xlens)

        if y_labels is None or y_lens is None:
            # TODO: implement
            # We are performing inference
            probs = None
            pass
        else:
            # During training we are using teacher-forcing
            probs = self.decoder(encoding, encoding_lens, y_labels, y_lens)

        return probs
