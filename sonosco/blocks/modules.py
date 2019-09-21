import torch
import logging
import torch.nn as nn
import torch.nn.functional as functional

from sonosco.config.global_settings import DROPOUT

LOGGER = logging.getLogger(__name__)

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.ByteTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return functional.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, batch_norm=True, bidirectional=False):
        super(BatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, *args):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x, *args)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x

    def forward_one_step(self, x, *args, **kwargs):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, h = self.rnn(x, *args, **kwargs)
        return x, h


class TDSBlock(nn.Module):
    """TDS block.
    Args:
        channel (int):
        kernel_size (int):
        in_freq (int):
        dropout (float):
    """

    def __init__(self, channel, kernel_size, in_freq, dropout):
        super().__init__()

        self.channel = channel
        self.in_freq = in_freq

        self.conv2d = nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=(kernel_size, 1),
                                stride=(1, 1),
                                padding=(kernel_size // 2, 0))
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(in_freq * channel, eps=1e-6)

        # second block
        self.conv1d_1 = nn.Conv2d(in_channels=in_freq * channel,
                                  out_channels=in_freq * channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.dropout2_1 = nn.Dropout(p=dropout)

        self.conv1d_2 = nn.Conv2d(in_channels=in_freq * channel,
                                  out_channels=in_freq * channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.dropout2_2 = nn.Dropout(p=dropout)
        self.layer_norm2 = nn.LayerNorm(in_freq * channel, eps=1e-6)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`
        """
        bs, _, time, _ = xs.size()

        # first block
        residual = xs
        xs = self.conv2d(xs)
        xs = torch.relu(xs)
        if DROPOUT:
            xs = self.dropout1(xs)

        xs = xs + residual  # `[B, out_ch, T, feat_dim]`

        # layer normalization
        bs, out_ch, time, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`
        xs = self.layer_norm1(xs)
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)  # `[B, out_ch * feat_dim, T, 1]`

        # second block
        residual = xs
        xs = self.conv1d_1(xs)
        xs = torch.relu(xs)
        if DROPOUT:
            xs = self.dropout2_1(xs)
        xs = self.conv1d_2(xs)
        if DROPOUT:
            xs = self.dropout2_2(xs)
        xs = xs + residual  # `[B, out_ch * feat_dim, T, 1]`

        # layer normalization
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`
        xs = self.layer_norm2(xs)
        xs = xs.view(bs, time, out_ch, feat_dim).contiguous().transpose(2, 1)

        return xs


class SubsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_freq, dropout):
        super().__init__()

        self.conv1d = nn.Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=(2, 1),
                                stride=(2, 1),
                                padding=(0, 0))
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(in_freq * out_channel, eps=1e-6)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`
        """
        bs, _, time, _ = xs.size()

        xs = self.conv1d(xs)
        xs = torch.relu(xs)
        if DROPOUT:
            xs = self.dropout(xs)

        # layer normalization
        bs, out_ch, time, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`
        xs = self.layer_norm(xs)
        xs = xs.view(bs, time, out_ch, feat_dim).contiguous().transpose(2, 1)

        return xs


class Linear(nn.Module):

    def __init__(self, in_size, out_size, bias=True, dropout=0, weight_norm=False):
        """Linear layer with regularization.
        Args:
            in_size (int):
            out_size (int):
            bias (bool): if False, remove a bias term
            dropout (float):
            weight_norm (bool):
        """
        super(Linear, self).__init__()

        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, name='weight', dim=0)

    def forward(self, xs):
        """Forward pass.
        Args:
            xs (FloatTensor):
        Returns:
            xs (FloatTensor):
        """
        if DROPOUT:
            return self.dropout(self.fc(xs))
        return self.fc(xs)
