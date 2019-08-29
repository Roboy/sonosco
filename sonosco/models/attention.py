import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sonosco.config.global_settings import CUDA_ENABLED

SOFT_WINDOW_SIGMA = 4.0


class DotAttention(nn.Module):

    def __init__(self, query_dim):
        super().__init__()
        self.scale = 1.0 / np.sqrt(query_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys, values, soft_window_enabled: bool = True):
        """
        :param queries: [B,T_dec,Q] (hidden state, decoder output, etc.)
        :param keys: [B,T_enc,K] (encoder outputs)
        :param values: [B,T_enc,V] (encoder outputs)
        :param mask: [B,T_enc,K] (mask)
        :return: Tuple[summary vector [B,T_dec,V], scores [B,T_dec,T_enc]
        """
        assert queries.size()[-1] == keys.size()[-1]

        # compute scores
        keys = keys.permute(0, 2, 1)  # [B,T_enc,K] -> [B,K,T_enc]
        scores = torch.bmm(queries, keys)  # [B,T_dec,Q]*[B,K,T_enc] = [B,T_dec,T_enc]

        xv, yv = torch.meshgrid([torch.arange(0, scores.shape[1]), torch.arange(0, scores.shape[2])])
        xv, yv = xv.type(torch.float32), yv.type(torch.float32)

        if soft_window_enabled:
            soft_window = (1. / (2. * SOFT_WINDOW_SIGMA ** 2)) * (xv - (scores.shape[1] / scores.shape[2] * yv))
            if CUDA_ENABLED:
                soft_window = soft_window.cuda()
            scores = self.softmax(scores.mul_(self.scale) - soft_window)
        else:
            scores = self.softmax(scores.mul_(self.scale))

        # TODO: add soft window pre-training
        # weight values
        summaries = torch.bmm(scores, values)  # [B,T_dec,T_enc]*[B,T_enc,V] -> [B,T_dec,V]

        return summaries, scores


class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()
        # TODO: move this out of this class?
        # self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(
            attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)
        # # concat -> (N, To, 2*H)
        # concated = torch.cat((attention_output, queries), dim=2)
        # # TODO: Move this out of this class?
        # # output -> (N, To, H)
        # output = torch.tanh(self.linear_out(
        #     concated.view(-1, 2*hidden_size))).view(batch_size, -1, hidden_size)

        return attention_output, attention_distribution
