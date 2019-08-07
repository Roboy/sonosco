import torch
import torch.nn as nn
import numpy as np


class DotAttention(nn.Module):

    def __init__(self, query_dim):
        super().__init__()

        self.scale = 1.0 / np.sqrt(query_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys, values):
        """

        :param queries: [B,T_dec,Q] (hidden state, decoder output, etc.)
        :param keys: [B,T_enc,K] (encoder outputs)
        :param values: [B,T_enc,V] (encoder outputs)
        :return: Tuple[summary vector [B,T_dec,V], scores [B,T_dec,T_enc]
        """
        assert queries.size()[-1] == keys.size()[-1]

        # compute scores
        keys = keys.permute(0, 2, 1)  # [B,T_enc,K] -> [B,K,T_enc]
        scores = torch.bmm(queries, keys)  # [B,T_dec,Q]*[B,K,T_enc] = [B,T_dec,T_enc]
        scores = self.softmax(scores.mul_(self.scale))

        # TODO: add soft window pre-training
        # weight values
        summaries = torch.bmm(scores, values)  # [B,T_dec,T_enc]*[B,T_enc,V] -> [B,T_dec,V]

        return summaries, scores
