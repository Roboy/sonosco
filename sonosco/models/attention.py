import torch
import torch.nn as nn
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
