import math
from collections import OrderedDict
from dataclasses import field
from typing import Dict
from torch import nn

from sonosco.models.modules import MaskConv, BatchRNN, SequenceWise, InferenceBatchSoftmax
from sonosco.model.serialization import serializable


@serializable
class DeepSpeech2(nn.Module):
    rnn_type: type = nn.LSTM
    labels: str = "abc"
    rnn_hid_size: int = 768
    nb_layers: int = 5
    audio_conf: Dict[str, str] = field(default_factory=dict)
    bidirectional: bool = True
    version: str = '0.0.1'

    def __post_init__(self):
        super(DeepSpeech2, self).__init__()
        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_in_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_in_size = int(math.floor(rnn_in_size + 2 * 20 - 41) / 2 + 1)
        rnn_in_size = int(math.floor(rnn_in_size + 2 * 10 - 21) / 2 + 1)
        rnn_in_size *= 32

        rnns = [('0', BatchRNN(input_size=rnn_in_size, hidden_size=self.rnn_hid_size, rnn_type=self.rnn_type, batch_norm=False))]
        rnns.extend([(f"{x + 1}", BatchRNN(input_size=self.rnn_hid_size, hidden_size=self.rnn_hid_size, rnn_type=self.rnn_type))
                     for x in range(self.nb_layers - 1)])
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.rnn_hid_size),
            nn.Linear(self.rnn_hid_size, num_classes, bias=False)
        )

        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        # if x.is_cuda and self.mixed_precision:
        #     x = x.half()
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    def __repr__(self):
        rep = f"DeepSpeech2 version: {self.version}\n" + \
              "=======================================" + \
              "Recurrent Neural Network Properties\n" + \
              f"  RNN Type:  \t{self.rnn_type.__name__.lower()}\n" + \
              f"  RNN Layers:\t{self.hidden_layers}\n" + \
              f"  RNN Size:  \t{self.hidden_size}\n" + \
              f"  Classes:   \t{len(self.labels)}\n" + \
              "---------------------------------------\n" + \
              "Model Features\n" + \
              f"  Labels:       \t{self.labels}\n" + \
              f"  Sample Rate:  \t{self.audio_conf.get('sample_rate', 'n/a')}\n" + \
              f"  Window Type:  \t{self.audio_conf.get('window', 'n/a')}\n" + \
              f"  Window Size:  \t{self.audio_conf.get('window_size', 'n/a')}\n" + \
              f"  Window Stride:\t{self.audio_conf.get('window_stride', 'n/a')}"
        return rep
