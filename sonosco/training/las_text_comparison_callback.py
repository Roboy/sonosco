import logging
from typing import Dict

import torch

from sonosco.model.serialization import serializable

from sonosco.training.abstract_callback import AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


@serializable
class LasTextComparisonCallback(AbstractCallback):
    log_dir: str
    labels: str
    args: Dict[str, str]
    samples: int = 4

    def __post_init__(self):
        # samples should be less than batch size
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __call__(self,
                 epoch,
                 step,
                 performance_measures,
                 context,
                 validation: bool = False):
        if step == 0 or step % context.test_step > 0:
            return

        model = context.model
        decoder = context.decoder
        batch = next(iter(context.test_data_loader))
        batch = context._recursive_to_cuda(batch)
        batch_x, batch_y, input_lens, target_lens = batch
        transcriptions = []

        split_targets = []
        offset = 0
        for size in target_lens:
            split_targets.append(batch_y[offset:offset + size])
            offset += size

        groundtruths = decoder.convert_to_strings(split_targets)

        for i, el in enumerate(batch_x):
            transcriptions.append(model.recognize(el[0].transpose(0, 1), input_lens[i:i+1], self.labels, self.args)[0])

        for transcription, groundtruth in zip(transcriptions, groundtruths):
            trans = decoder.convert_to_strings(torch.tensor([transcription['yseq']]))
            comparison = f"Transcription: {trans}. Groundtruth: {groundtruth}"
            LOGGER.info(comparison)
            self.writer.add_text("inference_text_comparison", comparison, step)
