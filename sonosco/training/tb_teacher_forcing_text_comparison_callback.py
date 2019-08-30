import logging
import torch

from sonosco.model.serialization import serializable

from sonosco.training.abstract_callback import AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


@serializable
class TbTeacherForcingTextComparisonCallback(AbstractCallback):
    log_dir: str
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
        batch_x = batch_x.squeeze(1).transpose(1, 2)
        split_targets = torch.split(batch_y, target_lens.tolist())
        model_output, lens, loss = model(batch_x, input_lens, split_targets)

        transcriptions, decoded_offsets = decoder.decode(model_output, lens)
        groundtruths = decoder.convert_to_strings(split_targets)

        for transcription, groundtruth in zip(transcriptions, groundtruths):
            comparison = f"Teacher-Forching Transcription: {transcription[0]}. Groundtruth: {groundtruth}"
            LOGGER.info(comparison)
            self.writer.add_text("teacher_forcing_text_comparison", comparison, step)
