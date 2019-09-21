import logging
import torch

from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from sonosco.serialization import serializable
from ..abstract_callback import AbstractCallback, ModelTrainer

LOGGER = logging.getLogger(__name__)


@serializable
class TbTeacherForcingTextComparisonCallback(AbstractCallback):
    """
    Perform decoding using teacher forcing and compare predictions to groundtruth and visualize in tensorboard.

    Args:
        log_dir: tensorboard output directory
        samples: number of samples to compare and visualize at a time

    """
    log_dir: str
    samples: int = 4

    def __post_init__(self) -> None:
        """
        Post initialization.
        """
        # samples should be less than batch size
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __call__(self,
                 epoch: int,
                 step: int,
                 performance_measures: Dict,
                 context: ModelTrainer,
                 validation: bool = False) -> None:
        """
        Execute teacher forcing text comparison during inference callback.

        Args:
            epoch: epoch step
            step: step inside of the epoch
            performance_measures: performance measures dictionary
            context: model trainer
            validation: should validation dataloader be used for comparison

        """
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
