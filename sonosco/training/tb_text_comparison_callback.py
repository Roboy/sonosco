import logging

from sonosco.model.serialization import serializable

from sonosco.training.abstract_callback import AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


@serializable
class TbTextComparisonCallback(AbstractCallback):
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

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_lens:
            split_targets.append(batch_y[offset:offset + size])
            offset += size

        groundtruths = decoder.convert_to_strings(split_targets)

        transcriptions = list()

        for i in range(min(self.samples, batch_x.shape[0])):
            # import pdb; pdb.set_trace()
            sample_x, sample_len = batch_x[i].transpose(1, 2), input_lens[i]
            out, output_lens, attention = model(sample_x, sample_len)
            decoded_output, decoded_offsets = decoder.decode(out, output_lens)
            transcriptions.append(decoded_output)

        for transcription, groundtruth in zip(transcriptions, groundtruths):
            comparison = f"Transcription: {transcription[0]}. Groundtruth: {groundtruth}"
            LOGGER.info(comparison)
            self.writer.add_text("inference_text_comparison", comparison, step)
