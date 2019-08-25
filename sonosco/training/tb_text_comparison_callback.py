import logging
from sonosco.training.abstract_callback import  AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


class TbTextComparisonCallback(AbstractCallback):

    def __init__(self, log_dir: str, samples: int = 4):
        # samples should be less than batch size
        self.samples = samples
        self.writer = SummaryWriter(log_dir=log_dir)

    def __call__(self,
                 epoch,
                 step,
                 performance_measures,
                 context,
                 validation: bool = False):

        model = context.model
        decoder = context.decoder
        batch_x, batch_y, input_lens, target_lens = next(iter(context.val_data_loader))

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_lens:
            split_targets.append(batch_y[offset:offset + size])
            offset += size

        groundtruths = decoder.convert_to_strings(split_targets)

        transcriptions = list()

        for i in range(min(self.samples, batch_x.shape[0])):
            sample_x, sample_len = batch_x[i].transpose(1, 2), input_lens[i]
            out, output_lens, attention = model(sample_x, sample_len)
            decoded_output, decoded_offsets = decoder.decode(out, output_lens)
            transcriptions.append(decoded_output)

        for transcription, groundtruth in zip(transcriptions, groundtruths):
            comparison = f"Transcription: {transcription[0]}. Groundtruth: {groundtruth}"
            LOGGER.info(comparison)
            self.writer.add_text("text_comparison", comparison, step)
