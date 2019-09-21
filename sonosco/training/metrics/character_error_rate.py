import logging
import torch

from typing import Tuple

LOGGER = logging.getLogger(__name__)


# TODO: restructure the code to not use the decoder
def character_error_rate(model_out: torch.Tensor, batch: Tuple, decoder=None) -> float:
    """
    Calculate character error rate based on the model output and groundtruth.

    Args:
        model_out: model output tensor
        batch: batch with groundtruth data
        decoder: decoder

    Returns: character error rate for the given output

    """
    inputs, targets, input_percentages, target_sizes = batch

    # unflatten targets
    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size

    out, output_sizes = model_out

    decoded_output, _ = decoder.decode(out, output_sizes)
    target_strings = decoder.convert_to_strings(split_targets)
    cer = 0
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        try:
            cer += decoder.cer(transcript, reference) / float(len(reference))
        except ZeroDivisionError:
            pass
    del out
    cer *= 100/len(target_strings)

    return cer
