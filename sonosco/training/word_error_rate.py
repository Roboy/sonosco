import logging
import sys
import numpy as np
import torch
LOGGER = logging.getLogger(__name__)


def WER(model_out, batch, decoder):
    inputs, targets, input_percentages, target_sizes = batch
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

    # unflatten targets
    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size

    out, output_sizes = model_out

    decoded_output, _ = decoder.decode(out, output_sizes)
    target_strings = decoder.convert_to_strings(split_targets)
    wer = 0
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
    del out

    wer *= 100

    return wer