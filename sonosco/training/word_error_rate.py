import logging

LOGGER = logging.getLogger(__name__)


def word_error_rate(model_out, batch, decoder=None):
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
    wer = 0
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        try:
            wer += decoder.wer(transcript, reference) / float(len(reference.split()))
        except ZeroDivisionError:
            pass
    del out

    wer *= 100/len(target_strings)
    return wer
