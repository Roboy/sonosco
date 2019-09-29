.. _metrics:

Metrics
========

You can define metrics that get evaluated every epoch step. An example metric is the word error:
::
   def word_error_rate(model_out: torch.Tensor, batch: Tuple, context=None) -> float:
      inputs, targets, input_percentages, target_sizes = batch

      # unflatten targets
      split_targets = []
      offset = 0
      for size in target_sizes:
          split_targets.append(targets[offset:offset + size])
          offset += size

      out, output_sizes = model_out

      decoded_output, _ = context.decoder.decode(out, output_sizes)
      target_strings = context.decoder.convert_to_strings(split_targets)
      wer = 0
      for x in range(len(target_strings)):
          transcript, reference = decoded_output[x][0], target_strings[x][0]
          try:
              wer += decoder.wer(transcript, reference) / float(len(reference.split()))
          except ZeroDivisionError:
              pass
      del out

      wer *= 100.0 / len(target_strings)
      return wer

The metric is some arbitrary function that gets the model output, the batch and the context,
which is the modeltrainer, so that within the metric you can access all parameters of the modeltrainer.
The metric returns then the metric to the model trainer, that prints it out every epoch step and can be used
from within the [Callbacks](#callbacks).

Sonosco already provides predefined metrics, such as Character Error Rate (CER) and Word Error Rate (WER).
