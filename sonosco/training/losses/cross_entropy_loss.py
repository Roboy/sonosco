import torch
import torch.nn

from typing import Tuple, List


def cross_entropy_loss(batch: Tuple, model: torch.nn.Module) -> Tuple[float, Tuple[torch.Tensor, List]]:
    """
    Calculate cross entropy loss for a batch and a model.

    Args:
        batch: batch of groundtruth data
        model: torch model

    Returns: loss, model output, length of each sample in the output batch

    """
    batch_x, batch_y, input_lengths, target_lengths = batch
    # check out the _collate_fn in loader to understand the next transformations
    batch_x = batch_x.squeeze(1).transpose(1, 2)
    batch_y = torch.split(batch_y, target_lengths.tolist())
    model_output, lens, loss = model(batch_x, input_lengths, batch_y)
    return loss, (model_output, lens)
