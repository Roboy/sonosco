import torch
import torch.nn.functional as F


def cross_entropy_loss(batch, model):
    batch_x, batch_y, input_lengths, target_lengths = batch
    # check out the _collate_fn in loader to understand the next transformations
    batch_x = batch_x.squeeze(1).transpose(1, 2)
    batch_y = torch.split(batch_y, target_lengths.tolist())
    model_output, lens, loss = model(batch_x, input_lengths, batch_y)
    return loss, (model_output, lens)


def torch_cross_entropy_loss(model_out, batch):
    model, ignore_id = model_out
    return F.cross_entropy(model_out, batch.view(-1),
                           ignore_index=ignore_id,
                           reduction='elementwise_mean')
