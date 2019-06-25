import os.path
from random import random
from datetime import datetime

import numpy as np
import torch

from models.deepspeech2 import DeepSpeech2

import json
import os
import random
import time

import torch.distributed as dist
import torch.utils.data.distributed

try:
    from apex.fp16_utils import FP16_Optimizer
    from apex.parallel import DistributedDataParallel
except Exception as e:
    print(f"Apex import failed: {e}")

from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from loader import AudioDataLoader, AudioDataset, BucketingSampler, DistributedBucketingSampler
from decoders.greedy_decoder import GreedyDecoder
from utils import convert_model_to_half, reduce_tensor, check_loss

models = {"deepspeech2": DeepSpeech2}

sttime = datetime.now()
print(f"Time of start: {sttime}")


def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelWrapper(object):
    DEF_PATH = "examples/checkpoints/"

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", models["deepspeech2"])

        if kwargs.get("continue"):
            path = kwargs.get("from", ModelWrapper.get_default_path())
            self.model.package = torch.load(path, map_location=lambda storage, loc: storage)
            self.model.load_model(path)

        self.save_path = kwargs.get("save", ModelWrapper.DEF_PATH + str(datetime.now().timestamp()))

        self.cuda = kwargs.get("cuda")
        self.apex = kwargs.get("apex") if self.cuda else False
        self.half = self.apex if self.apex else kwargs.get("half")

    def train(self, seed, cuda, mixed_precision, world_size, gpu_rank, rank, save_folder, dist_backend, dist_url,
              epochs, continue_from, finetune, labels_path, sample_rate, window_size, window_stride, window,
              hidden_size, hidden_layers, labels, supported_rnns, bidirectional, no_shuffle, no_sorta_grad, rnn_type,
              train_manifest, augment, batch_size, num_workers, momentum, lr, static_loss_scale, dynamic_loss_scale,
              val_manifest, max_norm, silent, checkpoint_per_batch, checkpoint, learning_anneal, model_path):
        # Set seeds for determinism
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        device = torch.device("cuda" if cuda else "cpu")
        if mixed_precision and not cuda:
            raise ValueError('If using mixed precision training, CUDA must be enabled!')
        distributed = world_size > 1
        main_proc = True
        device = torch.device("cuda" if cuda else "cpu")
        if distributed:
            if gpu_rank:
                torch.cuda.set_device(int(gpu_rank))
            dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                    world_size=world_size, rank=rank)
            main_proc = rank == 0  # Only the first proc should save models
        save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

        loss_results, cer_results, wer_results = torch.Tensor(epochs), torch.Tensor(epochs), torch.Tensor(epochs)
        best_wer = None

        avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None
        if continue_from:  # Starting from previous model
            print("Loading checkpoint model %s" % continue_from)

            labels = self.model.labels
            audio_conf = self.model.audio_conf
            if not finetune:  # Don't want to restart training
                optim_state = self.model.package['optim_dict']
                start_epoch = int(self.model.get('epoch', 1)) - 1  # Index start at 0 for training
                start_iter = self.model.package.get('iteration', None)
                if start_iter is None:
                    start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                    start_iter = 0
                else:
                    start_iter += 1
                avg_loss = int(self.model.package.get('avg_loss', 0))
                loss_results, cer_results, wer_results = self.model.package['loss_results'], \
                                                         self.model.package['cer_results'], \
                                                         self.model.package['wer_results']
        else:
            with open(labels_path) as label_file:
                labels = str(''.join(json.load(label_file)))

            audio_conf = dict(sample_rate=sample_rate,
                              window_size=window_size,
                              window_stride=window_stride,
                              window=window)

            rnn_type = rnn_type.lower()
            assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
            model = self.model(rnn_hidden_size=hidden_size,
                               nb_layers=hidden_layers,
                               labels=labels,
                               rnn_type=supported_rnns[rnn_type],
                               audio_conf=audio_conf,
                               bidirectional=bidirectional,
                               mixed_precision=mixed_precision)

        decoder = GreedyDecoder(labels)

        train_dataset = AudioDataset(audio_conf=audio_conf, manifest_filepath=train_manifest, labels=labels,
                                     normalize=False, augment=augment)
        test_dataset = AudioDataset(audio_conf=audio_conf, manifest_filepath=val_manifest, labels=labels,
                                    normalize=False, augment=False)
        if not distributed:
            train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
        else:
            train_sampler = DistributedBucketingSampler(train_dataset, batch_size=batch_size,
                                                        num_replicas=world_size, rank=rank)

        train_loader = AudioDataLoader(train_dataset, num_workers=num_workers, batch_sampler=train_sampler)
        test_loader = AudioDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        if (not no_shuffle and start_epoch != 0) or no_sorta_grad:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle(start_epoch)

        model = model.to(device)
        if mixed_precision:
            model = convert_model_to_half(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=lr,
                                    momentum=momentum, nesterov=True, weight_decay=1e-5)
        if distributed:
            model = DistributedDataParallel(model)
        if mixed_precision:
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=static_loss_scale,
                                       dynamic_loss_scale=dynamic_loss_scale)
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
        print(model)
        print("Number of parameters: %d" % self.model.get_param_size(model))

        criterion = CTCLoss()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for epoch in range(start_epoch, epochs):
            model.train()
            end = time.time()
            start_epoch_time = time.time()
            for i, (data) in enumerate(train_loader, start=start_iter):
                if i == len(train_sampler):
                    break
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                # measure data loading time
                data_time.update(time.time() - end)
                inputs = inputs.to(device)

                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH

                float_out = out.float()  # ensure float32 for loss
                loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                loss = loss / inputs.size(0)  # average the loss by minibatch

                if distributed:
                    loss = loss.to(device)
                    loss_value = reduce_tensor(loss, world_size).item()
                else:
                    loss_value = loss.item()

                # Check to ensure valid loss was calculated
                valid_loss, error = check_loss(loss, loss_value)
                if valid_loss:
                    optimizer.zero_grad()
                    # compute gradient
                    if mixed_precision:
                        optimizer.backward(loss)
                        optimizer.clip_master_grads(max_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                else:
                    print(error)
                    print('Skipping grad update')
                    loss_value = 0

                avg_loss += loss_value
                losses.update(loss_value, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if not silent:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time,
                        loss=losses))
                if checkpoint_per_batch > 0 and i > 0 and (i + 1) % checkpoint_per_batch == 0 and main_proc:
                    file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                    print("Saving checkpoint model to %s" % file_path)
                    torch.save(self.model.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                    loss_results=loss_results,
                                                    wer_results=wer_results, cer_results=cer_results,
                                                    avg_loss=avg_loss),
                               file_path)
                del loss, out, float_out

            avg_loss /= len(train_sampler)

            epoch_time = time.time() - start_epoch_time
            print(f"Elapsed time from start: {datetime.now() - sttime}")
            print('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

            start_iter = 0  # Reset start iteration for next epoch
            total_cer, total_wer = 0, 0
            model.eval()
            with torch.no_grad():
                for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    inputs, targets, input_percentages, target_sizes = data
                    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                    inputs = inputs.to(device)

                    # unflatten targets
                    split_targets = []
                    offset = 0
                    for size in target_sizes:
                        split_targets.append(targets[offset:offset + size])
                        offset += size

                    out, output_sizes = model(inputs, input_sizes)

                    decoded_output, _ = decoder.decode(out, output_sizes)
                    target_strings = decoder.convert_to_strings(split_targets)
                    wer, cer = 0, 0
                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer += decoder.cer(transcript, reference) / float(len(reference))
                    total_cer += cer
                    total_wer += wer
                    del out
                wer = total_wer / len(test_loader.dataset)
                cer = total_cer / len(test_loader.dataset)
                wer *= 100
                cer *= 100
                loss_results[epoch] = avg_loss
                wer_results[epoch] = wer
                cer_results[epoch] = cer
                print('Validation Summary Epoch: [{0}]\t'
                      'Average WER {wer:.3f}\t'
                      'Average CER {cer:.3f}\t'.format(
                    epoch + 1, wer=wer, cer=cer))

            values = {
                'loss_results': loss_results,
                'cer_results': cer_results,
                'wer_results': wer_results
            }

            if main_proc and checkpoint:
                file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
                torch.save(self.model.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results),
                           file_path)
            # anneal lr
            param_groups = optimizer.optimizer.param_groups if mixed_precision else optimizer.param_groups
            for g in param_groups:
                g['lr'] = g['lr'] / learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            if main_proc and (best_wer is None or best_wer > wer):
                print("Found better validated model, saving to %s" % model_path)
                torch.save(self.model.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results), model_path)
                best_wer = wer

                avg_loss = 0
                if not no_shuffle:
                    print("Shuffling batches...")
                    train_sampler.shuffle(epoch)

    def validate(self):
        pass

    def test(self):
        torch.set_grad_enabled(False)
        device = torch.device("cuda" if cuda else "cpu")
        model = load_model(device, model_path, cuda)

        if decoder == "beam":
            from decoder import BeamCTCDecoder

            decoder = BeamCTCDecoder(model.labels, lm_path=lm_path, alpha=alpha, beta=beta,
                                     cutoff_top_n=cutoff_top_n, cutoff_prob=cutoff_prob,
                                     beam_width=beam_width, num_processes=lm_workers)
        elif decoder == "greedy":
            decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        else:
            decoder = None
        target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        test_dataset = AudioDataset(audio_conf=model.audio_conf, manifest_filepath=test_manifest,
                                    labels=model.labels, normalize=True)
        test_loader = AudioDataLoader(test_dataset, batch_size=batch_size,
                                      num_workers=num_workers)
        total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
        output_data = []
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out, output_sizes = model(inputs, input_sizes)

            if save_output:
                # add output to data array, and continue
                output_data.append((out.cpu().numpy(), output_sizes.numpy()))

            decoded_output, _ = decoder.decode(out, output_sizes)
            target_strings = target_decoder.convert_to_strings(split_targets)
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                wer_inst = decoder.wer(transcript, reference)
                cer_inst = decoder.cer(transcript, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference)
                if verbose:
                    print("Ref:", reference.lower())
                    print("Hyp:", transcript.lower())
                    print("WER:", float(wer_inst) / len(reference.split()), "CER:", float(cer_inst) / len(reference),
                          "\n")

        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
        if save_output:
            np.save(output_path, output_data)

    def infer(self, sound):
        pass

    @staticmethod
    def get_default_path(def_path: str) -> str:
        """
        Returns the path to the latest checkpoint in the default location
        :param def_path: default path where checkpoints are stored
        :return: the path to the latest checkpoint
        """
        latest_subdir = max([os.path.join(def_path, d) for d in os.listdir(def_path)], key=os.path.getmtime)
        default = latest_subdir + "/final.pth"
        return default

    def print_training_info(self, epoch, loss, cer, wer):
        print(f"\nTraining Information\n " + \
              f"- Epoch:\t{epoch}\n " + \
              f"- Current Loss:\t{loss}\n " + \
              f"- Current CER: \t{cer}\n" + \
              f"- Current WER: \t{wer}")
