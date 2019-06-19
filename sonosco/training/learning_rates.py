import logging
import sys

from .abstract_callback import AbstractCallback


LOGGER = logging.getLogger(__name__)


class StepwiseLearningRateReduction(AbstractCallback):
    """
    Reduces the learning rate of the optimizer every N epochs.
    Args:
        epoch_steps (int): number of epochs after which learning rate is reduced
        reduction_factor (float): multiplicative factor for learning rate reduction
        min_lr (float): lower bound for learning rate
    """

    def __init__(self, epoch_steps, reduction_factor, min_lr=None):
        self._epoch_steps = epoch_steps
        self._reduction_factor = reduction_factor
        self._min_lr = min_lr

    def __call__(self, epoch, step, performance_measures, context):
        # execute at the beginning of every Nth epoch
        if epoch > 0 and step == 0 and epoch % self._epoch_steps == 0:

            # reduce lr for each param group (necessary for e.g. Adam)
            for param_group in context.optimizer.param_groups:
                new_lr = param_group['lr'] * self._reduction_factor

                if self._min_lr is not None and new_lr < self._min_lr:
                    continue

                param_group['lr'] = new_lr
                LOGGER.info("Epoch {}: Reducing learning rate to {}".format(epoch, new_lr))


class ScheduledLearningRateReduction(AbstractCallback):
    """
    Reduces the learning rate of the optimizer for every scheduled epoch.
    Args:
        epoch_schedule (list of int): defines at which epoch the learning rate will be reduced
        reduction_factor (float): multiplicative factor for learning rate reduction
        min_lr (float): lower bound for learning rate
    """

    def __init__(self, epoch_schedule, reduction_factor, min_lr=None):
        self._epoch_schedule = sorted(epoch_schedule)
        self._reduction_factor = reduction_factor
        self._min_lr = min_lr

    def __call__(self, epoch, step, performance_measures, context):

        if not self._epoch_schedule:    # stop if schedule is empty
            return

        next_epoch_step = self._epoch_schedule[0]
        if epoch >= next_epoch_step and step == 0:

            # reduce lr for each param group (necessary for e.g. Adam)
            for param_group in context.optimizer.param_groups:
                new_lr = param_group['lr'] * self._reduction_factor

                if self._min_lr is not None and new_lr < self._min_lr:
                    continue

                param_group['lr'] = new_lr
                LOGGER.info("Epoch {}: Reducing learning rate to {}".format(epoch, new_lr))

            self._epoch_schedule.pop(0)


class ReduceLROnPlateau(AbstractCallback):
    """
    Reduce the learning rate if the train or validation loss plateaus.
    Args:
        monitor (string): name of the relevant loss or metric (usually 'val_loss')
        factor (float): factor by which the lr is decreased at each step
        patience (int): number of epochs to wait on plateau for loss improvement before reducing lr
        min_delta (float): minimum improvement necessary to reset patience
        cooldown (int): number of epochs to cooldown after a lr reduction
        min_lr (float): minimum value the learning rate can decrease to
        verbose (bool): print to console
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10, min_delta=0, cooldown=0, min_lr=0, verbose=False):
        self.monitor = monitor
        if factor >= 1.0 or factor < 0:
            raise ValueError('ReduceLROnPlateau does only support a factor in [0,1[.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best_loss = sys.float_info.max

    def __call__(self, epoch, step, performance_measures, context):

        if self.monitor not in performance_measures:
            return

        if step != len(context.train_data_loader)-1: # only continue at end of epoch
            return

        if self.cooldown_counter > 0:   # in cooldown phase
            self.cooldown_counter -= 1
            self.wait = 0

        current_loss = performance_measures[self.monitor]
        if (self.best_loss - current_loss) >= self.min_delta:   # loss improved, save and reset wait counter
            self.best_loss = current_loss
            self.wait = 0

        elif self.cooldown_counter <= 0:    # no improvement and not in cooldown

            if self.wait >= self.patience:  # waited long enough, reduce lr
                for param_group in context.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * self.factor
                    if new_lr >= self.min_lr:   # only decrease if there is still enough buffer space
                        if self.verbose:
                            LOGGER.info("Epoch {}: Reducing learning rate from {} to {}".format(epoch, old_lr, new_lr)) #TODO print per param group?
                        param_group['lr'] = new_lr
                self.cooldown_counter = self.cooldown   # new cooldown phase after lr reduction
                self.wait = 0
            else:
                self.wait += 1
