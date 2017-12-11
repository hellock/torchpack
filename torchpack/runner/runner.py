import logging
import os
import time
from collections import defaultdict
from enum import Enum

import torch
from torchpack.io import load_checkpoint, save_checkpoint
from torchpack.runner.lr_updater import LrUpdater
from torchpack.runner.meters import AverageMeter


class Signal(Enum):

    PRE_TRAIN_EPOCH = ('epoch', 'pre_train_epoch')
    PRE_VAL_EPOCH = ('epoch', 'pre_val_epoch')
    POST_TRAIN_EPOCH = ('epoch', 'post_train_epoch')
    POST_VAL_EPOCH = ('epoch', 'post_val_epoch')
    PRE_TRAIN_ITER = ('iter', 'pre_train_iter')
    PRE_VAL_ITER = ('iter', 'pre_val_iter')
    POST_TRAIN_ITER = ('iter', 'post_train_iter')
    POST_VAL_ITER = ('iter', 'post_val_iter')
    # signal bundles
    PRE_EPOCH = [('epoch', 'pre_train_epoch'), ('epoch', 'pre_val_epoch')]
    POST_EPOCH = [('epoch', 'post_train_epoch'), ('epoch', 'post_val_epoch')]
    PRE_ITER = [('iter', 'pre_train_iter'), ('iter', 'pre_val_iter')]
    POST_ITER = [('iter', 'post_train_iter'), ('iter', 'post_val_iter')]


class Runner(object):

    def __init__(self,
                 model,
                 optimizer_config,
                 batch_processor,
                 work_dir,
                 log_level=logging.INFO):
        self.model = model
        self.optimizer = self.set_optimizer(optimizer_config)
        self.batch_processor = batch_processor
        self.work_dir = work_dir
        self.triggers = defaultdict(list)
        self.logger = self.init_logger(work_dir, log_level)

        self.epoch = 0
        self.num_iters = 0
        self.num_epoch_iters = 0
        self.mode = None

    def set_optimizer(self, config):
        if isinstance(config['algorithm'], str):
            optim_cls = getattr(torch.optim, config['algorithm'])
        elif isinstance(config['algorithm'], torch.optim.Optimizer):
            optim_cls = config['algorithm']
        else:
            raise ValueError('"{}" is not an implemented optimizer algorithm'.
                             format(config['algorithm']))
        optimizer = optim_cls(self.model.parameters(), **config['params'])
        return optimizer

    def init_logger(self, log_dir=None, level=logging.INFO):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            filename = time.strftime('%Y%m%d_%H%M%S',
                                     time.localtime()) + '.log'
            log_file = os.path.join(log_dir, filename)
            logger.addHandler(logging.FileHandler(log_file, 'w'))
        return logger

    def trigger(self, signal):
        if signal in self.triggers:
            measure = self.epoch + 1 if signal.value[
                0] == 'epoch' else self.num_epoch_iters + 1
            for trigger in self.triggers[signal]:
                func = trigger['trigger']
                kwargs = trigger['kwargs']
                interval = trigger['interval']
                if measure % interval == 0:
                    func(self, **kwargs)

    def register_trigger(self, signal, trigger, interval=1, **kwargs):
        if not isinstance(signal, Signal):
            raise TypeError(
                '"signal" must be a Signal, not {} type'.format(type(signal)))
        if isinstance(signal.value, list):
            for s in signal.value:
                self.register_trigger(self,
                                      Signal(s), trigger, interval, **kwargs)
        else:
            assert callable(trigger)
            assert isinstance(interval, int) and interval > 0
            self.triggers[signal].append({
                'trigger': trigger,
                'kwargs': kwargs,
                'interval': interval
            })

    def load_checkpoint(self, filename):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename)

    def save_checkpoint(self, dir, filename_tmpl='epoch_{}.pth'):
        save_checkpoint(
            self.model,
            self.epoch + 1,
            self.num_iters,
            our_dir=dir,
            filename_tmpl=filename_tmpl)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.trigger(Signal.PRE_TRAIN_EPOCH)
        self.meter = AverageMeter()
        t = time.time()
        for i, data_batch in enumerate(data_loader):
            # measure data loading time
            self.meter.update({'data_time': time.time() - t})
            # iter number in this epoch
            self.num_epoch_iters = i
            # send a PRE_TRAIN_ITER signal
            self.trigger(Signal.PRE_TRAIN_ITER)
            # process a batch of data
            self.outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            # update meter
            self.meter.update(self.outputs['log_vars'],
                              self.outputs['num_samples'])
            # measure batch iteration time
            self.meter.update({'batch_time': time.time() - t})
            t = time.time()
            # send a POST_TRAIN_ITER signal
            self.trigger(Signal.POST_TRAIN_ITER)

            self.num_iters += 1

        self.trigger(Signal.POST_TRAIN_EPOCH)
        self.epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.trigger(Signal.PRE_VAL_EPOCH)
        self.meter = AverageMeter()
        t = time.time()
        for i, data_batch in enumerate(data_loader):
            # measure data loading time
            self.meter.update({'data_time': time.time() - t})
            self.num_epoch_iters = i
            self.trigger(Signal.PRE_VAL_ITER)
            self.outputs = self.batch_processor(
                self.model, data_batch, train_mode=False, **kwargs)
            self.meter.update(self.outputs['log_vars'],
                              self.outputs['num_samples'])
            # measure elapsed time
            self.meter.update({'batch_time': time.time() - t})
            t = time.time()
            self.trigger(Signal.POST_VAL_ITER)
        self.trigger(Signal.POST_VAL_EPOCH)

    def resume(self, checkpoint):
        checkpoint = self.load_checkpoint(checkpoint)
        self.epoch = checkpoint['epoch'] + 1
        self.num_iters = checkpoint['num_iters']
        self.logger.info('resume from checkpoint %s, epoch %d, iter %d',
                         checkpoint, self.epoch, self.num_iters)

    def run(self, data_loaders, workflow, max_epoch, **kwargs):
        assert isinstance(data_loaders, list)
        while self.epoch < max_epoch:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run an epoch'.
                        format(mode))
                epoch_runner = getattr(self, mode)
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epoch:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

    def _register_lr_updater(self, lr_config):

        def update_lr(runner, **kwargs):
            runner.lr = LrUpdater.update(runner.optimizer, runner.epoch,
                                         **kwargs)

        assert 'policy' in lr_config
        self.register_trigger(Signal.PRE_TRAIN_EPOCH, update_lr, **lr_config)

    def _register_checkpoint(self, checkpoint_config):

        def checkpoint(runner, **kwargs):
            save_checkpoint(runner.model, runner.epoch, runner.num_iters,
                            **kwargs)

        checkpoint_config['out_dir'] = self.work_dir
        self.register_trigger(Signal.POST_TRAIN_EPOCH, checkpoint,
                              **checkpoint_config)

    def _register_bp(self):

        def back_propagate(runner):
            runner.optimizer.zero_grad()
            runner.outputs['loss'].backward()
            runner.optimizer.step()

        self.register_trigger(Signal.POST_TRAIN_ITER, back_propagate)

    def _register_loss_logger(self, log_config):

        def log_loss(runner):
            if runner.mode == 'train':
                log_info = 'Epoch [{}][{}/{}]\tlr: {:.5f}\t'.format(
                    runner.epoch + 1, runner.num_epoch_iters + 1,
                    len(runner.data_loader), runner.lr)
            elif runner.mode == 'val':
                log_info = 'Epoch(val) [{}]\t'.format(runner.epoch + 1)

            log_info += (
                'Time {avg[batch_time]:.3f} (Data {avg[data_time]:.3f})\t'
                'Loss {avg[loss]:.4f}').format(avg=runner.meter.avg)
            if len(runner.outputs['log_vars']) > 1:
                loss_items = []
                for var in runner.outputs['log_vars']:
                    if var == 'loss':
                        continue
                    else:
                        loss_items.append(
                            '{}: {:.4f}'.format(var, runner.meter.avg[var]))
                log_info += ' (' + ', '.join(loss_items) + ')'
            runner.logger.info(log_info)

        self.register_trigger(
            Signal.POST_TRAIN_ITER, log_loss, interval=log_config['interval'])
        self.register_trigger(Signal.POST_VAL_EPOCH, log_loss)

    def _register_reset_meter(self, interval):
        self.register_trigger(
            Signal.POST_TRAIN_ITER,
            lambda runner: runner.meter.reset(),
            interval=interval)

    def default_triggers(self,
                         lr_config,
                         checkpoint_config,
                         log_config,
                         reset_meter=True):
        """Register several default triggers"""
        self._register_lr_updater(lr_config)
        self._register_checkpoint(checkpoint_config)
        self._register_bp()
        self._register_loss_logger(log_config)
        if reset_meter:
            self._register_reset_meter(log_config['interval'])
