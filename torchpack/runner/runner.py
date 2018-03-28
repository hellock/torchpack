import logging
import os
import time

import torch
from torchpack.io import load_checkpoint, save_checkpoint
from torchpack.runner import hooks
from torchpack.runner.hooks import (Hook, LrUpdaterHook, CheckpointSaverHook,
                                    MeterHook, OptimizerStepperHook)


class Runner(object):

    def __init__(self,
                 model,
                 optimizer,
                 batch_processor,
                 work_dir,
                 log_level=logging.INFO):
        self.model = model
        self.optimizer = self.set_optimizer(optimizer)
        self.batch_processor = batch_processor
        self.work_dir = work_dir
        self.hooks = []
        self.logger = self.init_logger(work_dir, log_level)

        self.epoch = 0
        self.num_iters = 0
        self.num_epoch_iters = 0
        self.mode = None

    def set_optimizer(self, optimizer):
        if isinstance(optimizer, dict):
            optim_cls = getattr(torch.optim, optimizer['algorithm'])
            optimizer = optim_cls(self.model.parameters(), **optimizer['args'])
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(
                '"optimizer" must be either an Optimizer object or a dict')
        return optimizer

    def current_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

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

    def register_hook(self, hook):
        assert isinstance(hook, Hook)
        self.hooks.append(hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename)

    def save_checkpoint(self, out_dir, filename_tmpl='epoch_{}.pth'):
        save_checkpoint(
            self.model,
            self.epoch + 1,
            self.num_iters,
            out_dir=out_dir,
            filename_tmpl=filename_tmpl)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self.num_epoch_iters = i
            self.call_hook('before_train_iter')
            self.outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self.num_iters += 1
        self.call_hook('after_train_epoch')
        self.epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        for i, data_batch in enumerate(data_loader):
            self.num_epoch_iters = i
            self.call_hook('before_val_iter')
            self.outputs = self.batch_processor(
                self.model, data_batch, train_mode=False, **kwargs)
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch')

    def resume(self, checkpoint, resume_optimizer=True):
        checkpoint = self.load_checkpoint(checkpoint)
        self.epoch = checkpoint['epoch']
        self.num_iters = checkpoint['num_iters']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('resumed epoch %d, iter %d', self.epoch,
                         self.num_iters)

    def run(self, data_loaders, workflow, max_epoch, **kwargs):
        assert isinstance(data_loaders, list)
        self.logger.info('Start running, workflow: %s, max: %d epochs',
                         workflow, max_epoch)
        self.call_hook('before_run')
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
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_logger_hooks(self, log_config):
        self.register_hook(MeterHook())
        log_interval = log_config['interval']
        for logger_name, args in log_config['hooks']:
            if isinstance(logger_name, str):
                logger_cls = getattr(hooks, logger_name)
            elif isinstance(logger_name, type):
                logger_cls = logger_name
            else:
                raise TypeError(
                    'logger name must be a string of hook type, not {}'.format(
                        logger_name))
            kwargs = args.copy()
            kwargs['reset_meter'] = False
            if 'interval' not in kwargs:
                kwargs['interval'] = log_interval
            self.register_hook(logger_cls(**kwargs))
        self.hooks[-1].reset_meter = True

    def register_default_hooks(self,
                               lr_config,
                               grad_clip_config=None,
                               checkpoint_config=None,
                               log_config=None):
        """Register several default hooks"""
        if grad_clip_config is None:
            grad_clip_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_hook(LrUpdaterHook(**lr_config))
        self.register_hook(OptimizerStepperHook(**grad_clip_config))
        self.register_hook(CheckpointSaverHook(**checkpoint_config))
        if log_config is not None:
            self.register_logger_hooks(log_config)
