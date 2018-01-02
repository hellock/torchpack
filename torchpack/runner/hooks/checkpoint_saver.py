from torchpack.io import save_checkpoint
from torchpack.runner.hooks import Hook


class CheckpointSaverHook(Hook):

    def __init__(self, interval, out_dir=None, **kwargs):
        self.interval = interval
        self.out_dir = out_dir
        self.args = kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        save_checkpoint(runner.model, runner.epoch + 1, runner.num_iters,
                        self.out_dir, **self.args)
