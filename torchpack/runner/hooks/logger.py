from torchpack.runner.hooks import Hook


class LoggerHook(Hook):
    """Base class for logger hooks."""

    def __init__(self, interval=10, reset_meter=True, ignore_last=True):
        self.interval = interval
        self.reset_meter = reset_meter
        self.ignore_last = ignore_last

    def log(self, runner):
        pass

    def log_and_reset(self, runner):
        self.log(runner)
        if self.reset_meter:
            runner.meter.reset()

    def after_train_iter(self, runner):
        if not self.every_n_inner_iters(runner, self.interval):
            if not self.end_of_epoch(runner):
                return
            elif self.ignore_last:
                return
        self.log_and_reset(runner)

    def after_val_epoch(self, runner):
        self.log_and_reset(runner)
