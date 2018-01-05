from tensorboardX import SummaryWriter

from torchpack.runner.hooks import LoggerHook


class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir,
                 interval=10,
                 reset_meter=True,
                 ignore_last=True):
        super(TensorboardLoggerHook, self).__init__(interval, reset_meter,
                                                    ignore_last)
        self.writer = SummaryWriter(log_dir)

    def log(self, runner):
        for var in runner.outputs['log_vars']:
            tag = '{}/{}'.format(var, runner.mode)
            self.writer.add_scalar(tag, runner.meter.avg[var],
                                   runner.num_iters)

    def after_run(self, runner):
        self.writer.close()
