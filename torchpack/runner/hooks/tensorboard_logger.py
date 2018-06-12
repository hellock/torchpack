from .logger import LoggerHook
from ..utils import master_only


class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir,
                 interval=10,
                 reset_meter=True,
                 ignore_last=True):
        super(TensorboardLoggerHook, self).__init__(interval, reset_meter,
                                                    ignore_last)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX '
                              'to use TensorboardLoggerHook.')
        else:
            self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        if runner.rank != 0:
            return
        for var in runner.outputs['log_vars']:
            tag = '{}/{}'.format(var, runner.mode)
            self.writer.add_scalar(tag, runner.meter.avg[var],
                                   runner.num_iters)

    @master_only
    def after_run(self, runner):
        self.writer.close()
