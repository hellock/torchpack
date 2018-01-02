from torchpack.runner.hooks import Hook


class BasicLoggerHook(Hook):

    def __init__(self, interval, reset_meter=True, ignore_last=True):
        self.interval = interval
        self.reset_meter = reset_meter
        self.ignore_last = ignore_last

    def _log(self, runner, progress_info):
        log_info = progress_info + (
            'Time {avg[batch_time]:.3f} (Data {avg[data_time]:.3f})\t'
            'Loss {avg[loss]:.4f}').format(avg=runner.meter.avg)
        if len(runner.outputs['log_vars']) > 1:
            loss_items = []
            for var in runner.outputs['log_vars']:
                if var == 'loss':
                    continue
                loss_items.append(
                    '{}: {:.4f}'.format(var, runner.meter.avg[var]))
            log_info += ' (' + ', '.join(loss_items) + ')'
        runner.logger.info(log_info)
        if self.reset_meter:
            runner.meter.reset()

    def after_train_iter(self, runner):
        if not self.every_n_inner_iters(runner, self.interval):
            if not self.end_of_epoch(runner):
                return
            elif self.ignore_last:
                return
        progress_info = 'Epoch [{}][{}/{}]\tlr: {:.5f}\t'.format(
            runner.epoch + 1, runner.num_epoch_iters + 1,
            len(runner.data_loader), runner.lr)
        self._log(runner, progress_info)

    def after_val_epoch(self, runner):
        progress_info = 'Epoch(val) [{}][{}]\t'.format(
            runner.epoch, runner.num_epoch_iters + 1)
        self._log(runner, progress_info)
