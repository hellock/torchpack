from torchpack.runner.hooks import LoggerHook


class TextLoggerHook(LoggerHook):

    def log(self, runner):
        if runner.mode == 'train':
            lr_str = ', '.join(
                ['{:.5f}'.format(lr) for lr in runner.current_lr()])
            log_info = 'Epoch [{}][{}/{}]\tlr: {}\t'.format(
                runner.epoch + 1, runner.num_epoch_iters + 1,
                len(runner.data_loader), lr_str)
        else:
            log_info = 'Epoch({}) [{}][{}]\t'.format(
                runner.mode, runner.epoch, runner.num_epoch_iters + 1)
        log_info += ('Time {avg[batch_time]:.3f} (Data {avg[data_time]:.3f})\t'
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
