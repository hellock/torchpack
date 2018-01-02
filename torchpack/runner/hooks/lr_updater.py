from torchpack.runner.hooks import Hook


class LrUpdater(object):

    @staticmethod
    def fixed(epoch, base_lr):
        return base_lr

    @staticmethod
    def step(epoch, base_lr, step, gamma=0.1):
        if isinstance(step, int):
            return base_lr * (gamma**(epoch // step))
        assert isinstance(step, list)
        for s in step:
            assert isinstance(s, int)
        exp = len(step)
        for i, s in enumerate(step):
            if epoch < s:
                exp = i
                break
        return base_lr * gamma**exp

    @staticmethod
    def exp(epoch, base_lr, gamma):
        return base_lr * gamma**epoch

    @staticmethod
    def custom(epoch, base_lr, multipliers):
        assert isinstance(multipliers, list)
        for m in multipliers:
            assert isinstance(m, tuple)
        multiplier = 1
        for step, m in multipliers:
            if epoch < step:
                break
            multiplier = m
        return base_lr * multiplier


class LrUpdaterHook(Hook):

    def __init__(self, policy, **kwargs):
        if isinstance(policy, str):
            update_fn = getattr(LrUpdater, policy)
        elif callable(policy):
            update_fn = policy
        else:
            raise TypeError('"policy" must be a method name or method')
        self.update_fn = update_fn
        self.update_args = kwargs

    def before_train_epoch(self, runner):
        base_lr = runner.optimizer.defaults['lr']
        lr = self.update_fn(runner.epoch, base_lr, **self.update_args)
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = lr
        runner.lr = lr
