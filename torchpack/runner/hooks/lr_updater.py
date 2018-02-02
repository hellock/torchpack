from __future__ import division

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

    def __init__(self,
                 policy,
                 warm_up=None,
                 warm_up_iters=0,
                 warm_up_ratio=0.1,
                 **kwargs):
        # validate the "policy" argument
        if isinstance(policy, str):
            update_fn = getattr(LrUpdater, policy)
        elif callable(policy):
            update_fn = policy
        else:
            raise TypeError('"policy" must be a method name or method')
        # validate the "warm_up" argument
        if warm_up is not None:
            if warm_up not in ['constant', 'linear']:
                raise ValueError(
                    '"{}" is not supported for warming up, currently supported'
                    ' types are "constant" and "linear"'.format(warm_up))
        if warm_up is not None:
            assert warm_up_iters > 0, \
                '"warm_up_iters" must be a positive integer'
            assert 0 < warm_up_ratio <= 1.0, \
                '"warm_up_ratio" must be in range (0,1]'

        self.update_fn = update_fn
        self.warm_up = warm_up
        self.warm_up_iters = warm_up_iters
        self.warm_up_ratio = warm_up_ratio
        self.update_args = kwargs

        self.normal_lr = 0

    def _set_lr(self, runner, lr):
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = lr
        runner.lr = lr

    def before_train_epoch(self, runner):
        base_lr = runner.optimizer.defaults['lr']
        lr = self.update_fn(runner.epoch, base_lr, **self.update_args)
        self._set_lr(runner, lr)
        self.normal_lr = lr

    def before_train_iter(self, runner):
        if self.warm_up is None:
            return
        cur_iters = runner.num_iters
        if cur_iters < self.warm_up_iters:
            if self.warm_up_type == 'constant':
                # only need to set lr at the first iteration of each epoch
                if runner.num_epoch_iters != 0:
                    return
                lr = self.normal_lr * self.warm_up_ratio
            elif self.warm_up_type == 'linear':
                k = (1 - cur_iters / self.warm_up_iters) * (
                    1 - self.warm_up_ratio)
                lr = self.normal_lr * (1 - k)
            self._set_lr(runner, lr)
        elif cur_iters == self.warm_up_iters:
            self._set_lr(runner, self.normal_lr)
