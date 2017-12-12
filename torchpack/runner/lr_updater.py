class LrUpdater(object):

    @staticmethod
    def fixed(epoch, base_lr):
        return base_lr

    @staticmethod
    def step(epoch, base_lr, step):
        return base_lr * (0.1**(epoch // step))

    @staticmethod
    def multistep(epoch, base_lr, steps):
        exp = len(steps)
        for i, step in enumerate(steps):
            if epoch < step:
                exp = i
                break
        return base_lr * 0.1**exp

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

    @staticmethod
    def update(optimizer, epoch, policy, **kwargs):
        if isinstance(policy, str):
            method = getattr(LrUpdater, policy)
        elif callable(policy):
            method = policy
        else:
            raise TypeError('"policy" must be a string or method')
        base_lr = optimizer.defaults['lr']
        lr = method(epoch, base_lr, **kwargs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
