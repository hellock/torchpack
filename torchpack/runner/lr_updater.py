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
