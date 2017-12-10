from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = defaultdict(float)
        self.sum = defaultdict(float)
        self.avg = defaultdict(float)
        self.count = defaultdict(int)
        self.reset()

    def reset(self, keys=None):
        if isinstance(keys, str):
            keys = [keys]
        elif keys is None:
            keys = self.val.keys()
        for k in keys:
            self.val[k] = 0
            self.sum[k] = 0
            self.avg[k] = 0
            self.count[k] = 0

    def update(self, pairs, n=1):
        for k, v in pairs.items():
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]
