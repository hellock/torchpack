import torch
from torch.utils.data.sampler import Sampler


class MultiSourceRandomSampler(Sampler):

    def __init__(self, sources, weights=None, shuffle=True):
        self.sources = sources
        self.weights = weights
        self.shuffle = shuffle
        if weights is not None:
            assert isinstance(weights, list)
            assert len(weights) == len(sources)
            for w in weights:
                assert w >= 0 and w <= 1
            self.num_samples = sum([
                int(round(src_num * w))
                for src_num, w in zip(sources, weights)
            ])
        else:
            self.num_samples = sum([len(src) for src in sources])

    def __iter__(self):
        inds = []
        cnt = 0
        for src_num in self.sources:
            if self.shuffle:
                inds.append(torch.randperm(src_num).long() + cnt)
            else:
                inds.append(torch.arange(src_num).long() + cnt)
            cnt += src_num
        if self.weights is None:
            return iter(torch.cat(inds))
        else:
            selected = []
            for i, w in enumerate(self.weights):
                num = int(round(w * len(inds[i])))
                selected.append(inds[i][:num])
            selected = torch.cat(selected)
            return iter(selected[torch.randperm(len(selected))])

    def __len__(self):
        return self.num_samples
