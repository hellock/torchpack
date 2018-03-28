from torch.nn.utils.clip_grad import clip_grad_norm

from torchpack.runner.hooks import Hook


class OptimizerStepperHook(Hook):

    def __init__(self, grad_clip=False, max_norm=35, norm_type=2):
        self.grad_clip = grad_clip
        self.max_norm = max_norm
        self.norm_type = norm_type

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        clip_grad_norm(
            filter(lambda p: p.requires_grad, runner.model.parameters()),
            max_norm=self.max_norm,
            norm_type=self.norm_type)
        runner.optimizer.step()
