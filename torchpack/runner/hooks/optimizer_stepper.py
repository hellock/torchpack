from torchpack.runner.hooks import Hook


class OptimizerStepperHook(Hook):

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        runner.optimizer.step()
