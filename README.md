# torchpack

Torchpack is a set of interfaces to simplify the usage of PyTorch.

Documentation is ongoing.


## Example

```python
######################## file1: config.py #######################
work_dir = './demo'  # dir to save log file and checkpoints
optimizer = dict(
    algorithm='SGD', args=dict(lr=0.001, momentum=0.9, weight_decay=5e-4))
workflow = [('train', 2), ('val', 1)]  # train 2 epochs and then validate 1 epochs, iteratively
max_epoch = 16
lr_policy = dict(policy='step', step=12)  # decrese learning rate by 10 every 12 epochs
checkpoint_cfg = dict(interval=1)  # save checkpoint at every epoch
log_cfg = dict(interval=50)  # log at every 50 iterations

######################### file2: main.py ########################
import torch
from torchpack import Config, Runner
from collections import OrderedDict

# define how to process a batch and return a dict
def batch_processor(model, data, train_mode):
    img, label = data
    volatile = False if train_mode else True
    img_var = torch.autograd.Variable(img, volatile=volatile)
    label_var = torch.autograd.Variable(label, requires_grad=False)
    pred = model(img)
    loss = F.cross_entropy(pred, label_var)
    accuracy = get_accuracy(pred, label_var)
    log_vars = OrderedDict()
    log_vars['loss'] = loss.data[0]
    log_vars['accuracy'] = accuracy.data[0]
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs

cfg = Config.from_file('config.py')  # or config.yaml/config.json
model = resnet18()
runner = Runner(model, cfg.optimizer, batch_processor, cfg.work_dir)
runner.register_default_hooks(cfg.lr_policy, cfg.checkpoint_cfg, cfg.log_cfg)

runner.run([train_loader, val_loader], cfg.workflow, cfg.max_epoch)
```