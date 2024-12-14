import torch
import torcharc


loss_spec = {
    'type': 'BCEWithLogitsLoss',
    'reduction': 'mean',
    'pos_weight': 10.0,
}
criterion = torcharc.build_criterion(loss_spec)

pred = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
loss = criterion(pred, target)
loss


arc = {
    'type': 'Linear',
    'in_features': 8,
    'layers': [64, 32],
    'batch_norm': True,
    'activation': 'ReLU',
    'dropout': 0.2,
    'init': {
        'type': 'normal_',
        'std': 0.01,
    },
}
model = torcharc.build(arc)
m = torch.compile(model)
optim_spec = {
    'type': 'Adam',
    'lr': 0.001,
}
optimizer = torcharc.build_optimizer(optim_spec, model)
optimizer
