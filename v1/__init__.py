from contextlib import suppress
from torcharc import module_builder
from torcharc.module import dag
import torch


def build(arc: dict) -> torch.nn.Module:
    '''Interface method to build a DAGNet or a simple torch.nn module. See arf_ref.py for arc references.'''
    if 'dag_in_shape' in arc:
        return dag.DAGNet(arc)
    else:
        return module_builder.build_module(arc)


# additional convenience methods to build criterion and optimizer

def build_criterion(loss_spec: dict) -> torch.nn.Module:
    '''Build criterion (loss function) from loss spec'''
    criterion_cls = getattr(torch.nn, loss_spec.pop('type'))
    # any numeric arg has to be tensor; scan and try-cast
    for k, v in loss_spec.items():
        with suppress(Exception):
            loss_spec[k] = torch.tensor(v)
    criterion = criterion_cls(**loss_spec)
    return criterion


def build_optimizer(optim_spec: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    '''Build optimizer from optimizer spec'''
    optim_cls = getattr(torch.optim, optim_spec.pop('type'))
    optimizer = optim_cls(model.parameters(), **optim_spec)
    return optimizer
